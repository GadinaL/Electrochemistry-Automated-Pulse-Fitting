import numpy as np
import matplotlib.pyplot as plt
import time
import Gadina_Sobolev_fit
from scipy.optimize import curve_fit
import timeit
import datetime
import emcee
from tqdm import tqdm
from functools import partial
from multiprocessing import Pool
import pickle
import os
from tqdm.contrib.concurrent import process_map
from scipy.fft import rfft
import re

# Disable multithreading for numpy and MKL to avoid conflicts with parallel processing in this script
os.environ["OMP_NUM_THREADS"] = "1"
os.environ['OPENBLAS_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'


def Data_check(inspect):
    if inspect == True:
        time_array = np.linspace(0, 2000 / frequency_acquisition, 2000)
        reaction_middle = int(np.prod(current_array.shape) / 2)
        plt.figure(figsize=(12, 6))
        plt.subplot(122)
        plt.plot(time_array[0:2000], current_array[0:2000], 'k')
        plt.xlabel('Time (s)')
        plt.ylabel('Current (A)')
        plt.title('Reaction start')

        plt.subplot(121)
        plt.plot(time_array[0:2000], voltage_array[0:2000], 'k')
        plt.xlabel('Time (s)')
        plt.ylabel('Voltage (V)')
        plt.title('Reaction start')

        plt.tight_layout

        plt.figure(figsize=(12, 6))
        plt.subplot(122)
        plt.plot(time_array[0:2000], current_array[reaction_middle:reaction_middle + 2000], 'k')
        plt.xlabel('Time (s)')
        plt.ylabel('Current (A)')
        plt.title('Reaction middle')

        plt.subplot(121)
        plt.plot(time_array[0:2000], voltage_array[reaction_middle:reaction_middle + 2000], 'k')
        plt.xlabel('Time (s)')
        plt.ylabel('Voltage (V)')
        plt.title('Reaction middle')

        plt.tight_layout

        plt.figure(figsize=(12, 6))
        plt.subplot(122)
        plt.plot(time_array[0:2000], current_array[-2001:-1], 'k')
        plt.xlabel('Time (s)')
        plt.ylabel('Current (A)')
        plt.title('Reaction end')

        plt.subplot(121)
        plt.plot(time_array[0:2000], voltage_array[-2001:-1], 'k')
        plt.xlabel('Time (s)')
        plt.ylabel('Voltage (V)')
        plt.title('Reaction end')

        plt.tight_layout
        plt.show()

    # Methods used by Markov cnain Monte Carlo sampling (emcee library) to estimate the uncertainties of model parameters


def log_prior(theta):
    Y0, Rs, Rf = theta
    if 0.0 < Y0 < 0.001 and 0.0 < Rs < 1000.0 and 0.0 < Rf < 1500.0:
        return 0.0
    return -np.inf


def log_likelihood(theta, jobname, yerr, nY, frequency_acquisition):
    Y0, Rs, Rf = theta
    # model = Gadina_Sobolev_fit.predicted_current_pulse(dict_of_voltage_array_pulse_noisy[jobname], Y0, Rs, Rf, nY, frequency_acquisition)
    model = Gadina_Sobolev_fit.predicted_current_pulse_precomp_rfft(
        dict_of_voltage_array_pulse_noisy_precomp_fft[jobname],
        dict_of_voltage_array_pulse_noisy_precomp_n[jobname],
        Y0, Rs, Rf, nY,
        frequency_acquisition)
    sigma2 = yerr ** 2
    return -0.5 * np.sum((dict_of_current_array_pulse[jobname] - model) ** 2 / sigma2 + np.log(sigma2))


def log_probability(theta, jobname, yerr, nYmc, frequency_acquisition):
    lp = log_prior(theta)
    if not np.isfinite(lp):
        return -np.inf
    return lp + log_likelihood(theta, jobname, yerr, nY=nYmc,
                               frequency_acquisition=frequency_acquisition)


def emcee_worker(job):
    """
    Run an emcee job using parameters loaded (unpickled) from a file at `emceejobs/{job}'.
    The voltage and current datasets are loaded from global dictionaries.
    The results of the job are saved to a backend file (.h5 format).

    Parameters
    ----------
    job: str
        The name of the job to run. This is used to identify the filename containing the parameters,
        as well as the dictionary key to used load the current and voltage pulses from the global dictionaries.
    """
    global NUMBER_OF_STEPS_TO_BE_SAMPLED_BY_EACH_EMCEE_WORKER
    pickle_filename = f'emceejobs/{job}'
    # unpickle the emcee job from the pickle file
    with open(pickle_filename, 'rb') as f:
        emcee_job_data = pickle.load(f)
    pos, nwalkers, ndim, filename, _, _, std_deviation_current, nYmc, frequency_acquisition = emcee_job_data
    backend = emcee.backends.HDFBackend(filename)
    backend.reset(nwalkers, ndim)
    sampler = emcee.EnsembleSampler(
        nwalkers, ndim, log_probability, args=(job, std_deviation_current, nYmc, frequency_acquisition),
        backend=backend,
    )
    sampler.run_mcmc(pos, NUMBER_OF_STEPS_TO_BE_SAMPLED_BY_EACH_EMCEE_WORKER, progress=True)


def Pulse_analyser(data_string, inspect):
    ##Variables

    global packing_factor
    global voltage_array
    global current_array
    global frequency_acquisition
    global decimation_factor
    global pulse_threshold
    global current_amplification_factor
    global voltage_attenuation_factor

    # Data importation
    data = np.load(data_string)
    print('Loading data...')
    current_array = data[:, 1] / current_amplification_factor
    voltage_array = data[:, 0] * voltage_attenuation_factor
    data = data[0:0]
    print('Loaded')

    # Data first visualisation
    Data_check(inspect)

    ##Begining/end detection and cut
    # Begining
    std_deviation_current = float(np.std(current_array[-20000:]))
    std_deviation_voltage = float(np.std(voltage_array[-20000:]))

    pulse_threshold_mask = np.logical_or(voltage_array < -pulse_threshold, voltage_array > pulse_threshold)
    pulse_threshold_mask_index = np.where(pulse_threshold_mask)[0]
    current_array = current_array[pulse_threshold_mask_index[0] - 4:pulse_threshold_mask_index[
        -1]]  # cut begining of reaction (above the threshold)
    voltage_array = voltage_array[pulse_threshold_mask_index[0] - 4:pulse_threshold_mask_index[
        -1]]  # cut begining of reaction (above the threshold)

    # Ending
    neg_to_positive_crossing_mask = np.logical_and(voltage_array[:-1] < 0,
                                                   voltage_array[1:] > 0)  # detect the passage from neg to positive
    positive_to_negative_crossing_mask = np.logical_and(voltage_array[:-1] > 0, voltage_array[1:] < 0)
    sign_change_mask = np.logical_or(neg_to_positive_crossing_mask, positive_to_negative_crossing_mask)
    sign_change_indices = np.where(sign_change_mask)[0]
    pulse_length = sign_change_indices[1] - sign_change_indices[0]
    current_array = current_array[:-pulse_length]  # cut end of reaction
    voltage_array = voltage_array[:-pulse_length]  # cut end of reaction

    Data_check(inspect)

    print('Pulses located, begining and end of data cut')

    packing_size = int(packing_factor * 1000 / (
        pulse_length))  # Pack to get same number of point whatever the pulse width (for 20kHz acquisition and 500 packing, 1M points per fit)

    pulse_pair_indices = np.where(positive_to_negative_crossing_mask)[0]
    pulse_pair_indices = pulse_pair_indices[packing_size::packing_size]  # Packing pulses by packing factor
    individual_pulses_voltage = np.split(voltage_array,
                                         pulse_pair_indices - 1)  # split pulses by zero crossing
    individual_pulses_current = np.split(current_array, pulse_pair_indices - 1)  # split pulses by zero crossing
    individual_pulses_voltage = individual_pulses_voltage[::decimation_factor]
    individual_pulses_current = individual_pulses_current[::decimation_factor]

    print('Pulses packed and decimated')

    # Variable reset
    pulse_threshold_mask_index = pulse_threshold_mask_index[0:0]
    pulse_threshold_mask = pulse_threshold_mask[0:0]
    neg_to_positive_crossing_mask = neg_to_positive_crossing_mask[0:0]
    positive_to_negative_crossing_mask = positive_to_negative_crossing_mask[0:0]
    current_array = current_array[0:0]
    voltage_array = voltage_array[0:0]

    return individual_pulses_voltage, individual_pulses_current, std_deviation_voltage, std_deviation_current


def Pulse_fit(individual_pulses_voltage, individual_pulses_current, std_deviation_voltage, std_deviation_current,
              string_count, emcee_check, use_capacitor_instead_of_CPE, inspect):
    ##Variables
    global packing_factor
    global voltage_array
    global current_array
    global frequency_acquisition
    global decimation_factor
    global noise_factor
    ##Pulse crawling
    pulse_index = 0
    pulse_below_threshold = 0
    total_point_below_threshold = 0
    pulse_not_fitted = 0
    fit_failed = False
    ier_Ftol_count = 0
    ier_Xtol_count = 0
    ier_Xtol__Ftol_count = 0
    ier_mystery_count = 0

    # Time
    start = time.time()

    ##Initial parameters calculation parameters       
    Y0 = 6.5 * (10 ** -5)  # variable capcitance of CPE
    Rs = 450  # solution resistance
    Rf = 130  # charge transfer resistance

    if use_capacitor_instead_of_CPE == True:
        nY = 1
    else:
        nY = 0.7794317572  # variable power of CPE

    # Report initialisation
    Y0_report = np.array([])
    Rs_report = np.array([])
    Rf_report = np.array([])
    Charge_report = np.array([])
    Faradaic_charge_report = np.array([])
    Error_report = np.array([])
    Fit_failed_report = np.array([])

    initial_guess = [Y0, Rs, Rf]

    for pulse in tqdm(individual_pulses_voltage):
        # Pulse noising
        print("\nData points analysed in the pack: ", pulse.size)
        voltage_array_pulse = pulse  # Copy of the pulse_array analysed
        current_array_pulse = individual_pulses_current[pulse_index]
        voltage_array_pulse_noisy = voltage_array_pulse + np.random.normal(0, noise_factor * std_deviation_voltage,
                                                                           voltage_array_pulse.shape)  # Noising of voltage data

        # Fit boundaries
        bounds = [
            (0, np.inf),  # Bounds Y0 (CPE capacitance)
            (0, np.inf),  # Bounds for Rs (solution resistance)
            (0, np.inf)]  # Bounds for Rf (charge transfer resistance)

        lower_bounds = [x[0] for x in bounds]
        upper_bounds = [x[1] for x in bounds]

        kwargs = {
            'x_scale': initial_guess
        }

        # Curve fit
        if pulse_threshold < np.max(pulse):

            pulse_size = voltage_array_pulse.size
            if pulse_size % 2 != 0:
                voltage_array_pulse = voltage_array_pulse[:-1]  # Cut to even number for FT
                current_array_pulse = current_array_pulse[:-1]  # Cut to even number for FT

            # Fit 1st attempt
            try:
                start_time = timeit.default_timer()

                partialfunc = partial(Gadina_Sobolev_fit.predicted_current_pulse, nY=nY,
                                      frequency_acquisition=frequency_acquisition)
                optimal_parameters, covariance_matrix, infodict, mesg, ier = curve_fit(partialfunc,
                                                                                       voltage_array_pulse_noisy,
                                                                                       current_array_pulse,
                                                                                       p0=initial_guess,
                                                                                       bounds=[lower_bounds,
                                                                                               upper_bounds],
                                                                                       sigma=std_deviation_current * np.ones_like(
                                                                                           voltage_array_pulse),
                                                                                       absolute_sigma=True, xtol=1e-15,
                                                                                       ftol=1e-15, full_output=True,
                                                                                       maxfev=5000, **kwargs)

                errors_of_parameters = np.sqrt(np.diag(covariance_matrix))
                elapsed = round(timeit.default_timer() - start_time)
                print(f'Time spent fitting: {elapsed} s')
            except:
                # Fit 2nd attempt from original guess
                try:

                    initial_guess = [Y0, Rs, Rf]
                    optimal_parameters, covariance_matrix, infodict, mesg, ier = curve_fit(
                        Gadina_Sobolev_fit.predicted_current_pulse, voltage_array_pulse_noisy, current_array_pulse,
                        p0=initial_guess, bounds=[lower_bounds, upper_bounds],
                        sigma=std_deviation_current * np.ones_like(voltage_array_pulse), absolute_sigma=True,
                        xtol=1e-15, ftol=1e-15, full_output=True, maxfev=2000)

                    errors_of_parameters = np.sqrt(np.diag(covariance_matrix))
                    elapsed = timeit.default_timer() - start_time
                    print(f'Time spent fitting: {elapsed} s')
                except:
                    fit_failed = True
                    print('Fit failed')
                    Fit_failed_report = np.append(Fit_failed_report, pulse_index)

                    # Post-fitting
            # finally:

            if fit_failed == True:
                pulse_not_fitted += 1
                fit_failed = False

            else:
                # Fit results
                if inspect == True:

                    print("Pulse: ", pulse_index, "\nOptimal parameters: Y0=", optimal_parameters[0], " Rs=",
                          optimal_parameters[1], " Rf=", optimal_parameters[2])
                    parameter_names = ['Y0', 'Rs', 'Rf']
                    for parameter_id, parameter_name in enumerate(parameter_names):
                        print(
                            f'{parameter_name} = {optimal_parameters[parameter_id]} +- {errors_of_parameters[parameter_id]}, '
                            f'relative error is: {errors_of_parameters[parameter_id] / optimal_parameters[parameter_id]:.1%}')

                    plt.imshow(covariance_matrix, cmap='seismic', vmin=-1 * np.max(np.abs(covariance_matrix)),
                               vmax=np.max(np.abs(covariance_matrix)))
                    plt.colorbar()
                    plt.show()

                initial_guess = [optimal_parameters[0], optimal_parameters[1], optimal_parameters[2]]
                # Predicted total and Faradaic current evaluation
                pulse_predicted_current = Gadina_Sobolev_fit.predicted_current_pulse(voltage_array_pulse,
                                                                                     *optimal_parameters, nY=nY,
                                                                                     frequency_acquisition=frequency_acquisition)
                pulse_faradaic_current = Gadina_Sobolev_fit.predicted_faradaic_current_pulse(voltage_array_pulse,
                                                                                             *optimal_parameters, nY=nY,
                                                                                             frequency_acquisition=frequency_acquisition)

                ##Integration current
                integration_faradaic_curent = np.trapz(np.abs(pulse_faradaic_current), dx=1 / frequency_acquisition)
                integration_total_curent = np.trapz(np.abs(pulse_predicted_current), dx=1 / frequency_acquisition)

                # Report update
                Y0_report = np.append(Y0_report, optimal_parameters[0])
                Rs_report = np.append(Rs_report, optimal_parameters[1])
                Rf_report = np.append(Rf_report, optimal_parameters[2])

                Faradaic_charge_report = np.append(Faradaic_charge_report, integration_faradaic_curent)
                Charge_report = np.append(Charge_report, integration_total_curent)

                # Median error on the current prediction
                median_relative_error = np.median(
                    abs((current_array_pulse - pulse_predicted_current) / current_array_pulse))

                Error_report = np.append(Error_report, median_relative_error)

                # Visual check of the current prediction
                if inspect == True:
                    print('Median error as percentage:', median_relative_error * 100, "\n")
                    print("Iterations:", infodict['nfev'], "\nTermination cause:", mesg, ", ier:", ier)
                    print("Percentage in the pulse used in reaction",
                          round((integration_faradaic_curent / integration_total_curent * 100), 2), "%\n")
                  
                    font_properties={'family': 'Arial', 'size': 21, 'weight': 'bold'}
                    time_stamp= np.arange(len(current_array_pulse)) * (1/20000)*1000
                    plt.figure(figsize = (8, 6))
                    plt.rcParams["figure.autolayout"] = True
                    overlapping = 0.150
                    line1=plt.plot(time_stamp,current_array_pulse, c='k', alpha=0.7, linewidth=1.5)
                    line2=plt.plot(time_stamp,pulse_faradaic_current, c='r', alpha=0.7,linewidth=1.5)
                    line3=plt.plot(time_stamp,pulse_predicted_current, c='b', alpha=0.7,linewidth=1.5)
                    plt.legend(["Measured","Faradaic","Calculated"], prop=font_properties,loc='lower left')
                    font_properties={'family': 'Arial', 'size': 21, 'weight': 'bold'}
                    plt.xlabel('Time (ms)', fontdict=font_properties)
                    plt.ylabel('Current (A)', fontdict=font_properties)
                    #plt.title('Pulse number:'+str(pulse_index))
                    # Change the axis line thickness
                    ax = plt.gca()  # Get the current axes
                    ax.spines['bottom'].set_linewidth(2) # Bottom axis line thickness
                    ax.spines['left'].set_linewidth(2)   # Left axis line thickness
                    ax.spines['top'].set_linewidth(0) # Bottom axis line thickness
                    ax.spines['right'].set_linewidth(0)   # Left axis line thickness
                    #ax.tick_params(axis='both', which='major', labelsize=12, labelcolor='black')
                    
                    from matplotlib.font_manager import FontProperties
                    # Create a FontProperties object with custom properties
                    tick_font_properties = FontProperties(family='Arial', size=19, weight='bold')
                    # Apply FontProperties to X and Y tick labels
                    for label in ax.get_xticklabels():
                        label.set_fontproperties(tick_font_properties)

                    for label in ax.get_yticklabels():
                        label.set_fontproperties(tick_font_properties)
                    plt.xlim(40,260)
                    plt.ylim(-0.033,0.033)
                    plt.savefig('fitting results.eps', format='eps', dpi=300)
                    plt.tight_layout()
                    plt.show() 

                if ier == 2:
                    ier_Ftol_count += 1
                elif ier == 3:
                    ier_Xtol_count += 1
                elif ier == 4:
                    ier_Xtol__Ftol_count += 1
                elif ier == 1:
                    ier_mystery_count += 1

                global integration_full_faradaic_curent
                integration_full_faradaic_curent += integration_faradaic_curent  # Cumulate faradaic charges measured
                global integration_full_curent
                integration_full_curent += integration_total_curent  # Cumulate faradaic charges measured

                # Emcee
                if emcee_check == True:
                    print("emcee on going")
                    Y0mc = 5.0 * (10 ** -5)
                    Rsmc = 400.0
                    Rfmc = 125.0

                    nYmc = nY
                    theta = np.array([Y0mc, Rsmc, Rfmc])

                    pos = theta + theta * 1e-3 * np.random.randn(64, theta.size)
                    nwalkers, ndim = pos.shape

                    filename = f"Dataset{string_count}_Pulse{pulse_index}.h5"

                    # Generate emcee job description and pickle it to file for later execution
                    emcee_job = [pos, nwalkers, ndim, filename, voltage_array_pulse_noisy, current_array_pulse,
                                 std_deviation_current, nYmc, frequency_acquisition]
                    pickle_filename = f'emceejobs/Dataset{string_count}_Pulse{pulse_index}.pkl'
                    with open(pickle_filename, 'wb') as f:
                        pickle.dump(emcee_job, f)

        else:
            pulse_below_threshold += 1
            total_point_below_threshold += pulse.size
            print("Pulse below threshold")

        # Pulse update
        pulse_index += 1

        # Parameters results export
    np.savetxt("Y0 decimated variation " + str(string_count), Y0_report, delimiter=',')
    np.savetxt("Rs decimated variation " + str(string_count), Rs_report, delimiter=',')
    np.savetxt("Rf decimated variation " + str(string_count), Rf_report, delimiter=',')
    np.savetxt("Charge decimated variation " + str(string_count), Charge_report, delimiter=',')
    np.savetxt("Fitting decimated error " + str(string_count), Error_report, delimiter=',')
    np.savetxt("Faradaic decimated variation " + str(string_count), Faradaic_charge_report, delimiter=',')

    print('Total of pulses analysed:', pulse_index - pulse_below_threshold, 'Total of point below threshold:',
          total_point_below_threshold)
    print('Total of pulses non fitted:', pulse_not_fitted)
    print('Ftol reached: ', ier_Ftol_count, ', Xtol reached: ', ier_Xtol_count, ', FTol and Xtol converged: ',
          ier_Xtol__Ftol_count, ', ??? reached: ', ier_mystery_count)
    print('Fit failed: ', Fit_failed_report)
    # End

    if inspect == True:
        plt.figure(figsize=(12, 6))
        plt.rcParams["figure.autolayout"] = True
        plt.plot(Y0_report, c='k', alpha=0.7, linewidth=1.5)
        plt.legend(["Y0"])
        plt.xlabel('Time - Points')
        plt.ylim(0, 0.0004)
        plt.title('Pulse number:' + str(pulse_index))
        plt.figure(figsize=(12, 6))
        plt.rcParams["figure.autolayout"] = True
        plt.plot(Rs_report, c='k', alpha=0.7, linewidth=1.5)
        plt.legend(["Rs"])
        plt.xlabel('Time - Points')
        plt.ylim(0, 600)
        plt.title('Pulse number:' + str(pulse_index))
        plt.figure(figsize=(12, 6))
        plt.rcParams["figure.autolayout"] = True
        plt.plot(Rf_report, c='k', alpha=0.7, linewidth=1.5)
        plt.legend(["Rf"])
        plt.xlabel('Time - Points')
        plt.ylim(0, 200)
        plt.title('Pulse number:' + str(pulse_index))
        plt.show()

        # Time
    end = time.time()
    duration = (end - start)
    print('Time to analyse full data: ', duration // 60, 'min, ', round((end - start) % 60), 's')


##Library##
data_string1 = r'c:\Users\UNIST\Desktop\Louis Korea\Publication\rAP\Publication package\Sample Data\13V_2pt5h_50ms.npy'         #Example of raw path file to fit wherever is store the data
data_string2 = r'13V_2pt5h_50ms.npy'         #Example of raw path file to fit directly with the data in same folder than code
data_string3 = r''
data_string4 = r''
data_string5 = r''
data_string6 = r''


##Target List
#data_string_array = np.array([data_string1, data_string2])  # Multiple samples
data_string_array = np.array([data_string1])  #Single sample
######

##Variables
integration_full_faradaic_curent = 0
integration_full_curent = 0
current_amplification_factor = 10
voltage_attenuation_factor = 2

packing_factor = 500
frequency_acquisition = 20000  # in Hz
reaction_scale = 0.2  # mmol
pulse_threshold = 3  # in Volt, trigger the beginning of the spectrum
noise_factor = 32  # FActor used to noise the data before fitting
emcee_check = False  # Do you want to cross-validate by emcee? Warning: higher time of computation required.
# Using emcee not recommended for slow computers.

use_capacitor_instead_of_CPE = False  # Do you want to use a capacitor instead of a CPE? CPE is used by default in our model.
inspect = False  # Do you want to inspect the data processing? Recommended only for troubleshooting.
decimation_factor = 1#10  #1 # analysed performed every decimation_factor pulses.
length_of_trimmed_array = 1000000 #100000
number_of_files_to_be_analysed = len(data_string_array) # Counted from the start of data_string_array. 13 files means all files.
NUMBER_OF_PARALLEL_THREADS = 4 #80  # set this to the number of CPUs/threads your want to occupy on your computer
NUMBER_OF_STEPS_TO_BE_SAMPLED_BY_EACH_EMCEE_WORKER = 500

# # NOTE:
# To give you an idea of computational resources required, here are some examples.
# Time is listed per each 500 steps of EMCEE sampling. That is, there are 64 walkers, and each of them samples 500 steps.
# Executed in 80 parallel threads on Intel(R) Xeon(R) Gold 6230 CPU @ 2.10GHz (40 cores, 2 threads per core).
# Decimation=1 with encee_check=True and length_of_trimmed_array=100_000 takes 18 Gb of RAM and about 4 hours on 80 CPUs
# Decimation=1 with encee_check=True and length_of_trimmed_array=1000_000 takes 182 Gb of RAM and about 8 days on 80 CPUs
# Decimation=10 with encee_check=True and length_of_trimmed_array=1000_000 takes 30 Gb of RAM and 12 hours on 80 CPUs

# # DEBUG/TESTING VALUES OF THE 5 PARAMETERS ABOVE
# decimation_factor = 60
# length_of_trimmed_array = 50000
# number_of_files_to_be_analysed = len(data_string_array)
# NUMBER_OF_PARALLEL_THREADS = 4
# NUMBER_OF_STEPS_TO_BE_SAMPLED_BY_EACH_EMCEE_WORKER = 50


##Parameters display
execution_date = datetime.datetime.now()
parameters = ('Version code: {} \nDate of execution: {} \nFrequency acquisition: {} \nNoise used: {} \n'
              'Decimation factor: {} \nPacking factor: {} \nPulse threshold: {} \nReaction scale: {}').format(
    __file__, execution_date, frequency_acquisition, noise_factor, decimation_factor, packing_factor,
    pulse_threshold, reaction_scale)
print(parameters)

# if the /emceejobs folder does not exist, create it
if emcee_check:
    if not os.path.exists('emceejobs'):
        os.makedirs('emceejobs')

# if the EMCEE jobs were already submitted as files in the /emceejobs folder, load them into RAM to accelerate
# parallel processing with EMCEE.
# If the /emceejobs folder is empty, then list_of_jobs will be the empty and this piece for code.

if emcee_check:
    # make a list of all the .pkl files in the emceejobs directory
    dict_of_voltage_array_pulse_noisy = {}
    dict_of_current_array_pulse = {}
    dict_of_voltage_array_pulse_noisy_precomp_fft = {}
    dict_of_voltage_array_pulse_noisy_precomp_n = {}

    list_of_jobs = os.listdir('emceejobs')

    # lens = []
    for job in list_of_jobs:
        # use re.match to get the dataset id and pulse id from filenames like Dataset10_Pulse16.pkl
        match = re.match('Dataset(\d+)_Pulse(\d+).pkl', job)
        string_count = int(match.group(1))
        pulse_index = int(match.group(2))

        pickle_filename = f'emceejobs/{job}'
        # pickle the emcee job to the emceejobs/pickle_filename
        with open(pickle_filename, 'rb') as f:
            emcee_job_data = pickle.load(f)

        pos, nwalkers, ndim, filename, voltage_array_pulse_noisy, current_array_pulse, std_deviation_current, nYmc, frequency_acquisition = emcee_job_data

        # To speed up FFT, we are trimming the array to length that has very few prime factors by removing almost equal
        # numbers of points from the beginning and end of the array
        middle_index_of_the_pulse = len(current_array_pulse) // 2
        from_idx = middle_index_of_the_pulse - length_of_trimmed_array // 2
        to_idx = from_idx + length_of_trimmed_array

        # precompute ffts for voltage
        Vfft = rfft(voltage_array_pulse_noisy[from_idx:to_idx])
        n = voltage_array_pulse_noisy[from_idx:to_idx].size

        dict_of_current_array_pulse[job] = np.copy(current_array_pulse[from_idx:to_idx])
        dict_of_voltage_array_pulse_noisy[job] = np.copy(voltage_array_pulse_noisy[from_idx:to_idx])
        dict_of_voltage_array_pulse_noisy_precomp_fft[job] = np.copy(Vfft)
        dict_of_voltage_array_pulse_noisy_precomp_n[job] = n


def analysis_worker(passed_args):
    """
    Worker function for parallel processing of data strings. This function is called by the pool.map function.
    The processing inside it has the following steps:
    1. Data-cut and packing
    2. Fitting
    3. Faradaic equivalent calculation
    4. Reporting
    Parameters
    ----------
    passed_args:
        tuple: (string_count, data_string)
    """
    global integration_full_faradaic_curent
    global execution_date, duration, equivalent_electron, frequency_acquisition, noise_factor, decimation_factor, \
        packing_factor, pulse_threshold, reaction_scale, data_string
    string_count, data_string = passed_args
    start = time.time()

    # Data-cut and packing
    individual_pulses_voltage, individual_pulses_current, std_deviation_voltage, std_deviation_current = Pulse_analyser(
        data_string, inspect)
    # Fitting
    Pulse_fit(individual_pulses_voltage, individual_pulses_current, std_deviation_voltage, std_deviation_current,
              string_count, emcee_check, use_capacitor_instead_of_CPE, inspect)
    # Faradaic equivalent calculation
    electron_per_reaction = integration_full_faradaic_curent * decimation_factor / 96485 * 1000  # integration A.s, Faraday constant in A.s/mol, *1000 for mmol
    equivalent_electron = electron_per_reaction / reaction_scale

    end = time.time()
    duration = (end - start)

    # Reporting
    print('Time to analyse full data:', duration // 60, 'min, ', round((end - start) % 60), 's')
    report = ('Version code: {} \nExecuted on: {}\nTime needed: {} \nEquivalent of electron delivered: {} \n'
              'Frequency acquisition: {} \nNoise used: {} \nDecimation factor: {} \nPacking factor: {} \n'
              'Pulse threshold: {} \nReaction scale: {}, \nData:{}, \nData set:{}').format(
        __file__, execution_date, duration, equivalent_electron, frequency_acquisition, noise_factor, decimation_factor,
        packing_factor, pulse_threshold, reaction_scale, data_string, data_string_array)
    with open("Report analysis " + str(string_count) + ".txt", "w") as text_file:
        text_file.write(report)

    # Reinitialisation for next iteration
    string_count += 1
    integration_full_faradaic_curent = 0

if __name__ == '__main__':
    # make a list of passed args for pool.map, each one is a tuple of (string_count, data_string)
    passed_args = [(i, data_string) for i, data_string in enumerate(data_string_array[:number_of_files_to_be_analysed])]
    nthreads_here = min(NUMBER_OF_PARALLEL_THREADS, len(passed_args))
    print(f'Running pulse analysis: a total of {len(passed_args)} files, split over {nthreads_here} parallel threads.')
    with Pool(nthreads_here) as pool:
        pool.map(analysis_worker, passed_args)

    if emcee_check:
        # make sure all the threads have finished saving the EMCEE job description files
        time.sleep(5)
        # load the list of job files
        list_of_jobs = os.listdir('emceejobs')
        for job in list_of_jobs:
            # use re.match to get the dataset id and pulse id from filenames like Dataset10_Pulse16.pkl
            match = re.match('Dataset(\d+)_Pulse(\d+).pkl', job)
            string_count = int(match.group(1))
            pulse_index = int(match.group(2))

            pickle_filename = f'emceejobs/{job}'
            # pickle the emcee job to the emceejobs/pickle_filename
            with open(pickle_filename, 'rb') as f:
                emcee_job_data = pickle.load(f)

            pos, nwalkers, ndim, filename, voltage_array_pulse_noisy, current_array_pulse, std_deviation_current, nYmc, frequency_acquisition = emcee_job_data

            # To speed up FFT, we are trimming the array to length that has very few prime factors by removing almost equal
            # numbers of points from the beginning and end of the array
            middle_index_of_the_pulse = len(current_array_pulse) // 2
            from_idx = middle_index_of_the_pulse - length_of_trimmed_array // 2
            to_idx = from_idx + length_of_trimmed_array

            # precompute ffts for voltage
            Vfft = rfft(voltage_array_pulse_noisy[from_idx:to_idx])
            n = voltage_array_pulse_noisy[from_idx:to_idx].size

            dict_of_current_array_pulse[job] = np.copy(current_array_pulse[from_idx:to_idx])
            dict_of_voltage_array_pulse_noisy[job] = np.copy(voltage_array_pulse_noisy[from_idx:to_idx])
            dict_of_voltage_array_pulse_noisy_precomp_fft[job] = np.copy(Vfft)
            dict_of_voltage_array_pulse_noisy_precomp_n[job] = n

        nthreads_here = min(NUMBER_OF_PARALLEL_THREADS, len(list_of_jobs))
        print(f'Running MCMC (emcee): a total of {len(list_of_jobs)} jobs split over {nthreads_here} parallel threads.')
        res = process_map(emcee_worker, list_of_jobs, max_workers=nthreads_here)
