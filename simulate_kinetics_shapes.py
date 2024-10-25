import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
import pickle

starting_concentration = 0.2e-3 / 2.5e-3  # mol / L


def get_yields_at_given_times_general(rate_constants, time_points, timesteps=3000, saturation_concentration=0.2e-3 / 2.5e-3 * 0.05):
    """
    Computes the dependence of yields of the substances on time, given the rate constants of the reactions.

    :param rate_constants: This is array of five values [k0, k1, k2, k3, k4], where
    k0 is the zero-order rate constant for the first product formation,
    k1 is the first-order rate constant for the second product formation,
    k2 is the first-order rate constant for the degradation of second reduction product,
    k3 is the zero-order rate constant for degradation of the starting material,
    k4 is the first-order rate constant for the formation of the side product from the first reduction product.

    :param time_points: The concentrations will be evaluated at these time points, although integration is performed on
    a much finer time grid,
    :param timesteps: The timesteps of the integration. The higher the number, the more accurate the integration.
    :param effective_faradaic_current: The effective faradaic current used to scale the
    reaction rates. Must be in units of mol/L/ Set to None to disable the correction (scaling) of the reaction rates.
    :return: Array of shape [len(time_points) x 4] of yields of the substances at the given time points. Remember that the last
    column is the yield of the side product, which is not used in the fitting. Yield is evaluated with respect to
    starting_concentration, which is a global variable (bad, I know).
    """
    concentrations_over_time = np.zeros(shape=(timesteps, 4), dtype=np.float64)
    concentrations = np.array([starting_concentration, 0, 0, 0], dtype=np.float64)
    ts = np.linspace(0, time_points[-1], timesteps)
    dt = ts[1] - ts[0]
    for i, t in enumerate(ts):
        k1, k_1, k2, k_2, k3, k4, k5 = rate_constants
        A, B, C, D = concentrations
        actual_A = min(A, saturation_concentration)
        # These are partial derivatives with respect to time
        derivatives_of_concentrations = np.zeros_like(concentrations)
        # starting materials
        derivatives_of_concentrations[0] = -1 * k1 * actual_A - k3 * actual_A + k_1 * B
        # first reduction product
        derivatives_of_concentrations[1] = k1 * actual_A - k_1 * B - k4 * B + k_2 * C - k2 * B
        # second reduction product
        derivatives_of_concentrations[2] = k2 * B - k_2 * C - k5 * C
        # side product
        derivatives_of_concentrations[3] = k3 * actual_A + k4 * B + k5 * C

        concentrations += derivatives_of_concentrations * dt  # Euler's method

        # if any of concentrations became negative, make them zero
        concentrations = np.maximum(concentrations, 0)
        concentrations_over_time[i, :] = np.copy(concentrations)

    # resample the calculated time trace of concentrations at the input time points
    concentrations_at_timepoints = np.zeros((len(time_points), 4))
    for substance_index in range(4):
        ys = concentrations_over_time[:, substance_index]
        # interpolate ts, ys at time_points
        interpolated_ys = np.interp(time_points, ts, ys)
        concentrations_at_timepoints[:, substance_index] = interpolated_ys
    # convert concentrations to yields and return
    return concentrations_at_timepoints / starting_concentration


def get_yields_at_given_times_potentiostatic(rate_constants, time_points, timesteps=3000):
    """
    Computes the dependence of yields of the substances on time, given the rate constants of the reactions.
    In this potentiostatic version, the faradaic current is not used for scaling the reaction rates.

    :param rate_constants: This is array of five values [k0, k1, k2, k3, k4], where
    k0 is the zero-order rate constant for the first product formation,
    k1 is the first-order rate constant for the second product formation,
    k2 is the first-order rate constant for the degradation of second reduction product,
    k3 is the zero-order rate constant for degradation of the starting material,
    k4 is the first-order rate constant for the formation of the side product from the first reduction product.

    :param time_points: The concentrations will be evaluated at these time points, although integration is performed on
    a much finer time grid,
    :param timesteps: The timesteps of the integration. The higher the number, the more accurate the integration.
    :return: Array of shape [len(time_points) x 4] of yields of the substances at the given time points. Remember that the last
    column is the yield of the side product, which is not used in the fitting. Yield is evaluated with respect to
    starting_concentration, which is a global variable (bad, I know).
    """
    return get_yields_at_given_times_general(rate_constants, time_points, effective_faradaic_current=None,
                                             timesteps=timesteps)


def load_data_from_file(data_filename):
    """
    Load the data from the file, and return the time points, yields, and errors.
    :param data_filename: path to the data file. Data file must be tab-separated, and have columns 'Time', 'yield SM',
    'yield 1st', 'yield 2nd', 'stdev SM', 'stdev', 'stdev 2nd' in the first line. Yields are in percent, time in hours.
    :return: tuple of time points, yields, unpack like so:
        experimental_timepoints, yield_0, yield_1, yield_2, errors_1, errors_2, errors_3 = return_value
    """
    data = pd.read_csv(data_filename, sep='\t')
    experimental_timepoints = data['Time'].values
    yield_0 = data['yield SM'].values / 100
    yield_1 = data['yield 1st'].values / 100
    yield_2 = data['yield 2nd'].values / 100

    errors_1 = data['stdev SM'].values / 100
    errors_2 = data['stdev'].values / 100
    errors_3 = data['stdev 2nd'].values / 100

    errors_1[errors_1 == 0] = 0.005
    errors_2[errors_2 == 0] = 0.005
    errors_3[errors_3 == 0] = 0.005
    return experimental_timepoints, yield_0, yield_1, yield_2, errors_1, errors_2, errors_3


def plot_for_one_file(data_filename = 'data/yield_curves/Yield_25ms_Time_Potentiostatic.txt',
                      saturation_concentration = 0.2e-3 / 2.5e-3 * 0.05,
                      figname_suffix=''):
    only_file_name_without_extension = data_filename.split('/')[-1].split('.')[0]
    figname = only_file_name_without_extension + figname_suffix + '.png'
    # data_filename = 'data/yield_curves/Yield_Time_Potentiostatic_DC.txt'
    experimental_timepoints, yield_0, yield_1, yield_2, errors_1, errors_2, errors_3 = load_data_from_file(
        data_filename)

    # experimental_times, repeated three times
    ts = np.concatenate([experimental_timepoints, experimental_timepoints, experimental_timepoints])
    # experimental yields, all concatenated
    ys = np.concatenate([yield_0, yield_1, yield_2])

    def potentiostatic_model(*args):
        """
        This is the model that is used for fitting the potentiostatic data. The first argument is the time series,
        and the rest of the arguments are the rate constants of the reactions.
        :param args:
        :return: Yields of the substances at the given time points, concatenated into a one-dimensional array like so:
            [array_with_yield_of_starting_material, array_with_yield_of_first_reduction_product, array_with_yield_of_second_reduction_product]
        """
        # Unpacking of the args
        ts = args[0]
        # I know that there are three plots in the ys, so time series is repeated three times in the ts. I only want
        # the first repetition
        actual_len = int(round(len(ts) / 3))
        ts = ts[:actual_len]
        # kinetic rates start from the second argument onward
        ks = args[1:]
        yields = get_yields_at_given_times_general(ks, ts, timesteps=3000,
                                                          saturation_concentration=saturation_concentration)
        # drop the last yield, because it's the side product yield
        yields = yields[:, :-1]
        # concatenate the three yields into one array, for the curve_fit to work
        yields2 = np.concatenate([yields[:, 0], yields[:, 1], yields[:, 2]])
        return yields2

    # # do curve fitting of funt to ts, ys
    lower_bounds = [0, 0, 0, 0, 0, 0, 0]
    upper_bounds = [np.inf] * 7

    sigmas = 0.04664769 * ys + 0.02803713 # these are best-fit coefficients from the script error_analysis.py
    # sigmas = np.concatenate([errors_1, errors_2, errors_3]) # uncomment this to use instrumental errors from 3 replications

    p0 = [0.5, 1e-3, 0.07, 1e-3, 0.14, 0.01, 1e-3]
    popt, pcov = curve_fit(potentiostatic_model, ts, ys, p0=p0, bounds=(lower_bounds, upper_bounds),
                           sigma=sigmas, absolute_sigma=True, xtol=None, ftol=1e-12)
    perr = np.sqrt(np.diag(pcov))
    print(f'Data file: {data_filename}')
    print('Fitted parameters:')
    notes = ['K1  formation of first reduction product from starting material (forward), [1/hour]',
             'K_1 formation of first reduction product from starting material (backward), [1/hour]',
             'K2  formation of second reduction product from first reduction product (forward), [1/hour]',
             'K_2 formation of second reduction product from first reduction product (backward), [1/hour]',
             'K3  formation of side product from starting material, [1/hour]',
             'K4  formation of side product from first reduction product [1/hour]',
             'K5  formation of side prodict from second reduction product [1/hour]']
    for i in range(len(popt)):
        print(f'({notes[i]}) {popt[i]}+-{100*perr[i] / popt[i]:.1f}%')

    sigmas_experimental = np.concatenate([errors_1, errors_2, errors_3])
    chisq = np.sum((ys - potentiostatic_model(ts, *popt))**2 / sigmas_experimental**2)
    print(f'Chi squared = {chisq}\n')

    font_properties = {'family': 'Arial', 'size': 21, 'weight': 'bold'}
    fig, ax = plt.subplots(1, 1, figsize=(8, 6))
    plt.rcParams["figure.autolayout"] = True
    time_correction = 1
    colors = ('black', 'red', 'blue')
    alpha = 1
    linestyle = '--'
    capsize = 4
    elinewidth = 1
    ax.errorbar(experimental_timepoints, 100*yield_0, yerr=100*errors_1, fmt='o', label='yield SM', color=colors[0],
                alpha=alpha, capsize=capsize, elinewidth=elinewidth)
    ax.errorbar(experimental_timepoints, 100*yield_1, yerr=100*errors_2, fmt='o', label='yield 1', color=colors[1],
                alpha=alpha, capsize=capsize, elinewidth=elinewidth)
    ax.errorbar(experimental_timepoints, 100*yield_2, yerr=100*errors_3, fmt='o', label='yield 2', color=colors[2],
                alpha=alpha, capsize=capsize, elinewidth=elinewidth)

    ts = np.linspace(0, experimental_timepoints[-1] / time_correction, 400)
    ys_fitted = potentiostatic_model(np.concatenate([ts, ts, ts]), *popt)
    ys_fitted = [ys_fitted[:len(ts)], ys_fitted[len(ts):2 * len(ts)], ys_fitted[2 * len(ts):]]
    ax.plot(ts * time_correction, 100*ys_fitted[0], label='yield SM fitted', color=colors[0], linestyle=linestyle)
    ax.plot(ts * time_correction, 100*ys_fitted[1], label='yield 1 fitted', color=colors[1], linestyle=linestyle)
    ax.plot(ts * time_correction, 100*ys_fitted[2], label='yield 2 fitted', color=colors[2], linestyle=linestyle)
    ax.set_ylim(0, 100)
    ax.set_xlabel('Time (h)', fontdict=font_properties)
    ax.set_ylabel('Yield (%)', fontdict=font_properties)
    ax.set_xlim(0, np.max(experimental_timepoints)*1.02)

    ax.spines['bottom'].set_linewidth(2)  # Bottom axis line thickness
    ax.spines['left'].set_linewidth(2)  # Left axis line thickness
    ax.spines['top'].set_linewidth(0)  # Bottom axis line thickness
    ax.spines['right'].set_linewidth(0)  # Left axis line thickness

    from matplotlib.font_manager import FontProperties
    # Create a FontProperties object with custom properties
    tick_font_properties = FontProperties(family='Arial', size=19, weight='bold')
    # Apply FontProperties to X and Y tick labels
    for label in ax.get_xticklabels():
        label.set_fontproperties(tick_font_properties)

    for label in ax.get_yticklabels():
        label.set_fontproperties(tick_font_properties)

    fig.savefig(f'figures/{figname}', dpi=300)
    # change figname extension to .eps
    fig.savefig(f'figures/{figname.split(".")[0]}.eps', dpi=300)

    # plt.show()
    plt.close('all')

    return popt, perr


def plot_model_only(popt, figname, saturation_concentration, title='', alpha=1, scale_factor=1, do_clf=True, ax=None):

    def potentiostatic_model(*args):
        """
        This is the model that is used for fitting the potentiostatic data. The first argument is the time series,
        and the rest of the arguments are the rate constants of the reactions.
        :param args:
        :return: Yields of the substances at the given time points, concatenated into a one-dimensional array like so:
            [array_with_yield_of_starting_material, array_with_yield_of_first_reduction_product, array_with_yield_of_second_reduction_product]
        """
        # Unpacking of the args
        ts = args[0]
        # I know that there are three plots in the ys, so time series is repeated three times in the ts. I only want
        # the first repetition
        actual_len = int(round(len(ts) / 3))
        ts = ts[:actual_len]
        # kinetic rates start from the second argument onward
        ks = args[1:]
        yields = get_yields_at_given_times_general(ks, ts, timesteps=3000,
                                                          saturation_concentration=saturation_concentration)
        # drop the last yield, because it's the side product yield
        yields = yields[:, :-1]
        # concatenate the three yields into one array, for the curve_fit to work
        yields2 = np.concatenate([yields[:, 0], yields[:, 1], yields[:, 2]])
        return yields2

    if do_clf:
        fig, ax = plt.subplots(1, 1, figsize=(5.5, 4))
    time_correction = 1
    colors = ('black', 'red', 'blue')
    # alpha = 1
    linestyle = '--'
    capsize = 4
    elinewidth = 1

    ts = np.linspace(0, 20, 1000)
    ys_fitted = potentiostatic_model(np.concatenate([ts, ts, ts]), *popt)
    ys_fitted = [ys_fitted[:len(ts)], ys_fitted[len(ts):2 * len(ts)], ys_fitted[2 * len(ts):]]
    ax.plot(ts/scale_factor, 100*ys_fitted[0], label='yield SM fitted', color=colors[0], linestyle=linestyle, alpha=alpha)
    ax.plot(ts/scale_factor, 100*ys_fitted[1], label='yield 1 fitted', color=colors[1], linestyle=linestyle, alpha=alpha)
    ax.plot(ts/scale_factor, 100*ys_fitted[2], label='yield 2 fitted', color=colors[2], linestyle=linestyle, alpha=alpha)
    ax.set_ylim(0, 100)
    ax.set_xlabel('Time (h)')
    ax.set_ylabel('Yield (%)')
    ax.set_xlim(0, np.max(ts)*1.02)
    plt.title(title)

    if do_clf:
        fig.savefig(f'figures/{figname}', dpi=300)
    if do_clf:
        plt.clf()
    time_of_max_yield = ts[np.argmax(ys_fitted[1])]
    return time_of_max_yield, np.max(ys_fitted[1])
    # plt.show()


if __name__ == '__main__':
    # satconc = {'tri': 0.01519, 'sin': 0.01747, 'square': 0.02461, 'square12': 0.02461, 'square25': 0.02461, 'square50': 0.02342}
    # temps = {'tri': 32.5, 'sin': 34.8, 'square': 42, 'square12': 42, 'square25': 42, 'square50': 40.8}
    satconc = {'tri': 0.0179, 'sin': 0.0210, 'square': 0.03, 'square12': 0.03, 'square25': 0.03, 'square50': 0.03}
    temps = {'tri': 32.7, 'sin': 35.1, 'square': 42, 'square12': 42, 'square25': 42, 'square50': 40.8}
    filenames = {'tri': 'Yield_25ms_Time_Potentiostatic13V_Triangle.txt',
                 'sin': 'Yield_25ms_Time_Potentiostatic13V_Sinusoidal.txt',
                 'square': 'Yield_25ms_Time_Potentiostatic.txt',
                 'square12': 'Yield_12ms_Time_Potentiostatic.txt',
                 'square25': 'Yield_25ms_Time_Potentiostatic.txt',
                 'square50': 'Yield_50ms_Time_Potentiostatic.txt'}
    fit_k = dict()

    for shape in ['tri', 'sin', 'square']:
        fit_k[shape] = plot_for_one_file(f'data/yield_curves/{filenames[shape]}',
                                         saturation_concentration=satconc[shape],
                                         figname_suffix=f'_model_vs_time')


    with open('data/fit_k.pkl', 'wb') as f:
        pickle.dump(fit_k, f)

    # unpickle back
    with open('data/fit_k.pkl', 'rb') as f:
        fit_k = pickle.load(f)

    k_labels = ['k_1', 'k_{-1}', 'k_2'] # and so on

    k_indices = [0, 2]

    fig, ax1 = plt.subplots(1, 1, figsize=(8, 6))
    # add a second y axis on the right
    ax2 = ax1.twinx()
    axarr = [ax1, "don't use", ax2]
    colors = ['red', 'haha', 'blue']

    for k_index in k_indices:
        # use categorical on x axis, ['tri', 'sin', 'square']
        # use the fitted rate constants to plot the bar plot
        x = np.arange(3)
        width = 0.35
        k1 = [fit_k[shape][0][k_index] for shape in ['tri', 'sin', 'square']]
        k1_error = [fit_k[shape][1][k_index] for shape in ['tri', 'sin', 'square']]

        # plt.errorbar(x, k1, yerr=k1_error, fmt='o', label='k1', capsize=4, elinewidth=1)
        # axarr[k_index].set_xticks(x)
        # axarr[k_index].set_xticklabels(['triangle', 'sine', 'square'])
        # axarr[k_index].set_ylabel(f'${k_labels[k_index]}$ (1/hour)')
        # fig.savefig(f'figures/fitted_{k_labels[k_index]}_vs_shape.png', dpi=300)
        # plt.show()

        plt.rcParams["figure.autolayout"] = True
        ax = axarr[k_index]

        x = [1/(273.15 + temps[shape]) for shape in ['tri', 'sin', 'square']]
        x = np.array(x)
        k1 = [fit_k[shape][0][k_index] for shape in ['tri', 'sin', 'square']]
        k1_error = [fit_k[shape][1][k_index] for shape in ['tri', 'sin', 'square']]
        ax.errorbar(1000*x, k1, yerr=k1_error, fmt='o', label='k1', capsize=4, elinewidth=1, color=colors[k_index])
        # set scale to log
        ax.set_yscale('log')
        ax.set_ylabel(f'${k_labels[k_index]}$ (1/hour), logarithmic scale')
        ax.set_xlabel('1000/T (K$^{-1}$)')

        line_x = x
        line_y = np.log(k1)

        # fit line to the data
        a,b = np.polyfit(line_x, line_y, 1)
        # print(f'Fitted line: y = {a:.2f}x + {b:.2f}')
        # make a best fit of the data
        def func(T_inv, k0, u_over_R):
            return k0 * np.exp(-u_over_R*T_inv)

        popt, pcov = curve_fit(func, x, k1, p0=(np.exp(b), -a), sigma=k1_error, absolute_sigma=True, bounds=[(0, 0), (np.inf, np.inf)])
        # print(f'Fitted parameters: {popt}')
        perr = np.sqrt(np.diag(pcov))

        print(f'Activation energy: ({popt[1] * 8.314:.2E} +- {perr[1]*8.314:.2E}) J/mol')
        print(f'Pre-exponential factor: ({popt[0]:.2E} +- {perr[0]:.2E}) 1/hour')
        # b = U/R, so U = b*R
        U = popt[1] * 8.314
        x_fit = np.linspace(np.min(x), np.max(x), 100)
        y_fit = func(x_fit, *popt)
        ax.plot(1000*x_fit, y_fit, label='fit', linestyle='--', color=colors[k_index])
        ax.set_xlim(3.16, 3.28)
        if k_index == 0:
            ax.set_ylim(0.38, 1.57)
        else:
            ax.set_ylim(0.01, 0.1)

        # place y axis ticks on 5, 6, 7, 8, 9, 10
        if k_index == 0:
            ax.set_yticks([0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1, 1.2, 1.4])
            ax.set_yticklabels([f'{i:.1f}' for i in ax.get_yticks()])
        elif k_index == 2:
            ax.set_yticks([0.01, 0.02, 0.03, 0.04, 0.06, 0.08])
            ax.set_yticklabels([f'{i:.2f}' for i in ax.get_yticks()])
        # set ticklabels to one digit after comma
        # ax.set_title('Fitted k_1 rate constant')

        font_properties = {'family': 'Arial', 'size': 21, 'weight': 'bold'}
        overlapping = 0.150
        font_properties = {'family': 'Arial', 'size': 21, 'weight': 'bold'}

        ax.set_ylabel(f'${k_labels[k_index]}$ (1/hour), logarithmic scale', fontdict=font_properties)
        ax.set_xlabel('1000/T (K$^{-1}$)', fontdict=font_properties)
        # Change the axis line thickness
        ax.spines['bottom'].set_linewidth(2)  # Bottom axis line thickness
        ax.spines['left'].set_linewidth(2)  # Left axis line thickness
        ax.spines['top'].set_linewidth(0)  # Bottom axis line thickness
        ax.spines['right'].set_linewidth(2)  # Left axis line thickness
        # ax.tick_params(axis='both', which='major', labelsize=12, labelcolor='black')

        from matplotlib.font_manager import FontProperties

        # Create a FontProperties object with custom properties
        tick_font_properties = FontProperties(family='Arial', size=19, weight='bold')
        # Apply FontProperties to X and Y tick labels
        for label in ax.get_xticklabels():
            label.set_fontproperties(tick_font_properties)

        for label in ax.get_yticklabels():
            label.set_fontproperties(tick_font_properties)
            # plt.tight_layout()

        ax.tick_params(axis='y', labelcolor=colors[k_index])
        ax.yaxis.label.set_color(colors[k_index])
        # plt.show()

        # plt.tight_layout()
        fig.savefig(f'figures/fitted_{k_labels[k_index]}_vs_inverse_T.png', dpi=300)
        fig.savefig(f'figures/fitted_{k_labels[k_index]}_vs_inverse_T.eps', dpi=300)
    plt.show()

