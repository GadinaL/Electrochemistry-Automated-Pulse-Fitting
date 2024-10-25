import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
import pickle
from simulate_kinetics_shapes import *

starting_concentration = 0.2e-3 / 2.5e-3  # mol / L


if __name__ == '__main__':
    satconc = {'tri': 0.0179, 'sin': 0.0210, 'square': 0.03, 'square12': 0.03, 'square25': 0.03, 'square50': 0.03, 'DC': 0.0281}
    temps = {'tri': 32.7, 'sin': 35.1, 'square': 42, 'square12': 42, 'square25': 42, 'square50': 42, 'DC': 40.5}

    filenames = {'tri': 'Yield_25ms_Time_Potentiostatic13V_Triangle.txt',
                 'sin': 'Yield_25ms_Time_Potentiostatic13V_Sinusoidal.txt',
                 'square': 'Yield_25ms_Time_Potentiostatic.txt',
                 'square12': 'Yield_12ms_Time_Potentiostatic.txt',
                 'square25': 'Yield_25ms_Time_Potentiostatic.txt',
                 'square50': 'Yield_50ms_Time_Potentiostatic.txt',
                 'DC': 'Yield_Time_Potentiostatic_DC.txt'}
    fit_k = dict()

    shapes = ['square12', 'square25', 'square50', 'DC']
    pulse_lengths = [12, 25, 50, 1800000]
    for shape in shapes:
        fit_k[shape] = plot_for_one_file(f'data/yield_curves/{filenames[shape]}',
                                         saturation_concentration=satconc[shape],
                                         figname_suffix=f'_model_vs_time')


    with open('data/fit_k_pulselengths.pkl', 'wb') as f:
        pickle.dump(fit_k, f)

    # unpickle back
    with open('data/fit_k_pulselengths.pkl', 'rb') as f:
        fit_k = pickle.load(f)

    k_labels = ['k_1', 'k_{-1}', 'k_2'] # and so on

    # k_indices = [0]

    font_properties = {'family': 'Arial', 'size': 21, 'weight': 'bold'}
    fig, ax = plt.subplots(1, 1, figsize=(8, 6))
    ax2 = ax.twinx()
    axarr = [ax, 'this value should not be accessed', ax2]

    activation_energies = [7.09E4, 'this value should not be accessed', 8.43E+04]
    popts = []
    for k_index in [0, 2]:
        k1_dc_at_T_DC = fit_k['DC'][0][k_index]
        print(f'k{k_index}_dc_at_T_DC', k1_dc_at_T_DC)

        # k1_dc is at 38.9 C, while we need to calculate it at 42 C, at which all the square pulses happened
        T_DC = 40.5 + 273.15
        T_42 = 42 + 273.15
        k1_dc = k1_dc_at_T_DC * np.exp(activation_energies[k_index] / 8.314 * (1/T_DC - 1/T_42))
        print(f'Corrected k1_dc is {k1_dc}')
        x = pulse_lengths[:-1]
        width = 0.35
        k1 = [fit_k[shape][0][k_index] for shape in shapes[:-1]]
        k1_error = [fit_k[shape][1][k_index] for shape in shapes[:-1]]
        axarr[k_index].errorbar(x, k1, yerr=k1_error, fmt='o', label='k1', capsize=4, elinewidth=1,
                                color=f'C{k_index}')

        x = np.array(x)
        k1 = np.array(k1)
        k1_error = np.array(k1_error)

        # make a line fit
        def func(x, a, b):
            return k1_dc * (1 - b*np.exp(-a * x))
        popt, pcov = curve_fit(func, x, k1, p0=(1/50, 0.8), sigma=k1_error, absolute_sigma=True)
        x_line = np.linspace(3, 100, 55)
        y_line = func(x_line, *popt)
        perr = np.sqrt(np.diag(pcov))
        # print('K1 = a * L + b, where L is the length of the pulse')
        print('K = k_dc (1 - b*np.exp(-a * L)), where L is the length of the pulse')
        print(f'a = {popt[0]} +- {perr[0]}')
        print(f'b = {popt[1]:.2f} +- {perr[1]:.3f}')
        print(f'Inverse of "a" is the time constant of the exponential decay: {1/popt[0]:.2f} ms')
        # print(f'b = {popt[1]:.2f} +- {perr[1]:.3f}')

        axarr[k_index].plot(x_line, y_line, label='Linear fit', color=f'C{k_index}')
        axarr[k_index].axhline(y=k1_dc, color=f'black', linestyle='--', label='DC k1')
        axarr[k_index].set_ylabel(f'${k_labels[k_index]}$ (1/hour)', fontdict=font_properties)
        ax.set_xlabel('Pulse length (ms)', fontdict=font_properties)

        axarr[k_index].spines['bottom'].set_linewidth(2)  # Bottom axis line thickness
        axarr[k_index].spines['left'].set_linewidth(2)  # Left axis line thickness
        axarr[k_index].spines['top'].set_linewidth(0)  # Bottom axis line thickness
        axarr[k_index].spines['right'].set_linewidth(2)  # Left axis line thickness
        from matplotlib.font_manager import FontProperties

        # Create a FontProperties object with custom properties
        tick_font_properties = FontProperties(family='Arial', size=19, weight='bold')
        # Apply FontProperties to X and Y tick labels
        for label in axarr[k_index].get_xticklabels():
            label.set_fontproperties(tick_font_properties)

        for label in axarr[k_index].get_yticklabels():
            label.set_fontproperties(tick_font_properties)

        axarr[k_index].tick_params(axis='y', labelcolor=f'C{k_index}')
        axarr[k_index].yaxis.label.set_color(f'C{k_index}')

        plt.tight_layout()

        fig.savefig(f'figures/fitted_{k_labels[k_index]}_vs_pulse_length.png', dpi=300)
        fig.savefig(f'figures/fitted_{k_labels[k_index]}_vs_pulse_length.eps', dpi=300)

        popts.append(np.copy(popt))

    plt.show()