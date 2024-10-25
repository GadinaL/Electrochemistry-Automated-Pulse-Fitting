import matplotlib.pyplot as plt
import numpy as np

from simulate_kinetics_shapes import *

if __name__ == '__main__':
    filenames = {'tri': 'Yield_25ms_Time_Potentiostatic13V_Triangle.txt',
                 'sin': 'Yield_25ms_Time_Potentiostatic13V_Sinusoidal.txt',
                 'square': 'Yield_25ms_Time_Potentiostatic.txt',
                 'square12': 'Yield_12ms_Time_Potentiostatic.txt',
                 'square25': 'Yield_25ms_Time_Potentiostatic.txt',
                 'square50': 'Yield_50ms_Time_Potentiostatic.txt',
                    'DC': 'Yield_Time_Potentiostatic_DC.txt'}

    all_yields = []
    all_errors = []
    for key in ['tri', 'sin', 'square12', 'square25', 'square50', 'DC']:
        data_filename = 'data/yield_curves/' + filenames[key]
        experimental_timepoints, yield_0, yield_1, yield_2, errors_1, errors_2, errors_3 = load_data_from_file(
            data_filename)
        all_yields.extend(yield_0)
        all_yields.extend(yield_1)
        all_yields.extend(yield_2)
        all_errors.extend(errors_1)
        all_errors.extend(errors_2)
        all_errors.extend(errors_3)

    all_yields = np.array(all_yields)
    all_errors = np.array(all_errors)

    mask = np.array(all_yields) <= 0.8
    all_yields = np.array(all_yields)[mask]
    all_errors = np.array(all_errors)[mask]

    mask = np.array(all_yields) > 0
    all_yields = np.array(all_yields)[mask]
    all_errors = np.array(all_errors)[mask]

    # sort the data by yield using numpy
    sorted_indices = np.argsort(all_yields)
    all_yields = all_yields[sorted_indices]
    all_errors = all_errors[sorted_indices]

    plt.plot(100*np.array(all_yields), 100*np.array(all_errors), 'o', alpha=0.5)
    def func(x, a, b):
        return a*x + b

    # mask yields above 0.9
    popt, pcov = curve_fit(func, all_yields, all_errors, p0=(0.2, 0.01))
    # plot the line
    xs = np.linspace(0, 1, 100)
    ys = func(all_yields, *popt)
    plt.plot(100*all_yields, 100*ys, color='black')
    plt.ylabel('Absolute error of the yield (%)')
    plt.xlabel('Yield (%)')
    print(popt)
    plt.gcf().savefig('figures/error_analysis.png', dpi=300)
    plt.show()

    fig, ax = plt.subplots(1, 1, figsize=(6, 4))
    bins = [0, 10, 20, 30, 40, 50, 60, 70, 80]
    all_yields = np.array(all_yields)
    all_errors = np.array(all_errors)
    for i in range(len(bins) - 1):
        mask = (all_yields >= bins[i]/100) & (all_yields < bins[i+1]/100)
        # make boxplot
        ax.boxplot(all_errors[mask]*100, positions=[i], showfliers=False)
        print('Number of points in the bin', i, ':', np.sum(mask))
    ax.set_xticks(range(len(bins) - 1))
    ax.set_xticklabels([f'{bins[i]}-{bins[i+1]}' for i in range(len(bins) - 1)])

    plt.plot(100*all_yields/30*3 - 0.5, 100*ys, color='C2')

    plt.ylabel('Experimental RMSE of the yield (%) over three replicates')
    plt.xlabel('Range of the yield values (%)')

    plt.xlim(-0.5, 7.5)

    plt.tight_layout()
    fig.savefig('figures/error_analysis_boxplot.png', dpi=300)
    plt.show()
