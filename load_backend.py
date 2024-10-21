import emcee
import numpy as np
import corner
import matplotlib.pyplot as plt
from multiprocessing import Pool


def latex_pretty_uncertainties(mean, minus_sigma, plus_sigma):
    """
    Pretty print uncertainties in latex format, according to proper convention of reporting measurement results:
    standard error is reported with two significant digits. The median (or mean) is reported with the same number of
    decimal places as the standard deviation. Decimal exponent is written separately to the right like so:
    1.0481_{-0.0013}^{0.0013} \\\\times 10^{-4}

    Parameters
    ----------
    mean: float
    minus_sigma: float
    plus_sigma: float

    Returns
    -------
    str
        Pretty printed string in latex format like so: 1.0481_{-0.0013}^{0.0013} \\times 10^{-4}
    """
    # "detach" 10^x from the number
    decimal_exponent = np.floor(np.log10(np.abs(mean)))
    reduced_mean = mean / 10 ** decimal_exponent
    reduced_minus_sigma = minus_sigma / 10 ** decimal_exponent
    reduced_plus_sigma = plus_sigma / 10 ** decimal_exponent

    # find the location of two significant digits in the sigmas
    location_of_significant_digits_of_plus_sigma = np.floor(np.log10(reduced_plus_sigma))
    location_of_significant_digits_of_minus_sigma = np.floor(np.log10(reduced_minus_sigma))
    dig = min(location_of_significant_digits_of_plus_sigma, location_of_significant_digits_of_minus_sigma)

    # format to number of digits after comma equal to dig
    str_mean = "{0:.{1}f}".format(reduced_mean, int(-dig) + 1)
    str_sigma_plus = "{0:.{1}f}".format(reduced_plus_sigma, int(-dig) + 1)
    str_sigma_minus = "{0:.{1}f}".format(reduced_minus_sigma, int(-dig) + 1)

    # put everything together
    return r"{0}_{{-{1}}}^{{{2}}} \times 10^{{{3:.0f}}} ".format(
        str_mean, str_sigma_minus, str_sigma_plus, decimal_exponent)


def plot_corner_from_backend(backend_filename, do_show=True, tau=None):
    reader = emcee.backends.HDFBackend(backend_filename)

    if tau is None:
        tau = reader.get_autocorr_time()
    print(f'tau: {tau}')
    burnin = int(2 * np.max(tau))
    thin = int(0.5 * np.min(tau))
    samples = reader.get_chain(discard=burnin, flat=True, thin=thin)
    log_prob_samples = reader.get_log_prob(discard=burnin, flat=True, thin=thin)

    print("burn-in: {0}".format(burnin))
    print("thin: {0}".format(thin))
    print("flat chain shape: {0}".format(samples.shape))
    print("flat log prob shape: {0}".format(log_prob_samples.shape))

    best_index = np.argmax(reader.get_log_prob(flat=True))
    theta_max = reader.get_chain(flat=True)[best_index]
    print('The sample with highest probability is: Y_0={0}, R_s={1}, R_f={2}'.format(*theta_max))

    labels = ['Y_0', 'R_s', 'R_f']
    units = ['', '\Omega', '\Omega']
    labels_with_percentiles = []
    for i in range(3):
        mcmc = np.percentile(samples[:, i], [16, 50, 84])
        q = np.diff(mcmc)
        pretty_latex = latex_pretty_uncertainties(mcmc[1], q[0], q[1])
        labels_with_percentiles.append(r"$\mathrm{{{0}}} = {1} ~{2}$".format(labels[i], pretty_latex, units[i]))

    corner.corner(samples, labels=labels_with_percentiles, truths=theta_max)

    plt.gcf().suptitle(backend_filename)
    plt.gcf().set_size_inches(9, 9)
    # plt.tight_layout()

    if do_show:
        plt.show()

def plot_worker(file):
    print(f'Processing file: {file}')
    plot_corner_from_backend(file, do_show=False, tau=[92.00297665, 13.59598999, 49.22439449])
    filename_without_extension = file.split('.')[0]
    plt.gcf().savefig(f'corners/{filename_without_extension}.png', dpi=200)
    plt.close('all')

if __name__ == '__main__':
    import os
    # list all files with extension .h5
    list_of_h5_files = os.listdir('.')
    list_of_h5_files = [x for x in list_of_h5_files if x.endswith('.h5')]
    print(f'Number of files: {len(list_of_h5_files)}')
    with Pool(77) as pool:
        pool.map(plot_worker, list_of_h5_files)


    ### COMPARISON
    # plot_corner_from_backend('Dataset10_Pulse17_trim100k.h5', do_show=False)
    # plt.gcf().suptitle('Trimmed to 100k, but 5000 steps for each worker')
    # fig2 = plt.figure(2)
    # plot_corner_from_backend('Dataset10_Pulse17.h5', tau=[92.00297665, 13.59598999, 49.22439449],
    #                          do_show=False)
    # plt.gcf().suptitle('Trimmed to 1M, 500 steps for each worker. Forcing the autocorrs from 5k steps sample.')
    # plt.show()