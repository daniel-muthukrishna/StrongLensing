import os
import numpy as np
import emcee
from collections import OrderedDict
from chainconsumer import ChainConsumer
from src.mass_model import lnprob
import matplotlib.pyplot as plt


def run_mcmc(x_img, y_img, fig_dir, nwalkers=100, steps=200, prior_func=None, initial_params=None):

    # Starting positions for the MCMC walkers sampled from a uniform distribution
    if initial_params is None:
        initial = OrderedDict()
        initial[r'$x_{source}$'] = np.random.uniform(low=0, high=100, size=nwalkers)
        initial[r'$y_{source}$'] = np.random.uniform(low=0, high=100, size=nwalkers)
        initial[r'$\theta$'] = np.random.uniform(low=-np.pi, high=np.pi, size=nwalkers)
        initial[r'$b$'] = np.random.uniform(low=10, high=40, size=nwalkers)
        initial[r'$q$'] = np.random.uniform(low=0.2, high=1., size=nwalkers)
    else:
        initial = initial_params
    p0 = np.transpose(list(initial.values()))

    # MCMC setup
    ndim = 5

    # Run MCMC sampler with emcee
    sampler = emcee.EnsembleSampler(nwalkers, ndim, lnprob, args=(x_img, y_img, prior_func))
    pos, prob, state = sampler.run_mcmc(p0, steps)

    # Print chain
    samples = sampler.chain[:, 50:, :].reshape((-1, ndim))
    print(sampler.chain[:, 1, 0])
    samples = sampler.flatchain
    print(samples.shape)

    # Get best fit parameters
    samples_exp = samples.copy()
    samples_exp[:, 2] = np.exp(samples_exp[:, 2])
    best_fits = list(map(lambda v: (v[1]), zip(*np.percentile(samples_exp, [16, 50, 84], axis=0))))
    truth = dict(zip(initial.keys(), best_fits))
    print(truth)

    # Plot parameter contours and mcmc chains
    c = ChainConsumer()
    c.add_chain(samples, parameters=list(initial.keys()))
    c.configure(summary=True, cloud=True)
    c.plotter.plot(filename=os.path.join(fig_dir, 'parameter_contours.png'), truth=truth)
    fig = c.plotter.plot_walks(truth=truth, convolve=100)
    fig.savefig(os.path.join(fig_dir, 'mcmc_walks.png'))
    # plt.show()


def lensed_quasar():
    # Image positions taken from GAIA+SExtractor for WFI2033_F814W.fits
    # x_img = np.array([27.1051, 69.3759, 71.0171, 56.8204])
    # y_img = np.array([33.6946, 28.1791, 58.9579, 61.2290])
    x_img = np.array([27.1051, 69.3759, 27.5051, 69.8759, 28, 70, 26.5, 68.8])
    y_img = np.array([33.6946, 28.1791, 34, 28, 33.2, 28.5, 33, 27.6])
    fig_dir = 'Figures/lensed_quasar/'

    nwalkers = 1000
    steps = 2000

    # Starting positions for the MCMC walkers sampled from a uniform distribution
    initial = OrderedDict()
    initial[r'$x_{source}$'] = np.random.uniform(low=40, high=70, size=nwalkers)
    initial[r'$y_{source}$'] = np.random.uniform(low=30, high=50, size=nwalkers)
    initial[r'$\theta$'] = np.random.uniform(low=-np.pi, high=np.pi, size=nwalkers)
    initial[r'$b$'] = np.random.uniform(low=10, high=40, size=nwalkers)
    initial[r'$q$'] = np.random.uniform(low=0.2, high=1., size=nwalkers)

    def lnprior(pars):
        xsource, ysource, theta, b, q = pars
        if (28 < xsource < 69) and (28 < ysource < 60) and (10 < b < 40) and (-np.pi < theta < np.pi) and (0. < q < 1):
            return 0.0
        else:
            return -np.inf

    run_mcmc(x_img, y_img, fig_dir, nwalkers=nwalkers, steps=steps, prior_func=lnprior, initial_params=initial)


def macs0451():
    # Image positions taken from GAIA+SExtractor for MACS0451_F110W.fits
    x_img = np.array([2375.942110, 2378.5, 2379.816610, 2381.299088, 2384, 2385.927991, 2389.555816, 2457.694760, 2450.744242, 2442.833333, 2437.857924, 2433.064587, 2427.166666, 2424.099866, 2418.5, 2416.444081, 2462])
    y_img = np.array([3038.016677, 3024, 3012.367933, 2999.365293, 2983.5, 2970.435199, 2955.945319, 2737.545077, 2752.305849, 2766.166666, 2782.058508, 2795.293450, 2811.166666, 2823.079067, 2837.5, 2846.943113, 2728])
    fig_dir = 'Figures/MACS0451/'

    nwalkers = 1000
    steps = 2000

    # Starting positions for the MCMC walkers sampled from a uniform distribution
    initial = OrderedDict()
    initial[r'$x_{source}$'] = np.random.uniform(low=2700, high=4000, size=nwalkers)
    initial[r'$y_{source}$'] = np.random.uniform(low=2000, high=3700, size=nwalkers)
    initial[r'$\theta$'] = np.random.uniform(low=-np.pi, high=np.pi, size=nwalkers)
    initial[r'$b$'] = np.random.uniform(low=10, high=1500, size=nwalkers)
    initial[r'$q$'] = np.random.uniform(low=0.2, high=1., size=nwalkers)

    def lnprior(pars):
        xsource, ysource, theta, b, q = pars
        if (2700 < xsource < 4000) and (2000 < ysource < 3700) and (10 < b < 1500) and (-np.pi < theta < np.pi) and (0. < q < 1):
            return 0.0
        else:
            return -np.inf

    run_mcmc(x_img, y_img, fig_dir, nwalkers=nwalkers, steps=steps, prior_func=lnprior, initial_params=initial)


if __name__ == '__main__':
    # lensed_quasar()
    macs0451()
