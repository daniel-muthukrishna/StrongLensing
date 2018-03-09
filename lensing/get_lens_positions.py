import os
import numpy as np
import emcee
from collections import OrderedDict
from chainconsumer import ChainConsumer
from lensing.mass_model import lnprob
import matplotlib.pyplot as plt
from lensing.image_overplot_contours import plot_image_and_contours
from lensing.dist_ang import scale_einstein_radius


def run_mcmc(x_img, y_img, fig_dir, d=None, ndim=5, nwalkers=100, nsteps=200, prior_func=None, initial_params=None, fits_file=None, img_name=''):

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

    # Run MCMC sampler with emcee
    sampler = emcee.EnsembleSampler(nwalkers, ndim, lnprob, args=(x_img, y_img, d, prior_func))
    pos, prob, state = sampler.run_mcmc(p0, nsteps)

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
    c.plotter.plot(filename=os.path.join(fig_dir, 'parameter_contours%s.png' % img_name), truth=truth)
    fig = c.plotter.plot_walks(truth=truth, convolve=100)
    fig.savefig(os.path.join(fig_dir, 'mcmc_walks%s.png' % img_name))

    if fits_file:
        plot_image_and_contours(fits_file, samples, fig_dir, img_name)

    plt.show()


def lensed_quasar():
    # Image positions taken from GAIA+SExtractor for WFI2033_F814W.fits
    # x_img = np.array([27.1051, 69.3759, 71.0171, 56.8204])
    # y_img = np.array([33.6946, 28.1791, 58.9579, 61.2290])
    x_img = np.array([27.1051, 69.3759, 27.5051, 69.8759, 28, 70, 26.5, 68.8])
    y_img = np.array([33.6946, 28.1791, 34, 28, 33.2, 28.5, 33, 27.6])
    fig_dir = 'Figures/lensed_quasar/'

    ndim, nwalkers = 5, 1000
    n_steps = 2000

    # Starting positions for the MCMC walkers sampled from a uniform distribution
    initial = OrderedDict()
    initial[r'$x_{lens}$'] = np.random.uniform(low=40, high=70, size=nwalkers)
    initial[r'$y_{lens}$'] = np.random.uniform(low=30, high=50, size=nwalkers)
    initial[r'$\theta$'] = np.random.uniform(low=-np.pi, high=np.pi, size=nwalkers)
    initial[r'$b$'] = np.random.uniform(low=10, high=40, size=nwalkers)
    initial[r'$q$'] = np.random.uniform(low=0.2, high=1., size=nwalkers)

    def lnprior(pars):
        xsource, ysource, theta, b, q = pars
        if (28 < xsource < 69) and (28 < ysource < 60) and (10 < b < 40) and (-np.pi < theta < np.pi) and (0. < q < 1):
            return 0.0
        else:
            return -np.inf

    run_mcmc(x_img, y_img, fig_dir, d=1, nwalkers=nwalkers, nsteps=n_steps, prior_func=lnprior, initial_params=initial)


def macs0451():
    fig_dir = 'Figures/MACS0451/'
    nwalkers = 1500
    steps = 3000

    # ImageA Positions
    img_name = '_ImageA'
    x_img = np.array([2375.942110, 2378.5, 2379.816610, 2381.299088, 2384, 2385.927991, 2389.555816, 2457.694760, 2450.744242, 2442.833333, 2437.857924, 2433.064587, 2427.166666, 2424.099866, 2418.5, 2416.444081, 2462])
    y_img = np.array([3038.016677, 3024, 3012.367933, 2999.365293, 2983.5, 2970.435199, 2955.945319, 2737.545077, 2752.305849, 2766.166666, 2782.058508, 2795.293450, 2811.166666, 2823.079067, 2837.5, 2846.943113, 2728])

    # # Image 1.1 Positions
    # img_name = '_Image1.1'
    # x_img = np.array([3557.178601, 3548.271886, 3541.407488])
    # y_img = np.array([3363.943860, 3375.285957, 3385.515024])

    # # Image 6.2 Positions
    # img_name = '_Image6.2'
    # x_img = np.array([3486.371962])
    # y_img = np.array([3069.305065])

    # # Image 4.1 Positions
    # img_name = '_Image4.1'
    # x_img = np.array([3222.796159, 3227.700108])
    # y_img = np.array([3550.903781, 3542.180780])

    # # Image 3.1 Positions
    # img_name = '_Image3.1'
    # x_img = np.array([2933.063074, 3393.715824])
    # y_img = np.array([2943.400421, 3398.196336])



    # Starting positions for the MCMC walkers sampled from a uniform distribution
    initial = OrderedDict()
    initial[r'$x_{lens}$'] = np.random.uniform(low=2400, high=4000, size=nwalkers)
    initial[r'$y_{lens}$'] = np.random.uniform(low=2000, high=3700, size=nwalkers)
    initial[r'$\theta$'] = np.random.uniform(low=-np.pi, high=np.pi, size=nwalkers)
    initial[r'$b$'] = np.random.uniform(low=10, high=2000, size=nwalkers)
    initial[r'$q$'] = np.random.uniform(low=0.2, high=1., size=nwalkers)

    def lnprior(pars):
        xsource, ysource, theta, b, q = pars
        if (2400 < xsource < 4000) and (2000 < ysource < 3700) and (10 < b < 2000) and (-np.pi < theta < np.pi) and (0. < q < 1):
            return 0.0
        else:
            return -np.inf

    fits_file = '/Users/danmuth/PycharmProjects/StrongLensing/data/MACS0451/MACS0451_F110W.fits'

    run_mcmc(x_img, y_img, fig_dir, nwalkers=nwalkers, nsteps=steps, prior_func=lnprior, initial_params=initial, fits_file=fits_file, img_name=img_name)


def macs0451_multiple_sources():
    fig_dir = 'Figures/MACS0451/'
    ndim, nwalkers = 5, 100
    nsteps = 200
    img_name = '_multiple_sources2'
    z_lens = 0.43

    x_img, y_img, d = {}, {}, {}
    x_img['A'] = np.array([2375.942110, 2378.5, 2379.816610, 2381.299088, 2384, 2385.927991, 2389.555816, 2457.694760, 2450.744242, 2442.833333, 2437.857924, 2433.064587, 2427.166666, 2424.099866, 2418.5, 2416.444081, 2462])
    y_img['A'] = np.array([3038.016677, 3024, 3012.367933, 2999.365293, 2983.5, 2970.435199, 2955.945319, 2737.545077, 2752.305849, 2766.166666, 2782.058508, 2795.293450, 2811.166666, 2823.079067, 2837.5, 2846.943113, 2728])
    d['A'] = scale_einstein_radius(z_lens=z_lens, z_src=2.01)

    x_img['11'] = np.array([3557.178601, 3548.271886, 3541.407488])
    y_img['11'] = np.array([3363.943860, 3375.285957, 3385.515024])
    d['11'] = scale_einstein_radius(z_lens=z_lens, z_src=2.06)

    # x_img['62'] = np.array([3486.371962])
    # y_img['62'] = np.array([3069.305065])
    # d['62'] = scale_einstein_radius(z_lens=z_lens, zs=1.405)

    x_img['41'] = np.array([3222.796159, 3227.700108])
    y_img['41'] = np.array([3550.903781, 3542.180780])
    d['41'] = scale_einstein_radius(z_lens=z_lens, z_src=1.810)

    x_img['31'] = np.array([2933.063074, 3393.715824])
    y_img['31'] = np.array([2943.400421, 3398.196336])
    d['31'] = scale_einstein_radius(z_lens=z_lens, z_src=1.904)

    # Starting positions for the MCMC walkers sampled from a uniform distribution
    initial = OrderedDict()
    initial[r'$x_{lens}$'] = np.random.uniform(low=2400, high=4000, size=nwalkers)
    initial[r'$y_{lens}$'] = np.random.uniform(low=2000, high=3700, size=nwalkers)
    initial[r'$\theta$'] = np.random.uniform(low=-np.pi, high=np.pi, size=nwalkers)
    initial[r'$b$'] = np.random.uniform(low=10, high=2000, size=nwalkers)
    initial[r'$q$'] = np.random.uniform(low=0.2, high=1., size=nwalkers)

    def lnprior(pars):
        xlens, ylens, theta, b, q = pars
        if (2400 < xlens < 4000) and (2000 < ylens < 3700) and (-np.pi < theta < np.pi) and (10 < b < 2000) and (0. < q < 1):
            return 0.0
        else:
            return -np.inf

    fits_file = '/Users/danmuth/PycharmProjects/StrongLensing/data/MACS0451/MACS0451_F110W.fits'

    run_mcmc(x_img, y_img, fig_dir, d, ndim=ndim, nwalkers=nwalkers, nsteps=nsteps, prior_func=lnprior, initial_params=initial, fits_file=fits_file, img_name=img_name)


if __name__ == '__main__':
    # lensed_quasar()
    # macs0451()
    macs0451_multiple_sources()
