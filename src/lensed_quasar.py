import numpy as np
import emcee
from collections import OrderedDict
from chainconsumer import ChainConsumer
from src.mass_model import lnprob


def main():
    # Image positions taken from GAIA+SExtractor for WFI2033_F814W.fits
    x_img = np.array([27.1051, 69.3759, 71.0171, 56.8204])
    y_img = np.array([33.6946, 28.1791, 58.9579, 61.2290])

    # MCMC setup
    ndim, nwalkers = 5, 2000

    # Starting positions for the MCMC walkers sampled from a uniform distribution
    initial = OrderedDict()
    initial[r'$x_{source}$'] = np.random.uniform(low=50, high=60, size=nwalkers)
    initial[r'$y_{source}$'] = np.random.uniform(low=35, high=45, size=nwalkers)
    initial[r'$\theta$'] = np.random.uniform(low=-np.pi, high=np.pi, size=nwalkers)
    initial[r'$b$'] = np.random.uniform(low=10, high=40, size=nwalkers)
    initial[r'$q$'] = np.random.uniform(low=0.2, high=1., size=nwalkers)
    p0 = np.transpose(list(initial.values()))

    # Run MCMC sampler with emcee
    sampler = emcee.EnsembleSampler(nwalkers, ndim, lnprob, args=(x_img, y_img))
    pos, prob, state = sampler.run_mcmc(p0, 2000)

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

    # Plot parameter contours and mcmc chains
    c = ChainConsumer()
    c.add_chain(samples, parameters=list(initial.keys()))
    c.configure(summary=True, cloud=True)
    c.plotter.plot(filename='Figures/lensed_quasar/parameter_contours.png', truth=truth)
    fig = c.plotter.plot_walks(truth=truth, convolve=100)
    fig.savefig('Figures/lensed_quasar/mcmc_walks.png')


if __name__ == '__main__':
    main()
