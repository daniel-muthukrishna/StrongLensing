import numpy as np
import emcee
from collections import OrderedDict
from chainconsumer import ChainConsumer


def get_image_positions():
    """ Input image positions here. May rewrite to read positions directly from the sextractor file. """

    # Image positions taken from GAIA+SExtractor for WFI2033_F814W.fits
    x_img = np.array([27.1051, 69.3759, 71.0171, 56.8204])
    y_img = np.array([33.6946, 28.1791, 58.9579, 61.2290])

    return x_img, y_img


def align_coords(xin, yin, pars, revert=False):
    """ Change input coordinates (xin, yin) to be relative to the source (xmap, ymap) and then realign using theta. """
    xsource, ysource, theta, b, q = pars

    if revert is False:
        xmap = xin - xsource  # xin and xsource are coords directly from the image
        ymap = yin - ysource  # yin and ysource are coords directly from the image
        x = xmap * np.cos(theta) + ymap * np.sin(theta)
        y = ymap * np.cos(theta) - xmap * np.sin(theta)
    else:
        xmap = xin * np.cos(theta) - yin * np.sin(theta)
        ymap = yin * np.cos(theta) + xin * np.sin(theta)
        x = xmap + xsource
        y = ymap + ysource

    return x, y


def deflections(xin, yin, pars):
    """ Calculate deflection using a mass model. Returns the calculated source positions. """
    xsource, ysource = pars[0], pars[1]

    x, y = align_coords(xin, yin, pars, revert=False)
    xout, yout = sie_model(x, y, pars)
    x, y = align_coords(xout, yout, pars, revert=True)

    return x - xsource, y - ysource


def sie_model(x, y, pars):
    """ Singular Isothermal Elliptical Mass model for lens. Returns the calculated source positions. """
    xsource, ysource, theta, b, q = pars

    r = np.sqrt(x ** 2 + y ** 2)
    if q == 1.:
        q = 1.-1e-7  # Avoid divide-by-zero errors
    eps = np.sqrt(1. - q ** 2)

    xout = b * np.arcsinh(eps * y / q / r) * np.sqrt(q) / eps
    yout = b * np.arcsin(eps * -x / r) * np.sqrt(q) / eps
    xout, yout = -yout, xout

    return xout, yout


def pred_positions(x_img, y_img, pars):
    """ Predict the source positions by calculating the deflection. Returns the difference bete"""
    x, y = x_img.copy(), y_img.copy()
    x0, y0 = x_img.copy(), y_img.copy()

    xmap, ymap = deflections(x, y, pars)
    x0 = x0 - xmap
    y0 = y0 - ymap

    return x0, y0


def lnlike(pars):
    """ Calculate log-likelihood probability. Minimise the variance in the source position from all images. """
    x_img, y_img = get_image_positions()
    x_src, y_src = pred_positions(x_img, y_img, pars)

    return -0.5 * (x_src.var() + y_src.var())


def lnprior(pars):
    """ Log-prior for the parameters. """
    xsource, ysource, theta, b, q = pars
    if (-np.pi < theta < np.pi) and (0. < q < 1):
        return 0.0
    return -np.inf


def lnprob(pars):
    """ Log-probability is the log-prior + log-likelihood. """
    lp = lnprior(pars)

    if not np.isfinite(lp):
        return -np.inf
    return lp + lnlike(pars)


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
sampler = emcee.EnsembleSampler(nwalkers, ndim, lnprob)
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
c.plotter.plot(filename='Figures/parameter_contours.png', truth=truth)
fig = c.plotter.plot_walks(truth=truth, convolve=100)
fig.savefig('Figures/mcmc_walks.png')
