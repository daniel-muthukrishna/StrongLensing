import os
import numpy as np
import emcee
from collections import OrderedDict
from chainconsumer import ChainConsumer
from src.mass_model import lnprob
import matplotlib.pyplot as plt

""" Given source positions calculate what images are created """


def align_coords(x_in, y_in, pars, revert=False):
    """ Change input coordinates (xin, yin) to be relative to the source (xmap, ymap) and then realign using theta. """
    ximage, yimage, theta, b, q = pars

    if revert is False:
        xmap = ximage - x_in  # xmap is the image position wrt to Xsource at (0,0) in image plane
        ymap = yimage - y_in
        x = xmap * np.cos(theta) + ymap * np.sin(theta)  # Rotates coordinates from image plane to source plane
        y = ymap * np.cos(theta) - xmap * np.sin(theta)
    else:
        xmap = x_in * np.cos(theta) - y_in * np.sin(theta)  # Rotates coordinates from source plane to image plane (Xsource is still at Origin)
        ymap = y_in * np.cos(theta) + x_in * np.sin(theta)
        x = xmap + ximage  # x is the deflection position but
        y = ymap + yimage

    return x, y


def deflections(x_src, y_src, pars):
    """ Calculate deflection using a mass model. Returns the calculated source positions. """
    ximage, yimage = pars[0], pars[1]

    x, y = align_coords(x_src, y_src, pars, revert=False)  # Source positions rotated to source plane coordinates (wrt Xsource at (0,0))
    xout, yout = sie_model(x, y, pars)  # Calculated deflection positions in source plane coordinates
    x, y = align_coords(xout, yout, pars, revert=True)  # Calculated deflection positions in image plane coordinates (wrt Xsource at (0,0))

    x_deflect, y_deflect = x - ximage, y - yimage  # Calculated deflection positions in image plane coordinates (actual pixel positions instead of wrt Xsource)

    return x_deflect, y_deflect


def sie_model(x, y, pars):
    """ Singular Isothermal Elliptical Mass model for lens. Returns the calculated image positions. """
    ximage, yimage, theta, b, q = pars

    r = np.sqrt(x ** 2 + y ** 2)
    if q == 1.:
        q = 1.-1e-7  # Avoid divide-by-zero errors
    eps = np.sqrt(1. - q ** 2)

    xout = -np.sinh(y * eps / b / np.sqrt(q)) * r / eps
    yout = np.sinh(x * eps / b / np.sqrt(q)) * r * q / eps
    xout, yout = yout, -xout

    if xout == np.inf:
        xout = 1e99
    elif xout == -np.inf:
        xout = -1e99
    if yout == np.inf:
        yout = 1e99
    elif yout == -np.inf:
        yout = -1e99

    return xout, yout


def pred_positions(x_src, y_src, pars):
    """ Predict the image positions by calculating the deflection"""
    x, y = x_src.copy(), y_src.copy()
    x0, y0 = x_src.copy(), y_src.copy()

    x_deflect, y_deflect = deflections(x, y, pars)  # Calculated deflection positions in image plane coordinates (pixel positions)
    x_img = x0 + x_deflect  # Calculated source positions in pixels
    y_img = y0 + y_deflect

    return x_img, y_img


def lnlike(pars, x_src, y_src):
    """ Calculate log-likelihood probability. Minimise the variance in the source position from all images. """
    ximage, yimage, theta, b, q = pars
    x_img_calc, y_img_calc = pred_positions(x_src, y_src, pars)

    diffx = x_img_calc - ximage
    diffy = y_img_calc - yimage

    err = -0.5 * (diffx**2 + diffy**2)

    if abs(diffx) < 0.05 and abs(diffy) < 0.05:
        print(ximage, yimage, theta, b, q)

    return err


def lnprior(pars, prior_func=None):
    """ Log-prior for the parameters. """

    if prior_func is None:
        ximage, yimage, theta, b, q = pars

        if (0 < ximage < 100) and (0 < yimage < 100) and (10 < b < 40) and (-np.pi < theta < np.pi) and (0. < q < 1):
            return 0.0
        return -np.inf

    else:
        return prior_func(pars)


def lnprob(pars, x_src, y_src, prior_func=None):
    """ Log-probability is the log-prior + log-likelihood. """
    lp = lnprior(pars, prior_func)

    if not np.isfinite(lp):
        return -np.inf
    return lp + lnlike(pars, x_src, y_src)


def run_mcmc(x_src, y_src, fig_dir, prior_func=None):

    # MCMC setup
    ndim, nwalkers = 5, 200

    # Starting positions for the MCMC walkers sampled from a uniform distribution
    initial = OrderedDict()
    initial[r'$x_{image}$'] = np.random.uniform(low=0, high=100, size=nwalkers)
    initial[r'$y_{image}$'] = np.random.uniform(low=0, high=100, size=nwalkers)
    initial[r'$\theta$'] = np.random.uniform(low=-np.pi, high=np.pi, size=nwalkers)
    initial[r'$b$'] = np.random.uniform(low=10, high=40, size=nwalkers)
    initial[r'$q$'] = np.random.uniform(low=0.2, high=1., size=nwalkers)
    p0 = np.transpose(list(initial.values()))

    # Run MCMC sampler with emcee
    sampler = emcee.EnsembleSampler(nwalkers, ndim, lnprob, args=(x_src, y_src, prior_func))
    pos, prob, state = sampler.run_mcmc(p0, 100)

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
    c.plotter.plot(filename=os.path.join(fig_dir, 'parameter_contours_imagePos.png'), truth=truth)
    fig = c.plotter.plot_walks(truth=truth, convolve=100)
    fig.savefig(os.path.join(fig_dir, 'mcmc_walks_imagePos.png'))


if __name__ == '__main__':
    # ximage, yimage, theta, b, q = (69.3759,  28.1791, 1.55, 24.1, 0.954)
    # pars = (ximage, yimage, theta, b, q)
    x_src, y_src = np.array([53]), np.array([40])
    # x_img, y_img = pred_positions(x_src, y_src, pars)
    #
    # print(x_img, y_img)

    run_mcmc(x_src, y_src, fig_dir='/Users/danmuth/PycharmProjects/StrongLensing/Figures/lensed_quasar/')

    # plt.show()
