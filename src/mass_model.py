import numpy as np


def get_image_positions(x_img, y_img):
    """ Input image positions here. May rewrite to read positions directly from the sextractor file. """

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
    """ Predict the source positions by calculating the deflection. Returns the diff between img pos and deflections"""
    x, y = x_img.copy(), y_img.copy()
    x0, y0 = x_img.copy(), y_img.copy()

    xmap, ymap = deflections(x, y, pars)
    x0 = x0 - xmap
    y0 = y0 - ymap

    return x0, y0


def lnlike(pars, x_img, y_img):
    """ Calculate log-likelihood probability. Minimise the variance in the source position from all images. """
    x_img, y_img = get_image_positions(x_img, y_img)
    x_src, y_src = pred_positions(x_img, y_img, pars)

    return -0.5 * (x_src.var() + y_src.var())


def lnprior(pars):
    """ Log-prior for the parameters. """
    xsource, ysource, theta, b, q = pars
    if (-np.pi < theta < np.pi) and (0. < q < 1):
        return 0.0
    return -np.inf


def lnprob(pars, x_img, y_img):
    """ Log-probability is the log-prior + log-likelihood. """
    lp = lnprior(pars)

    if not np.isfinite(lp):
        return -np.inf
    return lp + lnlike(pars, x_img, y_img)
