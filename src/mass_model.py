import numpy as np


def get_image_positions(x_img, y_img):
    """ Input image positions here. May rewrite to read positions directly from the sextractor file. """

    return x_img, y_img


def align_coords(x_in, y_in, pars, revert=False):
    """ Change input coordinates (xin, yin) to be relative to the source (xmap, ymap) and then realign using theta. """
    xsource, ysource, theta, b, q = pars

    if revert is False:
        xmap = x_in - xsource  # xmap is the image position wrt to Xsource at (0,0) in image plane
        ymap = y_in - ysource
        x = xmap * np.cos(theta) + ymap * np.sin(theta)  # Rotates coordinates from image plane to source plane
        y = ymap * np.cos(theta) - xmap * np.sin(theta)
    else:
        xmap = x_in * np.cos(theta) - y_in * np.sin(theta)  # Rotates coordinates from source plane to image plane (Xsource is still at Origin)
        ymap = y_in * np.cos(theta) + x_in * np.sin(theta)
        x = xmap + xsource  # x is the deflection position but
        y = ymap + ysource

    return x, y


def deflections(x_img, y_img, pars):
    """ Calculate deflection using a mass model. Returns the calculated source positions. """
    xsource, ysource = pars[0], pars[1]

    x, y = align_coords(x_img, y_img, pars, revert=False)  # Image positions rotated to source plane coordinates (wrt Xsource at (0,0))
    xout, yout = sie_model(x, y, pars)  # Calculated deflection positions in source plane coordinates
    x, y = align_coords(xout, yout, pars, revert=True)  # Calculated deflection positions in image plane coordinates (wrt Xsource at (0,0))

    x_deflect, y_deflect = x - xsource, y - ysource  # Calculated deflection positions in image plane coordinates (actual pixel positions instead of wrt Xsource)

    return x_deflect, y_deflect


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

    x_deflect, y_deflect = deflections(x, y, pars)  # Calculated deflection positions in image plane coordinates (pixel positions)
    x_src = x0 - x_deflect  # Calculated source positions in pixels
    y_src = y0 - y_deflect


    return x_src, y_src


def lnlike(pars, x_img, y_img):
    """ Calculate log-likelihood probability. Minimise the variance in the source position from all images. """
    if isinstance(x_img, dict):
        for key in x_img.items():
            pass

    x_img, y_img = get_image_positions(x_img, y_img)
    x_src, y_src = pred_positions(x_img, y_img, pars)

    return -0.5 * (x_src.var() + y_src.var())


def lnprior(pars, prior_func=None):
    """ Log-prior for the parameters. """

    if prior_func is None:
        xsource, ysource, theta, b, q = pars

        if (-np.pi < theta < np.pi) and (0. < q < 1):
            return 0.0
        return -np.inf

    else:
        return prior_func(pars)


def lnprob(pars, x_img, y_img, prior_func=None):
    """ Log-probability is the log-prior + log-likelihood. """
    lp = lnprior(pars, prior_func)

    if not np.isfinite(lp):
        return -np.inf
    return lp + lnlike(pars, x_img, y_img)


if __name__ == '__main__':
    xsource, ysource, theta, b, q = (53, 40.8, 1.55, 24.1, 0.954)
    pars = (xsource, ysource, theta, b, q)
    x_img, y_img = np.array([69.3759]), np.array([28.1791])
    x_src, y_src = pred_positions(x_img, y_img, pars)

    print(x_src, y_src)
