import numpy as np
from astropy.cosmology import LambdaCDM


def scale_einstein_radius(b, zl, zs, H0=70, Om0=0.3, Ode0=0.7):
    if zl is None or zs is None:
        return b

    cosmo = LambdaCDM(H0=H0, Om0=Om0, Ode0=Ode0)
    d_ls = cosmo.angular_diameter_distance_z1z2(zl, zs)
    d_s = cosmo.angular_diameter_distance_z1z2(0, zs)
    d_l = cosmo.angular_diameter_distance_z1z2(0, zl)

    d = d_ls / d_s

    return b / d.value


def align_coords(x_in, y_in, pars, revert=False):
    """ Change input coordinates (xin, yin) to be relative to the lens (xmap, ymap) and then realign using theta. """
    xlens, ylens, theta, b, q = pars

    if revert is False:
        xmap = x_in - xlens  # xmap is the image position wrt to Xlens at (0,0) in image plane
        ymap = y_in - ylens
        x = xmap * np.cos(theta) + ymap * np.sin(theta)  # Rotates coordinates from image plane to lens plane
        y = ymap * np.cos(theta) - xmap * np.sin(theta)
    else:
        xmap = x_in * np.cos(theta) - y_in * np.sin(theta)  # Rotates coordinates from lens plane to image plane (Xlens is still at Origin)
        ymap = y_in * np.cos(theta) + x_in * np.sin(theta)
        x = xmap + xlens  # x is the deflection position but
        y = ymap + ylens

    return x, y


def deflections(x_img, y_img, zl, zs, pars):
    """ Calculate deflection using a mass model. Returns the calculated source positions. """
    xlens, ylens = pars[0], pars[1]

    x, y = align_coords(x_img, y_img, pars, revert=False)  # Image positions rotated to lens plane coordinates (wrt Xlens at (0,0))
    xout, yout = sie_model(x, y, zl, zs, pars)  # Calculated deflection positions in lens plane coordinates
    x, y = align_coords(xout, yout, pars, revert=True)  # Calculated deflection positions in image plane coordinates (wrt Xlens at (0,0))

    x_deflect, y_deflect = x - xlens, y - ylens  # Calculated deflection positions in image plane coordinates (actual pixel positions instead of wrt Xlens)

    return x_deflect, y_deflect


def sie_model(x, y, zl, zs, pars):
    """ Singular Isothermal Elliptical Mass model for lens. Returns the calculated source positions. """
    xlens, ylens, theta, b, q = pars

    b = scale_einstein_radius(b=b, zl=zl, zs=zs)

    r = np.sqrt(x ** 2 + y ** 2)
    if q == 1.:
        q = 1.-1e-7  # Avoid divide-by-zero errors
    eps = np.sqrt(1. - q ** 2)

    xout = b * np.arcsinh(eps * y / q / r) * np.sqrt(q) / eps
    yout = b * np.arcsin(eps * -x / r) * np.sqrt(q) / eps
    xout, yout = -yout, xout

    return xout, yout


def pred_positions(x_img, y_img, zl, zs, pars):
    """ Predict the source positions by calculating the deflection. Returns the diff between img pos and deflections"""
    x, y = x_img.copy(), y_img.copy()
    x0, y0 = x_img.copy(), y_img.copy()

    x_deflect, y_deflect = deflections(x, y, zl, zs, pars)  # Calculated deflection positions in image plane coordinates (pixel positions)
    x_src = x0 - x_deflect  # Calculated source positions in pixels
    y_src = y0 - y_deflect

    return x_src, y_src


def lnlike(pars, x_img, y_img, zl, zs):
    """ Calculate log-likelihood probability. Minimise the variance in the source position from all images. """
    if isinstance(x_img, dict):
        x_src, y_src = {}, {}
        lnlike_dict = {}

        for key in x_img:
            x_src[key], y_src[key] = pred_positions(x_img[key], y_img[key], zl, zs[key], pars)
            lnlike_dict[key] = -0.5 * (x_src[key].var() + y_src[key].var())

        return sum(lnlike_dict.values())

    else:
        x_src, y_src = pred_positions(x_img, y_img, zl, zs, pars)

        return -0.5 * (x_src.var() + y_src.var())


def lnprior(pars, prior_func=None):
    """ Log-prior for the parameters. """

    if prior_func is None:
        xlens, ylens, theta, b, q = pars

        if (-np.pi < theta < np.pi) and (0. < q < 1):
            return 0.0
        return -np.inf

    else:
        return prior_func(pars)


def lnprob(pars, x_img, y_img, zl, zs, prior_func=None):
    """ Log-probability is the log-prior + log-likelihood. """
    lp = lnprior(pars, prior_func)

    if not np.isfinite(lp):
        return -np.inf
    return lp + lnlike(pars, x_img, y_img, zl, zs)


# if __name__ == '__main__':
#     xlens, ylens, theta, b, q = (53, 40.8, 1.55, 24.1, 0.954)
#     pars = (xlens, ylens, theta, b, q)
#     x_img, y_img = np.array([69.3759]), np.array([28.1791])
#     x_src, y_src = pred_positions(x_img, y_img, pars)
#
#     print(x_src, y_src)
