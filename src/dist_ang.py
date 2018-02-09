from astropy.cosmology import LambdaCDM


def scale_einstein_radius(zl, zs, H0=70, Om0=0.3, Ode0=0.7):
    cosmo = LambdaCDM(H0=H0, Om0=Om0, Ode0=Ode0)
    d_ls = cosmo.angular_diameter_distance_z1z2(zl, zs)
    d_s = cosmo.angular_diameter_distance_z1z2(0, zs)
    d_l = cosmo.angular_diameter_distance_z1z2(0, zl)

    d = d_ls / d_s

    return d.value