import os
import numpy as np
from collections import OrderedDict
from chainconsumer import ChainConsumer
import matplotlib.pyplot as plt
from image_overplot_contours import plot_image_and_contours, plot_image
from dist_ang import scale_einstein_radius
from pylens import pylens, MassModels
from imageSim import SBObjects
import myEmcee
import pymc

SCRIPT_DIR = os.path.dirname(os.path.realpath(__file__))
ROOT_DIR = os.path.join(SCRIPT_DIR, '..')


def run_mcmc(img_xobs, img_yobs, fig_dir, d, lenses, pars, cov, nwalkers=100, nsteps=200, burn=20, fits_file=None, img_name='', mass_pos=None):
    names = img_xobs.keys()

    # Define likelihood function
    @pymc.observed
    def logL(value=0., tmp=pars):
        """ Calculate log-likelihood probability.
        Minimise the variance in the source position from all images. """
        for lens in lenses:
            lens.setPars()

        x_src, y_src = {}, {}
        lnlike_dict = {}

        for name in names:
            x_src[name], y_src[name] = pylens.getDeflections(lenses, [img_xobs[name], img_yobs[name]], d[name])
            lnlike_dict[name] = -0.5 * (x_src[name].var() + y_src[name].var())
        print(sum(lnlike_dict.values()), [lens.b for lens in lenses])
        return sum(lnlike_dict.values())

    # Run MCMC
    sampler = myEmcee.Emcee(pars+[logL], cov, nwalkers=nwalkers, nthreads=46)
    sampler.sample(nsteps)

    # Plot chains
    result = sampler.result()
    posterior, samples, _, best = result
    print("best", best)
    import pylab
    for j in range(nwalkers):
        pylab.plot(posterior[:, j])
    for i in range(len(pars)):
        pylab.figure()
        for j in range(nwalkers):
            pylab.plot(samples[:, j, i])

    # Trim initial samples (ie the burn-in) and concatenate chains
    samples = samples[burn:].reshape(((nsteps-burn) * nwalkers, len(pars)))

    # Plot parameter contours and mcmc chains
    param_names = ['$b_{lens%d}$' % i for i in range(len(pars))]
    c = ChainConsumer()
    c.add_chain(samples, parameters=param_names)
    c.configure(summary=True, cloud=True)
    fig = c.plotter.plot()
    fig.savefig(os.path.join(fig_dir, 'parameter_contours.png'), transparent=False)
    fig = c.plotter.plot_walks(convolve=100)
    fig.savefig(os.path.join(fig_dir, 'mcmc_walks.png'), transparent=False)

    if fits_file:
        fig = plt.figure(figsize=(13, 13))
        plot_image_and_contours(fits_file, samples, fig_dir, img_name, save=False, fig=fig)
        plot_source_and_pred_lens_positions(best, img_xobs, img_yobs, d, fig_dir, plotimage=False, mass_pos=mass_pos)


def plot_source_and_pred_lens_positions(pars, img_xobs, img_yobs, d, fig_dir, threshold=0.01, plotimage=False, fits_file=None, mass_pos=None):
    if plotimage:
        fig = plt.figure(figsize=(13, 13))
        plot_image(fits_file, fig)
    names = img_xobs.keys()

    lenses = []
    for b, (lx, ly) in zip(pars, mass_pos):
        LX = pymc.Uniform('lx', 0., 5000., value=lx)
        LY = pymc.Uniform('ly', 0., 5000., value=ly)
        LB = pymc.Uniform('lb', 0., 5000., value=b)
        lens = MassModels.SIS('', {'x': LX, 'y': LY, 'b': LB})
        lenses += [lens]

    colors = (col for col in ['#1f77b4', '#2ca02c', '#9467bd', '#17becf', '#e377c2', 'lime'])
    markers = (marker for marker in ['x', 'o', '*', '+', 'v', 'D'])

    x_src, y_src = {}, {}
    image_plane, image_coords_pred = {}, {}
    X1, Y1, Q1, P1, S1, srcs = {}, {}, {}, {}, {}, {}
    sa = (2000, 4500)
    pix_scale = 10.
    x, y = np.meshgrid(np.arange(sa[0], sa[1], pix_scale), np.arange(sa[0], sa[1], pix_scale))
    plt.xlim(sa[0], sa[1])
    plt.ylim(sa[0], sa[1])
    for name in names:
        x_src[name], y_src[name] = pylens.getDeflections(lenses, [img_xobs[name], img_yobs[name]], d[name])
        xs = np.median(x_src[name])
        ys = np.median(y_src[name])
        lnlike = -0.5 * (x_src[name].var() + y_src[name].var())
        print(name, xs, ys, lnlike)

        col = next(colors)
        plt.scatter(img_xobs[name], img_yobs[name], marker=next(markers), c='white', label="%s obs" % name, alpha=0.8)
        plt.scatter(x_src[name], y_src[name], marker='.', alpha=0.5, c=col, label="%s pred src" % name)

        # CALC IMG POS
        # Assume gaussian surface brightness at (xs, ys)
        X1[name] = pymc.Uniform('X1%s' % name, 0., 5000., value=xs)
        Y1[name] = pymc.Uniform('Y1%s' % name, 0., 5000., value=ys)
        Q1[name] = pymc.Uniform('Q1%s' % name, 0.2, 1., value=1.)
        P1[name] = pymc.Uniform('P1%s' % name, -180., 180., value=0.)
        S1[name] = pymc.Uniform('N1%s' % name, 0., 10000., value=6.)
        srcs[name] = SBObjects.Gauss('', {'x': X1[name], 'y': Y1[name], 'q': Q1[name], 'pa': P1[name], 'sigma': S1[name]})
        # Get Image plane
        x_src_all, y_src_all = pylens.getDeflections(lenses, [x, y], d=d[name])
        image_plane[name] = srcs[name].pixeval(x_src_all, y_src_all)
        image_indexes_pred = np.where(image_plane[name] > threshold)
        image_coords_pred[name] = np.array([x[image_indexes_pred], y[image_indexes_pred]])
        plt.scatter(image_coords_pred[name][0], image_coords_pred[name][1], marker='x', alpha=0.5, c=col, label="%s pred img" % name)

    print(x_src, y_src)
    plt.legend(loc='upper right')
    plt.savefig(os.path.join(fig_dir, 'image_with_contours_and_images.png'))


def macs0451_multiple_sources():
    fig_dir = os.path.join(ROOT_DIR, 'Figures/MACS0451_multiple_sis_model/')
    if not os.path.exists(fig_dir):
        os.makedirs(fig_dir)

    fits_file = '/home/djm241/PycharmProjects/StrongLensing/data/MACS0451/MACS0451_F110W.fits'
    if not os.path.isfile(fits_file):
        fits_file = '/Users/danmuth/PycharmProjects/StrongLensing/data/MACS0451/MACS0451_F110W.fits'

    img_name = ''
    z_lens = 0.43

    img_xobs, img_yobs, d = OrderedDict(), OrderedDict(), OrderedDict()
    img_xobs['A'] = np.array([2375.942110, 2378.5, 2379.816610, 2381.299088, 2384, 2385.927991, 2389.555816, 2457.694760, 2450.744242, 2442.833333, 2437.857924, 2433.064587, 2427.166666, 2424.099866, 2418.5, 2416.444081, 2462])
    img_yobs['A'] = np.array([3038.016677, 3024, 3012.367933, 2999.365293, 2983.5, 2970.435199, 2955.945319, 2737.545077, 2752.305849, 2766.166666, 2782.058508, 2795.293450, 2811.166666, 2823.079067, 2837.5, 2846.943113, 2728])
    d['A'] = scale_einstein_radius(z_lens=z_lens, z_src=2.013)

    img_xobs['B'] = np.array([3276.693717, 3261.382557, 3427.351819, 3417.043471, 3497.163625, 3486.371962])
    img_yobs['B'] = np.array([3482.795501, 3482.854177, 2592.719350, 2590.191799, 3075.107748, 3069.305065])
    d['B'] = scale_einstein_radius(z_lens=z_lens, z_src=1.405)

    img_xobs['11'] = np.array([3557.178601, 3548.271886, 3541.407488])
    img_yobs['11'] = np.array([3363.943860, 3375.285957, 3385.515024])
    d['11'] = scale_einstein_radius(z_lens=z_lens, z_src=2.06)

    img_xobs['31'] = np.array([2933.063074, 2943.400421, 2890.687234, 2878.906523])
    img_yobs['31'] = np.array([3393.715824, 3398.196336, 3044.431729, 3042.460964])
    d['31'] = scale_einstein_radius(z_lens=z_lens, z_src=1.904)

    img_xobs['41'] = np.array([3222.796159, 3227.700108])
    img_yobs['41'] = np.array([3550.903781, 3542.180780])
    d['41'] = scale_einstein_radius(z_lens=z_lens, z_src=1.810)

    img_xobs['C'] = np.array([3799.999263, 3794.5, 3863.057095, 3861.1])
    img_yobs['C'] = np.array([3358.972702, 3367.9, 3059.195359, 3069.9])
    d['C'] = scale_einstein_radius(z_lens=z_lens, z_src=2.0)

    mass_pos = [(3152.682, 2077.862),
                (2049.887, 2192.357),
                (2033.300, 2788.669),
                (3361.396, 2940.682),
                (3173.280, 2941.325),
                (3968.242, 2968.118),
                (4516.949, 2908.437),
                (3275.158, 3263.097),
                (2439.668, 3291.189),
                (2808.780, 3427.886)]

    lenses = []
    pars = []
    cov = []
    for lx, ly in mass_pos:
        LX = pymc.Uniform('lx', 1000., 5000., value=lx)
        LY = pymc.Uniform('ly', 1000., 5000., value=ly)
        LB = pymc.Uniform('lb', 10., 2000., value=1.50779124e+03)
        lens = MassModels.SIS('', {'x': LX, 'y': LY, 'b': LB})
        lenses += [lens]
        pars += [LB]
        cov += [400.]

    cov = np.array(cov)

    nwalkers = 1000
    nsteps = 1000
    burn = 50

    # best_lens = [  3.21895080e+03,   3.03726175e+03,   6.67192222e+02, 2.26586238e-01,   5.91357933e+00]
    # plot_source_and_pred_lens_positions(best_lens, img_xobs, img_yobs, d, fig_dir, threshold=0.01, plotimage=True, fits_file=fits_file)

    run_mcmc(img_xobs, img_yobs, fig_dir, d, lenses, pars, cov, nwalkers=nwalkers, nsteps=nsteps, burn=burn, fits_file=fits_file, img_name=img_name, mass_pos=mass_pos)


if __name__ == '__main__':
    # lensed_quasar()
    # macs0451()
    macs0451_multiple_sources()

    plt.show()
