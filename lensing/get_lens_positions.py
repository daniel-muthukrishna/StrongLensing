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


def run_mcmc(img_xobs, img_yobs, fig_dir, d, lenses, pars, cov, nwalkers=100, nsteps=200, burn=20, fits_file=None, img_name=''):
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
        print(sum(lnlike_dict.values()), float(lens.x), float(lens.y), float(lens.b), float(lens.q), float(lens.pa))
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
    param_names = ['$x_{lens}$', '$y_{lens}$', '$b_{lens}$', '$q_{lens}$', '$pa_{lens}$']
    if len(pars) == 7:
        param_names += ['$b_{shear}$', '$pa_{shear}$']
    c = ChainConsumer()
    c.add_chain(samples, parameters=param_names)
    c.configure(summary=True, cloud=True)
    fig = c.plotter.plot()
    fig.savefig(os.path.join(fig_dir, 'parameter_contours%s.png'), transparent=False)
    fig = c.plotter.plot_walks(convolve=100)
    fig.savefig(os.path.join(fig_dir, 'mcmc_walks%s.png'), transparent=False)

    if fits_file:
        plot_image_and_contours(fits_file, samples, fig_dir, img_name, save=False)
        plot_source_and_pred_lens_positions(best, img_xobs, img_yobs, d, fig_dir)

    plt.show()


def plot_source_and_pred_lens_positions(pars, img_xobs, img_yobs, d, fig_dir, threshold=0.01):
    names = img_xobs.keys()
    try:
        xlens, ylens, blens, qlens, plens = pars
        bshear, pshear = 0., 0.
    except ValueError:  # includes shear
        xlens, ylens, blens, qlens, plens, bshear, pshear = pars

    # Define lens mass model
    LX = pymc.Uniform('lx', 0., 5000., value=xlens)
    LY = pymc.Uniform('ly', 0., 5000., value=ylens)
    LB = pymc.Uniform('lb', 0., 5000., value=blens)
    LQ = pymc.Uniform('lq', 0.2, 1., value=qlens)
    LP = pymc.Uniform('lp', -180., 180., value=plens)
    XB = pymc.Uniform('xb', -200., 200., value=bshear)
    XP = pymc.Uniform('xp', -180., 180., value=pshear)
    lens = MassModels.SIE('', {'x': LX, 'y': LY, 'b': LB, 'q': LQ, 'pa': LP})
    lenses = [lens]
    if len(pars) == 7:
        shear = MassModels.ExtShear('',{'x':LX,'y':LY,'b':XB,'pa':XP})
        lenses += [shear]

    colors = (col for col in ['#1f77b4', '#2ca02c', '#9467bd', '#17becf', '#e377c2'])
    markers = (marker for marker in ['x', 'o', '*', '+', 'v'])

    x_src, y_src = {}, {}
    image_plane, image_coords_pred = {}, {}
    X1, Y1, Q1, P1, S1, srcs = {}, {}, {}, {}, {}, {}
    print(float(lens.x), float(lens.y), float(lens.b), float(lens.q), float(lens.pa))
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
        print(float(lens.x), float(lens.y), float(lens.b), float(lens.q), float(lens.pa))
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
    fig_dir = os.path.join(ROOT_DIR, 'Figures/MACS0451_2/')
    if not os.path.exists(fig_dir):
        os.makedirs(fig_dir)

    fits_file = '/home/djm241/PycharmProjects/StrongLensing/data/MACS0451/MACS0451_F110W.fits'
    if not os.path.isfile(fits_file):
        fits_file = '/Users/danmuth/PycharmProjects/StrongLensing/data/MACS0451/MACS0451_F110W.fits'

    img_name = '_multiple_sources'
    z_lens = 0.43

    img_xobs, img_yobs, d = OrderedDict(), OrderedDict(), OrderedDict()
    img_xobs['A'] = np.array([2375.942110, 2378.5, 2379.816610, 2381.299088, 2384, 2385.927991, 2389.555816, 2457.694760, 2450.744242, 2442.833333, 2437.857924, 2433.064587, 2427.166666, 2424.099866, 2418.5, 2416.444081, 2462])
    img_yobs['A'] = np.array([3038.016677, 3024, 3012.367933, 2999.365293, 2983.5, 2970.435199, 2955.945319, 2737.545077, 2752.305849, 2766.166666, 2782.058508, 2795.293450, 2811.166666, 2823.079067, 2837.5, 2846.943113, 2728])
    d['A'] = scale_einstein_radius(z_lens=z_lens, z_src=2.01)

    img_xobs['B'] = np.array([3276.693717, 3261.382557, 3427.351819, 3417.043471, 3497.163625, 3486.371962])
    img_yobs['B'] = np.array([3482.795501, 3482.854177, 2592.719350, 2590.191799, 3075.107748, 3069.305065])
    d['B'] = scale_einstein_radius(z_lens=z_lens, z_src=1.405)

    img_xobs['11'] = np.array([3557.178601, 3548.271886, 3541.407488]) #3490.676982, 3498.161408
    img_yobs['11'] = np.array([3363.943860, 3375.285957, 3385.515024]) #3447.666750, 3440.809843
    d['11'] = scale_einstein_radius(z_lens=z_lens, z_src=2.06)

    img_xobs['31'] = np.array([2933.063074, 2943.400421]) #2890.687234, 2878.906523
    img_yobs['31'] = np.array([3393.715824, 3398.196336]) #3044.431729, 3042.460964
    d['31'] = scale_einstein_radius(z_lens=z_lens, z_src=1.904)

    img_xobs['41'] = np.array([3222.796159, 3227.700108])
    img_yobs['41'] = np.array([3550.903781, 3542.180780])
    d['41'] = scale_einstein_radius(z_lens=z_lens, z_src=1.810)

    # img_xobs['C'] = np.array([3799.999263, 3794.5, 3863.057095, 3861.1])
    # img_yobs['C'] = np.array([3358.972702, 3367.9, 3059.195359, 3069.9])
    # d['C'] = scale_einstein_radius(z_lens=z_lens, z_src=?)

    # Define lens mass model
    LX = pymc.Uniform('lx', 2400., 4000., value=3.13876545e+03)
    LY = pymc.Uniform('ly', 2000., 3700., value=2.97884105e+03)
    LB = pymc.Uniform('lb', 10., 2000., value=1.50779124e+03)
    LQ = pymc.Uniform('lq', 0.2, 1., value=4.90424861e-01)
    LP = pymc.Uniform('lp', -180., 180., value=1.04010643e+02)
    XB = pymc.Uniform('xb', -200., 200., value=0.)
    XP = pymc.Uniform('xp', -180., 180., value=0.)
    lens = MassModels.SIE('', {'x': LX, 'y': LY, 'b': LB, 'q': LQ, 'pa': LP})
    shear = MassModels.ExtShear('', {'x': LX, 'y': LY, 'b': XB, 'pa': XP})
    lenses = [lens]
    pars = [LX, LY, LB, LQ, LP]
    cov = [400., 400., 400., 0.3, 50.]

    # lenses += [shear]
    # pars += [XB, XP]
    # cov += [5., 50.]

    cov = np.array(cov)

    nwalkers = 2000
    nsteps = 6000
    burn = 200

    best_lens = [3125.402837830007, 3069.6947816268207, 181.7467825143057, 0.5324488078055447, 87.2235065814847]
    # plot_source_and_pred_lens_positions(best_lens, img_xobs, img_yobs, d, fig_dir, threshold=0.01)

    run_mcmc(img_xobs, img_yobs, fig_dir, d, lenses, pars, cov, nwalkers=nwalkers, nsteps=nsteps, burn=burn, fits_file=fits_file, img_name=img_name)


if __name__ == '__main__':
    # lensed_quasar()
    # macs0451()
    macs0451_multiple_sources()





# def lensed_quasar():
#     # Image positions taken from GAIA+SExtractor for WFI2033_F814W.fits
#     # x_img = np.array([27.1051, 69.3759, 71.0171, 56.8204])
#     # y_img = np.array([33.6946, 28.1791, 58.9579, 61.2290])
#     x_img = np.array([27.1051, 69.3759, 27.5051, 69.8759, 28, 70, 26.5, 68.8])
#     y_img = np.array([33.6946, 28.1791, 34, 28, 33.2, 28.5, 33, 27.6])
#     fig_dir = 'Figures/lensed_quasar/'
#
#     ndim, nwalkers = 5, 1000
#     n_steps = 2000
#
#     # Starting positions for the MCMC walkers sampled from a uniform distribution
#     initial = OrderedDict()
#     initial[r'$x_{lens}$'] = np.random.uniform(low=40, high=70, size=nwalkers)
#     initial[r'$y_{lens}$'] = np.random.uniform(low=30, high=50, size=nwalkers)
#     initial[r'$\theta$'] = np.random.uniform(low=-np.pi, high=np.pi, size=nwalkers)
#     initial[r'$b$'] = np.random.uniform(low=10, high=40, size=nwalkers)
#     initial[r'$q$'] = np.random.uniform(low=0.2, high=1., size=nwalkers)
#
#     def lnprior(pars):
#         xsource, ysource, theta, b, q = pars
#         if (28 < xsource < 69) and (28 < ysource < 60) and (10 < b < 40) and (-np.pi < theta < np.pi) and (0. < q < 1):
#             return 0.0
#         else:
#             return -np.inf
#
#     run_mcmc(x_img, y_img, fig_dir, d=1, nwalkers=nwalkers, nsteps=n_steps, prior_func=lnprior, initial_params=initial)
#
#
# def macs0451():
#     fig_dir = 'Figures/MACS0451/'
#     nwalkers = 1500
#     steps = 3000
#
#     # ImageA Positions
#     img_name = '_ImageA'
#     x_img = np.array([2375.942110, 2378.5, 2379.816610, 2381.299088, 2384, 2385.927991, 2389.555816, 2457.694760, 2450.744242, 2442.833333, 2437.857924, 2433.064587, 2427.166666, 2424.099866, 2418.5, 2416.444081, 2462])
#     y_img = np.array([3038.016677, 3024, 3012.367933, 2999.365293, 2983.5, 2970.435199, 2955.945319, 2737.545077, 2752.305849, 2766.166666, 2782.058508, 2795.293450, 2811.166666, 2823.079067, 2837.5, 2846.943113, 2728])
#
#     # # Image 1.1 Positions
#     # img_name = '_Image1.1'
#     # x_img = np.array([3557.178601, 3548.271886, 3541.407488])
#     # y_img = np.array([3363.943860, 3375.285957, 3385.515024])
#
#     # # Image 6.2 Positions
#     # img_name = '_Image6.2'
#     # x_img = np.array([3486.371962])
#     # y_img = np.array([3069.305065])
#
#     # # Image 4.1 Positions
#     # img_name = '_Image4.1'
#     # x_img = np.array([3222.796159, 3227.700108])
#     # y_img = np.array([3550.903781, 3542.180780])
#
#     # # Image 3.1 Positions
#     # img_name = '_Image3.1'
#     # x_img = np.array([2933.063074, 3393.715824])
#     # y_img = np.array([2943.400421, 3398.196336])
#
#
#
#     # Starting positions for the MCMC walkers sampled from a uniform distribution
#     initial = OrderedDict()
#     initial[r'$x_{lens}$'] = np.random.uniform(low=2400, high=4000, size=nwalkers)
#     initial[r'$y_{lens}$'] = np.random.uniform(low=2000, high=3700, size=nwalkers)
#     initial[r'$\theta$'] = np.random.uniform(low=-np.pi, high=np.pi, size=nwalkers)
#     initial[r'$b$'] = np.random.uniform(low=10, high=2000, size=nwalkers)
#     initial[r'$q$'] = np.random.uniform(low=0.2, high=1., size=nwalkers)
#
#     def lnprior(pars):
#         xsource, ysource, theta, b, q = pars
#         if (2400 < xsource < 4000) and (2000 < ysource < 3700) and (10 < b < 2000) and (-np.pi < theta < np.pi) and (0. < q < 1):
#             return 0.0
#         else:
#             return -np.inf
#
#     fits_file = '/Users/danmuth/PycharmProjects/StrongLensing/data/MACS0451/MACS0451_F110W.fits'
#
#     run_mcmc(x_img, y_img, fig_dir, nwalkers=nwalkers, nsteps=steps, prior_func=lnprior, initial_params=initial, fits_file=fits_file, img_name=img_name)
