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


def run_mcmc(img_xobs, img_yobs, fig_dir, d, lenses, srcs, pars, cov, nwalkers=100, nsteps=200, burn=20, fits_file=None, img_name='', mass_pos=None, flux_dependent_b=False, masses_flux=None, threshold=0.01, sa=(2000, 4500), pix_scale=10.):
    names = img_xobs.keys()
    x, y = np.meshgrid(np.arange(sa[0], sa[1], pix_scale), np.arange(sa[0], sa[1], pix_scale))

    # Define likelihood function
    @pymc.observed
    def logL(value=0., tmp=pars):
        """ Calculate log-likelihood probability.
        Minimise the variance in the source position from all images. """
        for key in srcs:
            srcs[key].setPars()
        for lens in lenses:
            lens.setPars()
            if lens.b < 0:
                return -1e99

        x_src, y_src = {}, {}
        lnlike_dict = {}
        for name in names:
            # Get Image plane
            x_src_all, y_src_all = pylens.getDeflections(lenses, [x, y], d=d[name])
            image_plane = srcs.pixeval(x_src_all, y_src_all)
            image_indexes_pred = np.where(image_plane > threshold)
            image_coords_pred = np.array([x[image_indexes_pred], y[image_indexes_pred]])
            if not image_coords_pred.size:  # If it's an empty list
                return -1e30
            img_xpred, img_ypred = image_coords_pred

            num_pred_points = image_coords_pred.shape[1]
            num_obs_points = 10*len(img_xobs[name])
            penalise_more_points_weight = (num_pred_points/num_obs_points)

            # Map each observed image to the single closest predicted image
            pred_arg = []
            for xo, yo in zip(img_xobs[name], img_yobs[name]):
                xdist = np.abs(img_xpred - xo)  # pixel distance between xobs and xpredicted
                ydist = np.abs(img_ypred - yo)
                dist = xdist**2 + ydist**2
                pred_arg.append(np.argmin(dist))  # The index of the pred_img that the given observed image is closest to
            pred_arg = np.array(pred_arg)

            # these pred images are the ones that are being compared with the list of the obs images
            img_xpred_compare = np.array([img_xpred[i] for i in pred_arg])
            img_ypred_compare = np.array([img_ypred[i] for i in pred_arg])

            lnlike_dict[name] = -0.5 * (np.sum((img_xpred_compare - img_xobs[name]) ** 2 + (img_ypred_compare - img_yobs[name]) ** 2)) * (1+penalise_more_points_weight)

        print(sum(lnlike_dict.values()), num_pred_points, num_obs_points) #, [lens.b for lens in lenses])
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
    param_names = []
    for name in names:
        param_names += ['$x%s_{src}$' % name, '$y%s_{src}$' % name, '$\sigma %s_{src}$' % name]
    param_names += ['$x_{lens}$', '$y_{lens}$', '$b_{lens}$', '$q_{lens}$', '$pa_{lens}$']
    param_names += [par.__name__ for par in pars[-2:]]
    c = ChainConsumer()
    c.add_chain(samples, parameters=param_names)
    c.configure(summary=True, cloud=True)
    fig = c.plotter.plot()
    fig.savefig(os.path.join(fig_dir, 'parameter_contours.png'), transparent=False)
    fig = c.plotter.plot_walks(convolve=100)
    fig.savefig(os.path.join(fig_dir, 'mcmc_walks.png'), transparent=False)

    if fits_file:
        fig = plt.figure('image_and_position', figsize=(13, 13))
        plot_image_and_contours(fits_file, samples, fig_dir, img_name, save=False, figname='image_and_position')
        plot_source_and_pred_lens_positions(best, img_xobs, img_yobs, d, fig_dir, threshold=threshold, plotimage=False, mass_pos=mass_pos, flux_dependent_b=flux_dependent_b, masses_flux=masses_flux, sa=sa, pix_scale=pix_scale)


def plot_source_and_pred_lens_positions(pars, img_xobs, img_yobs, d, fig_dir, threshold=0.01, plotimage=False, fits_file=None, mass_pos=None, flux_dependent_b=False, masses_flux=None, sa=(2000, 4500), pix_scale=10.):
    if plotimage:
        fig = plt.figure('image_and_position', figsize=(13, 13))
        plot_image(fits_file, figname='image_and_position')
    names = img_xobs.keys()

    xsrc, ysrc, sigsrc = {}, {}, {}
    for i, name in enumerate(names):
        xsrc[name], ysrc[name], sigsrc[name] = pars[3 * i: 3 * (i + 1)]

    xlens, ylens, blens, qlens, plens = pars[-7:-2]

    # Define source positions as a Guassian surface brightness profile
    X1, Y1, Q1, P1, S1, srcs = {}, {}, {}, {}, {}, {}
    for name in names:
        X1[name] = pymc.Uniform('X1%s' % name, 0., 5000., value=xsrc[name])
        Y1[name] = pymc.Uniform('Y1%s' % name, 0., 5000., value=ysrc[name])
        Q1[name] = pymc.Uniform('Q1%s' % name, 0.2, 1., value=1.)
        P1[name] = pymc.Uniform('P1%s' % name, -180., 180., value=0.)
        S1[name] = pymc.Uniform('N1%s' % name, 0., 10000., value=sigsrc[name])
        srcs[name] = SBObjects.Gauss('', {'x': X1[name], 'y': Y1[name], 'q': Q1[name],'pa': P1[name], 'sigma': S1[name]})

    # Define lens mass model
    LX = pymc.Uniform('lx', 0., 5000., value=xlens)
    LY = pymc.Uniform('ly', 0., 5000., value=ylens)
    LB = pymc.Uniform('lb', 0., 5000., value=blens)
    LQ = pymc.Uniform('lq', 0., 1., value=qlens)
    LP = pymc.Uniform('lp', -180., 180., value=plens)
    lens = MassModels.SIE('', {'x': LX, 'y': LY, 'b': LB, 'q': LQ, 'pa': LP})
    lenses = [lens]

    if flux_dependent_b:
        for (lx, ly), flux in zip(mass_pos, masses_flux):
            slope, intercept = pars[-2:]
            LX = pymc.Uniform('lx', 0., 5000., value=lx)
            LY = pymc.Uniform('ly', 0., 5000., value=ly)
            LB = slope * np.log(flux) + intercept
            lens = MassModels.SIS('', {'x': LX, 'y': LY, 'b': LB})
            lenses += [lens]
            print('lens einstein radius', lens.b)
    else:
        for b, (lx, ly) in zip(pars[-2:], mass_pos):
            LX = pymc.Uniform('lx', 0., 5000., value=lx)
            LY = pymc.Uniform('ly', 0., 5000., value=ly)
            LB = pymc.Uniform('lb', 0., 5000., value=b)
            lens = MassModels.SIS('', {'x': LX, 'y': LY, 'b': LB})
            lenses += [lens]
            print('lens einstein radius', lens.b)

    colors = (col for col in ['#1f77b4', '#2ca02c', '#9467bd', '#17becf', '#e377c2', '#ADFF2F'])
    markers = (marker for marker in ['x', 'o', '*', '+', 'v', 'D'])

    x_src, y_src = {}, {}
    image_plane, image_coords_pred = {}, {}
    X1, Y1, Q1, P1, S1, srcs = {}, {}, {}, {}, {}, {}
    x, y = np.meshgrid(np.arange(sa[0], sa[1], pix_scale), np.arange(sa[0], sa[1], pix_scale))
    plt.xlim(sa[0], sa[1])
    plt.ylim(sa[0], sa[1])
    for name in names:
        plt.figure('image_and_position')

        col = next(colors)
        plt.scatter(img_xobs[name], img_yobs[name], marker=next(markers), c='white', label="%s obs" % name, alpha=0.8)
        plt.scatter(x_src[name], y_src[name], marker='.', alpha=0.5, c=col, label="%s pred src" % name)

        # Get Image plane
        x_src_all, y_src_all = pylens.getDeflections(lenses, [x, y], d=d[name])
        image_plane[name] = srcs[name].pixeval(x_src_all, y_src_all)
        image_indexes_pred = np.where(image_plane[name] > threshold)
        image_coords_pred[name] = np.array([x[image_indexes_pred], y[image_indexes_pred]])

        image_plane_norm = (image_plane[name] - image_plane[name].min())/(image_plane[name].max() - image_plane[name].min())
        rgba = np.zeros((len(image_coords_pred[name][0]), 4))
        rgba[:, 0:3] = list(int(col[i:i+2], 16)/256. for i in (1, 3, 5))
        rgba[:, 3] = 0.8 * np.array(image_plane_norm[image_indexes_pred])
        plt.scatter(image_coords_pred[name][0], image_coords_pred[name][1], marker='x', color=rgba, label="%s pred img" % name)
        plt.figure(name)
        plt.title(name)
        plt.imshow(image_plane[name], interpolation='nearest', origin='lower')

    print(x_src, y_src)
    plt.figure('image_and_position')
    plt.legend(loc='upper right')
    plt.savefig(os.path.join(fig_dir, 'image_with_contours_and_images.png'))


def macs0451_multiple_sources():
    fig_dir = os.path.join(ROOT_DIR, 'Figures/penalise1MACS0451_multiple_sis_model_log')
    if not os.path.exists(fig_dir):
        os.makedirs(fig_dir)

    fits_file = '/home/djm241/PycharmProjects/StrongLensing/data/MACS0451/MACS0451_F110W.fits'
    if not os.path.isfile(fits_file):
        fits_file = '/Users/danmuth/PycharmProjects/StrongLensing/data/MACS0451/MACS0451_F110W.fits'

    sa = (2000, 4500)
    pix_scale = 10.

    img_name = ''
    z_lens = 0.43

    img_xobs, img_yobs, d, init = OrderedDict(), OrderedDict(), OrderedDict(), OrderedDict()
    img_xobs['A'] = np.array([2375.942110, 2378.5, 2379.816610, 2381.299088, 2384, 2385.927991, 2389.555816, 2457.694760, 2450.744242, 2442.833333, 2437.857924, 2433.064587, 2427.166666, 2424.099866, 2418.5, 2416.444081, 2462])
    img_yobs['A'] = np.array([3038.016677, 3024, 3012.367933, 2999.365293, 2983.5, 2970.435199, 2955.945319, 2737.545077, 2752.305849, 2766.166666, 2782.058508, 2795.293450, 2811.166666, 2823.079067, 2837.5, 2846.943113, 2728])
    d['A'] = scale_einstein_radius(z_lens=z_lens, z_src=2.013)
    init['A'] = {'xsrc': 3.49212756e+03, 'ysrc': 3.08381379e+03, 'sigsrc': 8.06085547e+00}

    img_xobs['B'] = np.array([3276.693717, 3261.382557, 3427.351819, 3417.043471, 3497.163625, 3486.371962])
    img_yobs['B'] = np.array([3482.795501, 3482.854177, 2592.719350, 2590.191799, 3075.107748, 3069.305065])
    d['B'] = scale_einstein_radius(z_lens=z_lens, z_src=1.405)
    init['B'] = {'xsrc': 3.00192697e+03, 'ysrc': 2.96770223e+03, 'sigsrc': 2.17208719e+00}

    img_xobs['11'] = np.array([3557.178601, 3548.271886, 3541.407488])
    img_yobs['11'] = np.array([3363.943860, 3375.285957, 3385.515024])
    d['11'] = scale_einstein_radius(z_lens=z_lens, z_src=2.06)
    init['11'] = {'xsrc': 3034, 'ysrc': 3053, 'sigsrc': 3.}

    img_xobs['31'] = np.array([2933.063074, 2943.400421, 2890.687234, 2878.906523])
    img_yobs['31'] = np.array([3393.715824, 3398.196336, 3044.431729, 3042.460964])
    d['31'] = scale_einstein_radius(z_lens=z_lens, z_src=1.904)
    init['31'] = {'xsrc': 3034, 'ysrc': 3053, 'sigsrc': 3.}

    img_xobs['41'] = np.array([3222.796159, 3227.700108])
    img_yobs['41'] = np.array([3550.903781, 3542.180780])
    d['41'] = scale_einstein_radius(z_lens=z_lens, z_src=1.810)
    init['41'] = {'xsrc': 3034, 'ysrc': 3053, 'sigsrc': 3.}

    img_xobs['C'] = np.array([3799.999263, 3794.5, 3863.057095, 3861.1])
    img_yobs['C'] = np.array([3358.972702, 3367.9, 3059.195359, 3069.9])
    d['C'] = scale_einstein_radius(z_lens=z_lens, z_src=2.0)
    init['C'] = {'xsrc': 3034, 'ysrc': 3053, 'sigsrc': 3.}

    names = img_xobs.keys()
    # Define source positions as a Guassian surface brightness profile
    X1, Y1, Q1, P1, S1, srcs = {}, {}, {}, {}, {}, {}
    pars, cov = [], []
    for name in names:
        X1[name] = pymc.Uniform('X1%s' % name, 2300., 3900., value=init[name]['xsrc'])
        Y1[name] = pymc.Uniform('Y1%s' % name, 2300., 3900., value=init[name]['ysrc'])
        Q1[name] = pymc.Uniform('Q1%s' % name, 1., 1., value=1.)
        P1[name] = pymc.Uniform('P1%s' % name, 0., 0., value=0.)
        S1[name] = pymc.Uniform('N1%s' % name, 0., 10., value=init[name]['sigsrc'])
        srcs[name] = SBObjects.Gauss('', {'x': X1[name], 'y': Y1[name], 'q': Q1[name],'pa': P1[name], 'sigma': S1[name]})

        pars += [X1[name], Y1[name], S1[name]]  # List of parameters
        cov += [400., 400., 1.5]  # List of initial `scatter' for emcee

    # Define overall lens SIE mass model
    LX = pymc.Uniform('lx', 3100., 3500., value=3.29213154e+03)
    LY = pymc.Uniform('ly', 2850., 3150., value=3.04899548e+03)
    LB = pymc.Uniform('lb', 100., 3000., value=2.56554040e+02)
    LQ = pymc.Uniform('lq', 0.01, 1., value=1.24379589e-01)
    LP = pymc.Uniform('lp', -180., 180., value=5.67806821e+00)
    lens = MassModels.SIE('', {'x': LX, 'y': LY, 'b': LB, 'q': LQ, 'pa': LP})
    lenses = [lens]
    pars = [LX, LY, LB, LQ, LP]
    cov = [400., 400., 400., 0.3, 50.]

    # Define individual lens models for each major light source that are in the same section on the color magnitude plot
    masses_pos = [(2965.5288, 1933.9563), (2049.9021, 2192.387), (3141.9883, 2217.7527), (4607.7075, 2167.5984), (3549.3555, 2162.8872), (2068.0742, 2216.7585), (4268.5576, 2319.8298), (4663.2427, 2328.0593), (3323.5115, 2425.0659), (2221.5608, 2512.1736), (2950.5073, 2629.0552), (2981.6868, 2652.0073), (2033.7075, 2788.6877), (1949.4614, 2728.1226), (4241.4492, 2750.6653), (3361.4163, 2940.6357), (3173.2246, 2941.2471), (4516.8184, 2908.7585), (3382.9978, 2781.5076), (2898.4788, 2840.1252), (1942.7456, 2877.2471), (2899.8923, 2899.4895), (4041.6626, 2880.5015), (4162.7109, 2886.1875), (3989.6919, 2935.3879), (3303.6855, 2938.9148), (3968.1621, 2968.2993), (4613.0122, 2897.4099), (4649.1987, 2862.4167), (3261.9656, 2921.2605), (3386.9814, 2911.084), (3414.7791, 2936.4338), (2298.7419, 2965.8403), (3427.4221, 3072.7476), (3227.8262, 3095.5388), (3275.3184, 3263.1567), (4284.9653, 3240.2017), (3227.1746, 3250.7136), (2487.8906, 3230.6677), (2559.6587, 3186.9429), (2017.4875, 3195.2239), (2740.4167, 3208.8381), (3146.053, 3237.4194), (2401.4067, 3203.291), (2808.7485, 3427.9224), (4322.1392, 3317.6938), (3397.1904, 3356.5364), (3074.5276, 3448.4421), (3954.7253, 3590.8535), (3116.0625, 3523.1008), (2640.4075, 3624.4426), (3082.8025, 3548.231), (4583.7495, 3697.9534), (2768.9014, 3617.7961), (2639.042, 3615.2815), (2255.7852, 3900.9021), (2476.9795, 4028.5874)]
    masses_flux = np.array([378.1141, 1527.1989999999998, 773.9109, 294.5334, 275.1775, 346.9561, 780.823, 201.5256, 244.387, 391.2144, 954.7488, 273.3743, 2777.741, 636.8357, 317.5356, 2693.8559999999998, 3436.873, 1680.48, 219.1088, 591.2639, 624.871, 590.7428, 695.0563, 485.1097, 901.0128, 947.5237, 939.0687, 290.4246, 226.5177, 717.8644, 325.5481, 285.941, 266.1354, 232.6961, 257.5104, 1943.4389999999999, 957.1483, 1277.3110000000001, 821.7504, 362.3544, 343.2694, 381.5304, 260.1786, 306.7749, 1675.8770000000002, 442.8735, 549.5866, 910.3147, 979.7011, 389.7068, 707.3599, 433.32599999999996, 1193.312, 204.4583, 218.7119, 451.3262, 333.3452])
    masses_flux = masses_flux/200.
    print(masses_flux)

    flux_dependent_b = True
    if flux_dependent_b:
        # ------> b_sis = slope * flux ** 8 + intercept <-------- #
        slope = pymc.Uniform('slope', 0., 1000., value=1.53454884e+01)
        intercept = pymc.Uniform('intercept', -100., 1000., value=-7.22002250e-02)
        # n = pymc.Uniform('n', 0., 10., value=4.)
        pars += [slope, intercept]
        cov += [5., 5.]
        cov = np.array(cov)

        for (lx, ly), flux in zip(masses_pos, masses_flux):
            LX = pymc.Uniform('lx', 1000., 5000., value=lx)
            LY = pymc.Uniform('ly', 1000., 5000., value=ly)
            LB = slope * np.log(flux) + intercept
            lens = MassModels.SIS('', {'x': LX, 'y': LY, 'b': LB})
            lenses += [lens]
    else:
        for lx, ly in masses_pos:
            LX = pymc.Uniform('lx', 1000., 5000., value=lx)
            LY = pymc.Uniform('ly', 1000., 5000., value=ly)
            LB = pymc.Uniform('lb', 0., 2000., value=50.)
            lens = MassModels.SIS('', {'x': LX, 'y': LY, 'b': LB})
            lenses += [lens]
            pars += [LB]
            cov += [30.]
        cov = np.array(cov)

    nwalkers = 200
    nsteps = 500
    burn = 100

    best_lens = [  3.29213154e+03,   3.04899548e+03,   2.56554040e+02,
         1.24379589e-01,   5.67806821e+00,   1.53454884e+01,
        -7.22002250e-02]
    # plot_source_and_pred_lens_positions(best_lens, img_xobs, img_yobs, d, fig_dir, threshold=0.01, plotimage=True, fits_file=fits_file, mass_pos=masses_pos, flux_dependent_b=flux_dependent_b, masses_flux=masses_flux, sa=sa, pix_scale=pix_scale)

    run_mcmc(img_xobs, img_yobs, fig_dir, d, lenses, srcs, pars, cov, nwalkers=nwalkers, nsteps=nsteps, burn=burn, fits_file=fits_file, img_name=img_name, mass_pos=masses_pos, flux_dependent_b=flux_dependent_b, masses_flux=masses_flux, threshold=0.01, sa=sa, pix_scale=pix_scale)


if __name__ == '__main__':
    macs0451_multiple_sources()

    plt.show()

# ('lens einstein radius', 9.7010135098293517)
# ('lens einstein radius', 31.123232742829785)
# ('lens einstein radius', 20.692384547683861)
# ('lens einstein radius', 5.8676553665962237)
# ('lens einstein radius', 4.8245294436482187)
# ('lens einstein radius', 8.3813361398254198)
# ('lens einstein radius', 20.8288326555842)
# ('lens einstein radius', 0.044410969011899837)
# ('lens einstein radius', 3.0035829982694477)
# ('lens einstein radius', 10.223676938654723)
# ('lens einstein radius', 23.914806967376407)
# ('lens einstein radius', 4.7236416718172345)
# ('lens einstein radius', 40.30294487989763)
# ('lens einstein radius', 17.700857442448438)
# ('lens einstein radius', 7.0215981753773287)
# ('lens einstein radius', 39.832384945140078)
# ('lens einstein radius', 43.570368256933044)
# ('lens einstein radius', 32.59093860703485)
# ('lens einstein radius', 1.3280916087498631)
# ('lens einstein radius', 16.561466504599888)
# ('lens einstein radius', 17.409807756475612)
# ('lens einstein radius', 16.547936065994946)
# ('lens einstein radius', 19.043298494350157)
# ('lens einstein radius', 13.52478772503275)
# ('lens einstein radius', 23.025861155692617)
# ('lens einstein radius', 23.798237735939018)
# ('lens einstein radius', 23.660691360376013)
# ('lens einstein radius', 5.6520755134669791)
# ('lens einstein radius', 1.8384008772105862)
# ('lens einstein radius', 19.538771227963167)
# ('lens einstein radius', 7.4040121261717227)
# ('lens einstein radius', 5.4133228194817944)
# ('lens einstein radius', 4.3118191478825478)
# ('lens einstein radius', 2.2513513904194489)
# ('lens einstein radius', 3.8062605859042975)
# ('lens einstein radius', 34.82185984251943)
# ('lens einstein radius', 23.953325273447231)
# ('lens einstein radius', 28.38132769759806)
# ('lens einstein radius', 21.612806155764652)
# ('lens einstein radius', 9.047705987286534)
# ('lens einstein radius', 8.217404931141175)
# ('lens einstein radius', 9.8390389824909121)
# ('lens einstein radius', 3.9644451129475469)
# ('lens einstein radius', 6.4925519630465844)
# ('lens einstein radius', 32.548848133608679)
# ('lens einstein radius', 12.126953760491178)
# ('lens einstein radius', 15.439771303783884)
# ('lens einstein radius', 23.18347313224303)
# ('lens einstein radius', 24.310709150642417)
# ('lens einstein radius', 10.164426691360866)
# ('lens einstein radius', 19.312561778293507)
# ('lens einstein radius', 11.792516569293539)
# ('lens einstein radius', 27.337459249020956)
# ('lens einstein radius', 0.26611678913053077)
# ('lens einstein radius', 1.3002691415663299)
# ('lens einstein radius', 12.417078379813411)
# ('lens einstein radius', 7.7672147674176211)
