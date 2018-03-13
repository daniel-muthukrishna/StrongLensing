import os
import sys
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import emcee
from collections import OrderedDict
from chainconsumer import ChainConsumer
import pymc
from dist_ang import scale_einstein_radius
from mass_model import pred_positions
from pylens import pylens, MassModels
from imageSim import SBObjects
import myEmcee
from image_overplot_contours import plot_image

SCRIPT_DIR = os.path.dirname(os.path.realpath(__file__))
ROOT_DIR = os.path.join(SCRIPT_DIR, '..')


def plot_img_pos(pars, pix_scale=1., threshold=0.8, fits_file=None, img_xobs=None, img_yobs=None, d=None):
    xsrc, ysrc, sigsrc = {}, {}, {}
    xsrc['A'], ysrc['A'], sigsrc['A'], xsrc['B'], ysrc['B'], sigsrc['B'], xlens, ylens, blens, qlens, plens = pars
    fig_dir = 'Figures/MACS0451/'
    sa = (2000, 4500)  # search area is 2000 pixels to 5000 pixels
    names = xsrc.keys()

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
    LQ = pymc.Uniform('lq', 0.2, 1., value=qlens)
    LP = pymc.Uniform('lp', -180., 180., value=plens)
    XB = pymc.Uniform('xb', -0.2, 0.2, value=0.)
    XP = pymc.Uniform('xp', -180., 180., value=0.)
    lens = MassModels.SIE('', {'x': LX, 'y': LY, 'b': LB, 'q': LQ, 'pa': LP})
    shear = MassModels.ExtShear('',{'x':LX,'y':LY,'b':XB,'pa':XP})
    lenses = [lens]

    x, y = np.meshgrid(np.arange(sa[0], sa[1], pix_scale), np.arange(sa[0], sa[1], pix_scale))

    image_plane, image_coords_pred = {}, {}
    for name in names:
        x_src, y_src = pylens.getDeflections(lenses, [x, y], d=d[name])
        image_plane[name] = srcs[name].pixeval(x_src, y_src)
        plt.figure()
        plt.imshow(image_plane[name], interpolation='nearest', origin='lower')
        plt.xlabel('%dx - %d pixels' % (pix_scale, sa[0]))
        plt.ylabel('%dy - %d pixels' % (pix_scale, sa[0]))
        plt.savefig(os.path.join(ROOT_DIR, fig_dir, 'image_plane%s.png' % name))
        image_coords_pred[name] = np.add(np.multiply(np.where(image_plane[name] > threshold)[::-1], pix_scale), sa[0])
        print(name, image_coords_pred[name])

    colors = (col for col in ['#1f77b4', '#2ca02c', '#9467bd', '#17becf', '#e377c2'])
    fig = plt.figure()
    plot_image(fits_file, fig)
    plt.xlim(sa[0], sa[1])
    plt.ylim(sa[0], sa[1])
    for name in names:
        plt.scatter(image_coords_pred[name][0], image_coords_pred[name][1], marker='.', alpha=0.3, c=next(colors))
    plt.savefig(os.path.join(ROOT_DIR, fig_dir, 'image_with_predicted_image_plane.png'))


    # print(pylens.getImgPos(x0=0, y0=0, b=21.6, sx=54.83, sy=38.98, lenses=lenses))


def get_macs0451_img_pos(pix_scale=1., threshold=0.8, fits_file=None, img_xobs=None, img_yobs=None, d=None):
    fig_dir = 'Figures/MACS0451/'
    sa = (2000, 4500)  # search area is 2000 pixels to 5000 pixels
    names = img_xobs.keys()
    # Define source positions as a Guassian surface brightness profile
    X1, Y1, Q1, P1, S1, srcs = {}, {}, {}, {}, {}, {}
    pars, cov = [], []
    for name in names:
        X1[name] = pymc.Uniform('X1%s' % name, 2300., 3900., value=3034.)
        Y1[name] = pymc.Uniform('Y1%s' % name, 2400., 3600., value=3053.)
        Q1[name] = pymc.Uniform('Q1%s' % name, 1., 1., value=1.)
        P1[name] = pymc.Uniform('P1%s' % name, 0., 0., value=0.)
        S1[name] = pymc.Uniform('N1%s' % name, 0., 10., value=3.)
        srcs[name] = SBObjects.Gauss('', {'x': X1[name], 'y': Y1[name], 'q': Q1[name],'pa': P1[name], 'sigma': S1[name]})

        pars += [X1[name], Y1[name], S1[name]]  # List of parameters
        cov += [400., 400., 1.5]  # List of initial `scatter' for emcee

    # Define lens mass model
    LX = pymc.Uniform('lx', 2900., 3400., value=3034)
    LY = pymc.Uniform('ly', 2600., 3500., value=2981)
    LB = pymc.Uniform('lb', 10., 2000., value=297.)
    LQ = pymc.Uniform('lq', 0.2, 1., value=0.333)
    LP = pymc.Uniform('lp', -180., 180., value=85.)
    XB = pymc.Uniform('xb', -0.2, 0.2, value=0.)
    XP = pymc.Uniform('xp', -180., 180., value=0.)
    lens = MassModels.SIE('', {'x': LX, 'y': LY, 'b': LB, 'q': LQ, 'pa': LP})
    # shear = MassModels.ExtShear('', {'x': LX, 'y': LY, 'b': XB, 'pa': XP})
    lenses = [lens]
    pars += [LX, LY, LB, LQ, LP]
    cov += [400., 400., 400., 0.3, 50.]
    cov = np.array(cov)

    # Get grid of x and y points
    x, y = np.meshgrid(np.arange(sa[0], sa[1], pix_scale), np.arange(sa[0], sa[1], pix_scale))

    # MCMC setup
    nwalkers = 100
    nsteps = 1000

    # Define likelihood function
    @pymc.observed
    def logL(value=0., tmp=pars):
        for key in srcs:
            srcs[key].setPars()
        for lens in lenses:
            lens.setPars()
        lnlike = {}

        for name in names:
            # Calculate deflections
            x_src, y_src = pylens.getDeflections(lenses, [x, y], d[name])

            # Get list of predicted image coordinates
            image_plane = srcs[name].pixeval(x_src, y_src)
            image_coords_pred = np.add(np.multiply(np.where(image_plane > threshold)[::-1], pix_scale), sa[0])  # Only if brightness > threshold

            if not image_coords_pred.size:  # If it's an empty list
                return -1e30
            img_xpred, img_ypred = image_coords_pred

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

            lnlike[name] = -0.5 * (np.sum((img_xpred_compare - img_xobs[name]) ** 2 + (img_ypred_compare - img_yobs[name]) ** 2))

            print(name, img_xpred_compare, lnlike[name], float(srcs[name].x), float(srcs[name].y), float(srcs[name].sigma), float(lens.x), float(lens.y), float(lens.b), float(lens.q), float(lens.pa))
            print(name, img_ypred_compare, lnlike[name], float(srcs[name].x), float(srcs[name].y), float(srcs[name].sigma), float(lens.x), float(lens.y), float(lens.b), float(lens.q), float(lens.pa))

        return sum(lnlike.values())

    # Run MCMC
    sampler = myEmcee.Emcee(pars+[logL], cov, nwalkers=nwalkers, nthreads=44)
    sampler.sample(nsteps)

    # Plot chains
    result = sampler.result()
    posterior, samples, _, best = result
    print best
    import pylab
    for j in range(nwalkers):
        pylab.plot(posterior[:, j])
    for i in range(len(pars)):
        pylab.figure()
        for j in range(nwalkers):
            pylab.plot(samples[:, j, i])

    # Trim initial samples (ie the burn-in) and concatenate chains
    burn = 200
    samples = samples[burn:].reshape(((nsteps-burn) * nwalkers, len(pars)))

    # Get best fit parameters
    samples_exp = samples.copy()
    samples_exp[:, 2] = np.exp(samples_exp[:, 2])
    best_fits = list(map(lambda v: (v[1]), zip(*np.percentile(samples_exp, [16, 50, 84], axis=0))))
    print('bestfits')
    print(best_fits)

    # Plot parameter contours and mcmc chains
    param_names = []
    for name in names:
        param_names += ['$x%s_{src}$' % name, '$y%s_{src}$' % name, '$\sigma %s_{src}$' % name]
    param_names += ['$x_{lens}$', '$y_{lens}$', '$b_{lens}$', '$q_{lens}$', '$pa_{lens}$']
    c = ChainConsumer()
    c.add_chain(samples, parameters=param_names)
    c.configure(summary=True, cloud=True)
    fig = c.plotter.plot()
    fig.savefig(os.path.join(ROOT_DIR, fig_dir, 'source_pos_parameter_contours.png'), transparent=False)
    fig = c.plotter.plot_walks(convolve=100)
    fig.savefig(os.path.join(ROOT_DIR, fig_dir, 'source_pos_mcmc_walks.png'), transparent=False)

    b = best
    print(b)
    plot_img_pos(pars=b, pix_scale=pix_scale, threshold=threshold, fits_file=fits_file, img_xobs=img_xobs, img_yobs=img_yobs, d=d)


def main():
    fits_file_macs0451 = '/home/djm241/PycharmProjects/StrongLensing/data/MACS0451/MACS0451_F110W.fits'

    # Observed Image positions
    img_xobs, img_yobs, d = {}, {}, {}
    z_lens = 0.43
    img_xobs['A'] = np.array([2375.942110, 2378.5, 2379.816610, 2381.299088, 2384, 2385.927991, 2389.555816, 2457.694760, 2450.744242, 2442.833333, 2437.857924, 2433.064587, 2427.166666, 2424.099866, 2418.5, 2416.444081, 2462])
    img_yobs['A'] = np.array([3038.016677, 3024, 3012.367933, 2999.365293, 2983.5, 2970.435199, 2955.945319, 2737.545077, 2752.305849, 2766.166666, 2782.058508, 2795.293450, 2811.166666, 2823.079067, 2837.5, 2846.943113, 2728])

    d['A'] = scale_einstein_radius(z_lens=z_lens, z_src=2.01)
    img_xobs['B'] = np.array([3276.693717, 3261.382557, 3427.351819, 3417.043471])
    img_yobs['B'] = np.array([3482.795501, 3482.854177, 2592.719350, 2590.191799])
    d['B'] = scale_einstein_radius(z_lens=z_lens, z_src=1.405)

    img_xobs['11'] = np.array([3557.178601, 3548.271886, 3541.407488])
    img_yobs['11'] = np.array([3363.943860, 3375.285957, 3385.515024])
    d['11'] = scale_einstein_radius(z_lens=z_lens, z_src=2.06)

    img_xobs['31'] = np.array([2933.063074, 3393.715824])
    img_yobs['31'] = np.array([2943.400421, 3398.196336])
    d['31'] = scale_einstein_radius(z_lens=z_lens, z_src=1.904)

    img_xobs['41'] = np.array([3222.796159, 3227.700108])
    img_yobs['41'] = np.array([3550.903781, 3542.180780])
    d['41'] = scale_einstein_radius(z_lens=z_lens, z_src=1.810)

    pix_scale = 10.
    threshold = 0.01
    pars = [3.49212756e+03,3.08381379e+03,8.06085547e+00,3.00192697e+03 ,2.96770223e+03,2.17208719e+00,3.13876545e+03,2.97884105e+03 ,1.50779124e+03,4.90424861e-01,1.04010643e+02]
    # plot_img_pos(pars, pix_scale=pix_scale, threshold=threshold, fits_file=fits_file_macs0451, img_xobs=img_xobs, img_yobs=img_yobs, d=d)
    get_macs0451_img_pos(pix_scale=pix_scale, threshold=threshold, fits_file=fits_file_macs0451, img_xobs=img_xobs, img_yobs=img_yobs, d=d)

    plt.show()


if __name__=='__main__':
    main()


# This was a good model for just sources A and B
# pars = [3.49212756e+03,3.08381379e+03,8.06085547e+00,3.00192697e+03 ,2.96770223e+03,2.17208719e+00,3.13876545e+03,2.97884105e+03 ,1.50779124e+03,4.90424861e-01,1.04010643e+02]
