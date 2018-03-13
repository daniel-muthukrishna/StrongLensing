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
    xsrcA, ysrcA, sigsrcA, xlens, ylens, blens, qlens, plens = pars
    fig_dir = 'Figures/lensed_quasar/'
    sa = (0, 100)  # search area is 2000 pixels to 5000 pixels

    # Define source positions as a Guassian surface brightness profile
    X1, Y1, Q1, P1, S1, srcs = {}, {}, {}, {}, {}, {}
    X1['A'] = pymc.Uniform('X1A', 0., 100., value=xsrcA)
    Y1['A'] = pymc.Uniform('Y1A', 0., 100., value=ysrcA)
    Q1['A'] = pymc.Uniform('Q1A', 0.2, 1., value=1.)
    P1['A'] = pymc.Uniform('P1A', -180., 180., value=0.)
    S1['A'] = pymc.Uniform('N1A', 0., 6., value=sigsrcA)
    srcs['A'] = SBObjects.Gauss('', {'x': X1['A'], 'y': Y1['A'], 'q': Q1['A'],'pa': P1['A'], 'sigma': S1['A']})

    # Define lens mass model
    LX = pymc.Uniform('lx', 0., 100., value=xlens)
    LY = pymc.Uniform('ly', 0., 100., value=ylens)
    LB = pymc.Uniform('lb', 0., 100., value=blens)
    LQ = pymc.Uniform('lq', 0.2, 1., value=qlens)
    LP = pymc.Uniform('lp', -180., 180., value=plens)
    XB = pymc.Uniform('xb', -0.2, 0.2, value=0.)
    XP = pymc.Uniform('xp', -180., 180., value=0.)
    lens = MassModels.SIE('', {'x': LX, 'y': LY, 'b': LB, 'q': LQ, 'pa': LP})
    shear = MassModels.ExtShear('',{'x':LX,'y':LY,'b':XB,'pa':XP})
    lenses = [lens]

    x, y = np.meshgrid(np.arange(sa[0], sa[1], pix_scale), np.arange(sa[0], sa[1], pix_scale))

    image_plane, image_coords_pred = {}, {}
    for name in ['A']:
        x_src, y_src = pylens.getDeflections(lenses, [x, y], d=d[name])
        image_plane[name] = srcs[name].pixeval(x_src, y_src)
        plt.figure()
        plt.imshow(image_plane[name], interpolation='nearest', origin='lower')
        plt.xlabel('%dx - %d pixels' % (pix_scale, sa[0]))
        plt.ylabel('%dy - %d pixels' % (pix_scale, sa[0]))
        plt.savefig(os.path.join(ROOT_DIR, fig_dir, 'image_plane%s.png' % name))
        image_coords_pred[name] = np.add(np.multiply(np.where(image_plane[name] > threshold), pix_scale)[::-1], sa[0])

        lnlike, img_xpred_compare, img_ypred_compare = calc_lnlike(image_coords_pred[name], img_xobs[name], img_yobs[name])

        print(img_xpred_compare, img_xobs[name], lnlike)
        print(img_ypred_compare, img_yobs[name], lnlike)

    colors = (col for col in ['#1f77b4', '#2ca02c', '#9467bd', '#17becf', '#e377c2'])
    fig = plt.figure()
    plot_image(fits_file, fig, vmax=10.)
    plt.xlim(sa[0], sa[1])
    plt.ylim(sa[0], sa[1])
    for name in ['A']:
        plt.scatter(image_coords_pred[name][0], image_coords_pred[name][1], marker='.', alpha=0.3, c=next(colors))
    plt.savefig(os.path.join(ROOT_DIR, fig_dir, 'image_with_predicted_image_plane.png'))


    # print(pylens.getImgPos(x0=0, y0=0, b=21.6, sx=54.83, sy=38.98, lenses=lenses))


def calc_lnlike(image_coords_pred, img_xobs, img_yobs):
    img_xpred, img_ypred = image_coords_pred

    # Map each observed image to the single closest predicted image
    pred_arg = []
    for xo, yo in zip(img_xobs, img_yobs):
        xdist = np.abs(img_xpred - xo)  # pixel distance between xobs and xpredicted
        ydist = np.abs(img_ypred - yo)
        dist = xdist ** 2 + ydist ** 2
        pred_arg.append(np.argmin(dist))  # The index of the pred_img that the given observed image is closest to
    pred_arg = np.array(pred_arg)

    # these pred images are the ones that are being compared with the list of the obs images
    img_xpred_compare = np.array([img_xpred[i] for i in pred_arg])
    img_ypred_compare = np.array([img_ypred[i] for i in pred_arg])

    lnlike = -0.5 * (np.sum((img_xpred_compare - img_xobs) ** 2 + (img_ypred_compare - img_yobs) ** 2))

    return lnlike, img_xpred_compare, img_ypred_compare


def get_quasar_img_pos(pix_scale=1., threshold=0.8, fits_file=None, img_xobs=None, img_yobs=None, d=None):
    fig_dir = 'Figures/lensed_quasar/'
    sa = (0, 100)  # search area is 2000 pixels to 5000 pixels

    # Define source positions as a Guassian surface brightness profile
    X1, Y1, Q1, P1, S1, srcs = {}, {}, {}, {}, {}, {}
    X1['A'] = pymc.Uniform('X1A', 0., 100., value=52.)
    Y1['A'] = pymc.Uniform('Y1A', 0., 100., value=48.)
    Q1['A'] = pymc.Uniform('Q1A', 1., 1., value=1.)
    P1['A'] = pymc.Uniform('P1A', 0., 0., value=0.)
    S1['A'] = pymc.Uniform('N1A', 0., 6., value=1.2)
    srcs['A'] = SBObjects.Gauss('', {'x': X1['A'], 'y': Y1['A'], 'q': Q1['A'],'pa': P1['A'], 'sigma': S1['A']})

    pars = [X1['A'], Y1['A'], S1['A']]  # List of parameters
    cov = [0.1, 0.1, 1.5]  # List of initial `scatter' for emcee

    # Define lens mass model
    LX = pymc.Uniform('lx', 0., 100., value=49.)
    LY = pymc.Uniform('ly', 0., 100., value=50.)
    LB = pymc.Uniform('lb', 10., 40., value=20.)
    LQ = pymc.Uniform('lq', 0.2, 1., value=0.8)
    LP = pymc.Uniform('lp', -180., 180., value=85.)
    XB = pymc.Uniform('xb', -0.2, 0.2, value=0.)
    XP = pymc.Uniform('xp', -180., 180., value=0.)
    lens = MassModels.SIE('', {'x': LX, 'y': LY, 'b': LB, 'q': LQ, 'pa': LP})
    # shear = MassModels.ExtShear('', {'x': LX, 'y': LY, 'b': XB, 'pa': XP})
    lenses = [lens]
    pars += [LX, LY, LB, LQ, LP]
    cov += [0.2, 0.2, 0.2, 0.05, 10.]
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

        for name in ['A']:
            # Calculate deflections
            x_src, y_src = pylens.getDeflections(lenses, [x, y], d[name])

            # Get list of predicted image coordinates
            image_plane = srcs[name].pixeval(x_src, y_src)
            image_coords_pred = np.add(np.multiply(np.where(image_plane > threshold), pix_scale)[::-1], sa[0])  # Only if brightness > threshold

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
    param_names = ['$xA_{src}$', '$yA_{src}$', '$\sigma A_{src}$']
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

    # print(image_coords_pred)
    # plt.show()
    # return image_coords_pred


def main():
    fits_file = '/home/djm241/PycharmProjects/StrongLensing/data/lensed_quasar/WFI2033_F814W.fits'

    # Observed Image positions
    img_xobs, img_yobs, d = {}, {}, {}
    z_lens = 0.43
    img_xobs['A'] = np.array([27.1051, 69.3759, 71.0171, 56.8204])
    img_yobs['A'] = np.array([33.6946, 28.1791, 58.9579, 61.2290])
    d['A'] = scale_einstein_radius(z_lens=z_lens, z_src=2.01)

    pix_scale = 1.
    threshold = 0.8
    pars = [  51.63853012, 42.01441355,  4.19782456, 53.78385282, 40.6113934, 32.57773203,  0.66409627, 125.49369795]
    # plot_img_pos(pars, pix_scale=pix_scale, threshold=threshold, fits_file=fits_file, img_xobs=img_xobs, img_yobs=img_yobs, d=d)
    get_quasar_img_pos(pix_scale=pix_scale, threshold=threshold, fits_file=fits_file, img_xobs=img_xobs, img_yobs=img_yobs, d=d)

    plt.show()


if __name__=='__main__':
    main()
