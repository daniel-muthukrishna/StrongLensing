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

SCRIPT_DIR = os.path.dirname(os.path.realpath(__file__))
ROOT_DIR = os.path.join(SCRIPT_DIR, '..')


def plot_img_pos(pars, pix_scale=1., threshold=0.8):
    xsrcA, ysrcA, sigsrcA, xsrcB, ysrcB, sigsrcB, xlens, ylens, blens, qlens, plens = pars
    fig_dir = 'Figures/MACS0451/'
    sa = (2000, 5000)  # search area is 2000 pixels to 5000 pixels

    # Define source positions as a Guassian surface brightness profile
    X1, Y1, Q1, P1, S1, srcs = {}, {}, {}, {}, {}, {}
    X1['A'] = pymc.Uniform('X1A', 0., 5000., value=xsrcA)
    Y1['A'] = pymc.Uniform('Y1A', 0., 5000., value=xsrcA)
    Q1['A'] = pymc.Uniform('Q1A', 0.2, 1., value=1.)
    P1['A'] = pymc.Uniform('P1A', -180., 180., value=0.)
    S1['A'] = pymc.Uniform('N1A', 0., 10000., value=sigsrcA)
    srcs['A'] = SBObjects.Gauss('', {'x': X1['A'], 'y': Y1['A'], 'q': Q1['A'],'pa': P1['A'], 'sigma': S1['A']})

    X1['B'] = pymc.Uniform('X1B', 0., 5000., value=xsrcB)
    Y1['B'] = pymc.Uniform('Y1B', 0., 5000., value=xsrcB)
    Q1['B'] = pymc.Uniform('Q1B', 0.2, 1., value=1.)
    P1['B'] = pymc.Uniform('P1B', -180., 180., value=0.)
    S1['B'] = pymc.Uniform('N1B', 0., 10000., value=sigsrcB)
    srcs['B'] = SBObjects.Gauss('', {'x': X1['B'], 'y': Y1['B'], 'q': Q1['B'],'pa': P1['B'], 'sigma': S1['B']})

    pars = [X1['A'], Y1['A'], S1['A'], X1['B'], Y1['B'], S1['B']]  # List of parameters
    cov = [300, 300, 2, 300, 300, 2]  # List of initial `scatter' for emcee

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
    # # OR MY MODEL I have checked that they get the same answer
    # x_lens, y_lens, theta, b, q = (51.5, 39.9, 1.1, 21.6, 1.0)
    # pars = (x_lens, y_lens, theta, b, q)

    x, y = np.meshgrid(np.arange(sa[0], sa[1], pix_scale), np.arange(sa[0], sa[1], pix_scale))

    x_src, y_src = pylens.getDeflections(lenses, [x, y], )
    # x_src, y_src = pred_positions(x, y, d=1, pars=pars)

    image_plane = srcs['A'].pixeval(x_src, y_src)
    plt.figure()
    plt.imshow(image_plane, interpolation='nearest', origin='lower')
    plt.xlabel('%dx - %d pixels' % (pix_scale, sa[0]))
    plt.ylabel('%dy - %d pixels' % (pix_scale, sa[0]))
    plt.savefig(os.path.join(ROOT_DIR, fig_dir, 'image_planeA.png'))
    print('A', np.add(np.multiply(np.where(image_plane > threshold), pix_scale), sa[0]))

    image_plane = srcs['B'].pixeval(x_src, y_src)
    plt.figure()
    plt.imshow(image_plane, interpolation='nearest', origin='lower')
    plt.xlabel('%dx - %d pixels' % (pix_scale, sa[0]))
    plt.ylabel('%dy - %d pixels' % (pix_scale, sa[0]))
    plt.savefig(os.path.join(ROOT_DIR, fig_dir, 'image_planeB.png'))
    print(np.add(np.multiply(np.where(image_plane > threshold), pix_scale), sa[0]))

    # print(pylens.getImgPos(x0=0, y0=0, b=21.6, sx=54.83, sy=38.98, lenses=lenses))


def get_macs0451_img_pos(pix_scale=1., threshold=0.8):
    fig_dir = 'Figures/MACS0451/'
    sa = (2000, 5000)  # search area is 2000 pixels to 5000 pixels

    # Define source positions as a Guassian surface brightness profile
    X1, Y1, Q1, P1, S1, srcs = {}, {}, {}, {}, {}, {}
    X1['A'] = pymc.Uniform('X1A', 2500., 3900., value=3034)
    Y1['A'] = pymc.Uniform('Y1A', 2400., 3500., value=3053)
    Q1['A'] = pymc.Uniform('Q1A', 0.2, 1., value=1.)
    P1['A'] = pymc.Uniform('P1A', -180., 180., value=0.)
    S1['A'] = pymc.Uniform('N1A', 0., 1000., value=1.3)
    srcs['A'] = SBObjects.Gauss('', {'x': X1['A'], 'y': Y1['A'], 'q': Q1['A'],'pa': P1['A'], 'sigma': S1['A']})

    X1['B'] = pymc.Uniform('X1B', 2500., 3900., value=3300)
    Y1['B'] = pymc.Uniform('Y1B', 2400., 3500., value=3100)
    Q1['B'] = pymc.Uniform('Q1B', 0.2, 1., value=1.)
    P1['B'] = pymc.Uniform('P1B', -180., 180., value=0.)
    S1['B'] = pymc.Uniform('N1B', 0., 1000., value=1.1)
    srcs['B'] = SBObjects.Gauss('', {'x': X1['B'], 'y': Y1['B'], 'q': Q1['B'],'pa': P1['B'], 'sigma': S1['B']})

    pars = [X1['A'], Y1['A'], S1['A'], X1['B'], Y1['B'], S1['B']]  # List of parameters
    cov = [300., 300., 2., 300., 300., 2.]  # List of initial `scatter' for emcee

    # Define lens mass model
    LX = pymc.Uniform('lx', 2900., 3400., value=3034)
    LY = pymc.Uniform('ly', 2600., 3500., value=2981)
    LB = pymc.Uniform('lb', 10., 1500., value=297.)
    LQ = pymc.Uniform('lq', 0.2, 1., value=0.333)
    LP = pymc.Uniform('lp', -180., 180., value=85.)
    XB = pymc.Uniform('xb', -0.2, 0.2, value=0.)
    XP = pymc.Uniform('xp', -180., 180., value=0.)
    lens = MassModels.SIE('', {'x': LX, 'y': LY, 'b': LB, 'q': LQ, 'pa': LP})
    # shear = MassModels.ExtShear('', {'x': LX, 'y': LY, 'b': XB, 'pa': XP})
    lenses = [lens]
    pars += [LX, LY, LB, LQ, LP]
    cov += [300., 300., 300., 0.3, 50.]
    cov = np.array(cov)

    # Get grid of x and y points
    x, y = np.meshgrid(np.arange(sa[0], sa[1], pix_scale), np.arange(sa[0], sa[1], pix_scale))

    # MCMC setup
    nwalkers = 1000
    nsteps = 3000
    z_lens = 0.43

    # Observed Image positions
    x_img, y_img, d = {}, {}, {}
    x_img['A'] = np.array([2375.942110, 2378.5, 2379.816610, 2381.299088, 2384, 2385.927991, 2389.555816, 2457.694760, 2450.744242, 2442.833333, 2437.857924, 2433.064587, 2427.166666, 2424.099866, 2418.5, 2416.444081, 2462])
    y_img['A'] = np.array([3038.016677, 3024, 3012.367933, 2999.365293, 2983.5, 2970.435199, 2955.945319, 2737.545077, 2752.305849, 2766.166666, 2782.058508, 2795.293450, 2811.166666, 2823.079067, 2837.5, 2846.943113, 2728])
    d['A'] = scale_einstein_radius(z_lens=z_lens, z_src=2.01)

    x_img['B'] = np.array([3276.693717, 3261.382557, 3427.351819, 3417.043471])
    y_img['B'] = np.array([3482.795501, 3482.854177, 2592.719350, 2590.191799])
    d['B'] = scale_einstein_radius(z_lens=z_lens, z_src=1.405)

    # Define likelihood function
    @pymc.observed
    def logL(value=0., tmp=pars):
        for key in srcs:
            srcs[key].setPars()
        for lens in lenses:
            lens.setPars()
        lnlike = {}

        for image_name in ['A', 'B']:
            # Calculate deflections
            x_src, y_src = pylens.getDeflections(lenses, [x, y], d[image_name])

            # Get list of predicted image coordinates
            image_plane = src[image_name].pixeval(x_src, y_src)
            image_coords_pred = np.add(np.multiply(np.where(image_plane > threshold), pix_scale), sa[0])  # Only if brightness > threshold
            print(image_name)
            if not image_coords_pred.size:  # If it's an empty list
                return -1e30
            img_xpred, img_ypred = image_coords_pred

            # Map each observed image to the single closest predicted image
            img_xobs, img_yobs = x_img[image_name], y_img[image_name]
            obs_arg = []
            for xo, yo in zip(img_xobs, img_yobs):
                xdist = np.abs(img_xpred - xo)  # pixel distance between xobs and xpredicted
                ydist = np.abs(img_ypred - yo)
                dist = xdist + ydist
                obs_arg.append(np.argmin(dist))  # The index of the obs_img that the given predicted image is closest to
            obs_arg = np.array(obs_arg)

            # these pred images are the ones that are being compared with the list of the obs images
            img_xpred_compare = np.array([img_xpred[i] for i in obs_arg])
            img_ypred_compare = np.array([img_ypred[i] for i in obs_arg])

            lnlike[image_name] = -0.5 * (np.sum(np.abs(img_xpred_compare - img_xobs) + np.abs(img_ypred_compare - img_yobs)))

        return sum(lnlike.values())

    # Run MCMC
    sampler = myEmcee.Emcee(pars+[logL], cov, nwalkers=nwalkers, nthreads=30)
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
    param_names = ['$xA_{src}$', '$yA_{src}$', '$\sigmaA_{src}$', '$xB_{src}$', '$yB_{src}$', '$\sigmaB_{src}$']
    param_names += ['$x_{lens}$', '$y_{lens}$', '$b_{lens}$', '$q_{lens}$', '$pa_{lens}$']
    c = ChainConsumer()
    c.add_chain(samples, parameters=param_names)
    c.configure(summary=True, cloud=True)
    c.plotter.plot(filename=os.path.join(ROOT_DIR, fig_dir, 'source_pos_parameter_contours.png'))
    fig = c.plotter.plot_walks(convolve=100)
    fig.savefig(os.path.join(ROOT_DIR, fig_dir, 'source_pos_mcmc_walks.png'))

    b = best_fits
    print(b)
    plot_img_pos(pars=b, pix_scale=pix_scale, threshold=threshold)

    # print(image_coords_pred)
    # plt.show()
    # return image_coords_pred


def main():
    pix_scale = 10.
    threshold = 0.01
    pars = [3014., 3184., 10., 3014., 3182., 1167., 1.0, 41.187714667301265]

    plot_img_pos(pars, pix_scale=pix_scale, threshold=threshold)
    # get_macs0451_img_pos(pix_scale=pix_scale, threshold=threshold)

    plt.show()


if __name__=='__main__':
    main()
