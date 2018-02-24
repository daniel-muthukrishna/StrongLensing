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

def get_quasar_img_pos():

    # Define source positions as a Guassian surface brightness profile
    X1 = pymc.Uniform('X1', 50., 60., value=54.83)
    Y1 = pymc.Uniform('Y1', 35., 45., value=38.98)
    R1 = pymc.Uniform('R1', 0., 20., value=3.95)
    Q1 = pymc.Uniform('Q1', 0.2, 1., value=0.8)
    P1 = pymc.Uniform('P1', -180., 180., value=54.)
    S1 = pymc.Uniform('N1', 0.6, 6., value=1.3)
    src = SBObjects.Gauss('', {'x':X1,'y':Y1,'q':Q1,'pa':P1,'sigma':S1})
    srcs = [src]

    # Define lens mass model
    LX = pymc.Uniform('lx', 50., 60., value=54.83)
    LY = pymc.Uniform('ly', 35., 45., value=38.98)
    LB = pymc.Uniform('lb', 10., 40., value=20.)
    LQ = pymc.Uniform('lq', 0.2, 1., value=0.8)
    LP = pymc.Uniform('lp', -180., 180., value=110.)
    XB = pymc.Uniform('xb', -0.2, 0.2, value=0.)
    XP = pymc.Uniform('xp', -180., 180., value=0.)
    lens = MassModels.SIE('', {'x': LX, 'y': LY, 'b': LB, 'q': LQ, 'pa': LP})
    shear = MassModels.ExtShear('',{'x':LX,'y':LY,'b':XB,'pa':XP})
    lenses = [lens]
    # # OR MY MODEL I have checked that they get the same answer
    # x_lens, y_lens, theta, b, q = (51.5, 39.9, 1.1, 21.6, 1.0)
    # pars = (x_lens, y_lens, theta, b, q)

    x, y = np.meshgrid(np.arange(0, 100, 1.), np.arange(0, 100, 1.))

    for lens in lenses:
        lens.setPars()
    x_src, y_src = pylens.getDeflections(lenses, [x, y])
    # x_src, y_src = pred_positions(x, y, d=1, pars=pars)

    image_plane = src.pixeval(x_src, y_src)
    plt.imshow(image_plane, interpolation='nearest', origin='lower')

    print(pylens.getImgPos(x0=0, y0=0, b=21.6, sx=54.83, sy=38.98, lenses=lenses))

    plt.show()

def get_macs0451_img_pos():
    fig_dir = 'Figures/MACS0451/'
    sa = (2000, 5000)  # search area is 2000 pixels to 5000 pixels
    near = 150  # The predicted and actual image positions are nearby if they are within this many pixels

    # Define source positions as a Guassian surface brightness profile
    X1 = pymc.Uniform('X1', 2000., 5000., value=3034)
    Y1 = pymc.Uniform('Y1', 2000., 5000., value=3053)
    Q1 = pymc.Uniform('Q1', 0.2, 1., value=0.3)
    P1 = pymc.Uniform('P1', -180., 180., value=30.)
    S1 = pymc.Uniform('N1', 0.6, 6., value=0.7)
    src = SBObjects.Gauss('', {'x': X1, 'y': Y1, 'q': Q1, 'pa': P1, 'sigma': S1})
    srcs = [src]
    pars = [X1, Y1, Q1, P1, S1]  # List of parameters
    cov = [900, 900, 0.3, 50., 2]  # List of initial `scatter' for emcee
    cov = np.array(cov)

    # Define lens mass model
    LX = pymc.Uniform('lx', 2000., 5000., value=3034)
    LY = pymc.Uniform('ly', 2000., 5000., value=3053)
    LB = pymc.Uniform('lb', 10., 3000., value=950.)
    LQ = pymc.Uniform('lq', 0.2, 1., value=0.8)
    LP = pymc.Uniform('lp', -180., 180., value=110.)
    XB = pymc.Uniform('xb', -0.2, 0.2, value=0.)
    XP = pymc.Uniform('xp', -180., 180., value=0.)
    lens = MassModels.SIE('', {'x': LX, 'y': LY, 'b': LB, 'q': LQ, 'pa': LP})
    # shear = MassModels.ExtShear('', {'x': LX, 'y': LY, 'b': XB, 'pa': XP})
    lenses = [lens]
    # pars += [LX, LY, LB, LQ, LP]

    x, y = np.meshgrid(np.arange(sa[0], sa[1], 1.), np.arange(sa[0], sa[1], 1.))

    for lens in lenses:
        lens.setPars()
    x_src, y_src = pylens.getDeflections(lenses, [x, y])
    print(x_src)

    nwalkers = 50
    nsteps = 300
    z_lens = 0.43

    # Observed Image positions
    x_img, y_img, d = {}, {}, {}
    x_img['A'] = np.array([2375.942110, 2378.5, 2379.816610, 2381.299088, 2384, 2385.927991, 2389.555816, 2457.694760, 2450.744242, 2442.833333, 2437.857924, 2433.064587, 2427.166666, 2424.099866, 2418.5, 2416.444081, 2462])
    y_img['A'] = np.array([3038.016677, 3024, 3012.367933, 2999.365293, 2983.5, 2970.435199, 2955.945319, 2737.545077, 2752.305849, 2766.166666, 2782.058508, 2795.293450, 2811.166666, 2823.079067, 2837.5, 2846.943113, 2728])
    d['A'] = scale_einstein_radius(z_lens=z_lens, z_src=2.01)

    # Define likelihood function
    @pymc.observed
    def logL(value=0., tmp=pars):

        for src in srcs:
            src.setPars()

        image_plane = src.pixeval(x_src, y_src)

        # Get list of predicted image coordinates TODO: Change this to look for the center of each predicted image (i.e. when value is close to 1.0)
        image_coords_pred = np.add(np.where(image_plane > 0.8), sa[0])

        # Choose the central point
        # new = []
        # i_prev, j_prev = 0, 0
        # for i, j in zip(*image_coords_pred):
        #     if abs(i - i_prev) != 1 and abs(j - j_prev) != 1:
        #         new.append([i, j])
        #     i_prev, j_prev = i, j
        # image_coords_pred = np.array(zip(*new))
        print(image_coords_pred)

        if not image_coords_pred.size: # If it's an empty list
            return -1e20

        img_xpred, img_ypred = image_coords_pred

        img_xobs, img_yobs = x_img['A'], y_img['A']
        obs_arg = []  # the predicted values used in the likelihood calculation because the positions actually are nearby the actual image positions
        # Every obsimg needs to map to one predimg
        for xo, yo in zip(img_xobs, img_yobs):
            xdist = np.abs(img_xpred - xo)  # pixel distance between xobs and xpredicted
            ydist = np.abs(img_ypred - yo)
            dist = xdist + ydist
            obs_arg.append(np.argmin(dist))  # The index of the obs_img that the given predicted image is closest to
        obs_arg = np.array(obs_arg)

        img_xpred_compare = np.array([img_xpred[i] for i in obs_arg])  # these pred images are the ones that are being compared with the list of the obs images
        img_ypred_compare = np.array([img_ypred[i] for i in obs_arg])

        return -0.5 * (np.sum(np.abs(img_xpred_compare - img_xobs) + np.abs(img_ypred_compare - img_yobs)))

    sampler = myEmcee.Emcee(pars+[logL], cov, nwalkers=nwalkers, nthreads=14)
    sampler.sample(nsteps)

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
    burn = 0
    samples = samples[burn:].reshape(((nsteps-burn) * nwalkers, len(pars)))

    # Get best fit parameters
    samples_exp = samples.copy()
    samples_exp[:, 2] = np.exp(samples_exp[:, 2])
    best_fits = list(map(lambda v: (v[1]), zip(*np.percentile(samples_exp, [16, 50, 84], axis=0))))
    # truth = dict(zip(initial.keys(), best_fits))
    # print(truth)

    # Plot parameter contours and mcmc chains
    c = ChainConsumer()
    c.add_chain(samples, parameters=['xsrc', 'ysrc', 'qsrc', 'psrc', 'ssrc'])
    c.configure(summary=True, cloud=True)
    c.plotter.plot(filename=os.path.join(ROOT_DIR, fig_dir, 'source_pos_parameter_contours.png'))
    fig = c.plotter.plot_walks(convolve=100)
    fig.savefig(os.path.join(ROOT_DIR, fig_dir, 'source_pos_mcmc_walks.png'))

    # print(image_coords_pred)
    # plt.show()
    # return image_coords_pred






if __name__=='__main__':
    # get_quasar_img_pos()
    get_macs0451_img_pos()
    plt.show()

# for x, y in zip(x_img['A'], y_img['A']):
#     ((img_x_pred - x) < 150).any() and ((img_x_pred - x) > 150).any()
# if ((img_xobs - xp) < near).any() and ((img_xobs - xp) > -near).any() and ((img_yobs - yp) < near).any() and (
#         (img_yobs - yp) > -near).any():  # Check if predicted position near to actual image position
#
#     pred.append([xp, yp])
# if pred:
#     pred = np.array(zip(*pred))
# else:
#     pred = np.array([img_xpred, img_ypred])
    # print(pylens.getImgPos(x0=0, y0=0, b=950, sx=3034, sy=3053, lenses=lenses))
    # plt.imshow(image_plane, interpolation='nearest', origin='lower')
    # plt.show()