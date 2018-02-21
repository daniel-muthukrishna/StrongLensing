import os
import sys
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import pymc

from mass_model import pred_positions

sys.path.insert(0, '/Users/danmuth/PycharmProjects/StrongLensing/Matts_scripts')
from pylens import pylens, MassModels
from imageSim import SBObjects


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
    sa = (2000, 5000)  # search area

    # Define source positions as a Guassian surface brightness profile
    X1 = pymc.Uniform('X1', 2000., 5000., value=3034)
    Y1 = pymc.Uniform('Y1', 2000., 5000., value=3053)
    # R1 = pymc.Uniform('R1', 0., 1000, value=1000)
    Q1 = pymc.Uniform('Q1', 0.2, 1., value=0.8)
    P1 = pymc.Uniform('P1', -180., 180., value=110.)
    S1 = pymc.Uniform('N1', 0.6, 6., value=1.3)
    src = SBObjects.Gauss('', {'x':X1,'y':Y1,'q':Q1,'pa':P1,'sigma':S1})
    srcs = [src]

    # Define lens mass model
    LX = pymc.Uniform('lx', 2000., 5000., value=3034)
    LY = pymc.Uniform('ly', 2000., 5000., value=3053)
    LB = pymc.Uniform('lb', 10., 3000., value=950.)
    LQ = pymc.Uniform('lq', 0.2, 1., value=0.8)
    LP = pymc.Uniform('lp', -180., 180., value=110.)
    XB = pymc.Uniform('xb', -0.2, 0.2, value=0.)
    XP = pymc.Uniform('xp', -180., 180., value=0.)
    lens = MassModels.SIE('', {'x': LX, 'y': LY, 'b': LB, 'q': LQ, 'pa': LP})
    shear = MassModels.ExtShear('',{'x':LX,'y':LY,'b':XB,'pa':XP})
    lenses = [lens, shear]
    # # OR MY MODEL I have checked that they get the same answer
    # x_lens, y_lens, theta, b, q = (3034, 3053, 1.1, 950, 1)
    # pars = (x_lens, y_lens, theta, b, q)

    x, y = np.meshgrid(np.arange(sa[0], sa[1], 1.), np.arange(sa[0], sa[1], 1.))

    for lens in lenses:
        lens.setPars()
    x_src, y_src = pylens.getDeflections(lenses, [x, y])
    #
    # x_src, y_src = pred_positions(x, y, d=1, pars=pars)

    image_plane = src.pixeval(x_src, y_src)
    plt.imshow(image_plane, interpolation='nearest', origin='lower')

    # print(pylens.getImgPos(x0=0, y0=0, b=950, sx=3034, sy=3053, lenses=lenses))

    # Get list of predicted image coordinates
    image_coords_pred = np.add(np.where(image_plane>0.8), sa[0])

    plt.show()


if __name__=='__main__':
    # get_quasar_img_pos()
    get_macs0451_img_pos()
