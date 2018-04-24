import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from itertools import takewhile

from image_overplot_contours import plot_image

SCRIPT_DIR = os.path.dirname(os.path.realpath(__file__))
ROOT_DIR = os.path.join(SCRIPT_DIR, '..')


def read_sextractor_output(filepath):
    with open(filepath, 'r') as FileObj:
        header = list(takewhile(lambda s: s.startswith('#'), FileObj))

    col_names = []
    for line in header:
        info = line.split()
        col_names.append(info[2])

    sex = pd.read_csv(filepath, comment='#', delim_whitespace=True, names=col_names)

    return sex


def get_positions(filepath, colour1=None, colour2=None, colour_limits=None, fits_file=None, fig_dir='.', addsavename=''):
    sex = read_sextractor_output(filepath)
    x = sex.X_IMAGE
    y = sex.Y_IMAGE

    plt.figure()
    if fits_file is not None:
        plot_image(fits_file, vmin=0, vmax=0.35)
    plt.scatter(x, y, marker='o', c='b', alpha=0.2)

    if colour_limits is not None:
        (x1, x2), (y1, y2) = colour_limits
        mask = (colour1 > x1) & (colour1 < x2) & (colour2 > y1) & (colour2 < y2)
        x = x[mask]
        y = y[mask]

    plt.scatter(x, y, marker='+', c='g', alpha=0.9)

    plt.xlim(min(x), max(x))
    plt.ylim(min(x), max(y))
    plt.savefig(os.path.join(fig_dir, 'cluster_members{}'.format(addsavename)))
    return x, y


def get_colour(filepath1, filepath2):
    # If one of the args is None return the other flux instead of colour
    if filepath1 is None:
        flux = read_sextractor_output(filepath2).FLUX_ISO
        return flux
    elif filepath2 is None:
        flux = read_sextractor_output(filepath1).FLUX_ISO
        return flux

    flux1 = read_sextractor_output(filepath1).FLUX_ISO
    flux2 = read_sextractor_output(filepath2).FLUX_ISO

    colour = flux1 / flux2

    return colour


def plot_colour_colour(colour1, colour2, name1='', name2='', fig_dir='.', colour_limits=None, addsavename='', fpathpos=None, axlims=None):
    fig, ax = plt.subplots(1)
    ax.scatter(colour1, colour2, marker='.', alpha=0.3)

    if fpathpos is not None:
        xpos, ypos = get_positions(fpathpos)
        labels = np.core.defchararray.add(np.around(xpos).values.astype(str), np.around(ypos).values.astype(str))
        for label, x, y in zip(labels, colour1, colour2):
            plt.annotate(
                label,
                xy=(x, y), xytext=(-20, 20),
                textcoords='offset points', ha='right', va='bottom',
                bbox=dict(boxstyle='round,pad=0.5', fc='yellow', alpha=0.5),
                arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=0'))

    if colour_limits is not None:
        (x1, x2), (y1, y2) = colour_limits
        rect = patches.Rectangle(xy=(x1, y1), width=x2-x1, height=y2-y1, fill=False)
        ax.add_patch(rect)

    ax.set(xlabel=name1, ylabel=name2)
    if axlims:
        (x1, x2), (y1, y2) = axlims
        ax.set_xlim(x1, x2)
        ax.set_ylim(y1, y2)
    fig.savefig(os.path.join(fig_dir, 'colour_colour_plot{}'.format(addsavename)))


def main():
    fig_dir = os.path.join(ROOT_DIR, 'Figures')
    fpath140 = os.path.join(ROOT_DIR, 'data/MACS0451/sex_140.asc')
    fpath110 = os.path.join(ROOT_DIR, 'data/MACS0451/sex_110.asc')
    fpath814 = os.path.join(ROOT_DIR, 'data/MACS0451/sex_814.asc')
    fpath606 = os.path.join(ROOT_DIR, 'data/MACS0451/sex_606.asc')

    colour1 = get_colour(fpath814, None)
    colour2 = get_colour(fpath606, fpath814)

    colour_limits = [(50, 750), (0.5, 0.72)]
    axlims = [(-10, 800), (-0.1, 2)]

    plot_colour_colour(colour1, colour2, name1='814A', name2='606A - 814A', fig_dir=fig_dir, colour_limits=colour_limits, addsavename='_814', axlims=axlims)  #, fpathpos=fpath110)
    get_positions(fpath110, colour1, colour2, colour_limits=colour_limits, fits_file=os.path.join(ROOT_DIR, 'data/MACS0451/MACS0451_F110W.fits'), fig_dir=fig_dir, addsavename='_814')
    plt.show()


if __name__ == '__main__':
    main()
    # sex MACS0451_F110W.fits,MACS0451_F814W.fits -CATALOG_NAME 'sex_814.asc'
