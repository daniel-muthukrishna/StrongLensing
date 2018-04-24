import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from itertools import takewhile

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


def get_positions(filepath):
    sex = read_sextractor_output(filepath)
    x = sex.X_IMAGE
    y = sex.Y_IMAGE

    # plt.figure()
    # plt.scatter(x, y)

    return x, y


def get_colour(filepath1, filepath2):
    sex1 = read_sextractor_output(filepath1)
    sex2 = read_sextractor_output(filepath2)

    colour = sex1.FLUX_ISO / sex2.FLUX_ISO

    return colour


def plot_colour_colour(colour1, colour2, name1='', name2='', fig_dir='.', fpathpos=None):
    plt.figure()
    plt.scatter(colour1, colour2)

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

    plt.xlabel(name1)
    plt.ylabel(name2)
    plt.xlim(-0.1, 2)
    plt.ylim(-0.1, 2)
    plt.savefig(os.path.join(fig_dir, 'colour_colour_plot.png'))


def main():
    fig_dir = os.path.join(ROOT_DIR, 'Figures')
    fpath110 = os.path.join(ROOT_DIR, 'data/MACS0451/sex_110.asc')
    fpath814 = os.path.join(ROOT_DIR, 'data/MACS0451/sex_814.asc')
    fpath606 = os.path.join(ROOT_DIR, 'data/MACS0451/sex_606.asc')

    colour1 = get_colour(fpath606, fpath814)
    colour2 = get_colour(fpath814, fpath110)

    plot_colour_colour(colour1, colour2, name1='606A - 814A', name2='814A - 1100A', fig_dir=fig_dir)#, fpathpos=fpath110)
    get_positions(fpath110)
    plt.show()


if __name__ == '__main__':
    main()
