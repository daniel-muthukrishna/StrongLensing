import os
import numpy as np
from astropy.io import fits
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm


def read_fits_image(fits_file):
    """Take in fits_file and return the image """
    hdu1 = fits.open(fits_file)
    image = hdu1[0].data
    # image = (image - image.min()) / (image.max() - image.min())

    return image


def plot_image(fits_file, fig, vmin=0, vmax=0.5):
    image = read_fits_image(fits_file)

    plt.imshow(image, vmin=vmin, vmax=vmax, cmap='hot', origin='lower')


def plot_image_and_contours(fits_file, samples, fig_dir='', img_name='', save=True):
    fig = plt.figure()

    plot_image(fits_file, fig)

    counts, xbins, ybins = np.histogram2d(samples[:, 0], samples[:, 1], bins=100, normed=LogNorm())
    plt.contour(counts.transpose(), extent=[xbins.min(), xbins.max(), ybins.min(), ybins.max()])
    plt.xlim(1700, 4900)
    plt.ylim(1650, 4450)
    if save:
        plt.savefig(os.path.join(fig_dir, 'image_with_contours%s.png' % img_name))



if __name__ == '__main__':
    fits_file_quasar = '/Users/danmuth/PycharmProjects/StrongLensing/data/lensed_quasar/WFI2033_F814W.fits'
    fits_file_macs0451 = '/Users/danmuth/PycharmProjects/StrongLensing/data/MACS0451/MACS0451_F110W.fits'

    read_fits_image(fits_file_macs0451)

    plt.show()
