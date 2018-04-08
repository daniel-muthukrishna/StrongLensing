import pyfits
import colorImage
import pylab


b = pyfits.open('../data/MACS0451/MACS0451_F606W.fits')[0].data
g = pyfits.open('../data/MACS0451/MACS0451_F814W.fits')[0].data
r = pyfits.open('../data/MACS0451/MACS0451_F110W.fits')[0].data

xGal = 3075
yGal = 3447

C = colorImage.ColorImage()

# img = C.createModel(b,g,r,center=(yGal,xGal))
#
# pylab.imshow(img,origin='lower',interpolation='nearest')
# pylab.show()

# Change the non-linear scaling

C.nonlin = 50.
img = C.createModel(b,g,r,center=(yGal,xGal))

pylab.imshow(img,origin='lower',interpolation='nearest')
pylab.xlim([2000, 4500])
pylab.ylim([2000, 4500])
pylab.show()