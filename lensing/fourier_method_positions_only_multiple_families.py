import numpy as np
from numpy import cos, sin
from collections import OrderedDict
from dist_ang import scale_einstein_radius

# Image positions
z_lens = 0.43
img_xobs, img_yobs, d = OrderedDict(), OrderedDict(), OrderedDict()

img_xobs['A'] = np.array(
    [2375.942110, 2378.5, 2379.816610, 2381.299088, 2384, 2385.927991, 2389.555816, 2457.694760, 2450.744242,
     2442.833333, 2437.857924, 2433.064587, 2427.166666, 2424.099866, 2418.5, 2416.444081, 2462])
img_yobs['A'] = np.array(
    [3038.016677, 3024, 3012.367933, 2999.365293, 2983.5, 2970.435199, 2955.945319, 2737.545077, 2752.305849,
     2766.166666, 2782.058508, 2795.293450, 2811.166666, 2823.079067, 2837.5, 2846.943113, 2728])
d['A'] = scale_einstein_radius(z_lens=z_lens, z_src=2.013)

img_xobs['B'] = np.array(
    [3276.693717, 3261.382557, 3427.351819, 3417.043471, 3497.163625, 3486.371962, 3437.902765, 3430.50000])
img_yobs['B'] = np.array(
    [3482.795501, 3482.854177, 2592.719350, 2590.191799, 3075.107748, 3069.305065, 3698.140748, 3691.16667])
d['B'] = scale_einstein_radius(z_lens=z_lens, z_src=1.405)

img_xobs['11'] = np.array([3557.178601, 3548.271886, 3541.407488])
img_yobs['11'] = np.array([3363.943860, 3375.285957, 3385.515024])
d['11'] = scale_einstein_radius(z_lens=z_lens, z_src=2.06)

img_xobs['31'] = np.array([2933.063074, 2943.400421, 2890.687234, 2878.906523])
img_yobs['31'] = np.array([3393.715824, 3398.196336, 3044.431729, 3042.460964])
d['31'] = scale_einstein_radius(z_lens=z_lens, z_src=1.904)

img_xobs['41'] = np.array([3222.796159, 3227.700108])
img_yobs['41'] = np.array([3550.903781, 3542.180780])
d['41'] = scale_einstein_radius(z_lens=z_lens, z_src=1.810)

img_xobs['C'] = np.array([3799.999263, 3794.5, 3863.057095, 3861.1, 3823.051920])
img_yobs['C'] = np.array([3358.972702, 3367.9, 3059.195359, 3069.9, 2692.454490])
d['C'] = scale_einstein_radius(z_lens=z_lens, z_src=1.8)


# Image position in polar coords
rimage, theta, num_of_images = OrderedDict(), OrderedDict(), OrderedDict()
zeropoint = (3100, 3000)
for key in img_xobs:
    img_xobs[key] = img_xobs[key] - zeropoint[0]
    img_yobs[key] = img_yobs[key] - zeropoint[1]
    rimage[key] = np.sqrt(img_xobs[key]**2 + img_yobs[key]**2)
    theta[key] = np.arctan2(img_yobs[key], img_xobs[key])

    # Setup
    num_of_images[key] = len(rimage[key])

num_of_fourier_terms = 40
d = []
C = np.zeros((sum(num_of_images.values())*2, num_of_fourier_terms*2 + len(rimage.keys())*2))
nrows = 0

for idx, key in enumerate(img_xobs):
    # Compute d matrix
    for rl, tl in zip(rimage[key], theta[key]):
        d += [rl*cos(tl), rl*sin(tl)]

    # Compute alpha, beta
    alpha, beta, alpha_hat, beta_hat = {}, {}, {}, {}
    for k in range(num_of_fourier_terms):
        alpha[k] = cos(theta[key]) * cos(k*theta[key]) + k*sin(theta[key])*sin(k*theta[key])
        beta[k] = cos(theta[key]) * sin(k*theta[key]) - k*sin(theta[key])*cos(k*theta[key])
        alpha_hat[k] = sin(theta[key])*cos(k*theta[key]) - k*cos(theta[key])*sin(k*theta[key])
        beta_hat[k] = sin(theta[key])*sin(k*theta[key]) + k*cos(theta[key])*cos(k*theta[key])

    # Compute C matrix
    for l in range(num_of_images[key]):
        C[nrows + l*2][idx*2:idx*2+2] = [1, 0]
        C[nrows + l*2+1][idx*2:idx*2+2] = [0, 1]
        for k in range(num_of_fourier_terms):
            C[nrows + l*2][idx*2+2+k*2:idx*2+2+(k+1)*2] = [alpha[k][l], beta[k][l]]
            C[nrows + l*2+1][idx*2+2+k*2:idx*2+2+(k+1)*2] = [alpha_hat[k][l], beta_hat[k][l]]
    nrows += num_of_images[key] * 2


d = np.matrix(d).transpose()
C = np.matrix(C)

# Remove column beta_0 as all terms are zero
C = np.delete(C, 3, axis=1)

# Set a1 = b1 = 0 and remove these columns
C = np.delete(C, 3, axis=1)
C = np.delete(C, 4, axis=1)

# Delete last column so that C is a (2n x 2n) matrix
while C.shape[0] < C.shape[1]:
    print("C shape is", C.shape, "removing last column...")
    C = np.delete(C, -1, axis=1)

# Solve for x: x = [zeta, eta, a0/2, a2, b2, a3, b3, ..., ak, bk]
x = np.linalg.pinv(C,rcond=1e-14) * d  # inverse using singular vector decomposition (SVD)

# Get source positions
xsrc, ysrc = OrderedDict(), OrderedDict()
for idx, key in enumerate(img_xobs):
    zeta, eta = x[idx*2:idx*2+2]

    xsrc[key] = zeta * cos(eta) + zeropoint[0]
    ysrc[key] = zeta * sin(eta) + zeropoint[1]
pass