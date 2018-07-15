import numpy as np
from numpy import cos, sin

# Image position
ximgA = np.array(
    [2375.942110, 2378.5, 2379.816610, 2381.299088, 2384, 2385.927991, 2389.555816, 2457.694760, 2450.744242,
     2442.833333, 2437.857924, 2433.064587, 2427.166666, 2424.099866, 2418.5, 2416.444081, 2462])
yimgA = np.array(
    [3038.016677, 3024, 3012.367933, 2999.365293, 2983.5, 2970.435199, 2955.945319, 2737.545077, 2752.305849,
     2766.166666, 2782.058508, 2795.293450, 2811.166666, 2823.079067, 2837.5, 2846.943113, 2728])
zeropoint = (3100, 3000)

# Lensed quasar
ximgA = np.array([69.3759, 56.8204, 71.0171, 27.1051])
yimgA = np.array([28.1791, 61.2290, 58.9579, 33.6946])
zeropoint = (50, 40)

ximgA = ximgA - zeropoint[0]
yimgA = yimgA - zeropoint[1]

# Image position in polar coords
rImageA = np.sqrt(ximgA**2 + yimgA**2)
thetaA = np.arctan2(yimgA, ximgA)

# Setup
num_of_fourier_terms = 18
num_of_images = len(rImageA)

# Compute d matrix
d = []
for rl, tl in zip(rImageA, thetaA):
    d += [rl*cos(tl), rl*sin(tl)]
d = np.matrix(d).transpose()

# Compute alpha, beta
alpha, beta, alpha_hat, beta_hat = {}, {}, {}, {}
for k in range(num_of_fourier_terms):
    alpha[k] = cos(thetaA) * cos(k*thetaA) + k*sin(thetaA)*sin(k*thetaA)
    beta[k] = cos(thetaA) * sin(k*thetaA) - k*sin(thetaA)*cos(k*thetaA)
    alpha_hat[k] = sin(thetaA)*cos(k*thetaA) - k*cos(thetaA)*sin(k*thetaA)
    beta_hat[k] = sin(thetaA)*sin(k*thetaA) + k*cos(thetaA)*cos(k*thetaA)

# Compute C matrix
C = np.zeros((num_of_images*2, 2+num_of_fourier_terms*2))
for l in range(num_of_images):
    C[l*2][0:2] = [1, 0]
    C[l*2+1][0:2] = [0, 1]
    for k in range(num_of_fourier_terms):
        C[l*2][2+k*2:2+(k+1)*2] = [alpha[k][l], beta[k][l]]
        C[l*2+1][2+k*2:2+(k+1)*2] = [alpha_hat[k][l], beta_hat[k][l]]
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
zeta, eta = x[0:2]

# Check solution is correct
print(np.allclose(np.dot(C, x), d))

# Source positions (zeta, eta)
xsrc = zeta * cos(eta) + zeropoint[0]
ysrc = zeta * sin(eta) + zeropoint[1]
pass