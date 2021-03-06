import numpy as np
from numpy import cos, sin

# Image position
ximgA = np.array(
    [2375.942110, 2378.5, 2379.816610, 2381.299088, 2384, 2385.927991, 2389.555816, 2457.694760, 2450.744242,
     2442.833333, 2437.857924, 2433.064587, 2427.166666, 2424.099866, 2418.5, 2416.444081, 2462])
yimgA = np.array(
    [3038.016677, 3024, 3012.367933, 2999.365293, 2983.5, 2970.435199, 2955.945319, 2737.545077, 2752.305849,
     2766.166666, 2782.058508, 2795.293450, 2811.166666, 2823.079067, 2837.5, 2846.943113, 2728])
zeropoint = (3000, 2900)


ximgA = np.array([69.3759, 56.8204, 71.0171, 27.1051])
yimgA = np.array([28.1791, 61.2290, 58.9579, 33.6946])
fluxA = np.array([16.71731, 20.75353, 26.23504, 18.3524])/16.71731
zeropoint = (50, 50)

ximgA = ximgA - zeropoint[0]
yimgA = yimgA - zeropoint[1]

# Image position in polar coords
rImageA = np.sqrt(ximgA**2 + yimgA**2)
thetaA = np.arctan2(yimgA, ximgA)

# Setup
num_of_fourier_terms = 7
num_of_images = len(rImageA)

# Compute d matrix
d = []
for rl, tl, fl in zip(rImageA, thetaA, fluxA):
    d += [rl*cos(tl), rl*sin(tl), (fl-1)*rImageA[0]*rl]
d = np.matrix(d).transpose()

# Compute alpha, beta
alpha, beta, alpha_hat, beta_hat, gamma, delta = {}, {}, {}, {}, {}, {}
for k in range(num_of_fourier_terms):
    alpha[k] = cos(thetaA) * cos(k*thetaA) + k*sin(thetaA)*sin(k*thetaA)
    beta[k] = cos(thetaA) * sin(k*thetaA) - k*sin(thetaA)*cos(k*thetaA)
    alpha_hat[k] = sin(thetaA)*cos(k*thetaA) - k*cos(thetaA)*sin(k*thetaA)
    beta_hat[k] = sin(thetaA)*sin(k*thetaA) + k*cos(thetaA)*cos(k*thetaA)
    gamma[k] = (1 - k**2) * (fluxA*rImageA*cos(k*thetaA[0]) - rImageA[0]*cos(k*thetaA))
    delta[k] = (1 - k**2) * (fluxA*rImageA*sin(k*thetaA[0]) - rImageA[0]*sin(k*thetaA))

# Compute C matrix
C = np.zeros((num_of_images*3, 2+num_of_fourier_terms*2))
for l in range(num_of_images):
    C[l*3][0:2] = [1, 0]
    C[l*3+1][0:2] = [0, 1]
    C[l*3+2][0:2] = [0, 0]
    for k in range(num_of_fourier_terms):
        C[l*3][2+k*2:2+(k+1)*2] = [alpha[k][l], beta[k][l]]
        C[l*3+1][2+k*2:2+(k+1)*2] = [alpha_hat[k][l], beta_hat[k][l]]
        C[l*3+2][2+k*2:2+(k+1)*2] = [gamma[k][l], delta[k][l]]
C = np.matrix(C)

# Remove column beta_0 as all terms are zero
C = np.delete(C, 3, axis=1)

# Set a1 = b1 = 0 and remove these columns
C = np.delete(C, 3, axis=1)
C = np.delete(C, 4, axis=1)

# Delete last column so that C is a (3n x 3n) matrix
while C.shape[0] < C.shape[1]:
    print("C shape is", C.shape, "removing last column...")
    C = np.delete(C, -1, axis=1)


# Solve directly:
# x = np.linalg.solve(C, d)

# Solve using Singular Value Decomposition
# U, s, V = np.linalg.svd(C, full_matrices=True)
# Cinv_svd = np.dot(np.dot(V.T,np.linalg.inv(np.diag(s))),U.T)
x = np.linalg.pinv(C) * d

# x = [zeta, eta, a0/2, a2, b2, a3, b3, ..., ak, bk]
zeta, eta = x[0:2]

# Check solution is correct
print(np.allclose(np.dot(C, x), d))

# Source positions (zeta, eta)
xsrc = zeta * cos(eta) + zeropoint[0]
ysrc = zeta * sin(eta) + zeropoint[1]
pass



