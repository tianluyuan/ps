import numpy as np
from photospline import glam_fit, ndsparse

from matplotlib import pyplot as plt

# bin centers
PHI = np.array([2.5,   7.5,  12.5,  17.5,  22.5,  27.5,  32.5,  37.5,  42.5,
                47.5,  52.5,  57.5,  62.5,  67.5,  72.5,  77.5,  82.5,  87.5,
                92.5,  97.5, 102.5, 107.5, 112.5, 117.5, 122.5, 127.5, 132.5,
                137.5, 142.5, 147.5, 152.5, 157.5, 162.5, 167.5, 172.5, 177.5])
CTH = np.array([-0.99, -0.97, -0.95, -0.93, -0.91, -0.89, -0.87, -0.85, -0.83,
                -0.81, -0.79, -0.77, -0.75, -0.73, -0.71, -0.69, -0.67, -0.65,
                -0.63, -0.61, -0.59, -0.57, -0.55, -0.53, -0.51, -0.49, -0.47,
                -0.45, -0.43, -0.41, -0.39, -0.37, -0.35, -0.33, -0.31, -0.29,
                -0.27, -0.25, -0.23, -0.21, -0.19, -0.17, -0.15, -0.13, -0.11,
                -0.09, -0.07, -0.05, -0.03, -0.01,  0.01,  0.03,  0.05,  0.07,
                0.09,  0.11,  0.13,  0.15,  0.17,  0.19,  0.21,  0.23,  0.25,
                0.27,  0.29,  0.31,  0.33,  0.35,  0.37,  0.39,  0.41,  0.43,
                0.45,  0.47,  0.49,  0.51,  0.53,  0.55,  0.57,  0.59,  0.61,
                0.63,  0.65,  0.67,  0.69,  0.71,  0.73,  0.75,  0.77,  0.79,
                0.81,  0.83,  0.85,  0.87,  0.89,  0.91,  0.93,  0.95,  0.97,
                0.99])

# values
# take the log for stability (ymmv)
VLU = np.log(np.load('mwe.npy'))

# knot locations (a bit of magic)
NKNOTS_PHI = 23
NKNOTS_CTH = 53
# make sure to extend beyond the range of the coordinates for better fits
PHI_KNOTS = np.linspace(-17.5, 180+17.5, NKNOTS_PHI)
CTH_KNOTS = np.linspace(-1.1, 1.1, NKNOTS_CTH)

# scaling for photon yield
LIGHTFACTOR = 32582.0*5.21

@np.vectorize
def vecfeval_abs(a, phi, cth):
    return LIGHTFACTOR*np.exp()


def plot_values():
    plt.clf()
    plt.imshow(LIGHTFACTOR*np.exp(VLU),
               extent=[-1, 1, 0, 180],
               vmin=0,
               vmax=0.003,
               origin='lower',
               cmap='viridis',
               aspect='auto')
    plt.grid(None)
    plt.xlabel('cth')
    plt.ylabel('phi')
    plt.savefig('vlu.png')


def plot_spline(a):
    plt.clf()
    phi_fine = np.linspace(PHI[0], PHI[-1], 200)
    cth_fine = np.linspace(CTH[0], CTH[-1], 200)
    vals = np.asarray([a.evaluate_simple([phi, cth]) for phi in phi_fine for cth in cth_fine])
    vals = LIGHTFACTOR * np.exp(vals.reshape(200, 200))
    plt.imshow(vals,
               extent=[-1, 1, 0, 180],
               vmin=0,
               vmax=0.003,
               origin='lower',
               cmap='viridis',
               aspect='auto')
    plt.grid(None)
    # [plt.axvline(_, 0, 180, color='k', linewidth=0.5, linestyle='--') for _ in CTH_KNOTS]
    # [plt.axhline(_, -1, 1, color='k', linewidth=0.5, linestyle='--') for _ in PHI_KNOTS]
    plt.xlabel('cth')
    plt.ylabel('phi')
    plt.xlim(-1, 1)
    plt.ylim(0, 180)
    plt.savefig('spl.png')


_data, w = ndsparse.from_data(VLU, 1000*np.ones(VLU.shape))
spline = glam_fit(_data,
                  w,
                  [PHI, CTH],
                  [PHI_KNOTS, CTH_KNOTS],
                  [2, 2], # second order polynominals
                  [0.01, 0.01], # penalty, increasing can help prevent overfitting
                  penaltyOrder=[2, 2]) # 2nd order regularization (curvature)
spline.write('out.fits')

plot_values()
plot_spline(spline)
