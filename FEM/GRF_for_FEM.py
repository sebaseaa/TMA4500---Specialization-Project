'''
GRFs modified to be Firedrake friendly.

Note: this is built on the source code in DeepXDE.
'''

from scipy import interpolate
import numpy as np

# due to having firedrake and deeponet in two different virtual environments
# we make the function space class here again (slightly modified from vanilla_deeponet/modified_GRF_classes.py, to not use DeepXDE)
def _cov_exp_sine_squared(X : np.ndarray, Y: np.ndarray, lx: float, ly: float, px=1.0, py=1.0) -> np.ndarray:
    '''Product of 1D ExpSineSquared kernels per axis (strictly periodic per axis).'''

    X = np.asarray(X)
    Y = np.asarray(Y)

    dx = np.abs(X[:, None, 0] - Y[None, :, 0])
    dy = np.abs(X[:, None, 1] - Y[None, :, 1])

    kx = np.exp(-2.0 * np.sin(np.pi * dx / px) ** 2 / (lx ** 2))
    ky = np.exp(-2.0 * np.sin(np.pi * dy / py) ** 2 / (ly ** 2))
    return kx * ky


class PeriodicGRF2D:
    '''
    Gaussian random field on [0,1]^2 with periodicity in x and y. 
    
    Note: this is built on the source code in DeepXDE
    '''

    def __init__(self, kernel="ExpSineSquared", length_scale=0.25,
                 N=100, interp="splinef2d"):
        self.N = N
        self.interp = interp
        self.x = np.linspace(0.0, 1.0, num=N)
        self.y = np.linspace(0.0, 1.0, num=N)

        xv, yv = np.meshgrid(self.x, self.y, indexing="xy")
        self.X = np.stack([xv.ravel(), yv.ravel()], axis=1)

        if np.isscalar(length_scale):
            lx = ly = float(length_scale)
        else:
            lx, ly = map(float, length_scale)

        K = _cov_exp_sine_squared(self.X, self.X, lx, ly, px=1.0, py=1.0)
        jitter = 1e-12
        self.L = np.linalg.cholesky(K + jitter * np.eye(K.shape[0]))

    def random(self, size):
        z = np.random.randn(self.N * self.N, size)
        return (self.L @ z).T  

    def _wrap_periodic(self, x):
        xw = np.copy(x)
        xw[..., 0] = np.mod(xw[..., 0], 1.0)
        xw[..., 1] = np.mod(xw[..., 1], 1.0)
        return xw

    def eval_batch(self, features, xs):
        pts = self._wrap_periodic(xs)[..., :2]
        grids = features.reshape(-1, self.N, self.N)
        vals = [
            interpolate.interpn((self.x, self.y), g, pts, method=self.interp)
            for g in grids
        ]
        arr = np.vstack(vals)
        return np.real(arr).astype(np.float64)
