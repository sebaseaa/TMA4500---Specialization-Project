import os
import math
import numpy as np
import torch
import deepxde as dde

dde.backend.set_default_backend("pytorch")
dde.config.set_default_float("float32")
dde.config.set_default_autodiff("reverse")
torch.set_default_dtype(torch.float32)

# time parameters
K = 10
T = 1.0
dt_np = np.float32(T / K)
n_steps = 10

# GRF and sensors
l_domain = 1.0
Nx_s, Ny_s = 20, 20

# DeepONet / training parameters
num_functions = 500
num_test_funcs = 100
num_domain = 600
num_boundary = 100
batch_size = 256

lr = 3e-4
warmup = 1000
tmax = 9000
l_bfgs_steps = 5000

branch_layers = [128, 256]
trunk_layers = [128, 128]
activation = "tanh"
init = "Glorot normal"
num_outputs = 2
output_strategy = "split_branch"
loss_weights_base = [1, 1, 1, 1, 1, 1]
ru_weight_stage_after_1 = 1.0


this_dir = os.path.dirname(os.path.abspath(__file__))
save_dir = this_dir
save_prefix = "split_trunk"

# geometry and BCS
geom_xy = dde.geometry.Rectangle([0.0, 0.0], [l_domain, l_domain])

def on_x0(X, on_b): 
    return on_b and np.isclose(X[0], 0.0)

def on_y0(X, on_b): 
    return on_b and np.isclose(X[1], 0.0)

bc_u_per_x0 = dde.icbc.PeriodicBC(geom_xy, component_x=0, on_boundary=on_x0, derivative_order=0)
bc_u_per_y0 = dde.icbc.PeriodicBC(geom_xy, component_x=1, on_boundary=on_y0, derivative_order=0)
bc_v_per_x0 = dde.icbc.PeriodicBC(geom_xy, component_x=0, on_boundary=on_x0, component=1)
bc_v_per_y0 = dde.icbc.PeriodicBC(geom_xy, component_x=1, on_boundary=on_y0, component=1)

# GRF space utils
# much of this is similar to the source code of DeepXDE
def periodic_exp_sine_squared_1d(x : torch.Tensor, z: torch.Tensor, length_scale=7.0, period=1.0) -> torch.Tensor:
    '''Periodic exponential sine squared kernel in 1D.'''
    diff = x - z.T
    s = torch.sin(math.pi * diff / period)
    return torch.exp(-2.0 * (s ** 2) / (length_scale ** 2))


def periodic_exp_sine_squared_2d(x : torch.Tensor, z: torch.Tensor, lx=7.0, ly=7.0, period_x=1.0, period_y=1.0,
                                 ) -> torch.Tensor:
    '''Build the two-dimensional periodic exponential sine squared kernel.'''

    kx = periodic_exp_sine_squared_1d(x[:, :1], z[:, :1], length_scale=lx, period=period_x)
    ky = periodic_exp_sine_squared_1d(x[:, 1:2], z[:, 1:2], length_scale=ly, period=period_y)
    return kx * ky


def stable_cholesky(K : torch.Tensor, jitter : float =1e-8, max_tries : int =8) -> torch.Tensor:
    '''Cholesky decomposition with added jitter for numerical stability.'''

    K = 0.5 * (K + K.T)

    diag_mean = torch.diagonal(K).abs().mean()
    if diag_mean <= 0:
        diag_mean = K.new_tensor(1.0)

    jitter_local = jitter * diag_mean
    I = torch.eye(K.shape[0], device=K.device, dtype=K.dtype)

    for _ in range(max_tries):
        try:
            return torch.linalg.cholesky(K + jitter_local * I)
        except RuntimeError:
            jitter_local *= 10.0

    raise RuntimeError(
        f"Cholesky failed even with jitter={float(jitter_local):.3e}; "
        "matrix may be badly conditioned."
    )


def laplacian_fd(U: np.ndarray, Lx: float = l_domain, Ly: float = l_domain) -> np.ndarray:
    '''
    Compute the 2D Laplacian of a batch of periodic grids using finite differences.
    '''

    # ensure numerical stability and correct dtype
    U = np.nan_to_num(U, copy=False).astype(np.float32, copy=False)
    B, Ny, Nx = U.shape

    # grid 
    hx = Lx / Nx
    hy = Ly / Ny
    inv_hx2 = np.float32(1.0 / (hx * hx))
    inv_hy2 = np.float32(1.0 / (hy * hy))

    # shifted grids for FD
    U_xp = np.roll(U, -1, axis=2)
    U_xm = np.roll(U,  1, axis=2)
    U_yp = np.roll(U, -1, axis=1)
    U_ym = np.roll(U,  1, axis=1)

    # stencil
    lap = (U_xp - 2.0 * U + U_xm) * inv_hx2 \
        + (U_yp - 2.0 * U + U_ym) * inv_hy2

    return lap.astype(np.float32, copy=False)


def periodic_bilinear_eval_batch(U : np.ndarray, pts: np.ndarray, Lx: float = l_domain, Ly: float = l_domain) -> np.ndarray:
    '''Periodic bilinear interpolation of batch of grids U at points pts.'''

    U = np.nan_to_num(U, copy=False).astype(np.float32, copy=False)
    pts = np.nan_to_num(pts, copy=False).astype(np.float32, copy=False)

    B, Ny, Nx = U.shape

    # map physical coordinates to grid indices
    x = (pts[:, 0] % Lx) * (Nx / Lx)
    y = (pts[:, 1] % Ly) * (Ny / Ly)

    # integer grid indices
    j0 = np.floor(x).astype(np.int64)
    i0 = np.floor(y).astype(np.int64)
    j1 = (j0 + 1) % Nx
    i1 = (i0 + 1) % Ny

    # local coordinates within the grid cell
    tx = (x - j0).astype(np.float32)[:, None]
    ty = (y - i0).astype(np.float32)[:, None]

    # corner values for bilinear interpolation
    U_i0j0 = U[:, i0, j0]
    U_i0j1 = U[:, i0, j1]
    U_i1j0 = U[:, i1, j0]
    U_i1j1 = U[:, i1, j1]

    # bilinear interpolation 
    ax = (1.0 - tx.T) * U_i0j0 + tx.T * U_i0j1
    bx = (1.0 - tx.T) * U_i1j0 + tx.T * U_i1j1
    out = (1.0 - ty.T) * ax + ty.T * bx

    return out.astype(np.float32, copy=False)


class FunctionSpacePairGRF:
    '''
    Function space for (u, v) pairs defined by a 2D periodic GRF sampled on the sensor grid.
    Differs from the source code in DeepXDE by sampling both u and v.
    If eval_batch is called at the sensor locations, it returns the branch input directly.
    If eval_batch is called at other points, it computes u and its laplacian via finite differences 
    and interpolation.
    '''

    def __init__(self, sensors_np, lx=7.0, ly=7.0,
                 device="cpu", dtype=torch.float32):
        # similar to source code
        self.device = torch.device(device)
        self.dtype = dtype

        self.X_np = np.asarray(sensors_np, dtype=np.float32)
        self.M = self.X_np.shape[0]

        self.X = torch.as_tensor(self.X_np, dtype=self.dtype, device=self.device)
        self.lx = float(lx)
        self.ly = float(ly)

        with torch.no_grad():
            K_XX = periodic_exp_sine_squared_2d(self.X, self.X,
                                                lx=self.lx, ly=self.ly)
            L = stable_cholesky(K_XX)

        self.L = L
        self.eval_pts_ref = None  # used to detect branch evaluation

    def random(self, size: int):
        # similar to source code, but samples both u and v
        B = int(size)
        M = self.M

        z_u = torch.randn(B, M, dtype=self.dtype, device=self.device)
        z_v = torch.randn(B, M, dtype=self.dtype, device=self.device)

        with torch.no_grad():
            uX = z_u @ self.L.T
            vX = z_v @ self.L.T

        feats = torch.cat([uX, vX], dim=1)
        return feats.cpu().numpy().astype(np.float32)

    def eval_batch(self, features, xs):
        '''
        Similar to source code, but:

        - If xs == sensors:
            return branch input [u(X), v(X)].
        - Else:
            compute u(xs) and its laplacian via finite differences and interpolation.
        '''
        ft = np.asarray(features, dtype=np.float32)
        M = self.M

        uX_np = ft[:, :M]
        vX_np = ft[:, M:]
        xs_np = np.asarray(xs, dtype=np.float32)

        # detect branch 
        is_branch = (
            self.eval_pts_ref is not None
            and xs_np.shape[0] == self.eval_pts_ref.shape[0]
            and np.allclose(xs_np, self.eval_pts_ref, rtol=0.0, atol=0.0)
        )

        if is_branch:
            return ft.astype(np.float32)

        # reshape to grid for finite differences
        B = ft.shape[0]
        Ugrid = uX_np.reshape(B, Ny_s, Nx_s)
        Vgrid = vX_np.reshape(B, Ny_s, Nx_s)

        lap_grid = laplacian_fd(Ugrid)

        # interpolate to collocation points
        uY = periodic_bilinear_eval_batch(Ugrid, xs_np)
        lapY = periodic_bilinear_eval_batch(lap_grid, xs_np)
        vY = periodic_bilinear_eval_batch(Vgrid, xs_np)

        # return concatenated result
        return np.concatenate([uY, lapY, vY], axis=1).astype(np.float32)


class PrevStepFunctionSpace:
    '''
    Function space used for stages >= 1.

    Instead of sampling (u,v) directly from a GRF, we:
      1) sample a base GRF (u^0, v^0) on sensors
      2) push it through the previously trained operator
      3) use the result as the new branch input
    '''

    def __init__(self, base_space, sensors_np, prev_net):
        self.base = base_space
        self.sensors = sensors_np.astype(np.float32)
        self.prev_net = prev_net

        self.Mloc = self.sensors.shape[0]
        self._last_branch0_list = []
        self._last_branch_s_list = []

    def random(self, size):
        '''
        Generate branch data by:
        GRF -> previous operator -> sensor values.
        '''
        feats = self.base.random(size)
        self._last_branch0_list = []
        self._last_branch_s_list = []
        self.base.eval_pts_ref = self.sensors.copy()

        for i in range(size):
            branch0 = feats[i:i+1, :].astype(np.float32)

            # evaluate previous operator at sensors
            u_s, v_s, _ = net_autograd_uv_and_lap(
                self.prev_net, branch0, self.sensors, grad_requires=False
            )

            branch_s = np.concatenate(
                [u_s.reshape(1, self.Mloc), v_s.reshape(1, self.Mloc)],
                axis=1
            ).astype(np.float32)

            self._last_branch0_list.append(branch0)
            self._last_branch_s_list.append(branch_s)

        return feats

    def eval_batch(self, features, xs):
        '''
        Evaluation follows DeepXDE convention:
        
        - branch points: return cached pushed-forward branch
        - collocation points: compute u and its laplacian with autograd
        '''

        xs_np = np.asarray(xs, dtype=np.float32)
        is_branch = (
            self.base.eval_pts_ref is not None
            and xs_np.shape[0] == self.base.eval_pts_ref.shape[0]
            and np.allclose(xs_np, self.base.eval_pts_ref, rtol=0.0, atol=0.0)
        )
        B = np.asarray(features).shape[0]

        if is_branch:
            return np.concatenate(self._last_branch_s_list[:B], axis=0)

        P = xs_np.shape[0]
        rows = []
        for i in range(B):
            branch0 = self._last_branch0_list[i]
            u, v, lap = net_autograd_uv_and_lap(
                self.prev_net, branch0, xs_np, grad_requires=True
            )
            rows.append(
                np.concatenate([u.reshape(1, P),
                                lap.reshape(1, P),
                                v.reshape(1, P)], axis=1)
            )

        return np.concatenate(rows, axis=0).astype(np.float32)

# sensors
xs = np.linspace(0, l_domain, Nx_s, endpoint=False)
ys = np.linspace(0, l_domain, Ny_s, endpoint=False)
Xs, Ys = np.meshgrid(xs, ys, indexing="xy")
sensors = np.stack([Xs.ravel(), Ys.ravel()], axis=1).astype(np.float32)
M = sensors.shape[0]
func_space = FunctionSpacePairGRF(
    sensors_np=sensors, lx=7, ly=7, device="cpu", dtype=torch.float32
)
func_space.eval_pts_ref = sensors.copy()

def _parse_aux(aux : torch.Tensor, P: int, like: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    '''
    Send auxilary variables (i.e., u0, lap0,v0)
    '''

    aux = torch.as_tensor(aux, dtype=like.dtype, device=like.device)

    u0   = aux[0:P, :]
    lap0 = aux[P:2 * P, :]
    v0   = aux[2 * P:3 * P, :]
    return u0, lap0, v0


def _normalize_y_shape(y: torch.Tensor) -> torch.Tensor:
    '''
    Normalize shape of y.
    '''

    if not torch.is_tensor(y):
        y = torch.as_tensor(y)

    if y.ndim == 3 and y.shape[0] == 1:
        y = y[0]  

    return y


def pde_FE(x : torch.Tensor, y: torch.Tensor, aux: torch.Tensor) -> list[torch.Tensor]:
    '''
    Forward Euler residual:

    Ru = u_{t+1} - u_t - dt * v_t
    Rv = v_{t+1} - v_t - dt * Δu_t
    '''
    y = _normalize_y_shape(y)  
    u1 = y[:, 0:1]
    v1 = y[:, 1:2]
    P  = u1.shape[0]

    u0, lap0, v0 = _parse_aux(aux, P, like=u1)

    dt = u1.new_tensor(dt_np)

    Ru = u1 - u0 - dt * v0
    Rv = v1 - v0 - dt * lap0
    return [torch.nan_to_num(Ru), torch.nan_to_num(Rv)]


def make_net() -> dde.nn.DeepONetCartesianProd:
    '''
    Make the time-stepping DeepONet.
    '''

    branch = [2 * M] + branch_layers
    trunk  = [2]     + trunk_layers
    net = dde.nn.DeepONetCartesianProd(
        branch, trunk, activation, init,
        num_outputs=num_outputs,
        multi_output_strategy=output_strategy,
    )
    return net

def net_autograd_uv_and_lap(prev_net : dde.nn.DeepONetCartesianProd, branch_np: np.ndarray, trunk_np: np.ndarray, grad_requires: bool = True) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    '''
    Compute u, v, and laplacian of u via autograd using prev_net.
    '''

    xb = torch.as_tensor(branch_np, dtype=torch.float32)
    xt = torch.as_tensor(trunk_np,  dtype=torch.float32)
    if xt.ndim == 1:
        xt = xt[None, :]

    if grad_requires:
        xt.requires_grad_(True)

    y = prev_net([xb, xt])         
    u = y[:, 0:1]
    v = y[:, 1:2]

    if not grad_requires:
        lap = torch.zeros_like(u)
    else:
        gu = torch.autograd.grad(u.sum(), xt, create_graph=True)[0]
        u_xx = torch.autograd.grad(gu[:, 0].sum(), xt, create_graph=True)[0][:, 0:1]
        u_yy = torch.autograd.grad(gu[:, 1].sum(), xt, create_graph=True)[0][:, 1:2]
        lap = u_xx + u_yy

    return u.detach().cpu().numpy(), v.detach().cpu().numpy(), lap.detach().cpu().numpy()


def make_dataset(stage_idx: int, trained_tags_before_this_stage: list[str] = None) -> dde.data.PDEOperatorCartesianProd:
    '''
    Stage 0: GRF data
    Stage >= 1: data generated with GRF and all previously trained operators.
    '''
    pde_data = dde.data.PDE(
        geometry=geom_xy,
        pde=pde_FE,
        bcs=[bc_u_per_x0, bc_u_per_y0, bc_v_per_x0, bc_v_per_y0],
        num_domain=num_domain,
        num_boundary=num_boundary,
    )

    if stage_idx == 0:
        fs_used = func_space
    else:
        prev_tag = trained_tags_before_this_stage[stage_idx - 1]
        prev_net = make_net()

        prev_path = os.path.join(save_dir, prev_tag)
        if not os.path.exists(prev_path):
            prev_path = prev_path + ".pt"

        ckpt = torch.load(prev_path, map_location="cpu")
        if isinstance(ckpt, dict) and "model_state_dict" in ckpt:
            ckpt = ckpt["model_state_dict"]
        prev_net.load_state_dict(ckpt, strict=False)

        fs_used = PrevStepFunctionSpace(func_space, sensors, prev_net)

    data = dde.data.PDEOperatorCartesianProd(
        pde=pde_data,
        function_space=fs_used,
        evaluation_points=sensors,
        num_function=num_functions,
        num_test=num_test_funcs,
        batch_size=batch_size,
    )
    return data


def warmup_cosine(step: int) -> float:
    '''
    Learning rate decay
    '''
    if step < warmup:
        return (step + 1) / warmup
    t = step - warmup
    return 0.1 + 0.9 * 0.5 * (1 + np.cos(np.pi * t / tmax))
