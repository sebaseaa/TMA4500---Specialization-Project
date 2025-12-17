'''
Utilities for predicting the solution to the 2D wave equation using time stepping DeepONet.
'''

import numpy as np
import torch
import deepxde as dde
import matplotlib.pyplot as plt
from matplotlib import cm

dde.backend.set_default_backend("pytorch")
dde.config.set_default_float("float32")
torch.set_default_dtype(torch.float32)

# sensors
Nx_s, Ny_s = 20, 20
M = Nx_s * Ny_s

# time grid
T, K   = 1.0, 10
dt_step = T / K

# visualization grid
Nx_viz = 101

# wave equation parameters
two_pi = 2.0 * np.pi
c      = 1.0
omega  = two_pi * c  

# colormaps
cmap_field = "viridis"
cmap_error = "inferno"


def plot_rel_L2_wave(times: list, rel_errors: list, fname: str = "relL2_vs_time_DeepONet.png") -> None:
    '''Plot relative L^2 error vs time.'''

    plt.figure(figsize=(6.5, 4))
    plt.plot(times, rel_errors, "-o", markersize=4)
    plt.xlabel("Time")
    plt.ylabel(r"Relative $L^2$ error")
    plt.title(r"Relative $L^2$ error vs time (time-stepping DeepONet)")
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.tight_layout()
    plt.savefig(fname, dpi=300)
    plt.close()


def build_sensors(Nx: int, Ny: int) -> np.ndarray:
    '''Build sensor grid on [0,1]^2 with Nx x Ny points.'''

    xs = np.linspace(0.0, 1.0, Nx, endpoint=False)
    ys = np.linspace(0.0, 1.0, Ny, endpoint=False)
    Xs, Ys = np.meshgrid(xs, ys, indexing="xy")
    S = np.stack([Xs.ravel(), Ys.ravel()], axis=1).astype(np.float32)
    return S


def build_viz_grid(N: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    '''Build visualization grid on [0,1]^2 with N x N points.'''

    x = np.linspace(0.0, 1.0, N, dtype=np.float32)
    y = np.linspace(0.0, 1.0, N, dtype=np.float32)
    X, Y = np.meshgrid(x, y, indexing="xy")
    XY = np.stack([X.ravel(), Y.ravel()], axis=1).astype(np.float32)
    return X, Y, XY


def u_spatial_part_xy(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    '''Spatial part for the manufactured solution.'''

    return np.sin(two_pi * x) + np.cos(two_pi * y)


def analytic_u_xy_t(x: np.ndarray, y: np.ndarray, t: float) -> np.ndarray:
    '''Analytic displacement u(x,y,t).'''

    return u_spatial_part_xy(x, y) * np.cos(omega * t)


def analytic_v_xy_t(x: np.ndarray, y: np.ndarray, t: float) -> np.ndarray:
    '''Analytic v=u_t(x,y,t).'''

    return -omega * u_spatial_part_xy(x, y) * np.sin(omega * t)


def rel_l2(a: np.ndarray, b: np.ndarray, eps: float = 1e-12) -> float:
    '''Relative L^2 norm between arrays a and b.'''

    a = np.asarray(a)
    b = np.asarray(b)
    num = np.linalg.norm(a - b)
    den = np.linalg.norm(b)         
    return float(num / (den + eps)) # add eps to avoid division by zero


def branch_from_uv_on_sensors(u_sensor_vals: np.ndarray, v_sensor_vals: np.ndarray) -> np.ndarray:
    '''Build branch input from sensor u and v values.'''

    return np.concatenate([u_sensor_vals.T, v_sensor_vals.T], axis=1).astype(np.float32)


def make_branch_from_ic(sensors_xy: np.ndarray, t0: float = 0.0) -> np.ndarray:
    '''Build initial branch input from analytic (u,v) at time t0 on sensor locations.'''

    x = sensors_xy[:, 0:1]
    y = sensors_xy[:, 1:2]
    u0 = analytic_u_xy_t(x, y, t0).astype(np.float32)
    v0 = analytic_v_xy_t(x, y, t0).astype(np.float32)
    return branch_from_uv_on_sensors(u0, v0)


def make_net(M: int) -> dde.nn.DeepONetCartesianProd:
    '''Create DeepONet with specified architecture for 2M branch inputs and (x,y) trunk inputs.'''

    return dde.nn.DeepONetCartesianProd(
        [2 * M, 128, 256],  # branch
        [2, 128, 128],      # trunk (x,y)
        "tanh",
        "Glorot normal",
        num_outputs=2,
        multi_output_strategy="split_branch",
    )


def restore_model(pt_path: str, M: int) -> dde.Model:
    '''Restore the model from given path.'''
    
    geom = dde.geometry.Rectangle([0.0, 0.0], [1.0, 1.0])

    def pde_dummy(x, y):
        return y

    data_dummy = dde.data.PDE(geom, pde_dummy, [], num_domain=1, num_boundary=0)

    net = make_net(M)
    model = dde.Model(data_dummy, net)
    model.compile("L-BFGS")
    model.restore(pt_path, verbose=0)
    return model


@torch.no_grad()
def predict_uv(
    model: dde.model,
    branch_row_np: np.ndarray,
    XY_np: np.ndarray,
    chunk: int = 4096,
):
    '''Predict u and v on points XY_np using model and branch input branch_row_np.'''

    net = model.net
    p0 = next(net.parameters())
    dev, dt = p0.device, p0.dtype

    xb = torch.as_tensor(branch_row_np, dtype=dt, device=dev)
    xt = torch.as_tensor(XY_np,  dtype=dt, device=dev)         

    outs = []
    P = xt.shape[0]
    step_size = max(1, min(chunk, P))

    for s in range(0, P, step_size):
        e = min(P, s + step_size)
        yp = net([xb, xt[s:e, :]])
        if yp.ndim == 3:
            yp = yp[0]              
        outs.append(yp)

    y = torch.cat(outs, 0).cpu().numpy()
    return y[:, 0], y[:, 1]


def predict_on_points(
    model: dde.model,
    branch_row: np.ndarray,
    XY_points: np.ndarray,
    side: int,
) -> tuple[np.ndarray, np.ndarray]:
    '''Predict u and v on a grid of points using the model and branch input.'''

    u, v = predict_uv(model, branch_row, XY_points)
    return u.reshape(side, side), v.reshape(side, side)


def predict_on_sensors(
    model: dde.model,
    branch_row: np.ndarray,
    sensors_xy: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    '''Predict u and v on sensor points using the model and branch input.'''

    u, v = predict_uv(model, branch_row, sensors_xy)
    return u.reshape(-1, 1).astype(np.float32), v.reshape(-1, 1).astype(np.float32)


def save_3d(
    Xg: np.ndarray,
    Yg: np.ndarray,
    U_pred: np.ndarray,
    U_true: np.ndarray,
    t_value: float,
    rel_L2_u: float,
    out_path: str,
    elev: int = 30,
    azim: int = 35,
) -> None:
    '''Save 3D plot: predicted vs true vs error.'''

    Err = np.abs(U_pred - U_true)

    vmin = float(min(U_pred.min(), U_true.min()))
    vmax = float(max(U_pred.max(), U_true.max()))
    zabs = max(abs(vmin), abs(vmax))
    zlim = (-zabs, zabs)

    emax = float(Err.max())
    err_zlim = (0.0, emax if emax > 0 else 1.0)

    fig = plt.figure(figsize=(16, 5.5))

    ax1 = fig.add_subplot(1, 3, 1, projection="3d")
    s1 = ax1.plot_surface(
        Xg,
        Yg,
        U_pred,
        rstride=2,
        cstride=2,
        linewidth=0,
        antialiased=True,
        cmap=cm.get_cmap(cmap_field),
        vmin=-zabs,
        vmax=zabs,
    )
    ax1.set_title(f"Predicted solution at t={t_value:.2f}")
    ax1.set_xlabel("x")
    ax1.set_ylabel("y")
    ax1.set_zlabel("u")
    ax1.set_zlim(*zlim)
    ax1.view_init(elev=elev, azim=azim)

    ax2 = fig.add_subplot(1, 3, 2, projection="3d")
    s2 = ax2.plot_surface(
        Xg,
        Yg,
        U_true,
        rstride=2,
        cstride=2,
        linewidth=0,
        antialiased=True,
        cmap=cm.get_cmap(cmap_field),
        vmin=-zabs,
        vmax=zabs,
    )
    ax2.set_title(f"Analytic solution at t={t_value:.2f}")
    ax2.set_xlabel("x")
    ax2.set_ylabel("y")
    ax2.set_zlim(*zlim)
    ax2.view_init(elev=elev, azim=azim)

    cbar_main = fig.colorbar(s2, ax=[ax1, ax2], shrink=0.70, pad=0.06)
    cbar_main.set_label("u(x,y,t)")

    ax3 = fig.add_subplot(1, 3, 3, projection="3d")
    s3 = ax3.plot_surface(
        Xg,
        Yg,
        Err,
        rstride=2,
        cstride=2,
        linewidth=0,
        antialiased=True,
        cmap=cm.get_cmap(cmap_error),
        vmin=err_zlim[0],
        vmax=err_zlim[1],
    )
    ax3.set_title("Absolute error")
    ax3.set_xlabel("x")
    ax3.set_ylabel("y")
    ax3.set_zlim(*err_zlim)
    ax3.view_init(elev=elev, azim=azim)

    cbar_err = fig.colorbar(s3, ax=ax3, shrink=0.70, pad=0.06)
    cbar_err.set_label("Error")

    for ax in (ax1, ax2, ax3):
        ax.set_box_aspect((1, 1, 0.5))

    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
