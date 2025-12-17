'''
Use DeepONet to extrapolate wave equation solutions with GRF ICs with different length scales,
comparing against FEM reference solutions.
'''

import os
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D   
from scipy import interpolate
import deepxde as dde

dde.backend.set_default_backend("pytorch")
from modified_GRF_classes import PeriodicGRF2D

# colormaps
cmap_field = "viridis"
cmap_error = "inferno"

# wave speed
c = 1.0

# paths and checkpoint
ckpt_path = "vanilladeeponet-16000.pt"      
ref_dir = "fem_grf_reference"                
save_root = "extrapolation_results"    
os.makedirs(save_root, exist_ok=True)

# sensor grid (must match training)
Nx_s, Ny_s = 20, 20

def rel_l2(pred: np.ndarray, true: np.ndarray) -> float:
    '''Relative L^2 norm between arrays pred and true.'''

    pred = np.asarray(pred, dtype=np.float64)
    true = np.asarray(true, dtype=np.float64)
    num = np.linalg.norm((pred - true).ravel())
    den = np.linalg.norm(true.ravel())
    return float(num / (den + 1e-12))


def make_trunk_inputs(Xg: np.ndarray, Yg: np.ndarray, t_val: float) -> np.ndarray:
    '''Build trunk input [x, y, t] for a given time t_val on grid (Xg, Yg).'''

    T = np.zeros_like(Xg, dtype=np.float32) + np.float32(t_val)
    return np.stack([Xg.ravel(), Yg.ravel(), T.ravel()], axis=1).astype(np.float32)


def plot_snapshot_3d(
    Xg: np.ndarray,
    Yg: np.ndarray,
    u_true_T: np.ndarray,
    u_pred_T: np.ndarray,
    t_value: float,
    rel_l2_val: float,
    fname: str,
) -> None:
    '''3D plot: predicted, FEM reference, and absolute error at time t_value.'''

    err = np.abs(u_pred_T - u_true_T)

    vmin = float(min(u_true_T.min(), u_pred_T.min()))
    vmax = float(max(u_true_T.max(), u_pred_T.max()))
    zabs = max(abs(vmin), abs(vmax))
    vmin, vmax = -zabs, zabs
    z_lim = (-zabs, zabs)

    err_max = float(err.max())
    err_z_lim = (0.0, err_max)

    fig = plt.figure(figsize=(16, 5.5), dpi=150)

    ax1 = fig.add_subplot(1, 3, 1, projection="3d")
    s1 = ax1.plot_surface(
        Xg,
        Yg,
        u_pred_T,
        linewidth=0,
        antialiased=True,
        cmap=cm.get_cmap(cmap_field),
        vmin=vmin,
        vmax=vmax,
    )
    ax1.set_title(f"Predicted solution at t={t_value:.2f}")
    ax1.set_xlabel("x")
    ax1.set_ylabel("y")
    ax1.set_z_lim(*z_lim)
    ax1.view_init(elev=30, azim=35)

    ax2 = fig.add_subplot(1, 3, 2, projection="3d")
    s2 = ax2.plot_surface(
        Xg,
        Yg,
        u_true_T,
        linewidth=0,
        antialiased=True,
        cmap=cm.get_cmap(cmap_field),
        vmin=vmin,
        vmax=vmax,
    )
    ax2.set_title(f"FEM solution at t={t_value:.2f}")
    ax2.set_xlabel("x")
    ax2.set_ylabel("y")
    ax2.set_z_lim(*z_lim)
    ax2.view_init(elev=30, azim=35)

    cbar_main = fig.colorbar(s1, ax=[ax1, ax2], shrink=0.70, pad=0.06)
    cbar_main.set_label("u(x,y,t)")

    ax3 = fig.add_subplot(1, 3, 3, projection="3d")
    s3 = ax3.plot_surface(
        Xg,
        Yg,
        err,
        linewidth=0,
        antialiased=True,
        cmap=cm.get_cmap(cmap_error),
        vmin=0.0,
        vmax=err_max,
    )
    ax3.set_title("Absolute error")
    ax3.set_xlabel("x")
    ax3.set_ylabel("y")
    ax3.set_z_lim(*err_z_lim)
    ax3.view_init(elev=30, azim=35)

    cbar_err = fig.colorbar(s3, ax=ax3, shrink=0.70, pad=0.06)
    cbar_err.set_label("Error")

    for ax in (ax1, ax2, ax3):
        ax.set_box_aspect((1, 1, 0.5))

    fig.savefig(fname, dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_rel_L2(times: np.ndarray, rel_errors: list[float], fname: str, l: float) -> None:
    '''Plot relative L^2 error vs time for a given length scale l.'''

    plt.figure(figsize=(6.5, 4))
    plt.plot(times, rel_errors, "-o", markersize=4)
    plt.xlabel("Time")
    plt.ylabel(r"Relative $L^2$ error")
    plt.title(rf"Relative $L^2$ error vs time for $l={l}$")
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.tight_layout()
    plt.savefig(fname, dpi=300)
    plt.close()


# sensor grid 
xs = np.linspace(0.0, 1.0, Nx_s)
ys = np.linspace(0.0, 1.0, Ny_s)
Xs, Ys = np.meshgrid(xs, ys, indexing="xy")
sensors = np.vstack((np.ravel(Xs), np.ravel(Ys))).T.astype(np.float32)
m = sensors.shape[0]

# dummy TimePDE to build model, then restore weights
geom_xy = dde.geometry.Rectangle([0.0, 0.0], [1.0, 1.0])
timedomain = dde.geometry.TimeDomain(0.0, 1.0)
geomtime = dde.geometry.GeometryXTime(geom_xy, timedomain)


def pde_dummy(x, u, v):
    return u


data_dummy = dde.data.TimePDE(
    geomtime,
    pde_dummy,
    [],
    num_domain=1,
    num_boundary=0,
    num_initial=0,
)

dim_x = 3
net = dde.nn.DeepONetCartesianProd(
    [m, 128, 128, 128],      # branch
    [dim_x, 128, 128, 128],  # trunk (x,y,t)
    "tanh",
    "Glorot normal",
)
model = dde.Model(data_dummy, net)
model.compile("L-BFGS")
print(f"Restoring mode from: {ckpt_path}")
model.restore(ckpt_path, verbose=1)

length_scales = [1, 7, 12]

for ell in length_scales:
    print(f"\n Evaluating DeepONet vs FEM for l = {ell}")

    # FEM reference file
    npz_path = os.path.join(ref_dir, f"fem_reference_ell{int(ell)}.npz")
    ref = np.load(npz_path)

    times = ref["times"]          
    x_eval_1d = ref["x_eval_1d"]  
    y_eval_1d = ref["y_eval_1d"]
    u_all = ref["u_all"]        
    features = ref["features"] 
    N_grf = int(ref["N_grf"])

    Xg, Yg = np.meshgrid(x_eval_1d, y_eval_1d, indexing="xy")
    Ny_eval, Nx_eval = u_all.shape[1], u_all.shape[2]

    # GRF object to reuse eval_batch at sensor locations
    func_space = PeriodicGRF2D(
        kernel="ExpSineSquared",
        length_scale=ell,
        N=N_grf,
    )

    v_at_sensors = func_space.eval_batch(features, sensors)[0]
    v_branch = v_at_sensors[None, :].astype(np.float32)

    save_dir = os.path.join(save_root, f"ell{int(ell)}")
    os.makedirs(save_dir, exist_ok=True)

    rel_errors: list[float] = []

    for i, t in enumerate(times):
        u_true_t = u_all[i]  

        trunk_t = make_trunk_inputs(Xg, Yg, t)
        u_pred_t = model.predict((v_branch, trunk_t)).reshape(Ny_eval, Nx_eval)

        err_t = rel_l2(u_pred_t, u_true_t)
        rel_errors.append(err_t)

        print(f"t={t:.1f}, rel-L2 = {err_t:.3e}")

        fname3d = os.path.join(save_dir, f"snapshot_3D_t{t:.1f}.png")
        plot_snapshot_3d(Xg, Yg, u_true_t, u_pred_t, t, err_t, fname3d)

    fname_rel = os.path.join(save_dir, "relL2_vs_time.png")
    plot_rel_L2(times, rel_errors, fname_rel, l=ell)
    print(f"Saved rel-L2 curve to {fname_rel}")
