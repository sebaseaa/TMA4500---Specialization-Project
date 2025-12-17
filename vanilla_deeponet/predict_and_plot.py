'''
Plot the predicted solution of the 2D wave equation using a trained DeepONet,
and compare to the analytical solution.
'''

import os
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D 
from matplotlib import cm
from mpl_toolkits.axes_grid1 import make_axes_locatable
import deepxde as dde

dde.backend.set_default_backend("pytorch")

# config, paths, grids
c = 1.0
this_dir = os.path.dirname(os.path.abspath(__file__))
ckpt_path = os.path.normpath(os.path.join(this_dir, "vanilladeeponet-16000.pt"))
outdir = os.path.join(this_dir, "visualizations_wave2d")
os.makedirs(outdir, exist_ok=True)
cmap_field = "viridis"
cmap_error = "inferno"

# sensor and evaluation grids
Nx_s, Ny_s = 20, 20              
Nx_eval = 50                       
times = np.linspace(0.0, 1.0, 11)  
t_surface = 0.50

def rel_l2(pred : np.ndarray, true: np.ndarray) -> float:
    '''Relative L2 norm between prediction and true solution.'''
    
    pred = np.asarray(pred, dtype=np.float64)
    true = np.asarray(true, dtype=np.float64)
    return np.linalg.norm((pred - true).ravel()) / (np.linalg.norm(true.ravel()) + 1e-12)

def make_trunk_inputs(Xg : np.ndarray, Yg: np.ndarray, t_val: float) -> np.ndarray:
    '''Create trunk inputs [x,y,t].'''

    T = np.zeros_like(Xg, dtype=np.float32) + np.float32(t_val)
    return np.stack([Xg.ravel(), Yg.ravel(), T.ravel()], axis=1).astype(np.float32)

def plot_snapshot(
    u_true_T : np.ndarray,
    u_pred_T : np.ndarray,
    Xg : np.ndarray,
    Yg : np.ndarray,
    t_value : float,
    rel_l2_val : float,
    fname: str | None = None,
    cmap_main: str = cmap_field,
    figsize: tuple = (9.6, 3.6),
    dpi: int = 160,
):
    '''3D: predicted, analytical, and absolute error at time t_value.'''

    vmin = float(min(u_true_T.min(), u_pred_T.min()))
    vmax = float(max(u_true_T.max(), u_pred_T.max()))
    err  = np.abs(u_pred_T - u_true_T)

    fig, axs = plt.subplots(1, 3, figsize=figsize, dpi=dpi, constrained_layout=True)

    # predicted 
    im_pred = axs[0].pcolormesh(
        Xg, Yg, u_pred_T,
        shading="auto",
        cmap=cmap_main,
        vmin=vmin, vmax=vmax,
        rasterized=True,
    )
    axs[0].set_title("Predicted solution")
    axs[0].set_xticks([]); axs[0].set_yticks([])

    # analytical
    im_true = axs[1].pcolormesh(
        Xg, Yg, u_true_T,
        shading="auto",
        cmap=cmap_main,
        vmin=vmin, vmax=vmax,
        rasterized=True,
    )
    axs[1].set_title("Analytic Solution")
    axs[1].set_xticks([]); axs[1].set_yticks([])

    # error
    im_err = axs[2].pcolormesh(
        Xg, Yg, err,
        shading="auto",
        cmap=cmap_error,
        rasterized=True,
    )
    axs[2].set_title(f"Absolute error (rel L2 = {rel_l2_val:.2e})")
    axs[2].set_xticks([]); axs[2].set_yticks([])

    cbar_tp = fig.colorbar(im_true, ax=axs[:2], shrink=0.9)
    cbar_tp.set_label("u(x, y, t)")

    divider = make_axes_locatable(axs[2])
    cax_err = divider.append_axes("right", size="4.5%", pad=0.05)
    cbar_err = fig.colorbar(im_err, cax=cax_err)
    cbar_err.set_label("error")

    fig.suptitle(f"t = {t_value}", y=1.02)
    if fname:
        fig.savefig(fname, bbox_inches="tight")
        plt.close(fig)
    else:
        plt.show()

def plot_rel_L2_wave(times: np.ndarray, errs: np.ndarray, fname: str) -> None:
    '''Plot relative L2 error over time.'''

    plt.figure(figsize=(6.5, 4))
    plt.plot(times, errs, "-o", markersize=4)
    plt.xlabel("Time")
    plt.ylabel(r"Relative $L^2$ error")
    plt.title(r"Relative $L^2$ error vs time (DeepONet)")
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.tight_layout()
    plt.savefig(fname, dpi=300)
    plt.show()

# analytical solution
two_pi = 2.0 * np.pi
def ic_new(x, y):
    return np.sin(two_pi * x) + np.cos(two_pi * y)

def u_true(x, y, t, c=c):
    return (np.sin(two_pi * x) + np.cos(two_pi * y)) * np.cos(two_pi * c * t)

# evaluation grid
x = np.linspace(0, 1, Nx_eval, dtype=np.float32)
y = np.linspace(0, 1, Nx_eval, dtype=np.float32)
Xg, Yg = np.meshgrid(x, y, indexing="xy")

# sensors
xs = np.linspace(0, 1, Nx_s, dtype=np.float32)
ys = np.linspace(0, 1, Ny_s, dtype=np.float32)
Xs, Ys = np.meshgrid(xs, ys, indexing="xy")
sensors = np.vstack((np.ravel(Xs), np.ravel(Ys))).T.astype(np.float32)
m = sensors.shape[0]

# branch vector IC
v_branch_ic = ic_new(sensors[:, 0], sensors[:, 1])[None, :].astype(np.float32)  

# restore model
geom_xy = dde.geometry.Rectangle([0.0, 0.0], [1.0, 1.0])
timedomain = dde.geometry.TimeDomain(0.0, 1.0)
geomtime = dde.geometry.GeometryXTime(geom_xy, timedomain)

def pde_dummy(x, u, v):  
    return u

data_dummy = dde.data.TimePDE(geomtime, pde_dummy, [], num_domain=1, num_boundary=0, num_initial=0)

dim_x = 3  
net = dde.nn.DeepONetCartesianProd(
    [m, 128, 128, 128],    
    [dim_x, 128, 128, 128],  
    "tanh",
    "Glorot normal",
)
model = dde.Model(data_dummy, net)
model.compile("L-BFGS")
print(f"Restoring model (with optimizer) from: {ckpt_path}")
model.restore(ckpt_path, verbose=1)


def predict_u(Xmat : np.ndarray, Ymat: np.ndarray, t_scalar: float) -> np.ndarray:
    '''Predict u on grid at time t_scalar.'''

    trunk = make_trunk_inputs(Xmat, Ymat, t_scalar)  
    pred = model.predict((v_branch_ic, trunk)).reshape(Xmat.shape[0], Xmat.shape[1])
    return pred


# paths to store figures
surfaces_root = os.path.join(outdir, "surfaces")
combined_dir = os.path.join(surfaces_root, "combined")
os.makedirs(combined_dir, exist_ok=True)

# times to plot surfaces
t_list = np.linspace(0.0, 1.0, 11) 

# angles
angles = [(30, 35)]

# global limits
gmin, gmax = np.inf, -np.inf
emax = 0.0
for Tk in t_list:
    Up = predict_u(Xg, Yg, Tk)
    Ua = u_true(Xg, Yg, Tk)
    err = np.abs(Up - Ua)
    gmin = min(gmin, Up.min(), Ua.min())
    gmax = max(gmax, Up.max(), Ua.max())
    emax = max(emax, err.max())

zabs = max(abs(gmin), abs(gmax))
zlim = (-zabs, zabs)
vmin, vmax = -zabs, zabs

err_zlim = (0.0, float(emax))
err_vmin, err_vmax = 0.0, float(emax)

def add_surface(ax : Axes3D, X: np.ndarray, Y: np.ndarray, Z: np.ndarray, title: str, elev: float, azim: float,
                vmin: float | None = None, vmax: float | None = None, 
                zlim: tuple | None = None, cmap=cm.viridis):
    '''Plot surface on given Axes3D.'''

    surf = ax.plot_surface(
        X, Y, Z,
        rstride=2, cstride=2,
        linewidth=0, antialiased=True,
        cmap=cmap, vmin=vmin, vmax=vmax,
    )
    ax.set_xlabel("x"); ax.set_ylabel("y")
    ax.set_title(title)
    if zlim is not None:
        ax.set_zlim(*zlim)
    ax.view_init(elev=elev, azim=azim)
    return surf

def render_plot(Tk : float, elev: float, azim: float) -> str:
    '''Plot side-by-side 3D plot at time Tk and given angles.'''

    Up = predict_u(Xg, Yg, Tk)
    Ua = u_true(Xg, Yg, Tk)
    err = np.abs(Up - Ua)

    fig = plt.figure(figsize=(16, 5.5))

    # prediction
    ax1 = fig.add_subplot(1, 3, 1, projection="3d")
    s1 = add_surface(
        ax1, Xg, Yg, Up,
        f"Predicted solution at t={Tk:.2f}",
        elev, azim,
        vmin=vmin, vmax=vmax, zlim=zlim,
        cmap=cm.get_cmap(cmap_field),
    )

    # analytic
    ax2 = fig.add_subplot(1, 3, 2, projection="3d")
    s2 = add_surface(
        ax2, Xg, Yg, Ua,
        f"Analytic solution at t={Tk:.2f}",
        elev, azim,
        vmin=vmin, vmax=vmax, zlim=zlim,
        cmap=cm.get_cmap(cmap_field),
    )

    # error
    ax3 = fig.add_subplot(1, 3, 3, projection="3d")
    s3 = add_surface(
        ax3, Xg, Yg, err,
        f"Absolute error",
        elev, azim,
        vmin=err_vmin, vmax=err_vmax, zlim=err_zlim,
        cmap=cm.get_cmap(cmap_error),
    )

    cbar_main = fig.colorbar(s2, ax=[ax1, ax2], shrink=0.70, pad=0.06)
    cbar_main.set_label("u(x,y,t)")

    cbar_err = fig.colorbar(s3, ax=ax3, shrink=0.70, pad=0.06)
    cbar_err.set_label("error")

    for ax in (ax1, ax2, ax3):
        ax.set_box_aspect((1, 1, 0.5))

    fpath = os.path.join(combined_dir, f"plot_t{Tk:.2f}_e{int(elev)}_a{int(azim)}.png")
    fig.savefig(fpath, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return fpath

# render and save plots
out_paths = []
for Tk in t_list:
    for (elev, azim) in angles:
        out_paths.append(render_plot(Tk, elev, azim))

print(f"Saved {len(out_paths)} side-by-side 3D plots to:\n  {combined_dir}")

# compute errors
errs = []
for tval in times:
    u_pred_t = predict_u(Xg, Yg, tval).astype(np.float32)
    u_true_t = u_true(Xg, Yg, tval).astype(np.float32)
    err_t = rel_l2(u_pred_t, u_true_t)
    errs.append(err_t)
    
    fname = os.path.join(outdir, f"comparison_t{tval:.1f}.png")
    plot_snapshot(
        u_true_t, u_pred_t,
        Xg, Yg,
        t_value=f"{tval:.1f}",
        rel_l2_val=err_t,
        fname=fname,
    )
    print(f"[t={tval:.1f}] rel-L2 = {err_t:.3e} -> {fname}")

curve_path = os.path.join(outdir, "analytical_relL2_vs_time.png")
plot_rel_L2_wave(times, errs, fname=curve_path)
print(f"Saved curve: {curve_path}")

print(f"\nAll figures saved to: {outdir}\n")
