'''
Plotting utilites for the FEM solver.
'''

import firedrake as fd
import matplotlib.pyplot as plt
from firedrake.pyplot import tripcolor
from mpl_toolkits.mplot3d import Axes3D  
import numpy as np
from matplotlib import cm


cmap_field = "viridis"
cmap_error = "inferno"


def plot_energy(times: list[float], energies: list[float]) -> None:
    '''Plot discrete energy vs time (same size as convergence plots).'''

    plt.figure(figsize=(6.4, 4.8))
    plt.plot(times, energies, "-o", markersize=3)
    plt.xlabel("Time")
    plt.ylabel("Energy")
    plt.title("Energy conservation vs time")
    plt.grid(True, alpha=0.6, linestyle="--")
    plt.tight_layout()
    plt.savefig("FEM_energy.png", dpi=200)
    plt.show()

def plot_solution_3d(
    u_n: fd.Function,
    t: float = 1.0,
    c: float = 1.0,
    ngrid: int | None = None,
    fname: str = "FEM_wave_3D.png",
) -> None:
    '''Plot FEM solution vs true solution and error in 3D.'''

    V = u_n.function_space()
    mesh = V.mesh()

    if ngrid is None:
        ngrid = 50
    nx = ny = int(ngrid)

    xs = np.linspace(0.0, 1.0, nx, endpoint=False)
    ys = np.linspace(0.0, 1.0, ny, endpoint=False)
    Xg, Yg = np.meshgrid(xs, ys, indexing="xy")
    pts_list = [(float(x), float(y)) for x, y in zip(Xg.ravel(), Yg.ravel())]
    Uz = np.asarray(u_n.at(pts_list, dont_raise=False)).reshape(ny, nx)

    two_pi = 2.0 * np.pi
    u_true = (np.sin(two_pi * Xg) + np.cos(two_pi * Yg)) * np.cos(two_pi * c * t)
    u_err = np.abs(Uz - u_true)

    gmin = min(Uz.min(), u_true.min())
    gmax = max(Uz.max(), u_true.max())
    zabs = max(abs(gmin), abs(gmax))
    zlim = (-zabs, zabs)
    vmin, vmax = -zabs, zabs

    err_max = float(u_err.max())
    err_zlim = (0.0, err_max)

    fig = plt.figure(figsize=(16, 5.5))

    ax1 = fig.add_subplot(1, 3, 1, projection="3d")
    s1 = ax1.plot_surface(
        Xg, Yg, Uz,
        linewidth=0, antialiased=True,
        cmap=cm.get_cmap(cmap_field),
        vmin=vmin, vmax=vmax,
    )
    ax1.set_title(f"FEM solution at t={t:.2f}")
    ax1.set_xlabel("x"); ax1.set_ylabel("y")
    ax1.set_zlim(*zlim)
    ax1.view_init(elev=30, azim=35)

    ax2 = fig.add_subplot(1, 3, 2, projection="3d")
    s2 = ax2.plot_surface(
        Xg, Yg, u_true,
        linewidth=0, antialiased=True,
        cmap=cm.get_cmap(cmap_field),
        vmin=vmin, vmax=vmax,
    )
    ax2.set_title(f"Analytic solution at t={t:.2f}")
    ax2.set_xlabel("x"); ax2.set_ylabel("y")
    ax2.set_zlim(*zlim)
    ax2.view_init(elev=30, azim=35)

    cbar_main = fig.colorbar(s1, ax=[ax1, ax2], shrink=0.70, pad=0.06)
    cbar_main.set_label("u(x,y,t)")

    ax3 = fig.add_subplot(1, 3, 3, projection="3d")
    s3 = ax3.plot_surface(
        Xg, Yg, u_err,
        linewidth=0, antialiased=True,
        cmap=cm.get_cmap(cmap_error),
        vmin=0.0, vmax=err_max,
    )
    ax3.set_title("Absolute error")
    ax3.set_xlabel("x"); ax3.set_ylabel("y")
    ax3.set_zlim(*err_zlim)
    ax3.view_init(elev=30, azim=35)

    cbar_err = fig.colorbar(s3, ax=ax3, shrink=0.70, pad=0.06)
    cbar_err.set_label("Error")

    for ax in (ax1, ax2, ax3):
        ax.set_box_aspect((1, 1, 0.5))

    plt.savefig(fname, dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_rel_L2(sampled_t: list[float], rel_errors: list[float],
                fname: str = "relL2_vs_time.png") -> None:
    '''Plot relative L2 error vs time.'''

    plt.figure(figsize=(6.5, 4))
    plt.plot(sampled_t, rel_errors, "-o", markersize=4)
    plt.xlabel("Time")
    plt.ylabel(r"Relative $L^2$ error")
    plt.title(r"Relative $L^2$ error vs time (FEM)")
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.tight_layout()
    plt.savefig(fname, dpi=300)
    plt.show()

