'''
Plot the analytical test solution of the 2D wave equation.
'''


import os
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D 
from matplotlib import cm

Nx = 101
Ny = 101
c = 1.0
two_pi = 2.0 * np.pi

cmap_field = "viridis"

x = np.linspace(0, 1, Nx, endpoint=False)
y = np.linspace(0, 1, Ny, endpoint=False)
X, Y = np.meshgrid(x, y, indexing="xy")


def u_exact(x, y, t):
    return (np.sin(two_pi * x) + np.cos(two_pi * y)) * np.cos(two_pi * c * t)

times = [0.1,0.15,0.2,0.25,0.3,0.35,0.4,0.45,0.5,0.55,0.6,0.65,0.7,0.75,0.8]

out_dir = "analytic_plots/"  
os.makedirs(out_dir, exist_ok=True)


for t in times:
    U = u_exact(X, Y, t)
    zabs = np.max(np.abs(U))
    zlim = (-zabs, zabs)
    vmin, vmax = -zabs, zabs

    fig = plt.figure(figsize=(8, 5.5))
    ax = fig.add_subplot(1, 1, 1, projection="3d")

    surf = ax.plot_surface(
        X, Y, U,
        linewidth=0,
        antialiased=True,
        cmap=cm.get_cmap(cmap_field),
        vmin=vmin,
        vmax=vmax,
    )

    ax.set_title(f"Analytic solution at t = {t:.2f}")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("u(x,y,t)")

    ax.set_zlim(*zlim)
    ax.view_init(elev=30, azim=35)    
    ax.set_box_aspect((1, 1, 0.5))     

    cbar = fig.colorbar(surf, ax=ax, shrink=0.70, pad=0.06)
    cbar.set_label("u(x,y,t)")

    filename = os.path.join(out_dir, f"analytic_wave_3D_t{t:.3f}.png")
    plt.savefig(filename, dpi=300, bbox_inches="tight")
    plt.close(fig)

    print(f"Saved: {filename}")
