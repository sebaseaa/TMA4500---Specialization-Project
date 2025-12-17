'''
Produce initial condition plots for different length scales of periodic GRFs.
'''

from modified_GRF_classes import PeriodicGRF2D
import numpy as np
import matplotlib.pyplot as plt

length_scales = [7, 5, 3, 1, 0.5]
titles = [fr"Periodic kernel, l = {ls}" for ls in length_scales]
cmap = "jet"      
Nx_eval = Ny_eval = 128 

xe = np.linspace(0.0, 1.0, Nx_eval)
ye = np.linspace(0.0, 1.0, Ny_eval)
Xe, Ye = np.meshgrid(xe, ye, indexing="xy")
xy_grid = np.stack([Xe.ravel(), Ye.ravel()], axis=1).astype(np.float32)

for ls, title in zip(length_scales, titles):
    fs = PeriodicGRF2D(kernel="ExpSineSquared", length_scale=ls)
    feat = fs.random(size=1)  
    vals = fs.eval_batch(feat, xy_grid)
    field = np.array(vals).reshape(Ny_eval, Nx_eval)

    plt.figure(figsize=(5, 4), dpi=150)
    im = plt.pcolormesh(Xe, Ye, field, shading="auto",
                        cmap=cmap, rasterized=True)
    plt.title(title)
    plt.axis("off")
    cbar = plt.colorbar(im)
    cbar.set_label("u(x,y)")
    plt.tight_layout()
    plt.show()
