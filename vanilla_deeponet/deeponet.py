'''
Solve 2D wave equation using DeepONet.
'''

import deepxde as dde
import matplotlib.pyplot as plt
import numpy as np
import torch
import os
from modified_GRF_classes import *

dde.backend.set_default_backend("pytorch")
torch.set_default_dtype(torch.float32)

if torch.backends.mps.is_available():
    os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"
    device = torch.device("mps")
else:
    device = torch.device("cpu")
print(f"Using device: {device}")

# wave speed and domain
c = 1.0  
geom_xy = dde.geometry.Rectangle([0.0, 0.0], [1, 1])
timedomain = dde.geometry.TimeDomain(0.0, 1)
geomtime = dde.geometry.GeometryXTime(geom_xy, timedomain)
sin, cos, concat = torch.sin, torch.cos, torch.cat
dim_x = 3

# pde
def pde(x, u, v):
    u_xx = dde.grad.hessian(u, x, i=0, j=0)
    u_yy = dde.grad.hessian(u, x, i=1, j=1)
    u_tt = dde.grad.hessian(u, x, i=2, j=2)
    return u_tt - (c**2) * (u_xx + u_yy)


# initial conditions velocity
def ic_ut(x, u, v):
    u_t = dde.grad.jacobian(u, x, j=2)
    return u_t

def on_initial(X, _on_b):
    x, y, t = X
    return np.isclose(t, 0.0, atol=1e-6)

def on_initial_interior(X, on_b):
    x, y, t = X
    return (np.isclose(t, 0.0, atol=1e-6)) and (not on_b)

ic_u = dde.icbc.DirichletBC(
    geomtime,
    lambda X, aux: aux,   
    on_initial_interior    
)
ic_ut0 = dde.icbc.OperatorBC(geomtime, ic_ut, on_initial)

# bc
def on_x0(X, on_b):
    if not on_b:
        return False
    x, y, t = X
    return np.isclose(x, 0.0)

def on_y0(X, on_b):
    if not on_b:
        return False
    x, y, t = X
    return np.isclose(y, 0.0)

# periodic in x,y
bc_per_x0 = dde.icbc.PeriodicBC(geomtime, component_x=0, on_boundary=on_x0, derivative_order=0)
bc_per_y0 = dde.icbc.PeriodicBC(geomtime, component_x=1, on_boundary=on_y0, derivative_order=0)

# make pde data
pde_data = dde.data.TimePDE(
    geomtime,
    pde,
    [ic_u, ic_ut0, bc_per_x0, bc_per_y0],
    num_domain=200,
    num_boundary=100, 
    num_initial=200 
)

# random ICs 
func_space = PeriodicGRF2D(kernel="ExpSineSquared", length_scale=7)

# sensors
Nx_s, Ny_s = 20, 20
xs = np.linspace(0, 1, Nx_s)
ys = np.linspace(0, 1, Ny_s)
Xs, Ys = np.meshgrid(xs, ys, indexing="xy")
sensors = np.vstack((np.ravel(Xs), np.ravel(Ys))).T

# data in cartesian product form
data = dde.data.PDEOperatorCartesianProd(
    pde_data,
    func_space,
    sensors, 
    1000, 
    function_variables=[0, 1],  
    num_test=100,
    batch_size=128,
)

# define branch and trunk
net = dde.nn.DeepONetCartesianProd(
    [sensors.shape[0], 128, 128, 128], # branch
    [dim_x, 128, 128, 128],            # trunk
    "tanh",
    "Glorot normal",
)

# model
net.to(device)
model = dde.Model(data, net)
loss_weights = [1, 1.5, 1, 1, 1]
model.compile("adam", lr=1e-3, loss_weights=loss_weights)
losshistory, train_state = model.train(iterations=400, display_every=50)
dde.utils.plot_loss_history(losshistory)
dde.optimizers.LBFGS_options["iter_per_step"] = 100
dde.optimizers.set_LBFGS_options(
    maxiter=15000
)
model.compile("L-BFGS")
losshistory, train_state = model.train()
dde.utils.plot_loss_history(losshistory)
model.save("vanilladeeponet")
