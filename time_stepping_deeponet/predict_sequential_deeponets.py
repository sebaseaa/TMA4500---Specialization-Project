'''
Predict the solution to the 2D wave equation using a sequence of DeepONet models.
'''

import os
import numpy as np
import matplotlib.pyplot as plt
from predicting_utils import *

# paths for models and directories
this_dir = os.path.dirname(os.path.abspath(__file__))
models_dir = this_dir
out_dir = os.path.join(this_dir, "time_stepping_results")
os.makedirs(out_dir, exist_ok=True)

# build grids and initial condition
sensors = build_sensors(Nx_s, Ny_s)
Xg, Yg, XY = build_viz_grid(Nx_viz)
t0 = 0.0
branch_current = make_branch_from_ic(sensors, t0=t0)

# models
model_files = [
    "split_branch_00_to_01-15000.pt",
    "split_branch_01_to_02-15000.pt",
    "split_branch_02_to_03-15000.pt",
    "split_branch_03_to_04-15000.pt",
    "split_branch_04_to_05-15000.pt",
    "split_branch_05_to_06-15000.pt",
    "split_branch_06_to_07-15000.pt",
    "split_branch_07_to_08-15000.pt",
    "split_branch_08_to_09-15000.pt",
    "split_branch_09_to_10-15000.pt",
]

model_paths = [os.path.join(models_dir, f) for f in model_files]


# rollout
times = []
errs_u, errs_v = [], []
u_grids, v_grids = [], []

accum_time = 0.0

for k, model_path in enumerate(model_paths):
    fname = os.path.basename(model_path)

    accum_time += dt_step
    t_val = accum_time

    print(f"Loading model {k}  file={fname}")
    model = restore_model(model_path, M)

    # predict on sensors 
    u_sens_pred, v_sens_pred = predict_on_sensors(model, branch_current, sensors)
    branch_current = branch_from_uv_on_sensors(u_sens_pred, v_sens_pred)

    # predict on visualisation grid
    u_grid_pred, v_grid_pred = predict_on_points(model, branch_current, XY, side=Nx_viz)

    u_grids.append(u_grid_pred)
    v_grids.append(v_grid_pred)
    times.append(t_val)

    # analytic reference
    IC_grid = u_spatial_part_xy(Xg, Yg)
    u_true = IC_grid * np.cos(omega * t_val)
    v_true = -omega * IC_grid * np.sin(omega * t_val)

    # errors
    ru = rel_l2(u_grid_pred, u_true)
    rv = rel_l2(v_grid_pred, v_true)
    errs_u.append(ru)
    errs_v.append(rv)

    print(f"[t={t_val:.2f}] rel-L2(u)={ru:.3e}  rel-L2(v)={rv:.3e}")

    # plotting
    save_3d(
        Xg, Yg, u_grid_pred, u_true,
        t_value=t_val, rel_L2_u=ru,
        out_path=os.path.join(out_dir, f"plot_u_t{t_val:.3f}.png"),
        elev=30, azim=35,
    )

    save_3d(
        Xg, Yg, v_grid_pred, v_true,
        t_value=t_val, rel_L2_u=rv,
        out_path=os.path.join(out_dir, f"plot_v_t{t_val:.3f}.png"),
        elev=30, azim=35,
    )


plot_rel_L2_wave(times, errs_u, fname=os.path.join(out_dir, "relL2_vs_time_DeepONet.png"))
print(f"\n Outputs saved in: {out_dir}")
