'''
Solve the 2D wave equation with periodic boundary conditions using FEM,
with initial conditions sampled from a periodic Gaussian random field (GRF).
'''

import firedrake as fd
import numpy as np
import os
from GRF_for_FEM import PeriodicGRF2D

# slightly modified from FEM/solver.py, to allow for GRF initial conditions
def solve_wave_equation(
    c_val: float = 1.0,
    nx: int = 100,
    ny: int = 100,
    t_end: float = 1.0,
    dt_val: float = 0.001,
    u0_callable=None,
    v0_callable=None,
) -> tuple:
    '''Solve the 2D wave equation with periodic BCs and initial conditions from a GRF.'''

    # grid, function space, params
    mesh = fd.PeriodicRectangleMesh(nx, ny, 1.0, 1.0, direction="both")
    V = fd.FunctionSpace(mesh, "CG", 1)
    W = V * V
    c = fd.Constant(c_val)
    dt = fd.Constant(dt_val)

    x_sym, y_sym = fd.SpatialCoordinate(mesh)

    # coordinates of CG1 dofs
    ux = fd.Function(V).interpolate(x_sym)
    uy = fd.Function(V).interpolate(y_sym)
    x_dofs = ux.dat.data_ro.copy()
    y_dofs = uy.dat.data_ro.copy()

    # u initial
    u_n = fd.Function(V, name="u_n")
    if u0_callable is None:
        u0_expr = fd.sin(2.0 * fd.pi * x_sym) + fd.cos(2.0 * fd.pi * y_sym)
        u_n.interpolate(u0_expr)
    else:
        vals_u = u0_callable(x_dofs, y_dofs)
        vals_u = np.asarray(vals_u, dtype=float).reshape(-1)
        u_n.dat.data[:] = vals_u

    # v initial
    v_n = fd.Function(V, name="v_n")
    if v0_callable is None:
        v_n.interpolate(fd.Constant(0.0))
    else:
        vals_v = v0_callable(x_dofs, y_dofs)
        vals_v = np.asarray(vals_v, dtype=float).reshape(-1)
        v_n.dat.data[:] = vals_v

    (u_trial, v_trial) = fd.TrialFunctions(W)
    (phi, psi) = fd.TestFunctions(W)

    a = (
        u_trial * phi * fd.dx
        - 0.5 * dt * v_trial * phi * fd.dx
        + v_trial * psi * fd.dx
        + 0.5 * dt * (c**2) * fd.dot(fd.grad(u_trial), fd.grad(psi)) * fd.dx
    )

    # rhs helper function
    def rhs(u, v):
        return (
            u * phi * fd.dx
            + 0.5 * dt * v * phi * fd.dx
            + v * psi * fd.dx
            - 0.5 * dt * (c**2) * fd.dot(fd.grad(u), fd.grad(psi)) * fd.dx
        )

    w_np1 = fd.Function(W, name="w_np1")
    problem = fd.LinearVariationalProblem(a, rhs(u_n, v_n), w_np1)
    solver = fd.LinearVariationalSolver(problem)

    def energy(u : fd.Function, v: fd.Function) -> float:
        '''Computes the energy of the wave system at current time.'''   

        return 0.5 * (
            fd.assemble(v * v * fd.dx)
            + (c_val**2) * fd.assemble(fd.dot(fd.grad(u), fd.grad(u)) * fd.dx)
        )

    energies = [energy(u_n, v_n)]
    times = [0.0]
    t = 0.0

    snapshots_u = [u_n.copy(deepcopy=True)]
    snapshots_v = [v_n.copy(deepcopy=True)]

    # solver loop
    while t < t_end - 1e-15:
        solver.solve()
        u_np1, v_np1 = w_np1.subfunctions
        u_n.assign(u_np1)
        v_n.assign(v_np1)

        problem = fd.LinearVariationalProblem(a, rhs(u_n, v_n), w_np1)
        solver = fd.LinearVariationalSolver(problem)

        t += dt_val
        times.append(t)
        energies.append(energy(u_n, v_n))
        snapshots_u.append(u_n.copy(deepcopy=True))
        snapshots_v.append(v_n.copy(deepcopy=True))

    return snapshots_u, snapshots_v, times, energies


def sample_to_regular_grid(u_fem : fd.Function, nx_eval : int =64, ny_eval : int =64) -> tuple:
    '''Sample a Firedrake function u_fem to a regular grid of size (nx_eval, ny_eval) on [0,1]^2.'''

    xs = np.linspace(0.0, 1.0, nx_eval, endpoint=False)
    ys = np.linspace(0.0, 1.0, ny_eval, endpoint=False)
    Xg, Yg = np.meshgrid(xs, ys, indexing="xy")
    pts = [(float(x), float(y)) for x, y in zip(Xg.ravel(), Yg.ravel())]
    Uvals = np.asarray(u_fem.at(pts, dont_raise=True)).reshape(ny_eval, nx_eval)
    return xs, ys, Uvals


out_dir = "fem_grf_reference"
os.makedirs(out_dir, exist_ok=True)
length_scales = [1.0, 7.0, 12.0]

nx_fem = ny_fem = 100
nx_eval = ny_eval = 64
dt_val = 0.001
t_end = 1.0
times_out = np.linspace(0.0, 1.0, 11)  

for i_ls, ell in enumerate(length_scales):
    print(f"\n Length scale l = {ell}")

    # make GRF and fix seed so its reproducible
    np.random.seed(1234 + i_ls)
    func_space = PeriodicGRF2D(length_scale=ell, N=100)

    features = func_space.random(size=1)  

    # GRF IC at CG1 dofs
    def u0_grf_callable(x_arr, y_arr):
        pts = np.stack([x_arr, y_arr], axis=1)
        vals = func_space.eval_batch(features, pts)  
        return vals[0]

    snapshots_u, snapshots_v, times_fem, energies = solve_wave_equation(
        c_val=1.0,
        nx=nx_fem,
        ny=ny_fem,
        t_end=t_end,
        dt_val=dt_val,
        u0_callable=u0_grf_callable,
        v0_callable=None,
    )

    # sample at desired times on regular grid
    u_list = []
    for t_target in times_out:
        idx = int(round(t_target / dt_val))  
        u_func = snapshots_u[idx]
        x_eval_1d, y_eval_1d, U = sample_to_regular_grid(
            u_func, nx_eval=nx_eval, ny_eval=ny_eval
        )
        u_list.append(U)

    u_all = np.stack(u_list, axis=0)  

    # save everything needed for DeepONet
    fname = os.path.join(out_dir, f"fem_reference_ell{int(ell)}.npz")
    np.savez(
        fname,
        length_scale=ell,
        N_grf=func_space.N,
        times=times_out,
        x_eval_1d=x_eval_1d,
        y_eval_1d=y_eval_1d,
        u_all=u_all,
        features=features,
    )
    print(f"Saved reference data to {fname}")
