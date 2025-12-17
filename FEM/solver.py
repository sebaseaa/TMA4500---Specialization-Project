'''
Solves the 2D wave equation using FEM.
'''

import firedrake as fd
from plotting_utils import *


def solve_wave_equation(c_val: float = 1,
                        nx: int = 100, ny: int = 100,
                        t_end: float = 1, dt_val: float = 0.001) -> tuple:
    '''Solves the wave equation using FEM with periodic BCs.'''

    # mesh, space, params
    mesh = fd.PeriodicRectangleMesh(nx, ny, 1.0, 1.0, direction="both")
    V = fd.FunctionSpace(mesh, "CG", 1)
    W = V * V
    c  = fd.Constant(c_val)
    dt = fd.Constant(dt_val)

    # ICs
    x, y = fd.SpatialCoordinate(mesh)
    u0_expr = fd.sin(2.0*fd.pi*x) + fd.cos(2.0*fd.pi*y)
    v0_expr = fd.Constant(0.0)

    u_n = fd.Function(V, name="u_n")
    u_n.interpolate(u0_expr)
    v_n = fd.Function(V, name="v_n")
    v_n.interpolate(v0_expr)

    # trial, test
    (u_trial, v_trial) = fd.TrialFunctions(W)
    (phi, psi) = fd.TestFunctions(W)

    # linear and bilinear form
    a = (
        u_trial*phi*fd.dx
        - 0.5*dt*v_trial*phi*fd.dx
        + v_trial*psi*fd.dx
        + 0.5*dt*(c**2)*fd.dot(fd.grad(u_trial), fd.grad(psi))*fd.dx
    )

    L = (
        u_n*phi*fd.dx
        + 0.5*dt*v_n*phi*fd.dx
        + v_n*psi*fd.dx
        - 0.5*dt*(c**2)*fd.dot(fd.grad(u_n), fd.grad(psi))*fd.dx
    )

    w_np1 = fd.Function(W, name="w_np1")
    problem = fd.LinearVariationalProblem(a, L, w_np1)
    solver  = fd.LinearVariationalSolver(problem)

    def energy(u, v) -> float:
        '''Computes the energy of the wave system at current time.'''
        return 0.5*(fd.assemble(v*v*fd.dx)
                    + (c_val**2)*fd.assemble(fd.dot(fd.grad(u), fd.grad(u))*fd.dx))

    energies = [energy(u_n, v_n)]
    times = [0.0]
    t = 0.0

    while t < t_end - 1e-15:
        solver.solve()
        u_np1, v_np1 = w_np1.subfunctions
        u_n.assign(u_np1); v_n.assign(v_np1)

        # rebuild RHS for next step
        L = (
            u_n*phi*fd.dx
            + 0.5*dt*v_n*phi*fd.dx
            + v_n*psi*fd.dx
            - 0.5*dt*(c**2)*fd.dot(fd.grad(u_n), fd.grad(psi))*fd.dx
        )
        solver.problem = fd.LinearVariationalProblem(a, L, w_np1)

        t += dt_val
        times.append(t)
        energies.append(energy(u_n, v_n))

    return u_n, v_n, times, energies


def rel_L2_error(u_h: fd.Function, t: float, c_val: float) -> float:
    '''Computes the relative L2 error of u_h at time t.'''

    V = u_h.function_space()
    mesh = V.mesh()
    x, y = fd.SpatialCoordinate(mesh)
    two_pi = 2.0 * fd.pi
    u_true_expr = (fd.sin(two_pi * x) + fd.cos(two_pi * y)) * fd.cos(two_pi * c_val * t)
    num = fd.sqrt(fd.assemble((u_h - u_true_expr)**2 * fd.dx))
    den = fd.sqrt(fd.assemble(u_true_expr**2 * fd.dx))

    return float(num / den)


def rel_L2_over_times_single_run(
    c_val: float = 1.0,
    nx: int = 32,
    ny: int = 32,
    dt_val: float = 0.001,
    t_end: float = 1.0,
    targets=[0.1*k for k in range(int(round(1/0.1))+1)],
):
    '''Compute relative L2 error at specified times in a single run.'''

    targets = sorted(set(targets))
    tol = 0.5*dt_val + 1e-14
    next_idx = 0

    # set up 
    mesh = fd.PeriodicRectangleMesh(nx, ny, 1.0, 1.0, direction="both")
    V = fd.FunctionSpace(mesh, "CG", 1)
    W = V*V
    c = fd.Constant(c_val)
    dt = fd.Constant(dt_val)
    x, y = fd.SpatialCoordinate(mesh)
    u0_expr = fd.sin(2.0*fd.pi*x) + fd.cos(2.0*fd.pi*y)
    u_n = fd.Function(V)
    u_n.interpolate(u0_expr)
    v_n = fd.Function(V)
    v_n.interpolate(fd.Constant(0.0))

    (uT, vT) = fd.TrialFunctions(W)
    (phi, psi) = fd.TestFunctions(W)
    a = (uT*phi - 0.5*dt*vT*phi + vT*psi
         + 0.5*dt*(c**2)*fd.dot(fd.grad(uT), fd.grad(psi)))*fd.dx

    def rhs(u, v):
        return (u*phi + 0.5*dt*v*phi + v*psi
                - 0.5*dt*(c**2)*fd.dot(fd.grad(u), fd.grad(psi)))*fd.dx

    w_np1 = fd.Function(W)
    problem = fd.LinearVariationalProblem(a, rhs(u_n, v_n), w_np1)
    solver = fd.LinearVariationalSolver(problem)

    sampled_t, rel_errors = [], []

    # include t=0 
    if any(abs(0.0 - tt) <= tol for tt in targets):
        sampled_t.append(0.0)
        rel_errors.append(rel_L2_error(u_n, 0.0, c_val))
        next_idx = 1 

    t = 0.0
    while t < t_end - 1e-15:
        solver.solve()
        u_np1, v_np1 = w_np1.subfunctions
        u_n.assign(u_np1); v_n.assign(v_np1)

        # rebuild RHS for next step
        problem = fd.LinearVariationalProblem(a, rhs(u_n, v_n), w_np1)
        solver = fd.LinearVariationalSolver(problem)

        t += dt_val

        while next_idx < len(targets) and abs(t - targets[next_idx]) <= tol:
            tt = targets[next_idx]
            sampled_t.append(tt)
            rel_errors.append(rel_L2_error(u_n, tt, c_val))
            next_idx += 1

    return sampled_t, rel_errors


u_n, v_n, times, energies = solve_wave_equation(
    c_val=1,
    dt_val=0.001,
    t_end=1.0,
    nx=100,
    ny=100,
)

plot_energy(times, energies)
plot_solution_3d(u_n)
sampled_t, rel_errors = rel_L2_over_times_single_run(
    c_val=1.0, nx=100, ny=100, dt_val=0.001, t_end=1.0,
)
plot_rel_L2(sampled_t, rel_errors)
