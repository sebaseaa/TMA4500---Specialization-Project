'''
Convergence analysis for the FEM solver.
'''

import firedrake as fd
import numpy as np
import matplotlib.pyplot as plt

def solution(mesh: fd.MeshGeometry,tC: fd.Constant, c: float) -> tuple:
    '''True solution.'''

    x, y = fd.SpatialCoordinate(mesh)
    two_pi = 2.0 * fd.pi
    u_star = (fd.sin(two_pi*x) + fd.cos(two_pi*y)) * fd.cos(two_pi*c*tC)
    v_star = -(two_pi*c)*(fd.sin(two_pi*x) + fd.cos(two_pi*y)) * fd.sin(two_pi*c*tC)
    f1 = fd.Constant(0.0)
    f2 = fd.Constant(0.0)

    return u_star, v_star, f1, f2

def solve_wave_equation(c_val: float = 1.0,
                        nx: int = 100, ny: int = 100,
                        t_end: float = 1.0, dt_val: float = 0.001,
                        manufactured=solution) -> tuple:
    '''Solves the first-order wave system with periodic BCs using implicit midpoint.'''

    # mesh, space, params
    mesh = fd.PeriodicRectangleMesh(nx, ny, 1.0, 1.0, direction="both")
    V = fd.FunctionSpace(mesh, "CG", 1)
    W = V * V
    c = fd.Constant(c_val)
    dt = fd.Constant(dt_val)

    # initial conditions
    tC = fd.Constant(0.0)
    u0_expr, v0_expr, f1_expr, f2_expr = manufactured(mesh, tC, c_val)

    u_n = fd.Function(V, name="u_n"); u_n.interpolate(u0_expr)
    v_n = fd.Function(V, name="v_n"); v_n.interpolate(v0_expr)

    # trial, test
    (u_trial, v_trial) = fd.TrialFunctions(W)
    (phi, psi) = fd.TestFunctions(W)

    dx = fd.dx
    a = (
        u_trial * phi * dx
        - 0.5 * dt * v_trial * phi * dx
        + v_trial * psi * dx
        + 0.5 * dt * (c**2) * fd.dot(fd.grad(u_trial), fd.grad(psi)) * dx
    )

    # rhs helper
    def rhs_form(u, v, f1, f2):
        return (
            u * phi * dx
            + 0.5 * dt * v * phi * dx
            + v * psi * dx
            - 0.5 * dt * (c**2) * fd.dot(fd.grad(u), fd.grad(psi)) * dx
            + dt * f1 * phi * dx
            + dt * f2 * psi * dx
        )

    w_np1 = fd.Function(W, name="w_np1")

    def energy(u : fd.Function, v: fd.Function) -> float:
        '''Computes the energy of the wave system at current time.'''
        return 0.5 * (
            fd.assemble(v * v * dx)
            + (c_val**2) * fd.assemble(fd.dot(fd.grad(u), fd.grad(u)) * dx)
        )

    energies = []
    times = []

    t = 0.0
    energies.append(energy(u_n, v_n))
    times.append(t)

    # solver loop
    while t < t_end - 1e-15:
        t_mid = t + 0.5 * dt_val
        tC.assign(t_mid)
        _, _, f1_expr, f2_expr = manufactured(mesh, tC, c_val)

        L = rhs_form(u_n, v_n, f1_expr, f2_expr)
        problem = fd.LinearVariationalProblem(a, L, w_np1)
        solver = fd.LinearVariationalSolver(problem)

        solver.solve()
        u_np1, v_np1 = w_np1.subfunctions
        u_n.assign(u_np1)
        v_n.assign(v_np1)

        t += dt_val
        times.append(t)
        E = energy(u_n, v_n)
        energies.append(E)

    return u_n, v_n, times, energies, mesh, V


def convergence_study_spacelist(
    nx_list=(10, 20, 40, 80, 160),
    c: float = 1.0,
    t_end: float = 1.0,
    dt_fixed: float = 0.001,
    manufactured=solution,
) -> tuple:
    '''Spatial convergence study. Returns hs, errors, and observed rate p.'''

    errs, hs = [], []

    for nx in nx_list:
        ny = nx
        h = 1.0 / nx
        hs.append(h)

        # solve
        u_h, v_h, times, energies, mesh, V = solve_wave_equation(
            c_val=c,
            nx=nx, ny=ny,
            t_end=t_end,
            dt_val=dt_fixed,
            manufactured=manufactured,
        )

        # exact solution at final time
        tC = fd.Constant(t_end)
        u_exact_expr, _, _, _ = manufactured(mesh, tC, c)
        u_exact = fd.Function(V)
        u_exact.interpolate(u_exact_expr)

        # L2 error
        err = fd.errornorm(u_exact, u_h, norm_type="L2")
        errs.append(err)

        print(f"h = {h:.3e},  L2 error = {err:.6e}")

    # fit slope
    hs_arr = np.array(hs)
    errs_arr = np.array(errs)
    p = np.polyfit(np.log(hs_arr), np.log(errs_arr), 1)[0]
    print(f"\n Observed spatial convergence rate p ≈ {p:.2f}")

    return hs_arr, errs_arr, p


def convergence_study_timelist(
    dt_values=(0.1, 0.05, 0.025, 0.0125, 0.00625),
    nx: int = 1000,
    c: float = 1.0,
    t_end: float = 1.0,
    manufactured=solution,
):
    '''Temporal convergence study. Returns dts, errors, and observed rate p.'''

    errors = []
    dts = []

    for dt_val in dt_values:
        u_h, v_h, times, energies, mesh, V = solve_wave_equation(
            c_val=c,
            nx=nx, ny=nx,
            t_end=t_end,
            dt_val=dt_val,
            manufactured=manufactured,
        )

        tC_exact = fd.Constant(t_end)
        u_exact_expr, _, _, _ = manufactured(mesh, tC_exact, c)
        u_exact = fd.Function(V, name="u_exact")
        u_exact.interpolate(u_exact_expr)

        err = fd.errornorm(u_exact, u_h, norm_type="L2")
        errors.append(err)
        dts.append(dt_val)

        print(f"dt = {dt_val:.3e},  L2 error = {err:.6e}")

    dts_arr = np.array(dts)
    errs_arr = np.array(errors)
    p = np.polyfit(np.log(dts_arr), np.log(errs_arr), 1)[0]
    print(f"\n Observed temporal convergence rate p ≈ {p:.2f}")

    return dts_arr, errs_arr, p


def combined_convergence_plot(
    nx_list=(10, 20, 40, 80, 160),
    dt_values=(0.1, 0.05, 0.025, 0.0125),
    c: float = 1.0,
    t_end: float = 1.0,
    dt_fixed_space: float = 0.001,
    nx_time: int = 1000,
    manufactured_space=solution,
    manufactured_time=solution,
    filename: str = "FEM_convergence_space_time.png",
) -> None:
    '''Run both convergence studies and plot them on the same figure.'''

    hs, errs_space, p_space = convergence_study_spacelist(
        nx_list=nx_list,
        c=c,
        t_end=t_end,
        dt_fixed=dt_fixed_space,
        manufactured=manufactured_space,
    )

    dts, errs_time, p_time = convergence_study_timelist(
        dt_values=dt_values,
        nx=nx_time,
        c=c,
        t_end=t_end,
        manufactured=manufactured_time,
    )

    plt.figure()

    plt.loglog(hs, errs_space, "-o", label=f"Spatial (p ≈ {p_space:.2f})")
    plt.loglog(dts, errs_time, "-s", label=f"Temporal (p ≈ {p_time:.2f})")

    xmin = min(hs.min(), dts.min())
    ymin = min(errs_space.min(), errs_time.min())
    ref_steps = np.array([xmin, max(hs.max(), dts.max())])
    ref_errors = ymin * (ref_steps / xmin)**2  # slope 2

    plt.loglog(ref_steps, ref_errors, "k--", label="Reference slope = 2")
    plt.gca().invert_xaxis()
    plt.grid(True, which="both", ls=":")
    plt.xlabel(r"Step size")
    plt.ylabel(r"$L^2$ error")
    plt.title("Spatial and temporal convergence")
    plt.legend()
    plt.tight_layout()
    plt.savefig(filename, dpi=200)
    plt.show()


combined_convergence_plot(
    nx_list=(10, 20, 40, 80),
    dt_values=(0.1, 0.05, 0.025, 0.0125),
    c=1.0,
    t_end=0.3,
    dt_fixed_space=0.001,
    nx_time=500,
    manufactured_space=solution,
    manufactured_time=solution,
    filename="FEM_convergence_space_time.png",
)
