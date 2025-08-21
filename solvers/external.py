import numpy as np
import cvxpy as cp
import scipy.sparse as spr

import sys

from mosek.fusion import (
    Model,
    Domain,
    Expr,
    Matrix,
    ObjectiveSense,
    SolutionStatus,
)

import clarabel


"""
min <C, X>
X
s.t. A(X) = b            →      <A_i, X> = b_i, i = 1, ..., m
        X ⪰ 0
"""


def sdp_cvx(C, A, b, solver, verbose):

    n = C.shape[0]
    m = len(A)

    # Define the optimization variable X.
    X = cp.Variable((n, n), symmetric=True)

    # Define the objective function: min <C, X>
    objective = cp.Minimize(cp.trace(C @ X))

    # Define the constraints
    linear_constr = [cp.trace(A[i] @ X) == b[i] for i in range(m)]
    psd_constr = [X >> 0]
    constraints = linear_constr + psd_constr

    # Formulate the problem
    problem = cp.Problem(objective, constraints)

    # Solve the problem
    print("Solving cvx...")
    problem.solve(solver=solver, verbose=verbose)  # Using SCS solver explicitly
    print(f" - Problem Status: {problem.status}")

    if problem.status == cp.OPTIMAL or problem.status == cp.OPTIMAL_INACCURATE:
        # Print the optimal value
        print(f" - Optimal Value: {problem.value:.4f}")

        # Retrieve the optimal solution for X
        X_optimal = X.value
        S_optimal = psd_constr[0].dual_value
        y_optimal = np.array([c.dual_value for c in linear_constr])
        f_optimal = problem.value

    else:
        raise ValueError(f"Problem could not be solved. Status: {problem.status}")

    return X_optimal, S_optimal, y_optimal, f_optimal


def sdp_mosek(C, A, b, verbose):

    n = C.shape[0]
    m = len(A)

    print("Setting up Mosek problem...")
    with Model("PoP SDP") as M:
        if verbose:
            M.setSolverParam("log", 1)
            M.setLogHandler(sys.stdout)
        # Define the optimization variable X.
        X = M.variable("X", [n, n], Domain.inPSDCone())

        # Define the objective function: min <C, X>
        M.objective(ObjectiveSense.Minimize, Expr.dot(C, X))

        # Define the constraints
        constraints = []
        for i in range(m):
            constraints.append(M.constraint(Expr.dot(A[i], X), Domain.equalsTo(b[i])))

        # Solve the problem
        print("Solving Mosek problem...")
        M.setSolverParam("intpntMaxIterations", 100)
        M.solve()
        status = M.getPrimalSolutionStatus()
        if status == SolutionStatus.Optimal:
            print("Solution is optimal")
        else:
            raise ValueError(f"Problem could not be solved. Status: {status}")

        X_optimal = X.level()
        S_optimal = X.dual()
        y_optimal = np.array([c.level() for c in constraints])
        f_optimal = M.primalObjValue()

        return X_optimal, S_optimal, y_optimal, f_optimal


def sdp_clarabel(C, A, b, verbose):

    n = C.shape[0]
    m = len(A)

    # Dimension of the vectorized symmetric matrix using triangle storage
    nvec = n * (n + 1) // 2
    sqrt2 = np.sqrt(2.0)

    # Helpers for UPPER-triangular column-stacked order (i <= j, column-major)
    # Vector order: (0,0), (0,1), (1,1), (0,2), (1,2), (2,2), ...
    # - svec_from_matrix: scaled (sqrt(2) on off-diagonals)
    # - tri_from_matrix: unscaled raw upper-triangle entries
    # - smat_from_tri: build symmetric from unscaled upper-triangle entries
    # - smat_from_svec: build symmetric from svec-scaled entries
    def svec_from_matrix(M: np.ndarray) -> np.ndarray:
        v = np.empty(nvec, dtype=float)
        k = 0
        for j in range(n):
            for i in range(0, j + 1):
                if i == j:
                    v[k] = M[i, j]
                else:
                    v[k] = sqrt2 * M[i, j]
                k += 1
        return v

    def tri_from_matrix(M: np.ndarray) -> np.ndarray:
        v = np.empty(nvec, dtype=float)
        k = 0
        for j in range(n):
            for i in range(0, j + 1):
                v[k] = M[i, j]
                k += 1
        return v

    def smat_from_tri(v: np.ndarray) -> np.ndarray:
        X = np.zeros((n, n), dtype=float)
        k = 0
        for j in range(n):
            for i in range(0, j + 1):
                if i == j:
                    X[i, j] = v[k]
                else:
                    val = v[k]
                    X[i, j] = val
                    X[j, i] = val
                k += 1
        return X

    def smat_from_svec(v: np.ndarray) -> np.ndarray:
        X = np.zeros((n, n), dtype=float)
        k = 0
        for j in range(n):
            for i in range(0, j + 1):
                if i == j:
                    X[i, j] = v[k]
                else:
                    val = v[k] / sqrt2
                    X[i, j] = val
                    X[j, i] = val
                k += 1
        return X

    # Clarabel problem data: minimize q^T x, subject to A_clar x + s = b_clar, s in K
    # Enforce x in PSDTriangle via first block rows: -I * x + s_psd = 0  => s_psd = x ∈ PSD
    # Enforce linear equalities via ZeroCone: Aeq * x + s_zero = b, s_zero ∈ {0}
    P = spr.csc_matrix((nvec, nvec))

    # Use unscaled triangle vector x; objective vector must weight off-diagonals by 2
    q = np.empty(nvec, dtype=float)
    k = 0
    for j in range(n):
        for i in range(0, j + 1):
            if i == j:
                q[k] = C[i, j]
            else:
                q[k] = 2.0 * C[i, j]
            k += 1

    if m > 0:
        rows = []
        for Ai in A:
            row = np.empty(nvec, dtype=float)
            k = 0
            for j in range(n):
                for i in range(0, j + 1):
                    if i == j:
                        row[k] = Ai[i, j]
                    else:
                        row[k] = 2.0 * Ai[i, j]
                    k += 1
            rows.append(row)
        Aeq = spr.csc_matrix(np.vstack(rows))
    else:
        Aeq = spr.csc_matrix((0, nvec))

    # Map unscaled x to svec(X) for PSDTriangle cone argument: s_psd = T x, with T diag
    weights = np.empty(nvec, dtype=float)
    k = 0
    for j in range(n):
        for i in range(0, j + 1):
            weights[k] = 1.0 if i == j else sqrt2
            k += 1
    T = spr.diags(weights, format="csc")
    # Want s_psd = x ∈ PSD, but cone uses s = A x + ... with s ∈ K.
    # Set A_psd = -T so that A_psd x + s = 0  => s = T x ∈ PSDTriangle
    A_psd = -T

    A_clar = spr.vstack([A_psd, Aeq], format="csc")
    b_clar = np.concatenate([np.zeros(nvec), np.asarray(b, dtype=float)])

    cones = [clarabel.PSDTriangleConeT(n), clarabel.ZeroConeT(m)]

    settings = clarabel.DefaultSettings()
    settings.verbose = bool(verbose)

    solver = clarabel.DefaultSolver(P, q, A_clar, b_clar, cones, settings)
    solution = solver.solve()

    x = solution.x
    z = solution.z
    y = getattr(solution, "y", None)

    # Recover primal matrix X
    X_optimal = smat_from_tri(x)

    # Dual matrix S corresponds to the PSD cone dual variable (first block of z)
    S_optimal = smat_from_svec(z[:nvec])

    # Dual multipliers for equality constraints are the last m entries of y
    if y is not None and len(y) >= nvec + m:
        y_optimal = np.array(y[nvec:])
    else:
        # Fallback: use the ZeroCone dual from z if y not provided
        y_optimal = np.array(z[nvec:])

    # Objective value: q^T x since P = 0
    f_optimal = float(q @ x)

    return X_optimal, S_optimal, y_optimal, f_optimal
