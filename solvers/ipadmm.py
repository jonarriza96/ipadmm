import numpy as np
import cvxpy as cp
import scipy as sc
import scipy.sparse as spr

import time
import sys

"""
We solve
    min <C, X>
    X
    s.t. A(X) = b            →      <A_i, X> = b_i, i = 1, ..., m
         X ⪰ 0

with an interior-point ADMM method as:
    min <C, X>
    X,S
    s.t.
         A(x) = b            →      <A_i, X> = b_i, i = 1, ..., m
         S = X
         S ⪰ 0 .

We do this by solving it with the following steps:

1. Solve
    min <C, X> + ρ1/2*||X-S||_F^2 - trace(Y @ (X-S)) + ρ2/2*||A(X) - b||_2^2 - y.T @ (A(X) - b))
    X

2. Solve
    min - μ*log(det(S)) + ρ1/2*||X-S||_F^2 - trace(Y @ (X-S))
    S
    s.t. S ⪰ 0

3. Update Y and y
    Y = Y - ρ1 * (X - S)
    y = y - ρ2 * (A(X) - b)

4. Update μ, ρ1, ρ2
    μ = σ * μ
    ρ1 = τ * ρ1
    ρ2 = τ * ρ2
    where τ is a scaling factor depending on the residuals.

"""


def scale_sdp_data(C, A, b):
    """
    Scales the constraint matrices and vector of a semidefinite program (SDP).

    This function normalizes each linear constraint of the SDP:
        <A_i, X> = b_i
    by dividing both A_i and b_i by the Frobenius norm of A_i. This is a
    standard pre-processing step to improve the numerical stability and
    convergence of SDP solvers, especially for ADMM-based methods.

    The objective matrix C is returned unchanged.

    Args:
        C (np.ndarray): The objective matrix (n x n).
        A (List[np.ndarray]): A list of m constraint matrices (each n x n).
        b (np.ndarray): The constraint vector of length m.

    Returns:
        Tuple[np.ndarray, List[np.ndarray], np.ndarray, np.ndarray]: A tuple containing:
            - C_scaled (np.ndarray): The original, unscaled objective matrix.
            - A_scaled (List[np.ndarray]): The list of scaled constraint matrices.
            - b_scaled (np.ndarray): The scaled constraint vector.
            - scaling_factors (np.ndarray): The computed scaling factors for each constraint.
    """
    # Ensure b is a numpy array for vectorized operations
    b = np.asarray(b)

    # Get the number of constraints
    num_constraints = len(A)
    if num_constraints != len(b):
        raise ValueError(
            "The number of constraint matrices in A must match the length of b."
        )

    # Create copies to avoid modifying the original input data
    A_scaled = [a.copy() for a in A]
    b_scaled = b.copy().astype(float)  # Use float for scaling

    # Array to store the scaling factors for diagnostics or un-scaling dual variables
    scaling_factors = np.ones(num_constraints)

    print(f"Scaling SDP data...")

    # Iterate through each constraint to calculate and apply the scaling factor
    for i in range(num_constraints):
        # Calculate the Frobenius norm of the constraint matrix A_i
        norm_Ai = np.linalg.norm(A[i], "fro")

        # To avoid division by zero, only scale if the norm is significant
        if norm_Ai > 1e-8:
            scaler = 1.0 / norm_Ai
            scaling_factors[i] = scaler

            # Apply the scaling to both the matrix A_i and the scalar b_i
            A_scaled[i] *= scaler
            b_scaled[i] *= scaler
        else:
            # If the norm is zero or close to it, don't scale this constraint.
            # This corresponds to a constraint like <0, X> = 0, which is trivial.
            print(
                f"Warning: Constraint {i} has a near-zero norm. Skipping scaling for this constraint."
            )

    C_nf = np.linalg.norm(C, "fro")
    A_nf = [np.linalg.norm(A_i, "fro") for A_i in A]
    b_n2 = np.linalg.norm(b, 2)
    Cs_nf = np.linalg.norm(C, "fro")
    As_nf = [np.linalg.norm(A_i, "fro") for A_i in A_scaled]
    bs_n2 = np.linalg.norm(b_scaled, 2)
    print(
        "\tUnscaled:",
        f"‖C‖_F: {C_nf.item():.2e}",
        f"Σ(‖A_i‖_F)/m: {np.mean([round(A_i.item(), 2) for A_i in A_nf]):.2e}",
        f"‖b‖_2: {b_n2:.2e}",
    )
    print(
        "\tScaled:",
        f"  ‖C‖_F: {Cs_nf.item():.2e}",
        f"Σ(‖A_i‖_F)/m: {np.mean([round(A_i.item(), 2) for A_i in As_nf]):.2e}",
        f"‖b‖_2: {bs_n2:.2e}\n",
    )
    return C, A_scaled, b_scaled, scaling_factors


def A_linop(X, A_list):
    """Linear operator A(X) -> R^m"""
    # Numpy variant
    if isinstance(X, np.ndarray):
        return np.array([np.tensordot(Ai, X, axes=2) for Ai in A_list])
    # CVXPY variant
    elif isinstance(X, cp.expressions.expression.Expression):
        return cp.hstack([cp.trace(Ai @ X) for Ai in A_list])
    else:
        raise TypeError("X must be either a numpy array or a CVXPY Expression.")


def Ast_linop(y, A_list):
    """Adjoint operator A*(y) -> S^n"""
    # Numpy variant
    if isinstance(y, np.ndarray):
        X = np.zeros_like(A_list[0])
        for yi, Ai in zip(y, A_list):
            X += yi * Ai
        return X
    # CVXPY variant
    elif isinstance(y, cp.expressions.expression.Expression):
        # y is a CVXPY vector expression
        X = 0
        for i in range(len(A_list)):
            X += y[i] * A_list[i]
        return X
    else:
        raise TypeError("y must be either a numpy array or a CVXPY Expression.")


def cg(H, rhs, X0=None, tol=1e-8, maxiter=1000):
    """
    Solve H(X) = rhs using Conjugate Gradient.

    Parameters
    ----------
    H : callable
        Function H(X) that returns a matrix of same shape as X.
    rhs : np.ndarray
        Right-hand side matrix.
    X0 : np.ndarray, optional
        Initial guess for X. If None, zeros are used.
    tol : float
        Relative tolerance for convergence.
    maxiter : int
        Maximum number of iterations.

    Returns
    -------
    X : np.ndarray
        Solution matrix.
    info : int
        0 if converged, 1 if max iterations reached.
    """

    if X0 is None:
        X = np.zeros_like(rhs)
    else:
        X = X0.copy()

    R = rhs - H(X)  # residual
    P = R.copy()
    rsold = np.tensordot(R, R)

    for k in range(maxiter):
        HP = H(P)
        alpha = rsold / np.tensordot(P, HP)
        X += alpha * P
        # X = (X + X.T) / 2
        R -= alpha * HP
        rsnew = np.tensordot(R, R)
        if np.sqrt(rsnew) < tol:
            return X, 0  # converged
        P = R + (rsnew / rsold) * P
        rsold = rsnew

    return X, 1  # max iterations reached


def X_update(C, A, b, X, S, Y, y, rho1, rho2, method="cg"):
    """
    Solves
    min <C, X> + ρ1/2*||X-S||_F^2 - trace(Y @ (X-S)) + ρ2/2*||A(X) - b||_2^2 - y.T @ (A(X) - b))
    X

    method:
        - "cg": iterative conjugate gradient (default)
        - "direct": explicit system solve via np.linalg.solve
    """

    if method == "cg":
        H = lambda X: (rho1 * X + rho2 * Ast_linop(A_linop(X, A), A))
        rhs = rho1 * S + Y + rho2 * Ast_linop(b, A) + Ast_linop(y, A) - C
        X_sol, info = cg(H, rhs, maxiter=2000)
        if info == 1:
            print(f"CG max iterations reached")

    elif method == "direct":
        n = C.shape[0]
        # Build A_mat with rows vec(Ai)^T
        A_mat = (
            np.vstack([Ai.reshape(1, -1) for Ai in A])
            if len(A) > 0
            else np.zeros((0, n * n))
        )
        Hmat = rho1 * np.eye(n * n) + rho2 * (A_mat.T @ A_mat)

        rhs = rho1 * S + Y + rho2 * Ast_linop(b, A) + Ast_linop(y, A) - C
        rhs_vec = rhs.reshape(-1)

        x_vec = np.linalg.solve(Hmat, rhs_vec)
        X_sol = x_vec.reshape(n, n)

    elif method == "cvx":
        X = cp.Variable((X.shape[0], X.shape[1]))
        objective = cp.Minimize(
            cp.trace(C @ X)
            + rho1 / 2 * cp.norm(X - S, "fro") ** 2
            - cp.trace(Y @ (X - S))
            + rho2 / 2 * cp.norm(A_linop(X, A) - b, 2) ** 2
            - y.T @ (A_linop(X, A) - b)
        )

        try:
            problem = cp.Problem(objective, [])
            problem.solve(verbose=0)
        except Exception:
            problem = cp.Problem(objective, [])
            problem.solve(verbose=1)

        if problem.status == cp.OPTIMAL:
            X_sol = X.value
            X_sol = 0.5 * (X_sol + X_sol.T)

        else:
            raise ValueError(
                "Problem did not solve to optimality. Status:", problem.status
            )
    else:
        raise ValueError(f"Unknown method '{method}'. Use 'cg' or 'direct'.")

    X_sol = 0.5 * (X_sol + X_sol.T)
    return X_sol


def S_update(X, Y, mu, rho, method="care", eps=1e-14):
    """
    Solves
    min - μ*log(det(S)) + ρ1/2*||X-S||_F^2 - trace(Y @ (X-S))
    S
    s.t. S ⪰ 0

    method:
        - "care": current Riccati-based update
        - "proj": projection onto PSD cone of V = X - Y/ρ
    """
    if method == "care":
        I = np.eye(X.shape[0])
        A = -1 / 2 * (Y - rho * X).T
        B = I
        Q = mu * I
        R = 1 / rho * I
        S_sol = sc.linalg.solve_continuous_are(A, B, Q, R)

    elif method in ("proj"):
        V = X - (1.0 / rho) * Y
        V = 0.5 * (V + V.T)
        w, U = np.linalg.eigh(V)
        w = np.maximum(w, eps)  # eps=0 → pure PSD projection
        S_sol = U @ np.diag(w) @ U.T

    elif method == "cvx":
        S = cp.Variable((X.shape[0], X.shape[1]))
        log_det_term = -mu * cp.log_det(S)
        norm_sq_term = (rho / 2) * cp.sum_squares(X - S)  # Squared Frobenius norm
        inner_prod_term = -cp.trace(Y @ (X - S))
        objective = cp.Minimize(log_det_term + norm_sq_term + inner_prod_term)

        try:
            problem = cp.Problem(objective, [])
            problem.solve(verbose=0)
        except Exception:
            problem = cp.Problem(objective, [])
            problem.solve(verbose=1)

        if problem.status == cp.OPTIMAL:
            S_sol = S.value
        else:
            raise ValueError(
                "Problem did not solve to optimality. Status:", problem.status
            )
    else:
        raise ValueError(f"Unknown method '{method}'. Use 'care' or 'proj'.")

    S_sol = 0.5 * (S_sol + S_sol.T)
    return S_sol


def calculate_kkt_residuals(X, S, Y, y, C, A, b, scale=True):
    f = np.trace(C @ X)
    r_prim = np.linalg.norm(A_linop(X, A) - b, 2)
    r_dual = np.linalg.norm(C - Ast_linop(y, A) - Y, "fro")
    r_cons = np.linalg.norm(X - S, "fro")
    r_gap = np.abs((f - b.T @ y))

    if scale:
        r_prim = r_prim / (1 + np.linalg.norm(b, 2))
        r_dual = r_dual / (
            1 + np.linalg.norm(C - Ast_linop(y, A), "fro") + np.linalg.norm(Y, "fro")
        )
        r_cons = r_cons / (1 + np.linalg.norm(X, "fro") + np.linalg.norm(S, "fro"))
        r_gap = r_gap / (1 + np.abs(f) + np.abs(b.T @ y))

    return r_prim, r_dual, r_cons, r_gap


def sdp_ipadmm(C, A, b, params={}, init=None):

    # -------------------------------- Parameters -------------------------------- #
    max_iter = params.get("max_iter", 500)
    X_backend = params.get("X_backend", "cvx")
    S_backend = params.get("S_backend", "cvx")
    scaling = params.get("scaling", True)

    # barrier parameters
    mu = params.get("mu", 1.0)
    sigma = params.get("sigma", 0.8)
    eta = params.get("eta", 0.5)

    # ADMM parameters
    rho1 = params.get("rho1", 1.0)
    rho2 = params.get("rho2", 1.0)
    r_factor = params.get("r_factor", 10.0)
    tau = params.get("tau", 1.5)
    rho_max = params.get("rho_max", 1e2)
    rho_min = params.get("rho_min", 1e-2)

    # tolerances
    eps_prim = params.get("eps_prim", 1e-4)
    eps_dual = params.get("eps_dual", 1e-4)
    eps_cons = params.get("eps_cons", 1e-4)
    eps_gap = params.get("eps_gap", 1e-4)

    # ---------------------------------- Solver ---------------------------------- #
    n = C.shape[0]
    m = len(A)

    print("\n" + "-" * 20 + " IPADMM " + "-" * 20 + "\n")

    print(f" n: {n}, m: {m}")
    print(
        f" εp: {eps_prim:.1e}, εd: {eps_dual:.1e}, εc: {eps_cons:.1e}, εg: {eps_gap:.1e}"
    )
    print(f" μ: {mu}, σ: {sigma}, η: {eta}")
    print(f" ρ1: {rho1}, ρ2: {rho2}, τ: {tau}, r_factor: {r_factor}")
    print(f" scaling: {scaling}")
    print(f" X_backend: {X_backend}, S_backend: {S_backend}")
    print("\n")

    if scaling:
        C, A, b, _ = scale_sdp_data(C, A, b)

    # initial values
    if init is None:
        X0 = np.eye(n)
        S0 = np.eye(n)
        Y0 = np.zeros((n, n))
        y0 = np.zeros(m)
    else:
        X0, S0, Y0, y0 = init
    r_prim, r_dual, r_cons, r_gap = calculate_kkt_residuals(
        X=X0, S=S0, Y=Y0, y=y0, C=C, A=A, b=b, scale=True
    )
    print(
        f"Initial residuals: → "
        f"r_prim: {r_prim:.2e}, r_dual: {r_dual:.2e}, r_cons: {r_cons:.2e}, r_gap: {r_gap:.2e}\n"
    )
    # sys.exit()

    # iterate
    X = [X0]
    S = [S0]
    Y = [Y0]
    y = [y0]
    for it in range(max_iter):
        X0, S0 = X[it], S[it]
        Y0, y0 = Y[it], y[it]

        # steps 1 to 3
        t0 = time.time()
        X1 = X_update(
            C=C,
            A=A,
            b=b,
            X=X0,
            S=S0,
            Y=Y0,
            y=y0,
            rho1=rho1,
            rho2=rho2,
            method=X_backend,
        )
        t1 = time.time()
        S1 = S_update(X=X1, Y=Y0, mu=mu, rho=rho1, method=S_backend)
        t2 = time.time()
        Y1 = Y0 - rho1 * (X1 - S1)
        y1 = y0 - rho2 * (A_linop(X1, A) - b)
        t3 = time.time()

        # kkt residuals, gap and cost value
        f = np.trace(C @ X1)
        r_prim, r_dual, r_cons, r_gap = calculate_kkt_residuals(
            X=X1, S=S1, Y=Y1, y=y1, C=C, A=A, b=b, scale=True
        )

        # step residuals
        rX = (
            f
            + rho1 / 2 * np.linalg.norm(X1 - S1, "fro") ** 2
            - np.trace(Y1 @ (X1 - S1))
            + rho2 / 2 * np.linalg.norm(A_linop(X1, A) - b, 2) ** 2
            - y1.T @ (A_linop(X1, A) - b)
        )
        detS = np.linalg.det(S1)
        if detS > -1e-4:
            logdet = np.log(np.abs(np.linalg.det(S1)))
        else:
            raise ValueError("S is not positive definite")
        rS = (
            -mu * logdet
            + rho1 / 2 * np.linalg.norm(X1 - S1, "fro") ** 2
            - np.trace(Y1 @ (X1 - S1))
        )

        # calculate residuals
        r1_primal = np.linalg.norm(X1 - S1, "fro")
        r1_dual = rho1 * np.linalg.norm(S1 - S0, "fro")
        # r1_dual = np.linalg.norm(S1 - S0, "fro")

        r2_primal = np.linalg.norm(A_linop(X1, A) - b, 2)
        r2_dual = rho2 * np.linalg.norm(Ast_linop(X1 - X0, A), "fro")
        # r2_dual = np.linalg.norm(Ast_linop(X1 - X0, A), "fro")

        # update penalty parameters
        rho_update = 1
        if rho_update:
            if r1_primal > r_factor * r1_dual:
                rho1 = min(rho1 * tau, rho_max)
            elif r1_dual > r_factor * r1_primal:
                rho1 = max(rho1 * 1 / tau, rho_min)

            if r2_primal > r_factor * r2_dual:
                rho2 = min(rho2 * tau, rho_max)
            elif r2_dual > r_factor * r2_primal:
                rho2 = max(rho2 * 1 / tau, rho_min)

        if max(r_prim, r_dual, r_cons) < eta * mu:
            mu = sigma * mu

        t4 = time.time()

        # print
        t = [t1 - t0, t2 - t1, t3 - t2, t4 - t3]
        print(
            f"{it} |  f:{f:.3f} | ",
            f"rp:{r_prim:.2e},  rd:{r_dual:.2e}, rc:{r_cons:.2e}, rg:{r_gap:.2e} | ",
            f"r1p:{r1_primal:.2e}, r1d:{r1_dual:.2e}, r2p:{r2_primal:.2e}, r2d:{r2_dual:.2e} | ",
            f"rX:{rX:.2e}, rS:{rS:.2e} | ",
            f"μ:{mu:.2e}, ρ1:{rho1:.2f}, ρ2:{rho2:.2f} | ",
            # f"tX: {t[0]:.1e}, tS: {t[1]:.1e}, tYy: {t[2]:.1e}, tr: {t[3]:.1e}, tt:{sum(t):.1e}",
        )

        # iterate
        X.append(X1)
        S.append(S1)
        Y.append(Y1)
        y.append(y1)

        # check convergence
        residuals = np.array([r_prim, r_dual, r_cons, r_gap])
        eps = np.array([eps_prim, eps_dual, eps_cons, eps_gap])
        if all(residuals < eps):
            print(f"Converged!")
            break

    return X, S, Y, y, f
