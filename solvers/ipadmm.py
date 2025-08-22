import numpy as np
import cvxpy as cp
import scipy as sc
import scipy.sparse as spr

import time

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


def X_update(C, A, b, S, Y, y, rho1, rho2):

    H = lambda X: (rho1 * X + rho2 * Ast_linop(A_linop(X, A), A))
    rhs = rho1 * S + Y + rho2 * Ast_linop(b, A) + Ast_linop(y, A) - C
    X, info = cg(H, rhs, maxiter=2000)
    if info == 1:
        print(f"CG max iterations reached")
    return X


def S_update(X, Y, mu, rho):
    I = np.eye(X.shape[0])
    A = -1 / 2 * (Y - rho * X).T
    B = I
    Q = mu * I
    R = 1 / rho * I

    sol_care = sc.linalg.solve_continuous_are(A, B, Q, R)

    return sol_care


def sdp_ipadmm(C, A, b, params={}):

    # -------------------------------- Parameters -------------------------------- #
    max_iter = params.get("max_iter", 500)

    # barrier parameters
    mu = params.get("mu", 1.0)
    sigma = params.get("sigma", 0.5)
    eta_res = params.get("eta_res", 0.2)
    eta_gap = params.get("eta_gap", 0.5)

    # ADMM parameters
    rho1 = params.get("rho1", 1.0)
    rho2 = params.get("rho2", 1.0)
    r_factor = params.get("r_factor", 10.0)
    tau = params.get("tau", 1.5)

    # tolerances
    eps_prim = params.get("eps_prim", 1e-4)
    eps_dual = params.get("eps_dual", 1e-4)
    eps_cons = params.get("eps_cons", 1e-4)
    eps_gap = params.get("eps_gap", 1e-4)

    # ---------------------------------- Solver ---------------------------------- #
    n = C.shape[0]
    m = len(A)

    print("\n" + "-" * 20 + " IPADMM " + "-" * 20)
    print(f" n: {n}, m: {m}")
    print(
        f" εp: {eps_prim:.2e}, εd: {eps_dual:.2e}, εc: {eps_cons:.2e}, εg: {eps_gap:.2e}"
    )
    print(f" μ: {mu}, σ: {sigma}, ηr: {eta_res}, ηg: {eta_gap}")
    print(f" ρ1: {rho1}, ρ2: {rho2}, τ: {tau}")
    print("\n")

    # initial values
    X0 = np.eye(n)
    S0 = np.eye(n)
    Y0 = np.zeros((n, n))
    y0 = np.zeros(m)

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
        X1 = X_update(C=C, A=A, b=b, S=S0, Y=Y0, y=y0, rho1=rho1, rho2=rho2)
        t1 = time.time()
        S1 = S_update(X=X1, Y=Y0, mu=mu, rho=rho1)
        t2 = time.time()
        Y1 = Y0 - rho1 * (X1 - S1)
        y1 = y0 - rho2 * (A_linop(X1, A) - b)
        t3 = time.time()

        # kkt residuals, gap and cost value
        f = np.trace(C @ X1)
        Z1 = C - Ast_linop(y1, A)
        r_prim = np.linalg.norm(A_linop(X1, A) - b, 2) / (1 + np.linalg.norm(b, 2))
        r_dual = np.linalg.norm(Z1 - Y1, "fro") / (
            1 + np.linalg.norm(Z1, "fro") + np.linalg.norm(Y1, "fro")
        )
        r_cons = np.linalg.norm(X1 - S1, "fro") / (
            1 + np.linalg.norm(X1, "fro") + np.linalg.norm(S1, "fro")
        )
        gap = np.abs(np.trace(Z1 @ X1)) / (1 + np.abs(f) + np.abs(b @ y1))

        # calculate residuals
        r1_primal = np.linalg.norm(X1 - S1, "fro")
        r1_dual = rho1 * np.linalg.norm(S1 - S0, "fro")

        r2_primal = np.linalg.norm(A_linop(X1, A) - b, 2)
        r2_dual = rho2 * np.linalg.norm(Ast_linop(X1 - X0, A), "fro")

        r = max(r1_primal, r1_dual)
        s = max(r2_primal, r2_dual)

        if r1_primal > r_factor * r1_dual:
            rho1 = min(rho1 * tau, 1e4)
        elif r1_dual > r_factor * r1_primal:
            rho1 = max(rho1 * 1 / tau, 1e-4)

        # update penalty parameters
        if r2_primal > r_factor * r2_dual:
            rho2 = min(rho2 * tau, 1e4)
        elif r2_dual > r_factor * r2_primal:
            rho2 = max(rho2 * 1 / tau, 1e-4)

        if (max(r_prim, r_dual, r_cons) < eta_res * mu) and (gap / n < eta_gap * mu):
            mu = sigma * mu
        else:
            mu = mu

        t4 = time.time()

        # print
        t = [t1 - t0, t2 - t1, t3 - t2, t4 - t3]
        print(
            f"{it} |  f:{f:.3f} gap:{gap:.2e} | ",
            f"rp:{r_prim:.2e},  rd:{r_dual:.2e}, rc:{r_cons:.2e} | ",
            f"r:{r:.2e}, s:{s:.2e} | ",
            f"μ:{mu:.2e}, ρ1:{rho1:.2f}, ρ2:{rho2:.2f} | ",
            # f"tX: {t[0]:.1e}, tS: {t[1]:.1e}, tYy: {t[2]:.1e}, tr: {t[3]:.1e}, tt:{sum(t):.1e}",
        )

        # iterate
        X.append(X1)
        S.append(S1)
        Y.append(Y1)
        y.append(y1)

        # check convergence
        residuals = np.array([r_prim, r_dual, r_cons, gap])
        eps = np.array([eps_prim, eps_dual, eps_cons, eps_gap])
        if all(residuals < eps):
            print(f"Converged!")
            break

    return X, S, Y, y, f
