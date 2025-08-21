# %%
"""
We show that the solution to the problem

min <C, X> + ρ1/2*||X-S||_F^2 - trace(Y1 @ (X-S)) + ρ2/2*||A(X) - b||_2^2 - y2.T @ (A(X) - b))
    X

reduces to solving the unconstrained quadratic problem
    (ρ1 * I + A_star_A) @ X =  ρ1*S + Y1 + ρ2*A_star(b) + A_star(y2) - C,

which is equivalen to solving H @ x = rhs .

Given that H ⪰ 0 (as long as ρ1 > 0), we can solve this problem with conjugate gradient.
This is motivated by:
    - Large and sparse problems: CG only requires matrix-vector products
    - Memory efficient: Only stores vectors of size n^2, not the full n^2 x n^2 matrix
    - Exact solutions: In exact arithmetic CV converges at most n^2 steps (fewer with preconditioning)
"""

import numpy as np
import cvxpy as cp


def generate_random_sdp(n, m):
    print(f"Generating a random, solvable Semidefinite Program (SDP) with:")
    print(f" - Matrix size (n x n): {n}x{n}")
    print(f" - Number of constraints (m): {m}\n")

    # --- 2. Generate Random Data Guaranteed to be Feasible and Bounded ---

    # Create a list of 'm' random symmetric matrices A_i for the constraints
    A = []
    for i in range(m):
        A_i_rand = np.random.randn(n, n)
        A.append(A_i_rand + A_i_rand.T)

    # To guarantee the primal is feasible, we create a known primal solution.
    # The identity matrix is positive semidefinite, so it's a safe choice.
    X_primal_known = np.eye(n)
    # Then, we construct 'b' from this known solution.
    b = np.squeeze([np.trace(A_i.T @ X_primal_known) for A_i in A])

    # To guarantee the primal is bounded, we create a known strictly feasible
    # solution for the dual problem.
    y_dual_known = np.random.randn(m)

    # We also need a positive definite matrix S.
    # We create it by multiplying a random matrix P by its transpose.
    # P @ P.T is always positive semidefinite; it's positive definite if P has full rank.
    P = np.random.randn(n, n)
    S_dual_known = P @ P.T  # + 1e-4 * np.eye(n)

    # Now, construct C to ensure the dual is strictly feasible.
    # C = sum(y_i * A_i) + S, which means S = C - sum(y_i * A_i).
    # Since S is positive definite (S > 0), the dual constraint is strictly satisfied.
    C = sum(y_dual_known[i] * A[i] for i in range(m)) + S_dual_known

    return C, A, b


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


np.random.seed(0)

n = 5
m = 3
rho1 = np.random.rand()
rho2 = np.random.rand()


# generate raddom problem date
C, A, b = generate_random_sdp(n, m)
S_rand = np.random.randn(n, n)
Y_rand = np.random.randn(n, n)
y2 = np.random.randn(m)
S = S_rand + S_rand.T  # Make S symmetric
Y1 = Y_rand + Y_rand.T  # Make Y symmetric


# ------------------------------ Solve with CVX ------------------------------ #
X = cp.Variable((n, n))

objective = cp.Minimize(
    cp.trace(C @ X)
    + rho1 / 2 * cp.norm(X - S, "fro") ** 2
    - cp.trace(Y1 @ (X - S))
    + rho2 / 2 * cp.norm(A_linop(X, A) - b, 2) ** 2
    - y2.T @ (A_linop(X, A) - b)
)

constraints = []

problem = cp.Problem(objective, constraints)
problem.solve()
if problem.status == cp.OPTIMAL:
    sol_cvx = X.value
    pass
else:
    raise ValueError("Problem did not solve to optimality. Status:", problem.status)


# ------------------------------- Solve with CG ------------------------------ #

A_mat = np.stack([Ai.flatten() for Ai in A], axis=0)  # shape m x (n*n)
A_star_A = A_mat.T @ A_mat
I = np.eye(n * n)

H = lambda X: (rho1 * X + rho2 * Ast_linop(A_linop(X, A), A))
rhs = rho1 * S + Y1 + rho2 * Ast_linop(b, A) + Ast_linop(y2, A) - C
sol_cg, info = cg(H, rhs, maxiter=2000)

if info == 1:
    print(f"CG max iterations reached")


print(f"CVX ~= CG: {np.allclose(sol_cvx, sol_cg, atol=1e-4)}")
