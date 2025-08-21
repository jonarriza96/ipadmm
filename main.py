# %%
import numpy as np
import scipy as sc
import cvxpy as cp

from solvers.external import sdp_cvx
from solvers.ipadmm import solve_ipadmm


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
    # P @ P.T is always ¬positive semidefinite; it's positive definite if P has full rank.
    P = np.random.randn(n, n)
    S_dual_known = P @ P.T  # + 1e-4 * np.eye(n)

    # Now, construct C to ensure the dual is strictly feasible.
    # C = sum(y_i * A_i) + S, which means S = C - sum(y_i * A_i).
    # Since S is positive definite (S > 0), the dual constraint is strictly satisfied.
    C = sum(y_dual_known[i] * A[i] for i in range(m)) + S_dual_known

    return C, A, b


if __name__ == "__main__":
    np.random.seed(42)

    n = 5  # dimension of X
    m = 3  # number of linear equality constraints

    # define the problem
    C, A, b = generate_random_sdp(n, m)

    # solve with cvx
    X_cvx, S_cvx, y_cvx, f_cvx = sdp_cvx(
        C=C, A=A, b=b, solver=cp.CLARABEL, verbose=True
    )

    # solve ipdamm
    X_ipadmm, S_ipadmm, Y_ipadmm, y_ipadmm = solve_ipadmm(C, A, b)
