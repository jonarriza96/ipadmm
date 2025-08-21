# %%
import numpy as np
import scipy as sc
import cvxpy as cp

from ipadmm import solve_ipadmm


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


def sdp_cvx(C, A, b, solver):
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
    problem.solve(solver=solver)  # Using SCS solver explicitly
    print(f" - Problem Status: {problem.status}")

    if problem.status == cp.OPTIMAL or problem.status == cp.OPTIMAL_INACCURATE:
        # Print the optimal value
        print(f" - Optimal Value: {problem.value:.4f}")

        # Retrieve the optimal solution for X
        X_optimal = X.value
        S_optimal = psd_constr[0].dual_value
        y_optimal = np.array([c.dual_value for c in linear_constr])

    else:
        raise ValueError(
            f"The problem could not be solved to optimality. Status: {problem.status}"
        )

    return X_optimal, S_optimal, y_optimal


if __name__ == "__main__":
    np.random.seed(42)

    n = 5  # dimension of X
    m = 3  # number of linear equality constraints

    # define the problem
    C, A, b = generate_random_sdp(n, m)

    # solve with cvx
    X_cvx, S_cvx, y_cvx = sdp_cvx(C, A, b, cp.CLARABEL)

    # solve ipdamm
    X_ipadmm, S_ipadmm, Y_ipadmm, y_ipadmm = solve_ipadmm(C, A, b)
