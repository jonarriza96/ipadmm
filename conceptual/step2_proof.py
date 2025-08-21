# %%
"""
We show that the solution to the problem
    min_S -mu*log(det(S)) + rho/2*||X-S||_F^2 - trace(Y @ (X-S))
is the solution to the CARE
    A.T @ S + S @ A - S @ B @ R_inv @ B.T @ S + Q = 0
where
    A = -1/2 * (Y - rho * X).T
    B = I
    R = 1/rho * I
    Q = mu * I
"""

import cvxpy as cp
import numpy as np
import scipy as sc

n = 5
mu = 0.5
rho = 1.0


# -------------------------------- cvxpy solve ------------------------------- #

# Create random symmetric matrices for X and Y
np.random.seed(0)
X_rand = np.random.randn(n, n)
Y_rand = np.random.randn(n, n)
X = X_rand + X_rand.T  # Make X symmetric
Y = Y_rand + Y_rand.T  # Make Y symmetric

# 2. Define the CVXPY variable
S = cp.Variable((n, n))

# 3. Construct the objective function using CVXPY atoms
log_det_term = -mu * cp.log_det(S)
norm_sq_term = (rho / 2) * cp.sum_squares(X - S)  # Squared Frobenius norm
inner_prod_term = -cp.trace(Y @ (X - S))

objective = cp.Minimize(log_det_term + norm_sq_term + inner_prod_term)

# 4. Define the constraints
constraints = []

# 5. Form and solve the problem
problem = cp.Problem(objective, constraints)
problem.solve()

# 6. Display the results
if problem.status == cp.OPTIMAL:
    sol_cvx = S.value
    pass
else:
    raise ValueError("Problem did not solve to optimality. Status:", problem.status)


# ------------------------- Conversion to  CARE form ------------------------- #
# A.T @ S + S @ A - S @ B @ R_inv @ B.T @ S + Q = 0

I = np.eye(n)
A = -1 / 2 * (Y - rho * X).T
B = I
Q = mu * I
R = 1 / rho * I

sol_care = sc.linalg.solve_continuous_are(A, B, Q, R)


# --------------------------------- Validate --------------------------------- #

S = sol_care
S_inv = np.linalg.inv(S)

eq_opt_cond = mu * S_inv - (Y - rho * X) - rho * S
eq_care = A.T @ S + S @ A - S @ B @ np.linalg.inv(R) @ B.T @ S + Q


print(f"CVX ~= CARE: {np.allclose(sol_cvx, sol_care, atol=1e-4)}")
print(f"Optimality conditions at S* == 0: {np.allclose(eq_opt_cond, 0)}")
print(f"CARE at S*: {np.allclose(eq_opt_cond, 0)}")
