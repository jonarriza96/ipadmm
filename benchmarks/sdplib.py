# %%
import numpy as np
import cvxpy as cp


from solvers.external import sdp_cvx, sdp_mosek, sdp_clarabel
from solvers.ipadmm import sdp_ipadmm
from benchmarks.utils import load_sdplib


if __name__ == "__main__":

    solve_mosek = 0
    solve_clarabel = 1
    solve_ipadmm = 1

    problem_name = "control1"
    sdplib_path = "/Users/jonarrizabalaga/ipadmm/benchmarks/sdplib"

    print(f"Loading {problem_name} from {sdplib_path}...")
    C, A, b, f_sdplib = load_sdplib(sdplib_path=sdplib_path, problem_name=problem_name)
    print("done")

    # X_cvx, S_cvx, y_cvx, f = sdp_cvx(C=C, A=A, b=b, solver=cp.CLARABEL, verbose=True)
    if solve_mosek:
        X_mosek, S_mosek, y_mosek, f = sdp_mosek(C=C, A=A, b=b, verbose=True)
        print(f"\n|f_mosek - f_sdplib| : {np.abs(f - f_sdplib):.5e}\n")

    if solve_clarabel:
        X_clar, S_clar, y_clar, f = sdp_clarabel(
            C=C,
            A=A,
            b=b,
            verbose=True,
            # max_iter=13,  # ipadmm does not work
            max_iter=25,  # ipadmm works
        )
        print(f"\n|f_clarabel - f_sdplib| : {np.abs(f - f_sdplib):.5e}\n")

    if solve_ipadmm:

        params = {
            "max_iter": 500,
            "scaling": False,
            # "X_backend": "direct",
            # "S_backend": "proj",
            # "rho1": 100.0,
            # "rho2": 1.0,
            # "sigma": 0.95,
            # "tau": 2.0,
            # "r_factor": 5.0,
        }
        X_ipadmm, S_ipadmm, Y_ipadmm, y_ipadmm, f_ipadmm = sdp_ipadmm(
            C=C,
            A=A,
            b=b,
            params=params,
            # init=(X_mosek, X_mosek, S_mosek, y_mosek),
            init=(X_clar, X_clar, S_clar, y_clar),
        )
        print(f"\n|f_ipadmm - f_sdplib| : {np.abs(f_ipadmm - f_sdplib):.5e}\n")
