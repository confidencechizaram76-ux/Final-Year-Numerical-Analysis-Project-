import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla
import time
import tracemalloc
import csv
import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from QMR_krylov_solver import (
    SolverMonitor,
    build_preconditioner,
    qmr
)


# =============================================================
# PATHS
# =============================================================

MATRIX_FOLDER = (
    r"C:\Users\Administrator\.vscode-insiders"
    r"\Final Year Project\CDR_matrices\Pe_experiment"
)




RESULTS_FOLDER = os.path.join(
    MATRIX_FOLDER,
    "results"
)

os.makedirs(RESULTS_FOLDER, exist_ok=True)

# =============================================================
# EXPERIMENTAL PARAMETERS
# =============================================================

MATRIX_NAMES = [
    "CDR_n400_Pe0p01",
        "CDR_n400_Pe0p1",
        "CDR_n400_Pe1p0",
        "CDR_n400_Pe10p0",
        "CDR_n400_Pe100p0",
]

PRECONDITIONERS = [
    "none",
    "jacobi",
    "ilut"
]

TOL = 1e-8
MAXITER = 4000

ILUT_DROP_TOL = 1e-4
ILUT_FILL_FACTOR = 10


RESULTS_FILE = os.path.join(
    RESULTS_FOLDER,
    "QMR_CDR_Pe_results.csv"
)
HISTORY_FILE = os.path.join(
    RESULTS_FOLDER,
    "QMR_CDR_Pe_convergence_history.csv"
)

# =============================================================
# MAIN EXPERIMENT
# =============================================================

results = []
history_results = []


print()
print("=" * 90)
print("QMR CDR-Pe-MATRIX EXPERIMENT")
print("=" * 90)


for matrix_name in MATRIX_NAMES:

    print()
    print("#" * 90)
    print(f"MATRIX: {matrix_name}")
    print("#" * 90)

    # ---------------------------------------------------------
    # Load matrix
    # ---------------------------------------------------------

    matrix_file = os.path.join(
        MATRIX_FOLDER,
        matrix_name + ".npz"
    )

    A = sp.load_npz(matrix_file).tocsr()
    

    

    # ---------------------------------------------------------
    # Load right-hand side and exact solution
    # ---------------------------------------------------------

    b_file = os.path.join(
        MATRIX_FOLDER,
        matrix_name + "_b.npy"
        )

    x_true_file = os.path.join(
    MATRIX_FOLDER,
    matrix_name + "_uexact.npy"
    )

    b = np.load(b_file)
    x_true = np.load(x_true_file)

    # Initial guess
    x0 = np.zeros_like(x_true)
    n = A.shape[0]
    nnz = A.nnz

    sparsity = (
        nnz / (n * n)
    ) * 100.0

    print(f"Dimension : {n} x {n}")
    print(f"NNZ       : {nnz}")
    print(f"Sparsity  : {sparsity:.6f}%")

    # ---------------------------------------------------------
    # Compute matrix norm once
    # ---------------------------------------------------------

    norm_start = time.perf_counter()

    A_norm = spla.norm(
        A,
        ord=2
    )

    norm_time = (
        time.perf_counter()
        - norm_start
    )

    print(f"||A||_2   : {A_norm:.6e}")
    print(f"Norm time : {norm_time:.6e} s")

    # =========================================================
    # PRECONDITIONER LOOP
    # =========================================================

    for prec_name in PRECONDITIONERS:

        print()
        print("-" * 90)
        print(
            f"Running: {matrix_name} | "
            f"QMR | {prec_name.upper()}"
        )
        print("-" * 90)

        tracemalloc.start()

        monitor = SolverMonitor(
            A=A,
            b=b,
            x_true=x_true,
            tol=TOL,
            maxiter=MAXITER
        )

        monitor.A_norm = A_norm

        # -----------------------------------------------------
        # Preconditioner setup
        # -----------------------------------------------------

        setup_start = time.perf_counter()

        try:

            M = build_preconditioner(
                A=A,
                monitor=monitor,
                preconditioner=prec_name,
                drop_tol=ILUT_DROP_TOL,
                fill_factor=ILUT_FILL_FACTOR
            )

            monitor.setup_time = (
                time.perf_counter()
                - setup_start
            )

        except Exception as e:

            monitor.setup_time = (
                time.perf_counter()
                - setup_start
            )

            monitor.status = "Preconditioner Failure"
            monitor.breakdown_reason = str(e)

            current, peak = (
                tracemalloc.get_traced_memory()
            )

            monitor.peak_memory = peak

            tracemalloc.stop()

            results.append({
                "matrix": matrix_name,
                "n": n,
                "nnz": nnz,
                "sparsity_percent": sparsity,
                "solver": "QMR",
                "preconditioner": prec_name,
                "status": monitor.status,
                "iterations": 0,
                "N_A": 0,
                "N_AT": 0,
                "N_M": 0,
                "N_MT": 0,
                "N_prec": 0,
                "setup_time": monitor.setup_time,
                "solve_time": 0.0,
                "total_time": monitor.setup_time,
                "peak_memory_MB": (
                    monitor.peak_memory / 1024**2
                ),
                "final_residual": np.nan,
                "final_forward_error": np.nan,
                "final_backward_error": np.nan,
                "breakdown_reason": str(e)
            })

            print(
                f"Preconditioner failure: {e}"
            )

            continue

        # -----------------------------------------------------
        # Solve
        # -----------------------------------------------------

        solve_start = time.perf_counter()

        try:

            x = qmr(
                A=A,
                b=b,
                x0=x0,
                x_true=x_true,
                monitor=monitor,
                preconditioner=M
            )

        except Exception as e:

            monitor.status = "Solver Failure"
            monitor.breakdown_reason = str(e)

            x = x0.copy()

        monitor.solve_time = (
            time.perf_counter()
            - solve_start
        )

        monitor.total_time = (
            monitor.setup_time
            + monitor.solve_time
        )

        # -----------------------------------------------------
        # Memory
        # -----------------------------------------------------

        current_memory, peak_memory = (
            tracemalloc.get_traced_memory()
        )

        monitor.peak_memory = max(
            monitor.peak_memory,
            peak_memory
        )

        tracemalloc.stop()

        # -----------------------------------------------------
        # Final metrics
        # -----------------------------------------------------

        if len(monitor.residual_history) > 0:

            final_residual = (
                monitor.residual_history[-1]
            )

            final_forward_error = (
                monitor.forward_error_history[-1]
            )

            final_backward_error = (
                monitor.backward_error_history[-1]
            )

        else:

            final_residual = np.nan
            final_forward_error = np.nan
            final_backward_error = np.nan

        iterations = max(
            len(monitor.iterations) - 1,
            0
        )



        # -----------------------------------------------------
# Store convergence history
# -----------------------------------------------------
        for k, eta, ef, eb in zip(
            monitor.iterations,
            monitor.residual_history,
            monitor.forward_error_history,
            monitor.backward_error_history
        ):
             history_results.append({
                "matrix": matrix_name,
                "solver": "QMR",
                "preconditioner": prec_name,
                "iteration": k,
                "relative_residual": eta,
                "relative_forward_error": ef,
                "backward_error": eb
    })
        # -----------------------------------------------------
        # Store
        # -----------------------------------------------------

        results.append({
            "matrix": matrix_name,
            "n": n,
            "nnz": nnz,
            "sparsity_percent": sparsity,
            "solver": "QMR",
            "preconditioner": prec_name,
            "status": monitor.status,
            "iterations": iterations,
            "N_A": monitor.N_A,
            "N_AT": monitor.N_AT,
            "N_M": monitor.N_M,
            "N_MT": monitor.N_MT,
            "N_prec": monitor.N_prec,
            "setup_time": monitor.setup_time,
            "solve_time": monitor.solve_time,
            "total_time": monitor.total_time,
            "peak_memory_MB": (
                monitor.peak_memory / 1024**2
            ),
            "final_residual": final_residual,
            "final_forward_error": final_forward_error,
            "final_backward_error": final_backward_error,
            "breakdown_reason": (
                monitor.breakdown_reason
                if monitor.breakdown_reason
                is not None
                else ""
            )
        })

        # -----------------------------------------------------
        # Print Final Results
        # -----------------------------------------------------

        print(
            f"Status              : {monitor.status}"
        )

        print(
            f"Iterations          : {iterations}"
        )

        print(
            f"N_A                 : {monitor.N_A}"
        )

        print(
            f"N_AT                : {monitor.N_AT}"
        )

        print(
            f"N_M                 : {monitor.N_M}"
        )

        print(
            f"N_MT                : {monitor.N_MT}"
        )

        print(
            f"N_prec              : {monitor.N_prec}"
        )

        print(
            f"Setup time          : "
            f"{monitor.setup_time:.6e} s"
        )

        print(
            f"Solve time          : "
            f"{monitor.solve_time:.6e} s"
        )

        print(
            f"Total time          : "
            f"{monitor.total_time:.6e} s"
        )

        print(
            f"Peak memory         : "
            f"{monitor.peak_memory / 1024**2:.6f} MB"
        )

        print(
            f"Final residual      : "
            f"{final_residual:.6e}"
        )

        print(
            f"Final forward error : "
            f"{final_forward_error:.6e}"
        )

        print(
            f"Final backward error: "
            f"{final_backward_error:.6e}"
        )

        # -----------------------------------------------------
        # Print Iteration history
        # -----------------------------------------------------
        print()
        print(
            f"{'Iter':>6}"
            f"{'Rel. Residual':>20}"
            f"{'Rel. Forward Error':>22}"
            f"{'Backward Error':>20}"
        )

        print("-" * 68)

        for k, eta, ef, eb in zip(
            monitor.iterations,
            monitor.residual_history,
            monitor.forward_error_history,
            monitor.backward_error_history
        ):

            print(
                f"{k:6d}"
                f"{eta:20.6e}"
                f"{ef:22.6e}"
                f"{eb:20.6e}"
            )
# =============================================================
# SAVE Convergence History CSV
# =============================================================

fieldnames_1 = [
    "matrix",
    "solver",
    "preconditioner",
    "iteration",
    "relative_residual",
    "relative_forward_error",
    "backward_error"
]

with open(
    HISTORY_FILE,
    "w",
    newline=""
) as f:

    writer = csv.DictWriter(
        f,
        fieldnames=fieldnames_1 
    )

    writer.writeheader()
    writer.writerows(history_results)


# =============================================================
# SAVE Summary Results CSV
# =============================================================
fieldnames = [
    "matrix",
    "n",
    "nnz",
    "sparsity_percent",
    "solver",
    "preconditioner",
    "status",
    "iterations",
    "N_A",
    "N_AT",
    "N_M",
    "N_MT",
    "N_prec",
    "setup_time",
    "solve_time",
    "total_time",
    "peak_memory_MB",
    "final_residual",
    "final_forward_error",
    "final_backward_error",
    "breakdown_reason"
]


with open(
    RESULTS_FILE,
    "w",
    newline=""
) as f:

    writer = csv.DictWriter(
        f,
        fieldnames=fieldnames
    )

    writer.writeheader()
    writer.writerows(results)


# =============================================================
# COMPLETE
# =============================================================

print()
print("=" * 100)
print("QMR EXPERIMENT COMPLETE")
print("=" * 100)

print(
    f"Number of runs: {len(results)}"
)

print(
    f"Results saved to:\n{RESULTS_FILE}"
)


print(
    f"Convergence histories saved to:\n{HISTORY_FILE}"
)

print("=" * 100)