import numpy as np
import scipy
import scipy.sparse as sp
import scipy.sparse.linalg as spla
import time
import tracemalloc 

class SolverMonitor:
    """
    Common monitoring and performance accounting for all Krylov solvers.
    """

    def __init__(self, A, b, x_true, tol=1e-8, maxiter=4000):
        self.A = A
        self.b = np.asarray(b)
        self.x_true = np.asarray(x_true)

        self.tol = tol
        self.maxiter = maxiter

        # ---------------------------------------------------------
        # Operator-application counters
        # ---------------------------------------------------------
        self.N_A = 0       # Applications of A
        self.N_AT = 0      # Applications of A^T
        self.N_M = 0       # Applications of M^{-1}
        self.N_MT = 0      # Applications of M^{-T}

        # ---------------------------------------------------------
        # Histories
        # ---------------------------------------------------------
        self.iterations = []
        self.residual_history = []
        self.forward_error_history = []
        self.backward_error_history = []

        # ---------------------------------------------------------
        # Timing
        # ---------------------------------------------------------
        self.setup_time = 0.0
        self.solve_time = 0.0
        self.total_time = 0.0

        # ---------------------------------------------------------
        # Status
        # ---------------------------------------------------------
        self.status = "Not Started"
        self.breakdown_reason = None

        # ---------------------------------------------------------
        # Memory
        # ---------------------------------------------------------
        self.peak_memory = 0

        # ---------------------------------------------------------
        # Norms used repeatedly
        # ---------------------------------------------------------
        self.b_norm = np.linalg.norm(self.b)
        self.x_true_norm = np.linalg.norm(self.x_true)

        # A norm will be supplied later.
        self.A_norm = None

    @property
    def N_prec(self):
        """
        Total number of preconditioner applications.
        """
        return self.N_M + self.N_MT

    def record(self, x, iteration, r=None):

    # ---------------------------------------------------------
    # Compute the true residual only if it was not supplied.
    # ---------------------------------------------------------
        if r is None:
            r = self.b - self.A @ x

        # This is an actual application of A.
            self.N_A += 1

        r_norm = np.linalg.norm(r)

    # ---------------------------------------------------------
    # Relative true residual
    # ---------------------------------------------------------
        if self.b_norm != 0:
            eta = r_norm / self.b_norm
        else:
            eta = r_norm

    # ---------------------------------------------------------
    # Relative forward error
    # ---------------------------------------------------------
        if self.x_true_norm != 0:
            forward_error = (
            np.linalg.norm(x - self.x_true)
            / self.x_true_norm
                )
        else:
            forward_error = np.linalg.norm(
            x - self.x_true
                )

    # ---------------------------------------------------------
    # Relative backward error
    # ---------------------------------------------------------
        if self.A_norm is not None:

            denominator = (
            self.A_norm * np.linalg.norm(x)
            + self.b_norm
            )

            if denominator != 0:
                backward_error = (
                r_norm / denominator
            )
            else:
                backward_error = 0.0

        else:
            backward_error = np.nan

    # ---------------------------------------------------------
    # Store history
    # ---------------------------------------------------------
        self.iterations.append(iteration)
        self.residual_history.append(eta)
        self.forward_error_history.append(forward_error)
        self.backward_error_history.append(backward_error)

        return eta, forward_error, backward_error

        

        

    def update_peak_memory(self):

        current, peak = tracemalloc.get_traced_memory()

        self.peak_memory = max(
        self.peak_memory,
        peak
        )


class InstrumentedOperator:
    """
    Wrapper around A or A^T that counts matrix-vector products.
    """

    def __init__(self, A, monitor, transpose=False):
        self.A = A
        self.monitor = monitor
        self.transpose = transpose

        if transpose:
            self.shape = (A.shape[1], A.shape[0])
        else:
            self.shape = A.shape

        self.dtype = A.dtype

    def matvec(self, x):

        if self.transpose:
            self.monitor.N_AT += 1
            return self.A.T @ x

        else:
            self.monitor.N_A += 1
            return self.A @ x

    def as_linear_operator(self):

        return spla.LinearOperator(
            shape=self.shape,
            matvec=self.matvec,
            dtype=self.dtype
        )

class Preconditioner:
    """
    Base class for a left preconditioner.

    Provides:
        apply(x)           -> M^{-1} x
        apply_transpose(x) -> M^{-T} x
    """

    def __init__(self, A, monitor):
        self.A = A
        self.monitor = monitor

    def apply(self, x):
        """
        Apply M^{-1}.
        """
        self.monitor.N_M += 1
        return x

    def apply_transpose(self, x):
        """
        Apply M^{-T}.
        """
        self.monitor.N_MT += 1
        return x


class NoPreconditioner(Preconditioner):

    def apply(self, x):
        return x

    def apply_transpose(self, x):
        return x


    
class JacobiPreconditioner(Preconditioner):
    """
    Jacobi (diagonal) preconditioner.

    M = diag(A)
    """

    def __init__(self, A, monitor):
        super().__init__(A, monitor)

        diagonal = A.diagonal()

        if np.any(diagonal == 0):
            raise ValueError(
                "Jacobi preconditioner cannot be constructed: "
                "A contains a zero diagonal entry."
            )

        self.inv_diagonal = 1.0 / diagonal

    def apply(self, x):
        """
        Apply M^{-1} x.
        """

        self.monitor.N_M += 1

        return self.inv_diagonal * x

    def apply_transpose(self, x):
        """
        Apply M^{-T} x.

        For a real diagonal Jacobi preconditioner,
        M^{-T} = M^{-1}.
        """

        self.monitor.N_MT += 1

        return self.inv_diagonal * x


class ILUTPreconditioner(Preconditioner):
    """
    ILUT preconditioner based on scipy.sparse.linalg.spilu.
    """

    def __init__(
        self,
        A,
        monitor,
        drop_tol=1e-4,
        fill_factor=10
    ):
        super().__init__(A, monitor)

        # spilu requires CSC format.
        A_csc = A.tocsc()

        self.ilu = spla.spilu(
            A_csc,
            drop_tol=drop_tol,
            fill_factor=fill_factor
        )

    def apply(self, x):
        """
        Apply M^{-1} x.
        """

        self.monitor.N_M += 1

        return self.ilu.solve(x, trans='N')

    def apply_transpose(self, x):
        """
        Apply M^{-T} x.
        """

        self.monitor.N_MT += 1

        return self.ilu.solve(x, trans='T')

def build_preconditioner(
    A,
    monitor,
    preconditioner="none",
    drop_tol=1e-4,
    fill_factor=10
):
    """
    Construct the requested preconditioner.
    """

    name = preconditioner.lower()

    if name == "none":
        return NoPreconditioner(A, monitor)

    elif name == "jacobi":
        return JacobiPreconditioner(A, monitor)

    elif name == "ilut":
        return ILUTPreconditioner(
            A,
            monitor,
            drop_tol=drop_tol,
            fill_factor=fill_factor
        )

    else:
        raise ValueError(
            f"Unknown preconditioner: {preconditioner}"
        )



def gmres(
    A,
    b,
    x0,
    x_true,
    monitor,
    preconditioner,
    restart=30
):
    """
    Restarted left-preconditioned GMRES(m).

    Solves

        A x = b

    using the left-preconditioned system

        M^{-1} A x = M^{-1} b.

    Convergence is assessed using the true residual

        r_k = b - A x_k.

    Parameters
    ----------
    A : scipy sparse matrix
        System matrix.

    b : ndarray
        Right-hand side.

    x0 : ndarray
        Initial guess.

    x_true : ndarray
        Exact solution, used for forward-error monitoring.

    monitor : SolverMonitor
        Common monitoring object.

    preconditioner : Preconditioner
        Object providing M^{-1}.

    restart : int, optional
        GMRES restart length. Default is 30.

    Returns
    -------
    x : ndarray
        Computed solution.
    """

    x = np.asarray(x0, dtype=float).copy()
    b = np.asarray(b, dtype=float)

    n = len(b)

    # ---------------------------------------------------------
    # Instrument A
    # ---------------------------------------------------------
    A_op = InstrumentedOperator(A, monitor)
    A_linear = A_op.as_linear_operator()

    # ---------------------------------------------------------
    # Initial true residual
    # ---------------------------------------------------------
    r = b - A_linear @ x

    # ---------------------------------------------------------
    # Record initial iterate using the already computed true residual
    # ---------------------------------------------------------
    #r = b - A_linear @ x_cycle 
    
    eta, forward_error, backward_error = monitor.record(
        x,
        iteration=0,
        r=r
    )

    if eta <= monitor.tol:
        monitor.status = "Converged"
        return x

    iteration = 0

    # =========================================================
    # Restart cycles
    # =========================================================
    while iteration < monitor.maxiter:

        

        # -----------------------------------------------------
        # Left-preconditioned residual
        #
        # z = M^{-1} r
        # -----------------------------------------------------
        z = preconditioner.apply(r)

        beta = np.linalg.norm(z)

        if beta == 0.0:
            monitor.status = "Converged"
            return x

        # -----------------------------------------------------
        # Arnoldi basis
        # -----------------------------------------------------
        V = np.zeros(
            (n, restart + 1),
            dtype=float
        )

        H = np.zeros(
            (restart + 1, restart),
            dtype=float
        )

        V[:, 0] = z / beta

        # -----------------------------------------------------
        # Right-hand side of least-squares problem
        # -----------------------------------------------------
        g = np.zeros(restart + 1, dtype=float)
        g[0] = beta

        # Approximation at beginning of cycle
        x_cycle = x.copy()

        # =====================================================
        # Arnoldi process
        # =====================================================
        for j in range(restart):

            if iteration >= monitor.maxiter:
                break

            # -------------------------------------------------
            # Apply A
            # -------------------------------------------------
            w = A_linear @ V[:, j]

            # -------------------------------------------------
            # Apply M^{-1}
            # -------------------------------------------------
            w = preconditioner.apply(w)

            # -------------------------------------------------
            # Modified Gram-Schmidt
            # -------------------------------------------------
            for i in range(j + 1):

                H[i, j] = np.dot(
                    V[:, i],
                    w
                )

                w -= H[i, j] * V[:, i]

            # -------------------------------------------------
            # Normalize next Arnoldi vector
            # -------------------------------------------------
            H[j + 1, j] = np.linalg.norm(w)

            # -------------------------------------------------
            # Arnoldi breakdown / exact Krylov termination
            # -------------------------------------------------
            if H[j + 1, j] != 0.0:

                V[:, j + 1] = (
                    w / H[j + 1, j]
                )

            # -------------------------------------------------
            # Solve GMRES least-squares problem
            # -------------------------------------------------
            H_j = H[:j + 2, :j + 1]
            g_j = g[:j + 2]

            y, *_ = np.linalg.lstsq(
                H_j,
                g_j,
                rcond=None
            )

            # -------------------------------------------------
            # Form current approximation
            # -------------------------------------------------
            x_cycle = (
                x
                + V[:, :j + 1] @ y
            )

            iteration += 1


            # -------------------------------------------------
            # Compute true residual
            # -------------------------------------------------
            r = b - A_linear @ x_cycle


            # -------------------------------------------------
            # Record true residual and errors
            # -------------------------------------------------
            eta, forward_error, backward_error = monitor.record(
                x_cycle,
                iteration,
                r=r
            )

            # -------------------------------------------------
            # Convergence test
            # -------------------------------------------------
            if eta <= monitor.tol:

                x = x_cycle

                monitor.status = "Converged"

                return x

            # -------------------------------------------------
            # Krylov breakdown
            # -------------------------------------------------
            if H[j + 1, j] == 0.0:

                x = x_cycle

                monitor.status = "Breakdown"

                monitor.breakdown_reason = (
                    "Arnoldi process terminated because "
                    "the next Krylov basis vector has zero norm."
                )

                return x

        # -----------------------------------------------------
        # Restart
        # -----------------------------------------------------
        x = x_cycle

    # =========================================================
    # Maximum iteration limit reached
    # =========================================================

    monitor.status = "Not Converged"

    return x

# =============================================================
# TEST / EXPERIMENT DRIVER
# =============================================================

if __name__ == "__main__":

    

    # ---------------------------------------------------------
    # Test matrix
    # ---------------------------------------------------------
    n = 100

    A = sp.diags(
        diagonals=[
            -1.0 * np.ones(n - 1),
             4.0 * np.ones(n),
            -1.0 * np.ones(n - 1)
        ],
        offsets=[-1, 0, 1],
        format="csr"
    )

    # ---------------------------------------------------------
    # Exact solution
    # ---------------------------------------------------------
    x_true = np.ones(n)

    # ---------------------------------------------------------
    # Right-hand side
    # ---------------------------------------------------------
    b = A @ x_true

    # ---------------------------------------------------------
    # Initial guess
    # ---------------------------------------------------------
    x0 = np.zeros(n)

    # ---------------------------------------------------------
    # Preconditioners to test
    # ---------------------------------------------------------
    preconditioners = [
        "none",
        "jacobi",
        "ilut"
    ]

    # ---------------------------------------------------------
    # Store results
    # ---------------------------------------------------------
    results = []

    # =========================================================
    # Run each preconditioner
    # =========================================================

    for prec_name in preconditioners:

        # ---------------------------------------------------------
            # Start memory tracking
            # ---------------------------------------------------------
        tracemalloc.start()

        print()
        print("=" * 60)
        print(f"RUNNING GMRES(30) WITH: {prec_name.upper()}")
        print("=" * 60)

        # -----------------------------------------------------
        # Create a fresh monitor for each experiment
        # -----------------------------------------------------
        monitor = SolverMonitor(
            A=A,
            b=b,
            x_true=x_true,
            tol=1e-8,
            maxiter=4000
        )

        # -----------------------------------------------------
        # Compute ||A||_2
        #
        # Only suitable for this small test matrix.
        # Do NOT use A.toarray() for large SuiteSparse matrices.
        # -----------------------------------------------------
        monitor.A_norm = spla.norm(
                        A,
                        ord=2
                    )

        # -----------------------------------------------------
        # Preconditioner setup
        # -----------------------------------------------------
        setup_start = time.perf_counter()

        M = build_preconditioner(
            A=A,
            monitor=monitor,
            preconditioner=prec_name,
            drop_tol=1e-4,
            fill_factor=10
        )

        monitor.setup_time = (
            time.perf_counter()
            - setup_start
        )

        # -----------------------------------------------------
        # Solve
        # -----------------------------------------------------
        solve_start = time.perf_counter()

        x = gmres(
            A=A,
            b=b,
            x0=x0,
            x_true=x_true,
            monitor=monitor,
            preconditioner=M,
            restart=30
        )

        monitor.solve_time = (
            time.perf_counter()
            - solve_start
        )

        monitor.total_time = (
            monitor.setup_time
            + monitor.solve_time
        )

        # -----------------------------------------------------
        # Update peak memory
        # -----------------------------------------------------
        monitor.update_peak_memory()

        current_memory, peak_memory = (
            tracemalloc.get_traced_memory()
        )

        monitor.peak_memory = max(
            monitor.peak_memory,
            peak_memory
        )

        tracemalloc.stop()

        # -----------------------------------------------------
        # Store results
        # -----------------------------------------------------
        results.append({
            "preconditioner": prec_name,
            "status": monitor.status,
            "iterations": len(monitor.iterations) - 1,
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
            "final_residual": (
                monitor.residual_history[-1]
            ),
            "final_forward_error": (
                monitor.forward_error_history[-1]
            ),
            "final_backward_error": (
                monitor.backward_error_history[-1]
            )
        })

        # -----------------------------------------------------
        # Print individual result
        # -----------------------------------------------------
        print()
        print(f"Status:              {monitor.status}")
        print(
            f"Iterations:          "
            f"{len(monitor.iterations) - 1}"
        )

        print(
            f"N_A:                 "
            f"{monitor.N_A}"
        )

        print(
            f"N_AT:                "
            f"{monitor.N_AT}"
        )

        print(
            f"N_M:                 "
            f"{monitor.N_M}"
        )

        print(
            f"N_MT:                "
            f"{monitor.N_MT}"
        )

        print(
            f"N_prec:              "
            f"{monitor.N_prec}"
        )

        print(
            f"Setup time:          "
            f"{monitor.setup_time:.6e} s"
        )

        print(
            f"Solve time:          "
            f"{monitor.solve_time:.6e} s"
        )

        print(
            f"Total time:          "
            f"{monitor.total_time:.6e} s"
        )

        print(
            f"Peak memory:         "
            f"{monitor.peak_memory / 1024**2:.6f} MB"
        )

        print(
            f"Final relative residual: "
            f"{monitor.residual_history[-1]:.6e}"
        )

        print(
            f"Final relative forward error: "
            f"{monitor.forward_error_history[-1]:.6e}"
        )

        print(
            f"Final backward error: "
            f"{monitor.backward_error_history[-1]:.6e}"
        )

        # -----------------------------------------------------
        # Iteration history
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

    # =========================================================
    # Stop memory tracking
    # =========================================================

    

    # =========================================================
    # COMPARISON SUMMARY
    # =========================================================

    print()
    print("=" * 145)
    print("GMRES(30) PRECONDITIONER COMPARISON")
    print("=" * 145)

    print(
    f"{'Prec.':<12}"
    f"{'Status':<15}"
    f"{'Iter.':>7}"
    f"{'N_A':>8}"
    f"{'N_AT':>8}"
    f"{'N_M':>8}"
    f"{'N_MT':>8}"
    f"{'N_prec':>9}"
    f"{'Setup(s)':>14}"
    f"{'Solve(s)':>14}"
    f"{'Total(s)':>14}"
    f"{'Memory(MB)':>14}"
)

    print("-" * 145)

    for result in results:

        print(
            f"{result['preconditioner']:<12}"
            f"{result['status']:<15}"
            f"{result['iterations']:>7}"
            f"{result['N_A']:>8}"
            f"{result['N_AT']:>8}"
            f"{result['N_M']:>8}"
            f"{result['N_MT']:>8}"
            f"{result['setup_time']:>14.6e}"
            f"{result['solve_time']:>14.6e}"
            f"{result['total_time']:>14.6e}"
            f"{result['peak_memory_MB']:>14.6f}"
        )