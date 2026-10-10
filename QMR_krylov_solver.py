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



def qmr(A, b, x0, x_true, monitor, preconditioner,
        tol=1e-8, maxiter=4000):
    """
    Quasi-Minimal Residual (QMR) method for solving

        A x = b

    for a nonsymmetric linear system.

    This implementation follows the standard two-term recurrence
    formulation of preconditioned QMR without look-ahead.

    Parameters
    ----------
    A : scipy.sparse matrix
        Coefficient matrix.

    b : ndarray
        Right-hand-side vector.

    x0 : ndarray
        Initial guess.

    x_true : ndarray
        Exact/reference solution used by SolverMonitor
        for computing the relative forward error.

    monitor : SolverMonitor
        Object used to record residuals, errors, and
        operation counts.

    preconditioner : Preconditioner
        Preconditioner object providing

            apply(x)           -> M^{-1} x
            apply_transpose(x) -> M^{-T} x

        The "none" case uses NoPreconditioner, so
        identity operations are not counted.

    tol : float, optional
        Relative true-residual convergence tolerance.

    maxiter : int, optional
        Maximum number of QMR iterations.

    Returns
    -------
    x : ndarray
        Computed approximate solution.

    Notes
    -----
    The implementation uses:

        r       = b - A x
        y       = M^{-1} v
        z_tld   = M^{-T} w

    with the standard QMR two-term recurrences.

    The true residual

        r_true = b - A x

    is recomputed at every iteration for convergence testing
    and performance monitoring. Therefore, this introduces
    an additional A-matrix-vector product per iteration beyond
    the matrix-vector product required by the QMR recurrence.

    The function returns only x because the solver status is
    stored in monitor.status.
    """

    # =========================================================
    # INITIAL SOLUTION
    # =========================================================

    x = np.asarray(x0, dtype=float).copy()
    b = np.asarray(b, dtype=float)

    # ---------------------------------------------------------
    # Norm of b
    # ---------------------------------------------------------

    b_norm = np.linalg.norm(b)

    if b_norm == 0.0:
        b_norm = 1.0

    # =========================================================
    # INSTRUMENTED MATRIX OPERATORS
    # =========================================================

    A_op = InstrumentedOperator(
        A,
        monitor,
        transpose=False
    )

    AT_op = InstrumentedOperator(
        A,
        monitor,
        transpose=True
    )

    A_linear = A_op.as_linear_operator()
    AT_linear = AT_op.as_linear_operator()

    # =========================================================
    # INITIAL RESIDUAL
    #
    # r_0 = b - A x_0
    # =========================================================

    r = b - A_linear @ x

    # ---------------------------------------------------------
    # Record iteration 0.
    #
    # The residual is supplied explicitly, so monitor.record()
    # does NOT perform another A-matrix-vector product.
    # ---------------------------------------------------------

    monitor.record(
        x,
        iteration=0,
        r=r
    )

    # ---------------------------------------------------------
    # Initial convergence check
    # ---------------------------------------------------------

    relative_residual = np.linalg.norm(r) / b_norm

    if relative_residual <= tol:

        monitor.status = "Converged"

        return x

    # =========================================================
    # QMR INITIALIZATION
    # =========================================================
    #
    # v_tld = r
    #
    # y = M^{-1} v_tld
    #
    # rho = ||y||
    #
    # w_tld = r
    #
    # xi = ||w_tld||
    #
    # Notice that xi is the norm of w_tld itself.
    # M^{-T} is applied later when constructing z_tld.
    # =========================================================

    v_tld = r.copy()

    y = preconditioner.apply(v_tld)

    rho = np.linalg.norm(y)

    w_tld = r.copy()

    xi = np.linalg.norm(w_tld)

    # =========================================================
    # INITIALIZE QMR SCALARS
    # =========================================================

    theta = 0.0
    gamma = 1.0
    eta = -1.0

    theta_1 = 0.0
    gamma_1 = 1.0

    rho_1 = 0.0

    delta = 0.0
    ep = 0.0
    beta = 0.0

    # =========================================================
    # SEARCH / UPDATE DIRECTIONS
    # =========================================================

    p = None
    q = None

    d = None
    s = None

    # =========================================================
    # MAIN QMR ITERATION
    # =========================================================

    for iteration in range(1, maxiter + 1):

        # -----------------------------------------------------
        # BREAKDOWN CHECK: rho
        # -----------------------------------------------------

        if rho == 0.0 or not np.isfinite(rho):

            print("QMR breakdown: rho = 0.")

            monitor.status = "Breakdown"

            return x

        # -----------------------------------------------------
        # BREAKDOWN CHECK: xi
        # -----------------------------------------------------

        if xi == 0.0 or not np.isfinite(xi):

            print("QMR breakdown: xi = 0.")

            monitor.status = "Breakdown"

            return x

        # =====================================================
        # NORMALIZE LANCZOS VECTORS
        # =====================================================
        #
        # v = v_tld / rho
        #
        # y = y / rho
        #
        # w = w_tld / xi
        # =====================================================

        v = v_tld / rho

        y = y / rho

        w = w_tld / xi

        # -----------------------------------------------------
        # Compute
        #
        # delta = w^T y
        # -----------------------------------------------------

        delta = np.dot(w, y)

        if delta == 0.0 or not np.isfinite(delta):

            print("QMR breakdown: delta = 0.")

            monitor.status = "Breakdown"

            return x

        # =====================================================
        # APPLY TRANSPOSE PRECONDITIONER
        # =====================================================
        #
        # z_tld = M^{-T} w
        #
        # This is the shadow preconditioned vector.
        # =====================================================

        z_tld = preconditioner.apply_transpose(w)

        # =====================================================
        # CONSTRUCT SEARCH DIRECTIONS
        # =====================================================

        if iteration == 1:

            # -------------------------------------------------
            # First iteration
            # -------------------------------------------------

            p = y.copy()

            q = z_tld.copy()

        else:

            # -------------------------------------------------
            # Subsequent iterations
            #
            # p_k = y_k
            #       - (xi_k delta_k / ep_{k-1}) p_{k-1}
            #
            # q_k = z_k
            #       - (rho_k delta_k / ep_{k-1}) q_{k-1}
            # -------------------------------------------------

            p = (
                y
                - (xi * delta / ep) * p
            )

            q = (
                z_tld
                - (rho * delta / ep) * q
            )

        # =====================================================
        # MATRIX-VECTOR PRODUCT
        #
        # p_tld = A p
        # =====================================================

        p_tld = A_linear @ p

        # =====================================================
        # COMPUTE EPSILON
        #
        # ep = q^T A p
        # =====================================================

        ep = np.dot(q, p_tld)

        if ep == 0.0 or not np.isfinite(ep):

            print("QMR breakdown: epsilon = 0.")

            monitor.status = "Breakdown"

            return x

        # =====================================================
        # COMPUTE BETA
        #
        # beta = ep / delta
        # =====================================================

        beta = ep / delta

        if beta == 0.0 or not np.isfinite(beta):

            print("QMR breakdown: beta = 0.")

            monitor.status = "Breakdown"

            return x

        # =====================================================
        # UPDATE LANCZOS VECTOR
        #
        # v_tld = p_tld - beta v
        # =====================================================

        v_tld = p_tld - beta * v

        # =====================================================
        # APPLY LEFT PRECONDITIONER
        #
        # y = M^{-1} v_tld
        # =====================================================

        y = preconditioner.apply(v_tld)

        # -----------------------------------------------------
        # Store previous rho before updating it.
        # -----------------------------------------------------

        rho_1 = rho

        rho = np.linalg.norm(y)

        # =====================================================
        # UPDATE SHADOW LANCZOS VECTOR
        #
        # w_tld = A^T q - beta w
        # =====================================================

        w_tld = AT_linear @ q

        w_tld = w_tld - beta * w

        # -----------------------------------------------------
        # Update xi.
        #
        # IMPORTANT:
        # xi is the norm of w_tld itself.
        # No M^{-T} application occurs here.
        # -----------------------------------------------------

        xi = np.linalg.norm(w_tld)

        # =====================================================
        # STORE PREVIOUS QMR SCALARS
        # =====================================================

        gamma_1 = gamma

        theta_1 = theta

        # =====================================================
        # UPDATE THETA
        #
        # theta = rho / (gamma_1 beta)
        # =====================================================

        denominator = gamma_1 * beta

        if denominator == 0.0 or not np.isfinite(denominator):

            print("QMR breakdown: gamma_1 * beta = 0.")

            monitor.status = "Breakdown"

            return x

        theta = rho / denominator

        # =====================================================
        # UPDATE GAMMA
        #
        # gamma = 1 / sqrt(1 + theta^2)
        # =====================================================

        gamma = 1.0 / np.sqrt(
            1.0 + theta**2
        )

        if gamma == 0.0 or not np.isfinite(gamma):

            print("QMR breakdown: gamma = 0.")

            monitor.status = "Breakdown"

            return x

        # =====================================================
        # UPDATE ETA
        #
        # eta =
        #   -eta * rho_1 * gamma^2
        #   --------------------------------
        #   beta * gamma_1^2
        # =====================================================

        denominator = (
            beta * gamma_1**2
        )

        if denominator == 0.0 or not np.isfinite(denominator):

            print("QMR breakdown: beta * gamma_1^2 = 0.")

            monitor.status = "Breakdown"

            return x

        eta = (
            -eta
            * rho_1
            * gamma**2
            / denominator
        )

        if not np.isfinite(eta):

            print("QMR breakdown: eta is not finite.")

            monitor.status = "Breakdown"

            return x

        # =====================================================
        # UPDATE SOLUTION AND RESIDUAL DIRECTIONS
        # =====================================================

        if iteration == 1:

            # -------------------------------------------------
            # First iteration
            # -------------------------------------------------

            d = eta * p

            s = eta * p_tld

        else:

            # -------------------------------------------------
            # Subsequent iterations
            #
            # tmp = (theta_1 gamma)^2
            # -------------------------------------------------

            tmp = (
                theta_1 * gamma
            )**2

            d = (
                eta * p
                + tmp * d
            )

            s = (
                eta * p_tld
                + tmp * s
            )

        # =====================================================
        # UPDATE APPROXIMATE SOLUTION
        #
        # x_k = x_{k-1} + d_k
        # =====================================================

        x = x + d

        # =====================================================
        # UPDATE RECURSIVE RESIDUAL
        #
        # r_k = r_{k-1} - s_k
        #
        # This residual is maintained by QMR internally.
        # =====================================================

        r = r - s

        # =====================================================
        # TRUE RESIDUAL
        #
        # r_true = b - A x_k
        #
        # This is deliberately recomputed for:
        #
        # 1. convergence testing,
        # 2. residual-history recording,
        # 3. finite-precision verification.
        #
        # Consequently, this contributes one additional
        # A-matrix-vector product per iteration.
        # =====================================================

        true_r = b - A_linear @ x

        # =====================================================
        # RECORD ITERATION
        # =====================================================

        monitor.record(
            x,
            iteration=iteration,
            r=true_r
        )

        # =====================================================
        # RELATIVE TRUE RESIDUAL
        # =====================================================

        relative_residual = (
            np.linalg.norm(true_r)
            / b_norm
        )

        # =====================================================
        # CONVERGENCE TEST
        # =====================================================

        if relative_residual <= tol:

            monitor.status = "Converged"

            return x

    # =========================================================
    # MAXIMUM NUMBER OF ITERATIONS REACHED
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
        print(f"RUNNING QMR WITH: {prec_name.upper()}")
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

        x = qmr(
            A=A,
            b=b,
            x0=x0,
            x_true=x_true,
            monitor=monitor,
            preconditioner=M,
            maxiter=4000
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
    print("QMR PRECONDITIONER COMPARISON")
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
            f"{result['N_prec']:>9}"
            f"{result['setup_time']:>14.6e}"
            f"{result['solve_time']:>14.6e}"
            f"{result['total_time']:>14.6e}"
            f"{result['peak_memory_MB']:>14.6f}"
        )