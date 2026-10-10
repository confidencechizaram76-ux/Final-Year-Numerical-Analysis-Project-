from builtins import print
import os
import json
import numpy as np
import scipy.sparse as sp


# ============================================================
# 1. Construct the CDR matrix
# ============================================================

def cdr_matrix(n, Pe, epsilon=1.0, sigma=1.0):
    """
    Construct the sparse matrix for the 2-D convection-diffusion-
    reaction equation

        -epsilon * (u_xx + u_yy)
        + beta_x * u_x
        + beta_y * u_y
        + sigma * u = f

    on (0,1)^2 with homogeneous Dirichlet boundary conditions.

    Parameters
    ----------
    n : int
        Number of interior grid points in each coordinate direction.

    Pe : float
        Grid Peclet number:
            Pe = |beta| h / (2 epsilon)

    epsilon : float
        Diffusion coefficient.

    sigma : float
        Reaction coefficient.

    Returns
    -------
    A : scipy.sparse.csr_matrix
        Sparse system matrix.

    h : float
        Grid spacing.

    beta_x, beta_y : float
        Convection coefficients.
    """

    # Grid spacing
    h = 1.0 / (n + 1)

    # Diagonal convection direction
    d_x = 1.0 / np.sqrt(2.0)
    d_y = 1.0 / np.sqrt(2.0)

    # Choose |beta| so that the prescribed grid Peclet number
    # is obtained exactly.
    beta_magnitude = (2.0 * epsilon * Pe) / h

    beta_x = beta_magnitude * d_x
    beta_y = beta_magnitude * d_y

    # --------------------------------------------------------
    # Second derivative matrix
    # --------------------------------------------------------

    D2 = sp.diags(
        [
            np.ones(n - 1),
            -2.0 * np.ones(n),
            np.ones(n - 1)
        ],
        [-1, 0, 1],
        shape=(n, n),
        format="csr"
    ) / h**2

    # --------------------------------------------------------
    # First derivative matrix
    # --------------------------------------------------------

    D1 = sp.diags(
        [
            -np.ones(n - 1),
            np.ones(n - 1)
        ],
        [-1, 1],
        shape=(n, n),
        format="csr"
    ) / (2.0 * h)

    # Identity matrix
    I = sp.eye(n, format="csr")

    # 2-D operators
    Laplacian = sp.kron(I, D2) + sp.kron(D2, I)
    Dx = sp.kron(I, D1)
    Dy = sp.kron(D1, I)

    # Total system dimension
    N = n * n

    # CDR matrix
    A = (
        -epsilon * Laplacian
        + beta_x * Dx
        + beta_y * Dy
        + sigma * sp.eye(N, format="csr")
    )

    return A.tocsr(), h, beta_x, beta_y


# ============================================================
# 2. Manufactured exact solution and right-hand side
# ============================================================

def cdr_exact_solution_and_rhs(
    n,
    Pe,
    epsilon=1.0,
    sigma=1.0
):
    """
    Generate the manufactured exact solution

        u(x,y) = sin(pi*x) sin(pi*y)

    and the corresponding forcing function f(x,y).

    Returns
    -------
    u_exact : ndarray
        Exact solution sampled at the interior grid points.

    b : ndarray
        Right-hand-side vector obtained by sampling f(x,y).

    h : float
        Grid spacing.

    beta_x, beta_y : float
        Convection coefficients.
    """

    # Grid spacing
    h = 1.0 / (n + 1)

    # Interior grid points
    x = np.arange(1, n + 1) * h
    y = np.arange(1, n + 1) * h

    X, Y = np.meshgrid(x, y, indexing="ij")

    # Same convection coefficients used in A
    d_x = 1.0 / np.sqrt(2.0)
    d_y = 1.0 / np.sqrt(2.0)

    beta_magnitude = (2.0 * epsilon * Pe) / h

    beta_x = beta_magnitude * d_x
    beta_y = beta_magnitude * d_y

    # --------------------------------------------------------
    # Manufactured exact solution
    # --------------------------------------------------------

    u_exact = (
        np.sin(np.pi * X)
        * np.sin(np.pi * Y)
    )

    # --------------------------------------------------------
    # Corresponding forcing function
    #
    # f =
    # 2 epsilon pi^2 sin(pi x) sin(pi y)
    # + beta_x pi cos(pi x) sin(pi y)
    # + beta_y pi sin(pi x) cos(pi y)
    # + sigma sin(pi x) sin(pi y)
    # --------------------------------------------------------

    f = (
        2.0 * epsilon * np.pi**2 * u_exact
        + beta_x
        * np.pi
        * np.cos(np.pi * X)
        * np.sin(np.pi * Y)
        + beta_y
        * np.pi
        * np.sin(np.pi * X)
        * np.cos(np.pi * Y)
        + sigma * u_exact
    )

    # Convert the 2-D arrays into vectors
    u_exact = u_exact.ravel()
    b = f.ravel()

    return u_exact, b, h, beta_x, beta_y


# ============================================================
# 3. Save one CDR test case
# ============================================================

def save_cdr_case(
    output_folder,
    filename_base,
    experiment,
    n,
    Pe,
    epsilon=1.0,
    sigma=1.0
):
    """
    Generate and save one complete CDR test case.

    Files saved:
        A       -> sparse .npz
        b       -> .npy
        u_exact -> .npy
        metadata -> .json
    """

    # --------------------------------------------------------
    # Construct A
    # --------------------------------------------------------

    A, h, beta_x, beta_y = cdr_matrix(
        n=n,
        Pe=Pe,
        epsilon=epsilon,
        sigma=sigma
    )

    # --------------------------------------------------------
    # Construct u_exact and b
    # --------------------------------------------------------

    u_exact, b, _, _, _ = cdr_exact_solution_and_rhs(
        n=n,
        Pe=Pe,
        epsilon=epsilon,
        sigma=sigma
    )

    # System dimension
    N = n * n

    # --------------------------------------------------------
    # Save matrix
    # --------------------------------------------------------

    matrix_path = os.path.join(
        output_folder,
        filename_base + ".npz"
    )

    sp.save_npz(matrix_path, A)

    # --------------------------------------------------------
    # Save right-hand side
    # --------------------------------------------------------

    b_path = os.path.join(
        output_folder,
        filename_base + "_b.npy"
    )

    np.save(b_path, b)

    # --------------------------------------------------------
    # Save exact solution
    # --------------------------------------------------------

    u_path = os.path.join(
        output_folder,
        filename_base + "_uexact.npy"
    )

    np.save(u_path, u_exact)

    # --------------------------------------------------------
    # Calculate matrix properties
    # --------------------------------------------------------

    nnz = A.nnz

    density = nnz / (N * N)

    relative_asymmetry = (
        sp.linalg.norm(A - A.T)
        / sp.linalg.norm(A)
    )

    # --------------------------------------------------------
    # Metadata
    # --------------------------------------------------------

    metadata = {
        "experiment": experiment,

        "PDE": (
            "-epsilon*(u_xx + u_yy) "
            "+ beta_x*u_x + beta_y*u_y "
            "+ sigma*u = f"
        ),

        "domain": "(0,1)^2",

        "boundary_condition":
            "homogeneous Dirichlet",

        "discretization":
            "central finite differences",

        "manufactured_solution":
            "sin(pi*x)*sin(pi*y)",

        "n": n,
        "grid_spacing_h": h,
        "system_dimension_N": N,

        "epsilon": epsilon,
        "sigma": sigma,

        "grid_Peclet_number": Pe,

        "beta_x": beta_x,
        "beta_y": beta_y,

        "rows": A.shape[0],
        "columns": A.shape[1],

        "nnz": nnz,

        "density": density,

        "relative_asymmetry":
            relative_asymmetry
    }

    metadata_path = os.path.join(
        output_folder,
        filename_base + ".json"
    )

    with open(metadata_path, "w") as file:
        json.dump(
            metadata,
            file,
            indent=4
        )

    # --------------------------------------------------------
    # Confirmation
    # --------------------------------------------------------

    print(
        f"Saved {filename_base}: "
        f"N={N}, nnz={nnz}, Pe={Pe}"
    )


# ============================================================
# 4. Main generation program
# ============================================================

def main():

    # --------------------------------------------------------
    # General parameters
    # --------------------------------------------------------

    epsilon = 1.0
    sigma = 1.0

    # --------------------------------------------------------
    # Output folders
    # --------------------------------------------------------

    base_folder = "CDR_matrices"

    dimension_folder = os.path.join(
        base_folder,
        "dimension_experiment"
    )

    Pe_folder = os.path.join(
        base_folder,
        "Pe_experiment"
    )

    os.makedirs(
        dimension_folder,
        exist_ok=True
    )

    os.makedirs(
        Pe_folder,
        exist_ok=True
    )

    # ========================================================
    # EXPERIMENT 1
    #
    # Vary dimension while keeping Pe fixed
    # ========================================================

    dimension_values = [
        50,
        100,
        200,
        300,
        400,
        500,
        600
    ]

    fixed_Pe = 1.0

    print("\n========================================")
    print("CDR DIMENSION EXPERIMENT")
    print("========================================")

    for n in dimension_values:

        filename = (
            f"CDR_n{n:03d}_Pe1"
        )

        save_cdr_case(
            output_folder=dimension_folder,
            filename_base=filename,
            experiment="dimension_study",
            n=n,
            Pe=fixed_Pe,
            epsilon=epsilon,
            sigma=sigma
        )

    # ========================================================
    # EXPERIMENT 2
    #
    # Vary Pe while keeping dimension fixed
    # ========================================================

    fixed_n = 400

    Pe_values = [
        0.01,
        0.1,
        1.0,
        10.0,
        100.0
    ]

    print("\n========================================")
    print("CDR PECLET NUMBER EXPERIMENT")
    print("========================================")

    for Pe in Pe_values:

        # Replace decimal point in filename
        # to avoid awkward filenames.
        Pe_string = str(Pe).replace(".", "p")

        filename = (
            f"CDR_n{fixed_n:03d}_Pe{Pe_string}"
        )

        save_cdr_case(
            output_folder=Pe_folder,
            filename_base=filename,
            experiment="Pe_study",
            n=fixed_n,
            Pe=Pe,
            epsilon=epsilon,
            sigma=sigma
        )

    print("\n========================================")
    print("ALL CDR CASES GENERATED SUCCESSFULLY")
    print("========================================")


# ============================================================
# 5. Run the program
# ============================================================

if __name__ == "__main__":
    main()