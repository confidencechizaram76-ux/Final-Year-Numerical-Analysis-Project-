import numpy as np
from scipy.io import mmread
from scipy.sparse import csr_matrix
import os


# ============================================================
# 1. Folder containing the SuiteSparse .mtx files
# ============================================================

MATRIX_FOLDER = r"C:\Users\Administrator\.vscode-insiders\Final Year Project\application_matrices"


# ============================================================
# 2. Names of the six application matrices
# ============================================================

MATRIX_NAMES = [
    "swang1",
    "psmigr_2",
    "psmigr_3",
    "ex8",
    "rdist2",
    "utm3060"
]


# ============================================================
# 3. Output folder
# ============================================================

OUTPUT_FOLDER = os.path.join(MATRIX_FOLDER, "prepared_application_matrices")

os.makedirs(OUTPUT_FOLDER, exist_ok=True)


# ============================================================
# 4. Prepare each matrix
# ============================================================

for name in MATRIX_NAMES:

    print("\n" + "=" * 70)
    print(f"Preparing matrix: {name}")
    print("=" * 70)

    # --------------------------------------------------------
    # Locate the matrix file
    # --------------------------------------------------------

    matrix_file = os.path.join(MATRIX_FOLDER, name + ".mtx")

    if not os.path.exists(matrix_file):
        print(f"ERROR: File not found:")
        print(matrix_file)
        continue

    # --------------------------------------------------------
    # Load matrix
    # --------------------------------------------------------

    try:
        A = mmread(matrix_file)
        A = csr_matrix(A)

    except Exception as e:
        print(f"ERROR loading {name}: {e}")
        continue

    # --------------------------------------------------------
    # Basic matrix information
    # --------------------------------------------------------

    n, m = A.shape
    nnz = A.nnz

    if n != m:
        print(f"WARNING: {name} is not square.")
        print(f"Shape = {A.shape}")
        continue

    sparsity = (nnz / (n * n)) * 100

    print(f"Matrix        : {name}")
    print(f"Dimension     : {n} x {n}")
    print(f"Nonzeros      : {nnz}")
    print(f"Sparsity      : {sparsity:.6f}%")

    # --------------------------------------------------------
    # Construct the known exact solution
    #
    # x_true = [1, 1, ..., 1]^T
    # --------------------------------------------------------

    x_true = np.ones(n, dtype=float)

    # --------------------------------------------------------
    # Construct RHS
    #
    # b = A x_true
    # --------------------------------------------------------

    b = A @ x_true

    # --------------------------------------------------------
    # Initial guess
    #
    # x0 = [0, 0, ..., 0]^T
    # --------------------------------------------------------

    x0 = np.zeros(n, dtype=float)

    # --------------------------------------------------------
    # Verify that b is finite
    # --------------------------------------------------------

    if not np.all(np.isfinite(b)):
        print("WARNING: b contains NaN or Inf values.")

    # --------------------------------------------------------
    # Verify consistency:
    #
    # A x_true - b should be zero.
    # --------------------------------------------------------

    consistency_error = np.linalg.norm(A @ x_true - b)

    print(f"||Ax_true - b||_2 = {consistency_error:.6e}")

    # --------------------------------------------------------
    # Norm of b
    # --------------------------------------------------------

    b_norm = np.linalg.norm(b)

    print(f"||b||_2           = {b_norm:.6e}")

    # --------------------------------------------------------
    # Save the prepared data
    # --------------------------------------------------------

    output_file = os.path.join(
        OUTPUT_FOLDER,
        name + "_prepared.npz"
    )

    np.savez(
        output_file,
        x_true=x_true,
        b=b,
        x0=x0
    )

    print(f"Saved to:")
    print(output_file)


# ============================================================
# 5. Finished
# ============================================================

print("\n" + "=" * 70)
print("PREPARATION COMPLETE")
print("=" * 70)

print(f"\nPrepared matrices were saved in:")
print(OUTPUT_FOLDER)