import numpy as np
from scipy.io import mmread
from scipy.sparse import csr_matrix, identity
from scipy.sparse.linalg import norm as sparse_norm


# ============================================================
# 1. Matrix files
# ============================================================

matrix_files = {
    "swang1": "application_matrices/swang1.mtx",
    "psmigr_3": "application_matrices/psmigr_3.mtx",
    "psmigr_2": "application_matrices/psmigr_2.mtx",
    "utm3060": "application_matrices/utm3060.mtx",
    "rdist2": "application_matrices/rdist2.mtx",
    "ex8": "application_matrices/ex8.mtx"
}


# ============================================================
# 2. Expected matrix information from the project table
# ============================================================

expected = {
    "swang1": {
        "n": 3169,
        "nnz": 20841,
        "kappa": 5.990451
    },

    "psmigr_3": {
        "n": 3140,
        "nnz": 543160,
        "kappa": 1.328952e2
    },

    "psmigr_2": {
        "n": 3140,
        "nnz": 540022,
        "kappa": 1.720280e4
    },

    "utm3060": {
        "n": 3060,
        "nnz": 42211,
        "kappa": 1.927129e7
    },

    "rdist2": {
        "n": 3198,
        "nnz": 56834,
        "kappa": 1.367824e7
    },

    "ex8": {
        "n": 3096,
        "nnz": 90841,
        "kappa": 4.483542e10
    }
}


# ============================================================
# 3. Load matrices
# ============================================================

matrices = {}

for name, filename in matrix_files.items():

    A = mmread(filename)

    # Convert to CSR sparse format
    A = csr_matrix(A)

    # Remove explicitly stored zeros
    A.eliminate_zeros()

    matrices[name] = A


# ============================================================
# 4. Numerical symmetry
# ============================================================

def numerical_symmetry_measure(A):
    """
    Computes

        ||A - A^T||_F / ||A||_F

    A value close to zero indicates numerical symmetry.
    """

    numerator = sparse_norm(A - A.T, ord='fro')
    denominator = sparse_norm(A, ord='fro')

    return numerator / denominator


# ============================================================
# 5. Nonnormality measure
# ============================================================

def nonnormality_measure(A):
    """
    Computes the scale-invariant Henrici-type measure

        eta(A) = ||A^T A - A A^T||_F / ||A||_F^2

    for real matrices.
    """

    norm_A = sparse_norm(A, ord='fro')

    AT_A = A.T @ A
    A_AT = A @ A.T

    numerator = sparse_norm(AT_A - A_AT, ord='fro')

    return numerator / (norm_A ** 2)


# ============================================================
# 6. Condition number
# ============================================================

def condition_number_dense(A):
    """
    Computes the 2-norm condition number.

    This converts A to a dense matrix, so it should only be
    used for matrices of manageable size.
    """

    A_dense = A.toarray()

    return np.linalg.cond(A_dense, p=2)


# ============================================================
# 7. Validate all matrices
# ============================================================

print("=" * 90)
print("VALIDATION OF SUITESPARSE APPLICATION MATRICES")
print("=" * 90)

print()

for name, A in matrices.items():

    n = A.shape[0]
    nnz = A.nnz

    symmetry = numerical_symmetry_measure(A)
    eta = nonnormality_measure(A)

    print(f"Matrix: {name}")
    print("-" * 90)

    print(f"Dimension:              {A.shape}")
    print(f"Number of nonzeros:     {nnz}")
    print(f"Expected dimension:     {expected[name]['n']}")
    print(f"Expected nnz:           {expected[name]['nnz']}")

    # Dimension check
    dimension_ok = (
        A.shape[0] == expected[name]["n"]
        and A.shape[1] == expected[name]["n"]
    )

    # nnz check
    nnz_ok = nnz == expected[name]["nnz"]

    print(f"Dimension check:        {'PASS' if dimension_ok else 'CHECK'}")
    print(f"nnz check:              {'PASS' if nnz_ok else 'CHECK'}")

    print(f"Numerical symmetry:     {symmetry:.6e}")
    print(f"Nonnormality eta(A):    {eta:.6e}")

    print(f"Table kappa(A):         {expected[name]['kappa']:.6e}")

    print()


# ============================================================
# 8. Summary table
# ============================================================

print("=" * 90)
print("SUMMARY")
print("=" * 90)

print(
    f"{'Matrix':<12}"
    f"{'n':>8}"
    f"{'nnz':>12}"
    f"{'Sparsity (%)':>15}"
    f"{'Symmetry':>15}"
    f"{'eta(A)':>15}"
)

print("-" * 90)

for name, A in matrices.items():

    n = A.shape[0]
    nnz = A.nnz

    sparsity = 100 * nnz / (n ** 2)

    symmetry = numerical_symmetry_measure(A)
    eta = nonnormality_measure(A)

    print(
        f"{name:<12}"
        f"{n:>8}"
        f"{nnz:>12}"
        f"{sparsity:>15.6f}"
        f"{symmetry:>15.6e}"
        f"{eta:>15.6e}"
    )