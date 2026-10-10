# ================================================================
# FIXED-EIGENVALUE NONNORMAL MATRIX FAMILY
# Block construction with p = 5
#
# Project:
# Conditioning, Nonnormality, and Preconditioning in Sparse
# Nonsymmetric Linear Systems
#
# Matrix family:
#     A_alpha = V_alpha D V_alpha^{-1}
#
# V_alpha = I + alpha*S
#
# S is a 5 x 5 block upper-shift matrix with B on the first
# block superdiagonal.
#
# n = 1800
# p = 5
# block size = 360
# beta = 0.5
#
# alpha = 0.0, 0.1, ..., 1.0
# ================================================================

import os
import json
import numpy as np
import pandas as pd
import scipy.sparse as sp
import matplotlib.pyplot as plt

# ---------------------------------------------------------------
# 1. USER SETTINGS
# ---------------------------------------------------------------

N = 1800
P = 5
BETA = 0.5

ALPHA_VALUES = np.round(np.arange(0.0, 110, 10), 100)

# Output directory
OUTPUT_DIR = "fixed_eigenvalue_nonnormal_family_p5_n1800"

os.makedirs(OUTPUT_DIR, exist_ok=True)

# ---------------------------------------------------------------
# 2. CHECK DIMENSION
# ---------------------------------------------------------------

if N % P != 0:
    raise ValueError("N must be exactly divisible by P.")

BLOCK_SIZE = N // P

print("=" * 70)
print("FIXED-EIGENVALUE NONNORMAL MATRIX FAMILY")
print("=" * 70)

print(f"Matrix dimension n       : {N}")
print(f"Number of blocks p       : {P}")
print(f"Block size m             : {BLOCK_SIZE}")
print(f"Beta                     : {BETA}")
print(f"Alpha values             : {ALPHA_VALUES}")
print(f"Output directory         : {OUTPUT_DIR}")
print("=" * 70)


# ---------------------------------------------------------------
# 3. CONSTRUCT THE WITHIN-BLOCK MATRIX B
# ---------------------------------------------------------------

def construct_B(m, beta):
    """
    Construct

        B = I_m + beta * J_m,

    where J_m is the strict upper-shift matrix.
    """

    I_m = sp.eye(m, format="csr")

    upper_shift = sp.diags(
        np.ones(m - 1),
        offsets=1,
        shape=(m, m),
        format="csr"
    )

    B = I_m + beta * upper_shift

    return B


# ---------------------------------------------------------------
# 4. CONSTRUCT THE BLOCK-SHIFT MATRIX S
# ---------------------------------------------------------------

def construct_S(p, m, B):
    """
    Construct the p-by-p block matrix

        S = [ 0 B 0 ... 0
              0 0 B ... 0
              ...
              0 0 0 ... B
              0 0 0 ... 0 ]

    where every block is m-by-m.
    """

    Z = sp.csr_matrix((m, m))

    blocks = [
        [Z for _ in range(p)]
        for _ in range(p)
    ]

    for i in range(p - 1):
        blocks[i][i + 1] = B

    S = sp.bmat(blocks, format="csr")

    return S


# ---------------------------------------------------------------
# 5. CONSTRUCT V_alpha AND V_alpha^{-1}
# ---------------------------------------------------------------

def construct_V_and_inverse(alpha, S, N, P):
    """
    Construct

        V_alpha = I + alpha*S

    and, since S^P = 0,

        V_alpha^{-1}
          = I - alpha*S + alpha^2*S^2
            - ... + (-alpha)^(P-1) S^(P-1).
    """

    I = sp.eye(N, format="csr")

    # V_alpha
    V = I + alpha * S

    # Finite Neumann series for the inverse
    V_inv = I.copy()

    S_power = I.copy()

    for k in range(1, P):
        S_power = S_power @ S

        V_inv = (
            V_inv
            + ((-alpha) ** k) * S_power
        )

    return V.tocsr(), V_inv.tocsr()


# ---------------------------------------------------------------
# 6. CONSTRUCT THE FIXED EIGENVALUE MATRIX D
# ---------------------------------------------------------------

def construct_D(N):
    """
    Construct

        D = diag(lambda_1, ..., lambda_N)

    with eigenvalues uniformly distributed over [1, 10].
    """

    eigenvalues = np.linspace(
        1.0,
        10.0,
        N
    )

    D = sp.diags(
        eigenvalues,
        offsets=0,
        format="csr"
    )

    return D, eigenvalues


# ---------------------------------------------------------------
# 7. CONSTRUCT A_alpha
# ---------------------------------------------------------------

def construct_A(alpha, S, D, N, P):
    """
    Construct

        A_alpha = V_alpha D V_alpha^{-1}.
    """

    V, V_inv = construct_V_and_inverse(
        alpha,
        S,
        N,
        P
    )

    A = V @ D @ V_inv

    # Remove numerical entries that are essentially zero.
    A = A.tocsr()
    A.eliminate_zeros()

    return A, V, V_inv


# ---------------------------------------------------------------
# 8. FROBENIUS NORM OF A SPARSE MATRIX
# ---------------------------------------------------------------

def sparse_frobenius_norm(A):
    """
    Frobenius norm of a sparse matrix.
    """

    return np.sqrt(
        np.sum(
            np.abs(A.data) ** 2
        )
    )


# ---------------------------------------------------------------
# 9. NONNORMALITY MEASURE
# ---------------------------------------------------------------

def nonnormality_measure(A):
    """
    Compute

        eta(A) =
        ||A* A - A A*||_F / ||A||_F^2

    using sparse matrix operations.
    """

    AH = A.getH()

    AH_A = AH @ A
    A_AH = A @ AH

    commutator = AH_A - A_AH

    numerator = sparse_frobenius_norm(
        commutator
    )

    denominator = (
        sparse_frobenius_norm(A) ** 2
    )

    eta = numerator / denominator

    return float(eta)


# ---------------------------------------------------------------
# 10. 2-NORM CONDITION NUMBER
# ---------------------------------------------------------------

def condition_number_2(A):
    """
    Compute the 2-norm condition number.

    A dense SVD is used because the matrices are only 1800 x 1800.
    """

    A_dense = A.toarray()

    singular_values = np.linalg.svd(
        A_dense,
        compute_uv=False
    )

    sigma_max = singular_values[0]
    sigma_min = singular_values[-1]

    return float(
        sigma_max / sigma_min
    )


# ---------------------------------------------------------------
# 11. EIGENVALUE CHECK
# ---------------------------------------------------------------

def eigenvalue_check(A, expected_eigenvalues):
    """
    Verify that the eigenvalues remain equal to those prescribed
    in D.

    For this verification, the matrix is converted to dense form.
    """

    A_dense = A.toarray()

    computed_eigenvalues = np.linalg.eigvals(
        A_dense
    )

    computed_eigenvalues = np.sort(
        np.real_if_close(computed_eigenvalues)
    )

    expected = np.sort(
        expected_eigenvalues
    )

    max_error = np.max(
        np.abs(
            computed_eigenvalues - expected
        )
    )

    return float(max_error)


# ---------------------------------------------------------------
# 12. CONSTRUCT BASIC MATRICES
# ---------------------------------------------------------------

print("\nConstructing B...")

B = construct_B(
    BLOCK_SIZE,
    BETA
)

print(
    f"B dimensions : {B.shape}"
)

print(
    f"nnz(B)       : {B.nnz}"
)

print("\nConstructing S...")

S = construct_S(
    P,
    BLOCK_SIZE,
    B
)

print(
    f"S dimensions : {S.shape}"
)

print(
    f"nnz(S)       : {S.nnz}"
)

# Verify nilpotency
S_power = S.copy()

for _ in range(P - 1):
    pass

S_P = S
for _ in range(P - 1):
    S_P = S_P @ S

print(
    f"nnz(S^{P})      : {S_P.nnz}"
)

if S_P.nnz != 0:
    raise RuntimeError(
        f"S^{P} is not numerically zero."
    )

print(
    f"\nConfirmed: S^{P} = 0"
)

# Construct fixed D
print("\nConstructing D...")

D, expected_eigenvalues = construct_D(N)

print(
    f"D dimensions : {D.shape}"
)

print(
    f"Eigenvalue range : "
    f"[{expected_eigenvalues[0]:.1f}, "
    f"{expected_eigenvalues[-1]:.1f}]"
)

print(
    f"kappa_2(D)      : "
    f"{expected_eigenvalues[-1] / expected_eigenvalues[0]:.6f}"
)


# ---------------------------------------------------------------
# 13. STORAGE FOR RESULTS
# ---------------------------------------------------------------

results = []


# ---------------------------------------------------------------
# 14. GENERATE THE COMPLETE MATRIX FAMILY
# ---------------------------------------------------------------

for alpha in ALPHA_VALUES:

    print("\n" + "-" * 70)
    print(
        f"Generating matrix for alpha = {alpha:.1f}"
    )
    print("-" * 70)

    # Construct A, V and V^{-1}
    A, V, V_inv = construct_A(
        alpha,
        S,
        D,
        N,
        P
    )


    # -----------------------------------------------------------
    # Construct exact solution and right-hand side
    # -----------------------------------------------------------

    x_exact = np.ones(N, dtype=float)

    b = A @ x_exact


    # -----------------------------------------------------------
    # Matrix condition number
    # -----------------------------------------------------------

    print("Computing kappa_2(V_alpha)...")

    kappa_V = condition_number_2(
        V
    )

    print(
        f"kappa_2(V_alpha) = "
        f"{kappa_V:.8f}"
    )

    # -----------------------------------------------------------
    # Nonnormality
    # -----------------------------------------------------------

    print("Computing nonnormality measure...")

    eta = nonnormality_measure(
        A
    )

    print(
        f"eta(A_alpha) = "
        f"{eta:.10f}"
    )

    # -----------------------------------------------------------
    # Matrix condition number
    # -----------------------------------------------------------

    print("Computing kappa_2(A_alpha)...")

    kappa_A = condition_number_2(
        A
    )

    print(
        f"kappa_2(A_alpha) = "
        f"{kappa_A:.8f}"
    )

    # -----------------------------------------------------------
    # Eigenvalue verification
    # -----------------------------------------------------------

    print("Checking eigenvalues...")

    eigenvalue_error = eigenvalue_check(
        A,
        expected_eigenvalues
    )

    print(
        f"Maximum eigenvalue error = "
        f"{eigenvalue_error:.4e}"
    )

    # -----------------------------------------------------------
    # Sparsity information
    # -----------------------------------------------------------

    nnz_A = A.nnz

    density_A = (
        nnz_A / (N * N)
    )

    sparsity_A = (
        1.0 - density_A
    )

    # -----------------------------------------------------------
    # Save A_alpha as sparse .npz
    # -----------------------------------------------------------

    alpha_label = f"{alpha:.1f}"

    matrix_filename = (
        f"A_alpha_{alpha_label}.npz"
    )

    matrix_path = os.path.join(
        OUTPUT_DIR,
        matrix_filename
    )

    
    sp.save_npz(
        matrix_path,
        A.tocsr()
        )

    print(
        f"Saved sparse matrix: {matrix_path}"
    )

    # -----------------------------------------------------------
# Save right-hand-side vector b
# -----------------------------------------------------------

    b_filename = f"A_alpha_{alpha_label}_b.npy"

    b_path = os.path.join(
    OUTPUT_DIR,
    b_filename
    )

    np.save(
    b_path,
    b
    )

    print(f"Saved RHS vector: {b_path}")


# -----------------------------------------------------------
# Save exact solution u_exact
# -----------------------------------------------------------

    uexact_filename = f"A_alpha_{alpha_label}_uexact.npy"

    uexact_path = os.path.join(
    OUTPUT_DIR,
    uexact_filename
    )

    np.save(
    uexact_path,
    x_exact
    )

    print(f"Saved exact solution: {uexact_path}")
    # -----------------------------------------------------------
    # Save corresponding JSON metadata
    # -----------------------------------------------------------

    metadata = {
        "matrix_family":
            "Fixed-Eigenvalue Nonnormal Matrix Family",

        "matrix_definition":
            "A_alpha = V_alpha D V_alpha^{-1}",

        "dimension_n":
            N,

        "number_of_blocks_p":
            P,

        "block_size_m":
            BLOCK_SIZE,

        "beta":
            BETA,

        "alpha":
            float(alpha),

        "V_definition":
            "V_alpha = I + alpha*S",

        "V_inverse_definition":
            "V_alpha^{-1} = "
            "I - alpha*S + alpha^2*S^2 "
            "- alpha^3*S^3 + alpha^4*S^4",

        "nilpotency":
            f"S^{P} = 0",

        "D_definition":
            "Diagonal eigenvalues uniformly distributed "
            "from 1 to 10",

        "eigenvalue_min":
            float(expected_eigenvalues[0]),

        "eigenvalue_max":
            float(expected_eigenvalues[-1]),

        "kappa_2_V":
            float(kappa_V),

        "nonnormality_eta":
            float(eta),

        "kappa_2_A":
            float(kappa_A),

        "nnz_A":
            int(nnz_A),

        "density_A":
            float(density_A),

        "sparsity_A":
            float(sparsity_A),

        "maximum_eigenvalue_error":
            float(eigenvalue_error),

        "matrix_file":
            matrix_filename
    }

    json_filename = (
        f"A_alpha_{alpha_label}.json"
    )

    json_path = os.path.join(
        OUTPUT_DIR,
        json_filename
    )

    with open(
        json_path,
        "w",
        encoding="utf-8"
    ) as f:

        json.dump(
            metadata,
            f,
            indent=4
        )

    print(
        f"Saved metadata: {json_path}"
    )

    # -----------------------------------------------------------
    # Add row to results table
    # -----------------------------------------------------------

    results.append(
        {
            "alpha":
                alpha,

            "kappa_2(V_alpha)":
                kappa_V,

            "eta(A_alpha)":
                eta,

            "kappa_2(A_alpha)":
                kappa_A,

            "nnz(A_alpha)":
                nnz_A,

            "density":
                density_A,

            "sparsity":
                sparsity_A,

            "eigenvalue_error":
                eigenvalue_error
        }
    )


# ---------------------------------------------------------------
# 15. CREATE RESULTS TABLE
# ---------------------------------------------------------------

results_df = pd.DataFrame(
    results
)

print("\n")
print("=" * 90)
print("RESULTS TABLE")
print("=" * 90)

print(
    results_df.to_string(
        index=False,
        float_format=lambda x: f"{x:.8f}"
    )
)


# ---------------------------------------------------------------
# 16. SAVE RESULTS TABLE
# ---------------------------------------------------------------

csv_path = os.path.join(
    OUTPUT_DIR,
    "fixed_eigenvalue_nonnormal_results.csv"
)

results_df.to_csv(
    csv_path,
    index=False
)

print(
    f"\nResults table saved to:\n{csv_path}"
)


# ---------------------------------------------------------------
# 17. PLOT 1: CONDITION NUMBER OF V_alpha
# ---------------------------------------------------------------

plt.figure(
    figsize=(8, 5)
)

plt.plot(
    results_df["alpha"],
    results_df["kappa_2(V_alpha)"],
    marker="o"
)

plt.xlabel(r"$\alpha$")
plt.ylabel(r"$\kappa_2(V_\alpha)$")
plt.title(
    r"Condition Number of $V_\alpha$"
)

plt.grid(True)
plt.tight_layout()

plot_path = os.path.join(
    OUTPUT_DIR,
    "condition_number_V.png"
)

plt.savefig(
    plot_path,
    dpi=300,
    bbox_inches="tight"
)

plt.show()


# ---------------------------------------------------------------
# 18. PLOT 2: NONNORMALITY
# ---------------------------------------------------------------

plt.figure(
    figsize=(8, 5)
)

plt.plot(
    results_df["alpha"],
    results_df["eta(A_alpha)"],
    marker="o"
)

plt.xlabel(r"$\alpha$")
plt.ylabel(r"$\eta(A_\alpha)$")
plt.title(
    r"Nonnormality Measure $\eta(A_\alpha)$"
)

plt.grid(True)
plt.tight_layout()

plot_path = os.path.join(
    OUTPUT_DIR,
    "nonnormality_measure.png"
)

plt.savefig(
    plot_path,
    dpi=300,
    bbox_inches="tight"
)

plt.show()


# ---------------------------------------------------------------
# 19. PLOT 3: CONDITION NUMBER OF A_alpha
# ---------------------------------------------------------------

plt.figure(
    figsize=(8, 5)
)

plt.plot(
    results_df["alpha"],
    results_df["kappa_2(A_alpha)"],
    marker="o"
)

plt.xlabel(r"$\alpha$")
plt.ylabel(r"$\kappa_2(A_\alpha)$")
plt.title(
    r"Condition Number of $A_\alpha$"
)

plt.grid(True)
plt.tight_layout()

plot_path = os.path.join(
    OUTPUT_DIR,
    "condition_number_A.png"
)

plt.savefig(
    plot_path,
    dpi=300,
    bbox_inches="tight"
)

plt.show()


# ---------------------------------------------------------------
# 20. PLOT 4: COMBINED NONNORMALITY AND CONDITIONING
# ---------------------------------------------------------------

fig, ax1 = plt.subplots(
    figsize=(8, 5)
)

ax1.plot(
    results_df["alpha"],
    results_df["eta(A_alpha)"],
    marker="o",
    label=r"$\eta(A_\alpha)$"
)

ax1.set_xlabel(r"$\alpha$")
ax1.set_ylabel(
    r"$\eta(A_\alpha)$"
)

ax1.grid(True)

ax2 = ax1.twinx()

ax2.plot(
    results_df["alpha"],
    results_df["kappa_2(A_alpha)"],
    marker="s",
    label=r"$\kappa_2(A_\alpha)$"
)

ax2.set_ylabel(
    r"$\kappa_2(A_\alpha)$"
)

plt.title(
    r"Nonnormality and Conditioning versus $\alpha$"
)

fig.tight_layout()

plot_path = os.path.join(
    OUTPUT_DIR,
    "nonnormality_and_conditioning.png"
)

plt.savefig(
    plot_path,
    dpi=300,
    bbox_inches="tight"
)

plt.show()


# ---------------------------------------------------------------
# 21. FINAL SUMMARY
# ---------------------------------------------------------------

print("\n")
print("=" * 70)
print("GENERATION COMPLETE")
print("=" * 70)

print(
    f"Number of matrices generated : "
    f"{len(ALPHA_VALUES)}"
)

print(
    f"Matrix dimension              : "
    f"{N} x {N}"
)

print(
    f"Block structure                : "
    f"{P} x {P}"
)

print(
    f"Block size                     : "
    f"{BLOCK_SIZE} x {BLOCK_SIZE}"
)

print(
    f"Alpha range                    : "
    f"{ALPHA_VALUES[0]} to {ALPHA_VALUES[-1]}"
)

print(
    f"Output directory               : "
    f"{OUTPUT_DIR}"
)

print("=" * 70)