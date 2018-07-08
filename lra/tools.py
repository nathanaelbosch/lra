import numpy as np
import numba as nb
import matplotlib.pyplot as plt


N = 200


###############################################################################
# Setup
###############################################################################
@nb.jit
def xi(i):
    return i / (N+1)


@nb.jit
def generate_B1(N):
    B = np.ones((N, N, N))
    for i1 in range(N):
        for i2 in range(N):
            for i3 in range(N):
                B[i1, i2, i3] = np.sin(xi(i1)+xi(i2)+xi(i3))
    return B


@nb.jit
def generate_B2(N):
    B = np.ones((N, N, N))
    for i1 in range(N):
        for i2 in range(N):
            for i3 in range(N):
                B[i1, i2, i3] = np.sqrt(xi(i1)**2+xi(i2)**2+xi(i3)**2)
    return B


###############################################################################
# Tools
###############################################################################
def m_mode_matricisation(A, m):
    dim = len(A.shape)
    permutation = [m-1] + list(range(0, m-1)) + list(range(m, dim))
    out = A.transpose(permutation)
    return out.reshape(out.shape[0], np.prod(out.shape[1:]))


def refold(A, m, shape):
    dim = len(shape)
    permutation = [m-1] + list(range(0, m-1)) + list(range(m, dim))
    permuted_shape = [shape[p] for p in permutation]
    out = A.reshape(permuted_shape)
    un_permutation = list(range(1, m)) + [0] + list(range(m, dim))
    return out.transpose(un_permutation)


def m_mode_matrix_multiplication(A, B, m):
    A_mat = m_mode_matricisation(A, m)
    out_mat = B.dot(A_mat)
    out = refold(out_mat, m, A.shape[:m-1] + (B.shape[0], ) + A.shape[m:])
    return out


def frobenius_norm(A):
    return np.sqrt(np.sum(A**2))


###############################################################################
# HOSVD - Tucker Decomposition
###############################################################################
# HOSVD as described in the lecture consists of multiple steps:
# 1. Calculate SVD of matricizations
# 2. Truncate basis matrices
# 3. Form core tensor
# 4. Reconstruct tensor from core and basis matrices
# My implementations has these different parts in different functions
# @nb.jit
def compute_base_matrices(A, singular_values=False):
    """SVDs of matricisations

    Returns a list of the basis matrices

    We need the singular values for error estimation, otherwise not
    """
    dim = len(A.shape)

    # 1. Calculate SVD of matricisations
    U_list = []
    for i in range(dim):
        m = i+1
        mat = m_mode_matricisation(A, m)
        U, Sigma, _ = np.linalg.svd(mat, full_matrices=False)
        if singular_values:
            U_list.append((U, Sigma))
        else:
            U_list.append(U)

    return U_list


# @nb.jit
def compute_core(A, U_list):
    """Form the core tensor

    Returns basically a Tucker decomposition of A
    """
    dim = len(A.shape)
    C = A
    for i in range(dim-1, -1, -1):
        C = m_mode_matrix_multiplication(C, U_list[i].T, i+1)

    return C, U_list


# @nb.jit
def reconstruct(C, U_list):
    """Reconstructs the tensor given the Tucker decomposition"""
    dim = len(C.shape)
    A_tilde = C
    for i in range(dim-1, -1, -1):
        A_tilde = m_mode_matrix_multiplication(A_tilde, U_list[i], i+1)
    return A_tilde


###############################################################################
# Examples
###############################################################################
def full_hosvd(A):
    """Example on how to use the above functions: Full hosvd"""
    U_list = compute_base_matrices(A)
    C, _ = compute_core(A, U_list)
    A_tilde = reconstruct(C, U_list)
    return A_tilde


def truncated_hosvd(A, ranks, U_list_full=None):
    """Other example: HOSVD with truncation of the base matrices

    Can be given the full basis matrices to save computations, we use this
    in the next section.
    """
    if not U_list_full:
        U_list_full = compute_base_matrices(A)
    U_list = [U[:, :ranks[i]] for i, U in enumerate(U_list_full)]
    C, _ = compute_core(A, U_list)
    A_tilde = reconstruct(C, U_list)
    return A_tilde


###############################################################################
# Examples
###############################################################################
def aca(A, epsilon):
    """ACA with full pivoting as in the lecture"""
    # R0 = A
    Rk = A
    I_list = []
    J_list = []
    while frobenius_norm(Rk) > epsilon*frobenius_norm(A):
        i, j = np.unravel_index(Rk.argmax(axis=None), Rk.shape)
        I_list.append(i)
        J_list.append(j)
        delta = Rk[i, j]
        u = Rk[:, j]
        v = Rk[i, :].T / delta
        Rk = Rk - np.outer(u, v)

    R = A[I_list, :]
    U = np.linalg.inv(A[I_list, :][:, J_list])
    C = A[:, J_list]

    return C, U, R
