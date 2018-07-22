import numpy as np

from .tools import *


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
# Combine the above
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
