import numpy as np
import numba as nb
import matplotlib.pyplot as plt
import logging


N = 200


###############################################################################
# Setup
###############################################################################
@nb.jit
def xi(i):
    return i / (N+1)


@nb.jit
def f1(i1, i2, i3):
    return np.sin(xi(i1)+xi(i2)+xi(i3))


@nb.jit
def generate_B1(N):
    B = np.ones((N, N, N))
    for i1 in range(N):
        for i2 in range(N):
            for i3 in range(N):
                B[i1, i2, i3] = f1(i1, i2, i3)
    return B


@nb.jit
def f2(i1, i2, i3):
    return np.sqrt(xi(i1)**2+xi(i2)**2+xi(i3)**2)


@nb.jit
def generate_B2(N):
    B = np.ones((N, N, N))
    for i1 in range(N):
        for i2 in range(N):
            for i3 in range(N):
                B[i1, i2, i3] = f2(i1, i2, i3)
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
# Functions instead of tensors
###############################################################################
def functional_m_mode_matricization(f, tensor_shape, m):
    assert len(tensor_shape) == 3, 'Not implemented for dim>3'
    n1, n2, n3 = tensor_shape

    def matricized_f(i, j):
        if m == 1:
            i1, i2, i3 = i, j % n2, j//n2
        elif m == 2:
            i1, i2, i3 = j % n1, i, j//n1
        elif m == 3:
            i1, i2, i3 = j % n2, j//n2, i
        return f(i1, i2, i3)
    return matricized_f


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
# Regarding Ex 4
###############################################################################
def aca_full_pivoting(A, epsilon):
    """ACA with full pivoting as in the lecture

    Takes in a matrix, and returns the """
    # R0 = A
    Rk = A.copy()
    I_list = []
    J_list = []
    while frobenius_norm(Rk) > epsilon*frobenius_norm(A):
        i, j = np.unravel_index(np.argmax(np.abs(Rk), axis=None), Rk.shape)
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


def aca_partial_pivoting(A, epsilon):
    # R0 = A
    m, n = A.shape
    Rk = A.copy()
    I_list = []
    J_list = []
    I_good = []
    J_good = []
    u_list = []
    v_list = []
    i = 1
    Ak_norm = 0
    k = 1
    # while k == 1 or u.dot(u)*v.dot(v) > (epsilon*frobenius_norm(A))**2:
    # while frobenius_norm(Rk) > epsilon*frobenius_norm(A):
    while k == 1 or u.dot(u)*v.dot(v) > (epsilon**2)*Ak_norm:
        _Rk = np.abs(Rk[i, :].copy())
        _Rk[J_list] = -1
        j = np.argmax(_Rk)
        delta = Rk[i, j]
        # if delta == 0:
        if np.isclose(delta, 0):
            # print(len(I_list))
            # print(np.min((m, n)) - 1)
            if len(I_list) == np.min((m, n)) - 1:
                break
        else:
            u = Rk[:, j]
            v = Rk[i, :].T / delta

            Rk = Rk - np.outer(u, v)
            logging.debug(f'Residual norm: {frobenius_norm(Rk)}')
            logging.debug(f'Update norm: {np.linalg.norm(u, 2)*np.linalg.norm(v, 2)}')
            u_list.append(u)
            v_list.append(v)

            k += 1

            Ak_norm = (Ak_norm + u.dot(u) * v.dot(v) +
                np.sum([u.T.dot(u_list[l]) * (v_list[l].T).dot(v)
                        for l in range(0, k-1)]))
            logging.debug(f'Ak norm: {Ak_norm}')

            I_good.append(i)
            J_good.append(j)

        # if k <= 3:
        #     print(i, j)
        I_list.append(i)
        J_list.append(j)
        _u = np.abs(u.copy())
        _u[I_list] = -1
        i = np.argmax(_u)

    R = A[I_good, :]
    U = np.linalg.inv(A[I_good, :][:, J_good])
    C = A[:, J_good]

    return C, U, R


def aca_partial_pivoting_functional(f, shape, epsilon):
    """Function instead of tensor - Otherwise same functionality as above

    In order to take advantage of the partial pivoting we don't want to assume
    that we received the full tensor of function evaluations. Instead we pass
    a function, and evaluate depending on our needs."""
    m, n = shape
    fk = f
    # Rk = np.ones(m, n)
    I_list = []
    J_list = []
    I_good = []
    J_good = []
    i = 1
    # while frobenius_norm(Rk) > epsilon*frobenius_norm(A):

    # We are working in a functional manner
    def step(fk, u, v):
        def new_fk(i, j):
            return fk(i, j) - u[i]*v[j]
        return new_fk

    k = 1
    Ak_norm = 0
    u_list = []
    v_list = []
    while k == 1 or uk.dot(uk)*vk.dot(vk) > (epsilon**2)*Ak_norm:
        current_row = np.array([fk(i, j) for j in range(n)])
        _current_row = np.abs(current_row.copy())
        _current_row[J_list] = -1
        j = np.argmax(_current_row)
        delta = fk(i, j)
        if np.isclose(delta, 0):
            if len(I_list) == np.min((m, n)) - 1:
                break
        else:
            current_col = np.array([fk(k, j) for k in range(m)])
            uk = current_col
            vk = current_row.T / delta
            u_list.append(uk)
            v_list.append(vk)
            fk = step(fk, uk, vk)
            # Rk = Rk - np.outer(u, v)

            k += 1

            Ak_norm = (
                Ak_norm + uk.dot(uk) * vk.dot(vk) +
                np.sum([uk.T.dot(u_list[l]) * (v_list[l].T).dot(vk)
                        for l in range(0, k-1)]))
            I_good.append(i)
            J_good.append(j)

        I_list.append(i)
        J_list.append(j)

        _u = np.abs(uk.copy())
        _u[I_list] = -1
        i = np.argmax(_u)

    R = np.array([[f(i, j) for j in range(n)] for i in I_good])
    U = np.linalg.inv(np.array([[f(i, j) for j in J_good] for i in I_good]))
    C = np.array([[f(i, j) for j in J_good] for i in range(m)])

    return C, U, R
