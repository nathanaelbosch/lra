"""Tools to work with tensors and setup of the exercise"""
import numpy as np
import numba as nb


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
# When using functions instead of tensors
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
