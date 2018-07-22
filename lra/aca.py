import numpy as np
import logging

from .tools import *


def aca_full_pivoting(A, epsilon):
    """ACA with full pivoting as in the lecture

    Takes in a matrix, and returns the CUR decomposition
    """
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
    """ACA with partial pivoting

    Works as described in the pseudocode in the report
    """
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
    """Same functionality as above - receives a function instead of a tensor

    In order to take advantage of the partial pivoting we don't want to assume
    that we received the full tensor of function evaluations. Instead we pass
    a function, and evaluate depending on our needs.
    """
    m, n = shape
    fk = f
    I_list = []
    J_list = []
    I_good = []
    J_good = []
    i = 1

    # Working with functions requires this:
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
