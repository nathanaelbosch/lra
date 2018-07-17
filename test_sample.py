import numpy as np
from lra.tools import *


# Setup: Some simple example tensors
dummy_tensor = np.array(range(2*3*4)).reshape(2, 3, 4)
dummy_tensor.reshape(2, 3*4)
dummy_matrix = np.array(range(2*3)).reshape(3, 2)


def test_unfold_refold():
    assert (refold(m_mode_matricisation(
        dummy_tensor, 1), 1, dummy_tensor.shape) == dummy_tensor).all()


def test_matrix_multiplication():
    # Just checks if it runs to look for shape errors
    m_mode_matrix_multiplication(dummy_tensor, dummy_matrix, 1)


def test_full_hosvd():
    # Full HOSVD should keep the tensor quite similar
    assert frobenius_norm(full_hosvd(dummy_tensor)-dummy_tensor) < 1e-10


def test_f_matricization():
    B1 = generate_B1(10)
    B1_matricized = m_mode_matricisation(B1, 1)
    f1_matricized = functional_m_mode_matricization(f1, B1.shape, 1)
    other = np.ones_like(B1_matricized)
    m, n = other.shape
    for i in range(m):
        for j in range(n):
            other[i, j] = f1_matricized(i, j)
    assert np.allclose(other, B1_matricized)
