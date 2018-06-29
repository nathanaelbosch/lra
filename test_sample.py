import numpy as np
from tools import *


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
