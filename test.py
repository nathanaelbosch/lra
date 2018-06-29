import numpy as np
from tools import *


dummy_tensor = np.array(range(2*3*4)).reshape(2, 3, 4)
dummy_tensor.reshape(2, 3*4)
dummy_matrix = np.array(range(2*3)).reshape(3, 2)

print('Test matricization and refold')
assert (refold(m_mode_matricisation(
    dummy_tensor, 1), 1, dummy_tensor.shape) == dummy_tensor).all()

print('Test m mode matrix mult')
m_mode_matrix_multiplication(dummy_tensor, dummy_matrix, 1)

print('Test full hosvd')
assert frobenius_norm(full_hosvd(dummy_tensor)-dummy_tensor) < 1e-10
