"""Exercise 4

Computes the required things as written in the exercise description,
as well as every statement made in the section "Exercise 4" in the report.
"""
import time
import datetime as dt

from .tools import *
from .aca import *


epsilon = 1e-5

print('Computes all results stated in the section about exercise 4 ' +
      'of the report - takes around 5 minutes')

for i, (f, N) in enumerate(((f1, 400), (f2, 400), (f2, 1500))):
    start = time.time()
    name = 'B1' if i==0 else 'B2'
    print(f'Working on {name} with n={N}')

    # Core = A
    dimensions = 3
    C_list = []

    for m in range(1, dimensions+1):
        print(f'Mode {m}')
        if m == 1:
            # Work with the function insted of the tensor
            f_matricized = functional_m_mode_matricization(
                f, (N, N, N), m)
            Cm, Um, Rm = aca_partial_pivoting_functional(
                f_matricized, (N, N*N), epsilon)
        else:
            # We computed the tensor now anyways, so work with the tensor
            Core_mat = m_mode_matricisation(Core, m)
            Cm, Um, Rm = aca_partial_pivoting(Core_mat, epsilon)

        # Get the resulting shape for the refolding
        if m == 1:
            start_shape = (N, N, N)
        else:
            start_shape = Core.shape
        rank = Um.shape[0]
        new_shape = [rank if i+1==m else start_shape[i] for i in range(dimensions)]
        print(f'Shape this round: {new_shape}')

        Core = refold(Um.dot(Rm), m, new_shape)
        C_list.append(Cm)

    ranks = Core.shape
    print(f'Resulting ranks: {ranks}')

    # We can only compute the relative error for small N
    # A_tilde = reconstruct(Core, C_list)
    # print(f'Relative Error: {frobenius_norm(A_tilde - A) / frobenius_norm(A)}')

    end = time.time()
    time_needed = dt.timedelta(seconds=end-start)
    print('Time needed for this computation: {}'.format(time_needed))
