"""Exercise 1

Computes the required things as written in the exercise description,
as well as every statement made in the section "Exercise 1" in the report.
"""
import matplotlib.pyplot as plt

from .tools import *
from .hosvd import *


N=200
B1 = generate_B1(N)
B2 = generate_B2(N)

print('Computes the HOSVD for B1 and B2 - Takes about 2 minutes')


for rel_error in (1e-4, 1e-6):
    for i, A in enumerate((B1, B2)):
        name = 'B1' if i==0 else 'B2'
        print('Working on {}, using a maximal relative error of {:.0E}'.format(
            name, rel_error))

        full_list = compute_base_matrices(A, singular_values=True)
        U_list, Sigma_list = zip(*full_list)

        # Plot singular values as expected in the exercise, but only once
        if rel_error == 1e-4:
            m = 1
            for Sigma in Sigma_list:
                plt.semilogy(
                    range(len(Sigma)), Sigma, '.-',
                    color='#005293')
                plt.title(
                    f'{name} - singular values of {m}-mode matricization')
                try:
                    plt.savefig(
                        f'plots/{name}_{m}-mode-matricization_singular-values.png')
                except FileNotFoundError:
                    print('Could not save plot as the directory ./plots/ ' +
                          'does not exist')
                plt.gcf().clear()
                m = m+1

        # Compute the ranks that result in a bounded error as required
        max_error = rel_error * frobenius_norm(A)
        max_error_square = max_error ** 2
        # Vectorized computation using some numpy functionality
        sng_vals = np.stack(Sigma_list)
        sng_vals_cumsum = np.cumsum(sng_vals[:, ::-1], axis=1)
        ranks = np.sum(
            (np.cumsum(sng_vals[:, ::-1], axis=1) > max_error_square/3),
            axis=1)
        print(f'Resulting ranks: {ranks}')

        # Verify the result:
        A_tilde = truncated_hosvd(A, ranks, U_list)
        assert frobenius_norm(A_tilde - A) < max_error

        print('Resulting Relative Error: {}'.format(
            frobenius_norm(A_tilde - A) / frobenius_norm(A)))
