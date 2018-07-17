from .tools import *


N=300
B1 = generate_B1(N)
B2 = generate_B2(N)


###############################################################################
# Exercise 1
###############################################################################
for i, A in enumerate((B1, B2)):
    name = 'B1' if i==0 else 'B2'
    print(f'Working on {name}')

    full_list = compute_base_matrices(A, singular_values=True)
    U_list, Sigma_list = zip(*full_list)

    # Error bound:
    max_error = 10e-4 * frobenius_norm(A)
    max_error_square = max_error ** 2

    # Put them all in an array as vectorized is always good
    sng_vals = np.stack(Sigma_list)
    # Cumsum of the singular values, smallest to largest, by matricization
    sng_vals_cumsum = np.cumsum(sng_vals[:, ::-1], axis=1)
    # Giving 1/3 of the error to each component, what ranks does this give?
    ranks = np.sum(
        (np.cumsum(sng_vals[:, ::-1], axis=1) > max_error_square/3),
        axis=1)
    print('Ranks chosen:', ranks)

    # Verify the result:
    A_tilde = truncated_hosvd(A, ranks, U_list)
    assert frobenius_norm(A_tilde - A) < max_error

    """
    if False:
        # 2. Binary search to find ranks
        min_ranks = np.array((0, 0, 0))
        max_ranks = np.array([U.shape[1] for U in U_list])
        error = max_error + 1
        print('Max allowed error:', max_error)

        while np.all(max_ranks-min_ranks > 1) or error > max_error:
            ranks = min_ranks + ((max_ranks-min_ranks+1)//2)
            error = frobenius_norm(truncated_hosvd(A, ranks, U_list) - A)
            if error > max_error:
                min_ranks = ranks
            elif error < max_error:
                max_ranks = ranks
            print(ranks, error)
        print('Final error:', error)
        print('Final ranks:', ranks)
    """

import sys
sys.exit(1)

# Plot singular values
for i, A in enumerate((B1, B2)):
    for m in range(1, 4):
        name = 'B1' if i==0 else 'B2'
        _, Sigma, _ = np.linalg.svd(
            m_mode_matricisation(A, m), full_matrices=False)
        # plt.semilogy(
        plt.plot(
            range(len(Sigma)), Sigma, '.-',
            color='#005293')
        plt.title(f'{name} - singular values of {m}-mode matricization')
        plt.savefig(f'plots/{name}_{m}-mode-matricization_singular-values.png')
        plt.gcf().clear()
