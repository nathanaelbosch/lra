from tools import *


N=200
B1 = generate_B1(N)
B2 = generate_B2(N)


###############################################################################
# Exercise 1
###############################################################################
for i, A in enumerate((B1, B2)):
    name = 'B1' if i==0 else 'B2'
    print(f'Working on {name}')

    U_list_full = compute_base_matrices(A)

    # 2. Binary search to find ranks
    min_ranks = np.array((0, 0, 0))
    max_ranks = np.array([U.shape[1] for U in U_list_full])
    max_error = 10e-4 * frobenius_norm(A)
    error = max_error + 1
    print('Max allowed error:', max_error)

    while np.all(max_ranks-min_ranks > 1) or error > max_error:
        ranks = min_ranks + ((max_ranks-min_ranks+1)//2)
        error = frobenius_norm(truncated_hosvd(A, ranks, U_list_full) - A)
        if error > max_error:
            min_ranks = ranks
        elif error < max_error:
            max_ranks = ranks
        print(ranks, error)
    print('Final ranks:', ranks)
    print('Final error:', error)


# Plot singular values
for i, A in enumerate((B1, B2)):
    for m in range(1, 4):
        name = 'B1' if i==0 else 'B2'
        _, Sigma, _ = np.linalg.svd(
            m_mode_matricisation(A, m), full_matrices=False)
        plt.plot(range(len(Sigma)), Sigma, 'b.-')
        plt.title(f'{name} - singular values of {m}-mode matricization')
        plt.savefig(f'plots/{name}_{m}-mode-matricization_singular-values.png')
        plt.gcf().clear()
