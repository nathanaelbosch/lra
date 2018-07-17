from lra.tools import *


N=100
B1 = generate_B1(N)
B2 = generate_B2(N)


for i, A in enumerate((B1, B2)):
# for i, f in enumerate((f1, f2)):
    name = 'B1' if i==0 else 'B2'
    print(f'Working on {name}')

    Core = A
    dimensions = len(A.shape)
    C_list = []

    for m in range(1, dimensions+1):
        print(f'Mode {m}')
        # CUR decomposition of m-mode matricization
        # Cm, Um, Rm = aca_full_pivoting(Core_mat, 1e-10)
        if m == 0:
            f_matricized = functional_m_mode_matricization(
                f, (N, N, N), m)
            Cm, Um, Rm = aca_partial_pivoting(Core_mat, 1e-10)
        else:
            Core_mat = m_mode_matricisation(Core, m)
            Cm, Um, Rm = aca_partial_pivoting(Core_mat, 1e-5)

        # Get the resulting shape for the refolding
        start_shape = Core.shape
        rank = Um.shape[0]
        new_shape = [rank if i+1==m else start_shape[i] for i in range(dimensions)]
        print(f'Shape this round: {new_shape}')

        Core = refold(Um.dot(Rm), m, new_shape)
        C_list.append(Cm)

    A_tilde = reconstruct(Core, C_list)

    ranks = Core.shape
    print(f'Resulting ranks: {ranks}')
    print(f'Relative Error: {frobenius_norm(A_tilde - A) / frobenius_norm(A)}')
