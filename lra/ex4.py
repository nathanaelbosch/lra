from lra.tools import *


N=200
B1 = generate_B1(N)
B2 = generate_B2(N)


A = B1
U_list = []
for m in range(1, len(A.shape)+1):
    A_mat = m_mode_matricisation(A, m)
    C, U, R = aca(A_mat, 1e-15)
    A_mat_cur = C.dot(U).dot(R)
    U, _, _ = np.linalg.svd(A_mat_cur, full_matrices=False)
    U_list.append(U)

C, _ = compute_core(A, U_list)
A_tilde = reconstruct(C, U_list)

frobenius_norm(A_tilde - A)
