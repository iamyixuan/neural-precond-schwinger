import jax.numpy as np
import jax


def quick_ichol0(A):
    """
    Compute a quick ichol0 of a dense SPD matrix A.
    Input(s):
    A  - SPD dense matrix (np.array)
    Output(s):
    L0 - Dense lower triangular matrix (np.array)
    """
    Ad = A.todense()
    L = np.linalg.cholesky(Ad)
    L0 = np.multiply(Ad != 0.0, L)
    return L0


def ichol0(A):
    """
    Compute a ichol0 of a dense SPD matrix A.
    Input(s):
    A  - SPD dense matrix (np.array)
    Output(s):
    L0 - Dense lower triangular matrix (np.array)
    """
    n = A.shape[0]
    Ad = A#.todense()
    L0 = Ad.copy()

    for k in range(n):
        L0.at[k, k].set(np.sqrt(L0[k, k]))

    # Update along the column
    for i in range(k + 1, n):
        if L0[i, k] != 0.0:
            L0.at[i, k].set( L0[i, k] / L0[k, k])

    # Update across the row
    for j in range(k + 1, n):
        for i in range(j, n):
            if L0[i, j] != 0.0:
                L0.at[i, j].set(L0[i, j] - L0[i, k] * L0[j, k].conj())

    L0 = np.tril(L0)
    return L0
