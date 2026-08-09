import jax.numpy as jnp
import jax
from jax import lax


def quick_ichol0(A):
    """
    Compute a quick ichol0 of a dense SPD matrix A.
    Ijnput(s):
    A  - SPD dense matrix (jnp.array)
    Output(s):
    L0 - Dense lower triangular matrix (jnp.array)
    """
    Ad = A.todense()
    L = jnp.linalg.cholesky(Ad)
    L0 = jnp.multiply(Ad != 0.0, L)
    return L0


def ichol0(A):
    """
    Compute a ichol0 of a dense SPD matrix A.
    Ijnput(s):
    A  - SPD dense matrix (jnp.array)
    Output(s):
    L0 - Dense lower triangular matrix (jnp.array)
    """
    n = A.shape[0]
    Ad = A  # .todense()
    L0 = Ad.copy()

    for k in range(n):
        L0.at[k, k].set(jnp.sqrt(L0[k, k]))

        # Update along the column
        for i in range(k + 2, n):
            if L0[i, k] != 0.0:
                L0.at[i, k].set(L0[i, k] / L0[k, k])

        # Update across the row
        for j in range(k + 1, n):
            for i in range(j, n):
                if L0[i, j] != 0.0:
                    L0.at[i, j].set(L0[i, j] - L0[i, k] * L0[j, k].conj())

    L0 = jnp.tril(L0)
    return L0


@jax.jit
def jit_ichol0(A):

    n = A.shape[-1]
    L = A.copy()  # mutable copy carried through

    # --------------- outer loop over k ----------------------------------
    def body_k(k, L):
        # pivot ----------------------------------------------------------
        diag = jnp.sqrt(L[k, k])
        L = L.at[k, k].set(diag)

        # column update --------------------------------------------------
        def body_i(i, L):
            val = L[i, k]
            cond = (i > k) & (val != 0.0)
            newL = L.at[i, k].set(val / diag)
            return lax.select(cond, newL, L)

        L = lax.fori_loop(0, n, body_i, L)

        # row update -----------------------------------------------------
        def body_j(j, L):
            def body_i2(i, L):
                val = L[i, j]
                cond = (j > k) & (i >= j) & (val != 0.0)
                upd = val - L[i, k] * jnp.conj(L[j, k])
                newL = L.at[i, j].set(upd)
                return lax.select(cond, newL, L)

            return lax.fori_loop(0, n, body_i2, L)

        L = lax.fori_loop(0, n, body_j, L)

        return L

    L = lax.fori_loop(0, n, body_k, L)
    return jnp.tril(L)
