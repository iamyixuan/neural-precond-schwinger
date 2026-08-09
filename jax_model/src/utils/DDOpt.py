import jax
import jax.numpy as jnp
import numpy as np
from .gamma import gamma_factory


class Dirac_Matrix:
    def __init__(self, U, kappa):
        self.n_dim = 2
        self.time_index = 2

        self.U = U
        self.kappa = kappa

        # assert jnp.iscomplexobj(U)
        # assert len(U.shape) == 4
        # assert U.shape[1] == self.n_dim

        self.lattice_shape = U.shape[2:]
        self.gammas = jnp.stack(
            gamma_factory(n_dim=self.n_dim, incl_g5=False), axis=0
        )
        self.gammas = jnp.array(self.gammas).astype(U.dtype)
        self.gamma5 = 1j * jnp.dot(self.gammas[0], self.gammas[1])
        self.identity = jnp.eye(self.n_dim, dtype=U.dtype)

    def apply(self, x, dagger=False, gamma5=False):
        x = jnp.asarray(x, dtype=self.U.dtype)
        x = self._normalize_vector_shape(x)

        # if dagger:
        #     x = jnp.einsum("ij, ...j->...i", self.gamma5, x)
        def true_x(_):
            return jnp.einsum("ij, ...j->...i", self.gamma5, x)

        def false_x(_):
            return x

        x = jax.lax.cond(dagger, true_x, false_x, None)

        padded_x, slice_idx = self._antiperiodic_pad_vector(x)
        H = jnp.zeros_like(x)
        for mu in range(self.n_dim):
            forward_x = jnp.roll(padded_x, shift=-1, axis=1 + mu)
            forward_x = forward_x[slice_idx]
            forward_U = self.U[:, mu, ..., jnp.newaxis]
            # forward_U = self.U[:, mu, ...].reshape(list(self.U.shape[:-2]) + [1])
            forward_x = forward_U * forward_x

            backward_x = jnp.roll(padded_x, shift=1, axis=1 + mu)
            backward_x = backward_x[slice_idx]
            backward_U = jnp.roll(self.U[:, mu], shift=1, axis=1 + mu).conj()
            backward_U = backward_U[..., jnp.newaxis]
            # backward_U = backward_U.reshape(list(backward_U.shape[:-2]) + [1])
            backward_x = backward_U * backward_x

            H = H + jnp.einsum(
                "ij, ...j->...i", self.identity - self.gammas[mu], forward_x
            )
            H = H + jnp.einsum(
                "ij, ...j->...i", self.identity + self.gammas[mu], backward_x
            )

        y = x - (self.kappa * H)

        # if dagger:
        #     y = jnp.einsum("ij, ...j->...i", self.gamma5, y)
        def true_fun(_):
            return jnp.einsum("ij, ...j->...i", self.gamma5, y)

        def false_fun(_):
            return y

        y = jax.lax.cond(dagger, true_fun, false_fun, None)
        return y

    def _normalize_vector_shape(self, x):
        # assert len(x.shape) in [3, 4], f'unknown vector shape {x.shape}'
        # if len(x.shape) == 3:
        #     x = x.reshape((1,) + x.shape)  # dummy batch dimension
        # def true_branch(_):
        #     return x.reshape((1,) + x.shape)

        # def false_branch(_):
        #     return x

        # x = jax.lax.cond(x.ndim == 3, true_branch, false_branch, None)
        # assert x.shape[-1] == self.n_dim
        # assert x.shape[1:-1] == self.lattice_shape
        return x

    def _antiperiodic_pad_vector(self, x):
        padded_x = x.copy()

        s1 = [slice(None)] * len(x.shape)
        s1[self.time_index] = slice(0, 1)
        forward_pad = -padded_x[tuple(s1)]  # Convert list to tuple here

        s2 = [slice(None)] * len(x.shape)
        s2[self.time_index] = slice(
            x.shape[self.time_index] - 1, x.shape[self.time_index]
        )
        backward_pad = -padded_x[tuple(s2)]  # Convert list to tuple here

        padded_x = jnp.concatenate(
            [backward_pad, padded_x, forward_pad], axis=self.time_index
        )

        # Slice indices needed to recover the unpadded vector
        slice_idx = [slice(None)] * len(padded_x.shape)
        slice_idx[self.time_index] = slice(1, -1)

        # assert jnp.allclose(padded_x[tuple(slice_idx)], x), 'invalid padding'  # Convert list to tuple here
        return (padded_x, tuple(slice_idx))


class EvenOddPreconditionedOperator:
    def __init__(self, dirac_operator):
        self.D = dirac_operator
        self.shape = dirac_operator.lattice_shape
        X, T = self.shape
        xx, tt = jnp.meshgrid(jnp.arange(X), jnp.arange(T), indexing="ij")
        parity = (xx + tt) % 2
        self.even_mask = (parity == 0).flatten()
        self.odd_mask = ~self.even_mask

    def _mask(self, x, mask):
        return x.reshape(x.shape[0], -1, x.shape[-1])[:, mask]

    def _unmask(self, x_sub, mask):
        return x_sub

    def __call__(self, x_e):
        # Embed x_e to full lattice
        x_full = self._unmask(x_e, self.even_mask)

        # Step 1: D_oe x_e
        Doex = self.D.apply(x_full)
        Doex = self._mask(Doex, self.odd_mask)

        # Step 2: Solve D_oo z = Doex
        def solve_odd(rhs_odd):
            rhs_full = self._unmask(rhs_odd[None], self.odd_mask)
            z_full = self.D.apply(rhs_full, dagger=False)
            return self._mask(z_full, self.odd_mask)

        z = jax.vmap(solve_odd)(Doex)

        # Step 3: D_eo z
        z_full = self._unmask(z, self.odd_mask)
        Dcorr = self.D.apply(z_full, dagger=False)
        Dcorr = self._mask(Dcorr, self.even_mask)

        # Step 4: M_eo x_e = D_ee x_e - D_eo D_oo^{-1} D_oe x_e
        Dfull = self.D.apply(x_full, dagger=False)
        Dfull_even = self._mask(Dfull, self.even_mask)
        Meo_x = Dfull_even - Dcorr

        # Step 5: Apply adjoint: M_eo^† (M_eo x_e)
        Meo_full = self._unmask(Meo_x, self.even_mask)
        result = self.D.apply(Meo_full, dagger=True)
        return self._mask(result, self.even_mask)


class DiracGamma:
    def __init__(self, U, gammas, kappa=0.276):
        """
        gammas: the network output of shape (B, 2 * n_multiplies, 2, 2)
        """
        self.n_dim = 2
        self.time_index = 2

        self.U = U
        self.kappa = kappa

        self.lattice_shape = U.shape[2:]
        self.gammas = gammas.reshape(
            gammas.shape[0], gammas.shape[1] // 2, 2, 2, 2
        )  # shape (B, n_multiplies, 2, 2, 2)

        # TODO!! with multiple gamma matrices, we need to modify the gamma5 later
        self.gamma5 = 1j * jnp.einsum(
            "bnij, bnjk->bnik", self.gammas[:, :, 0], self.gammas[:, :, 1]
        )

        self.identity = jnp.eye(self.n_dim, dtype=U.dtype)
        self.identity = jnp.repeat(
            self.identity[None, ...], U.shape[0], axis=0
        )

    def apply(self, x, dagger=False, gamma5=False):
        x = jnp.asarray(x, dtype=self.U.dtype)
        x = self._normalize_vector_shape(x)

        def true_x(_):
            return jnp.einsum("bnij, b...j->b...i", self.gamma5, x)

        def false_x(_):
            return x

        x = jax.lax.cond(dagger, true_x, false_x, None)

        padded_x, slice_idx = self._antiperiodic_pad_vector(x)
        H = jnp.zeros_like(x)
        for mu in range(self.n_dim):
            forward_x = jnp.roll(padded_x, shift=-1, axis=1 + mu)
            forward_x = forward_x[slice_idx]
            forward_U = self.U[:, mu, ..., jnp.newaxis]
            forward_x = forward_U * forward_x

            backward_x = jnp.roll(padded_x, shift=1, axis=1 + mu)
            backward_x = backward_x[slice_idx]
            backward_U = jnp.roll(self.U[:, mu], shift=1, axis=1 + mu).conj()
            backward_U = backward_U[..., jnp.newaxis]
            backward_x = backward_U * backward_x

            # we can add an inner loop here over multiple (multiply of n_dim) gamma matrices
            for j in range(self.gammas.shape[1]):
                H = H + jnp.einsum(
                    "bij, b...j->b...i",
                    self.identity - self.gammas[:, j, mu],
                    forward_x,
                )
                H = H + jnp.einsum(
                    "bij, b...j->b...i",
                    self.identity + self.gammas[:, j, mu],
                    backward_x,
                )

        y = x - (self.kappa * H)

        def true_fun(_):
            return jnp.einsum("bnij, b...j->b...i", self.gamma5, y)

        def false_fun(_):
            return y

        y = jax.lax.cond(dagger, true_fun, false_fun, None)
        return y

    def _normalize_vector_shape(self, x):
        return x

    def _antiperiodic_pad_vector(self, x):
        padded_x = x.copy()

        s1 = [slice(None)] * len(x.shape)
        s1[self.time_index] = slice(0, 1)
        forward_pad = -padded_x[tuple(s1)]  # Convert list to tuple here

        s2 = [slice(None)] * len(x.shape)
        s2[self.time_index] = slice(
            x.shape[self.time_index] - 1, x.shape[self.time_index]
        )
        backward_pad = -padded_x[tuple(s2)]  # Convert list to tuple here

        padded_x = jnp.concatenate(
            [backward_pad, padded_x, forward_pad], axis=self.time_index
        )

        # Slice indices needed to recover the unpadded vector
        slice_idx = [slice(None)] * len(padded_x.shape)
        slice_idx[self.time_index] = slice(1, -1)

        # assert jnp.allclose(padded_x[tuple(slice_idx)], x), 'invalid padding'  # Convert list to tuple here
        return (padded_x, tuple(slice_idx))

