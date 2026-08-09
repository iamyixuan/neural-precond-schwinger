import time
from functools import partial
from typing import NamedTuple
from pathlib import Path
from tqdm import tqdm

import jax
from jax import config
config.update("jax_enable_x64", True)
import jax.numpy as jnp
from jax import lax
import numpy as np
from jax.scipy.linalg import solve_triangular
from jax import config
from .DDOpt import Dirac_Matrix
from .losses import HPD_opt
from .ichols import ichol0, jit_ichol0
from .metrics import construct_matrix
from .jax_solver import solve


def forward_substitution(L, b):
    """Solve L * y = b using forward substitution."""
    y = []
    b = b.ravel()
    for i in range(len(b)):
        y_i = (b[i] - jnp.dot(L[i, :i], jnp.array(y))) / L[i, i]
        y.append(y_i)
    return jnp.array(y)


def back_substitution(U, y):
    """Solve U * x = y using back substitution."""
    x = []
    y = y.ravel()
    n = len(y)
    for i in range(n - 1, -1, -1):
        x_i = (y[i] - jnp.vdot(U[i, i + 1 :], jnp.array(x[::-1]))) / U[i, i]
        x.append(x_i)
    return jnp.array(x[::-1])


# def preconditioner(L, U, v):
#     v_flat = v.reshape(v.shape[0], -1)  # (B, n)
#     y = solve_triangular(L, v_flat.T, lower=True)
#     z = solve_triangular(U, y, lower=False)
#     return z.reshape(v.shape)


# @jax.jit
# def batched_ichol_solve(L, U, v):
#     v_flat = v.reshape(v.shape[0], -1)  # (B, n)
#     y = solve_triangular(L, v_flat, lower=True)  # Solve L * y = v
#     z = solve_triangular(U, y, lower=False)
#     z = z.reshape(v.shape)
#     return z


@jax.jit
def batched_ichol_solve(L, v):
    B = v.shape[0]
    n = v.reshape(B, -1).shape[-1]
    v_flat = v.reshape(B, n)

    # 1) solve     L y = v
    y = solve_triangular(L, v_flat, lower=True)
    # 2) solve U z = y
    z = solve_triangular(L, y, lower=True, trans="C")

    return z.reshape(v.shape)


@jax.vmap
def preconditioner(L, U, v):
    """Apply ILU preconditioner M^{-1} to vector v."""
    y = forward_substitution(L, v)  # Solve L * y = v
    z = back_substitution(U, y)  # Solve U * z = y
    return z.reshape(v.shape)


def cg_solve(model, U1, kappa, p=1, use_ichol=False, **kwargs):
    """
    CG solve with different preconditioners
    Args:
        model: trained model
        U1: batched gauge config
    """
    if kwargs.get("U_tilde") is not None:
        print("Using provided U_tilde...")
        U_tilde = kwargs["U_tilde"]
    else:
        U_tilde = jax.vmap(model)(U1).squeeze()

    D = Dirac_Matrix(U1, kappa=kappa)
    M = Dirac_Matrix(U_tilde, kappa=kappa)

    key = jax.random.PRNGKey(0)
    b_real = jax.random.normal(key, (U1.shape[0], U1.shape[2], U1.shape[3], 2))
    b_imag = jax.random.normal(key, (U1.shape[0], U1.shape[2], U1.shape[3], 2))
    b = b_real + 1j * b_imag

    # time the two methods
    num_iter = 2000
    #
    DD_opt = jax.jit(lambda x: HPD_opt(D, x, 1))
    NN_M_opt = jax.jit(lambda x: HPD_opt(M, x, p))

    BATCH_SIZE = 1

    unpred_hist_list = []
    nnpred_hist_list = []

    for i in tqdm(range(U1.shape[0] // BATCH_SIZE)):
        U1_batch = U1[i * BATCH_SIZE : (i + 1) * BATCH_SIZE]
        Utilde_batch = U_tilde[i * BATCH_SIZE : (i + 1) * BATCH_SIZE]

        D_batch = Dirac_Matrix(U1_batch, kappa=kappa)
        M_batch = Dirac_Matrix(Utilde_batch, kappa=kappa)

        b_batch = b[i * BATCH_SIZE : (i + 1) * BATCH_SIZE]

        DD_opt = jax.jit(lambda x: HPD_opt(D_batch, x, 1))
        NN_M_opt = jax.jit(lambda x: HPD_opt(M_batch, x, p))


        pcg_state1, hist_no_pc, no_pc_time = solve(
            DD_opt, b_batch, max_iters=num_iter, tol=1e-8
        )
        # apply neural
        pcg_state2, hist_nn_pc, nn_pc_time = solve(
            DD_opt, b_batch, max_iters=num_iter, tol=1e-8, M=NN_M_opt
        )

        unpred_hist_list.append(hist_no_pc)
        nnpred_hist_list.append(hist_nn_pc)




    # apply IC preconditioner
    if use_ichol:
        print("Loading precomputed ICHOL matrices...")
        Ichol_path = Path(
            "./data/U1Configs/matrices/"
        )


        L0_name = kwargs["data_name"].replace("config", "L0-matrices")
        L0 = np.load(Ichol_path / L0_name)

        # process batches of 2 matrices
        max_iter = 0
        batch_size = 1
        num_batches = L0.shape[0] // batch_size
        hist_list_IC = []
        for i in tqdm(range(num_batches), desc="IC preconditioner solve"):
            L0_batch = jnp.array(L0[i * batch_size : (i + 1) * batch_size])

            D_batch = Dirac_Matrix(
                U1[i * batch_size : (i + 1) * batch_size], kappa=kappa
            )
            DD_opt_batch = jax.jit(lambda x: HPD_opt(D_batch, x, 1))
            b_batch = b[i * batch_size : (i + 1) * batch_size]

            ichol_M_batch = partial(batched_ichol_solve, L0_batch)

            _, hist_IC_pc_batch, IC_pc_time_batch = solve(
                DD_opt_batch,
                b_batch,
                num_iter,
                1e-8,
                0.0,
                ichol_M_batch,
            )
            if len(hist_IC_pc_batch) > max_iter:
                max_iter = len(hist_IC_pc_batch)
            hist_list_IC.append(hist_IC_pc_batch.tolist())

    else:

        hist_list_IC = [[0]]
        IC_pc_time = 0
    return (unpred_hist_list, nnpred_hist_list, hist_list_IC)


class EvenOddPreconditioner:
    def __init__(self, D_op, inner_max_iters=10, inner_tol=1e-4):
        self.D_op = D_op
        self.kappa = D_op.kappa
        self.inner_max_iters = inner_max_iters
        self.inner_tol = inner_tol

        # even-site mask, shape (1, X, T, 1) for broadcasting
        x_idx, y_idx = jnp.meshgrid(
            jnp.arange(D_op.lattice_shape[0]),
            jnp.arange(D_op.lattice_shape[1]),
            indexing="ij",
        )
        even_mask = ((x_idx + y_idx) % 2 == 0)[..., None]
        self.even_mask = even_mask[None, ...]

    def inner_solve(self, r_even):
        B = r_even.shape[0]
        x = jnp.zeros_like(r_even)
        r = r_even

        def A(x_):
            tmp = self.D_op.apply(x_, dagger=True)
            tmp = self.D_op.apply(tmp, dagger=False)
            return x_ - self.kappa**2 * tmp

        p = z = r
        gamma = jax.vmap(lambda a, b: jnp.vdot(a, b))(
            r.reshape(B, -1), z.reshape(B, -1)
        )

        eps = 1e-10

        def cond(state):
            x, r, p, gamma, i = state
            rnorm = jnp.linalg.norm(r.reshape(B, -1), axis=-1)
            return (i < self.inner_max_iters) & (rnorm > self.inner_tol).any()

        def body(state):
            x, r, p, gamma, i = state
            Ap = A(p)
            Ap_dot_p = jax.vmap(lambda a, b: jnp.vdot(a, b))(
                p.reshape(B, -1), Ap.reshape(B, -1)
            )
            Ap_dot_p = jnp.where(jnp.abs(Ap_dot_p) < eps, eps, Ap_dot_p)
            alpha = (gamma / Ap_dot_p)[:, None, None, None]

            x_new = x + alpha * p
            r_new = r - alpha * Ap
            z_new = r_new
            gamma_new = jax.vmap(lambda a, b: jnp.vdot(a, b))(
                r_new.reshape(B, -1), z_new.reshape(B, -1)
            )

            gamma_safe = jnp.where(jnp.abs(gamma) < eps, eps, gamma)
            beta = (gamma_new / gamma_safe)[:, None, None, None]
            p_new = z_new + beta * p
            return (x_new, r_new, p_new, gamma_new, i + 1)

        state = (x, r, p, gamma, 0)
        x_final, *_ = jax.lax.while_loop(cond, body, state)
        return x_final

    def __call__(self, r):
        r_even = r * self.even_mask
        x_even = self.inner_solve(r_even)
        return x_even * self.even_mask