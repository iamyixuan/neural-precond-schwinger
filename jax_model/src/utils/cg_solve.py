import time
from typing import NamedTuple

import jax
import jax.numpy as jnp
import numpy as np

from .DDOpt import Dirac_Matrix
from .losses import HPD_opt
from .ichols import ichol0
from .metrics import construct_matrix

# from jax import config
# config.update("jax_enable_x64", True)
def forward_substitution(L, b):
    """Solve L * y = b using forward substitution."""
    y = []
    b = b.ravel()
    for i in range(len(b)):
        y_i = (b[i] - jnp.vdot(L[i, :i], jnp.array(y))) / L[i, i]
        y.append(y_i)
    return jnp.array(y)

def back_substitution(U, y):
    """Solve U * x = y using back substitution."""
    x = []
    y = y.ravel()
    n = len(y)
    for i in range(n - 1, -1, -1):
        x_i = (y[i] - jnp.vdot(U[i, i + 1:], jnp.array(x[::-1]))) / U[i, i]
        x.append(x_i)
    return jnp.array(x[::-1])

@jax.vmap
def preconditioner(L, U, v):
    """Apply ILU preconditioner M^{-1} to vector v."""
    y = forward_substitution(L, v)  # Solve L * y = v
    z = back_substitution(U, y)    # Solve U * z = y
    
    X = int(jnp.sqrt(L.shape[0] / 2))
    return z.reshape(X, X, 2)


def cg_solve(model, U1):
    """
    CG solve with different preconditioners
    Args:
        model: trained model
        U1: batched gauge config
    """
    U_tilde = jax.vmap(model)(U1).squeeze()

    D = Dirac_Matrix(U1, kappa=0.276)
    M = Dirac_Matrix(U_tilde, kappa=0.276)

    key = jax.random.PRNGKey(0)
    b_real = jax.random.normal(key, (U1.shape[0], U1.shape[2], U1.shape[3], 2))
    b_imag = jax.random.normal(key, (U1.shape[0], U1.shape[2], U1.shape[3], 2))
    b = b_real + 1j * b_imag

    # time the two methods
    num_iter = 2000
    start = time.time()
    pcg_state1, hist_no_pc, no_pc_time = solve(lambda x: HPD_opt(D, x), b, num_iter, 1e-8, 0.0)
    end = time.time()

    print(f"Time taken for cg: {end - start}")
    print(hist_no_pc[-1].mean())

    # apply neural PC
    pcg_state2, hist_nn_pc, nn_pc_time = solve(
        lambda x: HPD_opt(D, x), b, num_iter, 1e-8, 0.0, lambda x: HPD_opt(M, x)
    )
    end = time.time()
    print(f"Time taken for pcg: {end - start}")
    print(hist_nn_pc[-1].mean())
    print(len(hist_nn_pc))

    # apply IC preconditioner
    """
    To apply ichol PC we need to 
        1. compute the matrices 
        2. perfrom ichol decomposition
        3. construct a solve as M_inv * x 
        4. pass this solve operator as the preconditioner to the CG solver
    """
    DD_mats = construct_matrix(
        lambda x: HPD_opt(D, x),
        U1.shape[0],
        L=U1.shape[2],
    )

    L0 = ichol0(DD_mats) 
    U0 = L0.conj().transpose((0, 2, 1))
    ichol_M = lambda x: preconditioner(L0, U0, x)
    
    print(DD_mats.shape, L0.shape, U0.shape)
    pcg_state3, hist_IC_pc, IC_pc_time = solve(
        lambda x: HPD_opt(D, x), b, num_iter, 1e-8, 0.0, ichol_M
    )


    print(hist_IC_pc[-1].mean())
    print(len(hist_IC_pc))
    return (hist_no_pc, hist_nn_pc, hist_IC_pc), (no_pc_time, nn_pc_time, IC_pc_time)
    


"""
    Preconditioned Congugate Gradient solver
    Adopted and modified from
    https://towardsdatascience.com/implementing-linear-operators-in-python-with-google-jax-c56be3a966c2
"""


def _identity(x):
    return x


class PCGState(NamedTuple):
    x: jnp.ndarray
    r: jnp.ndarray
    p: jnp.ndarray
    gamma: jnp.ndarray
    iterations: int


def solve_from(A, b, x0, max_iters=20, tol=1e-4, atol=0.0, M=_identity):
    """
    Modified version should directly work with batched systems,
    the stopping criterion is based on the worst case scenario

    Args:
        A: Linear operator that acts on vectors of shape (B, X, T, 2)
        b: RHS of the linear system of shape (B, X, T, 2)
        x0: Initial guess of the solution of shape (B, X, T, 2)
        max_iters: Maximum number of iterations
        tol: Relative tolerance, if all residuals satisfies in the batch
        atol: Absolute tolerance, if all residuals satisfies in the batch

    """
    # Boyd Conjugate Gradients slide 22
    b_norm_sqr = jnp.vdot(b, b)
    max_gamma = jnp.maximum(jnp.square(tol) * b_norm_sqr, jnp.square(atol))
    B = b.shape[0]

    def init():
        r0 = b - A(x0)
        p0 = z0 = M(r0)
        gamma = jax.vmap(jnp.vdot)(r0.reshape(B, -1), z0.reshape(B, -1))
        return PCGState(x=x0, r=r0, p=p0, gamma=gamma, iterations=1)

    def body(state):
        p = state.p
        Ap = A(p)
        alpha = (state.gamma / jax.vmap(jnp.vdot)(p.reshape(B, -1), Ap.reshape(B, -1)))[
            :, None, None, None
        ]

        x = state.x + alpha * p
        r = state.r - alpha * Ap
        z = M(r)
        gamma = jax.vmap(jnp.vdot)(r.reshape(B, -1), z.reshape(B, -1))
        beta = (gamma / state.gamma)[..., None, None, None]
        p = z + beta * p
        return PCGState(x=x, r=r, p=p, gamma=gamma, iterations=state.iterations + 1), (
            x,
            r,
            p,
            gamma,
            state.iterations + 1,
        )

    state = init()
    gamma = state.gamma
    rnorm_hist = []
    start_time = time.time()
    while (gamma > max_gamma).any() & (state.iterations < max_iters):
        state, (x, r, p, gamma, iterations) = body(state)
        gamma = (
            state.gamma
            if M is _identity
            else jax.vmap(jnp.vdot)(r.reshape(B, -1), r.reshape(B, -1))
        )
        rnorm = jnp.linalg.norm(r.reshape(B, -1), axis=-1)
        rnorm_hist.append(rnorm)
    end_time = time.time()
    time_taken = end_time - start_time
    return state, rnorm_hist, time_taken


# solve_from_jit = jit(
#     solve_from, static_argnames=("A", "max_iters", "tol", "atol", "M")
# )


def solve(A, b, max_iters=20, tol=1e-4, atol=0.0, M=_identity):
    x0 = jnp.zeros_like(b)
    return solve_from(A, b, x0, max_iters, tol, atol, M)
