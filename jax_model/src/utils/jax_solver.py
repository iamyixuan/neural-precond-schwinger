import jax
import jax.numpy as jnp
import time
from typing import NamedTuple

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
    # b_norm_sqr = jax.vmap(jnp.vdot)(b, b) # shape (B,)
    b_norm_sqr = jnp.vdot(b, b)
    max_gamma = jnp.maximum(jnp.square(tol) * b_norm_sqr, jnp.square(atol))
    B = b.shape[0]

    def init():
        r0 = b - A(x0)
        p0 = z0 = M(r0)
        gamma = jax.vmap(jnp.vdot)(r0.reshape(B, -1), z0.reshape(B, -1))
        return PCGState(x=x0, r=r0, p=p0, gamma=gamma, iterations=1)

    # @jax.jit
    def body(state):
        p = state.p
        Ap = A(p)
        alpha = (
            state.gamma
            / jax.vmap(jnp.vdot)(p.reshape(B, -1), Ap.reshape(B, -1))
        )[:, None, None, None]

        x = state.x + alpha * p
        r = state.r - alpha * Ap
        z = M(r)
        gamma = jax.vmap(jnp.vdot)(r.reshape(B, -1), z.reshape(B, -1))
        beta = (gamma / state.gamma)[..., None, None, None]
        p = z + beta * p
        return PCGState(
            x=x, r=r, p=p, gamma=gamma, iterations=state.iterations + 1
        ), (
            x,
            r,
            p,
            gamma,
            state.iterations + 1,
        )

    # state = init()
    # # dry run and time the compilation
    # # print("Dry run for compilation...")
    # s = time.time()
    # state, _ = body(state)
    # e = time.time()
    # reset the state
    state = init()

    gamma = state.gamma
    rnorm_hist = []
    start_time = time.time()
    while (gamma > max_gamma).any() & (state.iterations < max_iters):
        # print(f"Iteration {state.iterations}, max gamma: {gamma.max()}")
        state, (x, r, p, gamma, iterations) = body(state)
        gamma = (
            state.gamma
            if M is _identity
            else jax.vmap(jnp.vdot)(r.reshape(B, -1), r.reshape(B, -1))
        )
        rnorm = jnp.linalg.norm(r.reshape(B, -1), axis=-1)
        rnorm_hist.append(rnorm.squeeze().tolist())

    # _ = state.r.block_until_ready()
    end_time = time.time()
    time_taken = end_time - start_time
    return state, jnp.array(rnorm_hist), time_taken


def solve(A, b, max_iters=20, tol=1e-4, atol=0.0, M=_identity):
    x0 = jnp.zeros_like(b)
    return solve_from(A, b, x0, max_iters, tol, atol, M)