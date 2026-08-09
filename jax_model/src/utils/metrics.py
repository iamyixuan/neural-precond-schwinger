import equinox as eqx
import jax
import jax.numpy as jnp
from hilbertcurve.hilbertcurve import HilbertCurve
from .DDOpt import Dirac_Matrix
import numpy as np


def construct_matrix(opt, B, L=8):
    n = L * L * 2
    identity = jnp.identity(n)
    B_identity = jnp.repeat(identity[None, ...], B, axis=0)
    columns = []

    @jax.jit
    def apply_opt(i):
        e_i = B_identity[:, :, i]
        e_i = e_i.reshape(B, L, L, 2)
        return opt(e_i).reshape(B, n)

    col_indice = jnp.arange(n)
    columns = jax.vmap(apply_opt)(col_indice)
    M = jnp.transpose(columns, (1, 2, 0))
    return M


def hilbert_permutations(side: int):
    """
    Return (perm, inv) for an `side` × `side` grid.
    side must be 2^p  (8,16,32, …).
    """
    # Verify power‑of‑two
    if side & (side - 1):
        raise ValueError("side length must be a power of two (8,16,32,...)")

    p = int(np.log2(side))  # recursion level
    h = HilbertCurve(p, 2)  # 2‑D curve

    perm = np.empty(side * side, dtype=np.int32)
    for d in range(side * side):
        x, y = h.point_from_distance(d)
        perm[d] = y * side + x  # row‑major index

    inv = np.empty_like(perm)
    inv[perm] = np.arange(side * side)
    return perm, inv


# ------------------------------------------------------------
# 2.  Generic flatten / unflatten
# ------------------------------------------------------------
def hilbert_flatten(img_hwC: np.ndarray):
    """
    img_hwC :  (H, W, C)  with H==W==power‑of‑two
    returns  :  (H*W, C)  in Hilbert order
    """
    H, W, C = img_hwC.shape
    if H != W:
        raise ValueError("H and W must be equal for this helper.")
    perm, _ = hilbert_permutations(H)
    return img_hwC.reshape(-1, C)[perm]


def hilbert_unflatten(flat_NC: np.ndarray, side: int):
    """
    flat_NC : (side*side, C)
    side    : original image side length (8,16,32,…)
    returns : (side, side, C)
    """
    _, inv = hilbert_permutations(side)
    C = flat_NC.shape[-1]
    return flat_NC[inv].reshape(side, side, C)


##################################################


@jax.jit
def compute_condition_number(A):
    return jnp.linalg.cond(A)


def get_batch_matrix(f, b_size, v_size=128):
    """construct the matrix form of an operator
    Args:
        f: function that takes a batch of vectors and returns a batch of vectors
        v_size: size of the vector
    """
    identity = jnp.identity(v_size)
    identity = jnp.repeat(identity[None, ...], b_size, axis=0)
    M = jax.vmap(f, in_axes=-1, out_axes=-1)(identity)
    M = M.reshape(M.shape[0], v_size, v_size)
    return M


def load_model(configs, model, checkpoint):
    model = model(**configs)
    model = eqx.tree_deserialise_leaves(checkpoint + "model.eqx", model)
    model = eqx.nn.inference_mode(model)
    return model


# def construct_matrix(opt, B, L=8):
#     n = L * L * 2
#     identity = jnp.identity(n)
#     B_identity = jnp.repeat(identity[None, ...], B, axis=0)
#     columns = []
#
#     @jax.jit
#     def apply_opt(i):
#         e_i = B_identity[:, :, i]
#         e_i = e_i.reshape(B, L, L, 2)
#         return opt(e_i).reshape(B, n)
#
#     col_indice = jnp.arange(n)
#     columns = jax.vmap(apply_opt)(col_indice)
#     M = jnp.transpose(columns, (1, 2, 0))
#     return M


def construct_Dirac_Matrix(U1, v, kappa=0.276):
    if U1.shape[-3:] != (v, v, 2):
        U1 = U1.reshape(U1.shape[0], 2, v, v)
    return Dirac_Matrix(U1, kappa=kappa)


def estimate_flops(fn, *args, **kwargs):
    """
    Estimates total FLOPS for a JAX function given its inputs.

    Args:
        fn: The JAX function to analyze.
        *args, **kwargs: Dummy inputs matching the function's signature.

    Returns:
        float: Estimated number of floating point operations.
    """
    lowered = jax.jit(fn).lower(*args, **kwargs)
    analysis = lowered.cost_analysis()
    if analysis is None:
        raise RuntimeError(
            "Cost analysis failed. Ensure the function is JIT-compatible."
        )

    # 'flops' is the standard key returned by XLA cost analysis
    return analysis.get("flops", 0.0)
