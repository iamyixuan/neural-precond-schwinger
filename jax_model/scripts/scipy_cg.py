import numpy as np
import pickle
from scipy.sparse.linalg import cg
import scipy.sparse as sp
import scipy.sparse.linalg as spla
from tqdm import tqdm
import jax
from pathlib import Path
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--model", type=str, default="FNO")
parser.add_argument("--ML", type=int, default=16, help="Lattice size")
parser.add_argument("--DL", type=int, default=64, help="Lattice size")
parser.add_argument("--kappa", type=float, default=0.276)
parser.add_argument("--beta", type=float, default=2.0)
parser.add_argument(
    "--precond",
    type=str,
    default="unprecond",
    choices=["unprecond", "nn_pc", "ic_pc"],
    help="Preconditioner type",
)
args = parser.parse_args()

if args.precond == "unprecond":
    matrix_dir = Path(
        "./data/U1Configs/matrices/"
    )
    batch_size = 200
    save_name = "unprecond"
elif args.precond == "nn_pc":
    matrix_dir = Path(
        "./data/U1Configs/matrices/"
    )
    M_inv_dir = Path(
        "./data/U1Configs/matrices/L64_NN_precond/"
    )
    save_name = "nnprecond"
    batch_size = 200 // 50
elif args.precond == "ic_pc":
    matrix_dir = Path(
        "./data/U1Configs/matrices/"
    )
    L0_dir = Path(
        "./data/U1Configs/ichols/"
    )
    batch_size = 200 // 10
    save_name = "iCholprecond"
else:
    raise ValueError(f"Unknown preconditioner type: {args.precond}")


save_dir = f"plot_data/{args.model}/ML{args.ML}_p1/DL{args.DL}/kappa_{args.kappa}/beta_{args.beta}"


def solve_with_count(A, b, x0=None, rtol=1e-8, M=None, maxiter=None):
    """
    Solve Ax = b using conjugate gradient and return the number of iterations.
    """
    iters = 0
    b_norm = np.linalg.norm(b)
    residuals = []

    def cb_residual(xk):
        nonlocal iters, residuals
        iters += 1
        r = b - A.dot(xk)
        residuals.append(np.linalg.norm(r))

    if M is not None:
        n = A.shape[0]
        M = spla.LinearOperator(
            (n, n), matvec=apply_Minv, rmatvec=apply_Minv, dtype=A.dtype
        )
        x, info = cg(
            A, b, M=M, x0=x0, rtol=rtol, maxiter=maxiter, callback=cb_residual
        )
    else:
        x, info = cg(
            A, b, x0=x0, rtol=rtol, maxiter=maxiter, callback=cb_residual
        )
    return residuals, iters


cg_residuals = []

key = jax.random.PRNGKey(0)
b_real = jax.random.normal(key, (200, 64, 64, 2))
b_imag = jax.random.normal(key, (200, 64, 64, 2))
b = b_real + 1j * b_imag
b = np.array(b).reshape(200, -1)

for batch in range(200 // batch_size):
    print("batch size", batch_size, "number of batches", 200 // batch_size)

    if args.precond == "unprecond":
        datapath = matrix_dir / "matrices.l64-N200-b2.0-k0.276-unquenched.npy"
    elif args.precond == "nn_pc":
        datapath = matrix_dir / "matrices.l64-N200-b2.0-k0.276-unquenched.npy"
        M_inv_path = (
            M_inv_dir
            / f"M_inv.{args.model}.ML{args.ML}.L64.kappa0.276.beta2.0.batch{batch}.npy"
        )
        M_invs = np.load(M_inv_path, mmap_mode="r")
    elif args.precond == "ic_pc":
        datapath = (
            matrix_dir
            / f"matrices.l64-N200-b2.0-k0.276-unquenched-part-{batch}.npy"
        )
        L0path = (
            L0_dir
            / f"L0-matrices.l64-N200-b2.0-k0.276-unquenched-part-{batch}.npy"
        )

    mats = np.load(
        datapath, mmap_mode="r"
    )  # will be equal to batch size except for the nn_pc case
    L0s = np.load(L0path, mmap_mode="r") if args.precond == "ic_pc" else None

    for i in tqdm(range(batch_size)):
        if args.precond == "ic_pc":

            mat = sp.csr_matrix(mats[i])
            L0 = sp.csr_matrix(L0s[i])

            def apply_Minv(r: np.ndarray) -> np.ndarray:
                # Solve (L L^T) z = r   -> forward, then back substitution
                y = spla.spsolve_triangular(
                    L0, r, lower=True, unit_diagonal=False
                )
                z = spla.spsolve_triangular(
                    L0.conj().T, y, lower=False, unit_diagonal=False
                )
                return z

            residuals, iters = solve_with_count(
                mat,
                b[batch * batch_size + i],
                M=apply_Minv,
            )
            print(
                f"Matrix {i} in batch {batch} solved in {iters}/{len(residuals)} iterations."
            )

        elif args.precond == "nn_pc":
            assert mats.shape[0] == 200
            mat = sp.csr_matrix(mats[batch * batch_size + i])
            print(f"batch- {batch}, in batch idx-{i}, global idx-{batch * batch_size + i}")
            M_inv = sp.csr_matrix(M_invs[i])

            def apply_Minv(r: np.ndarray) -> np.ndarray:
                return M_inv.dot(r)

            residuals, iters = solve_with_count(
                mat,
                b[batch * batch_size + i],
                M=apply_Minv,
            )
            print(
                f"Matrix {i} in batch {batch} solved in {iters}/{len(residuals)} iterations."
            )

        else:
            mat = sp.csr_matrix(mats[i])
            residuals, iters = solve_with_count(mat, b[i])

        cg_residuals.append(residuals)


with open(f"{save_dir}/{save_name}_cg_hist.pickle", "wb") as f:
    pickle.dump(cg_residuals, f)
