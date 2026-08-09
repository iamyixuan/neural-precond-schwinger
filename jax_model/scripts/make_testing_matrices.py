from pathlib import Path
import jax
import jax.numpy as jnp
import numpy as np
import argparse
from src.utils.DDOpt import Dirac_Matrix


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


def HPD_opt(Dirac_Matrix_inst, x, power=4):
    for p in range(power):
        x = Dirac_Matrix_inst.apply(Dirac_Matrix_inst.apply(x), dagger=True)
    return x


def get_test_matrices(U1, kappa):
    D = Dirac_Matrix(U1, kappa=kappa)
    DD_opt = jax.jit(lambda x: HPD_opt(D, x, 1))
    print(f"Constructing matrices from U1 of size {U1.shape}...")
    DD_mats = construct_matrix(
        DD_opt,
        U1.shape[0],
        L=U1.shape[2],
    )
    print(f"Matrix construction done... of shape {DD_mats.shape}")
    return DD_mats


if __name__ == "__main__":
    jax.config.update("jax_enable_x64", True)

    parser = argparse.ArgumentParser()
    parser.add_argument("--DL", type=int, default=8, help="Lattice size")
    parser.add_argument("--kappa", type=float, default=0.276)
    parser.add_argument("--beta", type=float, default=2.0)

    args = parser.parse_args()
    args.save_dir = f"../"

    # if args.DL == 64:
    #     data_name = (
    #         f"config.l{args.DL}-N32-b{args.beta}-k{args.kappa}-unquenched.npy"
    #     )
    data_name = (
        f"config.l{args.DL}-N200-b{args.beta}-k{args.kappa}-unquenched.npy"
    )
    data_dir = Path("../../../data/U1Configs/")
    mat_dir = data_dir / "matrices"
    mat_dir.mkdir(parents=True, exist_ok=True)
    data_path = data_dir / data_name

    U1 = jax.numpy.load(data_path)
    U1 = jnp.exp(1j * U1)

    if U1.shape[-1] <= 16:
        DD_matrices = get_test_matrices(U1, args.kappa)

        # save the matrices
        print(f"Saving matrices (shape {DD_matrices.shape})")
        np.save(
            mat_dir / data_name.replace("config", "matrices", 1),
            DD_matrices,
        )
    else:
        print("Large sizes, saving by chunks...")
        if U1.shape[-1] == 64:
            chunk_size = 2
        else:
            chunk_size = 10
        num_chunks = U1.shape[0] // chunk_size

        mm = np.lib.format.open_memmap(
            mat_dir / data_name.replace("config", "matrices-last-two", 1),
            mode="w+",
            dtype=np.complex128,
            # shape=(U1.shape[0], U1.shape[-1] ** 2 * 2, U1.shape[-1] ** 2 * 2),
            shape=(2, U1.shape[-1] ** 2 * 2, U1.shape[-1] ** 2 * 2),
        )

        for i in range(num_chunks):
            start = i * chunk_size
            end = (i + 1) * chunk_size
            if i == num_chunks - 1:
                print(start, end)
                print(U1.shape[0])
                print(f"Processing chunk {i + 1}/{num_chunks}...")
                U1_chunk = U1[start:end]
                DD_matrices = get_test_matrices(U1_chunk, args.kappa)
                print(mm.shape, DD_matrices.shape)
                mm[:] = DD_matrices
            else:
                pass
        mm.flush()
        del mm
