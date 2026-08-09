import equinox as eqx
import argparse
import jax
import jax.numpy as jnp
import numpy as np
from src.utils.DDOpt import Dirac_Matrix, EvenOddPreconditionedOperator 
from src.utils.losses import HPD_opt
from src.utils.cg_solve import solve, EvenOddPreconditioner
from src.utils.metrics import load_model
from src.model.complexCNN import PrecondCNN
from src.model.FNO2d import FNO2d
from pathlib import Path


def cg_solve(model, U1, kappa, p=1, **kwargs):
    """
    CG solve with different preconditioners
    Args:
        model: trained model
        U1: batched gauge config
    """
    U_tilde = jax.vmap(model)(U1).squeeze()

    D = Dirac_Matrix(U1, kappa=kappa)
    M = Dirac_Matrix(U_tilde, kappa=kappa)
    D_evenOdd = EvenOddPreconditionedOperator(D)

    key = jax.random.PRNGKey(0)
    b_real = jax.random.normal(key, (U1.shape[0], U1.shape[2], U1.shape[3], 2))
    b_imag = jax.random.normal(key, (U1.shape[0], U1.shape[2], U1.shape[3], 2))
    b = b_real + 1j * b_imag

    # time the two methods
    num_iter = 2000
    #
    DD_opt = jax.jit(lambda x: HPD_opt(D, x, 1))
    NN_M_opt = jax.jit(lambda x: HPD_opt(M, x, p))
    M_evenOdd = EvenOddPreconditioner(D)

    # apply neural
    pcg_state2, hist_nn_pc, nn_pc_time = solve(
        DD_opt, b, max_iters=num_iter, tol=1e-8, M=NN_M_opt
    )
    pcg_state3, hist_evenOdd_pc, evenOdd_pc_time = solve(
        D_evenOdd, b, max_iters=num_iter, tol=1e-8 
    )
    return (hist_nn_pc, hist_evenOdd_pc), (
        nn_pc_time,
        evenOdd_pc_time,
    )


def main(args, configs):
    # log_dir = f"./logs/FNO_L{args.ML}_{args.loss}Precond_invloss/"
    if args.model == "FNO":
        model = load_model(configs, FNO2d, args.log_dir)
    elif args.model == "PrecondCNN":
        model = load_model(configs, PrecondCNN, args.log_dir)
    else:
        raise ValueError("Unknown model type")

    if args.DL == 64:
        data_name = (
            f"config.l{args.DL}-N32-b{args.beta}-k{args.kappa}-unquenched.npy"
        )
    else:
        data_name = (
            f"config.l{args.DL}-N200-b{args.beta}-k{args.kappa}-unquenched.npy"
        )

    data_dir = Path("../data/U1Configs/")
    data_path = data_dir / data_name

    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    U1 = jnp.load(data_path)
    U1_org = U1.copy()
    U1 = jnp.exp(1j * U1)

    hist, time = cg_solve(model, U1, kappa=args.kappa, p=args.p, **configs)
    print(
        "nn pc hist length:",
        len(hist[0]),
        "evenOdd pc hist length:",
        len(hist[1]),
    )
    print("nn pc time:", time[0]),
    print("evenOdd pc time:", time[1])
    np.savez(
        save_dir / "even_Oddhist.npz",
        hist_nn_pc=hist[0],
        hist_evenOdd_pc=hist[1],
        time_nn_pc=time[0],
        time_evenOdd_pc=time[1],
    )


if __name__ == "__main__":

    jax.config.update("jax_enable_x64", True)

    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="FNO")
    parser.add_argument("--ML", type=int, default=8, help="Lattice size")
    parser.add_argument("--DL", type=int, default=8, help="Lattice size")
    parser.add_argument("--loss", type=str, default="single")
    parser.add_argument("--kappa", type=float, default=0.276)
    parser.add_argument("--beta", type=float, default=2.0)
    parser.add_argument("--log_dir", type=str, default="./logs/")
    parser.add_argument("--p", type=int, default=1)

    args = parser.parse_args()
    args.save_dir = f"plot_data/{args.model}/ML{args.ML}_p{args.p}/DL{args.DL}/kappa_{args.kappa}/beta_{args.beta}"

    configs_CNN = {
        "inch": 2,
        "outch": 2,
        "activation": eqx.nn.PReLU(),
        "kernel_size": 3,
        "n_layers": 4,
        "hidden_dim": 16,
        "key": jax.random.PRNGKey(0),
    }
    configs_fno = {
        "in_channels": 2,
        "out_channels": 2,
        "modes": 8,
        "h_channels": 16,
        "activation": eqx.nn.PReLU(),
        "n_blocks": 4,
        "key": jax.random.PRNGKey(0),
    }

    if args.model == "FNO":
        main(args, configs_fno)
    elif args.model == "PrecondCNN":
        main(args, configs_CNN)
