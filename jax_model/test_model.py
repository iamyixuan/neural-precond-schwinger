import argparse
import os

import equinox as eqx
import jax
import jax.numpy as jnp
from precondFNN_U_tilde import PrecondFNN, U1DDDataset
from src.model.FNO2d import FNO2d
from src.utils.cg_solve import cg_solve
from src.utils.data import U1pathsDataset
from src.utils.metrics import (compute_condition_number,
                               construct_Dirac_Matrix, get_batch_matrix,
                               load_model)
from torch.utils.data import DataLoader


def I_Cholesky_M_invA(A):
    L = jnp.linalg.cholesky(A)
    L0 = jnp.multiply(A != 0.0, L)
    L0_inv = jnp.linalg.inv(L0)
    M_inv = L0_inv.conj().T @ L0_inv
    return M_inv @ A


def main(args, configs, network="FNO"):
    log_dir = f"./logs/FNO_L{args.model_L}_inverse_loss/"
    if network == "FNO":
        model = load_model(configs, FNO2d, log_dir)
    else:
        model = load_model(configs, PrecondFNN, log_dir)

    if args.data_L == 8:
        data_name = "config.l8-N200-b2.0-k0.276-unquenched-test.x.npy"
    elif args.data_L == 16:
        data_name = "config.l16-N200-b2.0-k0.276-unquenched-test.x.npy"
    elif args.data_L == 32:
        data_name = "config.l32-N200-b2.0-k0.276-unquenched-test.x.npy"
    elif args.data_L == 64:
        data_name = "config.l64-N32-b2.0-k0.276-unquenched-test.x.npy"

    data_dir = "../data/U1Configs/"
    data_path = os.path.join(data_dir, data_name)

    U1 = jnp.load(data_path)
    U1 = jnp.exp(1j * U1)

    hist, time = cg_solve(model, U1)
    no_pc_hist, nn_pc_hist, IC_pc_hist = hist
    no_pc_time, nn_pc_time, IC_pc_time = time
    jnp.savez(
        f"./plot_data/M{args.model_L}_D{args.data_L}_no_pc_hist.npz",
        hist=no_pc_hist,
        time=no_pc_time,
    )
    jnp.savez(
        f"./plot_data/M{args.model_L}_D{args.data_L}_nn_pc_hist.npz", 
        hist=nn_pc_hist,
        time=nn_pc_time,

    )
    jnp.savez(
        f"./plot_data/M{args.model_L}_D{args.data_L}_ic_pc_hist.npy",
        hist=IC_pc_hist,
        time=IC_pc_time,
    )

    assert False

    U_tilde = jax.vmap(model)(U1).squeeze()
    print(U1.shape, U_tilde.shape)

    if network != "FNO":
        U_tilde = U_tilde.reshape(U1.shape[0], 2, 8, 8)
    print(U1.shape, U_tilde.shape)
    assert len(U_tilde.shape) == 4

    X = U1.shape[-1]

    M = construct_Dirac_Matrix(U_tilde, v=X)
    D = construct_Dirac_Matrix(U1, v=X)

    def f_org(x):
        if x.shape[-3:] != (X, X, 2):
            x = x.reshape(x.shape[0], X, X, 2)
        Dx = D.apply(D.apply(x), dagger=True)
        return Dx

    def f_precond(x):
        if x.shape[-3:] != (X, X, 2):
            x = x.reshape(x.shape[0], X, X, 2)
        Dx = D.apply(D.apply(x), dagger=True)
        MDx = M.apply(M.apply(Dx), dagger=True)
        return MDx

    org_mat = get_batch_matrix(f_org, b_size=U1.shape[0], v_size=2 * X**2)
    IC_org_mat = jax.vmap(I_Cholesky_M_invA)(org_mat)
    precond_mat = get_batch_matrix(
        f_precond, b_size=U1.shape[0], v_size=2 * X**2
    )

    IC_cond_number = compute_condition_number(IC_org_mat)
    org_cond_number = compute_condition_number(org_mat)
    cond_number = compute_condition_number(precond_mat)

    q_50_org = jnp.percentile(org_cond_number, 50)
    q_50_nn = jnp.percentile(cond_number, 50)
    q_50_IC = jnp.percentile(IC_cond_number, 50)

    q_10_org = jnp.percentile(org_cond_number, 10)
    q_10_nn = jnp.percentile(cond_number, 10)
    q_10_IC = jnp.percentile(IC_cond_number, 10)

    q_90_org = jnp.percentile(org_cond_number, 90)
    q_90_nn = jnp.percentile(cond_number, 90)
    q_90_IC = jnp.percentile(IC_cond_number, 90)

    print(
        f"Original condition number: {q_10_org:.3f} | {q_50_org:3f} | {q_90_org:.3f}"
    )
    print(f"Condition number: {q_10_nn:.3f} | {q_50_nn:.3f} | {q_90_nn:.3f}")
    print(
        f"IC condition number: {q_10_IC:.3f} | {q_50_IC:.3f} | {q_90_IC:.3f}"
    )
    return org_cond_number, cond_number, IC_cond_number


if __name__ == "__main__":
    from plot import plot_hist, plot_sorted_scatter, plot_train, read_log

    parser = argparse.ArgumentParser()
    parser.add_argument("--model_L", type=int, default=8, help="Lattice size")
    parser.add_argument("--data_L", type=int, default=8, help="Lattice size")

    args = parser.parse_args()
    model_name = f"FNO_L{args.model_L}"

    log_dir = f"./logs/FNO_L{args.model_L}_inverse_loss/"

    print(f"Model name: {model_name}")
    configs = {
        "key": jax.random.PRNGKey(0),
        "in_dim": 128,
        "out_dim": 128,
        "activation": eqx.nn.PReLU(),
        "layer_sizes": [1024] * 3,
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

    org, pred, ic = main(args, configs_fno, network="FNO")
    fig, ax = plot_hist([org, pred], ["Original", "Preconditioned"])
    fig.savefig(
        f"../figures/{model_name}_condNum_hist_L{args.data_L}.pdf",
        bbox_inches="tight",
    )
    fig, ax = plot_sorted_scatter(
        [org, pred, ic], ["Original", "Preconditioned", "IC Preconditioned"]
    )
    fig.savefig(
        f"../figures/{model_name}_sorted_scatter_L{args.data_L}.pdf",
        bbox_inches="tight",
    )

    train_loss, val_loss, scale = read_log(log_dir + "/log.txt")
    fig, ax = plot_train(train_loss, val_loss)
    fig.savefig(
        f"../figures/{model_name}_loss_curves.pdf",
        bbox_inches="tight",
    )
