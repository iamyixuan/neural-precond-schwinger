import argparse

import equinox as eqx
import jax
import jax.numpy as jnp
from precondFNN_U_tilde import PrecondFNN, U1DDDataset
from src.model.FNO2d import FNO2d
from src.utils.metrics import (compute_condition_number,
                               construct_Dirac_Matrix, get_batch_matrix,
                               load_model)
from torch.utils.data import DataLoader 
from src.utils.data import U1pathsDataset


def main(args, configs, network="FNO", if_u_paths=False):
    if network == "FNO":
        model = load_model(configs, FNO2d, args.checkpoint)
    else:
        model = load_model(configs, PrecondFNN, args.checkpoint)

    if if_u_paths:
        valset = U1pathsDataset(args.data_path, mode="val")
        valloader = DataLoader(valset, batch_size=valset.__len__())
        for data in valloader:
            U_paths = jnp.asarray(data)
            U1 = U_paths[:, :2, ...]
            U_tilde = jax.vmap(model)(U_paths).squeeze()
    else:
        valset = U1DDDataset(args.data_path, mode="val")
        valloader = DataLoader(valset, batch_size=valset.__len__())
        for data in valloader:
            U1, DD, mask = data
            U1 = jnp.asarray(U1)
            DD = jnp.asarray(DD)
            mask = jnp.nonzero(jnp.array(mask))
            U_tilde = jax.vmap(model)(U1).squeeze()

    if network != "FNO":
        U_tilde = U_tilde.reshape(U1.shape[0], 2, 8, 8)
    print(U1.shape, U_tilde.shape)
    assert len(U_tilde.shape) == 4

    M = construct_Dirac_Matrix(U_tilde)
    D = construct_Dirac_Matrix(U1)

    def f_org(x):
        if x.shape[-3:] != (8, 8, 2):
            x = x.reshape(x.shape[0], 8, 8, 2)
        Dx = D.apply(D.apply(x), dagger=True)
        return Dx

    def f_precond(x):
        if x.shape[-3:] != (8, 8, 2):
            x = x.reshape(x.shape[0], 8, 8, 2)
        Dx = D.apply(D.apply(x), dagger=True)
        MDx = M.apply(M.apply(Dx), dagger=True)
        return MDx

    org_mat = get_batch_matrix(f_org, b_size=U1.shape[0])
    precond_mat = get_batch_matrix(f_precond, b_size=U1.shape[0])
    org_cond_number = compute_condition_number(org_mat)
    cond_number = compute_condition_number(precond_mat)
    print(f"Original condition number: {org_cond_number.mean()}")
    print(f"Condition number: {cond_number.mean()}")
    return org_cond_number, cond_number


if __name__ == "__main__":
    from plot import plot_hist, plot_sorted_scatter, plot_train, plt, read_log

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--checkpoint",
        type=str,
        default="./logs/U1_FNN_U_tilde_full_inverse_loss_multiU_tilde/",
        help="Model checkpoint",
    )
    parser.add_argument(
        "--data_path",
        type=str,
        default="../data/U1_DD_matrices.pt",
        help="Data path",
    )
    args = parser.parse_args()
    model_name = args.checkpoint.split("/")[-2]
    print(f"Model name: {model_name}")
    configs = {
        "key": jax.random.PRNGKey(0),
        "in_dim": 128,
        "out_dim": 128,
        "activation": eqx.nn.PReLU(),
        "layer_sizes": [1024] * 3,
    }
    configs_fno = {
        "in_channels": 18,
        "out_channels": 2,
        "modes": 8,
        "h_channels": 16,
        "activation": eqx.nn.PReLU(),
        "n_blocks": 4,
        "key": jax.random.PRNGKey(0),
    }

    org, pred = main(args, configs_fno, network="FNO", if_u_paths=True)
    fig, ax = plot_hist([org, pred], ["Original", "Preconditioned"])
    fig.savefig(f"../figures/{model_name}_condNum_hist.pdf", bbox_inches="tight")
    fig, ax = plot_sorted_scatter([org, pred], ["Original", "Preconditioned"])
    fig.savefig(
        f"../figures/{model_name}_sorted_scatter.pdf",
        bbox_inches="tight",
    )

    train_loss, val_loss, scale = read_log(args.checkpoint + "/log.txt")
    fig, ax = plot_train(train_loss, val_loss)
    fig.savefig(
        f"../figures/{model_name}_loss_curves.pdf",
        bbox_inches="tight",
    )
