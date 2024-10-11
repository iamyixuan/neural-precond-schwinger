import logging
import os

import equinox as eqx
import jax
import jax.numpy as jnp
import optax
import torch
from src.model.FNO2d import FNO2d
from src.utils.data import U1pathsDataset
from src.utils.DDOpt import Dirac_Matrix
from src.utils.losses import (construct_matrix, inverse_loss_multiU,
                              inverse_loss_U_paths)
from torch.utils.data import DataLoader

logger = logging.getLogger(__name__)


class ComplexLinear(eqx.Module):
    fc_layer: eqx.nn.Linear

    def __init__(self, *args, **kwargs):
        self.fc_layer = eqx.nn.Linear(*args, **kwargs)

    def __call__(self, x):
        x_real = x.real
        x_imag = x.imag
        return self.fc_layer(x_real) + 1j * self.fc_layer(x_imag)


def condition_number_loss_tilde(model, inputs):
    U1, DD, mask, _ = inputs
    U_tilde = jax.vmap(model)(U1).squeeze()
    U_tilde = U_tilde.reshape(U1.shape[0], 2, 8, 8)
    M = Dirac_Matrix(U_tilde, kappa=0.276)
    M = construct_matrix(
        M, B=U1.shape[0]
    )  # what if taking the lower triangular part of M
    M = model.scale * M + jnp.eye(M.shape[-1])
    MM = jnp.matmul(M, M.conj().transpose((0, 2, 1)))
    precond_sys = jnp.matmul(MM, DD)
    cond_number = jnp.linalg.cond(precond_sys)
    return jnp.mean(cond_number)


def train(
    model: FNO2d,
    trainloader: DataLoader,
    valloader: DataLoader,
    optim: optax.GradientTransformation,
    loss_name: str = "conditionNumber",
    configs: dict = None,
):
    print(f"Training with {loss_name} loss")
    if loss_name == "conditionNumber":
        loss_fn = condition_number_loss_tilde
    elif loss_name == "inverse":
        loss_fn = inverse_loss_U_paths
    else:
        raise NotImplementedError

    opt_state = optim.init(eqx.filter(model, eqx.is_array))

    @eqx.filter_jit
    def update_step(model, inputs, opt_state):
        model = eqx.nn.inference_mode(model, value=False)
        loss, grads = eqx.filter_value_and_grad(loss_fn)(model, inputs)
        updates, opt_state = optim.update(grads, opt_state)
        model = eqx.apply_updates(model, updates)
        return model, opt_state, loss

    @eqx.filter_jit
    def val_step(model, inputs):
        model = eqx.nn.inference_mode(model)
        return loss_fn(model, inputs)

    best_loss = 1e6
    patience = 0
    for _ in range(configs["num_epochs"]):
        running_loss = 0.0
        for i, U1_paths in enumerate(trainloader):
            key = jax.random.PRNGKey(i)
            k1, k2 = jax.random.split(key)
            U1_paths = jnp.asarray(U1_paths)
            inputs = (U1_paths, k2)
            model, opt_state, loss = update_step(model, inputs, opt_state)
            running_loss += loss

        for U_paths_val in valloader:
            U_paths_val = jnp.asarray(U_paths_val)

            inputs_val = (U_paths_val, k2)
            val_loss = val_step(model, inputs_val)

        if val_loss < best_loss:
            best_loss = val_loss
            best_model = model
            patience = 0
            print("Current best at epoch", _)
        else:
            patience += 1

        print(f"train Loss: {running_loss / (i + 1)}; val Loss: {val_loss}")
        # print(f"scale: {model.scale}")

        logger.info(
            f"train Loss: {running_loss / (i + 1)}, val Loss: {val_loss}"
        )

        if patience > 50:
            break

    return best_model


def main(data_path):
    logname = "FNO_U_tilde_paths"
    os.makedirs(f"./logs/{logname}", exist_ok=True)
    logging.basicConfig(
        filename=f"./logs/{logname}/log.txt", level=logging.INFO, filemode="w"
    )

    key = jax.random.PRNGKey(0)
    key, subkey = jax.random.split(key)
    model = FNO2d(
        in_channels=18,
        out_channels=2,
        modes=8,
        h_channels=16,
        activation=eqx.nn.PReLU(),
        n_blocks=4,
        key=key,
    )

    config = {"num_epochs": 1000, "batch_size": 128, "optim": optax.adam(1e-4)}
    # data_path = (
    #     "/home/seswar/Desktop/Academics/Code/nnprecond/datasets/DD_matrices.pt"
    # )
    trainset = U1pathsDataset(data_path, mode="train")
    valset = U1pathsDataset(data_path, mode="val")
    logger.info(
        f"Train size {trainset.__len__()}, Val size {valset.__len__()}"
    )

    trainloader = DataLoader(trainset, batch_size=config["batch_size"])
    valloader = DataLoader(valset, batch_size=valset.__len__())

    model = train(
        model,
        trainloader,
        valloader,
        config["optim"],
        "inverse",
        config,
    )
    eqx.tree_serialise_leaves(f"./logs/{logname}/model.eqx", model)


if __name__ == "__main__":
    data_path = "../data/U1_paths.pt"
    main(data_path)
