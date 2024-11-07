import logging
import os

import equinox as eqx
import jax
import jax.numpy as jnp
import optax
from src.model.FNO2d import FNO2d
from src.utils.data import RawU1Dataset
from src.utils.losses import inverse_loss
from torch.utils.data import DataLoader
from jax import config
config.update("jax_enable_x64", True)

logger = logging.getLogger(__name__)


def train(
    model: FNO2d,
    trainloader: DataLoader,
    valloader: DataLoader,
    optim: optax.GradientTransformation,
    configs: dict = None,
):
    loss_fn = inverse_loss

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
        for i, U1 in enumerate(trainloader):
            key = jax.random.PRNGKey(i)
            k1, k2 = jax.random.split(key)
            U1 = jnp.asarray(U1)
            inputs = (U1, k1)
            model, opt_state, loss = update_step(model, inputs, opt_state)
            running_loss += loss

        for U1_val in valloader:
            U1_val = jnp.asarray(U1_val)

            inputs_val = (U1_val, k2)
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


def main(args):
    data_dir = "../data/U1Configs/"
    if args.L == 8:
        data_name = "config.l8-N1600-b2.0-k0.276-unquenched.x.npy"
    elif args.L == 16:
        data_name = "config.l16-N200-b2.0-k0.276-unquenched-test.x.npy"
    elif args.L == 32:
        data_name = "config.l32-N200-b2.0-k0.276-unquenched-test.x.npy"
    elif args.L == 64:
        data_name = "config.l64-N200-b2.0-k0.276-unquenched-test.x.npy"

    data_path = os.path.join(data_dir, data_name)

    logname = f"FNO_L{args.L}_inverse_loss_doubleP"
    os.makedirs(f"./logs/{logname}", exist_ok=True)
    logging.basicConfig(
        filename=f"./logs/{logname}/log.txt", level=logging.INFO, filemode="w"
    )

    key = jax.random.PRNGKey(0)
    key, subkey = jax.random.split(key)
    model = FNO2d(
        in_channels=2,
        out_channels=2,
        modes=8,
        h_channels=16,
        activation=eqx.nn.PReLU(),
        n_blocks=4,
        key=key,
    )

    config = {"num_epochs": 10000, "batch_size": 128, "optim": optax.adam(1e-4)}
    trainset = RawU1Dataset(data_path, mode="train")
    valset = RawU1Dataset(data_path, mode="val")
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
        config,
    )
    eqx.tree_serialise_leaves(f"./logs/{logname}/model.eqx", model)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()

    parser.add_argument("--L", type=int, default=8)
    args = parser.parse_args()

    main(args)
