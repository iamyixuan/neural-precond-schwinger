import logging
import os
from omegaconf import OmegaConf
import hydra
from hydra.utils import instantiate
import wandb
import time
from pathlib import Path

import equinox as eqx
import jax
import jax.numpy as jnp
import optax
from src.model.complexCNN import PrecondCNN
from src.utils.data import RawU1Dataset
from src.utils.losses import inverse_loss, inverse_loss_double
from torch.utils.data import DataLoader
from jax import config

config.update("jax_enable_x64", True)

logger = logging.getLogger(__name__)


def train(
    model: PrecondCNN,
    trainloader: DataLoader,
    valloader: DataLoader,
    configs: OmegaConf,
    logdir: Path = None,
):
    data_config = configs.data
    train_config = configs.train
    wandb_config = configs.wandb

    wandb.init(
        project=wandb_config.project,
        name=wandb_config.name,
        config=OmegaConf.to_container(configs, resolve=True),
    )

    loss_fn = lambda model, inputs: inverse_loss(
        model,
        inputs,
        kappa=data_config.kappa,
        power=train_config.power,
        num_v=train_config.num_vec,
    )

    val_loss_fn = lambda model, inputs: inverse_loss(
        model,
        inputs,
        kappa=data_config.kappa,
        power=train_config.power,
        num_v=128,
    )
    optim = optax.adam(train_config.lr)

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
        return val_loss_fn(model, inputs)

    best_loss = 1e6
    patience = 0
    best_model = None
    for _ in range(train_config.num_epochs):
        running_loss = 0.0
        running_val_loss = 0.0
        for i, U1 in enumerate(trainloader):
            key = jax.random.PRNGKey(i)
            k1, k2 = jax.random.split(key)
            U1 = jnp.asarray(U1)
            inputs = (U1, k1)
            model, opt_state, loss = update_step(model, inputs, opt_state)
            running_loss += loss

        for j, U1_val in enumerate(valloader):
            U1_val = jnp.asarray(U1_val)

            inputs_val = (U1_val, k2)
            val_loss = val_step(model, inputs_val)
            running_val_loss += val_loss

        val_loss = running_val_loss / (j + 1)
        if val_loss < best_loss:
            best_loss = val_loss
            best_model = model
            eqx.tree_serialise_leaves(logdir / "best_model.eqx", model)
            patience = 0
            print("Current best at epoch", _)
        else:
            patience += 1

        print(f"train Loss: {running_loss / (i + 1)}; val Loss: {val_loss}")

        wandb.log(
            {
                "train_loss": running_loss / (i + 1),
                "val_loss": val_loss,
                "epoch": _,
            }
        )

        logger.info(
            f"train Loss: {running_loss / (i + 1)}, val Loss: {val_loss}"
        )

        if patience > train_config.patience:
            break
    wandb.finish()

    return best_model

@hydra.main(config_path=".", config_name="config")
def main(args):
    model_config = args.model
    data_config = args.data
    train_config = args.train

    data_dir = data_config.dir
    data_name = f"config.l{data_config.L}-N1600-b2.0-k{data_config.kappa}-unquenched.npy"

    data_path = os.path.join(data_dir, data_name)

    log_root = Path("./logs/")
    logname = f"CNN_L{data_config.L}_kappa{data_config.kappa}_power{train_config.power}/"
    logdir = log_root / logname
    logdir.mkdir(parents=True, exist_ok=True)

    logging.basicConfig(
        filename=(logdir / "log.txt"), level=logging.INFO, filemode="w"
    )

    key = jax.random.PRNGKey(train_config.seed)
    key, subkey = jax.random.split(key)

    model = PrecondCNN(
        inch=model_config.inch,
        outch=model_config.outch,
        activation=instantiate(model_config.activation),
        kernel_size=model_config.kernel_size,
        n_layers=model_config.n_layers,
        hidden_dim=model_config.hidden_dim,
        key=key,
    )

    trainset = RawU1Dataset(data_path, mode="train")
    valset = RawU1Dataset(data_path, mode="val")
    logger.info(
        f"Train size {trainset.__len__()}, Val size {valset.__len__()}"
    )

    trainloader = DataLoader(trainset, batch_size=train_config.batch_size)
    valloader = DataLoader(valset, batch_size=train_config.batch_size)

    start_time = time.time()
    model = train(model, trainloader, valloader, args, logdir)
    end_time = time.time()
    logger.info(f"Training time: {end_time - start_time} seconds")
    eqx.tree_serialise_leaves(logdir / "last_model.eqx", model)


if __name__ == "__main__":
    main()
