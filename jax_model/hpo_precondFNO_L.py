import os

import equinox as eqx
import jax
import jax.numpy as jnp
import optax
from deephyper.evaluator import RunningJob, profile
from src.model.FNO2d import FNO2d
from src.utils.data import RawU1Dataset
from src.utils.losses import inverse_loss
from torch.utils.data import DataLoader
from tqdm import tqdm

# get env variables
LENGTH = int(os.environ.get("L", 16))


def get_activation(activation):
    if activation == "relu":
        return jax.nn.relu
    elif activation == "swish":
        return jax.nn.swish
    elif activation == "sigmoid":
        return jax.nn.sigmoid
    elif activation == "softplus":
        return jax.nn.softplus
    elif activation == "prelu":
        return eqx.nn.PReLU()
    else:
        raise ValueError(f"Activation {activation} not supported")


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
    train_losses = []
    val_losses = []
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
        train_losses.append(running_loss / (i + 1))
        val_losses.append(val_loss)
        # print(f"scale: {model.scale}")

        if patience > 50 or jnp.isnan(val_loss):
            break

    return train_losses, val_losses


def train_stage(configs):
    key = jax.random.PRNGKey(0)
    key, subkey = jax.random.split(key)
    model = FNO2d(
        in_channels=2,
        out_channels=2,
        modes=configs["modes"],
        h_channels=configs["hidden_channels"],
        activation=get_activation(configs["activation"]),
        n_blocks=configs["n_blocks"],
        key=key,
    )

    train_config = {
        "num_epochs": 10000,
        "batch_size": configs["batch_size"],
        "optim": optax.adam(configs["lr"]),
    }

    data_dir = "../data/U1Configs/"
    if LENGTH == 8:
        data_name = "config.l8-N1600-b2.0-k0.276-unquenched.x.npy"
    elif LENGTH == 16:
        data_name = "config.l16-N200-b2.0-k0.276-unquenched-test.x.npy"
    elif LENGTH == 32:
        data_name = "config.l32-N200-b2.0-k0.276-unquenched-test.x.npy"
    elif LENGTH == 64:
        data_name = "config.l64-N200-b2.0-k0.276-unquenched-test.x.npy"

    data_path = os.path.join(data_dir, data_name)
    trainset = RawU1Dataset(data_path, mode="train")
    valset = RawU1Dataset(data_path, mode="val")

    trainloader = DataLoader(trainset, batch_size=train_config["batch_size"])
    valloader = DataLoader(valset, batch_size=valset.__len__())

    train_losses, val_losses = train(
        model,
        trainloader,
        valloader,
        train_config["optim"],
        train_config,
    )
    return train_losses, val_losses


@profile
def run(job: RunningJob):
    configs = job.parameters.copy()
    try:
        train_loss, val_loss = train_stage(configs)

        metadata = {
            "train_losses": train_loss,
            "val_losses": val_loss,
            # "scales": scales,
        }
        final_objective = -float(val_loss[-1])  # maximize neg condition number

    except Exception as e:
        print(e)
        metadata = {}
        val_loss = [1e6]
        final_objective = 'F'

    print(f"Objective: {final_objective}")
    # except Exception as e:
    #     print(e)
    #     final_objective = "F"
    #     metadata = {}

    return {"objective": final_objective, "metadata": metadata}


if __name__ == "__main__":
    from deephyper.evaluator import Evaluator
    from deephyper.problem import HpProblem
    from deephyper.search.hps import CBO

    problem = HpProblem()
    problem.add_hyperparameter(
        (8, 128),
        "hidden_channels",
        default_value=64,
    )  # hidden dimension
    problem.add_hyperparameter(
        (1, 20),
        "n_blocks",
        default_value=4,
    )  # number of layers
    problem.add_hyperparameter(
        (1, 8),
        "modes",
        default_value=2,
    )  # number of
    problem.add_hyperparameter(
        (1e-5, 1e-1, "log-uniform"),
        "lr",
        default_value=1e-3,
    )  # learning rate
    problem.add_hyperparameter(
        ["relu", "swish", "sigmoid", "softplus", "prelu"],
        "activation",
        default_value="relu",
    )  # activation function
    problem.add_hyperparameter(
        (1, 256),
        "batch_size",
        default_value=32,
    )  # batch size
    problem.add_hyperparameter(
        (0.0, 0.5),
        "dropout_rate",
        default_value=0.0,
    )  # dropout rate

    evaluator = Evaluator.create(run)
    log_dir = f"hpo_FNO_L{LENGTH}/"
    os.makedirs(log_dir, exist_ok=True)
    search = CBO(
        problem,
        evaluator,
        log_dir=log_dir,
        initial_points=[problem.default_configuration],
        acq_optimizer="mixedga",
        acq_optimizer_freq=1,
        kappa=5.0,
        scheduler={
            "type": "periodic-exp-decay",
            "period": 50,
            "kappa_final": 0.0001,
        },
        objective_scaler="identity",
    )
    results = search.search(max_evals=500)
