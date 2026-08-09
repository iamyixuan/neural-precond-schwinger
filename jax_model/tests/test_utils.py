import os
import pickle
from pathlib import Path
import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from tqdm import tqdm

from src.model.FNO2d import FNO2d
from src.model.complexCNN import PrecondCNN
from src.utils.cg_solve import cg_solve, ichol0
from src.utils.metrics import (
    compute_condition_number,
    construct_Dirac_Matrix,
    get_batch_matrix,
    load_model,
    construct_matrix,
)
from src.utils.losses import HPD_opt


def get_model_config(model_name):
    """
    Returns the hyperparameters configuration for the specified model architecture.

    Args:
        model_name (str): The name of the model ('PrecondCNN' or 'FNO').

    Returns:
        dict: A dictionary containing model-specific hyperparameters such as
              channels, layers, activation functions, and random keys.

    Raises:
        ValueError: If an unknown model_name is provided.
    """
    if model_name == "PrecondCNN":
        return {
            "inch": 2,
            "outch": 2,
            "activation": eqx.nn.PReLU(),
            "kernel_size": 3,
            "n_layers": 4,
            "hidden_dim": 16,
            "key": jax.random.PRNGKey(0),
        }
    elif model_name == "FNO":
        return {
            "in_channels": 2,
            "out_channels": 2,
            "modes": 8,
            "h_channels": 16,
            "activation": eqx.nn.PReLU(),
            "n_blocks": 4,
            "key": jax.random.PRNGKey(0),
        }
    else:
        raise ValueError(f"Unknown model type: {model_name}")


def load_trained_model(args, configs):
    """
    Loads a trained model state from the file system based on the provided arguments.

    Constructs the log directory path using model parameters (ML, p, etc.) and
    uses the project's `load_model` utility to deserialized the model.

    Args:
        args (argparse.Namespace): Command line arguments containing model specifications.
        configs (dict): The configuration dictionary for the model architecture.

    Returns:
        equinox.Module: The loaded model instance.

    Raises:
        ValueError: If the model type in args is unknown.
    """
    if args.model == "FNO":
        log_dir = (
            f"./logs/FNO_L{args.ML}_kappa0.276_Precond_invloss_power{args.p}/"
        )
        model = load_model(configs, FNO2d, log_dir)
    elif args.model == "PrecondCNN":
        log_dir = (
            f"./logs/CNN_L{args.ML}_kappa0.276_Precond_invloss_power{args.p}/"
        )
        model = load_model(configs, PrecondCNN, log_dir)
    else:
        raise ValueError("Unknown model type")

    print(f"Loaded {args.model} model from {log_dir}")
    return model


def load_and_preprocess_data(args, model):
    """
    Loads the U1 gauge configuration data and computes the transformed field U_tilde.

    It loads the raw data from `args.data_dir`, applies the exponential map to get U1,
    and then computes U_tilde either via a manual mapping or by applying the trained model.

    Args:
        args (argparse.Namespace): Command line arguments containing data paths and parameters.
        model (equinox.Module): The trained model to apply (if manual mapping is disabled).

    Returns:
        tuple:
            - U1 (jax.numpy.ndarray): The gauge field configurations (complex exp form).
            - U_tilde (jax.numpy.ndarray): The transformed/preconditioned field configurations.
            - data_name (str): The filename of the loaded data.

    Raises:
        FileNotFoundError: If the data file does not exist.
    """
    data_name = (
        f"config.l{args.DL}-N200-b{args.beta}-k{args.kappa}-unquenched.npy"
    )
    data_path = Path(args.data_dir) / data_name

    if not data_path.exists():
        raise FileNotFoundError(f"Data file not found at {data_path}")

    U1 = jnp.load(data_path)
    U1_org = U1.copy()

    # Transform U1
    U1 = jnp.exp(1j * U1)

    if args.general_mapping:
        print("Using manual mapping for U_tilde")
        U_tilde = 0.5 * jnp.exp(1j * (U1_org + jnp.pi))
    else:
        U_tilde = jax.vmap(model)(U1).squeeze()

    assert len(U_tilde.shape) == 4
    return U1, U_tilde, data_name


def setup_operators(args, U1, U_tilde):
    """
    Constructs the Dirac matrix operators and JIT-compiled functions for matrix application.

    Creates two main operators:
    1. f_org: Applies the operator D^dagger D (original system).
    2. f_precond: Applies the preconditioned operator M D^dagger D M^dagger (or similar structure).

    Args:
        args (argparse.Namespace): Command line arguments containing physics parameters (kappa).
        U1 (jax.numpy.ndarray): The original gauge field.
        U_tilde (jax.numpy.ndarray): The transformed field used for the preconditioner.

    Returns:
        tuple:
            - f_org (callable): JIT-compiled function for the original operator.
            - f_precond (callable): JIT-compiled function for the preconditioned operator.
            - M (object): The Dirac matrix object for U_tilde.
            - D (object): The Dirac matrix object for U1.
    """
    X = U1.shape[-1]
    M = construct_Dirac_Matrix(U_tilde, v=X, kappa=args.kappa)
    D = construct_Dirac_Matrix(U1, v=X, kappa=args.kappa)

    @jax.jit
    def f_org(x):
        if x.shape[-3:] != (X, X, 2):
            x = x.reshape(x.shape[0], X, X, 2)
        Dx = D.apply(D.apply(x), dagger=True)
        return Dx

    @jax.jit
    def f_precond(x):
        if x.shape[-3:] != (X, X, 2):
            x = x.reshape(x.shape[0], X, X, 2)
        Dx = D.apply(D.apply(x), dagger=True)
        MDx = M.apply(M.apply(Dx), dagger=True)
        return MDx

    return f_org, f_precond, M, D


def compute_cg_stats(hist):
    """
    Computes statistical metrics (mean and std dev) for the number of CG iterations.

    Args:
        hist (list): A list of iteration counts or history objects.

    Returns:
        tuple: (mean_iters, std_iters)
    """
    num_iters = [len(h) for h in hist]
    mean_iters = np.mean(num_iters)
    std_iters = np.std(num_iters)
    return mean_iters, std_iters


def run_cg_analysis(args, model, U1, U_tilde, data_name):
    """
    Runs the Conjugate Gradient (CG) solver analysis.

    Compares the performance of:
    1. No preconditioner.
    2. Neural Network (NN) preconditioner.
    3. Incomplete Cholesky (IC) preconditioner (if requested).

    Prints statistics and saves the iteration history to pickle files.

    Args:
        args (argparse.Namespace): Command line arguments.
        model (equinox.Module): The neural network model.
        U1 (jax.numpy.ndarray): The gauge field.
        U_tilde (jax.numpy.ndarray): The preconditioner field.
        data_name (str): Name of the dataset being analyzed.
    """
    print("Running CG analysis...")
    hist = cg_solve(
        model,
        U1,
        args.kappa,
        args.p,
        use_ichol=args.use_ichol,
        data_name=data_name,
        general_mapping=args.general_mapping,
        U_tilde=U_tilde,
    )
    no_pc_hist, nn_pc_hist, IC_pc_hist = hist
    print(
        f"Sample counts: NoPC={len(no_pc_hist)}, NNPC={len(nn_pc_hist)}, ICPC={len(IC_pc_hist)}"
    )

    no_pc_mean, no_pc_std = compute_cg_stats(no_pc_hist)
    nn_pc_mean, nn_pc_std = compute_cg_stats(nn_pc_hist)
    IC_pc_mean, IC_pc_std = compute_cg_stats(IC_pc_hist)

    print(f"CG stats for {data_name}:")
    print(f"No preconditioner: mean={no_pc_mean:.2f}, std={no_pc_std:.2f}")
    print(f"NN preconditioner: mean={nn_pc_mean:.2f}, std={nn_pc_std:.2f}")
    print(f"IC preconditioner: mean={IC_pc_mean:.2f}, std={IC_pc_std:.2f}")

    # Save histories
    save_path = Path(args.save_dir)
    save_path.mkdir(parents=True, exist_ok=True)

    if args.general_mapping:
        print("Saving CG history with manual mapping...")
        with open(save_path / "USimple_precond_cg_hist.pickle", "wb") as f:
            pickle.dump(nn_pc_hist, f)
    else:
        with open(save_path / "unprecond_cg_hist.pickle", "wb") as f:
            pickle.dump(no_pc_hist, f)
        with open(save_path / "nnprecond_cg_hist.pickle", "wb") as f:
            pickle.dump(nn_pc_hist, f)
        with open(save_path / "iCholprecond_cg_hist.pickle", "wb") as f:
            pickle.dump(IC_pc_hist, f)


def save_matrices_data(args, U1, U_tilde, f_org, f_precond):
    """
    Computes and saves the dense matrix representations of the operators.

    Useful for offline analysis or debugging. Saves:
    - DD_mat: Original operator matrix.
    - M_inv_mat: Preconditioned operator matrix.
    - U1, U_tilde: The fields.

    Args:
        args (argparse.Namespace): Command line arguments.
        U1 (jax.numpy.ndarray): The gauge field.
        U_tilde (jax.numpy.ndarray): The preconditioner field.
        f_org (callable): Function to apply original operator.
        f_precond (callable): Function to apply preconditioned operator.
    """
    print("Saving matrices...")
    save_path = Path(args.save_dir)
    save_path.mkdir(parents=True, exist_ok=True)

    org_mat = get_batch_matrix(f_org, U1.shape[0], U1.shape[2] ** 2 * 2)
    precond_mat = get_batch_matrix(
        f_precond, U1.shape[0], U1.shape[2] ** 2 * 2
    )

    jnp.savez(
        save_path / f"{args.model}_DD_M.npz",
        DD_mat=org_mat,
        M_inv_mat=precond_mat,
        U1=U1,
        U_tilde=U_tilde,
    )


def compute_condition_numbers(args, U1, U_tilde, f_org, f_precond, data_name):
    """
    Computes and logs the condition numbers of the system matrices.

    Handles logic for different system sizes:
    - Small systems: Computes full condition numbers for Original, NN, and IC.
    - Medium systems (16 < N <= 32): Batched computation for IC/Original.
    - Large systems (N > 32): Saves precomputed inverse matrices (M_inv) instead of full condition number
      due to computational cost, then exits.

    Args:
        args (argparse.Namespace): Command line arguments.
        U1 (jax.numpy.ndarray): The gauge field.
        U_tilde (jax.numpy.ndarray): The preconditioner field.
        f_org (callable): Original operator function.
        f_precond (callable): Preconditioned operator function.
        data_name (str): Dataset name used to look up precomputed IC matrices.
    """
    print("Computing condition numbers...")

    # Setup paths for ichol matrices relative to data_dir
    # Assumes structure: data_dir/matrices/
    data_path_root = Path(args.data_dir) / "matrices"

    L0 = 0
    if U1.shape[-1] <= 32:
        L0_name = data_name.replace("config", "L0-matrices")
        L0_path = data_path_root / L0_name

        if L0_path.exists():
            print(f"Loading precomputed ICHOL matrices from {L0_path}...")
            L0 = np.load(L0_path, mmap_mode="r")
        else:
            print(f"Warning: L0 matrix not found at {L0_path}. Using L0=0.")

    cond_number = None
    org_cond_number = None
    IC_cond_number = None

    X = U1.shape[-1]

    # CASE 1: Medium sized systems (16 < N <= 32)
    if 16 < X <= 32:
        print("Using batch processing for large U1 matrices (16 < N <= 32)...")
        batch_size = 1
        cond_num_list = []
        org_cond_num_list = []

        # Re-construct Dirac matrix per batch to save memory/apply specifically
        for i in tqdm(range(U1.shape[0] // batch_size)):
            U1_batch = U1[i * batch_size : (i + 1) * batch_size]
            X_batch = U1_batch.shape[-1]
            D_batch = construct_Dirac_Matrix(
                U1_batch, v=X_batch, kappa=args.kappa
            )
            # print("Constructed Dirac matrix for batch", i)

            @jax.jit
            def f_org_batch(x):
                if x.shape[-3:] != (X, X, 2):
                    x = x.reshape(x.shape[0], X, X, 2)
                Dx = D_batch.apply(D_batch.apply(x), dagger=True)
                return Dx

            org_mat_batch = get_batch_matrix(
                f_org_batch, U1_batch.shape[0], U1_batch.shape[2] ** 2 * 2
            )
            org_cond = compute_condition_number(org_mat_batch)
            org_cond.block_until_ready()
            org_cond_num_list.append(org_cond)

            if isinstance(L0, (np.ndarray, jnp.ndarray)):
                L0_batch = jnp.array(L0[i * batch_size : (i + 1) * batch_size])
                U0_batch = jnp.transpose(L0_batch, (0, 2, 1)).conj()
                IC_pc_mat = jnp.linalg.inv(L0_batch @ U0_batch) @ org_mat_batch
                IC_cond = compute_condition_number(IC_pc_mat)
                IC_cond.block_until_ready()
                cond_num_list.append(IC_cond)
            else:
                cond_num_list.append(jnp.array([0.0]))  # Placeholder

        org_cond_number = jnp.concatenate(org_cond_num_list, axis=0)
        IC_cond_number = jnp.concatenate(cond_num_list, axis=0)

        q_50_IC = jnp.percentile(IC_cond_number, 50)
        q_10_IC = jnp.percentile(IC_cond_number, 10)
        q_90_IC = jnp.percentile(IC_cond_number, 90)
        print(
            f"IC condition number: {q_10_IC:.3f} | {q_50_IC:.3f} | {q_90_IC:.3f}"
        )

    # CASE 2: Large systems (N > 32)
    elif X > 32:
        print("U1 too large (>32), skipping IC condition number computation")
        IC_cond_number = jnp.array([0])

        # Original logic: Save M_inv for large systems
        batch_size = 4
        print("Large systems, saving M_inv...")

        save_path_root = data_path_root / "L64_NN_precond"
        save_path_root.mkdir(parents=True, exist_ok=True)

        for i in tqdm(
            range(U1.shape[0] // batch_size), desc="Cond number batches"
        ):
            U1_batch = U1[i * batch_size : (i + 1) * batch_size]
            X_batch = U1_batch.shape[-1]
            U_tilde_batch = U_tilde[i * batch_size : (i + 1) * batch_size]

            M_batch = construct_Dirac_Matrix(
                U_tilde_batch, v=X_batch, kappa=args.kappa
            )

            NN_M_opt = jax.jit(lambda x: HPD_opt(M_batch, x, 1))

            @jax.jit
            def M_inv_batch(x):
                if x.shape[-3:] != (X, X, 2):
                    x = x.reshape(x.shape[0], X, X, 2)
                M_invx = NN_M_opt(x)
                return M_invx

            # Compile
            _ = M_inv_batch(U1_batch)

            matrix_batch = construct_matrix(
                M_inv_batch, B=batch_size, L=X_batch
            )

            np.save(
                save_path_root
                / f"M_inv.{args.model}.ML{args.ML}.L64.kappa0.276.beta2.0.batch{i}.npy",
                matrix_batch,
            )

        print(
            "Finished generating large system matrices. Exiting as per original logic."
        )
        return  # Replaces 'assert False'

    # CASE 3: Small systems
    else:
        org_mat = get_batch_matrix(f_org, U1.shape[0], U1.shape[2] ** 2 * 2)
        org_cond_number = compute_condition_number(org_mat)

        if isinstance(L0, (np.ndarray, jnp.ndarray)):
            U0 = jnp.transpose(L0, (0, 2, 1)).conj()
            IC_pc_mat = jnp.linalg.inv(L0 @ U0) @ org_mat
            IC_cond_number = compute_condition_number(IC_pc_mat)
        else:
            IC_cond_number = jnp.array([0])

        del org_mat

    # Common Neural Precond computation (unless it was the >32 case which returned early)
    precond_mat = get_batch_matrix(
        f_precond, U1.shape[0], U1.shape[2] ** 2 * 2
    )
    jax.clear_caches()

    cond_number = compute_condition_number(precond_mat)
    del precond_mat

    # Reporting
    q_50_org = jnp.percentile(org_cond_number, 50)
    q_50_nn = jnp.percentile(cond_number, 50)

    q_10_org = jnp.percentile(org_cond_number, 10)
    q_10_nn = jnp.percentile(cond_number, 10)

    q_90_org = jnp.percentile(org_cond_number, 90)
    q_90_nn = jnp.percentile(cond_number, 90)

    print(
        f"Original condition number: {q_10_org:.3f} | {q_50_org:3f} | {q_90_org:.3f}"
    )
    print(f"Condition number: {q_10_nn:.3f} | {q_50_nn:.3f} | {q_90_nn:.3f}")

    # Save
    save_path = Path(args.save_dir)
    save_path.mkdir(parents=True, exist_ok=True)

    jnp.savez(
        save_path / "condition_numbers.npz",
        org=org_cond_number,
        precond=cond_number,
        IC=IC_cond_number,
    )
