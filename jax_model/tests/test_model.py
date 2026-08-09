import argparse
from pathlib import Path
import jax
from jax import config
from test_utils import (
    get_model_config,
    load_trained_model,
    load_and_preprocess_data,
    setup_operators,
    run_cg_analysis,
    save_matrices_data,
    compute_condition_numbers,
)

config.update("jax_enable_x64", True)


def main():
    # use CPU
    jax.config.update("jax_enable_x64", True)

    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="FNO")
    parser.add_argument("--ML", type=int, default=8, help="Lattice size")
    parser.add_argument("--DL", type=int, default=8, help="Lattice size")
    parser.add_argument("--loss", type=str, default="single")
    parser.add_argument("--kappa", type=float, default=0.276)
    parser.add_argument("--beta", type=float, default=2.0)
    parser.add_argument("--log_dir", type=str, default="./logs/")
    parser.add_argument("--compute_condition_number", action="store_true")
    parser.add_argument("--save_matrices", action="store_true")
    parser.add_argument("--run_CG", action="store_true")
    parser.add_argument("--use_ichol", action="store_true")
    parser.add_argument("--p", type=int, default=1)
    parser.add_argument(
        "--general_mapping", action="store_true", help="Use mannual U_tilde"
    )
    parser.add_argument(
        "--log_root",
        type=str,
        default="/home/yixuan.sun/projects/neuralPrecond/ExperimentLogs/U1",
    )
    # Added argument for data directory
    parser.add_argument(
        "--data_dir",
        type=str,
        default="../data/U1Configs/",
        help="Directory containing U1 config files",
    )

    args = parser.parse_args()
    log_root = Path(args.log_root)

    # Construct save_dir
    if args.general_mapping:
        args.save_dir = (
            log_root
            / f"plot_data/MannualMap/DL{args.DL}/kappa_{args.kappa}/beta_{args.beta}"
        )
    else:
        args.save_dir = (
            log_root
            / f"plot_data/{args.model}/ML{args.ML}_p{args.p}/DL{args.DL}/kappa_{args.kappa}/beta_{args.beta}"
        )

    # Create save_dir if it doesn't exist
    args.save_dir.mkdir(parents=True, exist_ok=True)

    print(f"Running with p={args.p}, use_ichol={args.use_ichol}")

    configs = get_model_config(args.model)
    model = load_trained_model(args, configs)

    U1, U_tilde, data_name = load_and_preprocess_data(args, model)

    # Set up operators (needed for multiple steps)
    f_org, f_precond, M, D = setup_operators(args, U1, U_tilde)

    if args.run_CG:
        run_cg_analysis(args, model, U1, U_tilde, data_name)

    if args.save_matrices:
        save_matrices_data(args, U1, U_tilde, f_org, f_precond)

    if args.compute_condition_number:
        compute_condition_numbers(
            args, U1, U_tilde, f_org, f_precond, data_name
        )

    return str(args.log_dir)


if __name__ == "__main__":
    main()
