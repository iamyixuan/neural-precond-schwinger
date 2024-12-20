import argparse
import jax.numpy as jnp
from plot import plot_hist, plot_sorted_scatter, plot_train, read_log, plot_cg_hist

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--model_L", type=int, default=8, help="Lattice size")
    parser.add_argument("--data_L", type=int, default=8, help="Lattice size")

    args = parser.parse_args()
    # load  saved data

    # load CG solve hist
    up_cg_hist = jnp.load(
        f"./plot_data/FNO_L{args.model_L}_D{args.data_L}_singlePrecond_unpc_hist.npz"
    )["hist"]
    nn_cg_hist_single = jnp.load(
        f"./plot_data/FNO_L{args.model_L}_D{args.data_L}_singlePrecond_nn_pc_hist.npz"
    )["hist"]
    nn_cg_hist_double = jnp.load(
        f"./plot_data/FNO_L{args.model_L}_D{args.data_L}_doublePrecond_nn_pc_hist.npz"
    )["hist"]

    fig_prefix = f"../figures/FNO_L{args.model_L}_D{args.data_L}"
    fig = plot_cg_hist(
        [up_cg_hist, nn_cg_hist_single, nn_cg_hist_double],
        ["Uncond.", "Single", "Double"],
    )
    fig.savefig(
        f"{fig_prefix}_cg_hist.pdf",
        bbox_inches="tight",
    )

    if args.data_L == 8:
        cond_num_singlePrecond = jnp.load(
            f"./plot_data/FNO_L{args.model_L}_D{args.data_L}_singlePrecond_cond_number.npz"
        )
        cond_num_doublePrecond = jnp.load(
            f"./plot_data/FNO_L{args.model_L}_D{args.data_L}_doublePrecond_cond_number.npz"
        )
        org = cond_num_singlePrecond["org"]
        pred_single = cond_num_singlePrecond["precond"]
        pred_double = cond_num_doublePrecond["precond"]
        ic_precond = cond_num_singlePrecond["IC"]
        fig, ax = plot_hist(
            [org, pred_single, pred_double, ic_precond],
            ["Uncond.", "Single", "Double", "IC"],
        )
        fig.savefig(
            f"{fig_prefix}_condNum_hist_.pdf",
            bbox_inches="tight",
        )
        fig, ax = plot_sorted_scatter(
            [org, pred_single, pred_double, ic_precond],
            ["Uncond.", "Single", "Double", "IC"],
        )
        fig.savefig(
            f"{fig_prefix}_sorted_scatter.pdf",
            bbox_inches="tight",
        )

    # train_loss, val_loss, scale = read_log(log_dir + "/log.txt")
    # fig, ax = plot_train(train_loss, val_loss)
    # fig.savefig(
    #     f"../figures/{model_name}_loss_curves.pdf",
    #     bbox_inches="tight",
    # )
