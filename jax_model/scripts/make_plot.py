import os
import argparse
import jax.numpy as jnp
from plot import plot_hist, plot_sorted_scatter, plot_train, read_log, plot_cg_hist, plot_cg_iter

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--model_L", type=int, default=8, help="Lattice size")
    parser.add_argument("--data_L", type=int, default=8, help="Lattice size")
    parser.add_argument("--kappa", type=float, default=0.276)
    parser.add_argument("--save_dir", type=str, default="kU10.276_kSetup0.260")

    args = parser.parse_args()
    # load  saved data
    plot_dir = os.path.join("./plot_data/", args.save_dir)
    # load CG solve hist
    up_cg_hist = jnp.load(
        f"{plot_dir}/FNO_L{args.model_L}_D{args.data_L}_kappa{args.kappa}_singlePrecond_unpc_hist.npz"
    )["hist"]
    nn_cg_hist_single = jnp.load(
        f"{plot_dir}/FNO_L{args.model_L}_D{args.data_L}_kappa{args.kappa}_singlePrecond_nn_pc_hist.npz"
    )["hist"]
    ic_cg_hist = jnp.load(
        f"{plot_dir}/FNO_L{args.model_L}_D{args.data_L}_kappa{args.kappa}_singlePrecond_ic_pc_hist.npz"
    )["hist"]

    if args.data_L == 8:
        cg_list = [up_cg_hist, nn_cg_hist_single, ic_cg_hist]
        name_list = ["Uncond.", "NN", "IC"]
    else:
        cg_list = [up_cg_hist, nn_cg_hist_single]
        name_list = ["Uncond.", "NN"]

    fig_dir = f"../figures/{args.save_dir}/FNO_L{args.model_L}/D{args.data_L}/kappa{args.kappa}/"
    os.makedirs(fig_dir, exist_ok=True)
    fig = plot_cg_hist(
        cg_list,
        name_list,
    )
    fig.savefig(
        os.path.join(fig_dir, "cg_hist.pdf"),
        bbox_inches="tight",
    )



    if args.data_L == 8:
        cond_num_singlePrecond = jnp.load(
            f"{plot_dir}/FNO_L{args.model_L}_D{args.data_L}_kappa{args.kappa}_singlePrecond_cond_number.npz"
        )
        # cond_num_doublePrecond = jnp.load(
        #     f"./plot_data/FNO_L{args.model_L}_D{args.data_L}_doublePrecond_cond_number.npz"
        # )
        org = cond_num_singlePrecond["org"]
        pred_single = cond_num_singlePrecond["precond"]
        # pred_double = cond_num_doublePrecond["precond"]
        ic_precond = cond_num_singlePrecond["IC"]
        fig, ax = plot_hist(
            [org, pred_single, ic_precond],
            ["Uncond.", "Single", "IC"],
        )
        fig.savefig(
            os.path.join(fig_dir, "condNum_hist.pdf"),
            bbox_inches="tight",
        )
        fig, ax = plot_sorted_scatter(
            [org, pred_single, ic_precond],
            ["Uncond.", "Single", "IC"],
        )
        fig.savefig(
            os.path.join(fig_dir, "sorted_scatter.pdf"),
            bbox_inches="tight",
        )

        fig = plot_cg_iter(
        [up_cg_hist, nn_cg_hist_single],
        jnp.argsort(org),
        )  
        fig.savefig(
            os.path.join(fig_dir, "cg_iter.pdf"),
            bbox_inches="tight",
        )


    # train_loss, val_loss, scale = read_log(log_dir + "/log.txt")
    # fig, ax = plot_train(train_loss, val_loss)
    # fig.savefig(
    #     f"../figures/{model_name}_loss_curves.pdf",
    #     bbox_inches="tight",
    # )
