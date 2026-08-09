import re

import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np


def set_size(width, fraction=1, subplots=(1, 1), golden_ratio=True):
    """Set figure dimensions to avoid scaling in LaTeX.
    from https://jwalton.info/Embed-Publication-Matplotlib-Latex/

    Parameters
    ----------
    width: float or string
            Document width in points, or string of predined document type
    fraction: float, optional
            Fraction of the width which you wish the figure to occupy
    subplots: array-like, optional
            The number of rows and columns of subplots.
    Returns
    -------
    fig_dim: tuple
            Dimensions of figure in inches
    """
    if width == "thesis":
        width_pt = 426.79135
    elif width == "beamer":
        width_pt = 307.28987
    else:
        width_pt = width

    tex_fonts = {
        # Use LaTeX to write all text
        # "text.usetex": True,
        "font.family": "serif",
        # Use 10pt font in plots, to match 10pt font in document
        "axes.labelsize": 10,
        "font.size": 10,
        # Make the legend/label fonts a little smaller
        "legend.fontsize": 8,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
    }

    plt.rcParams.update(tex_fonts)

    # Width of figure (in pts)
    fig_width_pt = width_pt * fraction
    # Convert from pt to inches
    inches_per_pt = 1 / 72.27

    # Golden ratio to set aesthetic figure height
    # https://disq.us/p/2940ij3
    golden_ratio = (5**0.5 - 1) / 2

    # Figure width in inches
    fig_width_in = fig_width_pt * inches_per_pt
    # Figure height in inches
    if golden_ratio:
        fig_height_in = fig_width_in * golden_ratio * (subplots[0] / subplots[1])
    else:
        fig_height_in = fig_width_in

    return (fig_width_in, fig_height_in)


def set_figuresize(fig, width, height, golden_ratio=False):
    if golden_ratio:
        width = height * (1 + np.sqrt(5)) / 2
    fig.set_figwidth(width)
    fig.set_figheight(height)

    # set font size based on figure size
    if width >= 10:
        font_size = 16
    elif width >= 8:
        font_size = 14
    else:
        font_size = 12
    plt.rcParams.update({"font.size": font_size})
    plt.rcParams.update({"axes.labelsize": font_size})

    # set line width based on figure size
    if width >= 10:
        line_width = 3
    elif width >= 8:
        line_width = 2.5
    else:
        line_width = 1.5
    plt.rcParams.update({"lines.linewidth": line_width})
    return fig


def read_log(file_path):
    print(file_path)
    train_loss = []
    val_loss = []
    scale = []
    train_pattern = r"train Loss: (\d+\.\d+)"
    val_pattern = r"val Loss: (\d+\.\d+)"
    scale_pattern = r"scale: \[?([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)\]?"
    with open(file_path, "r") as f:
        for line in f:
            train_match = re.findall(train_pattern, line)
            val_match = re.findall(val_pattern, line)
            # scale_match = re.findall(scale_pattern, line)
            if len(train_match) == 0 or len(val_match) == 0:
                continue
            train_loss.append(float(train_match[0]))
            val_loss.append(float(val_match[0]))
            # scale.append(float(scale_match[0]))

    return train_loss, val_loss, scale


def plot_train(train_loss, val_loss):
    x = jnp.arange(1, len(train_loss) + 1)
    fig, ax = plt.subplots(figsize=set_size("thesis"))
    ax.plot(x, train_loss, label="train loss")
    ax.plot(x, val_loss, label="val loss")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.grid(linestyle="dotted")
    ax.legend()
    return fig, ax


def plot_multi_curves(log_list, label_list):
    fig, ax = plt.subplots(figsize=set_size("thesis"))
    ax2 = ax.twinx()
    for i, log in enumerate(log_list):
        train_loss, val_loss, scale = read_log(log)
        x = jnp.arange(1, len(train_loss) + 1)
        ax.plot(x, train_loss, label=label_list[i])
        ax2.plot(x, scale, label=label_list[i] + " scale", linestyle="--")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax2.set_ylabel(r"$\alpha$")
    # ax2.set_yscale("log")
    ax.grid(linestyle="dotted")
    ax.legend()
    return fig, ax


def plot_hist(hist_list, label_list):
    fig, ax = plt.subplots(figsize=set_size("thesis"))
    q90 = []
    for i, hist in enumerate(hist_list):
        q90.append(np.percentile(hist, 90))
        ax.hist(hist, bins=500, alpha=0.3, label=label_list[i])
    ax.set_xlabel("Condition number")
    ax.set_xlim(0, np.max(q90))
    ax.set_ylabel("Frequency")
    ax.legend()
    return fig, ax


def plot_sorted_scatter(point_lists, labels):
    fig, ax = plt.subplots(figsize=set_size(390))
    base_order = np.argsort(point_lists[0])
    q90 = []
    for i, points in enumerate(point_lists):
        q90.append(np.percentile(points, 90))
        ax.scatter(np.arange(len(points)), points[base_order], label=labels[i])
    ax.set_xlabel("Index")
    ax.set_ylabel("Condition number")
    # ax.set_ylim(0, np.max(q90))
    ax.set_yscale("log")
    ax.grid(linestyle="dotted")
    ax.legend()
    return fig, ax


def plot_condition_number_comparison(cond_list, label_list):
    """
    cond_list: list of condiiton number pairs (original, NPC) for various lattice sizes
    label_list: list of labels for the lattice sizes
    """
    fig, ax = plt.subplots(figsize=set_size(390))
    colors = [
        "#E69F00",  # orange
        "#56B4E9",  # light blue
        "#009E73",  # greenish
        "#0072B2",  # blue
        "#D55E00",  # reddish
        "#CC79A7",  # pinkish
        "#F0E442",  # yellow
    ]
    for i, cond in enumerate(cond_list):
        org_cond = cond[0]
        npc_cond = cond[1]
        base_order = np.argsort(org_cond)
        ax.scatter(
            np.arange(len(org_cond)),
            org_cond[base_order],
            c=colors[i],
            edgecolors="k",
            linewidths=0.3,
            alpha=.1,
        )
        ax.scatter(
            np.arange(len(npc_cond)),
            npc_cond[base_order],
            c=colors[i],
            marker="*",
            edgecolors="k",
            linewidths=0.1,
        )
        ax.scatter([], [], label=label_list[i], marker="s", c=colors[i])
    ax.scatter([], [], label="Unprecond.", c="k")
    ax.scatter([], [], label="NN-precond.", c="k", marker="*")
    ax.legend()
    ax.set_xlabel("Index")
    ax.set_ylabel("Condition number")
    ax.set_yscale("log")
    ax.grid(which="both", color="gray", linestyle="dotted", alpha=0.5)
    return fig

def plot_cg_comparison(cg_hist_list, label_list):
    fig, ax = plt.subplots(figsize=set_size(390))
    colors = [
        "#E69F00",  # orange
        "#56B4E9",  # light blue
        "#009E73",  # greenish
        "#0072B2",  # blue
        "#D55E00",  # reddish
        "#CC79A7",  # pinkish
        "#F0E442",  # yellow
    ]
    for i, hist in enumerate(cg_hist_list):
        no_pc_hist, nn_pc_hist = hist
        no_pc_hist = np.array(no_pc_hist)
        nnpc_hist = np.array(nn_pc_hist)
        ax.plot(no_pc_hist.mean(axis=1), linestyle="-.", color=colors[i])
        ax.plot(nnpc_hist.mean(axis=1), color=colors[i])
        ax.scatter([],[], label=label_list[i], color=colors[i], marker="s")

    ax.set_xlabel("Iteration")
    ax.grid(linestyle="dotted", which="both")
    ax.set_ylabel("Relative Residual")
    ax.set_yscale("log")
    ax.plot([],[], label="Unprecond", color="k", linestyle="-.")
    ax.plot([],[], label="NN-precond", color="k")
    ax.legend()
    return fig

def plot_cg_hist(hist_list, label_list):
    fig, ax = plt.subplots(figsize=set_size(390))
    ax.set_box_aspect(1 / 1.62)
    colors = [
        "#E69F00",  # orange
        "#56B4E9",  # light blue
        "#009E73",  # greenish
        "#0072B2",  # blue
        "#D55E00",  # reddish
        "#CC79A7",  # pinkish
        "#F0E442",  # yellow
    ]

    for i in range(len(hist_list)):
        ax.plot(np.mean(hist_list[i], axis=1), label=label_list[i], color=colors[i])
        # plot 1.96 * std
        # ax.fill_between(
        #     np.arange(hist_list[i].shape[0]),
        #     np.mean(hist_list[i], axis=1) - 1.96 * np.std(hist_list[i], axis=1),
        #     np.mean(hist_list[i], axis=1) + 1.96 * np.std(hist_list[i], axis=1),
        #     alpha=0.15,
        #     color=colors[i],
        # )
    ax.set_yscale("log")
    ax.grid(which="both", color="gray", linestyle="dotted", alpha=0.5)
    ax.set_xlabel("Iteration")
    ax.set_ylabel("Relative Residual")
    ax.legend()
    return fig

def plot_cg_time(time_list, label_list):
    fig, ax = plt.subplots(figsize=set_size(390))
    ax.set_box_aspect(1 / 1.62)
    colors = [
        "#E69F00",  # orange
        "#56B4E9",  # light blue
        "#009E73",  # greenish
        "#0072B2",  # blue
        "#D55E00",  # reddish
        "#CC79A7",  # pinkish
        "#F0E442",  # yellow
    ]
    # barplot
    ax.bar(np.arange(len(time_list)), time_list, color=colors)
    ax.set_xticks(np.arange(len(time_list)))
    ax.set_xticklabels(label_list)
    ax.set_ylabel("Time (s)")
    ax.grid(which="both", color="gray", linestyle="dotted", alpha=0.5)
    return fig




def plot_cg_iter(hist_list, sorted_idx, cutoff_tol=1e-8 * 128 * 200):
    fig, ax = plt.subplots(figsize=set_size(390))
    ax.set_box_aspect(1 / 1.62)
    # sorted_idx = np.argsort(np.argmax(hist_list[0] <= cutoff_tol, axis=0))
    for hist in hist_list:
        num_iter = np.argmax(hist < cutoff_tol, axis=0) + 1
        ax.scatter(np.arange(len(num_iter)), num_iter[sorted_idx])

    ax.set_xlabel("Index")
    ax.set_ylabel("Number of iterations")
    ax.grid(which="both", color="gray", linestyle="dotted", alpha=0.5)
    return fig


if __name__ == "__main__":
    from pathlib import Path
    import pickle

    """
    plot_data_dir = Path(
        "./plot_data/"
    )
    parent_files = [f"FNO_L{lattice}_D{lattice}_kappa0.276" for lattice in [8, 16, 32]]
    fnames = [
        f"FNO_L{lattice}_D{lattice}_kappa0.276_singlePrecond_cond_number.npz"
        for lattice in [8, 16, 32]
    ]

    cond_list = []
    for i, f in enumerate(parent_files):
        data = np.load(plot_data_dir / f / fnames[i])
        cond_list.append((data["org"], data["precond"]))

    fig = plot_condition_number_comparison(cond_list, ["L=8", "L=16", "L=32"])
    fig.savefig("condition_number_comparison.pdf", bbox_inches="tight")


    unprecond_file_list = ["FNO_L8_D8_kappa0.276_singlePrecond_unpc_hist.npz",
                           "FNO_L16_D16_kappa0.276_singlePrecond_unpc_hist.npz",
                           "FNO_L32_D32_kappa0.276_singlePrecond_unpc_hist.npz"]
    nnprecond_file_list = ["FNO_L8_D8_kappa0.276_singlePrecond_nn_pc_hist.npz",
                           "FNO_L16_D16_kappa0.276_singlePrecond_nn_pc_hist.npz",
                           "FNO_L32_D32_kappa0.276_singlePrecond_nn_pc_hist.npz"]

    hist_list = []
    for i, f in enumerate(parent_files):
        unprecond_data = np.load(plot_data_dir / f / unprecond_file_list[i])
        nnprecond_data = np.load(plot_data_dir / f / nnprecond_file_list[i])
        hist_list.append((unprecond_data["hist"], nnprecond_data["hist"]))
    fig = plot_cg_comparison(hist_list, ["L=8", "L=16", "L=32"])
    fig.savefig("cg_comparison.pdf", bbox_inches="tight")

    # plot cg hist for L=8 including Ichol 
    ic_hist = "FNO_L8_D8_kappa0.276_singlePrecond_ic_pc_hist.npz"
    unprecond_cg_L8 = np.load(plot_data_dir / parent_files[0] / unprecond_file_list[0])
    unprecond_cg_L8_hist, unprecond_cg_L8_time = unprecond_cg_L8["hist"], unprecond_cg_L8["time"]
    nnprecond_cg_L8 = np.load(plot_data_dir / parent_files[0] / nnprecond_file_list[0])
    nnprecond_cg_L8_hist, nnprecond_cg_L8_time = nnprecond_cg_L8["hist"], nnprecond_cg_L8["time"]
    ic_cg_L8 = np.load(plot_data_dir / parent_files[0] / ic_hist)
    ic_cg_L8_hist, ic_cg_L8_time = ic_cg_L8["hist"], ic_cg_L8["time"]
    fig = plot_cg_hist([unprecond_cg_L8_hist, nnprecond_cg_L8_hist, ic_cg_L8_hist], ["Unprecond", "NN-precond", "IC-precond"])
    fig.savefig("cg_hist_L8.pdf", bbox_inches="tight")

    fig= plot_cg_time([unprecond_cg_L8_time, nnprecond_cg_L8_time, ic_cg_L8_time], ["Unprecond", "NN-precond", "IC-precond"])
    fig.savefig("cg_time_L8.pdf", bbox_inches="tight")
    """

    ML=16
    DL=32
    with open(f"./plot_data/cg_hist_ML16_DL{DL}.pickle", "rb") as f:
        history = pickle.load(f)
    print(history.keys())   
    # plot transfer leanring hist
    colors = [
        "#E69F00",  # orange
        "#56B4E9",  # light blue
        "#009E73",  # greenish
        "#0072B2",  # blue
        "#D55E00",  # reddish
        "#CC79A7",  # pinkish
        "#F0E442",  # yellow
    ]

    # with open(f"plot_data/zero_shot_M{DL}_D{DL}_hist.pickle", "rb") as f:
    #     base_hist = pickle.load(f)
    
    # base_hist = np.array(base_hist['nnprecond'])

    # with open(f"plot_data/zero_shot_M{ML}_D{DL}_hist.pickle", "rb") as f:
    #     zero_hist = pickle.load(f)
    # unpred_hist = np.array(zero_hist['unprecond'])
    # nnprecond_hist = np.array(zero_hist['nnprecond'])

    # fig, ax = plt.subplots(figsize=set_size(390))
    # ax.plot(unpred_hist.mean(axis=1), label="Unprecond", color=colors[0])
    # for i, (key, val) in enumerate(history.items()):
    #     ax.plot(val.mean(axis=1), label=f"Ratio={key.split('_')[-1]}", color=colors[i+1])
    #     break
    # ax.plot(base_hist.mean(axis=1), label="From scratch", linestyle="--", color="black", linewidth=1)
    # ax.set_xlabel("Iteration")
    # ax.grid(linestyle="dotted", which="both")
    # ax.set_ylabel("Relative Residual")
    # # ax.set_title(f"CG convergence (ML{16}DL{DL})")
    # ax.set_yscale("log")
    # ax.legend()
    # fig.savefig(f"../figures/cg_hist_ML16_DL{DL}_seed123.pdf", format="pdf", bbox_inches="tight")


    with open(f"plot_data/zero_shot_M{ML}_D{DL}_hist.pickle", "rb") as f:
        zero_hist1 = pickle.load(f)
    unpred_hist = np.array(zero_hist1['unprecond'])
    nnprecond_hist1 = np.array(zero_hist1['nnprecond'])

    with open(f"plot_data/zero_shot_M{ML}_D{DL}_hist_seed123.pickle", "rb") as f:
        zero_hist2 = pickle.load(f)
    unpred_hist = np.array(zero_hist2['unprecond'])
    nnprecond_hist2 = np.array(zero_hist2['nnprecond'])
    fig = plot_cg_hist([unpred_hist, nnprecond_hist1, nnprecond_hist2], ["Unprecond", "NN-precond seed0", "NN-precond seed123"])
    fig.savefig("zero_shot_hist_M16_D32_seed0-123.pdf", bbox_inches="tight")

