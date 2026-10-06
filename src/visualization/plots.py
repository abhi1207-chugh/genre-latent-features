"""Plots for the report. Every figure is saved as PNG in results/figures/."""
import matplotlib

matplotlib.use("Agg")                    # save to file, no window
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from sklearn.decomposition import PCA

# Fixed colour + marker per genre (index = label). Colour-blind-safe palette;
# the marker shape is a second cue so genres are not told apart by colour alone.
GENRE_NAMES = ["Classical", "Country", "Disco", "Hip-Hop"]
GENRE_COLORS = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100"]
GENRE_MARKERS = ["o", "s", "^", "D"]
INK = "#333333"
GRID = "#e5e5e5"


def style_axes(ax):
    ax.grid(color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    for side in ["top", "right"]:
        ax.spines[side].set_visible(False)
    for side in ["left", "bottom"]:
        ax.spines[side].set_color("#999999")
    ax.tick_params(colors=INK, labelsize=9)


def plot_confusion_panels(panels, path, title):
    """Side-by-side confusion matrices. panels = [(name, matrix of counts), ...].

    Cell colour = share of the TRUE genre's items (row-normalised, one blue ramp);
    each cell shows the count and that percentage.
    """
    from matplotlib.colors import LinearSegmentedColormap
    blues = LinearSegmentedColormap.from_list("blues", ["#f4f8fd", "#86b6ef", "#2a78d6", "#0d366b"])

    fig, axes = plt.subplots(1, len(panels), figsize=(4.6 * len(panels), 4.4))
    for ax, (name, counts) in zip(np.atleast_1d(axes), panels):
        shares = counts / counts.sum(axis=1, keepdims=True)
        ax.imshow(shares, cmap=blues, vmin=0, vmax=1)
        for i in range(len(GENRE_NAMES)):
            for j in range(len(GENRE_NAMES)):
                ink = "white" if shares[i, j] > 0.55 else INK
                ax.text(j, i, f"{counts[i, j]}\n{shares[i, j]:.0%}", ha="center", va="center",
                        fontsize=9, color=ink)
        ax.set_xticks(range(4), GENRE_NAMES, fontsize=9, color=INK)
        ax.set_yticks(range(4), GENRE_NAMES, fontsize=9, color=INK)
        ax.set_xlabel("Predicted genre", color=INK, fontsize=10)
        ax.set_ylabel("True genre", color=INK, fontsize=10)
        ax.set_title(name, color=INK, fontsize=10)
        for side in ax.spines.values():
            side.set_visible(False)
    fig.suptitle(title, color=INK, fontsize=12)
    fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def plot_pareto(table, ae_reference, trivial_error, path):
    """Validation recon error (x) vs validation 5-NN accuracy (y) for the grid runs.

    Colour = γ, marker = λ, each point labelled with its config number.
    Lines: 1.5 x AE_reference (selection threshold), trivial 'output zeros' error,
    and AE_reference itself. The selected run gets a black ring; the Pareto
    front (runs not beaten on both axes) is joined by a grey line.
    """
    gammas = sorted(table["gamma"].unique())
    lambdas = sorted(table["lambda"].unique())
    colors = dict(zip(gammas, GENRE_COLORS[:len(gammas)]))
    markers = dict(zip(lambdas, ["o", "s", "^"]))

    fig, ax = plt.subplots(figsize=(8, 5.4))
    front = table[table["pareto"]].sort_values("val_recon")
    ax.plot(front["val_recon"], front["val_knn5_accuracy"], color="#bbbbbb",
            linewidth=2, zorder=1)

    for _, run in table.iterrows():
        ax.scatter(run["val_recon"], run["val_knn5_accuracy"], s=90,
                   color=colors[run["gamma"]], marker=markers[run["lambda"]],
                   edgecolors="white", linewidths=1.5, zorder=3)
        # Label above-right; below-right if another point sits just above (avoids overlap)
        crowded = ((abs(table["val_recon"] - run["val_recon"]) < 0.01) &
                   (table["val_knn5_accuracy"] > run["val_knn5_accuracy"]) &
                   (table["val_knn5_accuracy"] - run["val_knn5_accuracy"] < 0.006)).any()
        ax.annotate(f"#{int(run['config'])}", (run["val_recon"], run["val_knn5_accuracy"]),
                    textcoords="offset points", xytext=(7, -13 if crowded else 5),
                    fontsize=9, color=INK)
    chosen = table[table["selected"]].iloc[0]
    ax.scatter(chosen["val_recon"], chosen["val_knn5_accuracy"], s=320, facecolors="none",
               edgecolors=INK, linewidths=1.8, zorder=4)

    for x, style, text in [(1.5 * ae_reference, "-", f"1.5 × AE_ref = {1.5 * ae_reference:.3f}\n(selection threshold)"),
                           (trivial_error, "--", f"output zeros = {trivial_error:.3f}"),
                           (ae_reference, ":", f"AE_ref = {ae_reference:.3f}")]:
        ax.axvline(x, color="#777777", linestyle=style, linewidth=1.2, zorder=2)
        ax.text(x, 0.01, text, transform=ax.get_xaxis_transform(), rotation=90,
                va="bottom", ha="right", fontsize=8, color="#555555")

    handles = [Line2D([], [], linestyle="", marker="o", markersize=9, color=colors[g],
                      label=f"γ = {g}") for g in gammas]
    handles += [Line2D([], [], linestyle="", marker=markers[l], markersize=8, color="#777777",
                       label=f"λ = {l}") for l in lambdas]
    handles += [Line2D([], [], linestyle="", marker="o", markersize=13, markerfacecolor="none",
                       markeredgecolor=INK, label=f"selected (#{int(chosen['config'])})"),
                Line2D([], [], color="#bbbbbb", linewidth=2, label="Pareto front")]
    ax.legend(handles=handles, loc="center left", bbox_to_anchor=(1.01, 0.5), frameon=False, fontsize=9)

    ax.set_xlabel("Validation reconstruction error (MSE per value) — lower is better", color=INK, fontsize=10)
    ax.set_ylabel("Validation 5-NN accuracy on 64-D embeddings", color=INK, fontsize=10)
    ax.set_title("Grid runs (α = 0.1): reconstruction vs genre information", color=INK, fontsize=12)
    ax.set_xlim(ae_reference - 0.05, 1.5 * ae_reference + 0.02)
    style_axes(ax)
    fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def plot_pca_panels(panels, y, path, title):
    """Side-by-side 2-D PCA scatter plots, one panel per (name, features) pair.

    PCA is fit on the plotted points themselves (validation clips) - it is only
    a 2-D view; no model uses it.
    """
    fig, axes = plt.subplots(1, len(panels), figsize=(5.2 * len(panels), 4.6))
    # Draw points in random order (in 20 chunks), so no genre is always painted on top
    order = np.random.default_rng(42).permutation(len(y))
    for ax, (name, Z) in zip(axes, panels):
        pca = PCA(n_components=2, random_state=42).fit(Z)
        points = pca.transform(Z)
        for chunk in np.array_split(order, 20):
            for label in range(len(GENRE_NAMES)):
                idx = chunk[y[chunk] == label]
                ax.scatter(points[idx, 0], points[idx, 1], s=12, alpha=0.55,
                           color=GENRE_COLORS[label], marker=GENRE_MARKERS[label],
                           edgecolors="white", linewidths=0.3)

        # Zoom to the central 98% of points; a few loud outliers would squash the rest
        low, high = np.percentile(points, [1, 99], axis=0)
        margin = 0.1 * (high - low)
        ax.set_xlim(low[0] - margin[0], high[0] + margin[0])
        ax.set_ylim(low[1] - margin[1], high[1] + margin[1])
        inside = np.all((points >= low - margin) & (points <= high + margin), axis=1)

        explained = pca.explained_variance_ratio_
        ax.set_title(f"{name}\n({(~inside).sum()} outlier points outside view)",
                     color=INK, fontsize=11)
        ax.set_xlabel(f"PC1 ({explained[0]:.0%} var.)", color=INK, fontsize=9)
        ax.set_ylabel(f"PC2 ({explained[1]:.0%} var.)", color=INK, fontsize=9)
        style_axes(ax)

    handles = [Line2D([], [], linestyle="", marker=GENRE_MARKERS[i], color=GENRE_COLORS[i],
                      markersize=7, label=GENRE_NAMES[i]) for i in range(len(GENRE_NAMES))]
    fig.legend(handles=handles, loc="upper center", ncol=4, frameon=False,
               markerscale=1.8, fontsize=10, bbox_to_anchor=(0.5, 0.96))
    fig.suptitle(title, color=INK, fontsize=12, y=1.02)
    fig.tight_layout(rect=(0, 0, 1, 0.92))
    fig.savefig(path, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)
