"""The remaining report figures: architecture, loss curves, reconstructions, model comparison."""
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

from src.visualization.plots import GENRE_COLORS, GENRE_NAMES, INK, style_axes

TRAIN_COLOR, VAL_COLOR = GENRE_COLORS[0], GENRE_COLORS[1]      # blue, orange
MUTED = "#777777"


def save(fig, path):
    fig.savefig(path, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def plot_architecture(path):
    """Block diagram of the full model with the tensor shape after every stage."""
    fig, ax = plt.subplots(figsize=(13.5, 5.2))
    ax.set_xlim(0, 13.5)
    ax.set_ylim(0, 5.2)
    ax.axis("off")

    def box(x, y, w, h, text, color):
        ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.02,rounding_size=0.08",
                                    facecolor=color + "22", edgecolor=color, linewidth=1.5))
        ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=8.5, color=INK)

    def arrow(x1, y1, x2, y2):
        ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2), arrowstyle="-|>", mutation_scale=12,
                                     color=MUTED, linewidth=1.2))

    enc, dec, clf, cen = GENRE_COLORS[0], GENRE_COLORS[1], GENRE_COLORS[2], GENRE_COLORS[3]
    y = 3.3
    box(0.1, y, 1.2, 1.2, "1-s clip\n20 kHz\n20,000 samples\n↓ avg-pool 40\n500-D", MUTED)
    stages = ["Conv 1→16, k9\nBN, ReLU, Pool\n(16, 250)", "Conv 16→32, k7\nBN, ReLU, Pool\n(32, 125)",
              "Conv 32→64, k5\nBN, ReLU, Pool\n(64, 62)", "Conv 64→64, k3\nBN, ReLU, Pool\n(64, 31)",
              "Flatten 1984\nDropout 0.1\nLinear → 64"]
    x = 1.5
    arrow(1.3, y + 0.6, x, y + 0.6)
    for text in stages:
        box(x, y, 1.35, 1.2, text, enc)
        arrow(x + 1.35, y + 0.6, x + 1.5, y + 0.6)
        x += 1.5
    box(x, y + 0.1, 0.95, 1.0, "z\n64-D\nembedding", INK)
    ax.text(5.2, y + 1.45, "1D CNN encoder (153,824 parameters)", ha="center", fontsize=10, color=enc)

    zx = x + 0.45
    # Decoder (top right)
    box(10.6, 3.2, 2.85, 1.4, "Decoder (155,553 params)\nLinear 64→1984 → (64, 31)\nUpsample+Conv ×3 → (16, 250)\n"
        "Upsample → Conv 16→1\n→ 500-D reconstruction x̂", dec)
    arrow(x + 0.95, y + 0.6, 10.6, y + 0.6)
    # Classifier (bottom middle)
    box(6.2, 0.5, 2.8, 1.3, "Classifier (2,676 params)\n64 → 32 → 16 → 4 logits\nReLU, Dropout 0.1\n"
        "softmax only at inference", clf)
    arrow(zx, y + 0.1, 7.7, 1.8)
    # Center loss (bottom right)
    box(9.5, 0.5, 3.4, 1.3, "Center loss\none 64-D center per genre\nc_j moved with rate α\n(not trained by Adam)", cen)
    arrow(zx + 0.2, y + 0.1, 11.2, 1.8)

    ax.text(0.1, 2.3, "Loss:  L = γ · MSE(x, x̂)  +  (1 − γ) · CE(logits, y)  +  λ · ½‖z − c_y‖²",
            fontsize=11, color=INK, va="center")
    ax.text(0.1, 1.1, "Selected: γ = 0.9, λ = 0.1, α = 0.1\nAdam, lr 1e-4, batch 512\n"
            "early stopping on val total loss\n(min 300, patience 20, max 500 epochs)", fontsize=9, color=MUTED,
            va="center")
    save(fig, path)


def plot_loss_curves(history, best_epoch, path, title):
    """2 x 2 small multiples: total, recon, CE, center loss; train vs validation."""
    fig, axes = plt.subplots(2, 2, figsize=(11, 7), sharex=True)
    for ax, (term, name) in zip(axes.flat, [("total", "Total loss"), ("recon", "Reconstruction (MSE)"),
                                            ("ce", "Cross-entropy"), ("center", "Center loss")]):
        ax.plot(history["epoch"], history[f"train_{term}"], color=TRAIN_COLOR, linewidth=2, label="train")
        ax.plot(history["epoch"], history[f"val_{term}"], color=VAL_COLOR, linewidth=2, label="validation")
        ax.axvline(best_epoch, color=MUTED, linestyle="--", linewidth=1.2)
        ax.set_title(name, color=INK, fontsize=11)
        style_axes(ax)
    axes[0, 0].text(best_epoch, 0.98, f" best epoch {best_epoch}", transform=axes[0, 0].get_xaxis_transform(),
                    va="top", fontsize=8, color=MUTED)
    for ax in axes[1]:
        ax.set_xlabel("Epoch", color=INK, fontsize=10)
    axes[0, 0].legend(frameon=False, fontsize=9)
    fig.suptitle(title, color=INK, fontsize=12)
    fig.tight_layout()
    save(fig, path)


def plot_reconstructions(examples, path, title):
    """One panel per genre: standardized input vs reconstructions of two models.

    examples = [(genre_label, track_id, x, {model name: x_hat}), ...]
    """
    line_colors = [GENRE_COLORS[0], GENRE_COLORS[1]]
    fig, axes = plt.subplots(len(examples), 1, figsize=(11, 2.3 * len(examples)), sharex=True)
    for ax, (label, track_id, x, recons) in zip(axes, examples):
        ax.plot(x, color="#9a9a9a", linewidth=1.0, label="input (standardized, 500-D)")
        errors = []
        for color, (name, x_hat) in zip(line_colors, recons.items()):
            ax.plot(x_hat, color=color, linewidth=1.6, label=name)
            errors.append(f"{name.split(' (')[0]} MSE {((x_hat - x) ** 2).mean():.2f}")
        ax.set_title(f"{GENRE_NAMES[label]} — {track_id}   ({', '.join(errors)})",
                     color=INK, fontsize=10, loc="left")
        style_axes(ax)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=3, frameon=False, fontsize=9,
               bbox_to_anchor=(0.5, 0.965))
    axes[-1].set_xlabel("Position in the 1-second clip (500 pooled values, 2 ms each)", color=INK, fontsize=10)
    fig.suptitle(title, color=INK, fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    save(fig, path)


def plot_model_comparison(table, path, title):
    """Horizontal bars: clip and track accuracy per model (TEST). Error bars = std over seeds."""
    table = table.iloc[::-1].reset_index(drop=True)              # best at the top
    fig, ax = plt.subplots(figsize=(10, 0.55 * len(table) + 1.4))
    height = 0.38
    for offset, (column, color, name) in enumerate([("clip_accuracy", GENRE_COLORS[0], "clip accuracy (1-s clips)"),
                                                    ("track_accuracy", GENRE_COLORS[1], "track accuracy (whole songs)")]):
        positions = [i + (0.5 - offset) * height for i in range(len(table))]
        errors = table.get(f"{column}_std")
        ax.barh(positions, table[column], height=height * 0.92, color=color, label=name,
                xerr=errors, error_kw={"ecolor": INK, "elinewidth": 1, "capsize": 2})
        spread = errors.fillna(0) if errors is not None else [0] * len(table)
        for pos, value, err in zip(positions, table[column], spread):
            ax.text(value + err + 0.012, pos, f"{value:.2f}", va="center", fontsize=8, color=INK)
    ax.axvline(0.25, color=MUTED, linestyle="--", linewidth=1)
    ax.text(0.25, len(table) - 0.4, " chance 0.25", fontsize=8, color=MUTED)
    ax.set_yticks(range(len(table)), table["model"], fontsize=9, color=INK)
    ax.set_xlim(0, 1.05)
    ax.set_xlabel("Accuracy on the held-out TEST tracks", color=INK, fontsize=10)
    ax.legend(frameon=False, fontsize=9, loc="upper right")
    ax.set_title(title, color=INK, fontsize=12)
    style_axes(ax)
    ax.grid(axis="y", visible=False)
    fig.tight_layout()
    save(fig, path)
