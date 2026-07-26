"""Generate figures for training_llms/009_LLM_decoding.md.

Outputs (all .jpg, per repo convention):
  assets/decode_temperature.jpg   -- how temperature reshapes a softmax (T=0.5,1,2)
  assets/decode_topk_vs_topp.jpg  -- fixed-size (top-k) vs adaptive (top-p) truncation

Run with the repo 'general' conda env:
  /Users/mohitl/Documents/miniconda3/envs/general/bin/python scripts/make_decoding_figs.py
"""
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ASSETS = os.path.join(os.path.dirname(__file__), "..", "assets")

# Same soft, brand-neutral palette used across the repo's figures.
C = {
    "blue":  "#6C8EBF",
    "purple":"#9673A6",
    "green": "#82B366",
    "orange":"#D6795B",
    "grey":  "#555555",
    "lgrey": "#EDEDED",
    "yellow":"#E8B84B",
    "text":  "#222222",
}

DPI = 150


def save(fig, name):
    path = os.path.join(ASSETS, name)
    fig.savefig(path, dpi=DPI, bbox_inches="tight", facecolor="white",
                pil_kwargs={"quality": 90})
    plt.close(fig)
    print("wrote", os.path.normpath(path))


def softmax(z, T=1.0):
    z = np.asarray(z, dtype=float) / T
    z = z - z.max()
    e = np.exp(z)
    return e / e.sum()


# ---------------------------------------------------------------------------
# Figure 1: temperature reshaping the softmax
# ---------------------------------------------------------------------------
def fig_temperature():
    tokens = ["cat", "dog", "car", "the", "sky", "run", "blue", "of"]
    logits = [3.2, 2.6, 1.4, 1.0, 0.6, 0.2, -0.4, -1.0]
    temps = [0.5, 1.0, 2.0]
    titles = ["T = 0.5  (sharper / greedier)",
              "T = 1.0  (raw model)",
              "T = 2.0  (flatter / more random)"]
    colors = [C["blue"], C["purple"], C["orange"]]

    fig, axes = plt.subplots(1, 3, figsize=(12.5, 3.6), sharey=True)
    x = np.arange(len(tokens))
    for ax, T, title, col in zip(axes, temps, titles, colors):
        p = softmax(logits, T)
        ax.bar(x, p, color=col, edgecolor="white", width=0.72)
        ax.set_title(title, fontsize=11, color=C["text"])
        ax.set_xticks(x)
        ax.set_xticklabels(tokens, rotation=45, ha="right", fontsize=9)
        ax.set_ylim(0, 0.85)
        ax.spines[["top", "right"]].set_visible(False)
        ax.tick_params(colors=C["grey"])
        # annotate the top token's probability
        top = p.argmax()
        ax.text(top, p[top] + 0.02, f"{p[top]:.2f}", ha="center",
                fontsize=9, color=C["text"])
    axes[0].set_ylabel("probability", fontsize=10, color=C["text"])
    fig.suptitle("Temperature reshapes the SAME logits before sampling",
                 fontsize=12.5, color=C["text"], y=1.03)
    fig.tight_layout()
    save(fig, "decode_temperature.jpg")


# ---------------------------------------------------------------------------
# Figure 2: top-k (fixed size) vs top-p (adaptive) on two distributions
# ---------------------------------------------------------------------------
def fig_topk_vs_topp():
    tokens = ["a", "b", "c", "d", "e", "f", "g", "h", "i", "j"]
    x = np.arange(len(tokens))

    # A "peaked" distribution (model is confident) and a "flat" one (uncertain).
    peaked = softmax([4.0, 2.0, 1.2, 0.6, 0.2, -0.2, -0.6, -1.0, -1.4, -1.8])
    flat   = softmax([1.4, 1.2, 1.1, 1.0, 0.9, 0.8, 0.6, 0.4, 0.2, 0.0])

    k = 4          # top-k keeps a fixed 4 tokens
    p_thresh = 0.9 # top-p keeps smallest set reaching 0.90 mass

    def topp_count(p):
        order = np.argsort(p)[::-1]
        c = np.cumsum(p[order])
        return int(np.searchsorted(c, p_thresh) + 1)

    fig, axes = plt.subplots(2, 2, figsize=(12, 6.4))
    rows = [("Peaked\n(confident)", peaked),
            ("Flat\n(unsure)", flat)]

    for r, (rtitle, p) in enumerate(rows):
        order = np.argsort(p)[::-1]
        kept_k = set(order[:k].tolist())
        np_ = topp_count(p)
        kept_p = set(order[:np_].tolist())

        for col, (name, kept, tag) in enumerate([
                ("Top-k  (k = 4, always 4)", kept_k, f"keeps {k}"),
                (f"Top-p  (p = 0.9, adaptive)", kept_p, f"keeps {np_}")]):
            ax = axes[r, col]
            cols = [C["green"] if i in kept else C["lgrey"] for i in range(len(p))]
            ax.bar(x, p, color=cols, edgecolor="white", width=0.72)
            ax.set_title(f"{name}", fontsize=10.5, color=C["text"])
            ax.set_xticks(x)
            ax.set_xticklabels(tokens, fontsize=9)
            ax.set_ylim(0, max(p) * 1.25)
            ax.spines[["top", "right"]].set_visible(False)
            ax.tick_params(colors=C["grey"])
            ax.text(0.97, 0.9, tag, transform=ax.transAxes, ha="right",
                    fontsize=10, color=C["green"], fontweight="bold")
            if col == 0:
                ax.set_ylabel("probability", fontsize=10, color=C["text"])
        axes[r, 0].annotate(rtitle, xy=(-0.22, 0.5), xycoords="axes fraction",
                            rotation=90, va="center", ha="center",
                            fontsize=11, color=C["text"], fontweight="bold")

    fig.suptitle("Top-k keeps a fixed number of tokens; top-p adapts to the "
                 "shape of the distribution",
                 fontsize=12.5, color=C["text"], y=1.01)
    fig.tight_layout()
    fig.subplots_adjust(left=0.11)
    save(fig, "decode_topk_vs_topp.jpg")


if __name__ == "__main__":
    fig_temperature()
    fig_topk_vs_topp()
