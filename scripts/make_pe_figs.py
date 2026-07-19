"""Generate figures for training_llms/002_PositionEmbeddings.md.

Outputs (all .jpg, per repo convention):
  assets/pe_absolute_vs_relative.jpg -- schematic contrasting absolute vs relative PE
  assets/pe_sinusoidal_heatmap.jpg   -- position x dimension heatmap of sinusoidal PE
  assets/rope_rotation.jpg           -- a 2D pair rotated by angle m*theta at m=0,1,2,3
  assets/rope_multifreq.jpg          -- several dim pairs rotating at different speeds vs position

Run with the repo 'general' conda env:
  /Users/mohitl/Documents/miniconda3/envs/general/bin/python scripts/make_pe_figs.py
"""
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Arc

ASSETS = os.path.join(os.path.dirname(__file__), "..", "assets")

# A soft, consistent, brand-neutral palette (matches make_tokenizer_figs.py).
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


# ---------------------------------------------------------------------------
# 1. Absolute vs relative PE schematic
# ---------------------------------------------------------------------------
def fig_absolute_vs_relative():
    fig, (axA, axB) = plt.subplots(1, 2, figsize=(12, 4.2))
    words = ["The", "cat", "sat", "on", "the", "mat"]
    n = len(words)
    xs = np.arange(n)

    # --- Absolute ---
    axA.set_title("Absolute PE:  \"you are at slot $m$\"", fontsize=13,
                  fontweight="bold", color=C["text"])
    for i, w in enumerate(words):
        box = FancyBboxPatch((i - 0.42, -0.32), 0.84, 0.64,
                             boxstyle="round,pad=0.02,rounding_size=0.12",
                             fc=C["lgrey"], ec=C["grey"], lw=1.2)
        axA.add_patch(box)
        axA.text(i, 0, w, ha="center", va="center", fontsize=12, color=C["text"])
        axA.text(i, 0.75, f"pos {i}", ha="center", va="center", fontsize=12,
                 fontweight="bold", color=C["blue"])
        axA.annotate("", xy=(i, 0.33), xytext=(i, 0.62),
                     arrowprops=dict(arrowstyle="-|>", color=C["blue"], lw=1.6))
    axA.set_xlim(-0.8, n - 0.2)
    axA.set_ylim(-0.9, 1.1)
    axA.axis("off")
    axA.text(0.5, -0.85, "Each token gets a fixed code for its own index.",
             transform=axA.transAxes, ha="center", fontsize=10.5,
             style="italic", color=C["grey"])

    # --- Relative ---
    axB.set_title("Relative PE:  \"how far is key from query\"", fontsize=13,
                  fontweight="bold", color=C["text"])
    for i, w in enumerate(words):
        fc = C["yellow"] if i == 4 else C["lgrey"]
        box = FancyBboxPatch((i - 0.42, -0.32), 0.84, 0.64,
                             boxstyle="round,pad=0.02,rounding_size=0.12",
                             fc=fc, ec=C["grey"], lw=1.2)
        axB.add_patch(box)
        axB.text(i, 0, w, ha="center", va="center", fontsize=12, color=C["text"])
    q = 4  # query token index ("the")
    for i in range(n):
        d = i - q
        axB.text(i, 0.72, f"{d:+d}" if d != 0 else "query", ha="center",
                 va="center", fontsize=11,
                 fontweight="bold", color=(C["orange"] if d != 0 else C["grey"]))
        if d != 0:
            axB.annotate("", xy=(q, 0.42), xytext=(i, 0.42),
                         arrowprops=dict(arrowstyle="-|>", color=C["orange"],
                                         lw=1.3, alpha=0.55,
                                         connectionstyle="arc3,rad=-0.25"))
    axB.set_xlim(-0.8, n - 0.2)
    axB.set_ylim(-0.9, 1.15)
    axB.axis("off")
    axB.text(0.5, -0.85,
             "Only the offset (query$-$key) matters, so the same rule\n"
             "works no matter where the pair sits in the sequence.",
             transform=axB.transAxes, ha="center", fontsize=10.5,
             style="italic", color=C["grey"])

    fig.tight_layout()
    save(fig, "pe_absolute_vs_relative.jpg")


# ---------------------------------------------------------------------------
# 2. Sinusoidal PE heatmap
# ---------------------------------------------------------------------------
def fig_sinusoidal_heatmap():
    d_model = 128
    max_pos = 100
    pos = np.arange(max_pos)[:, None]
    i = np.arange(d_model)[None, :]
    # angle rates: 1 / 10000^(2*(i//2)/d)
    div = np.power(10000.0, (2 * (i // 2)) / d_model)
    angles = pos / div
    PE = np.zeros((max_pos, d_model))
    PE[:, 0::2] = np.sin(angles[:, 0::2])
    PE[:, 1::2] = np.cos(angles[:, 1::2])

    fig, ax = plt.subplots(figsize=(9.5, 5.2))
    im = ax.imshow(PE, aspect="auto", cmap="RdBu_r", vmin=-1, vmax=1,
                   origin="lower", interpolation="nearest")
    ax.set_xlabel("embedding dimension  $i$  (0 … 127)", fontsize=12)
    ax.set_ylabel("position  $m$", fontsize=12)
    ax.set_title("Sinusoidal position encoding  $PE[m, i]$\n"
                 "low dimensions = fast (short wavelength), "
                 "high dimensions = slow (long wavelength)",
                 fontsize=12.5, fontweight="bold", color=C["text"])
    cb = fig.colorbar(im, ax=ax, fraction=0.045, pad=0.02)
    cb.set_label("value", fontsize=11)
    # annotate the fast/slow ends (white bbox so text stays readable over the map)
    bbox = dict(boxstyle="round,pad=0.3", fc="white", ec=C["grey"], lw=1.0, alpha=0.92)
    ax.annotate("fast dims\n(distinguish\nnearby positions)",
                xy=(3, 50), xytext=(22, 40), fontsize=9.5, color=C["text"],
                ha="center", bbox=bbox,
                arrowprops=dict(arrowstyle="-|>", color=C["grey"], lw=1.4))
    ax.annotate("slow dims\n(distinguish\nfar-apart positions)",
                xy=(124, 50), xytext=(102, 40), fontsize=9.5, color=C["text"],
                ha="center", bbox=bbox,
                arrowprops=dict(arrowstyle="-|>", color=C["grey"], lw=1.4))
    fig.tight_layout()
    save(fig, "pe_sinusoidal_heatmap.jpg")


# ---------------------------------------------------------------------------
# 3. RoPE rotation of a single 2D pair
# ---------------------------------------------------------------------------
def fig_rope_rotation():
    fig, ax = plt.subplots(figsize=(6.2, 6.2))
    theta = np.deg2rad(35)          # rotation per position step
    v0 = np.array([1.0, 0.35])      # the raw 2D pair (d0, d1)
    v0 = v0 / np.linalg.norm(v0) * 1.0
    positions = [0, 1, 2, 3]
    cols = [C["grey"], C["blue"], C["green"], C["orange"]]

    # unit circle
    t = np.linspace(0, 2 * np.pi, 200)
    ax.plot(np.cos(t), np.sin(t), color=C["lgrey"], lw=1.4, zorder=0)
    ax.axhline(0, color="#cccccc", lw=1, zorder=0)
    ax.axvline(0, color="#cccccc", lw=1, zorder=0)

    for m, col in zip(positions, cols):
        a = m * theta
        R = np.array([[np.cos(a), -np.sin(a)], [np.sin(a), np.cos(a)]])
        v = R @ v0
        ax.add_patch(FancyArrowPatch((0, 0), (v[0], v[1]),
                     arrowstyle="-|>", mutation_scale=20, color=col, lw=2.4,
                     zorder=3))
        ax.text(v[0] * 1.12, v[1] * 1.12, f"$m={m}$", color=col, fontsize=13,
                fontweight="bold", ha="center", va="center")

    # angle arc between m=0 and m=1
    ax.add_patch(Arc((0, 0), 0.9, 0.9, angle=0,
                     theta1=np.rad2deg(np.arctan2(v0[1], v0[0])),
                     theta2=np.rad2deg(np.arctan2(v0[1], v0[0]) + theta),
                     color=C["blue"], lw=1.8))
    ax.text(0.62, 0.42, r"$\theta$", color=C["blue"], fontsize=15)

    ax.set_xlim(-1.3, 1.3)
    ax.set_ylim(-1.3, 1.3)
    ax.set_aspect("equal")
    ax.set_xlabel("dimension $d_0$", fontsize=12)
    ax.set_ylabel("dimension $d_1$", fontsize=12)
    ax.set_title("RoPE on one dimension pair $(d_0, d_1)$\n"
                 "position $m$ rotates the vector by angle $m\\theta$",
                 fontsize=12.5, fontweight="bold", color=C["text"])
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    save(fig, "rope_rotation.jpg")


# ---------------------------------------------------------------------------
# 4. Multi-frequency rotation: angle vs position for several pairs
# ---------------------------------------------------------------------------
def fig_rope_multifreq():
    d = 128
    positions = np.arange(0, 64)
    # frequencies omega_i = 10000^(-2i/d), i = 0..d/2-1
    idxs = [4, 12, 28, 63]     # a fast, two medium, and the slowest pair
    labels = ["pair 4 (fast)", "pair 12", "pair 28", "pair 63 (slowest)"]
    cols = [C["orange"], C["green"], C["blue"], C["purple"]]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12.5, 5.0),
                                   gridspec_kw={"width_ratios": [1.4, 1]})

    # dense sampling so the fast sinusoid reads as a smooth curve, not aliased
    dense = np.linspace(0, 63, 800)
    for i, lab, col in zip(idxs, labels, cols):
        omega = 10000.0 ** (-2.0 * i / d)
        ax1.plot(dense, np.sin(dense * omega), color=col, lw=2.2,
                 label=f"{lab},  $\\omega={omega:.1e}$")
    ax1.set_xlabel("position  $m$", fontsize=12)
    ax1.set_ylabel(r"$\sin(m\,\omega_i)$", fontsize=12)
    ax1.set_title("Different dimension pairs rotate at different speeds",
                  fontsize=12.5, fontweight="bold", color=C["text"])
    ax1.legend(fontsize=9.5, loc="lower left")
    ax1.axhline(0, color="#cccccc", lw=0.8)
    ax1.spines[["top", "right"]].set_visible(False)

    # right panel: the angle table intuition -- angles grow linearly with position,
    # slope = omega_i, drawn as accumulated rotation for a few pairs.
    for i, lab, col in zip(idxs, labels, cols):
        omega = 10000.0 ** (-2.0 * i / d)
        ax2.plot(positions, positions * omega, color=col, lw=2.2)
    ax2.set_xlabel("position  $m$", fontsize=12)
    ax2.set_ylabel(r"accumulated angle  $m\,\omega_i$  (radians)", fontsize=12)
    ax2.set_title("Angle = position $\\times$ frequency\n"
                  "fast pairs sweep huge angles, slow pairs barely move",
                  fontsize=12, fontweight="bold", color=C["text"])
    ax2.set_yscale("symlog")
    ax2.spines[["top", "right"]].set_visible(False)

    fig.tight_layout()
    save(fig, "rope_multifreq.jpg")


if __name__ == "__main__":
    fig_absolute_vs_relative()
    fig_sinusoidal_heatmap()
    fig_rope_rotation()
    fig_rope_multifreq()
    print("done.")
