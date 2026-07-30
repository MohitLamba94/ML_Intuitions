"""Generate figures for gated-attention.md (NeurIPS 2025 Best Paper: Gated Attention).

Outputs (all .jpg, per repo convention):
  assets/gated_positions_g1_g5.jpg      -- where the five gate positions G1..G5 sit in an attention block
  assets/gated_lowrank_nonlinearity.jpg -- W_V . W_O collapses to one low-rank linear map; the gate breaks it
  assets/gated_gate_sparsity_hist.jpg   -- gate-score distribution: G1 sparse near 0 (mean ~0.116) vs denser G2
  assets/gated_attention_sink_heatmap.jpg -- attention maps: baseline (bright first-token column) vs gated

All schematic / illustrative (synthetic data), matplotlib only. Numbers quoted in
labels (46.7% -> 4.8% first-token attention, mean gate 0.116) are from
Qiu et al., "Gated Attention for Large Language Models" (arXiv:2505.06708).

Run with the repo 'general' conda env:
  /Users/mohitl/Documents/miniconda3/envs/general/bin/python scripts/make_gated_attention_figs.py
"""
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

ASSETS = os.path.join(os.path.dirname(__file__), "..", "assets")

# Same brand-neutral palette as make_attention_figs.py / make_norm_figs.py.
C = {
    "blue":   "#6C8EBF",
    "purple": "#9673A6",
    "green":  "#82B366",
    "orange": "#D6795B",
    "red":    "#C0504D",
    "grey":   "#555555",
    "lgrey":  "#EDEDED",
    "yellow": "#E8B84B",
    "text":   "#222222",
}
DPI = 150
np.random.seed(7)


def save(fig, name):
    path = os.path.join(ASSETS, name)
    fig.savefig(path, dpi=DPI, bbox_inches="tight", facecolor="white",
                pil_kwargs={"quality": 90})
    plt.close(fig)
    print("wrote", os.path.normpath(path))


# ---------------------------------------------------------------------------
# 1. The five gate positions inside one attention block.
#    X -> {Wq,Wk,Wv} -> SDPA -> Wo -> out.  Gates G1..G5 hang off the pipeline.
#    G1 (post-SDPA, pre-Wo, query-conditioned) is the winner -> highlighted.
# ---------------------------------------------------------------------------
def fig_positions():
    fig, ax = plt.subplots(figsize=(12.5, 5.6))
    ax.set_xlim(0, 12.5)
    ax.set_ylim(0, 5.6)
    ax.axis("off")

    def box(x, y, w, h, label, fc, txt=None, fs=11, ec=None):
        ax.add_patch(FancyBboxPatch((x, y), w, h,
                     boxstyle="round,pad=0.02,rounding_size=0.06",
                     fc=fc, ec=ec or C["grey"], lw=1.4))
        ax.text(x + w / 2, y + h / 2, label, ha="center", va="center",
                fontsize=fs, color=txt or C["text"], zorder=5)

    def arrow(x1, y1, x2, y2, col=None, lw=1.8):
        ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2),
                     arrowstyle="-|>", mutation_scale=15,
                     color=col or C["grey"], lw=lw, zorder=1))

    yb = 2.4  # main pipeline baseline
    h = 0.9
    # main pipeline blocks
    box(0.2, yb, 1.3, h, "X\n(hidden)", C["lgrey"], fs=10)
    box(2.0, yb, 1.5, h, "Q,K,V\nWq Wk Wv", C["blue"], txt="white", fs=9.5)
    box(4.0, yb, 1.9, h, "SDPA\nsoftmax(QKᵀ)V", C["purple"], txt="white", fs=9)
    box(6.4, yb, 1.5, h, "output\nWo", C["green"], txt="white", fs=10)
    box(8.4, yb, 1.4, h, "out", C["lgrey"], fs=10)

    arrow(1.5, yb + h / 2, 2.0, yb + h / 2)
    arrow(3.5, yb + h / 2, 4.0, yb + h / 2)
    arrow(5.9, yb + h / 2, 6.4, yb + h / 2)
    arrow(7.9, yb + h / 2, 8.4, yb + h / 2)

    # gate glyphs: small circled x above/below where each gate multiplies.
    def gate(x, y, tag, col, best=False):
        r = 0.28
        ax.add_patch(plt.Circle((x, y), r, fc="white",
                     ec=col, lw=2.6 if best else 1.8, zorder=6))
        ax.text(x, y, "×", ha="center", va="center", fontsize=15,
                color=col, zorder=7)
        ax.text(x, y + (r + 0.28 if y > yb else -(r + 0.28)), tag,
                ha="center", va="center", fontsize=11,
                fontweight="bold" if best else "normal", color=col, zorder=7)

    # G1: post-SDPA, pre-Wo (on the arrow between SDPA and Wo) -- the winner.
    gx1 = 6.15
    gate(gx1, yb + h + 0.55, "G1", C["red"], best=True)
    arrow(gx1, yb + h + 0.27, gx1, yb + h + 0.02, col=C["red"], lw=2.2)
    ax.text(gx1, yb + h + 1.15, "best:\npost-SDPA,\nquery-conditioned",
            ha="center", va="bottom", fontsize=8.5, color=C["red"])

    # G4 on Q path, G3 on K path, G2 on V path (all inside/around QKV block).
    gate(2.75, yb - 0.75, "G4 · Q", C["grey"])
    gate(2.75, yb + h + 0.75, "G3 · K", C["grey"])
    gate(4.95, yb - 0.75, "G2 · V", C["orange"])
    ax.text(4.95, yb - 1.55, "value gate:\ndenser, weaker",
            ha="center", va="top", fontsize=8.5, color=C["orange"])
    # G5 after Wo (final output).
    gate(8.1, yb - 0.75, "G5 · out", C["grey"])

    ax.text(6.25, 5.25, "Five places a gate can sit in an attention block — G1 wins",
            ha="center", va="center", fontsize=13, fontweight="bold",
            color=C["text"])
    ax.text(6.25, 0.28,
            "Every gate multiplies its stream by  σ(X Wθ) ∈ (0,1), elementwise & per head.",
            ha="center", va="center", fontsize=10, color=C["grey"])
    save(fig, "gated_positions_g1_g5.jpg")


# ---------------------------------------------------------------------------
# 2. Low-rank collapse vs gate-injected non-linearity.
#    Top: V=X Wv (d_head) then out=Y Wo  -> the composed map Wv Wo is a single
#    linear map of rank <= d_head (a bottleneck), no non-linearity between.
#    Bottom: insert sigmoid gate between -> the path is now non-linear.
# ---------------------------------------------------------------------------
def fig_lowrank():
    fig, ax = plt.subplots(figsize=(12.0, 6.2))
    ax.set_xlim(0, 12.0)
    ax.set_ylim(0, 6.2)
    ax.axis("off")

    def box(x, y, w, h, label, fc, txt=None, fs=10.5):
        ax.add_patch(FancyBboxPatch((x, y), w, h,
                     boxstyle="round,pad=0.02,rounding_size=0.06",
                     fc=fc, ec=C["grey"], lw=1.4))
        ax.text(x + w / 2, y + h / 2, label, ha="center", va="center",
                fontsize=fs, color=txt or C["text"], zorder=5)

    def arrow(x1, y1, x2, y2, col=None, lw=1.8):
        ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2),
                     arrowstyle="-|>", mutation_scale=14,
                     color=col or C["grey"], lw=lw, zorder=1))

    h = 0.85
    # ---- top row: standard attention path ----
    yt = 4.5
    ax.text(0.1, yt + h + 0.55, "Standard attention: a linear bottleneck",
            ha="left", va="center", fontsize=12.5, fontweight="bold",
            color=C["text"])
    box(0.2, yt, 1.5, h, "Y\n(SDPA out)", C["purple"], txt="white", fs=9.5)
    box(2.4, yt, 1.6, h, "· Wv-side\n(d_head)", C["blue"], txt="white", fs=9)
    box(4.7, yt, 1.4, h, "· Wo", C["green"], txt="white")
    box(6.8, yt, 1.5, h, "out", C["lgrey"])
    arrow(1.7, yt + h / 2, 2.4, yt + h / 2)
    arrow(4.0, yt + h / 2, 4.7, yt + h / 2)
    arrow(6.1, yt + h / 2, 6.8, yt + h / 2)
    # brace: Wv and Wo collapse
    ax.annotate("", xy=(6.1, yt - 0.35), xytext=(2.4, yt - 0.35),
                arrowprops=dict(arrowstyle="-", color=C["red"], lw=1.6))
    ax.text(4.25, yt - 0.85,
            "two linear maps compose into ONE map of rank ≤ d_head — nothing non-linear between them",
            ha="center", va="center", fontsize=9.5, color=C["red"])

    # ---- bottom row: gated path ----
    yb = 1.4
    ax.text(0.1, yb + h + 0.55, "Gated attention: sigmoid breaks the bottleneck",
            ha="left", va="center", fontsize=12.5, fontweight="bold",
            color=C["text"])
    box(0.2, yb, 1.5, h, "Y\n(SDPA out)", C["purple"], txt="white", fs=9.5)
    box(2.4, yb, 1.9, h, "⊙ σ(X Wθ)\ngate", C["orange"], txt="white", fs=9.5)
    box(5.0, yb, 1.4, h, "· Wo", C["green"], txt="white")
    box(7.1, yb, 1.5, h, "out", C["lgrey"])
    arrow(1.7, yb + h / 2, 2.4, yb + h / 2)
    arrow(4.3, yb + h / 2, 5.0, yb + h / 2)
    arrow(6.4, yb + h / 2, 7.1, yb + h / 2)
    ax.text(4.35, yb - 0.6,
            "the gate is the only non-linearity between V and Wo → strictly more expressive",
            ha="center", va="center", fontsize=9.5, color=C["green"])

    # a little sigmoid inset on the right
    axx = fig.add_axes([0.74, 0.30, 0.20, 0.34])
    z = np.linspace(-6, 6, 200)
    axx.plot(z, 1 / (1 + np.exp(-z)), color=C["orange"], lw=2.2)
    axx.set_title("σ(z) ∈ (0,1)", fontsize=9)
    axx.axhline(0, color=C["grey"], lw=0.6)
    axx.axvline(0, color=C["grey"], lw=0.6)
    axx.set_xticks([]); axx.set_yticks([0, 1])
    axx.tick_params(labelsize=8)
    for s in axx.spines.values():
        s.set_edgecolor(C["lgrey"])
    save(fig, "gated_lowrank_nonlinearity.jpg")


# ---------------------------------------------------------------------------
# 3. Gate-score distribution: G1 (post-SDPA) is sparse, mass piled near 0
#    (mean ~0.116); G2 (value gate) is denser / more spread.
# ---------------------------------------------------------------------------
def fig_sparsity():
    fig, ax = plt.subplots(figsize=(8.6, 5.0))
    # G1: heavily right-skewed toward 0 (Beta with small mean).
    g1 = np.random.beta(0.55, 4.0, 60000)
    g1 = g1 * (0.116 / g1.mean())        # nudge mean to ~0.116
    g1 = np.clip(g1, 0, 1)
    # G2: broader, more mass in the middle (denser gating).
    g2 = np.random.beta(2.2, 2.6, 60000)

    bins = np.linspace(0, 1, 51)
    ax.hist(g1, bins=bins, density=True, color=C["red"], alpha=0.72,
            label=f"G1 · post-SDPA gate (mean ≈ 0.12) — sparse")
    ax.hist(g2, bins=bins, density=True, color=C["orange"], alpha=0.5,
            label="G2 · value gate (mean ≈ 0.45) — dense")
    ax.axvline(g1.mean(), color=C["red"], ls="--", lw=1.6)
    ax.set_xlabel("gate score  σ(X Wθ)", fontsize=11)
    ax.set_ylabel("density", fontsize=11)
    ax.set_title("The winning gate is sparse: most heads stay near 0 and 'abstain'",
                 fontsize=12.5, color=C["text"])
    ax.legend(fontsize=10, frameon=False)
    ax.set_xlim(0, 1)
    for s in ["top", "right"]:
        ax.spines[s].set_visible(False)
    ax.grid(axis="y", color=C["lgrey"], lw=0.8)
    ax.set_axisbelow(True)
    save(fig, "gated_gate_sparsity_hist.jpg")


# ---------------------------------------------------------------------------
# 4. Attention-sink heatmaps: baseline pours mass on the first token (bright
#    column 0), gated model spreads attention along the causal diagonal.
# ---------------------------------------------------------------------------
def fig_sink():
    n = 40
    ii, jj = np.meshgrid(np.arange(n), np.arange(n), indexing="ij")

    def causal_softmax(logits):
        mask = jj > ii
        logits = np.where(mask, -1e9, logits)
        logits = logits - logits.max(axis=1, keepdims=True)
        e = np.exp(logits)
        return e / e.sum(axis=1, keepdims=True)

    # baseline: strong constant bias onto column 0 (the sink) + mild local band.
    base_logits = np.zeros((n, n))
    base_logits += -0.25 * np.abs(ii - jj)          # weak locality
    base_logits[:, 0] += 6.0                          # huge sink bias on token 0
    base_logits[0, 0] = 6.0
    A_base = causal_softmax(base_logits)

    # gated: no sink bias, attention follows a local causal band.
    gate_logits = np.zeros((n, n))
    gate_logits += -0.35 * np.abs(ii - jj)
    gate_logits[:, 0] += 0.3                          # only tiny residual
    A_gate = causal_softmax(gate_logits)

    first_base = A_base[1:, 0].mean() * 100
    first_gate = A_gate[1:, 0].mean() * 100

    fig, axes = plt.subplots(1, 2, figsize=(11.0, 5.2))
    for ax, A, title, pct, col in [
        (axes[0], A_base, "Baseline softmax attention", first_base, C["red"]),
        (axes[1], A_gate, "Gated attention", first_gate, C["green"]),
    ]:
        im = ax.imshow(A, cmap="magma", aspect="equal", vmin=0, vmax=0.6)
        ax.set_title(title, fontsize=12)
        ax.set_xlabel("key position (0 = first token)", fontsize=10)
        ax.set_ylabel("query position", fontsize=10)
        ax.text(0.5, -0.19,
                f"avg attention on token 0: {pct:.1f}%",
                transform=ax.transAxes, ha="center", va="top",
                fontsize=11, color=col, fontweight="bold")
    fig.suptitle("The gate drains the attention sink: 46.7% → 4.8% of mass leaves token 0",
                 fontsize=13, fontweight="bold", y=1.02)
    fig.colorbar(im, ax=axes, fraction=0.025, pad=0.02, label="attention weight")
    save(fig, "gated_attention_sink_heatmap.jpg")


if __name__ == "__main__":
    fig_positions()
    fig_lowrank()
    fig_sparsity()
    fig_sink()
    print("done.")
