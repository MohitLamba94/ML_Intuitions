"""Generate figures for training_llms/005_Attention.md.

Outputs (all .jpg, per repo convention):
  assets/attn_matmul_flow.jpg   -- single-head attention as a matrix-multiply pipeline
  assets/attn_sqrt_dk.jpg       -- why 1/sqrt(d_k): softmax saturation as d_k grows
  assets/attn_multihead.jpg     -- split d_model into h heads, attend, concat, W^O
  assets/attn_mha_mqa_gqa.jpg   -- MHA vs GQA vs MQA head-sharing
  assets/attn_gqa_perf_vs_time.jpg    -- Ainslie et al. 2023 Fig 3 (recreated from paper data)
  assets/attn_gqa_uptraining.jpg      -- Ainslie et al. 2023 Fig 5 (recreated from paper data)
  assets/attn_gqa_time_vs_groups.jpg  -- Ainslie et al. 2023 Fig 6 (recreated from paper data)
  assets/attn_qknorm_stability.jpg    -- schematic: logit growth -> entropy collapse, QK-norm fix

The three GQA-paper figures are pgfplots/TikZ figures in the arXiv source (not
raster images), so we recreate them in matplotlib using the exact data points
extracted from arXiv:2305.13245 (figures/results.tex, uptraining_steps.tex,
time_vs_groups.tex). Credit Ainslie et al. (2023).

Run with the repo 'general' conda env:
  /Users/mohitl/Documents/miniconda3/envs/general/bin/python scripts/make_attention_figs.py
"""
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

ASSETS = os.path.join(os.path.dirname(__file__), "..", "assets")

# A soft, consistent, brand-neutral palette (matches make_norm_figs.py / make_pe_figs.py).
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
# 1. Single-head attention as a chain of matrix multiplies.
#    Q (n x d_k)  x  K^T (d_k x n)  ->  scores S (n x n)  -> row-softmax ->
#    A (n x n)  x  V (n x d_v)  ->  out (n x d_v).
#    We draw each operand as a little coloured matrix block with shape labels,
#    plus a real softmax'd score heatmap in the middle so the "who attends to
#    whom" picture is concrete.
# ---------------------------------------------------------------------------
def fig_matmul_flow():
    fig, ax = plt.subplots(figsize=(13.0, 5.2))
    ax.set_xlim(0, 13.0)
    ax.set_ylim(0, 5.2)
    ax.axis("off")

    def block(x, y, w, h, label, shape, fc, txt_col=None):
        ax.add_patch(FancyBboxPatch((x, y), w, h,
                     boxstyle="round,pad=0.02,rounding_size=0.06",
                     fc=fc, ec=C["grey"], lw=1.3))
        ax.text(x + w / 2, y + h / 2 + 0.16, label, ha="center", va="center",
                fontsize=12, fontweight="bold", color=txt_col or C["text"])
        ax.text(x + w / 2, y + h / 2 - 0.32, shape, ha="center", va="center",
                fontsize=9.5, color=txt_col or C["grey"])

    def op(x, y, sym):
        ax.text(x, y, sym, ha="center", va="center", fontsize=20,
                color=C["grey"])

    yc = 2.9  # vertical centre of the operand row

    # Q
    block(0.2, yc - 0.7, 1.5, 1.4, "$Q$", "$n \\times d_k$", C["blue"], "white")
    op(2.05, yc, r"$\times$")
    # K^T
    block(2.45, yc - 0.7, 1.5, 1.4, "$K^{\\top}$", "$d_k \\times n$",
          C["purple"], "white")
    op(4.35, yc, r"$=$")

    # scores heatmap S = softmax(QK^T / sqrt(dk)), causal
    n = 6
    rng = np.random.default_rng(3)
    raw = rng.normal(size=(n, n)) * 0.8
    raw = raw + np.linspace(0.4, -0.2, n)[None, :]      # mild structure
    mask = np.triu(np.ones((n, n)), k=1).astype(bool)    # causal: no future
    raw[mask] = -np.inf
    A = np.exp(raw - raw.max(axis=1, keepdims=True))
    A = A / A.sum(axis=1, keepdims=True)
    hm_x, hm_y, hm_s = 4.75, yc - 1.05, 2.1
    ax.imshow(A, extent=[hm_x, hm_x + hm_s, hm_y, hm_y + hm_s],
              cmap="Blues", vmin=0, vmax=1, aspect="auto", zorder=2)
    ax.add_patch(plt.Rectangle((hm_x, hm_y), hm_s, hm_s, fill=False,
                 ec=C["grey"], lw=1.3, zorder=3))
    ax.text(hm_x + hm_s / 2, hm_y + hm_s + 0.30,
            "$A=\\mathrm{softmax}\\!\\left(\\dfrac{QK^{\\top}}{\\sqrt{d_k}}\\right)$",
            ha="center", va="center", fontsize=12.5, color=C["text"])
    ax.text(hm_x + hm_s / 2, hm_y - 0.34, "$n \\times n$  (rows sum to 1)",
            ha="center", va="center", fontsize=9.5, color=C["grey"])
    ax.text(hm_x - 0.16, hm_y + hm_s / 2, "query $i$", ha="center",
            va="center", rotation=90, fontsize=8.5, color=C["grey"])
    ax.text(hm_x + hm_s / 2, hm_y - 0.02, "key $j$", ha="center",
            va="bottom", fontsize=8.5, color="white", zorder=4)

    op(7.35, yc, r"$\times$")
    # V
    block(7.75, yc - 0.7, 1.5, 1.4, "$V$", "$n \\times d_v$", C["green"], "white")
    op(9.65, yc, r"$=$")
    # output
    block(10.05, yc - 0.7, 1.7, 1.4, "output", "$n \\times d_v$", C["orange"],
          "white")

    # captions top and bottom
    ax.text(6.5, 4.85,
            "Single-head attention is just two matrix multiplies with a "
            "row-wise softmax in between",
            ha="center", fontsize=13, fontweight="bold", color=C["text"])
    ax.text(2.55, 1.35,
            "step 1: every query dotted with every key\n$\\Rightarrow$ an "
            "$n\\times n$ table of similarities",
            ha="center", fontsize=9.5, style="italic", color=C["grey"])
    ax.text(10.35, 1.35,
            "step 2: each output row is a weighted\naverage of the value rows",
            ha="center", fontsize=9.5, style="italic", color=C["grey"])
    fig.tight_layout()
    save(fig, "attn_matmul_flow.jpg")


# ---------------------------------------------------------------------------
# 2. Why 1/sqrt(d_k). Left: same "shape" of logits, but scaled up by growing
#    d_k, softmax collapses to near one-hot. Right: max softmax weight vs d_k,
#    unscaled (saturates -> 1) vs scaled (stays moderate).
# ---------------------------------------------------------------------------
def fig_sqrt_dk():
    fig, (axL, axR) = plt.subplots(1, 2, figsize=(12.5, 4.6))

    # --- left: softmax of logits whose spread grows like sqrt(d_k) ---
    base = np.array([1.0, 0.6, 0.2, -0.2, -0.6, -1.0, 0.1, -0.3])
    base = base / base.std()  # unit-variance "raw" logit shape
    dks = [8, 64, 512]
    xpos = np.arange(len(base))
    width = 0.26
    colours = [C["green"], C["blue"], C["orange"]]
    for k, dk in enumerate(dks):
        logits = base * np.sqrt(dk)         # unscaled scores grow like sqrt(d_k)
        p = np.exp(logits - logits.max())
        p = p / p.sum()
        axL.bar(xpos + (k - 1) * width, p, width,
                color=colours[k], label=f"$d_k={dk}$ (no scaling)")
    axL.set_title("Unscaled scores grow with $d_k$;\nsoftmax collapses toward "
                  "one-hot", fontsize=12, fontweight="bold", color=C["text"])
    axL.set_xlabel("key position $j$", fontsize=11)
    axL.set_ylabel("attention weight", fontsize=11)
    axL.set_xticks(xpos)
    axL.legend(fontsize=9.5, frameon=False)
    axL.spines[["top", "right"]].set_visible(False)

    # --- right: peakiness (max weight) vs d_k, scaled vs unscaled ---
    dk_grid = np.array([4, 8, 16, 32, 64, 128, 256, 512, 1024])
    rng = np.random.default_rng(0)
    trials = 400
    maxw_unscaled, maxw_scaled = [], []
    for dk in dk_grid:
        q = rng.normal(size=(trials, dk))
        k = rng.normal(size=(trials, 8, dk))       # 8 keys per query
        scores = np.einsum("td,tkd->tk", q, k)     # ~ variance d_k
        for scores_v, store in ((scores, maxw_unscaled),
                                (scores / np.sqrt(dk), maxw_scaled)):
            p = np.exp(scores_v - scores_v.max(axis=1, keepdims=True))
            p = p / p.sum(axis=1, keepdims=True)
            store.append(p.max(axis=1).mean())
    axR.plot(dk_grid, maxw_unscaled, "o-", color=C["orange"], lw=2,
             label="no scaling")
    axR.plot(dk_grid, maxw_scaled, "s-", color=C["blue"], lw=2,
             label="scaled by $1/\\sqrt{d_k}$")
    axR.axhline(1.0 / 8, color=C["grey"], ls="--", lw=1.2)
    axR.text(dk_grid[0], 1.0 / 8 + 0.03, "uniform (1/8)", fontsize=9,
             color=C["grey"])
    axR.set_xscale("log", base=2)
    axR.set_ylim(0, 1.05)
    axR.set_title("Average peak weight vs $d_k$\n(8 keys, random unit-variance "
                  "Q,K)", fontsize=12, fontweight="bold", color=C["text"])
    axR.set_xlabel("$d_k$ (log scale)", fontsize=11)
    axR.set_ylabel("mean of max softmax weight", fontsize=11)
    axR.legend(fontsize=10, frameon=False, loc="center right")
    axR.spines[["top", "right"]].set_visible(False)

    fig.suptitle("Dividing by $\\sqrt{d_k}$ keeps softmax in a responsive "
                 "regime instead of a saturated, near-zero-gradient one",
                 fontsize=13, fontweight="bold", color=C["text"])
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    save(fig, "attn_sqrt_dk.jpg")


# ---------------------------------------------------------------------------
# 3. Multi-head attention: split the d_model-wide token vectors into h slices,
#    run attention independently per slice, concat, mix with W^O.
# ---------------------------------------------------------------------------
def fig_multihead():
    fig, ax = plt.subplots(figsize=(13.0, 5.4))
    ax.set_xlim(0, 13.0)
    ax.set_ylim(0, 5.4)
    ax.axis("off")

    h = 4
    head_cols = [C["blue"], C["purple"], C["green"], C["orange"]]

    # input token matrix (n x d_model), shown as h coloured column-slices
    x0, y0, slice_w, tot_h = 0.4, 1.6, 0.55, 2.2
    ax.text(x0 + h * slice_w / 2, y0 + tot_h + 0.55,
            "token matrix\n$X\\;(n \\times d_{model})$", ha="center",
            fontsize=11, color=C["text"])
    for i in range(h):
        ax.add_patch(FancyBboxPatch((x0 + i * slice_w, y0), slice_w * 0.92, tot_h,
                     boxstyle="round,pad=0.01,rounding_size=0.04",
                     fc=head_cols[i], ec=C["grey"], lw=1.1, alpha=0.85))
    ax.annotate("", xy=(x0 + h * slice_w / 2, y0 - 0.18),
                xytext=(x0 + h * slice_w / 2, y0 - 0.02),
                arrowprops=dict(arrowstyle="-|>", color=C["grey"], lw=1.4))
    ax.text(x0 + h * slice_w / 2, y0 - 0.55,
            "split width into $h$ heads\n$d_k = d_{model}/h$", ha="center",
            fontsize=9.5, style="italic", color=C["grey"])

    # per-head attention blocks
    bx0, bw, bh = 3.4, 2.1, 0.78
    ys = [4.0, 2.9, 1.8, 0.7]
    for i in range(h):
        ax.add_patch(FancyBboxPatch((bx0, ys[i]), bw, bh,
                     boxstyle="round,pad=0.02,rounding_size=0.10",
                     fc=head_cols[i], ec=C["grey"], lw=1.3, alpha=0.85))
        ax.text(bx0 + bw / 2, ys[i] + bh / 2,
                f"head {i+1} attention\n$(n \\times d_k)$", ha="center",
                va="center", fontsize=9.5, color="white")
        # arrow from input to head
        ax.add_patch(FancyArrowPatch((x0 + i * slice_w + slice_w * 0.46,
                                      y0 + tot_h * (1 - (i + 0.5) / h)),
                                     (bx0, ys[i] + bh / 2),
                     arrowstyle="-|>", mutation_scale=12, color=head_cols[i],
                     lw=1.3, connectionstyle="arc3,rad=0.06"))
        ax.text(bx0 + bw / 2, ys[i] + bh + 0.02,
                "own $W^Q_i,W^K_i,W^V_i$", ha="center", va="bottom",
                fontsize=8, style="italic", color=C["grey"])

    # concat block
    cx, cy, cw, ch = 6.4, 1.6, 1.5, 2.2
    ax.add_patch(FancyBboxPatch((cx, cy), cw, ch,
                 boxstyle="round,pad=0.02,rounding_size=0.06",
                 fc=C["lgrey"], ec=C["grey"], lw=1.3))
    frac = ch / h
    for i in range(h):
        ax.add_patch(plt.Rectangle((cx + 0.12, cy + ch - (i + 1) * frac + 0.04),
                     cw - 0.24, frac - 0.08, fc=head_cols[i], ec="none",
                     alpha=0.85))
    ax.text(cx + cw / 2, cy + ch + 0.35, "concat\n$(n \\times d_{model})$",
            ha="center", fontsize=10.5, color=C["text"])
    for i in range(h):
        ax.add_patch(FancyArrowPatch((bx0 + bw, ys[i] + bh / 2),
                     (cx, cy + ch - (i + 0.5) * frac),
                     arrowstyle="-|>", mutation_scale=11, color=C["grey"],
                     lw=1.1, connectionstyle="arc3,rad=0.0"))

    # W^O
    ox, oy, ow, oh = 8.6, 2.3, 1.6, 0.9
    ax.add_patch(FancyArrowPatch((cx + cw, cy + ch / 2), (ox, oy + oh / 2),
                 arrowstyle="-|>", mutation_scale=14, color=C["grey"], lw=1.5))
    ax.add_patch(FancyBboxPatch((ox, oy), ow, oh,
                 boxstyle="round,pad=0.02,rounding_size=0.10",
                 fc=C["yellow"], ec=C["grey"], lw=1.4))
    ax.text(ox + ow / 2, oy + oh / 2, "mix\n$\\times\\,W^O$", ha="center",
            va="center", fontsize=10.5, color=C["text"])

    # output
    ex, ey, ew, eh = 10.8, 2.3, 1.9, 0.9
    ax.add_patch(FancyArrowPatch((ox + ow, oy + oh / 2), (ex, ey + eh / 2),
                 arrowstyle="-|>", mutation_scale=14, color=C["grey"], lw=1.5))
    ax.add_patch(FancyBboxPatch((ex, ey), ew, eh,
                 boxstyle="round,pad=0.02,rounding_size=0.08",
                 fc=C["orange"], ec=C["grey"], lw=1.4))
    ax.text(ex + ew / 2, ey + eh / 2, "output\n$(n \\times d_{model})$",
            ha="center", va="center", fontsize=10.5, color="white")

    ax.text(6.5, 5.15,
            "Multi-head attention: $h$ small attentions on slices of the width, "
            "run in parallel, then recombined",
            ha="center", fontsize=13, fontweight="bold", color=C["text"])
    ax.text(6.5, 0.28,
            "example: $d_{model}=512,\\; h=8 \\Rightarrow d_k=64$   —   total "
            "width (and compute) is preserved, just carved into $h$ subspaces",
            ha="center", fontsize=10, style="italic", color=C["grey"])
    fig.tight_layout()
    save(fig, "attn_multihead.jpg")


# ---------------------------------------------------------------------------
# 4. MHA vs GQA vs MQA: how many KV heads the query heads share.
# ---------------------------------------------------------------------------
def fig_mha_mqa_gqa():
    fig, axes = plt.subplots(1, 3, figsize=(13.0, 4.8))

    def draw(ax, title, subtitle, kv_of_query, n_kv):
        # 8 query heads across the top, n_kv KV heads across the bottom.
        n_q = 8
        qy, ky = 2.6, 0.6
        qw = 0.86
        ax.set_xlim(0, 8.4)
        ax.set_ylim(0, 4.3)
        ax.axis("off")
        kv_cols = [C["blue"], C["purple"], C["green"], C["orange"],
                   C["yellow"], C["grey"], C["blue"], C["purple"]]
        # KV heads (bottom), spaced across the width
        kv_x = np.linspace(0.6, 7.8 - 8.4 / n_kv + 0.6, n_kv) if n_kv > 1 else [4.2 - 0.5]
        kv_cx = []
        kvw = min(1.4, 7.2 / n_kv)
        step = 7.4 / n_kv
        for j in range(n_kv):
            cx = 0.5 + step * (j + 0.5)
            kv_cx.append(cx)
            ax.add_patch(FancyBboxPatch((cx - kvw / 2, ky), kvw, 0.7,
                         boxstyle="round,pad=0.02,rounding_size=0.10",
                         fc=kv_cols[j], ec=C["grey"], lw=1.3, alpha=0.85))
            ax.text(cx, ky + 0.35, f"KV{j+1}", ha="center", va="center",
                    fontsize=9.5, color="white")
        # query heads (top)
        for i in range(n_q):
            cx = 0.5 + 7.4 / n_q * (i + 0.5)
            j = kv_of_query[i]
            ax.add_patch(FancyBboxPatch((cx - qw / 2, qy), qw, 0.7,
                         boxstyle="round,pad=0.02,rounding_size=0.10",
                         fc=kv_cols[j], ec=C["grey"], lw=1.1, alpha=0.85))
            ax.text(cx, qy + 0.35, f"Q{i+1}", ha="center", va="center",
                    fontsize=8.5, color="white")
            ax.add_patch(FancyArrowPatch((cx, qy), (kv_cx[j], ky + 0.7),
                         arrowstyle="-", color=C["grey"], lw=0.9, alpha=0.7))
        ax.text(4.2, 4.05, title, ha="center", fontsize=13,
                fontweight="bold", color=C["text"])
        ax.text(4.2, 3.68, subtitle, ha="center", fontsize=9.5,
                style="italic", color=C["grey"])
        ax.text(0.2, qy + 0.35, "Q", ha="center", va="center", fontsize=10,
                color=C["text"], fontweight="bold")
        ax.text(0.2, ky + 0.35, "KV", ha="center", va="center", fontsize=10,
                color=C["text"], fontweight="bold")

    # MHA: 8 query heads, 8 KV heads (one each)
    draw(axes[0], "MHA  ($g=h$)", "8 query heads, 8 KV heads",
         kv_of_query=[0, 1, 2, 3, 4, 5, 6, 7], n_kv=8)
    # GQA: 8 query heads, 2 KV groups of 4
    draw(axes[1], "GQA  ($1<g<h$)", "8 query heads, 2 KV heads (groups of 4)",
         kv_of_query=[0, 0, 0, 0, 1, 1, 1, 1], n_kv=2)
    # MQA: 8 query heads, 1 KV head
    draw(axes[2], "MQA  ($g=1$)", "8 query heads, 1 shared KV head",
         kv_of_query=[0, 0, 0, 0, 0, 0, 0, 0], n_kv=1)

    fig.suptitle("Fewer KV heads $\\Rightarrow$ a smaller KV cache to stream "
                 "each decode step (MHA $\\to$ GQA $\\to$ MQA)",
                 fontsize=13, fontweight="bold", color=C["text"])
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    save(fig, "attn_mha_mqa_gqa.jpg")


# ---------------------------------------------------------------------------
# 5. Ainslie et al. (2023) Figure 3: quality vs speed. Recreated from the exact
#    coordinates in the arXiv source (figures/results.tex). Uptrained MQA/GQA
#    on T5-XXL sit far to the left (fast) at near-MHA-XXL quality.
# ---------------------------------------------------------------------------
def fig_gqa_perf_vs_time():
    # (time per sample, performance) straight from results.tex
    pts = {
        "MHA-Large": (0.372, 45.95,  C["grey"]),
        "MHA-XXL":   (1.514, 47.206, C["grey"]),
        "MQA-XXL":   (0.239, 46.571, C["orange"]),
        "GQA-XXL":   (0.275, 47.136, C["blue"]),
    }
    fig, ax = plt.subplots(figsize=(7.2, 4.6))
    for name, (t, p, col) in pts.items():
        ax.scatter([t], [p], s=130, color=col, edgecolor=C["grey"],
                   linewidth=1.0, zorder=3)
    # annotations positioned to avoid overlap
    ax.annotate("MHA-Large", (0.372, 45.95), xytext=(8, -4),
                textcoords="offset points", fontsize=10, color=C["text"])
    ax.annotate("MHA-XXL", (1.514, 47.206), xytext=(-8, -14),
                textcoords="offset points", fontsize=10, color=C["text"],
                ha="right")
    ax.annotate("MQA-XXL", (0.239, 46.571), xytext=(10, -3),
                textcoords="offset points", fontsize=10, color=C["text"])
    ax.annotate("GQA-XXL", (0.275, 47.136), xytext=(10, 2),
                textcoords="offset points", fontsize=10, color=C["text"])
    ax.set_xlim(0.0, 1.95)
    ax.set_ylim(45.65, 47.40)
    ax.set_xlabel("Time per sample (relative)", fontsize=11)
    ax.set_ylabel("Average performance", fontsize=11)
    ax.set_title("5%-uptrained MQA/GQA-8 keep almost all MHA-XXL quality\n"
                 "at a fraction of MHA-XXL's decoding time",
                 fontsize=12, fontweight="bold", color=C["text"])
    ax.grid(True, ls="--", color="#dddddd", lw=0.8)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    save(fig, "attn_gqa_perf_vs_time.jpg")


# ---------------------------------------------------------------------------
# 6. Ainslie et al. (2023) Figure 5: performance vs uptraining proportion for
#    T5-XXL. Data from figures/uptraining_steps.tex. GQA-8 recovers quality with
#    far less uptraining than MQA, and lands right on the MHA reference line.
# ---------------------------------------------------------------------------
def fig_gqa_uptraining():
    prop = [0.0, 0.05, 0.1]
    gqa = [56.713, 57.4, 57.56]
    mqa = [53.93, 56.92, 57.153]
    mha_ref = 57.537
    fig, ax = plt.subplots(figsize=(7.2, 4.3))
    ax.axhline(mha_ref, color=C["grey"], ls=":", lw=2.4,
               label="MHA (reference)")
    ax.plot(prop, gqa, "s-", color=C["blue"], lw=2.2, ms=7, label="GQA-8")
    ax.plot(prop, mqa, "^-", color=C["orange"], lw=2.2, ms=8, label="MQA")
    ax.set_xlabel("Uptraining proportion (fraction of pre-training)", fontsize=11)
    ax.set_ylabel("Performance", fontsize=11)
    ax.set_title("GQA-8 needs far less uptraining than MQA to recover\n"
                 "multi-head quality from an MHA checkpoint",
                 fontsize=12, fontweight="bold", color=C["text"])
    ax.grid(True, ls="--", color="#dddddd", lw=0.8)
    ax.legend(fontsize=10, frameon=False, loc="lower right")
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    save(fig, "attn_gqa_uptraining.jpg")


# ---------------------------------------------------------------------------
# 7. Ainslie et al. (2023) Figure 6: decode time per sample vs number of GQA
#    groups (log x), input 2048 / output 512. Data from time_vs_groups.tex.
#    1 (MQA) -> 8 groups is nearly flat; past 8 the cost climbs toward MHA.
# ---------------------------------------------------------------------------
def fig_gqa_time_vs_groups():
    groups = [1, 4, 8, 16, 32, 64]
    gqa_t = [0.489, 0.476, 0.514, 0.594, 0.800, 2.531]
    mqa_t = 0.489   # 1 group
    mha_t = 2.531   # 64 groups == full multi-head
    fig, ax = plt.subplots(figsize=(7.6, 4.3))
    # shade the "cheap" region 1..8 groups
    ax.axvspan(1, 8, color=C["green"], alpha=0.10)
    ax.axhline(mha_t, color=C["grey"], ls=":", lw=2.4, label="MHA")
    ax.axhline(mqa_t, color=C["orange"], ls=":", lw=2.0, label="MQA")
    ax.plot(groups, gqa_t, "s-", color=C["blue"], lw=2.2, ms=7, label="GQA")
    ax.set_xscale("log", base=2)
    ax.set_xticks(groups)
    ax.set_xticklabels([str(g) for g in groups])
    ax.set_xlim(0.9, 70)
    ax.set_ylim(0, 2.8)
    ax.set_xlabel("Number of GQA groups $g$ (log scale)", fontsize=11)
    ax.set_ylabel("Time per sample (s)", fontsize=11)
    ax.set_title("1 (MQA) → 8 groups is nearly free; beyond 8 the cost\n"
                 "climbs back toward full multi-head (64 groups)",
                 fontsize=12, fontweight="bold", color=C["text"])
    ax.text(2.6, 0.72, "cheap zone\n(1–8 groups)", fontsize=9,
            style="italic", color=C["green"], ha="center")
    ax.grid(True, ls="--", color="#dddddd", lw=0.8)
    ax.legend(fontsize=10, frameon=False, loc="upper left")
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    save(fig, "attn_gqa_time_vs_groups.jpg")


# ---------------------------------------------------------------------------
# 8. Schematic (illustrative, not measured): without QK-norm, attention logits
#    grow unbounded during training and softmax collapses to one-hot (attention
#    entropy -> 0), triggering loss spikes; QK-norm bounds the logits so entropy
#    stays healthy. Curves are stylized to convey the mechanism reported by
#    Dehghani et al. (ViT-22B) and Wortsman et al.
# ---------------------------------------------------------------------------
def fig_qknorm_stability():
    steps = np.linspace(0, 1, 200)
    # left: max attention logit magnitude (log scale)
    logit_no = 8.0 * 10 ** (3.8 * steps)          # blows up toward ~50k
    logit_norm = 9.0 + 1.2 * np.sin(steps * 25)   # bounded, small wiggle
    # right: attention entropy as a fraction of its max (ln n)
    ent_no = 0.05 + 0.85 / (1 + np.exp(16 * (steps - 0.62)))
    ent_norm = 0.66 + 0.02 * np.sin(steps * 20)

    fig, (axL, axR) = plt.subplots(1, 2, figsize=(12.5, 4.5))

    axL.semilogy(steps, logit_no, color=C["orange"], lw=2.4,
                 label="no QK-norm")
    axL.semilogy(steps, logit_norm, color=C["blue"], lw=2.4,
                 label="with QK-norm")
    axL.axhline(5e4, color=C["grey"], ls="--", lw=1.2)
    axL.text(0.02, 6.5e4, "one-hot softmax / fp16 overflow risk",
             fontsize=8.5, color=C["grey"])
    # divergence marker where the unnormalised curve blows up
    axL.scatter([0.9], [8.0 * 10 ** (3.8 * 0.9)], marker="x", s=120,
                color="#c0392b", lw=2.5, zorder=5)
    axL.annotate("loss spike /\ndivergence", (0.9, 8.0 * 10 ** (3.8 * 0.9)),
                 xytext=(-70, -6), textcoords="offset points", fontsize=9,
                 color="#c0392b", ha="right")
    axL.set_ylim(1, 2e5)
    axL.set_xlabel("training progress", fontsize=11)
    axL.set_ylabel("max attention logit  $|q\\cdot k|$  (log)", fontsize=11)
    axL.set_title("Logits grow without bound...", fontsize=12,
                  fontweight="bold", color=C["text"])
    axL.legend(fontsize=10, frameon=False, loc="center right")
    axL.spines[["top", "right"]].set_visible(False)

    axR.plot(steps, ent_no, color=C["orange"], lw=2.4, label="no QK-norm")
    axR.plot(steps, ent_norm, color=C["blue"], lw=2.4, label="with QK-norm")
    axR.set_ylim(0, 1.0)
    axR.set_xlabel("training progress", fontsize=11)
    axR.set_ylabel("attention entropy (fraction of max)", fontsize=11)
    axR.set_title("...so softmax collapses to one-hot\n(attention entropy → 0)",
                  fontsize=12, fontweight="bold", color=C["text"])
    axR.annotate("entropy collapse", (0.72, 0.12), xytext=(0.30, 0.30),
                 fontsize=9.5, color=C["orange"],
                 arrowprops=dict(arrowstyle="-|>", color=C["orange"], lw=1.4))
    axR.legend(fontsize=10, frameon=False, loc="center left")
    axR.spines[["top", "right"]].set_visible(False)

    fig.suptitle("Why QK-norm helps: bounding the logits keeps softmax "
                 "responsive instead of saturating into instability",
                 fontsize=13, fontweight="bold", color=C["text"])
    fig.tight_layout(rect=[0, 0, 1, 0.92])
    save(fig, "attn_qknorm_stability.jpg")


if __name__ == "__main__":
    fig_matmul_flow()
    fig_sqrt_dk()
    fig_multihead()
    fig_mha_mqa_gqa()
    fig_gqa_perf_vs_time()
    fig_gqa_uptraining()
    fig_gqa_time_vs_groups()
    fig_qknorm_stability()
    print("done.")
