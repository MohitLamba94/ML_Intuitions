"""Generate figures for kimi_k3.md (Kimi K3: Open Frontier Intelligence, arXiv:2607.24653).

Outputs (all .jpg, per repo convention), prefix `kimik3_`:
  kimik3_architecture.jpg        -- the block: 3x KDA + 1x Gated MLA, each paired with Stable LatentMoE, AttnRes across blocks, vision pathway
  kimik3_lower_bounded_decay.jpg -- Kimi Linear negative-softplus vs Kimi K3 scaled-sigmoid log-decay + diagonal-tile consequence
  kimik3_situ_glu.jpg            -- scalar responses of GLU vs SwiGLU (unbounded) vs SiTU-GLU (soft-capped, bound b1*b2=100)
  kimik3_quantile_balancing.jpg  -- imbalanced Top-k routing -> QB bias shift -> balanced load (m=8,n=4,k=1)
  kimik3_attn_residuals.jpg      -- standard residual (single-state bottleneck) vs AttnRes attention-over-depth (block form)
  kimik3_kda_recurrence.jpg      -- delta-rule state update with channel-wise decay; chunkwise parallel form
  kimik3_scaling_law.jpg         -- fitted loss-vs-FLOPs curves, K3 2.5x more efficient than K2
  kimik3_kcp_prefix_scan.jpg     -- KDA Context Parallelism: local state + cumulative transition, composed by prefix scan

All schematic / illustrative (synthetic data), matplotlib only. Numbers in labels
(gmin=-5, beta1=4, beta2=25, bound 100, 896/16 experts, 2.5x) are from the paper.

Run with the repo 'general' conda env:
  /Users/mohitl/Documents/miniconda3/envs/general/bin/python scripts/make_kimi_k3_figs.py
"""
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Circle, Rectangle

ASSETS = os.path.join(os.path.dirname(__file__), "..", "assets")

# Brand-neutral palette shared with the other repo figure scripts.
C = {
    "blue":   "#6C8EBF",
    "purple": "#9673A6",
    "green":  "#82B366",
    "orange": "#D6795B",
    "red":    "#C0504D",
    "grey":   "#555555",
    "lgrey":  "#EDEDED",
    "mgrey":  "#BBBBBB",
    "yellow": "#E8B84B",
    "teal":   "#4E9EA6",
    "text":   "#222222",
}
DPI = 150
np.random.seed(3)


def save(fig, name):
    path = os.path.join(ASSETS, name)
    fig.savefig(path, dpi=DPI, bbox_inches="tight", facecolor="white",
                pil_kwargs={"quality": 90})
    plt.close(fig)
    print("wrote", os.path.normpath(path))


def rbox(ax, x, y, w, h, label, fc, ec="none", txt="#222222", fs=10, lw=1.2, weight="normal"):
    """Rounded box centred at (x,y)."""
    p = FancyBboxPatch((x - w / 2, y - h / 2), w, h,
                       boxstyle="round,pad=0.02,rounding_size=0.08",
                       fc=fc, ec=ec, lw=lw, zorder=3)
    ax.add_patch(p)
    if label:
        ax.text(x, y, label, ha="center", va="center", color=txt,
                fontsize=fs, zorder=4, weight=weight)


def arrow(ax, x0, y0, x1, y1, color="#555555", lw=1.6, style="-|>", ls="-", rad=0.0, alpha=1.0):
    a = FancyArrowPatch((x0, y0), (x1, y1), arrowstyle=style,
                        mutation_scale=13, lw=lw, color=color, zorder=2,
                        linestyle=ls, alpha=alpha,
                        connectionstyle=f"arc3,rad={rad}")
    ax.add_patch(a)


# ---------------------------------------------------------------------------
# 1. Architecture overview
# ---------------------------------------------------------------------------
def fig_architecture():
    fig, ax = plt.subplots(figsize=(12.8, 8.2))
    ax.set_xlim(0, 12.8)
    ax.set_ylim(0, 8.4)
    ax.axis("off")
    ax.set_title("Kimi K3: one block = 3 KDA layers + 1 Gated MLA, each paired with a Stable LatentMoE FFN;\n"
                 "Attention Residuals let every layer attend back over previous block outputs",
                 fontsize=11.5, color=C["text"], weight="bold", pad=8)

    # ----- left: the token/channel stack (one expanded block) -----
    xcol = 3.7      # spine column (embedding, attn layers, final MLA, output)
    xmoe = 6.05     # LatentMoE column
    y0 = 0.95
    rbox(ax, xcol, y0, 2.5, 0.55, "Embedding", C["lgrey"], ec=C["grey"], fs=10)

    layers = [
        ("KDA  (linear attn)", C["blue"]),
        ("KDA  (linear attn)", C["blue"]),
        ("KDA  (linear attn)", C["blue"]),
        ("Gated MLA  (global, NoPE)", C["purple"]),
    ]
    ys = []
    y = y0 + 1.05
    for name, col in layers:
        rbox(ax, xcol, y, 2.5, 0.5, name, col, txt="white", fs=8.5)
        rbox(ax, xmoe, y, 1.45, 0.5, "LatentMoE", C["green"], txt="white", fs=8.5)
        ys.append(y)
        y += 1.1

    # vertical spine + within-layer attn->moe
    arrow(ax, xcol, y0 + 0.28, xcol, ys[0] - 0.25, color=C["grey"], lw=1.4)
    for i in range(len(ys) - 1):
        arrow(ax, xcol, ys[i] + 0.25, xcol, ys[i + 1] - 0.25, color=C["grey"], lw=1.3)
    for yy in ys:
        arrow(ax, xcol + 1.28, yy, xmoe - 0.75, yy, color=C["grey"], lw=1.1)

    ytop = ys[-1] + 1.1
    rbox(ax, xcol, ytop, 2.5, 0.5, "Gated MLA (final)", C["purple"], txt="white", fs=8.5)
    arrow(ax, xcol, ys[-1] + 0.25, xcol, ytop - 0.25, color=C["grey"], lw=1.3)
    rbox(ax, xcol, ytop + 0.9, 2.5, 0.5, "Output / next token", C["lgrey"], ec=C["grey"], fs=9)
    arrow(ax, xcol, ytop + 0.25, xcol, ytop + 0.9 - 0.25, color=C["grey"], lw=1.3)

    ax.text(xcol, 0.35, "93 layers total = 69 KDA + 24 MLA  (pattern: 3 KDA : 1 MLA)",
            ha="center", va="center", fontsize=8.2, style="italic", color=C["grey"])
    ax.text(xmoe, y0 + 0.02, "each LatentMoE:\n896 experts,\n16 active, 2 shared",
            ha="center", va="center", fontsize=7.6, color=C["green"])

    # ----- vision pathway feeding embedding (far left) -----
    vx = 1.0
    rbox(ax, vx, 0.5, 1.5, 0.45, "image / video", "white", ec=C["mgrey"], fs=8)
    rbox(ax, vx, 1.2, 1.5, 0.45, "MoonViT-V2", C["teal"], txt="white", fs=8)
    rbox(ax, vx, 1.9, 1.5, 0.45, "MLP projector", C["teal"], txt="white", fs=7.4)
    arrow(ax, vx, 0.73, vx, 0.98, color=C["grey"], lw=1.2)
    arrow(ax, vx, 1.43, vx, 1.68, color=C["grey"], lw=1.2)
    arrow(ax, vx + 0.5, 1.9, xcol - 1.28, y0, color=C["teal"], lw=1.4, rad=-0.2)

    # ----- right: AttnRes across blocks (attention over depth) -----
    ax.plot([7.35, 7.35], [0.5, 7.6], color=C["lgrey"], lw=1.0)
    bx = 9.7
    ax.text(bx, 7.5, "Attention Residuals (depth mixing)", ha="center", fontsize=10,
            weight="bold", color=C["orange"])
    block_ys = [1.2, 2.5, 3.8, 5.1]
    labels = ["embedding b0", "block 1  b1", "block 2  b2", "block 3  b3"]
    for by, lab in zip(block_ys, labels):
        rbox(ax, bx - 1.15, by, 1.9, 0.5, lab, C["lgrey"], ec=C["grey"], fs=8.5)
    # current layer pseudo-query
    qy = 6.6
    rbox(ax, bx - 1.15, qy, 1.9, 0.5, "layer l:  q = w_l", C["orange"], txt="white", fs=8.5)
    for by in block_ys:
        arrow(ax, bx - 0.2, by, bx - 0.2, qy, color=C["orange"], lw=1.1, rad=-0.3, alpha=0.85)
    ax.text(bx + 1.15, 3.9,
            "softmax over\nRMSNorm(keys)\n= a_{i->l}\n\nh_l =\nsum_i a_{i->l} v_i",
            ha="center", va="center", fontsize=8.2, color=C["text"])
    ax.text(bx - 0.4, 0.4, "standard residual keeps ONE running state;\nAttnRes keeps all block outputs and attends over them",
            ha="center", va="top", fontsize=7.8, style="italic", color=C["grey"])

    save(fig, "kimik3_architecture.jpg")


# ---------------------------------------------------------------------------
# 2. Lower-bounded decay
# ---------------------------------------------------------------------------
def fig_lower_bounded_decay():
    fig, axes = plt.subplots(1, 2, figsize=(12.2, 4.7))

    # (a) log-decay parameterization g vs z
    ax = axes[0]
    z = np.linspace(-6, 6, 400)
    softplus = np.log1p(np.exp(z))
    g_kl = -softplus                      # Kimi Linear, A=0: g=-softplus(z) in (-inf,0)
    gmin = -5.0
    g_k3 = gmin / (1 + np.exp(-z))         # Kimi K3: g = gmin * sigmoid(z) in (gmin,0)
    ax.plot(z, g_kl, color=C["blue"], lw=2.4, label="Kimi Linear:  g = -softplus(z)   (unbounded)")
    ax.plot(z, g_k3, color=C["red"], lw=2.4, label="Kimi K3:  g = gmin . sigmoid(z)   (bounded)")
    ax.axhline(gmin, color=C["red"], ls="--", lw=1.1, alpha=0.7)
    ax.text(5.6, gmin + 0.25, "gmin = -5", color=C["red"], fontsize=9, ha="right")
    ax.axhline(0, color=C["grey"], lw=0.8)
    ax.text(-5.8, -7.5, "as z grows negative,\nKimi Linear's g -> -inf\n(alpha -> 0, reciprocal\n1/decay overflows)",
            fontsize=8.2, color=C["blue"], va="top")
    ax.set_xlabel("decay logit  z", fontsize=10)
    ax.set_ylabel("per-step log-decay  g   (alpha = exp(g))", fontsize=10)
    ax.set_title("(a) Bounding the log-decay from below", fontsize=10.5, weight="bold")
    ax.set_ylim(-9.5, 0.6)
    ax.legend(fontsize=8.0, loc="lower right", framealpha=0.95)
    ax.grid(alpha=0.25)

    # (b) diagonal-tile computation consequence
    ax = axes[1]
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 10)
    ax.axis("off")
    ax.set_title("(b) Consequence for chunkwise tiles", fontsize=10.5, weight="bold")
    n = 4
    cell = 1.7
    x0, y0 = 1.0, 1.2
    for i in range(n):      # row (query tile), top-down
        for j in range(n):  # col (key tile)
            xx = x0 + j * cell
            yy = y0 + (n - 1 - i) * cell
            if j > i:
                fc, lab = "white", ""            # masked (future)
                ec = C["lgrey"]
            elif j == i:
                fc, lab = C["yellow"], "diag"     # diagonal tile
                ec = C["grey"]
            else:
                fc, lab = C["green"], "TC"        # off-diagonal, Tensor Core
                ec = C["grey"]
            ax.add_patch(Rectangle((xx, yy), cell * 0.92, cell * 0.92, fc=fc, ec=ec, lw=1.2))
            if lab:
                ax.text(xx + cell * 0.46, yy + cell * 0.46, lab, ha="center", va="center",
                        fontsize=8.5, color=C["text"])
    ax.text(5.0, 9.4, "Kimi Linear: diagonal tiles need slow position-pair math", fontsize=8.6,
            ha="center", color=C["orange"])
    ax.text(5.0, 0.5, "Kimi K3: bounded range keeps 1/decay inside BF16,\n"
                      "so EVERY causal tile (incl. diagonal) uses dense Tensor-Core matmul",
            fontsize=8.6, ha="center", color=C["green"])

    fig.suptitle("Kimi K3 replaces the unbounded negative-softplus decay with a scaled sigmoid bounded at gmin = -5",
                 fontsize=11.5, weight="bold", y=1.02)
    fig.tight_layout()
    save(fig, "kimik3_lower_bounded_decay.jpg")


# ---------------------------------------------------------------------------
# 3. SiTU-GLU vs SwiGLU vs GLU
# ---------------------------------------------------------------------------
def fig_situ_glu():
    b1, b2 = 4.0, 25.0
    x = np.linspace(-10, 100, 800)
    sig = 1 / (1 + np.exp(-x))

    fig, axes = plt.subplots(1, 3, figsize=(13.2, 4.4))

    # gate branch
    ax = axes[0]
    glu_gate = sig                              # GLU gate: sigmoid(x)
    swi_gate = x * sig                          # SwiGLU gate: Swish = x*sigmoid(x)
    situ_gate = b1 * np.tanh(x / b1) * sig      # SiTU gate: b1 tanh(x/b1) * sigmoid(x)
    ax.plot(x, glu_gate, color=C["mgrey"], lw=2.0, label="GLU:  sigma(x)")
    ax.plot(x, swi_gate, color=C["blue"], lw=2.2, label="SwiGLU:  x . sigma(x)")
    ax.plot(x, situ_gate, color=C["red"], lw=2.4, label="SiTU:  b1 tanh(x/b1) . sigma(x)")
    ax.axhline(b1, color=C["red"], ls="--", lw=1.0, alpha=0.7)
    ax.text(98, b1 + 1.5, "-> b1 = 4", color=C["red"], fontsize=8.5, ha="right")
    ax.set_title("Gate branch", fontsize=10.5, weight="bold")
    ax.set_xlabel("x"); ax.set_ylabel("f(x)")
    ax.legend(fontsize=8.0, loc="upper left"); ax.grid(alpha=0.25)
    ax.set_ylim(-2, 20)

    # up branch
    ax = axes[1]
    up_lin = x                                  # GLU / SwiGLU up branch: linear x
    up_situ = b2 * np.tanh(x / b2)              # SiTU up branch: b2 tanh(x/b2)
    ax.plot(x, up_lin, color=C["blue"], lw=2.2, label="GLU / SwiGLU:  x  (unbounded)")
    ax.plot(x, up_situ, color=C["red"], lw=2.4, label="SiTU:  b2 tanh(x/b2)")
    ax.axhline(b2, color=C["red"], ls="--", lw=1.0, alpha=0.7)
    ax.text(98, b2 + 3, "-> b2 = 25", color=C["red"], fontsize=8.5, ha="right")
    ax.set_title("Up branch", fontsize=10.5, weight="bold")
    ax.set_xlabel("x"); ax.set_ylabel("f(x)")
    ax.legend(fontsize=8.0, loc="upper left"); ax.grid(alpha=0.25)
    ax.set_ylim(-12, 100)

    # product (per-coordinate output)
    ax = axes[2]
    swi_out = swi_gate * up_lin
    situ_out = situ_gate * up_situ
    ax.plot(x, swi_out, color=C["blue"], lw=2.2, label="SwiGLU product (blows up)")
    ax.plot(x, situ_out, color=C["red"], lw=2.4, label="SiTU-GLU product (bounded)")
    ax.axhline(b1 * b2, color=C["red"], ls="--", lw=1.1, alpha=0.8)
    ax.text(98, b1 * b2 + 120, "|f| <= b1.b2 = 100", color=C["red"], fontsize=8.6, ha="right")
    ax.set_title("Product = one output coordinate", fontsize=10.5, weight="bold")
    ax.set_xlabel("x"); ax.set_ylabel("gate . up")
    ax.legend(fontsize=8.0, loc="upper left"); ax.grid(alpha=0.25)
    ax.set_ylim(-50, 900)

    fig.suptitle("SiTU-GLU soft-caps both SwiGLU factors: near-identical to SwiGLU around the origin, "
                 "but the product is bounded by b1.b2 = 100 (no low-precision overflow)",
                 fontsize=11.3, weight="bold", y=1.03)
    fig.tight_layout()
    save(fig, "kimik3_situ_glu.jpg")


# ---------------------------------------------------------------------------
# 4. Quantile Balancing
# ---------------------------------------------------------------------------
def fig_quantile_balancing():
    fig, axes = plt.subplots(1, 3, figsize=(13.2, 5.0))
    m, n = 8, 4
    tok_y = np.linspace(0.5, 7.5, m)[::-1]
    exp_y = np.linspace(1.5, 6.5, n)[::-1]

    # assignments: (a) imbalanced loads (4,3,1,0); (c) balanced (2,2,2,2)
    assign_a = [0, 0, 0, 0, 1, 1, 1, 2]        # token -> expert  (loads 4,3,1,0)
    assign_c = [0, 0, 1, 1, 2, 2, 3, 3]        # balanced loads 2,2,2,2
    ecolors = [C["red"], C["orange"], C["blue"], C["mgrey"]]

    def draw_bipartite(ax, assign, title, loads):
        ax.set_xlim(0, 6); ax.set_ylim(0, 8); ax.axis("off")
        ax.set_title(title, fontsize=10.5, weight="bold")
        for ti, ty in enumerate(tok_y):
            ax.add_patch(Circle((1.0, ty), 0.22, fc="white", ec=C["grey"], lw=1.2, zorder=3))
            ax.text(1.0, ty, f"t{ti+1}", ha="center", va="center", fontsize=7.5, zorder=4)
        for ei, ey in enumerate(exp_y):
            load = loads[ei]
            # heat: darker = overheated, faded = dying
            if load >= 4:
                fc = C["red"]
            elif load == 0:
                fc = "white"
            else:
                fc = C["blue"]
            ax.add_patch(Circle((5.0, ey), 0.30, fc=fc, ec=C["grey"], lw=1.2,
                                zorder=3, alpha=0.35 if load == 0 else 1.0,
                                ls="--" if load == 0 else "-"))
            ax.text(5.0, ey, f"E{ei+1}", ha="center", va="center", fontsize=8,
                    color="white" if load in (4, 3, 2, 1) else C["grey"], zorder=4)
            ax.text(5.55, ey, f"load {load}", ha="left", va="center", fontsize=8, color=C["grey"])
        for ti, e in enumerate(assign):
            ax.plot([1.22, 4.72], [tok_y[ti], exp_y[e]], color=ecolors[e], lw=1.4, alpha=0.8, zorder=1)

    draw_bipartite(axes[0], assign_a, "(a) Top-k routing: imbalanced", [4, 3, 1, 0])

    # (b) margins with per-expert cutoff
    ax = axes[1]
    ax.set_title("(b) QB: shift each expert's bias to\nthe (q+1)-th largest margin (q=2)", fontsize=10, weight="bold")
    ax.set_xlim(0, 5); ax.set_ylim(0, 8); ax.axis("off")
    rng = np.random.default_rng(1)
    for ei, ey in enumerate(exp_y):
        margins = np.sort(rng.uniform(0.2, 3.5, m))[::-1]
        base_x = 0.6
        for k, mg in enumerate(margins):
            col = ecolors[ei] if k < 2 else C["lgrey"]
            ax.add_patch(Rectangle((base_x, ey - 0.32 + k * 0.0), mg, 0.5, fc=C["mgrey"],
                                   ec="white", lw=0.4, alpha=0.0))
        # draw stacked bars horizontally as a small row of margin bars
        for k, mg in enumerate(margins):
            ax.add_patch(Rectangle((base_x + k * 0.42, ey - 0.28), 0.36, 0.55 * (mg / 3.5) + 0.15,
                                   fc=(C["blue"] if k < 2 else C["lgrey"]), ec="white", lw=0.5))
        # cutoff line after the 2nd (q=2)
        ax.axvline(base_x + 2 * 0.42 - 0.03, ymin=(ey - 0.5) / 8, ymax=(ey + 0.6) / 8,
                   color=C["red"], ls="--", lw=1.6)
        ax.text(4.9, ey, f"E{ei+1}", ha="right", va="center", fontsize=8, color=C["grey"])
    ax.text(2.5, 0.3, "keep 2 margins above the red cutoff per expert\n"
                      "=> exactly q = mk/n = 2 tokens each", ha="center", fontsize=8.2, color=C["red"])

    draw_bipartite(axes[2], assign_c, "(c) After QB: balanced (2,2,2,2)", [2, 2, 2, 2])

    fig.suptitle("Quantile Balancing sets each expert's bias to the router-score quantile matching its target load "
                 "(m=8 tokens, n=4 experts, k=1) - one jump, no learning rate",
                 fontsize=11.0, weight="bold", y=1.02)
    fig.tight_layout()
    save(fig, "kimik3_quantile_balancing.jpg")


# ---------------------------------------------------------------------------
# 5. Attention Residuals
# ---------------------------------------------------------------------------
def fig_attn_residuals():
    fig, axes = plt.subplots(1, 2, figsize=(12.6, 5.2))

    # (a) standard residual: single accumulating state
    ax = axes[0]
    ax.set_xlim(0, 10); ax.set_ylim(0, 8); ax.axis("off")
    ax.set_title("(a) Standard residual: one running state\n(a depth-wise RNN bottleneck)",
                 fontsize=10.5, weight="bold")
    xs = 5.0
    ys = np.linspace(0.8, 7.0, 6)
    labels = ["h1 (embed)", "h2", "h3", "h4", "h5", "hL"]
    for i, (yy, lab) in enumerate(zip(ys, labels)):
        rbox(ax, xs, yy, 2.4, 0.55, f"layer {i}:  {lab}", C["lgrey"], ec=C["grey"], fs=9)
        if i > 0:
            arrow(ax, xs, ys[i - 1] + 0.30, xs, yy - 0.30, color=C["grey"], lw=1.8)
    ax.text(xs + 1.5, 4.0, "h_l = h_{l-1} + f_l(h_{l-1})\n\nall past compressed\ninto ONE vector",
            ha="left", va="center", fontsize=8.6, color=C["grey"])

    # (b) AttnRes: attention over all previous layer/block outputs
    ax = axes[1]
    ax.set_xlim(0, 10); ax.set_ylim(0, 8); ax.axis("off")
    ax.set_title("(b) Attention Residuals: attend over depth\n(block form: N ~ 8 block summaries)",
                 fontsize=10.5, weight="bold")
    src_y = np.linspace(0.9, 5.2, 4)
    src_lab = ["b0 (embed)", "b1", "b2", "b3"]
    for yy, lab in zip(src_y, src_lab):
        rbox(ax, 2.0, yy, 1.7, 0.5, lab, C["lgrey"], ec=C["grey"], fs=8.5)
    rbox(ax, 6.4, 6.7, 2.9, 0.55, "layer l:  q = w_l", C["orange"], txt="white", fs=8.5)
    weights = [0.12, 0.5, 0.28, 0.1]
    for yy, w in zip(src_y, weights):
        arrow(ax, 2.85, yy, 5.4, 6.5, color=C["orange"], lw=0.8 + 5 * w, rad=-0.18, alpha=0.85)
    rbox(ax, 6.4, 4.3, 2.9, 0.55, "h_l = sum a_{i->l} v_i", C["green"], txt="white", fs=8.5)
    arrow(ax, 6.4, 6.4, 6.4, 4.6, color=C["grey"], lw=1.6)
    ax.text(4.2, 2.4, "softmax weights a_{i->l}\nover RMSNorm(keys):\nthick arrow = big weight",
            ha="center", va="center", fontsize=8.3, color=C["orange"])

    fig.suptitle("Attention Residuals do to DEPTH what attention did to sequence: "
                 "each layer selectively retrieves earlier representations instead of one accumulated state",
                 fontsize=11.0, weight="bold", y=1.02)
    fig.tight_layout()
    save(fig, "kimik3_attn_residuals.jpg")


# ---------------------------------------------------------------------------
# 6. KDA delta-rule recurrence + chunkwise
# ---------------------------------------------------------------------------
def fig_kda_recurrence():
    fig, axes = plt.subplots(1, 2, figsize=(12.8, 5.0))

    # (a) the per-step update pipeline
    ax = axes[0]
    ax.set_xlim(0, 10); ax.set_ylim(0, 8); ax.axis("off")
    ax.set_title("(a) One KDA step on the state matrix S (d_k x d_v)", fontsize=10.5, weight="bold")
    rbox(ax, 1.4, 4, 1.8, 0.9, "S_{t-1}", C["lgrey"], ec=C["grey"], fs=11)
    rbox(ax, 3.7, 4, 1.6, 0.9, "Diag(alpha_t)\ndecay", C["blue"], txt="white", fs=8.5)
    rbox(ax, 6.0, 4, 1.7, 0.9, "(I - beta_t k k^T)\nerase", C["purple"], txt="white", fs=8.0)
    rbox(ax, 8.4, 4, 1.7, 0.9, "+ beta_t k v^T\nwrite", C["green"], txt="white", fs=8.5)
    arrow(ax, 2.3, 4, 2.9, 4, color=C["grey"])
    arrow(ax, 4.5, 4, 5.15, 4, color=C["grey"])
    arrow(ax, 6.85, 4, 7.55, 4, color=C["grey"])
    rbox(ax, 8.4, 1.7, 1.7, 0.8, "S_t", C["orange"], txt="white", fs=11)
    arrow(ax, 8.4, 3.55, 8.4, 2.1, color=C["grey"])
    rbox(ax, 5.0, 6.4, 3.0, 0.7, "output  o_t = S_t^T q_t", C["teal"], txt="white", fs=9.5)
    arrow(ax, 8.4, 2.1, 6.5, 6.05, color=C["teal"], lw=1.3, rad=0.2, style="-|>")
    ax.text(5.0, 0.7, "channel-wise forget gate alpha_t in (e^-5, 1)  +  delta-rule write beta_t in (0,1)",
            ha="center", fontsize=8.4, color=C["grey"], style="italic")

    # (b) chunkwise: intra-chunk parallel + inter-chunk recurrent
    ax = axes[1]
    ax.set_xlim(0, 10); ax.set_ylim(0, 8); ax.axis("off")
    ax.set_title("(b) Chunkwise form: parallel inside a chunk,\nrecurrent across chunks", fontsize=10.5, weight="bold")
    cx = [1.7, 4.3, 6.9]
    for i, x in enumerate(cx):
        rbox(ax, x, 5.6, 1.9, 1.5, f"chunk {i+1}\n(intra: parallel\nmatmuls on\nTensor Cores)", C["blue"],
             txt="white", fs=7.6)
        rbox(ax, x, 2.6, 1.6, 0.7, f"state S[{i+1}]", C["orange"], txt="white", fs=8)
        arrow(ax, x, 4.85, x, 2.95, color=C["grey"], lw=1.3)
    for i in range(len(cx) - 1):
        arrow(ax, cx[i] + 0.8, 2.6, cx[i + 1] - 0.8, 2.6, color=C["red"], lw=1.8)
    ax.text(8.9, 5.6, "O = (decay . Q) S   [inter-chunk]\n        + A V~        [intra-chunk]",
            ha="center", va="center", fontsize=7.8, rotation=90, color=C["text"])
    ax.text(4.3, 1.4, "serial state carry (cheap, fixed size)  <-  the recurrence",
            ha="center", fontsize=8.2, color=C["red"])

    fig.suptitle("KDA = delta-rule fast-weight memory + channel-wise decay, computed chunkwise "
                 "(parallel within a chunk, a fixed-size recurrent state across chunks)",
                 fontsize=11.0, weight="bold", y=1.03)
    fig.tight_layout()
    save(fig, "kimik3_kda_recurrence.jpg")


# ---------------------------------------------------------------------------
# 7. Scaling law 2.5x
# ---------------------------------------------------------------------------
def fig_scaling_law():
    fig, ax = plt.subplots(figsize=(8.2, 5.2))
    F = np.logspace(20, 24, 400)
    # illustrative power-law loss L(C) = a*C^-b + c; K3 shifts the curve LEFT by 2.5x
    # in compute, i.e. L_K3(C) = L_K2(2.5*C), so K3 is strictly below K2.
    def loss(C, shift=1.0):
        return 1.6 * ((shift * C) / 1e20) ** (-0.055) + 1.55
    Lk2 = loss(F, shift=1.0)
    Lk3 = loss(F, shift=2.5)
    ax.plot(F, Lk2, color=C["blue"], lw=2.6, label="Kimi K2")
    ax.plot(F, Lk3, color=C["red"], lw=2.6, label="Kimi K3 (2.5x more efficient)")
    ax.set_xscale("log")
    ax.set_xlabel("Training compute  (FLOPs, log scale)", fontsize=10.5)
    ax.set_ylabel("Validation loss (held-out OOD)", fontsize=10.5)
    ax.set_title("Kimi K3 reaches the same loss as Kimi K2 with ~2.5x less compute",
                 fontsize=11.0, weight="bold")
    # horizontal 2.5x arrow at a fixed loss inside the plotted range
    ytar = 2.80
    xk2 = F[np.argmin(np.abs(Lk2 - ytar))]
    xk3 = F[np.argmin(np.abs(Lk3 - ytar))]
    ax.axhline(ytar, color=C["grey"], ls=":", lw=1.0)
    ax.plot([xk2], [ytar], "o", color=C["blue"], ms=6)
    ax.plot([xk3], [ytar], "o", color=C["red"], ms=6)
    ax.annotate("", xy=(xk2, ytar), xytext=(xk3, ytar),
                arrowprops=dict(arrowstyle="<|-|>", color=C["green"], lw=2.4))
    ax.text(np.sqrt(xk2 * xk3), ytar + 0.05, "2.5x less\ncompute", color=C["green"], fontsize=11,
            ha="center", va="bottom", weight="bold")
    ax.legend(fontsize=10, loc="upper right")
    ax.grid(alpha=0.25, which="both")
    fig.tight_layout()
    save(fig, "kimik3_scaling_law.jpg")


# ---------------------------------------------------------------------------
# 8. KDA Context Parallelism prefix scan
# ---------------------------------------------------------------------------
def fig_kcp_prefix_scan():
    fig, ax = plt.subplots(figsize=(12.6, 5.0))
    ax.set_xlim(0, 12.6); ax.set_ylim(0, 5.0)
    ax.axis("off")
    ax.set_title("KDA Context Parallelism: each rank holds a sequence segment; the incoming state is recovered "
                 "by an associative prefix scan over fixed-size fragments",
                 fontsize=11.0, weight="bold", pad=6)
    P = 4
    xs = np.linspace(1.6, 11.0, P)
    for i, x in enumerate(xs):
        rbox(ax, x, 3.7, 2.2, 0.9, f"rank {i}\nlocal tokens", C["blue"], txt="white", fs=8.5)
        rbox(ax, x, 2.15, 2.2, 0.7, "compute locally:\n M[i] , S~[i]", C["green"], txt="white", fs=7.6)
        arrow(ax, x, 3.25, x, 2.5, color=C["grey"], lw=1.3)
    # all-gather bar
    ax.add_patch(Rectangle((0.7, 1.15), 11.2, 0.42, fc=C["lgrey"], ec=C["grey"], lw=1.0))
    ax.text(6.3, 1.36, "one all-gather of the fixed-size fragments  (M[i], S~[i])",
            ha="center", va="center", fontsize=8.6, color=C["text"])
    for x in xs:
        arrow(ax, x, 2.0, x, 1.57, color=C["grey"], lw=1.1)
    # prefix scan recurrence at bottom
    for i in range(P - 1):
        arrow(ax, xs[i] + 0.4, 0.7, xs[i + 1] - 0.4, 0.7, color=C["red"], lw=1.8)
    for i, x in enumerate(xs):
        ax.add_patch(Circle((x, 0.7), 0.12, fc=C["orange"], ec=C["grey"], zorder=4))
    ax.text(6.3, 0.28, "prefix scan:   S[i+1] = M[i+1] . S[i] + S~[i+1]   (compose left to right)",
            ha="center", fontsize=8.8, color=C["red"])
    ax.text(0.75, 4.45, "each fragment (M[i], S~[i]) has fixed size  S in R^{d_k x d_v}  "
                        "-- independent of sequence length, so communication does not grow with context",
            ha="left", va="center", fontsize=7.9, color=C["grey"])
    save(fig, "kimik3_kcp_prefix_scan.jpg")


# ---------------------------------------------------------------------------
# 9. Fine-grained hybrid KDA-MLA prefix cache
# ---------------------------------------------------------------------------
def fig_prefix_cache():
    fig, ax = plt.subplots(figsize=(12.8, 4.4))
    ax.set_xlim(0, 12.8); ax.set_ylim(0, 4.4)
    ax.axis("off")
    ax.set_title("Fine-grained prefix caching for hybrid KDA-MLA: hash on 512-token blocks inside the coarse "
                 "physical block, so a hit can land deep inside it",
                 fontsize=11.0, weight="bold", pad=8)
    nblk = 12
    x0, w = 0.6, 0.98
    y_mla = 2.55
    hit_idx = 5   # B = 2560 = 5 * 512
    # physical block outline
    ax.add_patch(Rectangle((x0 - 0.05, y_mla - 0.05), nblk * w + 0.1, 0.9,
                           fc="none", ec=C["grey"], lw=1.6))
    ax.text(x0 + nblk * w / 2, y_mla + 1.25,
            "one physical cache block = 6144 tokens = 12 hash blocks of 512", ha="center",
            fontsize=8.6, color=C["grey"])
    for i in range(nblk):
        xx = x0 + i * w
        filled = i < hit_idx          # cached MLA blocks up to the hit
        ax.add_patch(Rectangle((xx, y_mla), w * 0.9, 0.8,
                               fc=(C["blue"] if filled else C["lgrey"]),
                               ec="white", lw=1.0))
        ax.text(xx + w * 0.45, y_mla + 0.4, "MLA" if filled else "", ha="center", va="center",
                fontsize=7.2, color="white")
        # KDA checkpoint markers below each boundary
        cy = y_mla - 0.45
        if i == hit_idx:
            ax.add_patch(Circle((xx, cy), 0.11, fc=C["orange"], ec=C["grey"], zorder=4))
        elif i in (0, 2):
            ax.add_patch(Circle((xx, cy), 0.10, fc=C["grey"], ec=C["grey"], zorder=4))
        else:
            ax.add_patch(Circle((xx, cy), 0.10, fc="white", ec=C["mgrey"], zorder=4))
    ax.text(x0 - 0.35, y_mla + 0.4, "MLA KV", ha="right", va="center", fontsize=8, color=C["blue"])
    ax.text(x0 - 0.35, y_mla - 0.45, "KDA ckpt", ha="right", va="center", fontsize=8, color=C["grey"])
    # hit boundary marker
    bx = x0 + hit_idx * w
    ax.annotate("hit boundary B = 2560 = 5 x 512", xy=(bx, y_mla - 0.45), xytext=(bx + 0.3, 0.75),
                fontsize=8.6, color=C["orange"],
                arrowprops=dict(arrowstyle="-|>", color=C["orange"], lw=1.4))
    ax.text(x0 + nblk * w / 2, 0.3,
            "on a hit: restore the KDA checkpoint at B, copy-on-write the partial MLA block, "
            "resume prefill from B with zero recompute of [0, B)",
            ha="center", fontsize=8.3, color=C["text"])
    # legend for checkpoint markers
    lx = x0 + (nblk - 0.2) * w
    ax.add_patch(Circle((lx, 1.75), 0.10, fc="white", ec=C["mgrey"])); ax.text(lx + 0.2, 1.75, "no ckpt", va="center", fontsize=7.4, color=C["grey"])
    ax.add_patch(Circle((lx, 1.45), 0.10, fc=C["grey"])); ax.text(lx + 0.2, 1.45, "persisted", va="center", fontsize=7.4, color=C["grey"])
    ax.add_patch(Circle((lx, 1.15), 0.10, fc=C["orange"])); ax.text(lx + 0.2, 1.15, "hit", va="center", fontsize=7.4, color=C["grey"])
    save(fig, "kimik3_prefix_cache.jpg")


if __name__ == "__main__":
    fig_architecture()
    fig_lower_bounded_decay()
    fig_situ_glu()
    fig_quantile_balancing()
    fig_attn_residuals()
    fig_kda_recurrence()
    fig_scaling_law()
    fig_kcp_prefix_scan()
    fig_prefix_cache()
    print("done")
