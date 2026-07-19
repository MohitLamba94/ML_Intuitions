"""Generate figures for training_llms/006_EffecientAttention.md.

Outputs (all .jpg, per repo convention):
  assets/eff_attn_landscape.jpg      -- taxonomy of efficient-attention families vs complexity
  assets/eff_attn_linear_assoc.jpg   -- the reassociation trick: (QK^T)V  vs  Q(K^T V)
  assets/eff_attn_sparse_patterns.jpg-- full / window / dilated / Longformer / BigBird masks
  assets/eff_attn_lowrank.jpg        -- Linformer: project K,V along the length axis n -> k
  assets/eff_attn_reformer_lsh.jpg   -- Reformer LSH: bucket similar q/k, attend within bucket

Run with the repo 'general' conda env:
  /Users/mohitl/Documents/miniconda3/envs/general/bin/python scripts/make_eff_attn_figs.py
"""
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

ASSETS = os.path.join(os.path.dirname(__file__), "..", "assets")

# Same brand-neutral palette as make_attention_figs.py / make_norm_figs.py.
C = {
    "blue":  "#6C8EBF",
    "purple":"#9673A6",
    "green": "#82B366",
    "orange":"#D6795B",
    "grey":  "#555555",
    "lgrey": "#EDEDED",
    "yellow":"#E8B84B",
    "red":   "#C0504D",
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
# 1. The landscape / taxonomy of efficient-attention families.
#    A dense O(n^2) card at the top, then one card per escape route, each with
#    its mechanism, complexity, and representative methods.
# ---------------------------------------------------------------------------
def fig_landscape():
    fig, ax = plt.subplots(figsize=(13.0, 7.2))
    ax.set_xlim(0, 13.0)
    ax.set_ylim(0, 7.2)
    ax.axis("off")

    def card(x, y, w, h, title, mech, comp, methods, fc):
        ax.add_patch(FancyBboxPatch((x, y), w, h,
                     boxstyle="round,pad=0.03,rounding_size=0.08",
                     fc=fc, ec=C["grey"], lw=1.4))
        ax.text(x + w / 2, y + h - 0.32, title, ha="center", va="center",
                fontsize=12.5, fontweight="bold", color=C["text"])
        ax.text(x + w / 2, y + h - 0.72, mech, ha="center", va="top",
                fontsize=8.8, style="italic", color=C["grey"])
        ax.text(x + w / 2, y + 0.95, comp, ha="center", va="center",
                fontsize=11.5, fontweight="bold", color=C["red"])
        ax.text(x + w / 2, y + 0.42, methods, ha="center", va="top",
                fontsize=8.2, color=C["text"])

    ax.text(6.5, 6.95, "Escaping the quadratic curse: the $n\\times n$ score "
            "matrix is the enemy — sparsify it, low-rank it, kernelize it away, "
            "or hash around it",
            ha="center", fontsize=13, fontweight="bold", color=C["text"])

    # Dense baseline card (top, wide).
    ax.add_patch(FancyBboxPatch((3.6, 5.35), 5.8, 1.20,
                 boxstyle="round,pad=0.03,rounding_size=0.08",
                 fc=C["lgrey"], ec=C["grey"], lw=1.4))
    ax.text(6.5, 6.28, "Dense (vanilla) attention", ha="center", va="center",
            fontsize=12.5, fontweight="bold", color=C["text"])
    ax.text(6.5, 5.92, "$O(n^2 d)$ compute,  $O(n^2)$ memory", ha="center",
            va="center", fontsize=11.5, fontweight="bold", color=C["red"])
    ax.text(6.5, 5.58, "form the full $n\\times n$ softmax matrix — "
            "Transformer (Vaswani 2017)", ha="center", va="center",
            fontsize=8.6, style="italic", color=C["grey"])
    ax.annotate("", xy=(6.5, 5.30), xytext=(6.5, 5.02),
                arrowprops=dict(arrowstyle="-|>", color=C["grey"], lw=1.6))
    ax.text(6.72, 5.16, "how do we avoid it?", fontsize=9, style="italic",
            color=C["grey"], va="center", ha="left")

    # Five escape-route cards along the bottom.
    y0, cw, ch, gap = 1.95, 2.30, 2.75, 0.18
    x0 = (13.0 - (5 * cw + 4 * gap)) / 2
    xs = [x0 + i * (cw + gap) for i in range(5)]
    card(xs[0], y0, cw, ch, "Sparse",
         "compute only a\nchosen subset of\nentries",
         "$O(n\\sqrt{n})$\nto $O(n)$",
         "Sparse Transf.,\nLongformer,\nBigBird", "#DCE6F1")
    card(xs[1], y0, cw, ch, "Low-rank",
         "the softmax matrix\nis ~low-rank:\nproject $n\\!\\to\\!k$",
         "$O(nk)$",
         "Linformer,\nNystromformer", "#E4DAEC")
    card(xs[2], y0, cw, ch, "Linear / kernel",
         "drop softmax, use\n$\\phi(q)\\!\\cdot\\!\\phi(k)$,\nreassociate",
         "$O(n d^2)$",
         "Transf.-as-RNN,\nPerformer\n(FAVOR+)", "#DBEAD5")
    card(xs[3], y0, cw, ch, "Hashing / cluster",
         "attend only within\nbuckets of similar\ntokens",
         "$O(n\\log n)$",
         "Reformer,\nRouting Transf.", "#FCE4D6")
    card(xs[4], y0, cw, ch, "Memory",
         "look up a large\nexternal store of\npast (k,v) pairs",
         "$O(n\\cdot m)$",
         "Memorizing\nTransformer", "#FDF3D0")

    # FlashAttention footnote box: exact, not approximate.
    ax.add_patch(FancyBboxPatch((1.6, 0.35), 9.8, 1.05,
                 boxstyle="round,pad=0.03,rounding_size=0.08",
                 fc="white", ec=C["blue"], lw=1.6, linestyle="--"))
    ax.text(6.5, 1.13, "Orthogonal escape route: FlashAttention (exact, no "
            "approximation)", ha="center", fontsize=11,
            fontweight="bold", color=C["blue"])
    ax.text(6.5, 0.68, "same math as dense — still $O(n^2)$ compute, but $O(n)$ "
            "memory by never writing the $n\\times n$ matrix to HBM.  "
            "See the companion hardware note.",
            ha="center", fontsize=9.0, color=C["grey"])

    fig.tight_layout()
    save(fig, "eff_attn_landscape.jpg")


# ---------------------------------------------------------------------------
# 2. The reassociation trick (centerpiece).
#    Top row: (Q K^T) V  -- builds the n x n matrix, O(n^2 d).
#    Bottom row: phi(Q) (phi(K)^T V) -- builds a d x d matrix, O(n d^2).
# ---------------------------------------------------------------------------
def fig_linear_assoc():
    fig, ax = plt.subplots(figsize=(13.0, 6.6))
    ax.set_xlim(0, 13.0)
    ax.set_ylim(0, 6.6)
    ax.axis("off")

    def block(x, y, w, h, label, shape, fc, txt_col=None, big=False):
        ax.add_patch(FancyBboxPatch((x, y), w, h,
                     boxstyle="round,pad=0.02,rounding_size=0.06",
                     fc=fc, ec=C["grey"], lw=1.3))
        ax.text(x + w / 2, y + h / 2 + 0.18, label, ha="center", va="center",
                fontsize=12, fontweight="bold", color=txt_col or C["text"])
        ax.text(x + w / 2, y + h / 2 - 0.30, shape, ha="center", va="center",
                fontsize=9.5, color=txt_col or C["grey"])

    def op(x, y, sym):
        ax.text(x, y, sym, ha="center", va="center", fontsize=19, color=C["grey"])

    # ---- Top: the naive order, forms n x n ----
    yT = 4.85
    ax.text(0.2, yT + 1.15, "Softmax forces this order:  first $QK^{\\top}$ "
            "(the $n\\times n$ matrix), then $\\times V$",
            fontsize=12.5, fontweight="bold", color=C["text"])
    block(0.4, yT - 0.6, 1.3, 1.2, "$Q$", "$n\\times d$", C["blue"], "white")
    op(1.95, yT, r"$\times$")
    block(2.2, yT - 0.6, 1.3, 1.2, "$K^{\\top}$", "$d\\times n$", C["purple"], "white")
    op(3.75, yT, r"$=$")
    block(4.05, yT - 0.85, 1.7, 1.7, "$QK^{\\top}$", "$n\\times n$",
          C["red"], "white", big=True)
    ax.text(4.9, yT - 1.15, "BIG: grows as $n^2$", ha="center", fontsize=8.8,
            style="italic", color=C["red"])
    op(6.05, yT, r"$\times$")
    block(6.35, yT - 0.6, 1.3, 1.2, "$V$", "$n\\times d$", C["green"], "white")
    op(7.9, yT, r"$=$")
    block(8.2, yT - 0.6, 1.5, 1.2, "out", "$n\\times d$", C["orange"], "white")
    ax.text(11.6, yT, "$O(n^2 d)$", ha="center", va="center", fontsize=15,
            fontweight="bold", color=C["red"])

    # divider
    ax.plot([0.3, 12.7], [3.15, 3.15], color=C["lgrey"], lw=2)

    # ---- Bottom: kernelized, reassociated, forms d x d ----
    yB = 1.55
    ax.text(0.2, yB + 1.35, "Kernel $\\phi$ removes softmax $\\Rightarrow$ "
            "reassociate:  first $\\phi(K)^{\\top}V$ (a small $d\\times d$ "
            "matrix), then $\\phi(Q)\\times$ it",
            fontsize=12.5, fontweight="bold", color=C["text"])
    block(0.4, yB - 0.6, 1.5, 1.2, "$\\phi(Q)$", "$n\\times d$", C["blue"], "white")
    op(2.15, yB, r"$\times$")
    ax.text(3.55, yB + 0.95, "compute this first",
            ha="center", fontsize=8.6, style="italic", color=C["green"])
    block(2.5, yB - 0.6, 1.5, 1.2, "$\\phi(K)^{\\top}$", "$d\\times n$",
          C["purple"], "white")
    op(4.25, yB, r"$\times$")
    block(4.55, yB - 0.6, 1.3, 1.2, "$V$", "$n\\times d$", C["green"], "white")
    op(6.1, yB, r"$=$")
    block(6.4, yB - 0.55, 1.15, 1.1, "$\\phi(K)^{\\top}V$", "$d\\times d$",
          C["green"], "white")
    ax.text(6.98, yB - 0.85, "SMALL: no $n$!", ha="center", fontsize=8.8,
            style="italic", color=C["green"])
    op(7.85, yB, r"$=$")
    block(8.2, yB - 0.6, 1.5, 1.2, "out", "$n\\times d$", C["orange"], "white")
    ax.text(11.6, yB, "$O(n d^2)$", ha="center", va="center", fontsize=15,
            fontweight="bold", color=C["green"])
    # bracket under phi(K)^T V pieces
    ax.annotate("", xy=(2.5, yB - 0.95), xytext=(5.85, yB - 0.95),
                arrowprops=dict(arrowstyle="-", color=C["green"], lw=1.4))
    ax.text(4.2, yB - 1.18, "reassociated: contract over $n$ first",
            ha="center", fontsize=8.4, color=C["green"])

    ax.text(6.5, 0.12, "Same three matrices — only the multiplication order "
            "changes.  Associativity turns the $n\\times n$ bottleneck into a "
            "$d\\times d$ one, and $d\\ll n$.",
            ha="center", fontsize=10, style="italic", color=C["text"])

    fig.tight_layout()
    save(fig, "eff_attn_linear_assoc.jpg")


# ---------------------------------------------------------------------------
# 3. Sparse attention patterns as n x n masks.
# ---------------------------------------------------------------------------
def fig_sparse_patterns():
    n = 24
    rng = np.random.default_rng(7)

    def full():
        return np.ones((n, n))

    def window(w=3):
        M = np.zeros((n, n))
        for i in range(n):
            for j in range(max(0, i - w), min(n, i + w + 1)):
                M[i, j] = 1
        return M

    def dilated(w=2, d=3):
        M = window(w)
        for i in range(n):
            for k in range(1, n):
                j1, j2 = i - k * d, i + k * d
                if 0 <= j1 < n:
                    M[i, j1] = 1
                if 0 <= j2 < n:
                    M[i, j2] = 1
        return M

    def longformer(w=2, ng=2):
        M = window(w)
        M[:ng, :] = 1  # global rows
        M[:, :ng] = 1  # global cols
        return M

    def bigbird(w=2, ng=1, nr=2):
        M = window(w)
        M[:ng, :] = 1
        M[:, :ng] = 1
        for i in range(n):  # a few random long-range links per row
            for j in rng.choice(n, size=nr, replace=False):
                M[i, j] = 1
        return M

    panels = [
        ("Full  $O(n^2)$", full(), C["red"]),
        ("Sliding window", window(), C["blue"]),
        ("+ Dilated / strided", dilated(), C["purple"]),
        ("Longformer\n(window + global)", longformer(), C["green"]),
        ("BigBird\n(window + global + random)", bigbird(), C["orange"]),
    ]
    fig, axes = plt.subplots(1, 5, figsize=(14.0, 3.4))
    for ax, (title, M, col) in zip(axes, panels):
        cmap = matplotlib.colors.ListedColormap(["white", col])
        ax.imshow(M, cmap=cmap, vmin=0, vmax=1)
        ax.set_title(title, fontsize=10.5, fontweight="bold", color=C["text"])
        ax.set_xticks([]); ax.set_yticks([])
        for s in ax.spines.values():
            s.set_edgecolor(C["grey"]); s.set_linewidth(1.2)
        frac = M.mean()
        ax.set_xlabel(f"{frac*100:.0f}% of entries", fontsize=9, color=C["grey"])
    fig.suptitle("Sparse attention: only compute a fixed subset of the "
                 "$n\\times n$ grid (row $i$ = which keys query $i$ may see)",
                 fontsize=12.5, fontweight="bold", color=C["text"], y=1.06)
    fig.tight_layout()
    save(fig, "eff_attn_sparse_patterns.jpg")


# ---------------------------------------------------------------------------
# 4. Linformer low-rank projection: shrink K, V along the length axis n -> k.
# ---------------------------------------------------------------------------
def fig_lowrank():
    fig, ax = plt.subplots(figsize=(12.5, 5.9))
    ax.set_xlim(0, 12.5)
    ax.set_ylim(0, 6.1)
    ax.axis("off")

    def block(x, y, w, h, label, shape, fc, txt_col=None):
        ax.add_patch(FancyBboxPatch((x, y), w, h,
                     boxstyle="round,pad=0.02,rounding_size=0.06",
                     fc=fc, ec=C["grey"], lw=1.3))
        ax.text(x + w / 2, y + h / 2 + 0.16, label, ha="center", va="center",
                fontsize=11.5, fontweight="bold", color=txt_col or C["text"])
        ax.text(x + w / 2, y + h / 2 - 0.30, shape, ha="center", va="center",
                fontsize=9, color=txt_col or C["grey"])

    def op(x, y, sym, fs=18):
        ax.text(x, y, sym, ha="center", va="center", fontsize=fs, color=C["grey"])

    ax.text(6.25, 5.85, "Linformer: the softmax matrix is approximately "
            "low-rank, so project $K,V$ from length $n$ down to a small fixed "
            "$k$ before attending",
            ha="center", fontsize=12.5, fontweight="bold", color=C["text"])

    # Projection step (top).
    yP = 3.9
    block(0.4, yP - 0.55, 1.35, 1.9, "$K$", "$n\\times d$", C["purple"], "white")
    op(1.95, yP + 0.4, r"$\rightarrow$")
    block(2.25, yP + 0.05, 1.3, 0.8, "$E K$", "$k\\times d$", C["purple"], "white")
    ax.text(2.9, yP - 0.35, "project length\n$n\\to k$", ha="center", fontsize=8,
            style="italic", color=C["grey"])

    block(4.6, yP - 0.55, 1.35, 1.9, "$V$", "$n\\times d$", C["green"], "white")
    op(6.15, yP + 0.4, r"$\rightarrow$")
    block(6.45, yP + 0.05, 1.3, 0.8, "$F V$", "$k\\times d$", C["green"], "white")

    ax.text(10.4, yP + 0.4, "$E, F \\in \\mathbb{R}^{k\\times n}$\nlearned, "
            "shared\nprojections", ha="center", fontsize=9, color=C["grey"])

    # Attention with the shrunk K', V' (bottom).
    yA = 1.35
    block(0.4, yA - 0.5, 1.3, 1.0, "$Q$", "$n\\times d$", C["blue"], "white")
    op(1.9, yA, r"$\times$")
    block(2.15, yA - 0.5, 1.4, 1.0, "$(EK)^{\\top}$", "$d\\times k$",
          C["purple"], "white")
    op(3.75, yA, r"$=$")
    block(4.0, yA - 0.5, 1.35, 1.0, "scores", "$n\\times k$", C["yellow"])
    ax.text(4.68, yA - 0.78, "small! not $n\\times n$", ha="center", fontsize=8,
            style="italic", color=C["red"])
    op(5.55, yA, r"$\times$")
    block(5.8, yA - 0.5, 1.3, 1.0, "$FV$", "$k\\times d$", C["green"], "white")
    op(7.3, yA, r"$=$")
    block(7.55, yA - 0.5, 1.5, 1.0, "out", "$n\\times d$", C["orange"], "white")
    ax.text(10.4, yA, "$O(nk)$,\nlinear in $n$\nfor fixed $k$", ha="center",
            va="center", fontsize=11, fontweight="bold", color=C["green"])

    fig.tight_layout()
    save(fig, "eff_attn_lowrank.jpg")


# ---------------------------------------------------------------------------
# 5. Reformer LSH: hash similar tokens into buckets, attend within a bucket.
#    Left: tokens as points, coloured by LSH bucket. Right: the resulting
#    block-diagonal attention pattern after sorting tokens by bucket.
# ---------------------------------------------------------------------------
def fig_reformer_lsh():
    fig, (axL, axR) = plt.subplots(1, 2, figsize=(12.0, 5.0),
                                   gridspec_kw={"width_ratios": [1.15, 1.0]})
    rng = np.random.default_rng(1)

    # Three latent clusters of token vectors (2-D stand-in for d-dim).
    centres = np.array([[-1.3, 0.9], [1.4, 1.0], [0.1, -1.4]])
    cols = [C["blue"], C["green"], C["orange"]]
    pts, lab = [], []
    for c in range(3):
        p = centres[c] + rng.normal(scale=0.42, size=(7, 2))
        pts.append(p); lab += [c] * 7
    pts = np.vstack(pts); lab = np.array(lab)

    for c in range(3):
        m = lab == c
        axL.scatter(pts[m, 0], pts[m, 1], s=110, color=cols[c],
                    edgecolor=C["grey"], linewidth=1.0, zorder=3,
                    label=f"bucket {c+1}")
    # A couple of LSH random-hyperplane boundaries.
    for a, b, c0 in [(1.0, 0.6, -0.2), (-0.7, 1.0, 0.3)]:
        xs = np.array([-2.4, 2.4])
        axL.plot(xs, -(a * xs + c0) / b, ls="--", color=C["grey"], lw=1.3,
                 zorder=1)
    axL.set_title("LSH hashes nearby q/k into the same bucket\n(random "
                  "hyperplanes = the hash)", fontsize=11, fontweight="bold",
                  color=C["text"])
    axL.set_xticks([]); axL.set_yticks([])
    axL.set_xlim(-2.6, 2.6); axL.set_ylim(-2.6, 2.6)
    axL.legend(fontsize=9, frameon=False, loc="upper center", ncol=3,
               bbox_to_anchor=(0.5, -0.02))
    for s in axL.spines.values():
        s.set_edgecolor(C["grey"])

    # Right: sort tokens by bucket -> block-diagonal attention mask.
    order = np.argsort(lab)
    sl = lab[order]
    n = len(sl)
    M = (sl[:, None] == sl[None, :]).astype(float)
    # colour each block by its bucket colour
    Mcol = np.ones((n, n, 3))
    for i in range(n):
        for j in range(n):
            if M[i, j]:
                from matplotlib.colors import to_rgb
                Mcol[i, j] = to_rgb(cols[sl[i]])
    axR.imshow(Mcol, interpolation="nearest")
    axR.set_title("Attend only within your bucket\n$\\Rightarrow$ block-diagonal, "
                  "$O(n\\log n)$", fontsize=11, fontweight="bold", color=C["text"])
    axR.set_xlabel("key j (sorted by bucket)", fontsize=9.5, color=C["grey"])
    axR.set_ylabel("query i (sorted by bucket)", fontsize=9.5, color=C["grey"])
    axR.set_xticks([]); axR.set_yticks([])
    for s in axR.spines.values():
        s.set_edgecolor(C["grey"]); s.set_linewidth(1.2)

    fig.suptitle("Reformer: replace the full $n\\times n$ grid with cheap "
                 "within-bucket attention", fontsize=12.5, fontweight="bold",
                 color=C["text"], y=1.02)
    fig.tight_layout()
    save(fig, "eff_attn_reformer_lsh.jpg")


if __name__ == "__main__":
    fig_landscape()
    fig_linear_assoc()
    fig_sparse_patterns()
    fig_lowrank()
    fig_reformer_lsh()
    print("done")
