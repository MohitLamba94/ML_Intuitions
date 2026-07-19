"""Generate the two original figures for training_llms/001_tokeniser.md.

Outputs (all .jpg, per repo convention):
  assets/tokenization_granularities.jpg  -- same sentence split 4 ways
  assets/bpe_merge_steps.jpg             -- BPE merge walk-through on a toy corpus

Run with the repo 'general' conda env:
  /Users/mohitl/Documents/miniconda3/envs/general/bin/python scripts/make_tokenizer_figs.py
"""
import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

ASSETS = os.path.join(os.path.dirname(__file__), "..", "assets")

# A soft, consistent palette (brand-neutral).
C = {
    "char":   "#6C8EBF",   # blue
    "byte":   "#9673A6",   # purple
    "subw":   "#82B366",   # green
    "word":   "#D6795B",   # warm orange
    "grey":   "#555555",
    "lgrey":  "#EDEDED",
    "merge":  "#E8B84B",   # highlight yellow
    "text":   "#222222",
}


def rounded_tokens(ax, tokens, y, color, label, xstart=0.055, xend=0.985):
    """Draw a horizontal row of rounded 'token' chips spanning [xstart, xend]."""
    n = len(tokens)
    gap = 0.006
    total = xend - xstart
    w = (total - gap * (n - 1)) / n
    ax.text(0.045, y, label, ha="right", va="center", fontsize=12.5,
            fontweight="bold", color=C["text"], transform=ax.transAxes)
    for i, tok in enumerate(tokens):
        x = xstart + i * (w + gap)
        box = FancyBboxPatch(
            (x, y - 0.028), w, 0.056,
            boxstyle="round,pad=0.002,rounding_size=0.012",
            transform=ax.transAxes, facecolor=color, edgecolor="white",
            linewidth=1.4, alpha=0.92, mutation_aspect=0.4)
        ax.add_patch(box)
        disp = tok if tok != " " else "␣"
        fs = 10.5 if len(disp) <= 3 else 9.0
        ax.text(x + w / 2, y, disp, ha="center", va="center",
                fontsize=fs, color="white", fontweight="bold",
                transform=ax.transAxes)
    ax.text(0.992, y, f"n={n}", ha="left", va="center", fontsize=10.5,
            color=C["grey"], transform=ax.transAxes)


def fig_granularities():
    sentence = "tokenizing text!"
    chars = list(sentence)
    # bytes: same as chars here (all ASCII) but note count; show as hex-ish ids
    byte_ids = [f"{ord(c):d}" for c in sentence]
    # a plausible BPE / subword split
    subwords = ["token", "izing", " ", "text", "!"]
    words = ["tokenizing", " ", "text!"]

    fig, ax = plt.subplots(figsize=(12.2, 6.4))
    ax.axis("off")
    ax.set_xlim(0, 1); ax.set_ylim(0, 1)

    ax.text(0.5, 0.955, "Same string, four tokenization granularities",
            ha="center", va="center", fontsize=16.5, fontweight="bold",
            color=C["text"], transform=ax.transAxes)
    ax.text(0.5, 0.905, 'input text:  "tokenizing text!"   (16 characters, 16 UTF-8 bytes)',
            ha="center", va="center", fontsize=11.5, color=C["grey"],
            style="italic", transform=ax.transAxes)

    rounded_tokens(ax, words,    0.78, C["word"], "word")
    rounded_tokens(ax, subwords, 0.60, C["subw"], "subword\n(BPE)")
    rounded_tokens(ax, chars,    0.42, C["char"], "character")
    rounded_tokens(ax, byte_ids, 0.24, C["byte"], "byte")

    # annotation arrow: shorter <-> longer sequence
    ax.annotate("", xy=(0.5, 0.075), xytext=(0.5, 0.155),
                arrowprops=dict(arrowstyle="-", color=C["grey"], lw=0),
                transform=ax.transAxes)
    ax.text(0.5, 0.135, "fewer tokens  ↑  (short sequence, big vocab)      "
                        "↓  more tokens  (long sequence, tiny vocab)",
            ha="center", va="center", fontsize=11, color=C["grey"],
            transform=ax.transAxes)
    ax.text(0.5, 0.055,
            "vocab size grows top→bottom is REVERSED: word≈10⁵–10⁶ ▸ subword≈10⁴–10⁵ ▸ char≈few×10² ▸ byte=256",
            ha="center", va="center", fontsize=10, color=C["text"],
            transform=ax.transAxes)

    fig.tight_layout()
    out = os.path.join(ASSETS, "tokenization_granularities.jpg")
    fig.savefig(out, dpi=150, bbox_inches="tight", pil_kwargs={"quality": 90},
                facecolor="white")
    plt.close(fig)
    print("saved", out)


def fig_bpe_steps():
    """BPE merge walk-through on toy corpus: low, lower, newest, widest."""
    fig, ax = plt.subplots(figsize=(12.2, 7.2))
    ax.axis("off")
    ax.set_xlim(0, 1); ax.set_ylim(0, 1)

    ax.text(0.5, 0.965, "BPE training: greedily merge the most frequent adjacent pair",
            ha="center", va="center", fontsize=16, fontweight="bold",
            color=C["text"], transform=ax.transAxes)
    ax.text(0.5, 0.922,
            "toy corpus (word : count):   low·5   lower·2   newest·6   widest·3   "
            "(· = word-end marker)",
            ha="center", va="center", fontsize=11, color=C["grey"],
            style="italic", transform=ax.transAxes)

    # rows: each shows the segmentation state after a merge
    rows = [
        ("start", "l o w ·  |  l o w e r ·  |  n e w e s t ·  |  w i d e s t ·",
         "vocab = individual characters/bytes", None),
        ("merge 1", "l o w ·  |  l o w e r ·  |  n e w [es] t ·  |  w i d [es] t ·",
         "pair (e,s) most frequent  →  new token  es", "es"),
        ("merge 2", "l o w ·  |  l o w e r ·  |  n e w [est] ·  |  w i d [est] ·",
         "pair (es,t) most frequent  →  new token  est", "est"),
        ("merge 3", "[lo] w ·  |  [lo] w e r ·  |  n e w est ·  |  w i d est ·",
         "pair (l,o) most frequent  →  new token  lo", "lo"),
        ("merge 4", "[low] ·  |  [low] e r ·  |  n e w est ·  |  w i d est ·",
         "pair (lo,w) most frequent  →  new token  low", "low"),
    ]

    y0 = 0.83
    dy = 0.135
    for i, (tag, seg, note, newtok) in enumerate(rows):
        y = y0 - i * dy
        # left tag
        ax.text(0.045, y, tag, ha="left", va="center", fontsize=12,
                fontweight="bold", color=C["subw"] if i else C["grey"],
                transform=ax.transAxes)
        # segmentation panel
        box = FancyBboxPatch((0.16, y - 0.045), 0.63, 0.09,
                             boxstyle="round,pad=0.004,rounding_size=0.01",
                             transform=ax.transAxes, facecolor=C["lgrey"],
                             edgecolor="#CCCCCC", linewidth=1.0,
                             mutation_aspect=0.5)
        ax.add_patch(box)
        ax.text(0.475, y, seg, ha="center", va="center", fontsize=10.6,
                family="monospace", color=C["text"], transform=ax.transAxes)
        # note on the right
        ax.text(0.80, y, note, ha="left", va="center", fontsize=9.6,
                color=C["grey"], transform=ax.transAxes)

    # merge-list callout at the bottom
    ax.text(0.045, 0.075,
            "ordered merge list  (this list IS the tokenizer):",
            ha="left", va="center", fontsize=11.5, fontweight="bold",
            color=C["text"], transform=ax.transAxes)
    ax.text(0.045, 0.032,
            "1) e+s→es    2) es+t→est    3) l+o→lo    4) lo+w→low    …   "
            "encode new text by replaying these rules in order",
            ha="left", va="center", fontsize=10.2, family="monospace",
            color=C["word"], transform=ax.transAxes)

    fig.tight_layout()
    out = os.path.join(ASSETS, "bpe_merge_steps.jpg")
    fig.savefig(out, dpi=150, bbox_inches="tight", pil_kwargs={"quality": 90},
                facecolor="white")
    plt.close(fig)
    print("saved", out)


if __name__ == "__main__":
    fig_granularities()
    fig_bpe_steps()
