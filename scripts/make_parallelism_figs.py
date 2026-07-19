"""Generate figures for training_llms/007_Parallelism.md.

Outputs (all .jpg, per repo convention):
  assets/par_collectives.jpg        -- the four workhorse collectives (all-reduce, all-gather,
                                        reduce-scatter, broadcast) as small before/after diagrams
  assets/par_memory_breakdown.jpg   -- where a training step's per-GPU memory goes (fp16 weights,
                                        fp32 master, ADAM m + v, fp16 grads, activations)
  assets/par_landscape.jpg          -- taxonomy: which axis each parallelism splits
  assets/par_data_parallel.jpg      -- data parallelism: replicate model, split batch, all-reduce grads
  assets/par_tensor_column_row.jpg  -- tensor parallelism: column vs row split + Megatron col->row
  assets/par_pipeline_bubble.jpg    -- pipeline schedules: naive (all-forward) vs GPipe vs 1F1B
  assets/par_zero_stages.jpg        -- ZeRO stages 1/2/3: what gets sharded, per-GPU memory shrinking
  assets/par_expert_parallel.jpg    -- MoE / expert parallelism: router + sharded experts + all-to-all
  assets/par_3d_parallelism.jpg     -- combining DP x TP x PP as a device mesh
  assets/par_batch_scaling.jpg      -- DP speedup vs per-GPU batch (comm-bound -> compute-bound)

The bar charts use numbers distilled from my own benchmarking suite
(parallelisms_from_scratch: 4-layer ~2.4B-param MLP, 4x H100, ADAM + fp16/fp32).
The schedule / mesh / collective figures are original schematics.

Run with the repo 'general' conda env:
  /Users/mohitl/Documents/miniconda3/envs/general/bin/python scripts/make_parallelism_figs.py
"""
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Rectangle

ASSETS = os.path.join(os.path.dirname(__file__), "..", "assets")

# A soft, consistent, brand-neutral palette (matches make_attention_figs.py etc.).
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


def rbox(ax, x, y, w, h, label, fc, txt=None, fs=10, sub=None, ec=None):
    ax.add_patch(FancyBboxPatch((x, y), w, h,
                 boxstyle="round,pad=0.02,rounding_size=0.05",
                 fc=fc, ec=ec or C["grey"], lw=1.2))
    if label:
        ax.text(x + w / 2, y + h / 2 + (0.11 if sub else 0), label, ha="center",
                va="center", fontsize=fs, fontweight="bold", color=txt or C["text"])
    if sub:
        ax.text(x + w / 2, y + h / 2 - 0.16, sub, ha="center", va="center",
                fontsize=fs - 2, color=txt or C["grey"])


def arrow(ax, x0, y0, x1, y1, col=None, lw=1.6, style="-|>"):
    ax.add_patch(FancyArrowPatch((x0, y0), (x1, y1), arrowstyle=style,
                 mutation_scale=14, color=col or C["grey"], lw=lw))


# ---------------------------------------------------------------------------
# 1. The four workhorse collectives.
#    Each mini-panel: 4 GPU cells before -> 4 GPU cells after, with a symbol.
# ---------------------------------------------------------------------------
def fig_collectives():
    fig, axes = plt.subplots(1, 4, figsize=(15.5, 3.9))
    W = 4
    titles = ["all-reduce", "all-gather", "reduce-scatter", "broadcast"]
    subt = [
        "sum & share\n(= reduce-scatter + all-gather)",
        "each keeps a shard\n-> everyone holds all shards",
        "sum, then split:\neach keeps 1/W of the sum",
        "one GPU's copy\n-> everyone",
    ]
    # colours per shard
    sc = [C["blue"], C["purple"], C["green"], C["orange"]]

    for ax, title, st, kind in zip(axes, titles, subt, range(4)):
        ax.set_xlim(0, 6); ax.set_ylim(0, 6.6); ax.axis("off")
        ax.set_title(title, fontsize=13, fontweight="bold", color=C["text"], pad=6)
        ax.text(3, 6.05, st, ha="center", va="center", fontsize=8.2,
                color=C["grey"])
        cw, ch = 1.0, 0.62
        xs = 0.35
        # "before" column (left) and "after" column (right)
        for g in range(W):
            y = 4.5 - g * 0.95
            # before
            if kind == 0 or kind == 2:      # start with distinct partials a,b,c,d
                rbox(ax, xs, y, cw, ch, "abcd" if False else f"g{g}", sc[g], fs=8,
                     txt="white")
            elif kind == 1:                  # all-gather: each has one shard
                rbox(ax, xs, y, cw*0.5, ch, ["A","B","C","D"][g], sc[g], fs=8, txt="white")
            elif kind == 3:                  # broadcast: only g0 has data
                if g == 0:
                    rbox(ax, xs, y, cw, ch, "X", C["blue"], fs=9, txt="white")
                else:
                    rbox(ax, xs, y, cw, ch, "", C["lgrey"], fs=8)
            # arrow
            arrow(ax, xs + cw + 0.15, y + ch/2, xs + cw + 1.0, y + ch/2)
            # after
            xa = xs + cw + 1.15
            if kind == 0:                    # all-reduce -> everyone has full sum
                rbox(ax, xa, y, cw, ch, "Σ", C["yellow"], fs=11)
            elif kind == 1:                  # all-gather -> everyone has ABCD
                for k in range(W):
                    rbox(ax, xa + k*0.28, y, 0.26, ch, "", sc[k], fs=6)
                ax.text(xa + 4*0.28 + 0.12, y+ch/2, "ABCD", fontsize=7, va="center")
            elif kind == 2:                  # reduce-scatter -> each keeps its 1/W of sum
                rbox(ax, xa, y, cw*0.5, ch, "Σ"+["A","B","C","D"][g], sc[g], fs=7, txt="white")
            elif kind == 3:                  # broadcast -> everyone has X
                rbox(ax, xa, y, cw, ch, "X", C["blue"], fs=9, txt="white")
    fig.suptitle("The collectives that power every parallelism strategy",
                 fontsize=14, fontweight="bold", color=C["text"], y=1.02)
    fig.tight_layout()
    save(fig, "par_collectives.jpg")


# ---------------------------------------------------------------------------
# 2. Where the memory goes (per GPU, one training step, ADAM + fp16/fp32).
#    Numbers (MB) from parallelisms_from_scratch, ~2.4B params.
# ---------------------------------------------------------------------------
def fig_memory_breakdown():
    comps = ["fp16\nweights", "fp32\nmaster", "ADAM\nmomentum", "ADAM\nvariance",
             "fp16\ngradients", "activations\n(+ overhead)"]
    vals = [4608, 9216, 9216, 9216, 4608, 18800]      # MB
    cols = [C["blue"], C["purple"], C["orange"], C["orange"], C["green"], C["grey"]]

    fig, (ax, ax2) = plt.subplots(1, 2, figsize=(13.5, 5.0),
                                  gridspec_kw={"width_ratios": [1.5, 1]})
    x = np.arange(len(comps))
    bars = ax.bar(x, vals, color=cols, edgecolor="white", width=0.72)
    for b, v in zip(bars, vals):
        ax.text(b.get_x()+b.get_width()/2, v+400, f"{v/1024:.1f} GB",
                ha="center", fontsize=9.5, color=C["text"])
    ax.set_xticks(x); ax.set_xticklabels(comps, fontsize=9)
    ax.set_ylabel("Memory per GPU (MB)", fontsize=11)
    ax.set_ylim(0, 21000)
    ax.spines[["top", "right"]].set_visible(False)
    ax.set_title("Per-GPU memory of one training step\n(~2.4B params, ADAM, mixed precision)",
                 fontsize=12, fontweight="bold", color=C["text"])
    # bracket the optimizer states
    ax.annotate("", xy=(3.42, 19800), xytext=(1.58, 19800),
                arrowprops=dict(arrowstyle="-", color=C["orange"], lw=1.4))
    ax.text(2.5, 20300, "ADAM states = 3x the fp16 model", ha="center",
            fontsize=9, style="italic", color=C["orange"])

    # right: the "static" pie -> what ZeRO can shard
    static = {"fp16 weights\n(replicated)": 4608, "fp32 master": 9216,
              "ADAM m": 9216, "ADAM v": 9216, "fp16 grads": 4608}
    labels = list(static.keys()); sv = list(static.values())
    pcols = [C["blue"], C["purple"], C["orange"], C["yellow"], C["green"]]
    wedges, _, _ = ax2.pie(sv, colors=pcols, autopct=lambda p: f"{p*sum(sv)/100/1024:.1f}G",
                           startangle=90, pctdistance=0.75,
                           wedgeprops=dict(edgecolor="white", linewidth=1.5),
                           textprops=dict(fontsize=8, color=C["text"]))
    ax2.legend(wedges, labels, loc="center left", bbox_to_anchor=(0.98, 0.5),
               frameon=False, fontsize=8.5)
    ax2.set_title("Static state ZeRO/FSDP can shard\n(everything but activations)",
                  fontsize=11, fontweight="bold", color=C["text"])
    fig.tight_layout()
    save(fig, "par_memory_breakdown.jpg")


# ---------------------------------------------------------------------------
# 3. The landscape: which axis each parallelism splits.
# ---------------------------------------------------------------------------
def fig_landscape():
    fig, ax = plt.subplots(figsize=(13.5, 6.2))
    ax.set_xlim(0, 13.5); ax.set_ylim(0, 6.2); ax.axis("off")
    rows = [
        ("Data Parallel (DP)",   "splits the BATCH",              C["blue"],
         "each GPU: full model, different examples"),
        ("Tensor Parallel (TP)", "splits WEIGHTS within a layer", C["purple"],
         "each GPU: a slice of every matrix"),
        ("Pipeline Parallel (PP)","splits LAYERS across depth",   C["green"],
         "each GPU: a contiguous block of layers"),
        ("ZeRO / FSDP",          "splits the STATE (opt/grad/param)", C["orange"],
         "each GPU: 1/W of optimizer states, grads, params"),
        ("Sequence / Context (SP/CP)","splits the SEQUENCE length", C["yellow"],
         "each GPU: a chunk of tokens"),
        ("Expert Parallel (EP)", "splits the EXPERTS (MoE)",      C["grey"],
         "each GPU: a subset of experts"),
    ]
    ax.text(6.75, 6.0, "What does each parallelism cut across?",
            ha="center", fontsize=14, fontweight="bold", color=C["text"])
    y = 5.05
    for name, what, col, note in rows:
        rbox(ax, 0.3, y, 3.7, 0.68, name, col, fs=11,
             txt="white" if col != C["yellow"] else C["text"])
        ax.text(4.4, y+0.34, what, fontsize=11.5, fontweight="bold",
                va="center", color=C["text"])
        ax.text(8.9, y+0.34, note, fontsize=9.3, va="center", color=C["grey"],
                style="italic")
        y -= 0.85
    save(fig, "par_landscape.jpg")


# ---------------------------------------------------------------------------
# 4. Data parallelism.
# ---------------------------------------------------------------------------
def fig_data_parallel():
    fig, ax = plt.subplots(figsize=(12.5, 5.4))
    ax.set_xlim(0, 12.5); ax.set_ylim(0, 5.4); ax.axis("off")
    W = 4
    ax.text(6.25, 5.15, "Data Parallelism: replicate the model, split the batch, average the gradients",
            ha="center", fontsize=13, fontweight="bold", color=C["text"])
    # global batch box
    rbox(ax, 0.3, 3.9, 2.1, 0.8, "Global batch", C["lgrey"], fs=10, sub="B examples")
    xs = [3.4, 5.7, 8.0, 10.3]
    for i, x in enumerate(xs):
        arrow(ax, 2.45, 4.3, x+0.05, 3.15)
        # per-GPU stack: shard of batch + full model replica
        rbox(ax, x, 2.5, 1.9, 0.6, f"batch shard {i+1}", C["blue"], fs=8.5,
             txt="white", sub="B/W examples")
        rbox(ax, x, 1.5, 1.9, 0.85, "FULL model copy", C["purple"], fs=9, txt="white",
             sub="same weights")
        ax.text(x+0.95, 3.25, f"GPU {i}", ha="center", fontsize=9,
                fontweight="bold", color=C["text"])
    # all-reduce bar underneath
    ax.add_patch(FancyBboxPatch((3.4, 0.55), 8.8, 0.6,
                 boxstyle="round,pad=0.02,rounding_size=0.05",
                 fc=C["yellow"], ec=C["grey"], lw=1.2))
    ax.text(7.8, 0.85, "all-reduce gradients, then divide by W  ->  identical weights everywhere",
            ha="center", fontsize=9.5, fontweight="bold", color=C["text"])
    for x in xs:
        arrow(ax, x+0.95, 1.45, x+0.95, 1.2, col=C["grey"], lw=1.2)
    save(fig, "par_data_parallel.jpg")


# ---------------------------------------------------------------------------
# 5. Tensor parallelism: column vs row, and Megatron col->row.
# ---------------------------------------------------------------------------
def fig_tensor_column_row():
    fig, axes = plt.subplots(1, 3, figsize=(15.5, 4.8))
    # -- column split --
    ax = axes[0]; ax.set_xlim(0, 6); ax.set_ylim(0, 6); ax.axis("off")
    ax.set_title("Column split:  Y = X W,  W cut by columns",
                 fontsize=11, fontweight="bold", color=C["text"])
    rbox(ax, 0.4, 3.5, 1.3, 1.6, "X", C["blue"], fs=12, txt="white", sub="B x F")
    ax.text(1.95, 4.3, "x", fontsize=16, ha="center", color=C["grey"])
    for k, col in enumerate([C["purple"], C["green"], C["orange"], C["yellow"]]):
        rbox(ax, 2.3+k*0.5, 3.5, 0.45, 1.6, "", col, fs=8)
    ax.text(3.05, 5.3, "W  (each GPU owns one column block)", fontsize=8.2,
            ha="center", color=C["grey"])
    ax.text(3.05, 3.15, "-> local Y_i of shape B x F/W", fontsize=8.5, ha="center",
            color=C["text"])
    ax.add_patch(FancyBboxPatch((0.6, 1.4), 4.8, 0.7,
                 boxstyle="round,pad=0.02,rounding_size=0.05",
                 fc=C["yellow"], ec=C["grey"], lw=1.1))
    ax.text(3.0, 1.75, "all-gather -> full Y (B x F)", fontsize=9.5, ha="center",
            fontweight="bold")

    # -- row split --
    ax = axes[1]; ax.set_xlim(0, 6); ax.set_ylim(0, 6); ax.axis("off")
    ax.set_title("Row split:  W cut by rows,  X pre-sharded",
                 fontsize=11, fontweight="bold", color=C["text"])
    for k, col in enumerate([C["purple"], C["green"], C["orange"], C["yellow"]]):
        rbox(ax, 0.4, 4.6-k*0.42, 1.3, 0.38, "", col, fs=8)
    ax.text(1.05, 5.25, "X shards", fontsize=8.2, ha="center", color=C["grey"])
    ax.text(1.95, 4.3, "x", fontsize=16, ha="center", color=C["grey"])
    for k, col in enumerate([C["purple"], C["green"], C["orange"], C["yellow"]]):
        rbox(ax, 2.3, 4.6-k*0.42, 1.3, 0.38, "", col, fs=8)
    ax.text(2.95, 5.25, "W row blocks", fontsize=8.2, ha="center", color=C["grey"])
    ax.text(3.0, 3.9, "-> partial sums (B x F each)", fontsize=8.5, ha="center",
            color=C["text"])
    ax.add_patch(FancyBboxPatch((0.6, 1.4), 4.8, 0.7,
                 boxstyle="round,pad=0.02,rounding_size=0.05",
                 fc=C["yellow"], ec=C["grey"], lw=1.1))
    ax.text(3.0, 1.75, "all-reduce -> full Y (B x F)", fontsize=9.5, ha="center",
            fontweight="bold")

    # -- Megatron col->row --
    ax = axes[2]; ax.set_xlim(0, 6); ax.set_ylim(0, 6); ax.axis("off")
    ax.set_title("Megatron: column THEN row\n(1 all-reduce per pair, output stays sharded between)",
                 fontsize=10.5, fontweight="bold", color=C["text"])
    steps = [("Column\n(no comm)", C["purple"], 4.6),
             ("Row\n(all-reduce)", C["green"], 3.4),
             ("Column\n(no comm)", C["purple"], 2.2),
             ("Row\n(all-reduce)", C["green"], 1.0)]
    for i, (lab, col, y) in enumerate(steps):
        rbox(ax, 1.6, y, 2.8, 0.85, lab, col, fs=9.5, txt="white")
        if i < 3:
            arrow(ax, 3.0, y, 3.0, y+0.35, col=C["grey"], lw=1.3)
    ax.text(5.3, 3.9, "all-reduce", fontsize=8, rotation=90, color=C["orange"], va="center")
    ax.text(5.3, 1.5, "all-reduce", fontsize=8, rotation=90, color=C["orange"], va="center")
    fig.suptitle("Tensor parallelism: split the matmul, pay one collective per layer",
                 fontsize=13, fontweight="bold", color=C["text"], y=1.03)
    fig.tight_layout()
    save(fig, "par_tensor_column_row.jpg")


# ---------------------------------------------------------------------------
# 6. Pipeline schedules: naive vs GPipe vs 1F1B.  Gantt-style timelines.
# ---------------------------------------------------------------------------
def fig_pipeline_bubble():
    fig, axes = plt.subplots(3, 1, figsize=(13.5, 7.2), sharex=True)
    W = 4
    fcol = C["blue"]; bcol = C["orange"]

    def draw(ax, forward_cells, backward_cells, title, util):
        ax.set_ylim(-0.5, W - 0.5); ax.set_xlim(0, 20)
        ax.set_yticks(range(W)); ax.set_yticklabels([f"GPU {i}" for i in range(W)],
                                                    fontsize=9)
        ax.invert_yaxis()
        for (g, t, m) in forward_cells:
            ax.add_patch(Rectangle((t, g-0.35), 1, 0.7, fc=fcol, ec="white"))
            ax.text(t+0.5, g, f"F{m}", ha="center", va="center", fontsize=7, color="white")
        for (g, t, m) in backward_cells:
            ax.add_patch(Rectangle((t, g-0.35), 1, 0.7, fc=bcol, ec="white"))
            ax.text(t+0.5, g, f"B{m}", ha="center", va="center", fontsize=7, color="white")
        ax.set_title(title, fontsize=11, fontweight="bold", color=C["text"], loc="left")
        ax.text(19.6, 0.0, util, ha="right", fontsize=9, style="italic",
                color=C["grey"])
        ax.spines[["top", "right", "left"]].set_visible(False)
        ax.tick_params(length=0)

    # (a) Naive: 1 micro-batch, full forward then full backward, one GPU at a time
    f = [(g, g, 0) for g in range(W)]
    b = [(g, 2*W - 1 - g, 0) for g in range(W)]
    draw(axes[0], f, b, "Naive (1 batch): only one GPU busy at a time -> huge bubble",
         "util ~ 1/W")

    # (b) GPipe: m micro-batches, all forwards then all backwards
    m = 4
    f = []
    for mb in range(m):
        for g in range(W):
            f.append((g, mb + g, mb))
    tstart = m + W - 1
    b = []
    for mb in range(m):
        for g in range(W):
            b.append((W-1-g, tstart + mb + (W-1-g), mb))
    draw(axes[1], f, b, "GPipe (m=4 micro-batches): fill the pipe, bubble shrinks with m",
         "bubble = (W-1)/(m+W-1)")

    # (c) 1F1B: interleave forward/backward to bound activation memory
    # simplified illustrative schedule
    f = []
    b = []
    # warmup forwards
    for g in range(W):
        for mb in range(W-g):
            f.append((g, g + mb, mb))
    # steady 1F1B (illustrative staggering)
    for g in range(W):
        base = W
        for k in range(m-(W-g)):
            f.append((g, base + g + 2*k, (W-g)+k))
    for g in range(W):
        for mb in range(m):
            b.append((g, W + g + 2*mb + 1, mb))
    draw(axes[2], f, b, "1F1B: same bubble as GPipe, but bounded activation memory",
         "steady-state: 1 fwd + 1 bwd")
    axes[2].set_xlabel("time  ->", fontsize=10)
    fig.suptitle("Pipeline parallelism: the bubble and how scheduling shrinks it",
                 fontsize=13, fontweight="bold", color=C["text"], y=0.99)
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    save(fig, "par_pipeline_bubble.jpg")


# ---------------------------------------------------------------------------
# 7. ZeRO stages: stacked per-GPU memory shrinking DP -> Z1 -> Z2 -> Z3.
#    Numbers in GB, per GPU (static state; activations excluded).
# ---------------------------------------------------------------------------
def fig_zero_stages():
    # Playbook accounting in units of Psi (# params), Adam + mixed precision:
    #   params(bf16)=2, grads(bf16)=2, opt states(fp32 master+m+v)=12  -> total 16 Psi.
    # Use data-parallel degree Nd=8 so the staircase is visible.
    Nd = 8
    fig, ax = plt.subplots(figsize=(12.0, 5.8))
    stages = ["Baseline DP\n(no sharding)", "ZeRO-1\n(shard opt states)",
              "ZeRO-2\n(+ shard grads)", "ZeRO-3 / FSDP\n(+ shard params)"]
    formulas = [r"$16\Psi$", r"$4\Psi + \frac{12\Psi}{N_d}$",
                r"$2\Psi + \frac{14\Psi}{N_d}$", r"$\frac{16\Psi}{N_d}$"]
    p, g, o = 2.0, 2.0, 12.0
    data = {
        "params (bf16), 2Ψ": [p, p, p, p/Nd],
        "grads (bf16), 2Ψ":  [g, g, g/Nd, g/Nd],
        "opt states (fp32), 12Ψ": [o, o/Nd, o/Nd, o/Nd],
    }
    cols = [C["blue"], C["green"], C["orange"]]
    x = np.arange(len(stages))
    bottom = np.zeros(len(stages))
    for (lab, vals), col in zip(data.items(), cols):
        ax.bar(x, vals, bottom=bottom, label=lab, color=col, edgecolor="white", width=0.62)
        bottom += np.array(vals)
    for i, tot in enumerate(bottom):
        ax.text(i, tot+0.5, f"{tot:.2f}Ψ", ha="center", fontsize=10.5,
                fontweight="bold", color=C["text"])
        ax.text(i, tot+1.4, formulas[i], ha="center", fontsize=10, color=C["grey"])
    ax.set_xticks(x); ax.set_xticklabels(stages, fontsize=9.5)
    ax.set_ylabel("Memory per GPU  (units of Ψ = # params)", fontsize=11)
    ax.set_ylim(0, 19)
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(frameon=False, fontsize=9.5, loc="upper right")
    ax.set_title(f"ZeRO / FSDP: shard progressively more state (shown at N_d = {Nd})\n"
                 "activations are NOT shardable and are excluded",
                 fontsize=12.5, fontweight="bold", color=C["text"])
    save(fig, "par_zero_stages.jpg")


# ---------------------------------------------------------------------------
# 7b. ZeRO-1 training step, unrolled (static recreation of dp_zero1.gif).
#     4 GPUs (rows) x 5 steps (columns). A full-width bar = full copy on every
#     GPU; a 1/Nd-width coloured sliver = each GPU only owns its shard.
# ---------------------------------------------------------------------------
def fig_zero1_steps():
    fig, ax = plt.subplots(figsize=(15.0, 5.2))
    Wg = 4
    ax.set_xlim(0, 15.0); ax.set_ylim(0, 5.2); ax.axis("off")
    steps = ["1. Forward\n(full params)", "2. Backward\n(full grads)",
             "3. Reduce-scatter\ngrads", "4. Optimizer step\n(on own shard)",
             "5. All-gather\nupdated params"]
    colx = [0.4, 3.3, 6.2, 9.1, 12.0]
    cw = 2.4
    gcols = [C["blue"], C["purple"], C["green"], C["orange"]]
    for c, (sx, title) in enumerate(zip(colx, steps)):
        ax.text(sx + cw/2, 4.9, title, ha="center", fontsize=9.5,
                fontweight="bold", color=C["text"])
        for r in range(Wg):
            y = 3.9 - r*0.85
            if c == 0:
                ax.text(sx - 0.15, y + 0.22, f"GPU{r}", ha="right", fontsize=7.5,
                        color=C["grey"])
            if c == 0:      # full params on every GPU
                rbox(ax, sx, y, cw, 0.44, "params", C["blue"], fs=7.5, txt="white")
            elif c == 1:    # full grads on every GPU
                rbox(ax, sx, y, cw, 0.44, "grads", C["green"], fs=7.5, txt="white")
            elif c == 2:    # each GPU keeps only its grad shard
                rbox(ax, sx + r*(cw/Wg), y, cw/Wg, 0.44, "", C["green"], fs=6)
                ax.add_patch(Rectangle((sx, y), cw, 0.44, fill=False, ec=C["lgrey"], lw=0.8))
            elif c == 3:    # each GPU updates its fp32/opt shard
                rbox(ax, sx + r*(cw/Wg), y, cw/Wg, 0.44, "", C["orange"], fs=6)
                ax.add_patch(Rectangle((sx, y), cw, 0.44, fill=False, ec=C["lgrey"], lw=0.8))
            elif c == 4:    # full params restored on every GPU
                rbox(ax, sx, y, cw, 0.44, "params", C["blue"], fs=7.5, txt="white")
        if c < len(colx)-1:
            arrow(ax, sx+cw+0.05, 2.2, colx[c+1]-0.05, 2.2, col=C["grey"], lw=1.3)
    # labels above the two collective arrows
    ax.text(5.95, 2.62, "reduce-scatter (grads)", ha="center",
            fontsize=7.5, color=C["orange"], style="italic")
    ax.text(11.75, 2.62, "all-gather (params)", ha="center",
            fontsize=7.5, color=C["orange"], style="italic")
    ax.text(7.5, 0.55, "Same total volume on the wire as plain DP's single all-reduce  "
            "(all-reduce = reduce-scatter + all-gather)  ->  optimizer sharding is 'free'.",
            ha="center", fontsize=9, style="italic", color=C["grey"])
    fig.suptitle("ZeRO-1: one training step, unrolled",
                 fontsize=13.5, fontweight="bold", color=C["text"], y=1.02)
    save(fig, "par_zero1_steps.jpg")


# ---------------------------------------------------------------------------
# 7c. Overlapping communication with compute (prefetching), recreation of
#     dp_zero1_overlap.svg idea: all-gather layer n+1's params while computing
#     layer n, so the comm is hidden behind compute.
# ---------------------------------------------------------------------------
def fig_zero_overlap():
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(13.0, 5.2))
    L = 4
    def gantt(ax, comm_hidden, title):
        ax.set_xlim(0, 12); ax.set_ylim(-0.5, 1.5); ax.axis("off")
        ax.text(0, 1.9, title, fontsize=11, fontweight="bold", color=C["text"])
        ax.text(-0.2, 1.0, "compute", ha="right", va="center", fontsize=9, color=C["grey"])
        ax.text(-0.2, 0.0, "comm",    ha="right", va="center", fontsize=9, color=C["grey"])
        t = 0.4
        for l in range(L):
            comp_w = 1.6
            comm_w = 1.0
            if comm_hidden:
                # comm for layer l+1 sits under compute of layer l (overlapped)
                ax.add_patch(Rectangle((t, 0.7), comp_w, 0.6, fc=C["blue"], ec="white"))
                ax.text(t+comp_w/2, 1.0, f"fwd L{l}", ha="center", va="center",
                        fontsize=7.5, color="white")
                if l < L-1:
                    ax.add_patch(Rectangle((t, -0.3), comp_w*0.9, 0.6, fc=C["orange"], ec="white"))
                    ax.text(t+comp_w*0.45, 0.0, f"gather L{l+1}", ha="center", va="center",
                            fontsize=7, color="white")
                t += comp_w + 0.1
            else:
                # comm then compute, serialized
                ax.add_patch(Rectangle((t, -0.3), comm_w, 0.6, fc=C["orange"], ec="white"))
                ax.text(t+comm_w/2, 0.0, f"gather L{l}", ha="center", va="center",
                        fontsize=7, color="white")
                t += comm_w + 0.05
                ax.add_patch(Rectangle((t, 0.7), comp_w, 0.6, fc=C["blue"], ec="white"))
                ax.text(t+comp_w/2, 1.0, f"fwd L{l}", ha="center", va="center",
                        fontsize=7.5, color="white")
                t += comp_w + 0.05
        ax.annotate("", xy=(t, -0.9), xytext=(0.4, -0.9),
                    arrowprops=dict(arrowstyle="-|>", color=C["grey"], lw=1.2))
        ax.text(t/2, -1.15, "time", fontsize=8, color=C["grey"], ha="center")
        return t
    t_bad = gantt(ax2, False, "Naive: gather params, THEN compute (comm stalls the GPU)")
    t_good = gantt(ax1, True, "Prefetch: all-gather layer n+1's params WHILE computing layer n")
    fig.suptitle("ZeRO-3 / FSDP: hide the parameter all-gather behind compute",
                 fontsize=13, fontweight="bold", color=C["text"], y=1.0)
    fig.tight_layout(rect=[0.05, 0, 1, 0.94])
    save(fig, "par_zero_overlap.jpg")


# ---------------------------------------------------------------------------
# 7d. Ring Attention: GPUs in a ring, KV blocks rotate while attention computes.
# ---------------------------------------------------------------------------
def fig_ring_attention():
    fig, ax = plt.subplots(figsize=(9.5, 8.0))
    ax.set_xlim(-5, 5); ax.set_ylim(-5.2, 5.6); ax.axis("off")
    ax.text(0, 5.2, "Ring Attention: rotate K,V around the ring, compute as you go",
            ha="center", fontsize=13, fontweight="bold", color=C["text"])
    R = 3.0
    pos = [(0, R), (R, 0), (0, -R), (-R, 0)]      # top, right, bottom, left
    gcols = [C["blue"], C["purple"], C["green"], C["orange"]]
    bw, bh = 2.0, 1.1
    for i, (x, y) in enumerate(pos):
        rbox(ax, x - bw/2, y - bh/2, bw, bh, f"GPU {i}", gcols[i], fs=11, txt="white",
             sub=f"holds Q,K,V chunk {i}")
    # curved arrows clockwise: 0->1->2->3->0
    for i in range(4):
        x0, y0 = pos[i]; x1, y1 = pos[(i+1) % 4]
        ax.add_patch(FancyArrowPatch((x0, y0), (x1, y1), arrowstyle="-|>",
                     mutation_scale=16, color=C["grey"], lw=1.8,
                     connectionstyle="arc3,rad=-0.28",
                     shrinkA=26, shrinkB=26))
    ax.text(2.55, 2.55, "send K,V\nto next GPU", fontsize=8.5, color=C["grey"],
            style="italic", ha="center")
    ax.text(0, 0.55, "each step", ha="center", fontsize=10, fontweight="bold",
            color=C["text"])
    ax.text(0, -0.55,
            "1. send my K,V onward (non-blocking)\n"
            "2. compute softmax(QKᵀ/√d)·V on the\n     K,V block I currently hold\n"
            "3. receive the next K,V block",
            ha="center", va="center", fontsize=8.6, color=C["text"])
    ax.text(0, -4.7,
            "After W steps every query has seen every key. Online softmax accumulates the\n"
            "partial outputs, so no GPU ever stores the full K,V — and the sends overlap the compute.",
            ha="center", fontsize=8.8, style="italic", color=C["grey"])
    save(fig, "par_ring_attention.jpg")


# ---------------------------------------------------------------------------
# 7e. Zig-Zag Ring Attention: causal mask makes naive assignment imbalanced;
#     zig-zag token assignment equalizes the per-GPU work.
# ---------------------------------------------------------------------------
def fig_ring_zigzag():
    n, W = 8, 4
    gcols = [C["blue"], C["purple"], C["green"], C["orange"]]
    # owner[row] = which GPU owns that query row
    seq_owner = [r // 2 for r in range(n)]                 # 0,0,1,1,2,2,3,3
    zz_owner  = [0, 1, 2, 3, 3, 2, 1, 0]                    # zig-zag pairing
    fig, axes = plt.subplots(1, 2, figsize=(14.0, 6.3))

    def draw(ax, owner, title):
        ax.set_xlim(-0.5, n + 2.6); ax.set_ylim(-2.4, n + 0.5); ax.axis("off")
        ax.text(n/2, n + 0.15, title, ha="center", fontsize=11.5,
                fontweight="bold", color=C["text"])
        load = [0] * W
        for i in range(n):          # query row i (drawn top-to-bottom)
            yy = n - 1 - i
            g = owner[i]
            for j in range(n):      # key col j; causal -> active if j <= i
                if j <= i:
                    ax.add_patch(Rectangle((j, yy), 1, 1, fc=gcols[g], ec="white", lw=0.6))
                    load[g] += 1
                else:
                    ax.add_patch(Rectangle((j, yy), 1, 1, fc=C["lgrey"], ec="white", lw=0.6))
            ax.text(-0.25, yy + 0.5, f"t{i}", ha="right", va="center", fontsize=7,
                    color=gcols[g])
        ax.text(n/2, -0.5, "keys  (causal: query t attends to keys ≤ t)", ha="center",
                fontsize=8.5, color=C["grey"])
        # per-GPU workload bars
        maxl = 15
        for g in range(W):
            ax.add_patch(Rectangle((n + 0.4, n - 1 - g*1.0), load[g]/maxl*2.0, 0.6,
                         fc=gcols[g], ec="white"))
            ax.text(n + 0.4, n - 0.55 - g*1.0, f"GPU{g}: {load[g]}", fontsize=7.5,
                    va="center", ha="left", color=C["text"])
        return load

    l1 = draw(axes[0], seq_owner, "Sequential: GPU0=t0-1 … GPU3=t6-7  →  imbalance")
    axes[0].text(n/2, -1.5, f"work per GPU: {l1}  →  later GPUs do far more",
                 ha="center", fontsize=8.8, color=C["orange"], style="italic")
    l2 = draw(axes[1], zz_owner, "Zig-zag: each GPU owns one early + one late chunk  →  balanced")
    axes[1].text(n/2, -1.5, f"work per GPU: {l2}  →  every GPU equal",
                 ha="center", fontsize=8.8, color=C["green"], style="italic")
    fig.suptitle("Causal masking breaks Ring Attention's balance; zig-zag token assignment fixes it",
                 fontsize=13, fontweight="bold", color=C["text"], y=1.0)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    save(fig, "par_ring_zigzag.jpg")


# ---------------------------------------------------------------------------
# 8. Expert / MoE parallelism.
# ---------------------------------------------------------------------------
def fig_expert_parallel():
    fig, ax = plt.subplots(figsize=(12.5, 5.4))
    ax.set_xlim(0, 12.5); ax.set_ylim(0, 5.4); ax.axis("off")
    ax.text(6.25, 5.25, "Expert (MoE) Parallelism: a router sends each token to a few experts",
            ha="center", fontsize=13, fontweight="bold", color=C["text"])
    # tokens
    tcols = [C["blue"], C["purple"], C["green"], C["orange"]]
    for i, col in enumerate(tcols):
        rbox(ax, 0.3, 3.7-i*0.55, 1.1, 0.45, f"tok {i}", col, fs=8, txt="white")
    # router
    rbox(ax, 2.1, 2.3, 1.5, 1.4, "Router\n(top-k)", C["yellow"], fs=10)
    for i in range(4):
        arrow(ax, 1.4, 3.9-i*0.55, 2.1, 3.0, col=C["grey"], lw=1.0)
    # experts on different GPUs
    ex = [("Expert 0\nGPU 0", C["blue"]), ("Expert 1\nGPU 1", C["purple"]),
          ("Expert 2\nGPU 2", C["green"]), ("Expert 3\nGPU 3", C["orange"])]
    for i, (lab, col) in enumerate(ex):
        rbox(ax, 5.4, 3.95-i*0.95, 2.0, 0.78, lab, col, fs=9, txt="white")
        arrow(ax, 3.6, 3.0, 5.4, 4.34-i*0.95, col=C["grey"], lw=1.0)
    # all-to-all label
    ax.add_patch(FancyBboxPatch((4.0, 0.4), 4.6, 0.55,
                 boxstyle="round,pad=0.02,rounding_size=0.05",
                 fc=C["lgrey"], ec=C["grey"], lw=1.1))
    ax.text(6.3, 0.67, "all-to-all: route tokens to experts, then gather results back",
            ha="center", fontsize=9, fontweight="bold", color=C["text"])
    ax.text(9.9, 3.0, "Each GPU stores only\nits experts' weights\n-> huge param count,\n"
                      "fixed compute per token", fontsize=9, color=C["grey"],
            style="italic", va="center")
    save(fig, "par_expert_parallel.jpg")


# ---------------------------------------------------------------------------
# 9. 3D parallelism: DP x TP x PP device mesh.
# ---------------------------------------------------------------------------
def fig_3d_parallelism():
    fig, ax = plt.subplots(figsize=(11.5, 6.0))
    ax.set_xlim(0, 11.5); ax.set_ylim(0, 6.0); ax.axis("off")
    ax.text(5.75, 5.7, "3D parallelism: combine DP x TP x PP on a device mesh",
            ha="center", fontsize=13.5, fontweight="bold", color=C["text"])
    # two DP replicas, each a PP x TP grid
    for rep in range(2):
        ox = 0.6 + rep * 5.6
        ax.text(ox+2.1, 5.15, f"DP replica {rep}  (different batch shard)",
                ha="center", fontsize=9.5, fontweight="bold", color=C["blue"])
        for pp in range(3):        # 3 pipeline stages (rows)
            for tp in range(2):    # 2 tensor shards (cols)
                col = [C["purple"], C["green"]][tp]
                rbox(ax, ox + tp*2.1, 3.5 - pp*1.05, 1.9, 0.85,
                     f"L{pp} / shard {tp}", col, fs=8, txt="white")
        # PP arrow down the left edge
        ax.annotate("", xy=(ox-0.15, 1.45), xytext=(ox-0.15, 4.3),
                    arrowprops=dict(arrowstyle="-|>", color=C["green"], lw=1.6))
        ax.text(ox-0.45, 2.85, "PP (layers)", rotation=90, fontsize=8.5,
                color=C["green"], va="center")
        # TP arrow across the top
        ax.annotate("", xy=(ox+4.2, 4.6), xytext=(ox+0.0, 4.6),
                    arrowprops=dict(arrowstyle="-|>", color=C["purple"], lw=1.6))
        ax.text(ox+2.0, 4.73, "TP (within layer)", fontsize=8.5, color=C["purple"],
                ha="center")
    ax.text(5.75, 0.5, "Rule of thumb: TP inside a node (NVLink), PP across nodes, "
                       "DP outermost; ZeRO shards state within each DP group.",
            ha="center", fontsize=9, style="italic", color=C["grey"])
    save(fig, "par_3d_parallelism.jpg")


# ---------------------------------------------------------------------------
# 10. Batch scaling: DP goes from comm-bound (slower) to compute-bound (faster).
#     Data points distilled from parallelisms_from_scratch report3/report4.
# ---------------------------------------------------------------------------
def fig_batch_scaling():
    fig, ax = plt.subplots(figsize=(9.5, 5.2))
    per_gpu_batch = np.array([768, 1536, 3072, 6144])   # examples per GPU
    # speedup of DP over single-GPU baseline (measured endpoints 0.89x, 1.42x; interp middle)
    dp = np.array([0.89, 1.10, 1.28, 1.42])
    dpzero = np.array([1.02, 1.15, 1.25, 1.33])
    ax.axhline(1.0, ls=":", color=C["grey"], lw=1.2)
    ax.text(6144, 1.02, "single-GPU baseline", ha="right", fontsize=8.5,
            color=C["grey"], style="italic")
    ax.plot(per_gpu_batch, dp, "o-", color=C["blue"], lw=2, label="Data Parallel")
    ax.plot(per_gpu_batch, dpzero, "s-", color=C["orange"], lw=2, label="DP + ZeRO-1")
    ax.axvspan(700, 1200, alpha=0.10, color=C["orange"])
    ax.text(950, 0.83, "comm-bound\n(DP slower!)", ha="center", fontsize=8.5,
            color=C["orange"])
    ax.axvspan(4000, 6500, alpha=0.10, color=C["green"])
    ax.text(5200, 0.83, "compute-bound\n(DP wins)", ha="center", fontsize=8.5,
            color=C["green"])
    ax.set_xscale("log", base=2)
    ax.set_xticks(per_gpu_batch); ax.set_xticklabels(per_gpu_batch)
    ax.set_xlabel("examples per GPU (larger batch ->)", fontsize=11)
    ax.set_ylabel("speedup vs single GPU", fontsize=11)
    ax.set_ylim(0.75, 1.55)
    ax.grid(True, ls="--", color="#dddddd", lw=0.8)
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(frameon=False, fontsize=10, loc="upper left")
    ax.set_title("Batch size is critical: DP only pays off once compute hides communication\n"
                 "(4x H100, ~2.4B MLP, ADAM + mixed precision)",
                 fontsize=11.5, fontweight="bold", color=C["text"])
    fig.tight_layout()
    save(fig, "par_batch_scaling.jpg")


if __name__ == "__main__":
    fig_collectives()
    fig_memory_breakdown()
    fig_landscape()
    fig_data_parallel()
    fig_tensor_column_row()
    fig_pipeline_bubble()
    fig_zero_stages()
    fig_zero1_steps()
    fig_zero_overlap()
    fig_ring_attention()
    fig_ring_zigzag()
    fig_expert_parallel()
    fig_3d_parallelism()
    fig_batch_scaling()
    print("done.")
