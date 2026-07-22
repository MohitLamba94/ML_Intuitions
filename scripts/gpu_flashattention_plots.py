"""
Figures for `gpu-tpu-matmul-flashattention.md` (hardware / matmul / FlashAttention note).

  Fig 1  roofline: arithmetic intensity vs attainable throughput, showing the
         memory-bound ramp, the math-bound ceiling, and three example GEMMs. -> Part 2
  Fig 2  tile quantization (partial edge tiles) + wave quantization (a tail
         wave that leaves most SMs idle).                                    -> Part 1
  Fig 3  GPU memory hierarchy: capacity vs bandwidth per level (A100-class,
         log scale) -- small+fast at top, big+slow at bottom.               -> Part 3
  Fig 4  CPU vs GPU floorplan cartoon: few fat latency cores + big cache vs
         a sea of thin throughput lanes.                                     -> Part 3
  Fig 5  systolic array: an NxN MAC grid, activations stream in from the left,
         weights held in place, each operand reused across a whole row/col.  -> Part 4
  Fig 6  A100 / H100 / B200 across HBM bandwidth, dense FP16 Tensor TFLOPS,
         and memory capacity.                                                -> Part 5
  Fig 7  FlashAttention tiling: Q/K/V split into SRAM-sized blocks, one block
         processed at a time with running (m, l) softmax statistics.         -> Part 6
  Fig 1b tiling data reuse: naive matmul refetches operands from HBM for every
         multiply (AI ~ 1); tiling loads a block into SRAM once and reuses each
         value T times (AI ~ T) -- how tiling manufactures arithmetic intensity. -> Part 1
  Fig 8  KV cache: without a cache every step recomputes all K/V (wasted); with
         a cache only the new token's K/V is computed and appended.          -> Part 8
  Fig 9  prefill vs decode: one wide compute-bound prefill GEMM followed by a
         run of thin memory-bound decode steps reading the growing cache.    -> Part 8
  Fig 10 kernel fusion: unfused chain of elementwise kernels each round-tripping
         HBM vs one fused kernel that keeps the intermediates on-chip.        -> Part 7

Pure numpy/matplotlib (no sklearn/torch). Deterministic (seeded). All numbers are
approximate, vendor-published, A100/H100/B200-class values (see note for citations).

Run:
    /Users/mohitl/Documents/miniconda3/envs/general/bin/python scripts/gpu_flashattention_plots.py
"""

import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, Rectangle
import numpy as np

# dataviz-skill categorical palette (light surface)
BLUE = "#2a78d6"   # slot 1
AQUA = "#1baf7a"   # slot 2
AMBER = "#e8a33d"  # slot 3 (highlight / warm)
RED = "#d1495b"    # slot 4 (waste / warning)
INK = "#0b0b0b"
INK2 = "#52514e"
GRID = "#dcdcd8"
SURFACE = "#fcfcfb"

HERE = os.path.dirname(os.path.abspath(__file__))
ASSETS = os.path.join(HERE, "..", "assets")


def _style(ax, grid=True):
    ax.set_facecolor(SURFACE)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(GRID)
    ax.tick_params(colors=INK2, labelsize=9)
    if grid:
        ax.grid(True, color=GRID, linewidth=0.8, alpha=0.7)
    ax.set_axisbelow(True)


def _bare(ax):
    ax.set_facecolor(SURFACE)
    for s in ("top", "right", "left", "bottom"):
        ax.spines[s].set_visible(False)
    ax.set_xticks([]); ax.set_yticks([])


def _save(fig, name):
    fig.patch.set_facecolor(SURFACE)
    path = os.path.join(ASSETS, name)
    fig.savefig(path, dpi=150, bbox_inches="tight", facecolor=SURFACE)
    plt.close(fig)
    print("wrote", os.path.normpath(path))


# ---------------------------------------------------------------- Fig 1  roofline
def fig_roofline():
    peak = 125.0        # V100 FP16 Tensor Core peak, TFLOP/s
    bw = 900.0          # HBM bandwidth, GB/s  -> 0.9 TB/s
    ridge = peak / (bw / 1000.0)   # FLOP/byte at the ridge point = 138.9

    ai = np.logspace(-1, 4, 400)                       # arithmetic intensity, FLOP/byte
    attainable = np.minimum(peak, (bw / 1000.0) * ai)  # TFLOP/s

    fig, ax = plt.subplots(figsize=(8.2, 5.0))
    _style(ax)
    ax.set_xscale("log"); ax.set_yscale("log")

    # shade the two regimes
    ax.axvspan(0.1, ridge, color=RED, alpha=0.06)
    ax.axvspan(ridge, 1e4, color=AQUA, alpha=0.07)

    ax.plot(ai, attainable, color=INK, lw=2.4, zorder=3)
    ax.axhline(peak, color=INK2, lw=1.0, ls=":")
    ax.axvline(ridge, color=INK2, lw=1.0, ls="--")
    ax.text(ridge * 1.1, 2.0, f"ridge point\n{ridge:.0f} FLOP/byte",
            color=INK2, fontsize=9)
    ax.text(0.12, peak * 1.15, f"peak compute = {peak:.0f} TFLOP/s", color=INK2, fontsize=9)

    ax.text(2.0, 1.4, "MEMORY-BOUND\n(bandwidth limits you)", color=RED,
            fontsize=10.5, ha="center", weight="bold")
    ax.text(900, 20, "MATH-BOUND\n(FLOPs limit you)", color=AQUA,
            fontsize=10.5, ha="center", weight="bold")

    pts = [(0.5, "GEMV / activation\n(AI < 1)", RED, (2.0, 0.55), "left"),
           (124, "GEMM\n8192x128x8192\n(AI ~ 124)", AMBER, (16, 45), "center"),
           (2730, "GEMM 8192^3\n(AI ~ 2730)", AQUA, (2730, 30), "center")]
    for x, lab, c, xytext, ha in pts:
        y = min(peak, (bw / 1000.0) * x)
        ax.scatter([x], [y], color=c, s=70, zorder=5, edgecolor=INK, linewidths=0.6)
        ax.annotate(lab, (x, y), xytext=xytext, color=INK, fontsize=8.5, ha=ha)

    ax.set_xlabel("arithmetic intensity  (FLOP per byte read/written)", color=INK2)
    ax.set_ylabel("attainable throughput  (TFLOP/s)", color=INK2)
    ax.set_title("The roofline: below the ridge you are starved for data, not compute\n"
                 "(V100-class numbers: 125 TFLOP/s peak, 0.9 TB/s HBM)",
                 color=INK, fontsize=10.5)
    ax.set_xlim(0.1, 1e4); ax.set_ylim(0.3, 300)
    _save(fig, "roofline_memory_vs_math.jpg")


# ---------------------------------------------------------------- Fig 1b tiling reuse
def fig_tiling_reuse():
    fig, (axL, axR) = plt.subplots(1, 2, figsize=(12.5, 5.4))

    def hbm_slab(ax, y=-0.15, h=0.7):
        ax.add_patch(Rectangle((0.2, y), 9.6, h, facecolor=RED, alpha=0.20,
                               edgecolor=INK2, linewidth=1.2))
        ax.text(5.0, y + h / 2, "HBM  (huge, slow: ~1.5 TB/s)", ha="center",
                va="center", color=INK, fontsize=9.5, weight="bold")

    def out_tile(ax, x0, y0, s=0.55, nb=3, color=BLUE, alpha=0.30):
        for i in range(nb):
            for j in range(nb):
                ax.add_patch(Rectangle((x0 + i * s, y0 + j * s), s * 0.9, s * 0.9,
                                       facecolor=color, alpha=alpha, edgecolor=INK2,
                                       linewidth=0.8))
        return (x0 + nb * s / 2, y0 + nb * s / 2)

    # ---- left: naive, refetch from HBM for every multiply --------------------
    _bare(axL)
    hbm_slab(axL)
    cx, cy = out_tile(axL, 3.4, 4.2, color=RED, alpha=0.28)
    axL.text(cx, 4.2 + 3 * 0.55 + 0.25, "output tile", ha="center", color=INK, fontsize=9)
    # many arrows straight from HBM to the output cells (re-reads)
    rng = np.random.default_rng(0)
    for _ in range(11):
        tx = 3.4 + rng.uniform(0.1, 1.5)
        ty = 4.2 + rng.uniform(0.1, 1.5)
        sx = rng.uniform(1.5, 8.5)
        axL.add_patch(FancyArrowPatch((sx, 0.55), (tx, ty), arrowstyle="-|>",
                     mutation_scale=9, color=RED, lw=1.0, alpha=0.55))
    axL.text(5.0, 3.15, "every multiply refetches its operands from HBM;\n"
                        "each row of $A$ re-read $N$ times, each col of $B$ re-read $M$ times",
             ha="center", color=INK, fontsize=9)
    axL.add_patch(Rectangle((2.0, 1.35), 6.0, 1.0, facecolor=SURFACE, edgecolor=RED,
                            linewidth=1.4))
    axL.text(5.0, 1.85, "HBM reads $\\approx 2MNK$    $\\Rightarrow$    AI $\\approx 1$\n"
                        "(memory-starved: Tensor Cores idle)", ha="center", va="center",
             color=RED, fontsize=9.5, weight="bold")
    axL.set_xlim(0, 10); axL.set_ylim(-0.6, 6.4)
    axL.set_title("Naive: no reuse", color=INK, fontsize=11)

    # ---- right: tiled, load once into SRAM then reuse ------------------------
    _bare(axR)
    hbm_slab(axR)
    # SRAM box in the middle holding the loaded tiles
    axR.add_patch(Rectangle((3.0, 1.9), 4.0, 1.5, facecolor=AQUA, alpha=0.18,
                            edgecolor=INK2, linewidth=1.3))
    axR.text(5.0, 3.05, "SRAM  (tiny, fast: ~19 TB/s)", ha="center", color=INK,
             fontsize=9.5, weight="bold")
    axR.text(5.0, 2.4, "$A$-tile $(T\\times K)$  +  $B$-tile $(K\\times T)$", ha="center",
             color=INK, fontsize=9.5)
    # one bulk load HBM -> SRAM
    axR.add_patch(FancyArrowPatch((5.0, 0.55), (5.0, 1.85), arrowstyle="-|>",
                 mutation_scale=16, color=BLUE, lw=2.6))
    axR.text(5.9, 1.2, "load $2TK$ once", ha="left", color=BLUE, fontsize=9)
    # reuse SRAM -> output tile
    cx, cy = out_tile(axR, 3.4, 4.4, color=AQUA, alpha=0.30)
    axR.text(cx, 4.4 + 3 * 0.55 + 0.25, "output tile", ha="center", color=INK, fontsize=9)
    for _ in range(11):
        tx = 3.4 + np.random.default_rng(_ + 5).uniform(0.1, 1.5)
        ty = 4.4 + np.random.default_rng(_ + 9).uniform(0.1, 1.5)
        axR.add_patch(FancyArrowPatch((5.0, 3.45), (tx, ty), arrowstyle="-|>",
                     mutation_scale=8, color=AQUA, lw=1.0, alpha=0.6))
    axR.text(5.0, 3.75, "each loaded value reused $T$ times from fast SRAM", ha="center",
             color=INK, fontsize=9)
    axR.text(5.0, -0.35, "HBM reads $\\approx 2TK$   $\\Rightarrow$   AI $\\approx T$   (compute-bound)",
             ha="center", va="center", color=AQUA, fontsize=9.5, weight="bold")
    axR.set_xlim(0, 10); axR.set_ylim(-0.8, 6.4)
    axR.set_title("Tiled: load once, reuse $T\\times$", color=INK, fontsize=11)

    fig.suptitle("How tiling manufactures arithmetic intensity: keep a block in SRAM and "
                 "reuse each value $T$ times,\nturning $\\sim\\!2MNK$ HBM reads into $\\sim\\!2TK$",
                 color=INK, fontsize=10.5, y=1.04)
    _save(fig, "tiling_reuse.jpg")


# ---------------------------------------------------------------- Fig 2  quantization
def fig_quantization():
    fig, (axL, axR) = plt.subplots(1, 2, figsize=(12, 4.8))

    # --- left: tile quantization -------------------------------------------
    _bare(axL)
    # output matrix 384 x 256, tiles of 128x128  -> 3 x 2 = 6 tiles, all full here;
    # to SHOW the effect use a matrix 384 x 260 so the last column of tiles is partial.
    W, H = 5.0, 3.0            # drawing box (columns x rows), 5 wide is odd on purpose
    ncol, nrow = 3, 3          # tile grid
    useful_frac_last_col = 0.18
    tw, th = W / ncol, H / nrow
    for i in range(ncol):
        for j in range(nrow):
            x0, y0 = i * tw, j * th
            partial = (i == ncol - 1)     # last column of tiles is mostly padding
            face = RED if partial else BLUE
            alpha = 0.20 if partial else 0.30
            axL.add_patch(Rectangle((x0, y0), tw, th, facecolor=face, alpha=alpha,
                                    edgecolor=INK2, linewidth=1.0))
            if partial:
                # the actually-useful sliver of the tile
                axL.add_patch(Rectangle((x0, y0), tw * useful_frac_last_col, th,
                                        facecolor=BLUE, alpha=0.55,
                                        edgecolor="none"))
    axL.add_patch(Rectangle((0, 0), W, H, fill=False, edgecolor=INK, linewidth=1.6))
    axL.text(W * 0.83, H + 0.18, "padding\n(wasted MACs)", color=RED, fontsize=9,
             ha="center")
    axL.text(W * 0.16, H + 0.18, "one 128x128 tile\n= one thread block", color=BLUE,
             fontsize=9, ha="center")
    axL.set_xlim(-0.3, W + 0.3); axL.set_ylim(-0.5, H + 0.9)
    axL.set_title("Tile quantization: the output doesn't divide evenly by the tile,\n"
                  "so edge tiles do a full tile's work for a sliver of useful output",
                  color=INK, fontsize=10)

    # --- right: wave quantization ------------------------------------------
    _style(axR, grid=False)
    n_sm = 108
    total_tiles = 117          # one full wave (108) + a 9-tile tail
    wave1 = np.ones(n_sm)
    tail = np.zeros(n_sm); tail[:total_tiles - n_sm] = 1.0
    xs = np.arange(n_sm)
    axR.bar(xs, wave1, width=1.0, color=BLUE, alpha=0.85, label="wave 1: all 108 SMs busy")
    axR.bar(xs, tail, width=1.0, bottom=wave1, color=RED, alpha=0.85,
            label="wave 2 (tail): only 9/108 SMs busy")
    axR.set_ylim(0, 2.4)
    axR.set_xlabel("streaming multiprocessor (SM) index, 0..107", color=INK2)
    axR.set_yticks([0.5, 1.5]); axR.set_yticklabels(["wave 1", "wave 2"], color=INK2)
    axR.legend(loc="upper right", fontsize=8.5, frameon=False)
    axR.set_title("Wave quantization: 117 tiles on 108 SMs means a second wave\n"
                  "that uses 9/108 = 8.3% of the GPU -- runtime nearly doubles",
                  color=INK, fontsize=10)
    _save(fig, "tile_wave_quantization.jpg")


# ---------------------------------------------------------------- Fig 3  memory hierarchy
def fig_memory_hierarchy():
    levels = ["Registers", "SRAM / shared\n(on-chip)", "L2 cache", "HBM (DRAM)"]
    capacity_MB = [27.0, 20.0, 40.0, 40000.0]     # A100-class, aggregate
    bandwidth_TBs = [100.0, 19.0, 5.0, 1.9]        # approximate
    y = np.arange(len(levels))[::-1]               # fast at top
    colors = [AQUA, BLUE, AMBER, RED]

    fig, (axL, axR) = plt.subplots(1, 2, figsize=(11.5, 4.4), sharey=True)

    _style(axL, grid=True); axL.set_xscale("log")
    axL.barh(y, capacity_MB, color=colors, alpha=0.85, height=0.6)
    for yi, v in zip(y, capacity_MB):
        lab = f"{v/1000:.0f} GB" if v >= 1000 else f"{v:.0f} MB"
        axL.text(v * 1.25, yi, lab, va="center", color=INK, fontsize=9)
    axL.set_yticks(y); axL.set_yticklabels(levels, color=INK, fontsize=9.5)
    axL.set_xlabel("capacity  (log scale)", color=INK2)
    axL.set_xlim(5, 2e5)
    axL.set_title("Capacity grows as you go down", color=INK, fontsize=10.5)

    _style(axR, grid=True); axR.set_xscale("log")
    axR.barh(y, bandwidth_TBs, color=colors, alpha=0.85, height=0.6)
    for yi, v in zip(y, bandwidth_TBs):
        axR.text(v * 1.12, yi, f"~{v:g} TB/s", va="center", color=INK, fontsize=9)
    axR.set_xlabel("bandwidth  (log scale)", color=INK2)
    axR.set_xlim(1, 400)
    axR.set_title("...but bandwidth shrinks", color=INK, fontsize=10.5)

    fig.suptitle("GPU memory hierarchy (A100-class, approximate): fast & tiny at the top,\n"
                 "huge & slow at the bottom -- keeping data high up is the whole game",
                 color=INK, fontsize=10.5, y=1.06)
    _save(fig, "gpu_memory_hierarchy.jpg")


# ---------------------------------------------------------------- Fig 4  CPU vs GPU
def fig_cpu_vs_gpu():
    fig, (axL, axR) = plt.subplots(1, 2, figsize=(11.5, 4.6))

    # --- CPU: a few fat cores + a big cache --------------------------------
    _bare(axL)
    axL.add_patch(Rectangle((0, 0), 10, 8, fill=False, edgecolor=INK, linewidth=1.6))
    # big shared cache band
    axL.add_patch(Rectangle((0.6, 0.6), 8.8, 2.2, facecolor=AMBER, alpha=0.28,
                            edgecolor=INK2))
    axL.text(5.0, 1.7, "large caches + control\n(out-of-order, branch prediction)",
             ha="center", va="center", color=INK, fontsize=9)
    for i in range(2):
        for j in range(2):
            x0, y0 = 0.9 + i * 4.4, 3.5 + j * 2.0
            axL.add_patch(Rectangle((x0, y0), 3.6, 1.6, facecolor=BLUE, alpha=0.5,
                                    edgecolor=INK, linewidth=1.2))
            axL.text(x0 + 1.8, y0 + 0.8, "core", ha="center", va="center",
                     color=INK, fontsize=10)
    axL.set_xlim(-0.4, 10.4); axL.set_ylim(-1.0, 9.2)
    axL.set_title("CPU: a few FAT cores tuned for latency\n"
                  "(finish one thread as fast as possible)", color=INK, fontsize=10.5)

    # --- GPU: a sea of small lanes -----------------------------------------
    _bare(axR)
    axR.add_patch(Rectangle((0, 0), 10, 8, fill=False, edgecolor=INK, linewidth=1.6))
    axR.add_patch(Rectangle((0.4, 0.3), 9.2, 0.7, facecolor=AMBER, alpha=0.28,
                            edgecolor=INK2))
    axR.text(5.0, 0.65, "small caches / control", ha="center", va="center",
             color=INK2, fontsize=8.5)
    nx, ny = 12, 8
    cw, ch = 9.2 / nx, 6.3 / ny
    for i in range(nx):
        for j in range(ny):
            x0 = 0.4 + i * cw
            y0 = 1.3 + j * ch
            axR.add_patch(Rectangle((x0 + 0.03, y0 + 0.03), cw - 0.08, ch - 0.08,
                                    facecolor=AQUA, alpha=0.55, edgecolor=SURFACE,
                                    linewidth=0.6))
    axR.set_xlim(-0.4, 10.4); axR.set_ylim(-1.0, 9.2)
    axR.set_title("GPU: thousands of THIN lanes tuned for throughput\n"
                  "(hide memory stalls behind many parallel threads)",
                  color=INK, fontsize=10.5)
    _save(fig, "cpu_vs_gpu_cores.jpg")


# ---------------------------------------------------------------- Fig 5  systolic array
def fig_systolic():
    fig, ax = plt.subplots(figsize=(7.4, 6.2))
    _bare(ax)
    n = 4
    step = 1.0
    x0, y0 = 1.5, 1.0
    for i in range(n):          # column
        for j in range(n):      # row
            cx = x0 + i * step
            cy = y0 + (n - 1 - j) * step
            ax.add_patch(Rectangle((cx - 0.32, cy - 0.32), 0.64, 0.64,
                                    facecolor=BLUE, alpha=0.30, edgecolor=INK2,
                                    linewidth=1.0))
            ax.text(cx, cy, "MAC", ha="center", va="center", color=INK, fontsize=7.5)

    # activations stream in from the left (rows)
    for j in range(n):
        cy = y0 + (n - 1 - j) * step
        ax.add_patch(FancyArrowPatch((x0 - 1.15, cy), (x0 - 0.4, cy),
                     arrowstyle="-|>", mutation_scale=13, color=AQUA, lw=2.0))
    ax.text(x0 - 1.5, y0 + (n - 1) * step + 0.9, "activations\nflow in $\\rightarrow$",
            color=AQUA, fontsize=10, ha="center", weight="bold")

    # weights loaded from the top (columns), partial sums accumulate downward
    for i in range(n):
        cx = x0 + i * step
        ax.add_patch(FancyArrowPatch((cx, y0 + n * step - 0.15), (cx, y0 + (n - 1) * step + 0.4),
                     arrowstyle="-|>", mutation_scale=13, color=AMBER, lw=2.0))
    ax.text(x0 + (n - 1) * step + 1.35, y0 + n * step - 0.1,
            "weights held\nin place $\\downarrow$", color=AMBER, fontsize=10,
            ha="center", weight="bold")

    # results out the bottom
    for i in range(n):
        cx = x0 + i * step
        ax.add_patch(FancyArrowPatch((cx, y0 - 0.4), (cx, y0 - 1.05),
                     arrowstyle="-|>", mutation_scale=13, color=INK2, lw=1.6))
    ax.text(x0 + (n - 1) * step / 2, y0 - 1.45, "partial sums accumulate out the bottom",
            color=INK2, fontsize=9.5, ha="center")

    ax.set_xlim(-0.6, x0 + n * step + 2.4); ax.set_ylim(y0 - 2.0, y0 + n * step + 1.3)
    ax.set_title("Systolic array (TPU): each value entering the grid is reused across\n"
                 "a whole row/column of MACs -- one HBM read feeds many multiplies",
                 color=INK, fontsize=10.5)
    _save(fig, "systolic_array.jpg")


# ---------------------------------------------------------------- Fig 6  generations
def fig_generations():
    gens = ["A100\n(Ampere)", "H100\n(Hopper)", "B200\n(Blackwell)"]
    colors = [INK2, BLUE, AQUA]
    hbm = [2.0, 3.0, 7.7]        # TB/s
    tflops = [312, 990, 2250]    # dense FP16 Tensor, TFLOP/s
    mem = [80, 80, 192]          # GB

    fig, axes = plt.subplots(1, 3, figsize=(12, 4.2))
    data = [(hbm, "HBM bandwidth (TB/s)", "{:.1f}"),
            (tflops, "dense FP16 Tensor (TFLOP/s)", "{:.0f}"),
            (mem, "memory capacity (GB)", "{:.0f}")]
    x = np.arange(3)
    for ax, (vals, title, fmt) in zip(axes, data):
        _style(ax, grid=True)
        ax.bar(x, vals, color=colors, alpha=0.85, width=0.62)
        for xi, v in zip(x, vals):
            ax.text(xi, v * 1.02, fmt.format(v), ha="center", va="bottom",
                    color=INK, fontsize=9.5)
        ax.set_xticks(x); ax.set_xticklabels(gens, color=INK, fontsize=9)
        ax.set_title(title, color=INK, fontsize=10.5)
        ax.set_ylim(0, max(vals) * 1.2)
    fig.suptitle("Three NVIDIA generations: bandwidth and compute climb together,\n"
                 "each jump paired with a lower-precision format (TF32 -> FP8 -> FP4)",
                 color=INK, fontsize=10.5, y=1.08)
    _save(fig, "gpu_generations.jpg")


# ---------------------------------------------------------------- Fig 7  flash attention
def fig_flash_tiling():
    fig, ax = plt.subplots(figsize=(9.0, 5.4))
    _bare(ax)

    # the full N x N scores matrix we REFUSE to materialize (dashed, faint)
    ax.add_patch(Rectangle((0, 0), 6, 6, fill=False, edgecolor=INK2, linewidth=1.2,
                           linestyle="--"))
    ax.text(3, 6.55, "the full $N\\times N$ scores matrix is never stored in HBM",
            color=INK2, fontsize=10, ha="center")

    nb = 3
    b = 6 / nb
    # grid of blocks
    for i in range(nb):
        for j in range(nb):
            x0, y0 = i * b, j * b
            active = (i == 1 and j == 1)
            face = AMBER if active else BLUE
            alpha = 0.55 if active else 0.14
            ax.add_patch(Rectangle((x0, y0), b, b, facecolor=face, alpha=alpha,
                                   edgecolor=INK2, linewidth=1.0))
    ax.text(1 * b + b / 2, 1 * b + b / 2, "one\n$Q_iK_j^\\top$\nblock\nin SRAM",
            ha="center", va="center", color=INK, fontsize=8.5, weight="bold")

    # Q blocks on the left (rows), K/V blocks on top (cols)
    for j in range(nb):
        ax.text(-0.6, (nb - 1 - j) * b + b / 2, f"$Q_{{{j+1}}}$", ha="center",
                va="center", color=BLUE, fontsize=11)
    for i in range(nb):
        ax.text(i * b + b / 2, 6.05, f"$K_{{{i+1}}},V_{{{i+1}}}$", ha="center",
                va="bottom", color=AQUA, fontsize=9.5)

    # running statistics panel on the right
    ax.add_patch(Rectangle((6.8, 1.4), 3.6, 3.1, facecolor=AQUA, alpha=0.10,
                           edgecolor=INK2, linewidth=1.0))
    ax.text(8.6, 4.1, "running softmax state\n(kept in SRAM)", ha="center",
            color=INK, fontsize=9.5, weight="bold")
    ax.text(8.6, 2.55, "$m$ = running max\n$\\ell$ = running sum of $e^{s-m}$\n"
                       "$O$ = rescaled output", ha="center", color=INK2, fontsize=9)

    ax.add_patch(FancyArrowPatch((6.15, 3.0), (6.75, 3.0), arrowstyle="-|>",
                 mutation_scale=14, color=INK2, lw=1.8))
    ax.text(3, -0.5, "walk over K/V blocks one at a time, update $(m,\\ell,O)$ on the fly "
                     "-- exact softmax, no full matrix",
            color=INK, fontsize=9.5, ha="center")

    ax.set_xlim(-1.2, 10.6); ax.set_ylim(-1.0, 6.8)
    ax.set_title("FlashAttention: tile Q/K/V into SRAM-sized blocks and stream the softmax",
                 color=INK, fontsize=11)
    _save(fig, "flash_attention_tiling.jpg")


# ---------------------------------------------------------------- Fig 8  KV cache
def fig_kv_cache():
    tokens = ["The", "cat", "sat", "on"]      # already generated
    new = "the"                                # token being generated at this step
    n = len(tokens)

    fig, (axL, axR) = plt.subplots(1, 2, figsize=(12, 4.6))

    def draw_row(ax, label_color, recompute):
        # draw K,V slots for the n past tokens + 1 new token
        for i, tok in enumerate(tokens + [new]):
            is_new = (i == n)
            if is_new:
                face = AMBER
            else:
                face = RED if recompute else BLUE
            alpha = 0.55 if (is_new or recompute) else 0.22
            ax.add_patch(Rectangle((i * 1.2, 0), 1.05, 1.0, facecolor=face, alpha=alpha,
                                   edgecolor=INK2, linewidth=1.0))
            ax.text(i * 1.2 + 0.52, 0.5, f"$K_{{{i}}},V_{{{i}}}$", ha="center", va="center",
                    color=INK, fontsize=9)
            ax.text(i * 1.2 + 0.52, 1.25, tok + ("  (new)" if is_new else ""),
                    ha="center", va="center", color=label_color, fontsize=8.5)
        ax.set_xlim(-0.2, (n + 1) * 1.2); ax.set_ylim(-0.4, 1.7)

    _bare(axL)
    draw_row(axL, INK2, recompute=True)
    axL.set_title("WITHOUT cache: recompute $K,V$ for every token, every step\n"
                  "(red = recomputed from scratch -- but weights are frozen, so identical)",
                  color=INK, fontsize=10)

    _bare(axR)
    draw_row(axR, INK2, recompute=False)
    # bracket the cached region
    axR.annotate("", xy=(-0.1, -0.28), xytext=(n * 1.2 - 0.15, -0.28),
                 arrowprops=dict(arrowstyle="<->", color=BLUE, lw=1.5))
    axR.text((n * 1.2) / 2 - 0.1, -0.6, "read from cache (computed once)", ha="center",
             color=BLUE, fontsize=9)
    axR.text(n * 1.2 + 0.52, -0.6, "compute\n& append", ha="center", color=AMBER,
             fontsize=9)
    axR.set_ylim(-0.9, 1.7)
    axR.set_title("WITH cache: compute $K,V$ only for the new token, append it\n"
                  "(blue = reused from cache; amber = the one new column)",
                  color=INK, fontsize=10)
    fig.suptitle("The KV cache removes redundant recomputation "
                 "($Q$ is not cached -- it is used once, then discarded)",
                 color=INK, fontsize=10.5, y=1.06)
    _save(fig, "kv_cache_redundancy.jpg")


# ---------------------------------------------------------------- Fig 9  prefill vs decode
def fig_prefill_decode():
    fig, ax = plt.subplots(figsize=(11, 4.4))
    _bare(ax)

    # prefill: one wide block
    ax.add_patch(Rectangle((0, 0.6), 3.0, 1.4, facecolor=AQUA, alpha=0.55,
                           edgecolor=INK, linewidth=1.4))
    ax.text(1.5, 1.3, "PREFILL\nwhole prompt\nin parallel", ha="center", va="center",
            color=INK, fontsize=10, weight="bold")
    ax.text(1.5, 0.25, "big GEMM -> COMPUTE-BOUND\n(high arithmetic intensity)", ha="center",
            color=AQUA, fontsize=9, weight="bold")

    # decode: many thin steps
    x0 = 3.6
    w = 0.55
    gap = 0.35
    n_steps = 7
    for s in range(n_steps):
        x = x0 + s * (w + gap)
        ax.add_patch(Rectangle((x, 0.6), w, 1.4, facecolor=RED, alpha=0.5,
                               edgecolor=INK, linewidth=1.0))
        ax.text(x + w / 2, 1.3, f"t{s+1}", ha="center", va="center", color=INK, fontsize=8)
        # each step reads the growing cache (arrow from a cache bar below)
    # growing cache bar underneath the decode steps
    cache_x1 = x0 + n_steps * (w + gap)
    ax.add_patch(Rectangle((x0 - 0.1, -0.55), cache_x1 - x0, 0.4, facecolor=BLUE,
                           alpha=0.25, edgecolor=INK2, linewidth=1.0))
    ax.text((x0 + cache_x1) / 2 - 0.1, -0.35, "KV cache in HBM (grows one token per step)",
            ha="center", color=BLUE, fontsize=8.5)
    for s in range(n_steps):
        x = x0 + s * (w + gap) + w / 2
        ax.add_patch(FancyArrowPatch((x, -0.13), (x, 0.55), arrowstyle="-|>",
                     mutation_scale=9, color=INK2, lw=0.9))
    ax.text((x0 + cache_x1) / 2 - 0.1, 0.25,
            "one token at a time, each reads the WHOLE cache -> MEMORY-BOUND (AI < 1)",
            ha="center", color=RED, fontsize=9, weight="bold")

    ax.annotate("", xy=(3.0, 2.25), xytext=(0.0, 2.25),
                arrowprops=dict(arrowstyle="<->", color=INK2, lw=1.2))
    ax.text(1.5, 2.4, "once", ha="center", color=INK2, fontsize=9)
    ax.annotate("", xy=(cache_x1, 2.25), xytext=(x0, 2.25),
                arrowprops=dict(arrowstyle="<->", color=INK2, lw=1.2))
    ax.text((x0 + cache_x1) / 2, 2.4, "repeated for every generated token", ha="center",
            color=INK2, fontsize=9)

    ax.set_xlim(-0.3, cache_x1 + 0.3); ax.set_ylim(-0.9, 2.7)
    ax.set_title("Two phases of LLM inference: a compute-bound prefill, then many "
                 "memory-bound decode steps", color=INK, fontsize=11)
    _save(fig, "prefill_vs_decode.jpg")


def fig_latency_throughput():
    """Latency & throughput vs batch size for Llama-2-13B on an H100. -> Part 9"""
    # Llama-2-13B-class config, bf16, context S = 1024
    V, D, F, L = 32000, 5120, 13824, 40
    N, K, H = 40, 40, 128          # query heads, kv heads (MHA here), head dim
    S = 1024
    bw = 3.35e12                   # H100 HBM bandwidth, bytes/s
    hbm = 80e9                     # H100 capacity, bytes

    P = 2 * V * D + 3 * D * F * L + (2 * D * N * H + 2 * D * K * H) * L
    w_bytes = 2 * P                                  # bf16 weights
    kv_per_seq = 2 * 2 * S * K * H * L               # 2(K,V) * 2 bytes * S * KH * L

    b = np.logspace(0, np.log10(256), 200)
    mem = w_bytes + b * kv_per_seq
    lat_ms = mem / bw * 1e3
    thru = b / (mem / bw)
    b_cliff = (hbm - w_bytes) / kv_per_seq            # ~64

    fig, axL = plt.subplots(figsize=(9.2, 5.2))
    _style(axL)
    axL.set_xscale("log")
    axL.plot(b, lat_ms, color=BLUE, lw=2.4, label="latency (ms/token)")
    axL.set_xlabel(r"batch size  $n_\mathrm{batch}$  (concurrent requests, log scale)", color=INK2, fontsize=10)
    axL.set_ylabel("latency  (ms / token)", color=BLUE, fontsize=10)
    axL.tick_params(axis="y", colors=BLUE)

    axR = axL.twinx()
    for s in ("top",):
        axR.spines[s].set_visible(False)
    axR.plot(b, thru, color=AQUA, lw=2.4, label="throughput (tok/s)")
    axR.set_ylabel("throughput  (tokens / s)", color=AQUA, fontsize=10)
    axR.tick_params(axis="y", colors=AQUA, labelsize=9)

    # memory cliff
    axL.axvline(b_cliff, color=RED, ls="--", lw=1.4)
    axL.axvspan(b_cliff, 256, color=RED, alpha=0.10)
    axL.text(b_cliff * 1.05, axL.get_ylim()[1] * 0.55,
             "exceeds 80 GB HBM\n" + r"($n_\mathrm{batch} > $" + f"{b_cliff:.0f}: does not fit)",
             color=RED, fontsize=9, weight="bold", va="center")

    # markers at B = 1, 64, 256
    for bi in (1, 64, 256):
        m = w_bytes + bi * kv_per_seq
        li = m / bw * 1e3
        ti = bi / (m / bw)
        axL.plot(bi, li, "o", color=BLUE, ms=6, zorder=5)
        axR.plot(bi, ti, "o", color=AQUA, ms=6, zorder=5)
        axL.annotate(r"$n_\mathrm{batch}$=" + f"{bi}\n{li:.0f} ms, {ti:.0f} tok/s",
                     xy=(bi, li), xytext=(0, -34 if bi != 256 else 12),
                     textcoords="offset points", ha="center", color=INK2, fontsize=8)

    axL.set_xlim(1, 256); axL.set_ylim(0, lat_ms.max() * 1.1)
    axR.set_ylim(0, thru.max() * 1.12)
    axL.set_title("Latency vs throughput vs batch size  —  Llama-2-13B on one H100 "
                  "(bf16, context 1024 tokens)", color=INK, fontsize=11.5)
    _save(fig, "latency_throughput_batch.jpg")


def fig_kernel_fusion():
    """Unfused (3 kernels round-tripping HBM) vs fused (1 kernel, on-chip). -> Part 7"""
    fig, (axL, axR) = plt.subplots(1, 2, figsize=(12, 5.2))
    for ax in (axL, axR):
        _bare(ax)
        ax.set_xlim(0, 10); ax.set_ylim(0, 10)

    def hbm_bar(ax):
        ax.add_patch(Rectangle((0.4, 0.4), 9.2, 1.1, facecolor=BLUE, alpha=0.22,
                               edgecolor=INK2, linewidth=1.0))
        ax.text(5.0, 0.95, "HBM (slow, off-chip)", ha="center", va="center",
                color=BLUE, fontsize=10, weight="bold")

    def arrow(ax, x0, y0, x1, y1, color, lw=1.4):
        ax.add_patch(FancyArrowPatch((x0, y0), (x1, y1), arrowstyle="-|>",
                     mutation_scale=12, color=color, lw=lw))

    # ---------------- LEFT: unfused, 3 kernels each round-tripping HBM
    hbm_bar(axL)
    labels = ["multiply\n(x*w)", "add\n(+b)", "sigmoid"]
    xs = [1.6, 4.5, 7.4]
    w = 2.0
    for x, lab in zip(xs, labels):
        axL.add_patch(Rectangle((x, 4.4), w, 2.2, facecolor=RED, alpha=0.45,
                                edgecolor=INK, linewidth=1.4))
        axL.text(x + w / 2, 5.5, lab, ha="center", va="center", color=INK,
                 fontsize=10, weight="bold")
        # read up from HBM, write back down to HBM
        arrow(axL, x + w * 0.35, 1.55, x + w * 0.35, 4.35, RED)
        arrow(axL, x + w * 0.65, 4.35, x + w * 0.65, 1.55, INK2)
        axL.text(x + w * 0.20, 3.0, "read", ha="center", color=RED, fontsize=8, rotation=90)
        axL.text(x + w * 0.80, 3.0, "write", ha="center", color=INK2, fontsize=8, rotation=90)
    # intermediates labelled as HBM round-trips between kernels
    axL.text((xs[0] + w + xs[1]) / 2, 5.5, "x*w", ha="center", color=INK2, fontsize=9, style="italic")
    axL.text((xs[1] + w + xs[2]) / 2, 5.5, "x*w+b", ha="center", color=INK2, fontsize=9, style="italic")
    arrow(axL, xs[0] + w, 5.5, xs[1], 5.5, INK2, lw=1.0)
    arrow(axL, xs[1] + w, 5.5, xs[2], 5.5, INK2, lw=1.0)
    axL.text(5.0, 8.7, "UNFUSED  —  3 kernels", ha="center", color=RED,
             fontsize=13, weight="bold")
    axL.text(5.0, 7.9, "every intermediate round-trips through HBM",
             ha="center", color=INK2, fontsize=10)
    axL.text(5.0, 0.0, "~8 HBM transfers   ·   3 launches", ha="center",
             color=RED, fontsize=10.5, weight="bold")

    # ---------------- RIGHT: fused, one kernel, on-chip
    hbm_bar(axR)
    # single tall kernel box with a shaded on-chip region inside
    axR.add_patch(Rectangle((2.0, 3.4), 6.0, 4.2, facecolor=AQUA, alpha=0.16,
                            edgecolor=INK, linewidth=1.6))
    axR.text(5.0, 7.15, "FUSED KERNEL", ha="center", color=AQUA, fontsize=11,
             weight="bold")
    # on-chip registers/SRAM region
    axR.add_patch(Rectangle((2.6, 3.8), 4.8, 2.7, facecolor=AMBER, alpha=0.18,
                            edgecolor=AMBER, linewidth=1.2, linestyle="--"))
    axR.text(5.0, 6.15, "registers / SRAM (on-chip)", ha="center", color="#b9791f",
             fontsize=9, style="italic")
    # the chain, entirely inside, no HBM arrows between steps
    axR.text(5.0, 5.3, "multiply  ->  add  ->  sigmoid", ha="center", va="center",
             color=INK, fontsize=11, weight="bold")
    axR.text(5.0, 4.55, "intermediates stay in registers", ha="center",
             color=INK2, fontsize=8.5)
    # read the 3 inputs once, write 1 result
    for i, (x, lab) in enumerate(zip([2.9, 3.7, 4.5], ["x", "w", "b"])):
        arrow(axR, x, 1.55, x, 3.35, AQUA)
        axR.text(x, 2.55, lab, ha="center", color=AQUA, fontsize=9, weight="bold")
    axR.text(2.2, 2.55, "read\nonce", ha="center", va="center", color=AQUA, fontsize=8)
    arrow(axR, 7.2, 3.35, 7.2, 1.55, INK2)
    axR.text(7.2, 2.55, "result", ha="center", color=INK2, fontsize=9, weight="bold")
    axR.text(5.0, 8.7, "FUSED  —  1 kernel", ha="center", color=AQUA,
             fontsize=13, weight="bold")
    axR.text(5.0, 7.9, "same math, intermediates never leave the chip",
             ha="center", color=INK2, fontsize=10)
    axR.text(5.0, 0.0, "~4 HBM transfers   ·   1 launch", ha="center",
             color=AQUA, fontsize=10.5, weight="bold")

    fig.suptitle("Kernel fusion:  y = sigmoid(x*w + b)  —  half the memory traffic for the same arithmetic",
                 color=INK, fontsize=12.5, y=0.99)
    _save(fig, "kernel_fusion.jpg")


if __name__ == "__main__":
    fig_roofline()
    fig_tiling_reuse()
    fig_quantization()
    fig_memory_hierarchy()
    fig_cpu_vs_gpu()
    fig_systolic()
    fig_generations()
    fig_flash_tiling()
    fig_kv_cache()
    fig_prefill_decode()
    fig_kernel_fusion()
    fig_latency_throughput()
    print("done")
