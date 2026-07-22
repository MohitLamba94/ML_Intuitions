# Why Hardware Makes Matrix Multiply Fast — GPUs, TPUs, and FlashAttention

*A single lens on a pile of folklore: why a $128\times128$ matmul beats a $125\times125$ one, what "tiling" means, how a GPU is built differently from a CPU, why TPUs exist, what actually changed from A100 → H100 → B200, and why FlashAttention was such a big deal.*

If you have spent any time around deep learning you have absorbed a set of half-explained rules of thumb: "make your dimensions multiples of 8," "pad to a power of two," "the matrix multiply is the expensive part," "memory bandwidth is the real bottleneck," "FlashAttention made attention fast without changing the math." Each of these is true, but stated on its own each sounds like superstition.

This note is written for someone who has **not** read the NVIDIA performance guides, the TPU paper, or the FlashAttention paper, and wants the mental model that ties all of it together. That model is one sentence: **modern deep-learning speed is governed less by how many arithmetic operations you do (FLOPs) than by how much data you move through the memory hierarchy.** Almost everything below is a consequence of that single fact — the matrix-shape quirks, the shape of GPU and TPU silicon, and FlashAttention's whole reason for existing.

We assume you are comfortable with what a matrix multiply *is* and with undergraduate ideas like caches and parallelism. We will not write CUDA. Where softmax mechanics matter (Part 6), we lean on the companion note [`absmax-mse-vs-softmax-ce.md`](./absmax-mse-vs-softmax-ce.md) rather than re-deriving them; the "a linear layer is a matrix multiply" picture is developed in [`lora.md`](./lora.md).

---

## Table of Contents

- [Setup and Notation](#setup-and-notation)
- [Part 1 — Why the *shape* of a matrix changes its speed (128 vs 125)](#part-1--why-the-shape-of-a-matrix-changes-its-speed-128-vs-125)
- [Part 2 — The roofline: memory-bound vs math-bound](#part-2--the-roofline-memory-bound-vs-math-bound)
- [Part 3 — GPU vs CPU: the fundamental difference](#part-3--gpu-vs-cpu-the-fundamental-difference)
- [Part 4 — TPUs and the systolic array](#part-4--tpus-and-the-systolic-array)
- [Part 5 — A100 → H100 → B200: what changed and why](#part-5--a100--h100--b200-what-changed-and-why)
- [Part 6 — FlashAttention: same math, far less data movement](#part-6--flashattention-same-math-far-less-data-movement)
- [Part 7 — Kernel fusion and Triton: how you actually write a data-light kernel](#part-7--kernel-fusion-and-triton-how-you-actually-write-a-data-light-kernel)
- [Part 8 — The KV cache: prefill, decode, and why inference is memory-bound](#part-8--the-kv-cache-prefill-decode-and-why-inference-is-memory-bound)
- [Part 9 — The math of compute- vs memory-bound: arithmetic intensity, latency, and throughput](#part-9--the-math-of-compute--vs-memory-bound-arithmetic-intensity-latency-and-throughput)
- [Sources](#sources)

---

## Setup and Notation

A few terms recur throughout. Read this once and refer back.

| Symbol / term | Meaning |
| --- | --- |
| **FLOP** | One floating-point operation (a multiply or an add). A matmul of shapes $M\times K$ by $K\times N$ costs about $2MNK$ FLOPs — one multiply and one add per inner-product term. |
| **byte moved** | One byte read from or written to memory. We count these separately from FLOPs — they are the other, often dominant, cost. |
| **(memory) bandwidth** | The *rate* at which data streams to/from memory, in **bytes per second** — the "width of the pipe." An H100's HBM runs at $\approx 3.35\times10^{12}$ byte/s $= 3.35$ TB/s. Do not confuse it with *latency* (the delay before the first byte arrives): bandwidth is throughput, how many bytes per second flow once the pipe is running. Time to move $N$ bytes $\approx N / \text{bandwidth}$, so this is the number that sets the cost of a memory-bound op. |
| **GEMM** | *General Matrix–Matrix multiply*, $C = A B$ with $A$ of shape $M\times K$, $B$ of shape $K\times N$, $C$ of shape $M\times N$. The workhorse of deep learning. |
| **GEMV** | Matrix–**vector** multiply (one of $M,N$ equals 1). |
| **HBM** | *High-Bandwidth Memory* — the large, off-chip DRAM on a GPU board (tens of GB). Big but relatively slow to reach. |
| **SRAM** | Fast on-chip memory (registers + "shared memory" / L1). Tiny (KB–MB) but an order of magnitude faster than HBM. |
| **SM** | *Streaming Multiprocessor* — one of the ~100+ parallel compute units on an NVIDIA GPU. |
| **warp** | A group of 32 GPU threads that execute the *same* instruction in lockstep (the SIMT model). |
| **Tensor Core** | A hardware unit inside each SM that does a small dense matrix multiply (e.g. a $16\times16$-ish block) in one shot, far faster than doing it with scalar multiply-adds. |
| **tile** | A sub-block of the output matrix that one thread block is responsible for computing. |
| **arithmetic intensity** | FLOPs performed per byte moved: $\dfrac{\text{FLOPs}}{\text{bytes}}$. The single number that decides whether you are compute- or memory-limited. |
| **ops:byte ratio** | A property of the *hardware*: peak FLOP/s divided by peak byte/s. Compare arithmetic intensity to this to see which resource runs out first. |

The two costs — FLOPs and bytes moved — race each other. Whichever takes longer sets your runtime. Parts 1 and 2 make that race precise.

---

## Part 1 — Why the *shape* of a matrix changes its speed (128 vs 125)

### 1.1 Tiling: the reason shape matters at all

A GPU does not compute the output matrix $C$ in one monolithic operation. It **cuts $C$ into rectangular tiles** — say $128\times128$ — and hands each tile to one *thread block* (a group of threads that lives on one SM). That block loads the slice of $A$ and the slice of $B$ it needs, multiplies them, and writes its tile of $C$. Stepping along the shared $K$ dimension in chunks, it accumulates the result.

Why bother tiling instead of streaming the whole thing? **Data reuse** — and to see why it matters we have to count *bytes moved*, not just FLOPs. A matmul is unusually rich in reuse: in $C = AB$, each element of $A$ is used in $N$ different output elements (once for every column of $B$), and each element of $B$ is used in $M$ of them (once for every row of $A$). The whole performance question is whether you *exploit* that reuse or throw it away. Consider the two extremes.

**The naive extreme: no reuse.** Suppose you compute each output element $C_{ij}=\sum_k A_{ik}B_{kj}$ independently, pulling its operands straight from HBM every time. Each output element then costs $2K$ reads (a length-$K$ row of $A$ and a length-$K$ column of $B$), so across all $MN$ outputs you read about $2MNK$ elements from HBM while doing about $2MNK$ FLOPs. That is roughly **one FLOP per element moved — arithmetic intensity ≈ 1.** From Part 2, an intensity of 1 sits far to the *left* of the roofline ridge (which is ~40–140): the operation is hopelessly **memory-starved**, and the Tensor Cores spend almost all their time waiting on HBM. The same row of $A$ gets dragged across the slow bus $N$ separate times; the same column of $B$, $M$ times. Enormous, redundant traffic.

**The tiled extreme: load once, reuse many times.** Now instead assign one $T\times T$ tile of $C$ to a thread block. It loads the corresponding $T\times K$ block of $A$ and $K\times T$ block of $B$ into on-chip SRAM/registers **once** — about $2TK$ reads from HBM — and then performs all $T^2K$ multiply-adds for that tile reading its operands from fast SRAM, not HBM. Every value that was loaded is now **reused about $T$ times** before it is discarded. HBM traffic for the tile drops from the naive $\sim 2T^2K$ down to $\sim 2TK$: a factor of $T$ fewer trips to slow memory. Equivalently, the arithmetic intensity climbs from $\approx 1$ to $\approx T$.

Put a number on it: with a $128\times128$ tile, each loaded value feeds roughly **128 multiply-adds** instead of one, so you make on the order of **128× fewer HBM reads**. That single change lifts arithmetic intensity from $\approx 1$ (deep in the memory-bound region) to $\approx 128$ — right around the roofline ridge — which is exactly what moves the matmul from *memory-starved* to *compute-bound*, where the Tensor Cores can finally run near peak. **This is what "tiling manufactures high arithmetic intensity" means:** the raw operation has intensity $\approx 1$, and tiling is the transformation that trades a big block of scarce SRAM for a $T$-fold cut in HBM traffic. It is the "move less data" principle of Part 2 applied at the level of a single matmul.

![Two side-by-side schematics. Left ("Naive: no reuse"): an output tile at top with many red arrows reaching straight down into a wide "HBM (huge, slow)" slab at the bottom, annotated that each row of A is re-read N times and each column of B M times; a callout reads "HBM reads ≈ 2MNK ⇒ AI ≈ 1 (memory-starved: Tensor Cores idle)". Right ("Tiled: load once, reuse T×"): a single thick blue arrow carries one bulk "load 2TK once" from the HBM slab up into an "SRAM (tiny, fast)" box holding the A-tile and B-tile; from that SRAM box many green arrows fan out to the output tile, annotated "each loaded value reused T times"; a callout reads "HBM reads ≈ 2TK ⇒ AI ≈ T (compute-bound)".](./assets/tiling_reuse.jpg){ width=100% }
*Tiling converts a memory-starved operation into a compute-bound one. Naively, every multiply refetches its operands from slow HBM (intensity ≈ 1). Tiled, a block is loaded into fast SRAM once and each value is reused $T$ times, cutting HBM traffic from $\sim\!2MNK$ to $\sim\!2TK$ (intensity ≈ $T$). (Recreated in our notation; the reuse/arithmetic-intensity argument follows the NVIDIA Matrix Multiplication and GPU-performance guides.)*

One practical wrinkle: a full $T\times K$ strip may itself be too big for SRAM when $K$ is large, so the block walks the shared $K$ dimension in **chunks**, loading a $T\times K_\text{chunk}$ slice of $A$ and a $K_\text{chunk}\times T$ slice of $B$ at a time and **accumulating** partial sums into the tile of $C$ (which stays resident in registers the whole time). The reuse story is unchanged: each $A$/$B$ element is still streamed in only once and reused across the tile; only the output tile is held in place.

That is also why tiling exists in the first place, and why the tile has a *fixed, hardware-chosen* size: it must be small enough that a tile's operands (plus the resident output tile) fit in the SM's limited SRAM/registers, yet large enough that the $T$-fold reuse actually pays for the load. Common tile sizes are $128\times128$, $256\times64$, $256\times128$, and so on — never $125\times117$. That fixed tile size is the root of every quirk below.

### 1.2 Tile quantization: partial tiles do full work

Here is the first place $128$ beats $125$. If your output matrix's dimensions are **not multiples of the tile size**, the tiles at the edges hang off the end of the matrix. The hardware still launches a *full* tile for them — it cannot launch a fractional thread block — so those edge tiles do nearly a full tile's worth of multiply-adds while producing only a sliver of useful output. The wasted work is called the **tile quantization** effect.

NVIDIA's own example: a $384\times256$ output with $128\times128$ tiles fits perfectly (3×2 = 6 tiles). But nudge one dimension just past a tile boundary and you spill into an extra row/column of tiles that are almost entirely padding — executing up to **1.5× more operations for 0.39% more actual data**. A $125\times125$ matrix pays exactly this tax: it still occupies the same tiles a $128\times128$ would, but three of those "128" columns and rows are wasted padding.

![A tiled output matrix: the leftmost columns of tiles are fully useful (blue), while the rightmost column of tiles is almost entirely padding (red) with only a thin useful sliver — the hardware still does a full tile's work there. Beside it, a stacked bar showing 117 tiles spread over 108 SMs: the first wave fills all 108 SMs, and a second "tail" wave uses only 9, leaving the GPU 92% idle during that wave.](./assets/tile_wave_quantization.jpg){ width=100% }
*Left: tile quantization — dimensions that don't divide the tile force nearly-empty edge tiles. Right: wave quantization — a tile count that doesn't divide the SM count leaves a nearly-empty tail wave. (Recreated in our notation; the $384\times256$ tile example and the 117-tiles-on-108-SMs example are from the NVIDIA Matrix Multiplication performance guide.)*

### 1.3 Wave quantization: partial waves leave SMs idle

The same "you can't do a fraction" problem repeats one level up. All the tiles are distributed across the GPU's SMs. An A100 has **108 SMs**, so it can run 108 tiles concurrently — one "wave." If your matmul produces, say, 117 tiles, you get one full wave of 108 plus a **tail wave of just 9 tiles**. That tail wave uses $9/108 = 8.3\%$ of the GPU but takes roughly as long as the full wave, so your effective throughput for that portion collapses and total runtime can nearly **double** for adding just a few tiles. This is the **wave quantization** effect (the right panel above).

The fix is the same in spirit: pick dimensions (and batch sizes) so the tile count is a multiple of, or comfortably larger than, the SM count, so the last wave is full or the tail is amortized over many waves.

### 1.4 Tensor Core alignment: the innermost granularity

There is a *third*, finest level of granularity. Tensor Cores — the units that make the matmul fast — consume operands in fixed small blocks (conceptually $16\times16$). To feed them without waste, the matrix dimensions need to be multiples of a hardware-friendly number. Concretely (with modern cuBLAS):

| Data type | Wants dimensions that are multiples of | Most efficient on A100 |
| --- | --- | --- |
| INT8 | 16 | 128 |
| FP16 | 8 | 64 |
| TF32 | 4 | 32 |

When $K$ (the contraction dimension) is not a multiple of 8 for FP16, the older library couldn't even use Tensor Cores; letting it do so was a documented **2–4× speedup**. If your dimension is misaligned, the hardware pads to the next multiple and the extra lanes compute zeros — pure waste.

### 1.5 Putting it together: why padding to a power of two is "free speed"

Now the folklore is just arithmetic. Padding $125 \to 128$ does three good things at once:

1. **Tensor Core alignment** — 128 is a multiple of 8/16/32/64, so no lanes are wasted at the innermost level.
2. **No tile quantization** — 128 is exactly the tile size, so no edge tiles are mostly padding.
3. **Cleaner wave packing** — nice round tile counts are more likely to fill whole waves of SMs.

The counterintuitive punchline is that the $128\times128$ matmul can be genuinely *faster in wall-clock time* than $125\times125$ **even though it does more arithmetic** — because it wastes nothing at any of the three granularities, while the "smaller" matrix quietly pays for full tiles, a ragged tail wave, and misaligned Tensor Core lanes. Powers of two (and multiples of 8/16/128) are not magic; they are simply the numbers that divide evenly at every level of the hardware's fixed block structure.

---

## Part 2 — The roofline: memory-bound vs math-bound

Part 1 was about *waste* at fixed granularities. Part 2 is the deeper question: even with a perfectly shaped matmul, what actually limits you — the arithmetic units, or the pipe feeding them data?

### 2.1 Arithmetic intensity vs the hardware's ops:byte ratio

Every kernel has an **arithmetic intensity**: FLOPs done per byte moved to/from HBM. Every GPU has an **ops:byte ratio**: its peak FLOP/s divided by its peak byte/s. The comparison is the whole story:

$$
\text{math-bound} \iff \frac{\text{FLOPs}}{\text{bytes}} \;>\; \frac{\text{peak FLOP/s}}{\text{peak byte/s}},
\qquad \text{else memory-bound.}
$$

Read this literally. The left side is what your *algorithm* asks for (work per byte). The right side is what the *hardware* offers (work it can do in the time it takes to move one byte). If your algorithm does more math per byte than the hardware's break-even point, the arithmetic units are the bottleneck (**math-bound**) — you are using the chip well. If it does less, the memory pipe is the bottleneck (**memory-bound**) — the arithmetic units are starved and idle, and buying more FLOPs would not help at all.

For a GEMM the arithmetic intensity works out to

$$
\text{AI} \;=\; \frac{2MNK}{2\,(MK + NK + MN)} \;=\; \frac{MNK}{MK + NK + MN},
$$

where the numerator $2MNK$ is the FLOP count and the denominator counts the bytes of the two inputs read plus the output written (times the bytes per element, which cancels into the constant). The key feature: intensity **grows with the matrix size**. Big square matmuls reuse each loaded byte many times; skinny ones barely reuse anything.

### 2.2 The roofline picture and two worked examples

Plot attainable throughput against arithmetic intensity and you get the famous **roofline**: a sloped line (you are on the memory "roof," throughput = bandwidth × intensity) that rises until it hits a flat ceiling (peak compute). The corner where they meet is the **ridge point**.

![Roofline plot on log-log axes. A sloped line rises from the bottom-left (the memory-bandwidth limit) until it meets a flat horizontal ceiling at 125 TFLOP/s (the compute limit); the corner is the ridge point at 139 FLOP/byte. The region left of the ridge is shaded and labeled memory-bound; right of it, math-bound. Three example points are marked: a GEMV/activation at arithmetic intensity under 1 (deep in memory-bound), a skinny GEMM at intensity ~124 (just short of the ridge, still memory-bound), and a large square GEMM at intensity ~2730 (well into math-bound, sitting on the compute ceiling).](./assets/roofline_memory_vs_math.jpg){ width=78% }
*The roofline. Left of the ridge you are starved for data; right of it you are limited by raw compute. (Recreated in our notation with V100-class numbers — 125 TFLOP/s peak, 0.9 TB/s HBM, ridge ≈ 139 FLOP/byte; the two GEMM examples and the framing are from the NVIDIA guides.)*

Concretely, on a V100 (ridge ≈ 139 FLOP/byte):

- A large square **$8192\times8192\times8192$** GEMM has AI ≈ **2730** — far above the ridge, comfortably **math-bound**. It reuses each byte thousands of times; the Tensor Cores run near peak.
- A skinny **$8192\times128\times8192$** GEMM has AI ≈ **124** — *below* the ridge, so it is **memory-bound** despite being a huge matmul. The thin $K=128$ dimension kills reuse.
- A **GEMV** (or an elementwise op like ReLU) has AI **< 1** — hopelessly memory-bound. There is essentially no reuse: you read each number, do one or two FLOPs, and move on.

### 2.3 Why this is the crux of the whole note

Here is the lesson that Parts 4 and 6 both cash in: **if you are memory-bound, adding FLOPs does nothing.** A faster arithmetic unit accelerates a bottleneck you don't have. The only way to go faster is to **move less data** — raise arithmetic intensity by reusing what you have already loaded. This is why memory bandwidth so often matters more than peak FLOPs in practice, why TPUs are built around maximizing reuse, and why FlashAttention wins by *reorganizing data movement* while doing the exact same (indeed, slightly more) arithmetic. Keep this sentence in mind: *the bottleneck for most real workloads is the pipe, not the pump.*

---

## Part 3 — GPU vs CPU: the fundamental difference

You don't need the microarchitecture nitty-gritty to get the one idea that matters. It is a difference of **goal**, and everything else follows.

### 3.1 Latency cores vs throughput cores

A **CPU is built to finish a single thread as fast as possible** — to minimize *latency*. Because most code is a serial chain of dependent steps sprinkled with unpredictable branches, a CPU spends most of its transistors *not* on arithmetic but on making one thread fast: large caches (so data is close), out-of-order execution and branch prediction (so it never stalls waiting to discover what to do next), deep pipelines. It has a handful of these big, clever cores.

A **GPU is built to finish an enormous number of independent threads per second** — to maximize *throughput*. It spends its transistors on thousands of small, simple arithmetic lanes and almost nothing on cleverness per lane. When one group of threads stalls waiting on memory, the GPU doesn't try to avoid the stall (as a CPU would); it just **switches to another group that is ready**, keeping the arithmetic units busy. It hides latency behind sheer parallelism instead of fighting it.

![Two floorplan cartoons side by side. Left, the CPU: a few large boxes labeled "core" plus a big band labeled "large caches + control (out-of-order, branch prediction)" — most of the area is control and cache. Right, the GPU: a dense grid of many small boxes (the arithmetic lanes) filling almost the entire chip, with only a thin band of "small caches / control" at the bottom.](./assets/cpu_vs_gpu_cores.jpg){ width=100% }
*The design split in one picture: a CPU spends area on a few fat latency-optimized cores plus large caches/control; a GPU spends area on a sea of thin throughput-optimized lanes. (Schematic, recreated; a standard illustration in the CUDA / NVIDIA GPU-architecture literature.)*

### 3.2 SIMT, warps, and occupancy

The GPU's simple lanes are ganged together: a **warp** of 32 threads executes the same instruction at the same time on 32 different data elements (the **SIMT**, single-instruction-multiple-thread, model). This is cheap because 32 lanes share one instruction decoder — it is why a GPU can afford so many lanes. The flip side is that heavily branchy code (where threads in a warp want to do different things) runs poorly; matmul, where every lane does the identical multiply-add, is the perfect fit.

**Occupancy** is the term for having enough warps resident on each SM so that whenever some are stalled on memory, others are ready to run. High occupancy is how the "hide latency behind parallelism" trick actually works — you need a deep enough backlog of ready warps to paper over the long trip to HBM.

### 3.3 The memory hierarchy is the constraint

Both machines have a memory hierarchy, but on a GPU it is the thing you spend all your effort managing (recall Part 2). From fast/tiny to slow/huge: **registers → SRAM (on-chip "shared memory"/L1) → L2 cache → HBM (DRAM)**. As you go down, capacity grows by orders of magnitude but bandwidth falls.

![Two horizontal bar charts sharing the same four rows — Registers, SRAM/shared (on-chip), L2 cache, HBM (DRAM). Left chart, capacity on a log scale: registers ~27 MB, SRAM ~20 MB, L2 ~40 MB, HBM ~40 GB — capacity grows sharply going down. Right chart, bandwidth on a log scale: registers ~100 TB/s, SRAM ~19 TB/s, L2 ~5 TB/s, HBM ~1.9 TB/s — bandwidth shrinks going down.](./assets/gpu_memory_hierarchy.jpg){ width=100% }
*The GPU memory hierarchy (A100-class, approximate): each step down multiplies capacity but divides bandwidth. Tiling (Part 1) and FlashAttention (Part 6) are both about keeping working data as high up this pyramid as possible. (Recreated; figures are approximate A100-class values, on-chip SRAM ~19 TB/s and HBM ~1.5–2 TB/s per the FlashAttention paper and NVIDIA guides.)*

The A100 concretely: 108 SMs, a 40 MB L2 cache, and 40–80 GB of HBM at roughly 1.5–2.0 TB/s, versus on-chip SRAM at an estimated ~19 TB/s — a **~10× bandwidth cliff** between on-chip and off-chip. Every performance technique in this note is, at bottom, a strategy for staying on the fast side of that cliff.

---

## Part 4 — TPUs and the systolic array

If the GPU is a general throughput machine, the **TPU** (Google's Tensor Processing Unit) is the answer to a sharper question: *if you already know the workload is almost entirely matrix multiply, how little hardware can you get away with?*

### 4.1 Why build one at all

Google measured that MLPs, CNNs, and LSTMs — all dominated by matmul — made up **95% of their datacenters' neural-network inference demand**. For that workload, most of a CPU's (and much of a GPU's) transistor budget — caches, branch predictors, out-of-order machinery — is *overhead*: silicon and power spent making irregular general-purpose code fast, which a wall of matmuls simply doesn't need. A domain-specific chip can delete all of that and spend the area on multiply-accumulate units instead.

### 4.2 The systolic array: reuse baked into the wiring

The heart of the first TPU is a **systolic array**: a $256\times256$ grid of **65,536 multiply-accumulate (MAC) units**. Instead of each MAC fetching its operands from memory, operands **flow through the grid** — activations stream in from one edge, weights are held in place, and each value that enters is **reused across an entire row or column of MACs** as it propagates. Partial sums accumulate as they march through. The name "systolic" is the analogy to a heartbeat pumping data rhythmically through the array.

![A 4×4 grid of boxes labeled "MAC". Green arrows enter from the left ("activations flow in →"), amber arrows enter from the top ("weights held in place ↓"), and gray arrows exit the bottom ("partial sums accumulate out the bottom"). The picture conveys that each value entering an edge is reused by every MAC along its row or column.](./assets/systolic_array.jpg){ width=62% }
*A systolic array (small 4×4 stand-in for the TPU's 256×256). One value read from memory feeds a whole row or column of multipliers as it flows through — reuse is built into the wiring, not managed in software. (Schematic, recreated; the 256×256 / 65,536-MAC design is from Jouppi et al., "In-Datacenter Performance Analysis of a TPU," 2017.)*

Why this is the right shape: it is Part 2's "move less data" principle turned into physical wiring. In a naive matmul each MAC would need its own operand fetch; in a systolic array **one HBM read feeds many multiplies** because the value physically travels past many MACs. Arithmetic intensity is maximized by the layout itself, so the array can run near peak with a modest memory system — the original TPU pairs it with a large (28 MiB) *software-managed* on-chip buffer instead of a cache hierarchy.

### 4.3 The design philosophy: deterministic and lean

Because the workload is regular, the TPU drops the features that make general chips complex: **no caches, no branch prediction, no out-of-order execution, no speculation.** Execution is **deterministic**, which also makes it easy to hit a tight 99th-percentile latency target for serving. The result is a chip that is small and low-power for its throughput: the 2017 paper reports roughly **15–30× the performance** and **30–80× the performance-per-watt** of contemporary CPUs and GPUs on those inference workloads (with the usual caveats — it was measured against 2015-era parts, and later GPUs with better memory closed much of the gap). The durable lesson is not the specific multiplier but the principle: **specialize the silicon to the workload and you can delete everything the workload doesn't use.**

---

## Part 5 — A100 → H100 → B200: what changed and why

Rather than memorize spec sheets, read each generation as *solving a specific bottleneck* exposed by the previous one. The through-line is the same one from Part 2: feed the arithmetic units faster, and let them chew lower-precision numbers so there are more FLOPs per byte.

![Three grouped bar charts comparing A100 (Ampere), H100 (Hopper), and B200 (Blackwell). HBM bandwidth: 2.0, 3.0, 7.7 TB/s. Dense FP16 Tensor throughput: 312, 990, 2250 TFLOP/s. Memory capacity: 80, 80, 192 GB. Every metric climbs sharply from A100 to B200.](./assets/gpu_generations.jpg){ width=100% }
*Three NVIDIA generations. Bandwidth and compute rise together, and each jump is paired with a new lower-precision format that raises FLOPs-per-byte. (Recreated from vendor-published specs; dense FP16 Tensor throughput shown without sparsity.)*

**A100 (Ampere, 2020) — the baseline.** 108 SMs, HBM2 at ~1.5–2.0 TB/s, third-gen Tensor Cores with TF32/FP16, ~312 dense FP16 Tensor TFLOP/s. This is the machine most of the numbers in Parts 1–3 refer to.

**H100 (Hopper, 2022) — feed the beast, and go to FP8.** 132 SMs and HBM3 at ~3.0 TB/s (the first GPU with HBM3). Its interesting parts are all about *data movement and precision*:

- **Tensor Memory Accelerator (TMA):** a dedicated hardware unit that performs large asynchronous bulk copies between HBM and SRAM. On A100, threads themselves had to compute all the addresses and issue the loads; the TMA offloads that entirely, so the arithmetic threads keep computing instead of babysitting memory transfers — a direct attack on the Part 2 bottleneck.
- **Thread-block clusters + distributed shared memory:** groups of thread blocks on different SMs can be co-scheduled and read each other's SRAM directly, about **7× faster** than routing data through HBM. This lets a tile's working set be shared across SMs without a round-trip to slow memory.
- **Transformer Engine + FP8:** new 8-bit float formats (E4M3 and E5M2) with hardware that dynamically picks FP8 vs 16-bit per layer to hold accuracy. Halving the bytes per number doubles both effective bandwidth and Tensor Core throughput — NVIDIA quotes up to **9× faster training and 30× faster inference** on large language models versus A100.

**B200 (Blackwell, 2025) — bigger still, and go to FP4.** To beat the reticle limit (the largest die you can manufacture), a B200 is **two dies fused into one logical GPU** (208 billion transistors total) linked at 10 TB/s so software sees a single chip. It carries **192 GB of HBM3e at ~7.7 TB/s** and a **second-generation Transformer Engine** that adds **FP4/FP6** with per-block ("micro-tensor") scaling — pushing FLOPs-per-byte even higher for inference. In practice this lands at roughly **~2× the real training throughput** of H100 on large models.

The pattern across all three: **more and faster memory, better hardware for moving data on-chip, and ever-lower-precision number formats.** Two of those three levers are about data movement, and the third (lower precision) is itself a way to move fewer bytes per number. The chips keep confirming Part 2's thesis.

---

## Part 6 — FlashAttention: same math, far less data movement

Now the payoff. **FlashAttention** (Dao, Fu, Ermon, Rudra, Ré, 2022) made the attention layer several times faster *without approximating anything* — the output is bit-for-bit the exact same attention. It did this purely by being **IO-aware**: reorganizing which data lives where so it stops thrashing HBM. It is Part 2's lesson applied to the single most important operation in a transformer.

### 6.1 The problem: attention is memory-bound, not compute-bound

Standard attention, for sequence length $N$ and head dimension $d$, computes scores $S = QK^\top$ (an $N\times N$ matrix), then $P = \mathrm{softmax}(S)$ (also $N\times N$), then output $O = PV$. Here $Q,K,V$ are the query/key/value matrices of shape $N\times d$. The naive implementation **writes the full $N\times N$ matrix $S$ out to HBM, reads it back to apply softmax, writes $P$ back to HBM, and reads it again** to multiply by $V$.

That is a disaster for two reasons. First, memory: storing $N\times N$ is **$O(N^2)$** and blows up quadratically with sequence length. Second — the deeper point — the operation is **memory-bound**: the actual matmuls are cheap relative to the enormous traffic of shuttling that $N\times N$ matrix in and out of slow HBM several times. The GPU's Tensor Cores spend most of their time waiting on the memory pipe. By Part 2, throwing more FLOPs at it would do nothing; you must move less data.

### 6.2 The insight: count HBM traffic, and keep the matrix on-chip

FlashAttention's stated missing principle is exactly ours: attention algorithms should be **IO-aware — account for reads and writes between levels of GPU memory**, not just count FLOPs. The relevant numbers are the ones from Part 3, which the paper spells out for the A100: on-chip **SRAM at ~19 TB/s but only ~192 KB per SM**, versus **HBM at ~1.5–2.0 TB/s with 40 GB**. The whole game is to keep the attention computation up in that 19 TB/s SRAM and **never materialize the $N\times N$ matrix in HBM at all.**

The obstacle is softmax. Softmax over a row needs the whole row (to find the max for numerical stability and to sum the exponentials for normalization), which seems to force you to have the entire row of $S$ at once. FlashAttention breaks that dependency.

### 6.3 Tiling plus an online (streaming) softmax

Split $Q$, $K$, and $V$ into blocks sized to fit in SRAM. For each block of queries, **walk over the blocks of keys/values one at a time**, computing that small $Q_iK_j^\top$ block in SRAM. The trick is to carry a little **running softmax state** and rescale as you go:

- $m$ — the running maximum score seen so far (for numerical stability),
- $\ell$ — the running sum of exponentials $\sum e^{\,s-m}$ (the normalizer),
- $O$ — the running, rescaled output accumulator.

When a new block arrives with a larger max, you rescale the previously accumulated $\ell$ and $O$ by $e^{\,m_\text{old}-m_\text{new}}$ so everything stays on a consistent scale, then fold in the new block. After the last key/value block, $O$ divided by $\ell$ is *exactly* the standard softmax-weighted output — no approximation. (This is the same numerically-stable softmax, computed incrementally; for why softmax needs that max-subtraction and how its normalization works, see [`absmax-mse-vs-softmax-ce.md`](./absmax-mse-vs-softmax-ce.md).)

![Schematic of the attention computation. A dashed N×N grid labeled "the full N×N scores matrix is never stored in HBM," split into a 3×3 grid of blocks. Rows are labeled Q1, Q2, Q3; columns are labeled K1,V1 / K2,V2 / K3,V3. One block (Q2·K2ᵀ) is highlighted as being "in SRAM." An arrow points to a side panel, "running softmax state (kept in SRAM)," listing m = running max, ℓ = running sum of e^(s−m), O = rescaled output. A caption reads: walk over K/V blocks one at a time, update (m, ℓ, O) on the fly — exact softmax, no full matrix.](./assets/flash_attention_tiling.jpg){ width=88% }
*FlashAttention tiles Q/K/V into SRAM-sized blocks and streams the softmax, keeping only the small running statistics $(m,\ell,O)$ — so the $N\times N$ matrix never touches HBM. (Recreated in our notation; cf. Dao et al., 2022, Figure 1.)*

### 6.4 The backward pass: recompute instead of store

Training needs the attention matrix again for gradients. Storing it during the forward pass would reintroduce the $O(N^2)$ HBM cost we just eliminated. So FlashAttention **doesn't store it** — in the backward pass it **recomputes** each attention block on the fly from the saved statistics $(m,\ell)$ and the inputs. This trades extra FLOPs for far less memory traffic, which is a *great* trade precisely because attention is memory-bound (Part 2 again): the recompute FLOPs are nearly free, and we save the expensive HBM round-trips.

### 6.5 The results, and the one-sentence takeaway

By counting HBM accesses instead of FLOPs, FlashAttention needs roughly **9× fewer HBM accesses** than standard attention for typical sizes, and drops memory from **$O(N^2)$ to $O(N)$** in sequence length. That translates to real speedups: about **15% end-to-end on BERT-large** (seq 512), **3× on GPT-2** (seq 1K), and **2.4× on long-range-arena** (seq 1K–4K) — and it made previously-infeasible long-context settings (Path-X at 16K, Path-256 at 64K) trainable at all, purely by fitting in memory.

The takeaway ties the whole note into one bow: **FlashAttention does the exact same math as standard attention — in fact slightly more arithmetic, because of recomputation — yet runs several times faster, because it moves far less data.** That is the entire thesis of this note in a single result. Fast hardware is a pump; performance is almost always limited by the pipe. Whether you are padding a matrix to 128, wiring a systolic array, adding a Tensor Memory Accelerator, or streaming a softmax, the winning move is the same: **keep the data close, and move less of it.**

---

## Part 7 — Kernel fusion and Triton: how you actually write a data-light kernel

FlashAttention (Part 6) is one hand-crafted masterpiece. But its core trick — *do the whole computation in one pass without round-tripping intermediate results through HBM* — is a **general** technique with a name (**kernel fusion**) and, these days, a tool that lets ordinary people write such kernels without becoming CUDA experts (**Triton**). This part is the "how the sausage is made" companion to the rest of the note: the same move — move less data — expressed at the level of the code you actually write.

### 7.1 What a kernel is, and why launching many small ones is slow

A **kernel** is just a function that runs on the GPU. In eager PyTorch, *each* tensor operation is typically its own kernel: one for the multiply, one for the add, one for the activation, one for the layer-norm. A line like `y = sigmoid(x * w + b)` is not one GPU call — it is three, run back to back.

Two costs pile up when you do this, and both are the ghosts of Part 2:

- **HBM round-trips for intermediates.** Each kernel reads its inputs from HBM and writes its output back to HBM. So the temporary `x*w` is written all the way out to slow HBM, only to be immediately read back in by the very next kernel to add `b` — and again for the sigmoid. These intermediate tensors never needed to leave the chip, yet the naive pipeline shuttles every one of them through the slowest level of the hierarchy.
- **Launch overhead.** Every kernel launch carries a fixed CPU→GPU dispatch cost. For big matmuls this is negligible, but for the swarm of tiny elementwise ops in a real model it adds up, and it hurts most exactly when the tensors are small.

And notice *which* operations dominate here: activations, bias-adds, scaling, dropout, normalization. These are **elementwise** (or near-elementwise) ops with arithmetic intensity around 1 — one or two FLOPs per element loaded. By the roofline of Part 2 they are deeply **memory-bound**: the arithmetic is trivial, so runtime is set almost entirely by the HBM traffic. Making the GPU's FLOPs faster does nothing; the fix has to be *fewer bytes moved*.

### 7.2 Fusion: do the whole chain in one pass, staying on-chip

The fix is exactly FlashAttention's, generalized: **merge the chain of operations into a single kernel.** Load each input from HBM *once*, carry out the entire sequence of arithmetic while the values sit in fast registers / SRAM, and write only the *final* result back. Every intermediate stays on-chip and never touches HBM.

Take the same `sigmoid(x*w + b)`. Unfused, it is 3 kernels and roughly **8 memory operations** (read `x`, `w`, write the product; read it back with `b`, write the sum; read it back, write the sigmoid). Fused, it is **1 kernel and 4 memory operations** (read the three inputs once, write one result) — about **half the memory traffic and a third of the launches**, for the exact same math. That is the entire win, and it is Part 2 in miniature.

![Two panels contrasting unfused vs fused execution of sigmoid(x*w + b). Left "UNFUSED — 3 kernels": three separate kernel boxes (multiply, add, sigmoid) stacked, each with a red arrow reading its inputs down from an HBM bar and a red arrow writing its output back up to HBM, so the intermediates x*w and (x*w+b) make a full round trip; tallied count "~8 HBM transfers, 3 launches". Right "FUSED — 1 kernel": a single tall kernel box that reads x, w, b once from HBM (one set of blue down-arrows), runs multiply→add→sigmoid entirely inside a shaded "registers / SRAM (on-chip)" region with no HBM arrows between the steps, and writes one result back up; tallied "~4 HBM transfers, 1 launch". A caption reads: same math, half the memory traffic — intermediates never leave the chip.](./assets/kernel_fusion.jpg){ width=100% }
*Kernel fusion collapses a chain of memory-bound elementwise ops into one pass: inputs are read once, the whole computation happens in registers, and only the final result is written back — so the intermediates never round-trip through HBM. FlashAttention (Part 6) is this idea pushed to its limit on the attention block. (Recreated in our notation; cf. the PyTorch "Why is torch.compile so fast: kernel fusion" blog.)*

FlashAttention is simply this at the extreme: the *whole* attention block — $QK^\top$, softmax, and the $\times V$ — is fused into one kernel so the $N\times N$ scores matrix never round-trips. Fusion comes in a few recognizable flavors: **vertical** (a chain where each op's output feeds the next, as above), **horizontal** (several independent ops on the same input, done together), and **epilogue** fusion (a GEMM with its bias/activation tacked onto the end so the matmul's output is post-processed *before* it is written out).

### 7.3 Who does the fusing: `torch.compile` and Inductor

You could hand-write every fused kernel, but you rarely need to. `torch.compile()` traces your model into a graph, and its **Inductor** backend automatically spots fusible groups of operations and generates a single fused kernel for each. Crucially, the code it emits is not CUDA — its default target is **Triton**. So the everyday path to fusion is: write ordinary PyTorch, wrap it in `torch.compile`, and let the compiler collapse your elementwise chains into fused Triton kernels for you. (You can inspect what it generated with `TORCH_LOGS="output_code"`.)

### 7.4 Triton: writing GPU kernels in Python, at the *tile* level

When automatic fusion isn't enough — a novel attention variant, a custom activation, a fused normalization — you write the kernel yourself. Historically that meant **CUDA C++**: maximal control, but you personally manage thread indices, shared-memory bank conflicts, memory-coalescing patterns, and warp synchronization. It is expert work and slow to get right.

**Triton** (open-sourced by OpenAI in 2021) changes the abstraction level. You write **Python**, and — this is the key idea — you program at the level of a **tile** (a block of the tensor), not a single thread. Your code says *load this tile from HBM (`tl.load`), do arithmetic on it, store the result back (`tl.store`)*. The compiler then handles everything below that line: how threads are laid out within the block, how shared memory is allocated, how loads are coalesced, when warps synchronize, and which hardware instructions to emit — including automatically firing the asynchronous **TMA** loads on Hopper/Blackwell that we met in Part 5, with no extra code. On top of that, `@triton.autotune` will benchmark a menu of tile sizes and warp counts on the first run and cache whichever is fastest.

The mental model is a clean division of labor: **in CUDA you place every thread; in Triton you place every tile and the compiler places the threads.** You trade a sliver of peak control for a large gain in productivity, and for the great majority of kernels Triton lands within striking distance of hand-tuned CUDA.

The canonical teaching example is, fittingly, a **fused softmax**. PyTorch's `F.softmax` launches separate kernels for the exponential, the sum-reduction, and the divide, each round-tripping through HBM. A Triton kernel fuses all three into one load-compute-store pass; on an H100 this roughly doubles the achieved bandwidth (~350 → ~820 GB/s, about **2.3×**) purely by halving HBM traffic — the softmax version of the same lesson. This is why Triton now sits under so much of the stack: it is the default codegen of `torch.compile`, and FlashAttention variants, vLLM's operators (PagedAttention, RoPE, RMSNorm — see Part 8), and libraries like Liger-Kernel are all written in it.

### 7.5 The takeaway

Kernel fusion is the whole note's thesis restated at the level of the code you write: **the pump is fast, the pipe is the limit, so chain your operations and touch HBM as few times as you can.** FlashAttention was the heroic hand-built instance; fusion is the general principle; `torch.compile` applies it automatically; and Triton is how you write a custom one without hand-placing a single thread.

---

## Part 8 — The KV cache: prefill, decode, and why inference is memory-bound

FlashAttention (Part 6) made *training* attention fast. But when you actually *use* a trained model — generating text one token at a time — a different inefficiency dominates, and the fix, the **KV cache**, is again a pure data-movement trick. This part is where the note's thesis finally lands on the thing you interact with every day: an LLM answering a prompt.

### 7.1 The redundancy a cache removes

A decoder LLM generates **autoregressively**: to produce token $t+1$ it runs a full forward pass over the sequence so far, and each attention layer needs the keys and values of *every previous token*. Recall from Part 6 that attention turns the current token's query $Q$ against the keys $K$ and values $V$ of all tokens: $O = \mathrm{softmax}(QK^\top)V$, where $K$ and $V$ are the per-token projections $K_i = x_i W_K$, $V_i = x_i W_V$ (here $x_i$ is token $i$'s hidden state and $W_K, W_V$ are the frozen key/value projection weights).

The naive loop recomputes $K$ and $V$ for **all** tokens at **every** step. That is almost entirely wasted work, and here is the exact reason why: the projection weights $W_K, W_V$ are **frozen** at inference, and attention is **causal** (a token only attends to itself and earlier tokens). So $K_i$ and $V_i$ for an already-seen token $i$ are *bit-for-bit identical* on every future step — token $i$'s hidden state doesn't change when we append token $i+1$. Recomputing them is like re-deriving the same number over and over. Concretely, extending a 5-token sequence to 6 tokens re-derives about $25/36 \approx 69\%$ of the attention work it already did.

![Two rows of key/value slots for the tokens "The cat sat on" plus a new token "the". Left, WITHOUT a cache: all five columns are shaded red as "recomputed from scratch" every step, even though the first four are unchanged. Right, WITH a cache: the first four columns are shaded blue and bracketed as "read from cache (computed once)", and only the fifth column (the new token) is shaded amber and labeled "compute & append". Title notes that Q is not cached because it is used once then discarded.](./assets/kv_cache_redundancy.jpg){ width=100% }
*The KV cache eliminates redundant recomputation: since frozen weights + causal masking make each past $K_i,V_i$ identical across steps, we compute them once and reuse. (Recreated in our notation; cf. the HuggingFace "KV Cache from Scratch in nanoVLM" and "KV Cache basics" blogs.)*

### 7.2 What gets cached — and why only $K$ and $V$, not $Q$

The fix is to **store** the keys and values as we go. We keep a per-layer cache of $K$ and $V$, each of shape `(batch, num_heads, seq_len, head_dim)`, and on each decode step we compute $K,V$ for **only the new token** and append them:

$$
K \leftarrow \text{concat}(K_{\text{cache}},\, k_{\text{new}}), \qquad
V \leftarrow \text{concat}(V_{\text{cache}},\, v_{\text{new}}).
$$

Then attention for the new token runs its single query against the full cached $K,V$.

Why cache $K$ and $V$ but **not** the queries $Q$? Because their roles across time are opposite. A key/value pair for position $i$ is **reused by every future token** — $K_0,V_0$ are read when token 1 attends back to token 0, again when token 2 does, and so on forever. A query, by contrast, is **used exactly once**: token $t$'s query attends to the past, produces token $t$'s output, and is then never needed again. There is nothing to reuse, so there is nothing worth caching. Caching $Q$ would store data no future step ever reads.

### 7.3 Prefill vs decode — two phases with opposite bottlenecks

This splits generation into two phases with strikingly different hardware behavior — and this is exactly the Part 2 roofline showing up in production inference.

**Prefill** processes the entire input prompt in **one parallel forward pass**, computing $K,V$ for all prompt tokens at once and populating the cache. Because it multiplies a whole matrix of token states by the weights, prefill is a big, dense **GEMM with high arithmetic intensity — it is compute/math-bound** (right of the ridge in Part 2). The GPU's Tensor Cores run near peak; this is the phase that does "real" arithmetic.

**Decode** then emits tokens **one at a time**, each step running a *single* new token's query against the entire cached $K,V$. That is a matrix–**vector** product — a GEMV — with arithmetic intensity below 1. To generate one token, the GPU must **stream the entire KV cache (and the model weights) out of HBM** and do only a tiny amount of math on it. So decode is squarely **memory-bandwidth-bound**: its speed is set by how fast you can read the cache from HBM, *not* by how many FLOPs the chip can do.

![A timeline. On the left, one wide green block labeled "PREFILL — whole prompt in parallel", tagged "big GEMM → compute-bound (high arithmetic intensity)" and bracketed "once". On the right, a run of seven thin red blocks t1..t7, each an arrow reading from a blue bar underneath labeled "KV cache in HBM (grows one token per step)", tagged "one token at a time, each reads the WHOLE cache → memory-bound (AI < 1)" and bracketed "repeated for every generated token".](./assets/prefill_vs_decode.jpg){ width=100% }
*Prefill runs once and is compute-bound; decode runs per token and is memory-bound. This is the roofline of Part 2 dictating the economics of LLM serving. (Recreated in our notation; cf. the HuggingFace KV-cache blogs.)*

The cache turns per-step cost from re-deriving the whole history ($O(n^2)$ work across a generation) into just appending and reading ($O(n)$), which in the nanoVLM walkthrough is already a **~38% generation speedup** on a small model — and far more on long sequences.

### 7.4 The price of the cache — and how people shrink it

Caching trades compute for **memory**, and that memory is substantial. The cache size is

$$
\text{cache bytes} \;=\; 2 \,\cdot\, n_{\text{layers}} \,\cdot\, n_{\text{heads}} \,\cdot\, d_{\text{head}} \,\cdot\, \text{seq\_len} \,\cdot\, \text{batch} \,\cdot\, \text{bytes/elem},
$$

where the leading $2$ counts $K$ **and** $V$, $n_{\text{layers}}$ is the number of transformer layers, $n_{\text{heads}}\cdot d_{\text{head}}$ is the model width per token, and the rest scale it by how many tokens and requests you hold. The thing to notice is that it grows **linearly with sequence length and batch size** — every extra token of context, and every extra concurrent request, costs a fixed slice of HBM. For a LLaMA-7B-class model this works out to roughly **0.5 MB per token**, so a single 2K-token request eats on the order of **~1 GB** of HBM just for its cache (larger models or fp32 push this to several GB).

This is why the trade, while essentially **always worth it** for speed, creates the real bottleneck in LLM serving: the KV cache — not the model weights — is often what caps how long a context you can serve and how many requests you can batch, and (per 7.3) its size is exactly the bytes you must stream every decode step. So the whole game becomes **shrink the cache you have to move**:

- **Multi-Query / Grouped-Query Attention (MQA / GQA):** let many attention heads *share* one set of $K,V$ instead of each head having its own. This cuts the cache (and the bytes streamed per step) by the sharing factor with little quality loss — a direct attack on the memory-bound decode bottleneck.
- **PagedAttention (vLLM):** manage the cache in fixed-size *pages* like virtual memory, so requests don't each reserve a giant contiguous block. This slashes fragmentation and lets you pack far more concurrent sequences into the same HBM.

Both are the same move we have seen all note long, now applied to inference: **the pump is fast; the pipe is the limit — so move fewer bytes.** The KV cache removes redundant *compute*; MQA/GQA and PagedAttention then shrink the *memory traffic and footprint* that the cache itself creates.

---

## Part 9 — The math of compute- vs memory-bound: arithmetic intensity, latency, and throughput

Parts 2 and 8 told the story in words: big matmuls are compute-bound, inference decode is memory-bound. This part does the **arithmetic** that proves it. We will (a) rederive arithmetic intensity for a single matmul and pin down the exact batch size at which an H100 flips from memory- to compute-bound; (b) do the same accounting for the MLP and attention layers of a transformer, and see *why* prefill is compute-bound while generation is stuck below the line no matter what; and (c) turn that into concrete **latency** and **throughput** numbers, and the batch-size tradeoff that governs how LLMs are actually served. The whole part follows the Stanford CS336 lecture-10 accounting, rewritten in our notation.

### 9.1 Setup and notation for this part

A handful of inference-specific symbols, written with descriptive subscripts (matching Part 8's style). The last column gives the terse single letters the CS336 source uses, so you can cross-reference the lecture code if you want — but you never need to memorize those.

| Symbol | Meaning | CS336 |
| --- | --- | --- |
| $n_\text{batch}$ | Batch size. In training/prefill, examples in the batch; in **generation, the number of concurrent requests** served together. | $B$ |
| $n_\text{ctx}$ | Number of **context** tokens already present — the prompt during prefill, the cached history during generation (these are the key/value positions). | $S$ |
| $n_\text{new}$ | Number of tokens processed **in this pass** (the query positions). Prefill sees the whole prompt at once ($n_\text{new}=n_\text{ctx}$); generation emits one token at a time ($n_\text{new}=1$). | $T$ |
| $d_\text{model}$ | Model (hidden) dimension. | $D$ |
| $d_\text{ff}$ | MLP inner (feed-forward) dimension, typically $\sim 3$–$4\times d_\text{model}$. | $F$ |
| $n_\text{layers}$ | Number of transformer layers (same symbol as Part 8). | $L$ |
| $n_\text{qheads}$ | Number of **query** heads. | $N$ |
| $n_\text{kvheads}$ | Number of **key/value** heads (Part 8's $n_\text{heads}$ — under MQA/GQA, $n_\text{kvheads}<n_\text{qheads}$). | $K$ |
| $d_\text{head}$ | Head dimension (same as Part 8). Model width $n_\text{qheads}\,d_\text{head}\approx d_\text{model}$. | $H$ |
| $n_\text{vocab}$ | Vocabulary size. | $V$ |

Everything is **bf16** at inference, so **2 bytes per element** — that stray factor of 2 in every byte count below is just this. As in Part 2, an operation is compute-bound when its AI exceeds the hardware's ops:byte ratio, and memory-bound when it falls short.

### 9.2 Warm-up: the arithmetic intensity of one matrix multiply

Take the workhorse of an MLP: multiply activations $X$ (shape $n_\text{batch}\times d_\text{model}$) by a weight $W$ (shape $d_\text{model}\times d_\text{ff}$). Account for every FLOP and every byte crossing HBM, exactly as Part 2 taught:

- **Read $X$:** $2\,n_\text{batch}\,d_\text{model}$ bytes. **Read $W$:** $2\,d_\text{model}\,d_\text{ff}$ bytes. **Write** the result $Y$ ($n_\text{batch}\times d_\text{ff}$): $2\,n_\text{batch}\,d_\text{ff}$ bytes.
- **Compute** $Y = XW$: $2\,n_\text{batch}\,d_\text{model}\,d_\text{ff}$ FLOPs (one multiply + one add per inner-product term).

So the arithmetic intensity is

$$
\mathrm{AI} \;=\; \frac{2\,n_\text{batch}\,d_\text{model}\,d_\text{ff}}{2\,(n_\text{batch}\,d_\text{model} + d_\text{model}\,d_\text{ff} + n_\text{batch}\,d_\text{ff})} \;=\; \frac{n_\text{batch}\,d_\text{model}\,d_\text{ff}}{n_\text{batch}\,d_\text{model} + d_\text{model}\,d_\text{ff} + n_\text{batch}\,d_\text{ff}}.
$$

This is the same formula as Part 2's GEMM intensity, just named with our inference symbols. The important move is what happens **when $n_\text{batch}$ is much smaller than $d_\text{model}$ and $d_\text{ff}$** — precisely the inference regime, where the weight matrices are huge but you are pushing only a few tokens through them. Then the $d_\text{model}\,d_\text{ff}$ term dominates the denominator, the $d_\text{model}$ and $d_\text{ff}$ cancel against the numerator, and

$$
\mathrm{AI} \;\xrightarrow{\;n_\text{batch}\,\ll\, d_\text{model},\,d_\text{ff}\;}\; n_\text{batch}.
$$

The intensity of a skinny matmul is essentially **just its batch size**. Now compare against the hardware. For an H100, peak bf16 is $989\times10^{12}$ FLOP/s and HBM bandwidth is $3.35\times10^{12}$ byte/s, so its ops:byte ratio is

$$
\frac{989\times10^{12}\ \text{FLOP/s}}{3.35\times10^{12}\ \text{byte/s}} \;\approx\; 295\ \text{FLOP/byte}.
$$

**Why do we compare AI to this ratio — and what are the units?** First, the units, because they are the whole trick. Arithmetic intensity is $\mathrm{AI}=\dfrac{\text{FLOPs}}{\text{bytes}}$, so its units are **FLOP per byte**: how much math you do for each byte you fetch. The ops:byte ratio is $\dfrac{\text{peak FLOP/s}}{\text{peak byte/s}}$ — and the "per second" cancels top and bottom, leaving **FLOP per byte** as well. So the two numbers live in the *same units* and can be compared directly. AI (FLOP/byte) is a property of **your algorithm**; the ops:byte ratio (FLOP/byte) is a property of **the chip**.

The comparison is really a race between two clocks. Any operation spends time on two things, which a GPU runs *in parallel*: doing the arithmetic, and moving the data. Their durations are

$$
t_\text{compute} = \frac{\text{FLOPs}}{\text{peak FLOP/s}}, \qquad
t_\text{memory} = \frac{\text{bytes}}{\text{bandwidth}}.
$$

Because they overlap, the op finishes when the **slower** one finishes. It is **compute-bound** exactly when the compute clock is the slower one, $t_\text{compute} > t_\text{memory}$:

$$
\frac{\text{FLOPs}}{\text{peak FLOP/s}} \;>\; \frac{\text{bytes}}{\text{bandwidth}}
\;\;\Longleftrightarrow\;\;
\underbrace{\frac{\text{FLOPs}}{\text{bytes}}}_{\mathrm{AI}} \;>\; \underbrace{\frac{\text{peak FLOP/s}}{\text{bandwidth}}}_{\text{ops:byte}}.
$$

That rearrangement — multiply both sides by $\text{bandwidth}$ and divide by $\text{bytes}$ — is the *entire* reason we compare AI to the ops:byte ratio. If your work-per-byte beats the most work-per-byte the chip can sustain while the pipe is saturated, the arithmetic units are the bottleneck (compute-bound, good); if not, they sit idle waiting on HBM (memory-bound).

Now the specific claim. For the H100 the break-even is $295$ FLOP/byte, and our skinny matmul has $\mathrm{AI}\approx n_\text{batch}$. Substituting into $\mathrm{AI} > \text{ops:byte}$ gives $n_\text{batch} > 295$ — **that is all "compute-bound only when $n_\text{batch}>295$" means**: you must push more than 295 tokens through the weight matrix before the math takes longer than fetching the weights. The extreme case is $n_\text{batch}=1$ — a matrix–**vector** product (exactly one token) — where $\mathrm{AI}=1$: you stream the entire $d_\text{model}\times d_\text{ff}$ weight matrix from HBM to do a mere $2\,d_\text{model}\,d_\text{ff}$ FLOPs, using the chip at well under 1% of peak. **That is the inference workload**: thin tensors, tiny intensity, hopelessly memory-bound. This is the roofline's GEMV example from Part 2, now with a number attached.

### 9.3 The arithmetic intensity of a whole transformer layer

Now the real thing. We split each layer into its MLP and its attention, and analyze both in the abstract $(n_\text{ctx}, n_\text{new})$ form, then specialize to **prefill** ($n_\text{new}=n_\text{ctx}$) and **generation** ($n_\text{new}=1$).

**MLP layer.** A gated MLP (Part 8's width per token, with up/gate/down projections) reads the input $X$ ($n_\text{batch}\times n_\text{new}\times d_\text{model}$) and the three weight matrices, does three matmuls, and writes the intermediates and output:

$$
\text{FLOPs} = 6\,n_\text{batch}\,n_\text{new}\,d_\text{model}\,d_\text{ff}, \qquad
\text{bytes} = \underbrace{4\,n_\text{batch}\,n_\text{new}\,d_\text{model}}_{\text{read }X,\text{ write }Y} + \underbrace{4\,n_\text{batch}\,n_\text{new}\,d_\text{ff}}_{\text{write }U,G} + \underbrace{6\,d_\text{model}\,d_\text{ff}}_{\text{read 3 weights}}.
$$

The FLOP count is three matmuls of $2\,n_\text{batch}\,n_\text{new}\,d_\text{model}\,d_\text{ff}$ each; the byte count is the activations in and out plus the weights read once. Taking the same limit as before — the number of *tokens in flight* $n_\text{batch}\,n_\text{new}$ is much smaller than $d_\text{model},d_\text{ff}$ — the weight term $6\,d_\text{model}\,d_\text{ff}$ dominates the bytes and

$$
\mathrm{AI}_\text{MLP} \;\xrightarrow{\;n_\text{batch}\,n_\text{new}\,\ll\,d_\text{model},\,d_\text{ff}\;}\; n_\text{batch}\,n_\text{new}.
$$

Just the matmul result again, with the batch enlarged to "tokens in flight" $n_\text{batch}\,n_\text{new}$. So the MLP is compute-bound whenever $n_\text{batch}\,n_\text{new}$ clears $\sim 295$: **easy in prefill** (the whole prompt makes $n_\text{new}$ large), and **workable in generation** ($n_\text{new}=1$) *provided you batch enough concurrent requests* to push $n_\text{batch}$ up.

**Attention layer.** With FlashAttention (Part 6) the full score matrix (size $n_\text{new}\times n_\text{ctx}$) never touches HBM, so we only move $Q,K,V$ and the output. Reading $Q$ ($n_\text{batch}\times n_\text{new}\times d_\text{model}$), $K,V$ ($n_\text{batch}\times n_\text{ctx}\times d_\text{model}$ each), computing scores then the value-weighted sum, and writing the output:

$$
\text{FLOPs} = 4\,n_\text{batch}\,n_\text{ctx}\,n_\text{new}\,d_\text{model}, \qquad
\text{bytes} = \underbrace{4\,n_\text{batch}\,n_\text{new}\,d_\text{model}}_{Q,\text{ output}} + \underbrace{4\,n_\text{batch}\,n_\text{ctx}\,d_\text{model}}_{K,\,V}.
$$

Here comes the crucial cancellation. Both FLOPs and bytes carry the **same** factor $n_\text{batch}\,d_\text{model}$, so it drops out entirely:

$$
\mathrm{AI}_\text{attn} \;=\; \frac{4\,n_\text{batch}\,n_\text{ctx}\,n_\text{new}\,d_\text{model}}{4\,n_\text{batch}\,d_\text{model}\,(n_\text{ctx}+n_\text{new})} \;=\; \frac{n_\text{ctx}\,n_\text{new}}{n_\text{ctx}+n_\text{new}}.
$$

**The attention intensity does not depend on $n_\text{batch}$ at all.** Specializing:

- **Prefill** ($n_\text{new}=n_\text{ctx}$): $\mathrm{AI} = \dfrac{n_\text{ctx}^2}{2\,n_\text{ctx}} = \dfrac{n_\text{ctx}}{2}$. Long prompts give high intensity — compute-bound and healthy. (Notice the batch dimension is *absent*; long sequences, not big batches, are what save prefill attention.)
- **Generation** ($n_\text{new}=1$): $\mathrm{AI} = \dfrac{n_\text{ctx}}{n_\text{ctx}+1} < 1$. Below the line for *any* context length, and — since $n_\text{batch}$ cancels — **no amount of batching can lift it.** This is the fundamental bottleneck of transformer inference.

Why does batching rescue the MLP but not attention? Because of *what is shared*. In the MLP, **every sequence hits the same weights**: the $6\,d_\text{model}\,d_\text{ff}$ bytes of weights are read once and amortized over all $n_\text{batch}$ sequences, so a bigger $n_\text{batch}$ buys more FLOPs per weight-byte loaded — intensity climbs. In attention, **every sequence carries its own KV cache** (Part 8): the $K,V$ bytes scale with $n_\text{batch}$ in lockstep with the FLOPs, so there is nothing to amortize and $n_\text{batch}$ cancels. Serving more sequences just means doing more independent, low-intensity matrix–vector products side by side.

### 9.4 From intensity to latency and throughput

Because generation is memory-bound, its speed is set purely by **how many bytes must be streamed from HBM per token** — the actual arithmetic hides under the memory time. So we can estimate real performance just by counting bytes. Two quantities matter, both per model:

$$
\text{parameters}: \;\; P = \underbrace{2\,n_\text{vocab}\,d_\text{model}}_{\text{embed + unembed}} + \underbrace{3\,d_\text{model}\,d_\text{ff}\,n_\text{layers}}_{\text{MLP up/gate/down}} + \underbrace{(2\,d_\text{model}\,n_\text{qheads}\,d_\text{head} + 2\,d_\text{model}\,n_\text{kvheads}\,d_\text{head})\,n_\text{layers}}_{\substack{\text{attn: }Q\text{ \& output proj}\\ +\;K\text{ \& }V\text{ proj}}}
$$

Each term is a matrix's element count: the two embedding tables ($n_\text{vocab}\times d_\text{model}$), three MLP matrices ($d_\text{model}\times d_\text{ff}$) per layer, and per layer the query+output projections ($d_\text{model}\times n_\text{qheads}d_\text{head}$ each) plus the key+value projections ($d_\text{model}\times n_\text{kvheads}d_\text{head}$ each, smaller under GQA). The **memory footprint** in bytes is the parameters (bf16) plus one KV cache per concurrent sequence:

$$
M(n_\text{batch}) \;=\; \underbrace{2P}_{\text{weights}} \;+\; n_\text{batch}\cdot\underbrace{2\cdot 2\cdot n_\text{ctx}\, n_\text{kvheads}\, d_\text{head}\, n_\text{layers}}_{\text{KV cache per sequence}},
$$

where the KV-cache term is Part 8's formula: a factor $2$ for storing both $K$ and $V$, another $2$ for bytes/element, times $n_\text{ctx}$ tokens $\times\,(n_\text{kvheads}d_\text{head})$ width $\times\,n_\text{layers}$ layers. Since each generated token must read essentially this whole footprint once from HBM,

$$
\text{latency} \;=\; \frac{M(n_\text{batch})}{\text{memory bandwidth}} \quad(\text{seconds per token}),
\qquad
\text{throughput} \;=\; \frac{n_\text{batch}}{\text{latency}} \quad(\text{tokens per second}),
$$

the throughput being $n_\text{batch}$ because the $n_\text{batch}$ sequences each emit one token per latency period.

**The batch-size tradeoff.** Watch what $n_\text{batch}$ does to the two terms of $M(n_\text{batch})$. The weight term $2P$ is **fixed**; the KV term grows **linearly in $n_\text{batch}$**. So:

- **Larger $n_\text{batch}$ → worse latency**, because each token must now stream a bigger ($O(n_\text{batch})$) pile of KV cache.
- **Larger $n_\text{batch}$ → better throughput**, because the fixed cost of reading the weights $2P$ is amortized over more sequences — until the KV term dominates and gains flatten.

![Twin-axis line plot for Llama-2-13B on one H100 (context length 1024, bf16). The x-axis is the batch size (number of concurrent requests) on a log scale from 1 to 256. A rising blue curve (left axis) shows latency per token in milliseconds climbing from about 8 ms at batch 1 to tens of ms at large batch; a rising green curve (right axis) shows throughput in tokens/second climbing steeply at first then flattening. A vertical dashed line near batch 64 marks where total memory reaches the 80 GB HBM capacity; the region to its right is shaded red and labeled "exceeds 80 GB — does not fit". Markers annotate batch 1, 64, 256 with their (latency, throughput) values.](./assets/latency_throughput_batch.jpg){ width=90% }
*Increasing the batch size trades latency for throughput, and eventually runs out of HBM. Weights ($\approx$ 26 GB in bf16) are a fixed cost amortized over the batch; the KV cache grows linearly with $n_\text{batch}$ and dominates once the batch is large — pushing past the 80 GB card and giving diminishing throughput. (Recreated in our notation from the CS336 lecture-10 example; numbers are approximate.)*

Concretely, for a Llama-2-13B-class model ($\approx$ 13B params $\Rightarrow$ $\approx$ 26 GB of bf16 weights) on an 80 GB H100 with a 1K-token context: $n_\text{batch}=1$ gives the best latency ($\sim$8 ms/token) but poor throughput; raising to $n_\text{batch}=64$ multiplies throughput by more than $20\times$ at the cost of $\sim3\times$ latency; and $n_\text{batch}=256$ blows past the 80 GB card entirely, so it does not even fit — and by then throughput gains are already flattening. This is why serving systems pick a batch size deliberately: **small batches for latency-sensitive interactive use, large batches for throughput-sensitive bulk use.** Two more practical corollaries fall straight out:

- **Cheap parallelism vs. hard parallelism.** Launch $M$ independent *copies* of the model on $M$ devices and throughput scales by $M$ with latency unchanged — trivial but $M\times$ the hardware. To go bigger than one device *per copy* you must **shard** the weights and KV cache across devices (tensor/pipeline parallelism, Part 5's NVLink territory), which is the harder engineering.
- **Time-to-first-token (TTFT) is essentially prefill time.** Prefill is the compute-bound phase (9.3), so TTFT is governed by it — favor **smaller** batches during prefill for snappy first tokens, then **larger** batches during generation to maximize sustained throughput. This asymmetry is why prefill and decode are often scheduled and even served separately.

The whole part is one number chased through the stack: the arithmetic intensity of a thin matmul is just its batch, attention's generation intensity is stuck below 1 because $n_\text{batch}$ cancels, and so inference latency is set by the bytes you must stream — leaving batch size as the one knob that trades latency for throughput until the KV cache eats your HBM. It is Part 2's roofline, all the way down to the serving bill.

---

## Sources

- **NVIDIA — Matrix Multiplication Background User's Guide.** The definitive explanation of tiling, tile/wave quantization, Tensor Core alignment, and arithmetic intensity for GEMMs. <https://docs.nvidia.com/deeplearning/performance/dl-performance-matrix-multiplication/>
- **NVIDIA — GPU Performance Background User's Guide.** SMs, the SIMT/warp model, the memory hierarchy, and the math-bound vs memory-bound / ops:byte framing. <https://docs.nvidia.com/deeplearning/performance/dl-performance-gpu-background/>
- **NVIDIA — Hopper Architecture In-Depth** (developer blog). TMA, thread-block clusters and distributed shared memory, the Transformer Engine and FP8, HBM3, NVLink. <https://developer.nvidia.com/blog/nvidia-hopper-architecture-in-depth/>
- **Jouppi et al., "In-Datacenter Performance Analysis of a Tensor Processing Unit" (ISCA 2017).** The systolic array, the deterministic design philosophy, and the CPU/GPU/TPU comparison. arXiv:1704.04760 — <https://arxiv.org/abs/1704.04760>
- **Dao, Fu, Ermon, Rudra, Ré, "FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness" (NeurIPS 2022).** IO-awareness, tiling + online softmax, recomputation, and the SRAM/HBM numbers. arXiv:2205.14135 — <https://arxiv.org/abs/2205.14135>
- **NVIDIA Blackwell (B100/B200) architecture** — vendor materials and secondary summaries for the dual-die design, HBM3e, and the second-gen Transformer Engine / FP4. <https://www.nvidia.com/en-us/data-center/technologies/blackwell-architecture/>
- **OpenAI — "Introducing Triton: Open-source GPU programming for neural networks" (2021).** The tile-level (block, not thread) programming model, what the compiler automates, and the fused-softmax / matmul examples. <https://openai.com/index/triton/>
- **PyTorch — "Why is `torch.compile` so fast: kernel fusion."** Vertical/horizontal/epilogue fusion, the memory-traffic and launch-overhead argument, the worked `x*w+b→sigmoid` example, and Inductor's use of Triton as its codegen backend. <https://pytorch.org/blog/why-is-pytorch-compile-so-fast-kernel-fusion/>
- **Spheron — "OpenAI Triton Kernel Development on GPU Cloud" (2026 guide).** Practical Triton programming model, autotuning, TMA on Hopper/Blackwell, and the H100 fused-softmax bandwidth numbers. <https://www.spheron.network/blog/openai-triton-kernel-gpu-cloud-2026/>
- **HuggingFace — "KV Cache from Scratch in nanoVLM."** Why the cache exists, prefill vs decode, and a from-scratch PyTorch implementation with the generation loop. <https://huggingface.co/blog/kv-cache>
- **HuggingFace — "The KV Cache: How It Eliminates Redundancy" (atharv6f).** Concise conceptual take on what gets cached, why only K/V, and the memory-vs-compute tradeoff. <https://huggingface.co/blog/atharv6f/kv-cache-basics>

- **Stanford CS336, "Language Modeling from Scratch" — Lecture 10 (systems / inference).** The FLOP-and-byte accounting for a matmul, the MLP and attention layers, and the transformer performance stats (parameters, memory, latency, throughput) that Part 9 follows. <https://github.com/stanford-cs336/lectures/blob/main/lecture_10.py>
- **"How to Scale Your Model" (JAX ML scaling book) — Inference.** The naive-vs-cached inference picture and the prefill/generation, latency/throughput framing underlying Part 9. <https://jax-ml.github.io/scaling-book/inference/>

**Companion notes in this repo:** [`absmax-mse-vs-softmax-ce.md`](./absmax-mse-vs-softmax-ce.md) for softmax mechanics (needed in Part 6) and [`lora.md`](./lora.md) for the "a linear layer is a matrix multiply" picture.
