# Floating-Point Precision and Quantization: FP32, BF16, FP8, MXFP8, INT8, and the 4-/2-bit World

Modern ML runs on a confusing zoo of number formats — FP32, FP16, BF16, FP8 (in two flavours), microscaling MXFP8, INT8, and increasingly 4-bit and even **2-bit** weights. This note builds intuition for what these actually *are*, how a pile of bits turns into a number a human can read, what you pay and gain in **memory, speed, and accuracy** when you shrink them, why **microscaling** had to be invented, and how something as extreme as 2-bit inference is even possible.

The single idea underneath everything: **fewer bits per number means less memory and (on the right hardware) more arithmetic per second — at the cost of representing each number less faithfully.** The entire field is about giving up bits without giving up model quality. Quantization is closely tied to how models are trained and served, so this pairs well with [`training_llms/008_ScalingLaws.md`](./training_llms/008_ScalingLaws.md) (why models got huge in the first place) and [`gpu-tpu-matmul-flashattention.md`](./gpu-tpu-matmul-flashattention.md) (why memory bandwidth, not FLOPs, often decides inference speed).

---

## Table of Contents

- [Setup and Notation](#setup-and-notation)
- [1. Why precision is a first-class decision](#1-why-precision-is-a-first-class-decision)
- [2. How a handful of bits becomes a number](#2-how-a-handful-of-bits-becomes-a-number)
- [3. The everyday formats: FP32, FP16, BF16](#3-the-everyday-formats-fp32-fp16-bf16)
- [4. FP8: E4M3 and E5M2](#4-fp8-e4m3-and-e5m2)
- [5. Integers instead of floats: INT8 and the quantization formula](#5-integers-instead-of-floats-int8-and-the-quantization-formula)
- [6. Why one scale is not enough: microscaling and MXFP8](#6-why-one-scale-is-not-enough-microscaling-and-mxfp8)
  - [6.1 A short history of FP8 scaling recipes](#61-a-short-history-of-fp8-scaling-recipes)
- [7. Hardware: what your GPU actually supports](#7-hardware-what-your-gpu-actually-supports)
- [8. Going lower: 4-bit and 2-bit weight quantization](#8-going-lower-4-bit-and-2-bit-weight-quantization)
  - [8.1 GPTQ: round carefully using curvature](#81-gptq-round-carefully-using-curvature)
  - [8.2 AWQ: protect the channels that matter](#82-awq-protect-the-channels-that-matter)
  - [8.3 QuIP and QuIP#: making 2-bit respectable](#83-quip-and-quip-making-2-bit-respectable)
  - [8.4 NF4 and the practical 4-bit default](#84-nf4-and-the-practical-4-bit-default)
- [9. 8-bit KV cache](#9-8-bit-kv-cache)
- [10. Cheat sheet: picking a precision](#10-cheat-sheet-picking-a-precision)
- [Sources](#sources)

---

## Setup and Notation

These terms recur throughout; skim once and refer back.

| Symbol / term | Meaning |
|---|---|
| **sign** $S$ | 1 bit: $S=0$ positive, $S=1$ negative. |
| **exponent** $E$ | the bits that set the *scale* (which power of two you are near). More exponent bits ⇒ wider **dynamic range**. |
| **mantissa** $M$ (a.k.a. fraction / significand) | the bits that set the *precision* within that scale — how finely you can subdivide between two powers of two. |
| **bias** $b$ | a constant subtracted from the stored exponent so it can represent negative powers; $b = 2^{(\#\text{exp bits})-1}-1$. |
| **"ExMy"** | naming convention: a format with $x$ exponent bits and $y$ mantissa bits (plus 1 sign bit). E.g. **E4M3** = 4 exponent, 3 mantissa. |
| **dynamic range** | ratio of the largest to smallest magnitude the format can represent — governed almost entirely by exponent bits. |
| **precision** | how many distinct values sit between consecutive powers of two — governed by mantissa bits ($2^{M}$ steps). |
| **quantization** | mapping high-precision numbers (FP32/BF16) onto a smaller format (FP8/INT8/INT4), unavoidably introducing rounding error. |
| **scale** $s$ | a per-tensor / per-channel / per-block multiplier that stretches a group of values into a low-bit format's usable range. |
| **amax** | the maximum absolute value in a tensor or block; used to pick that group's scale. |
| **master weights** | a high-precision (FP32) copy of the weights kept for stable *updates* while the heavy compute runs low-precision. |
| $N$ | number of model parameters (e.g. $70\times10^9$ for a "70B" model). |

A useful mental split: **exponent = how big, mantissa = how exact.** Almost every design choice below is a negotiation between those two using a fixed bit budget.

---

## 1. Why precision is a first-class decision

Start with memory. A parameter stored in FP32 takes 4 bytes. A 70-billion-parameter model is therefore

$$70\times10^{9}\ \text{params} \times 4\ \text{bytes} = 280\ \text{GB}$$

just to *hold the weights* — before any activations, optimizer state, or KV cache. That does not fit on a single 80 GB GPU. Halve the bits to BF16 and it becomes 140 GB; halve again to FP8/INT8 and it is 70 GB (now it fits on one card); 4-bit gets you to 35 GB, and 2-bit to ~18 GB. Memory scales **linearly** with bits-per-value, so precision is the most direct lever you have on whether a model runs at all.

![Bar chart of the memory needed to store a 70-billion-parameter model at FP32 (280 GB), FP16/BF16 (140 GB), FP8/INT8 (70 GB), INT4/NF4 (35 GB), and 2-bit (18 GB), with a dashed reference line at 80 GB marking a single high-end GPU; each halving of bit-width halves the memory.](./assets/precision_memory_tradeoff.jpg){ width=70% }

There are two more payoffs beyond fitting in memory:

- **Bandwidth.** LLM *decoding* (generating one token at a time) is usually **memory-bandwidth-bound**, not compute-bound: the GPU spends its time streaming weights and the KV cache from HBM, not multiplying. Fewer bits means fewer bytes to move per token, so latency drops roughly in proportion — this is why quantization speeds up generation even when the math is unchanged (see [`gpu-tpu-matmul-flashattention.md`](./gpu-tpu-matmul-flashattention.md)).
- **Throughput.** On hardware with dedicated low-precision units (Tensor Cores), an 8-bit matmul can run at roughly **2× the FLOPs** of a 16-bit one, and 4-bit faster still — *if* the GPU supports that format natively (Section 7).

The cost is **accuracy**: every bit you remove makes each number a coarser approximation, and those rounding errors can accumulate through a deep network. The whole art is spending your bits where they matter and rounding cleverly where they do not. The rest of this note is that art.

---

## 2. How a handful of bits becomes a number

A floating-point number is just **scientific notation in binary**. In base 10 you write $5.75 = 1.4375 \times 10^{1}$: a sign, a fractional part ("mantissa"), and an exponent. Binary floats do exactly this with base 2, and the IEEE-754 rule for reconstructing the value from the stored bits is:

$$\text{value} = (-1)^{S}\ \times\ \underbrace{(1.M)_2}_{\text{significand}}\ \times\ 2^{\,E - b}$$

Unpacking each piece, because this one formula explains *every* float format in the note:

- $(-1)^S$ flips the sign — one bit, nothing subtle.
- $(1.M)_2$ is the **significand**. The leading `1.` is **implicit** ("the hidden bit"): normalized binary scientific notation always starts with a 1 before the point, so there is no reason to store it — you get that bit for free. $M$ are the mantissa bits *after* the point, so the significand lives in $[1, 2)$ and the mantissa's $y$ bits carve that interval into $2^{y}$ equal steps.
- $2^{E-b}$ is the **scale**. $E$ is the unsigned integer in the exponent bits; subtracting the bias $b$ lets it represent negative powers (so you can also encode tiny numbers). For an 8-bit exponent, $b=127$, giving effective exponents from about $-126$ to $+127$.

**A fully worked FP32 example.** Take the decimal number $5.75$.

1. In binary $5.75 = 101.11_2$. Normalize: $101.11_2 = 1.0111_2 \times 2^{2}$.
2. Sign: positive, so $S=0$.
3. Exponent: the true exponent is $2$, and with bias $b=127$ we store $E = 2 + 127 = 129 = 10000001_2$.
4. Mantissa: the bits after the point are `0111`, padded with zeros to 23 bits: `01110000000000000000000`.
5. Reconstruct: $(-1)^0 \times 1.0111_2 \times 2^{2}$. Now $1.0111_2 = 1 + \tfrac14 + \tfrac18 + \tfrac{1}{16} = 1.4375$, and $1.4375 \times 4 = 5.75$. ✓

The three edge cases you should recognize but need no depth on: an all-zero exponent encodes **subnormals** (numbers even closer to zero, filling the gap around 0), an all-ones exponent with zero mantissa is **±∞**, and all-ones with nonzero mantissa is **NaN**.

The one structural consequence to carry forward: because the exponent multiplies the whole significand, **floats have finer absolute spacing near zero and coarser spacing far from zero** — their *relative* precision (roughly $2^{-y}$) is constant. Integers, by contrast, are spaced uniformly. That difference is exactly why floats shine for the wide-ranging, mostly-small-with-rare-huge values found in neural networks, and it is the reason FP8 often beats INT8 (Sections 4–5).

---

## 3. The everyday formats: FP32, FP16, BF16

With the formula in hand, the 16- and 32-bit formats differ only in how they split their bits between exponent (range) and mantissa (precision).

![Horizontal bit-strip diagram comparing FP32 (1 sign, 8 exponent, 23 mantissa), FP16 (1/5/10), BF16 (1/8/7), FP8 E4M3 (1/4/3), FP8 E5M2 (1/5/2), and FP4 E2M1 (1/2/1); bar width is proportional to total bits, and each row is annotated with its total bit count and approximate representable range. Sign is orange, exponent blue, mantissa green.](./assets/float_formats_bitlayout.jpg){ width=95% }

- **FP32 (E8M23), 4 bytes.** The historical default: 8 exponent bits give a huge range (~$10^{-38}$ to $3.4\times10^{38}$) and 23 mantissa bits give ~7 decimal digits of precision. Accurate but bulky; today mostly a baseline and a place to keep sensitive accumulations.
- **FP16 (E5M10), 2 bytes.** Halves the memory. It keeps a generous 10 mantissa bits (good precision) but spends only **5 exponent bits**, so its range tops out around **65 504**. That narrow range is a real problem in training: gradients can underflow to zero or overflow to infinity, which is why FP16 training historically needed **loss scaling** (multiply the loss to push gradients into range, then divide back) to stay stable.
- **BF16 (E8M7), 2 bytes.** "Brain float." The key design move: **keep FP32's full 8-bit exponent** (identical range, ~$10^{38}$) and pay for it by cutting the mantissa to 7 bits (coarser precision). This is a bet that neural nets care far more about **range** than about the last few bits of precision — and it turned out to be right. BF16 largely killed the loss-scaling headache and is now the *default* training precision on modern accelerators. Note FP16 and BF16 are the **same size**; they just cut the pie differently. (NVIDIA's **TF32** is a related trick: FP32's 8-bit exponent with a 10-bit mantissa, used internally by Tensor Cores.)

**Mixed-precision training** is the recipe that makes these 16-bit formats safe. Keep a **high-precision (FP32) master copy** of the weights for the tiny weight *updates* (which would be rounded to nothing in 16-bit), but run the expensive forward and backward **matmuls in BF16/FP16**. You get 16-bit speed and memory on the heavy compute while the master weights accumulate small gradient steps faithfully. This **"compute low, accumulate high"** pattern — drop the bulk data to few bits, but keep the sensitive accumulation in higher precision — is the template for *every* scheme below: FP8, weight-only 4-bit, and quantized KV cache all do the same thing.

The takeaway: at 16 bits, the industry chose *range over precision* (BF16 over FP16) for training. That same instinct reappears at 8 bits.

---

## 4. FP8: E4M3 and E5M2

Below 16 bits, there are too few bits to serve everyone, so FP8 ships as **two standardized variants** (the OCP / NVIDIA Hopper conventions), and you pick per use-case:

- **E4M3** — 4 exponent, 3 mantissa bits. **More precision, less range** (magnitudes up to ~448). Used for the **forward pass**: weights and activations, where values are relatively well-behaved and you want the extra precision.
- **E5M2** — 5 exponent, 2 mantissa bits. **More range, less precision** (up to ~57 344). Used for **gradients** in the backward pass, which span a much wider dynamic range and would otherwise overflow.

This mirrors the FP16-vs-BF16 split one level down: when in doubt, spend bits on range for the quantities that swing hard (gradients) and on precision for the quantities that are tame (weights/activations).

**Just how few numbers is 8 bits?** Take E4M3, whose bias is $b=7$. Its largest finite value uses the biggest exponent and the near-biggest mantissa: $1.75 \times 2^{15-7} = 1.75 \times 2^{8} = 448$. (The very top mantissa code at that exponent is stolen for **NaN**, and E4M3 has **no infinity at all** — there simply aren't spare bit patterns for it.) Its smallest normal value is $2^{-6}\approx 0.0156$, with subnormals reaching down to $\approx 0.002$. All told, E4M3 has only about **126 distinct positive values** across its entire range — it is not a smooth number line but a *sparse set of allowed points*, and two nearby reals routinely round to the same code. This is the literal meaning of "8 bits are tiny," and exactly why **raw FP8 is unusable without scaling**: a layer whose activations mix $0.001$ with $90$ will see the small values flushed to zero and the large ones snapped onto a coarse grid.

**Why bother with FP8 floats instead of INT8?** Both are 8 bits. The difference is *how the 256 available codes are placed on the number line.* INT8 spaces its levels **uniformly**; FP8 spaces them **logarithmically** — dense near zero, sparse far away. Neural-network tensors are overwhelmingly small values with a few large outliers, so FP8's non-uniform grid wastes fewer codes and represents that distribution more faithfully. FP8 also needs no zero-point bookkeeping (next section). The practical rule: FP8 tends to win for tensors with **wide dynamic range** (especially activations with outliers); INT8 can match or beat it for tame, well-calibrated weights and is supported on far more hardware.

Crucially, even FP8 is almost never used with a *single* global scale — the values are first multiplied by a scale factor to line them up with FP8's usable range. How coarse or fine that scaling is turns out to be the whole game (Section 6).

---

## 5. Integers instead of floats: INT8 and the quantization formula

Integer quantization takes a different route: instead of a clever non-uniform grid, use a **uniform** grid of integers and attach a **scale** (and optionally an offset) to map that grid onto the real values. This is the classic "quantize to INT8" you hear about.

**Affine (asymmetric) quantization.** Given a tensor of floats $x$, pick a scale $s$ and an integer **zero-point** $z$, then

$$x_q = \mathrm{round}\!\left(\frac{x}{s}\right) + z, \qquad \hat{x} = s\,(x_q - z).$$

Reading it term by term: $x_q$ is the stored 8-bit integer; dividing by $s$ compresses the float range into integer range and rounding snaps to the nearest level; the **zero-point** $z$ is an integer shift so that real-value $0$ maps exactly onto some integer code (important when the data is not centered at zero, e.g. post-ReLU activations that are all non-negative). To use the number you **dequantize** with $\hat{x} = s(x_q - z)$, recovering an approximation of $x$. The scale and zero-point come from the observed min/max:

$$s = \frac{x_{\max} - x_{\min}}{q_{\max} - q_{\min}}, \qquad z = q_{\min} - \mathrm{round}\!\left(\frac{x_{\min}}{s}\right),$$

where $[q_{\min}, q_{\max}]$ is the integer range (e.g. $[-128, 127]$ for signed INT8).

**Symmetric quantization** is the common simplification: force the range symmetric around zero using the max absolute value ("**absmax**"), which makes $z=0$ and drops a subtraction from every dot-product — cheaper on hardware. Then simply $s = \max(|x|)/q_{\max}$ and $x_q = \mathrm{round}(x/s)$. Weights (centered near zero) almost always use symmetric; activations that are one-sided may use asymmetric.

**The outlier problem — and calibration.** Because $s$ is set by the extreme values, **a single huge outlier stretches the scale and coarsens every other value.** Fixes:

- **Calibration**: choose the clipping range not from the raw min/max but to minimize error — via a **percentile** (clip the top 0.1%), **MSE**, or **KL-divergence** between the original and quantized distributions. You run a small **calibration dataset** through the model to observe activation ranges.
- **Clipping**: deliberately saturate outliers to a chosen bound, trading large error on a few values for small error on the many.
- **Static vs dynamic**: weights are known ahead of time, so their scales are computed once (**static**). Activations depend on the input, so their scales are either precomputed from calibration data (static) or recomputed per input at run time (**dynamic**).

This outlier headache is exactly what motivates finer-grained scaling — the subject of the next section.

---

## 6. Why one scale is not enough: microscaling and MXFP8

Everything so far assumed **one scale per tensor**. That assumption breaks on real transformer tensors, which are full of **outlier channels** — a handful of features with magnitudes 10–100× the rest. With a single per-tensor scale, that outlier forces the scale large enough to fit it, and now the ocean of small values gets squeezed into just a few of the low-bit levels, destroying their precision.

The fix is to **share a scale over a smaller group of numbers**, so an outlier only pollutes its own neighborhood:

![Three side-by-side heatmaps of the same 8×8 weight matrix, showing which elements share one quantization scale under per-tensor (one box around everything), per-channel (one box per column), and microscaling / per-block (small 2×2 boxes) granularity. Two outlier cells are highlighted in orange; finer grouping keeps those outliers from forcing a coarse scale onto the many small values.](./assets/quant_scaling_granularity.jpg){ width=92% }

- **Per-tensor**: one scale for the whole matrix. Cheapest, most outlier-sensitive.
- **Per-channel (per-row/column)**: one scale per output channel. Standard for INT8 weights; helps a lot because outliers often concentrate in specific channels.
- **Per-block / microscaling**: split each row into small **blocks** (typically **32 consecutive elements**) and give each block its own scale. Now an outlier only stretches the scale of its own 32-element block; every other block keeps a scale matched to its local magnitude, so quantization error drops sharply.

**MXFP8 (Microscaling FP8)** is the standardized version of this idea, defined by the **Open Compute Project (OCP) Microscaling spec**. The recipe:

- Each **block of 32 elements** stores its values in ordinary FP8 (E4M3 or E5M2)...
- ...plus **one shared scale per block**, stored as an **8-bit power-of-two exponent** (the format **E8M0** / **UE8M0** — literally just an exponent, no mantissa, because a power-of-two scale is enough and is cheap to apply by shifting).

So a value is reconstructed as `FP8_element × 2^(block_scale)`. The overhead is tiny — one extra byte per 32 values, about 3% — but the accuracy gain is large enough that MXFP8 can be used for **training**, not just inference. The same family scales down further: **MXFP6, MXFP4** (S1E2M1 elements with a shared block scale), and **MXINT8**, trading element bit-width for range while keeping the block-scaling trick. In short, microscaling exists because **one scale can't simultaneously serve outliers and the bulk of small values — so give each small block its own.**

### 6.1 A short history of FP8 scaling recipes

Microscaling didn't arrive fully formed. FP8 *training* recipes evolved through three eras, each shrinking the group of numbers that shares a single scale:

- **Era A — per-tensor "delayed scaling" (Hopper / H100, ~2022–2024).** One scale for an *entire* tensor. To avoid an extra pass over the data just to find its range, NVIDIA's Transformer Engine `DelayedScaling` tracks each tensor's **amax** over *recent* training steps and sets the next step's scale as roughly $s = \text{amax} / \text{FP8}_{\max}$ — "delayed" because it reuses a slightly stale amax from previous steps. Cheap and H100-friendly, but a single outlier dominates the whole scale, so the other ~99% of values quantize poorly (many collapse to zero).
- **Era B — 128-element blockwise scaling.** Split each tensor into blocks of **128** consecutive values, each with its own scale (Transformer Engine `Float8BlockScaling`). An outlier now only spoils its own 128-value block, not the whole tensor — much more accurate. The cost is more scale metadata to move and stricter memory-layout requirements.
- **Era C — MXFP8, 32-element blocks (Blackwell, ~2024+).** Shrink the block to **32** and standardize the shared scale as **E8M0**. This is the microscaling described above, and it is the first era with *native* Tensor-Core support (Section 7).

Two nice consequences of blocks this small. First, **amax is computed per 32-element block**, so each block's scale is matched to its own local magnitudes — tiny gradients no longer share a scale with a distant outlier. Second, and less obvious: because the fine block scales give every group its own effective dynamic range, **E4M3 alone is usually enough for both the forward *and* backward pass**. The older habit of switching to wide-range E5M2 for gradients (Section 4) largely disappears, which simplifies the whole training recipe to a single element format.

---

## 7. Hardware: what your GPU actually supports

A format is only fast if the silicon has arithmetic units for it; otherwise the GPU must **emulate** it (unpack to a supported type, compute, repack) and you get the memory savings but *not* the speedup. This is why "which precision?" is inseparable from "which GPU?":

- **INT8 / FP16 / BF16**: broadly supported across NVIDIA Turing/Ampere (A100), AMD, and TPUs. Safe, portable baselines.
- **FP8 (E4M3/E5M2)**: first got **native Tensor Core** support on NVIDIA **Hopper (H100)** via its **Transformer Engine**, and on AMD **MI300**. On pre-Hopper GPUs FP8 is emulated.
- **MXFP8 / MXFP6 / MXFP4 (microscaling)**: this is the headline **Blackwell (B200 / GB200)** feature. Blackwell's **5th-generation Tensor Cores natively support the MX block-scaled formats** — the hardware reads the per-32-element E8M0 scales and the FP8 elements directly (via the `tcgen05` block-scaled matrix instructions), so an MXFP8 matmul runs at **full FP8 throughput (~2× BF16)** with **no emulation overhead**. On Hopper you can still *use* MXFP8 numerically, but the block-scaling is handled in software, so you don't get Blackwell's free-lunch speedup. So yes — the thing you heard is right: **MXFP8 is natively supported on B200-class Blackwell GPUs**, and that native support is much of the reason the format is suddenly everywhere.

Two low-level details explain jargon you'll hit in Blackwell FP8 write-ups. Blackwell's Tensor Cores compute in **128×128 tiles** and must fetch the block scales that go with each tile; stored in plain row-major order those scales are scattered relative to how the tile unit reads memory, so libraries **swizzle** them into a contiguous **128×4 layout** the hardware can load in one shot (Transformer Engine does this for you — you only touch it when hand-writing kernels). And when a report says a model was **"trained in MXFP8"** — e.g. on **B200/B300** — it means the heavy **GEMMs** (linear layers, attention projections) ran in 8-bit microscaled format, *not* that every byte was 8 bits: the **master weights, optimizer state, and loss** are still kept in higher precision, exactly the "compute low, accumulate high" pattern from Section 3.

The practical consequence: the "best" precision for a job is whichever your **deployment hardware** accelerates natively. Quantizing to a format your GPU must emulate buys memory but can even *lose* you speed.

---

## 8. Going lower: 4-bit and 2-bit weight quantization

At this point a fair reaction is: **how can a model possibly work in 2 bits per weight — that's only four distinct values?** The resolution is three ideas working together:

1. **Weight-only quantization.** Only the **weights** are stored in 4/2-bit; **activations stay in FP16/BF16**. At run time each weight block is **dequantized back to FP16 just before the matmul**, and the actual arithmetic happens in FP16. So the 4 values are never asked to do precise math directly — they only need to *reconstruct* a good FP16 weight. This is why the win is purely **memory and bandwidth**, not compute: for memory-bound *decoding* that is exactly the bottleneck that matters, so weight-only 4-bit gives large speedups with tiny quality loss.
2. **Fine-grained block scales.** As in microscaling, weights are quantized in small groups (e.g. 64 or 128 per group) each with its own scale, so 2 bits encode the *shape within a group* while the scale restores the magnitude.
3. **Smart rounding.** Naive round-to-nearest is what falls apart at 2–4 bits; the methods below round *intelligently*, accounting for which errors actually hurt the model's outputs. This is what turns "impossible" into "usable."

The methods differ in how they do step 3 (and, for QuIP, step 2).

### 8.1 GPTQ: round carefully using curvature

Naive rounding treats every weight independently and every rounding error as equally harmful. **GPTQ** ([Frantar et al. 2022](https://arxiv.org/abs/2210.17323)) instead quantizes a layer to minimize the change in that layer's **output**, not its weights. Concretely, it minimizes the reconstruction error $\lVert Wx - \hat{W}x\rVert^2$ over calibration inputs $x$, whose curvature is captured by the **Hessian** $H = XX^\top$ (the input second-moment matrix).

The intuition that makes it work: **when you round one weight, you can partly cancel the damage by nudging the not-yet-quantized weights in the same layer.** GPTQ processes weights in order, and after fixing each one to its rounded value it uses the Hessian to compute a compensating update to the remaining weights, so error does not pile up. Using second-order information means "sensitive" weights (large curvature — big output effect) are handled carefully while flat directions absorb more rounding. The payoff: GPT-scale models quantized to **3–4 bits in a few GPU-hours with negligible accuracy loss**, a mainstay of practical 4-bit serving.

### 8.2 AWQ: protect the channels that matter

**AWQ** (Activation-aware Weight Quantization, [Lin et al. 2023](https://arxiv.org/abs/2306.00978)) starts from a sharp observation: **weight importance is extremely skewed — protecting just ~1% of "salient" weight channels recovers most of the lost accuracy.** The twist is *how you find them*: salience is judged not by the weights' own magnitudes but by the **activation** magnitudes flowing through each channel — a channel that multiplies large activations matters more, because its error is amplified downstream.

Rather than store those 1% in higher precision (which would mean ugly, hardware-unfriendly **mixed precision**), AWQ keeps everything uniformly low-bit but applies an **equivalent transformation**: **scale up the salient channels before quantizing** (and fold the inverse scale into the next operation so the math is unchanged). Scaling a channel up before rounding shrinks its *relative* quantization error, effectively giving the important channels more precision without any mixed-precision bookkeeping. It needs no backprop or reconstruction — just calibration activation statistics — so it is fast and robust, and is one of the most widely deployed 4-bit methods.

### 8.3 QuIP and QuIP#: making 2-bit respectable

Two bits is where naive methods collapse entirely, and where **QuIP** ([Chee et al. 2023](https://arxiv.org/abs/2307.13304)) makes its point. Its core concept is **incoherence**: quantization is easiest when the weight (and Hessian) matrices have **uniform magnitudes and no directions aligned with the coordinate axes** — i.e. no concentrated outliers. Real weights are *not* incoherent, so QuIP **makes** them so: it multiplies the weights and Hessian by **random orthogonal matrices** before quantizing (and inverts the rotation afterward, which is free because it can be folded into neighboring layers).

Why this rescues 2-bit: a random rotation **smears any concentrated outlier across all coordinates**, turning a few catastrophic values into many small, uniform ones that a coarse 2-bit grid can represent well. It is the same "spread the outliers out" instinct as microscaling and AWQ, but achieved by rotating the coordinate system rather than by grouping or per-channel scaling. **QuIP#** (the follow-up) makes this fast and higher-quality by using cheap structured **Hadamard** rotations plus a lattice codebook, and is a big reason 2-bit LLMs went from "curiosity" to "sometimes actually usable." The honest caveat: even with these tricks, 2-bit still costs noticeable quality on hard reasoning tasks, so 4-bit remains the sweet spot for most deployments while 2-bit is reserved for extreme memory pressure.

### 8.4 NF4 and the practical 4-bit default

For everyday 4-bit, the pragmatic default is **NF4 (4-bit NormalFloat)**, introduced with **QLoRA** ([Dettmers et al. 2023](https://arxiv.org/abs/2305.14314)). Instead of uniform integer levels, NF4 places its 16 levels at the **quantiles of a normal distribution** — because neural-net weights are roughly Gaussian, this is *information-theoretically* a better fit, putting more levels where weights are dense (near zero). Combined with **group-wise scales** and **double quantization** (quantizing the scales themselves to save a bit more), NF4 lets you load a 70B model in ~35 GB and even **fine-tune** it via LoRA adapters on a single GPU. When someone says "just run it in 4-bit," they usually mean NF4 (bitsandbytes) or a GPTQ/AWQ INT4 build.

---

## 9. 8-bit KV cache

The **KV cache** stores the keys and values for every past token so attention doesn't recompute them each step (background: [`gpu-tpu-matmul-flashattention.md`](./gpu-tpu-matmul-flashattention.md)). For long contexts and large batches it becomes **enormous** — often larger than the weights — and because decoding re-reads the *entire* cache every single token, decode speed is throttled by the **bandwidth** of streaming it. That makes the KV cache a prime quantization target.

**8-bit KV cache** stores K and V in **FP8 (usually E4M3)** or INT8 instead of FP16/BF16, with **per-token or per-block scales** computed on the fly. When attention needs them, they are **dequantized back to FP16** just before the score and value dot-products — same pattern as weight-only quantization. The wins are direct and large:

- **Half the memory per cached token**, so you can hold **~2× the context length or ~2× the batch size** in the same HBM.
- **Half the memory traffic per attention step**, which is exactly the decode bottleneck — reported gains are disproportionately large (KV-cache quantization can help decode-heavy, long-context workloads more than quantizing the linear layers does).

The accuracy hit is typically small, with one asymmetry worth knowing: **keys are more sensitive than values** (errors in keys perturb the softmax over *all* positions), so some systems keep keys at higher precision or use finer scaling for keys than values. In practice FP8 KV cache is now a sane default for long-context serving; INT8 and even 4-bit KV caches exist for more aggressive memory savings at a larger quality cost.

---

## 10. Cheat sheet: picking a precision

| Format | Bits | Split (S/E/M) | Strength | Typical use | Native HW |
|---|---|---|---|---|---|
| **FP32** | 32 | 1/8/23 | max precision + range | baseline, sensitive accumulation | everything |
| **TF32** | 19* | 1/8/10 | FP32 range, faster matmul | internal Tensor Core math | Ampere+ |
| **BF16** | 16 | 1/8/7 | FP32 range, cheap | **default training** | Ampere+, TPU |
| **FP16** | 16 | 1/5/10 | precise, narrow range | training w/ loss scaling, inference | most GPUs |
| **FP8 E4M3** | 8 | 1/4/3 | precise 8-bit | fwd: weights/activations | Hopper, MI300, Blackwell |
| **FP8 E5M2** | 8 | 1/5/2 | wide-range 8-bit | gradients | Hopper, MI300, Blackwell |
| **MXFP8** | 8+ | FP8 + block scale | 8-bit good enough to *train* | training/inference matmuls | **Blackwell (native)** |
| **INT8** | 8 | integer + scale/zp | portable, fast | inference (weights/acts) | everything |
| **INT4 / NF4** | 4 | integer/quantile + group scale | 4× smaller weights | weight-only LLM inference & QLoRA | wide (often emulated compute) |
| **2-bit (QuIP#)** | 2 | rotated + codebook | extreme compression | memory-starved inference | emulated |

\*TF32 stores 19 meaningful bits inside a 32-bit slot.

**How to choose, in one breath:** train in **BF16** (or **MXFP8** on Blackwell); serve on Hopper/Blackwell in **FP8** for the best speed/quality; on older or portability-constrained hardware use **INT8**; when memory is the binding constraint, go **4-bit weight-only (NF4 / GPTQ / AWQ)**, reaching for **2-bit** only under real duress; and quantize the **KV cache to FP8** whenever contexts are long or batches are large. Always check that your **target GPU accelerates the format natively** — otherwise you save memory but forfeit the speedup.

---

## Sources

- Grootendorst, M. 2024, *A Visual Guide to Quantization* ([blog](https://newsletter.maartengrootendorst.com/p/a-visual-guide-to-quantization)) — float layouts, symmetric/asymmetric quantization, outliers/calibration.
- IEEE-754 / floating-point representation references (exponent bias, the $(-1)^S(1.M)2^{E-b}$ formula and worked examples).
- **Open Compute Project**, *OCP Microscaling (MX) Formats Specification v1.0* — MXFP8/6/4, MXINT8, 32-element blocks, E8M0 shared scale.
- **NVIDIA**, *Transformer Engine — MXFP8* docs and the *Blackwell architecture* materials (native MX Tensor Core support, `tcgen05` block-scaled MMA).
- **PyTorch**, *Enabling up to 41% faster pre-training: MXFP8 and DeepEP for DeepSeek-V3 on B200* ([blog](https://pytorch.org/blog/enabling-up-to-41-faster-pre-training-mxfp8-and-deepep-for-deepseek-v3-on-b200-with-torchtitan/)).
- Frantar et al. 2022, *GPTQ: Accurate Post-Training Quantization for Generative Pre-trained Transformers* ([arXiv:2210.17323](https://arxiv.org/abs/2210.17323)).
- Lin et al. 2023, *AWQ: Activation-aware Weight Quantization for LLM Compression and Acceleration* ([arXiv:2306.00978](https://arxiv.org/abs/2306.00978)).
- Chee et al. 2023, *QuIP: 2-Bit Quantization of Large Language Models With Guarantees* ([arXiv:2307.13304](https://arxiv.org/abs/2307.13304)); Tseng et al. 2024, *QuIP#* ([arXiv:2402.04396](https://arxiv.org/abs/2402.04396)).
- Dettmers et al. 2023, *QLoRA: Efficient Finetuning of Quantized LLMs* (NF4, double quantization) ([arXiv:2305.14314](https://arxiv.org/abs/2305.14314)).
- vLLM docs & blog, *FP8 / Quantized KV Cache* ([vLLM KV-cache](https://docs.vllm.ai/en/latest/features/quantization/quantized_kvcache/)).
- Dettmers & Zettlemoyer 2022, *The case for 4-bit precision: k-bit Inference Scaling Laws* ([arXiv:2212.09720](https://arxiv.org/abs/2212.09720)).
