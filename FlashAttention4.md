# FlashAttention-4: When Matmul Gets Faster Than the Rest of Attention

FlashAttention-4 (FA4) is an attention kernel designed around NVIDIA Blackwell GPUs such as the B200. Its central problem is not that matrix multiplication is too slow: on Blackwell, tensor cores became faster *relative to* the units that feed them and compute softmax. FA4 rearranges the work so those slower operations are less likely to leave the tensor cores idle. The main source is [Zadouri et al., *FlashAttention-4: Algorithm and Kernel Pipelining Co-Design for Asymmetric Hardware Scaling*](https://arxiv.org/pdf/2603.05451).

This is a note about *exact tiled attention*, not an approximation that changes which tokens can attend to which others. For the basic attention mechanism, see [Attention](training_llms/005_Attention.md); for approximate ways to change its computational cost, see [Efficient Attention](training_llms/006_EffecientAttention.md).

## Table of Contents

- [The problem: asymmetric hardware scaling](#the-problem-asymmetric-hardware-scaling)
- [The B200 architecture that matters here](#the-b200-architecture-that-matters-here)
- [Forward pass: keep matmul and softmax busy together](#forward-pass-keep-matmul-and-softmax-busy-together)
  - [Tile-level bottleneck](#tile-level-bottleneck)
  - [Ping-pong pipeline](#ping-pong-pipeline)
  - [Using another unit for some exponentials](#using-another-unit-for-some-exponentials)
  - [Rescale only when necessary](#rescale-only-when-necessary)
- [Backward pass: shared-memory traffic takes over](#backward-pass-shared-memory-traffic-takes-over)
- [Scheduling and reproducibility](#scheduling-and-reproducibility)
- [Results, scope, and takeaways](#results-scope-and-takeaways)
- [Source](#source)

## The problem: asymmetric hardware scaling

For one attention head, let $Q$, $K$, and $V$ be the query, key, and value matrices for a sequence of $L$ tokens; each has head dimension $d$. The scores are $S=QK^\top/\sqrt d$, the row-wise probabilities are $P=\mathrm{softmax}(S)$, and the output is $O=PV$. Computing $S$ and $O$ takes two large matrix multiplications. Softmax also needs row maxima, exponentials, and sums. Ordinary attention would materialize the $L\times L$ score and probability matrices in high-bandwidth memory (HBM); FlashAttention instead processes blocks and keeps these intermediates on-chip. This saves HBM traffic but does **not** remove the quadratic arithmetic of dense attention.

FlashAttention-3 (FA3) overlapped matrix multiplication and softmax on Hopper. Moving that kernel design unchanged to Blackwell misses a new constraint: the B200 offers roughly twice Hopper's BF16 tensor-core throughput, but its per-SM exponential throughput and measured shared-memory read bandwidth have not doubled. Faster matrix multiplies therefore make the *other* stages more visible. FA4 addresses both forward and backward passes, which have different bottlenecks.

## The B200 architecture that matters here

Think of an SM (streaming multiprocessor) as a workshop with several work areas:

| Hardware resource | What it does for attention | Why FA4 cares |
| --- | --- | --- |
| HBM / L2 | Supplies input and output tensors; L2 caches HBM accesses | Tiling avoids repeatedly storing and reading the full score matrix from HBM. |
| Shared memory (SMEM) | Programmable on-chip staging area for matrix operands | Re-reading operands can bottleneck the backward pass even when HBM traffic is low. |
| Tensor cores | Perform the large matrix multiplies (MMA) | BF16 peak on B200 is about 2.25 PFLOPs/s per GPU, versus about 1 PFLOP/s on Hopper in the paper's comparison. |
| Multifunction unit (MUFU) | Computes hardware exponentials used by softmax | Its measured capacity is 16 exponentials per clock per SM on B200, the same figure the paper gives for Hopper. |
| Tensor memory (TMEM) | Blackwell's 256 KB per-SM on-chip storage for tensor-core intermediates | Async matrix multiplies write accumulators here instead of tying up thread registers; this makes larger tiles and more overlap practical. |

The paper reports 8192 BF16 tensor-core operations per clock per SM on B200, versus 4096 on Hopper, but only 128 bytes per clock per SM of shared-memory read throughput on both. These are *different resources with different units*, not directly comparable rates; the relevant comparison is how long each resource takes to handle **one attention tile**.

A CTA (thread block) is a cooperating group of threads on one SM. Blackwell can also have a *pair* of CTAs cooperate on a single tensor-core operation (2-CTA mode): they share the work and stage less duplicated operand data. That feature matters most for FA4's backward pass. TMEM is not a replacement for HBM or SMEM: it is dedicated space for tensor-core inputs/intermediates and accumulators, with its own explicit management and synchronization.

## Forward pass: keep matmul and softmax busy together

### Tile-level bottleneck

Suppose a tile contains $M$ query rows, $N$ key/value rows, and head dimension $d$. Computing its scores ($QK^\top$) and contribution to the output ($PV$) costs about $4MNd$ floating-point operations in total: each matrix multiplication contributes $2MNd$. The tile also needs $MN$ exponentials. Dividing by the B200 per-SM rates above gives an illustrative lower bound in clock cycles:

$$
T_{\mathrm{MMA}}\approx\frac{4MNd}{8192},\qquad T_{\mathrm{exp}}\approx\frac{MN}{16}.
$$

Here $T_{\mathrm{MMA}}$ is tensor-core time and $T_{\mathrm{exp}}$ is hardware-exponential time, assuming perfect utilization and no other costs. For $M=N=d=128$, **both are 1024 cycles**; the paper estimates 768 cycles of shared-memory reads for that tile. These costs can overlap, so they are *not* times to add together or a prediction of kernel runtime. They show why softmax can stop an otherwise very fast tensor core from making progress. At $M=256$, $N=d=128$, the paper estimates 2048 cycles for each of MMA and exponentials, versus 1536 for shared-memory reads.

![Two horizontal-bar panels compare estimated B200 resource cycles per tile. For forward at M=N=d=128, tensor cores and exponentials each take 1024 cycles versus 768 for shared memory. For backward, shared memory takes 3328 versus 2560 for tensor cores and 1024 for exponentials.](assets/flashattention4_resource_limits.jpg)

*The slowest resource changes between forward and backward. These are resource lower bounds for one tile, not benchmark runtimes; overlapping work can hide part of one bar behind another. Numbers from Tables 1 and 3 of the FA4 paper, redrawn here.*

The forward memory estimate is also a clue about tile size. $QK^\top$ reads both operands from SMEM, while $PV$ can read its probability operand from TMEM and its value operand from SMEM. Enlarging a tile increases useful work per launch, but if a matrix instruction cannot consume the whole tile at once it rereads some SMEM operands. Larger is therefore not automatically better: tile shape balances reuse, SMEM reads, and the space needed for overlapping intermediates.

### Ping-pong pipeline

FA4 works on two 128-row query tiles at a time. While the tensor cores multiply one tile, a different set of threads computes softmax for the other; then their roles alternate. This resembles FA3's ping-pong schedule, but Blackwell's tensor cores produce larger tiles asynchronously into TMEM. FA4 assigns one softmax thread per score row and separates the output-correction work from the softmax critical path. It carefully schedules which score, probability, and output tiles occupy TMEM at each step: having enough arithmetic units is not helpful if intermediate tiles cannot coexist on chip.

In more concrete terms, a warpgroup drives the asynchronous matrix instructions and data transfers; two other warpgroups process alternating query tiles' softmax rows; a correction warpgroup handles output rescaling. Each softmax thread owns a full row of 128 scores, so it can find that row's maximum and sum without shuffling partial results across warps. But holding the row in registers strains register capacity. FA4 staggers when it writes portions of the probability tile to TMEM so the row's temporary values do not force register spills. The two softmax groups also avoid executing their exponential-heavy phases simultaneously: running them at once would compete for the same saturated MUFU rather than double its throughput.

In a simplified timeline, the overlap looks like this:

| Moment | Tensor cores | Other threads |
| --- | --- | --- |
| A | Multiply query tile 1 | Softmax query tile 0 |
| B | Multiply query tile 0's next block | Softmax query tile 1 |

There are synchronization and data-movement steps between these stages; the table shows the scheduling idea, not an executable instruction sequence.

TMEM is what makes this schedule different from simply launching more threads. Scores can be written there by the tensor cores, read by softmax threads, then replaced with lower-precision probabilities for the next matrix multiply. Meanwhile two output accumulators remain resident for the two query tiles. Scheduling the *lifetimes* of these buffers is necessary to get the hoped-for overlap within the 256 KB TMEM budget; the matrix instructions alone do not arrange it.

### Using another unit for some exponentials

Softmax needs $e^x$ for each score after scaling. The hardware exponential unit can become saturated, so FA4 computes **some**, not all, exponentials using floating-point multiply-add (FMA) instructions on other execution units. To do this, it writes $e^x=2^{x/\ln 2}$ and splits the base-2 exponent into an integer part (handled via floating-point exponent bits) and a fractional part (approximated by a short polynomial). This moves a portion of the work off the overloaded MUFU while still using hardware exponentials for most entries.

Why not emulate every exponential? Polynomial evaluation costs registers, register bandwidth, and latency; using it everywhere can cause spills and become slower. The paper reports tuning an emulated fraction of roughly 10-25% per row. On random fractional inputs, its degree-3 approximation has a maximum relative FP32 error around $8.8\times10^{-5}$, much larger than hardware's, but the paper's BF16-rounded comparison shows nearly the same error as hardware because BF16 rounding dominates. **That is a measured tradeoff for the studied BF16 path, not a blanket accuracy guarantee for every precision or workload.**

### Rescale only when necessary

Tiled softmax cannot know the maximum score of a row until it has seen all key blocks. It therefore maintains a reference maximum $m$, a running weight sum $\ell$, and an unnormalized weighted-value sum $a$. For a new block of scores $S_j$ and values $V_j$, with $j$ denoting the current key/value block, ordinary online softmax raises $m$ to the larger of its old value and the new block's row maximum. If $m$ rises, both $\ell$ and $a$ must be multiplied by a correction factor before adding the new block's contribution. Repeatedly correcting the full output vector $a$ costs time.

FA4 keeps the old reference maximum unless the new maximum exceeds it by more than a threshold $\tau$. If it does exceed the threshold, it rescales both running sums into the new reference frame. Otherwise, it adds the new block using the *old* reference:

$$
\begin{aligned}
\text{if }m_{\mathrm{new}}-m>\tau:&\quad
 a\leftarrow e^{m-m_{\mathrm{new}}}a+e^{S_j-m_{\mathrm{new}}}V_j,\quad
 \ell\leftarrow e^{m-m_{\mathrm{new}}}\ell+\mathrm{rowsum}(e^{S_j-m_{\mathrm{new}}}),\quad m\leftarrow m_{\mathrm{new}};\\
\text{otherwise}:&\quad
 a\leftarrow a+e^{S_j-m}V_j,\quad
 \ell\leftarrow\ell+\mathrm{rowsum}(e^{S_j-m}).
\end{aligned}
$$

Here $m_{\mathrm{new}}$ is the largest score seen so far (including this block), $e^{S_j-m}$ applies the exponential elementwise with row-wise broadcasting, and $\mathrm{rowsum}$ sums across the block's keys. The final attention output is $a/\ell$: numerator and denominator always use the *same* reference, so delaying a correction does not change that ratio in exact arithmetic. It does allow temporary weights above 1; the threshold bounds how large they get and controls the numerical tradeoff. The paper describes a threshold of **8 in base-2 score units** (a factor of 256); in the natural-log scores written above the equivalent threshold is $\ln 256\approx5.55$. This distinction matters because simply using $\tau=8$ with natural exponentials would mean a factor of about 2981, not 256. FA4 also makes the decision collectively within a warp to avoid divergent execution.

For a single query with two scalar values $v_1$ and $v_2$, suppose the first score is 0 and the second is $\ln 2$. With reference $m=0$, skipping the correction gives $a=v_1+2v_2$ and $\ell=1+2=3$, so the output is $(v_1+2v_2)/3$. Updating the reference to $\ln 2$ instead would give $a=v_1/2+v_2$ and $\ell=1/2+1=3/2$: the intermediate numbers change, but their ratio is identical. For larger score jumps, rescaling keeps those intermediate weights within a numerically manageable range.

## Backward pass: shared-memory traffic takes over

Training needs gradients with respect to $Q$, $K$, and $V$. Let $dO$ be the gradient arriving from the next layer. Backward reconstructs the attention probabilities $P$ from saved softmax statistics rather than storing the full $L\times L$ probability matrix. It then computes $dV=P^\top dO$ and $dP=dOV^\top$. The softmax gradient $dS$ follows by removing each row's probability-weighted mean from $dP$ and multiplying by $P$; finally $dQ=dSK/\sqrt d$ and $dK=dS^\top Q/\sqrt d$. That is **five** tile-level matrix multiplies: one to recompute scores and four to propagate gradients. See the [paper's backward computation graph](https://arxiv.org/pdf/2603.05451#page=10) for the detailed ordering.

The softmax step has a useful shortcut. For each query row $i$, let $D_i$ be the dot product of row $i$ of $dO$ and row $i$ of the forward output $O$. Then the gradient for score position $(i,j)$ is

$$
dS_{ij}=P_{ij}(dP_{ij}-D_i),\qquad D_i=\sum_{k=1}^{d}dO_{ik}O_{ik}.
$$

The index $j$ selects a key and $k$ runs over the $d$ value features. Why does the row correction use $O$ rather than a stored probability matrix? Because $O_i=\sum_j P_{ij}V_j$, so $dO_i\cdot O_i=\sum_jP_{ij}(dO_i\cdot V_j)=\sum_jP_{ij}dP_{ij}$. Thus $D_i$ is exactly the mean term that the softmax Jacobian subtracts, available from a short dot product per query row. Scores and probabilities still have to be recomputed, but the huge intermediate need not be kept in HBM.

At $M=N=d=128$, the paper estimates **2560 cycles** of tensor-core work, **1024** for exponentials, but **3328** for shared-memory accesses. Unlike forward, shared-memory traffic is the largest of these simplified resource costs. TMEM lets FA4 keep accumulators outside the regular register file and overlap softmax/gradient work with matrix multiplies from neighboring iterations.

That 3328-cycle estimate has three parts: **2048** for the operands of the five matrix multiplies, **256** for writing intermediate $dS$ to SMEM, and **1024** for writing and rereading the partial $dQ$ accumulator in FP32. Moving an accumulator to TMEM reduces register pressure but does not make all its operands magically free: backward still reads several inputs from SMEM. That is why simply reusing the forward pipeline would miss the dominant backward cost.

The additional lever is **2-CTA MMA**. Two CTAs work on a larger tile, partitioning the result and sharing operand staging instead of each redundantly reading all the needed data from SMEM. For one gradient multiplication ($dQ$), the necessary reduction crosses the pair: FA4 exchanges part of the intermediate $dS$ through distributed shared memory so each CTA can finish the rows it owns. In the paper's comparable per-work analysis, shared-memory cost drops from **3328 to 2688 cycles**, much closer to the **2560-cycle** MMA cost. The new arrangement also halves the number of global atomic updates used to combine $dQ$ contributions. Pairing has synchronization and data-exchange costs, so it is a targeted backward design, not a universal claim that every two-CTA kernel is faster.

![Two-panel schematic of a CTA pair. In 1-CTA mode both CTAs stage a full B operand in shared memory and write their own accumulator rows to tensor memory. In 2-CTA mode each stages half of B, the paired MMA uses both halves, and the CTAs exchange dS through distributed shared memory to complete the dQ reduction.](assets/flashattention4_cta_pair.jpg)

*Operand sharing removes duplicated SMEM reads, but $dQ$ needs a reduction along a different axis: each CTA must receive its peer's half of $dS$ for the query rows it owns. Conceptual reconstruction of the paper's Section 3.2.3, not an instruction-level trace.*

This is a deliberate trade: in the paper's 2-CTA accounting, matrix-operand SMEM reads fall from **2048 to 1536** cycles and partial-$dQ$ write/read cost from **1024 to 512**; exchanging $dS$ adds **384** cycles of distributed-SMEM traffic. The $dS$ write remains **256** cycles, giving $1536+512+384+256=2688$. Because each CTA now owns only half as many $dQ$ output rows, it also issues half as many global atomic additions. Those savings do not imply a literal $3328/2688$ runtime speedup: synchronization, other operations, and overlap still matter.

## Scheduling and reproducibility

Even a fast tile can leave the GPU underused if work is assigned poorly. With causal attention, later query blocks have more valid keys to process; with variable-length batches, different requests have different amounts of work. FA4 schedules longer jobs earlier (longest-processing-time first) while keeping related heads and batches near each other for cache reuse. This is a general scheduling idea, not specific to Blackwell. The paper reports causal scheduling gains of 4-8% for multi-head attention and 7-14% for grouped/multi-query attention in an **H200** experiment; those figures should not be mistaken for B200 hardware measurements.

The default backward pass uses global atomic additions to combine partial gradients, so floating-point addition order can vary between runs. FA4 also offers a **deterministic** backward mode: ordered, lock-based reductions give a fixed accumulation order, at a performance cost. Its scheduling reduces stalls; the paper reports up to 75% of the speed of the nondeterministic 1-CTA backward baseline for this mode. Determinism here concerns the reduction order, not a claim that all GPU training is reproducible in every setting.

## Results, scope, and takeaways

On the paper's B200 forward benchmarks with BF16 inputs, FA4 reports **1.1-1.3x** over cuDNN 9.13 and **2.1-2.7x** over its Triton baseline, reaching up to **1613 TFLOPs/s** (about **71%** of the stated theoretical BF16 peak). The runs vary sequence length from 1k to 32k, head dimension and masking, while keeping total tokens at 32k; these are **kernel throughput**, not end-to-end model speedups. The paper notes that newer cuDNN versions adopted similar techniques and can reach similar performance. Its backward plots also show gains on long sequences, but do not establish one universal backward speedup factor for every workload.

FA4 is written in Python-embedded **CuTe-DSL** and lowered to GPU code. The paper reports compiling one forward kernel in 2.5 s versus 55 s for FA3's C++ templates, and one backward kernel in 1.4 s versus 45 s. These are **compile times**, separate from runtime throughput; the paper also notes FA3 itself does not run on B200, so its compile comparison is not a B200 runtime head-to-head.

The useful mental model: FlashAttention saved slow HBM traffic; FA3 overlapped softmax and matmul; **FA4 rebalances a faster tensor core against the units around it**. In forward, keep the matrix engine busy while making softmax cheaper. In backward, reduce shared-memory reads and overlapping reductions. The most important number is not the GPU's peak FLOPs in isolation, but the time taken by the *slowest resource for a representative tile*.

## Source

- Ted Zadouri et al., [*FlashAttention-4: Algorithm and Kernel Pipelining Co-Design for Asymmetric Hardware Scaling*](https://arxiv.org/pdf/2603.05451), arXiv:2603.05451 (2026). See Sections 2-5 for the hardware model, algorithm, and benchmarks.