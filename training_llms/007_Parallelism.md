# Parallelism: Training a Model That Does Not Fit on One GPU

Every other note in this series has treated the model as something that lives on a single device: attention, normalization, activations — all described as if one GPU holds the weights, runs the forward pass, and takes the gradient step. For a modern large language model that assumption breaks almost immediately. A 70B-parameter model in mixed precision needs *hundreds* of gigabytes just for weights, gradients, and optimizer state — far more than the 80 GB of an H100 — and even when a model *does* fit, one GPU is simply too slow to churn through trillions of training tokens in any reasonable time. So we spread the work across many GPUs. That is what **parallelism** means here.

The central idea is disarmingly simple: *pick something to split across devices, and pay for the split in communication.* You can split the **batch** (each GPU sees different examples), the **weights inside a layer** (each GPU holds a slice of every matrix), the **layers themselves** (each GPU owns a contiguous depth-range of the network), the **optimizer/gradient/parameter state** (each GPU stores only its share), the **sequence** (each GPU handles a chunk of tokens), or the **experts** in a mixture-of-experts layer. Each choice removes a different bottleneck — memory or time — and introduces a different communication pattern. There is no free lunch: the art is matching the split to what is actually scarce, and arranging the communication so it overlaps with computation instead of stalling it.

This note builds that picture up strategy by strategy. It grew out of a from-scratch benchmarking suite I wrote (a 4-layer, ~2.4B-parameter MLP trained on 4×H100 with `torch.distributed`), so alongside the math and algorithms I fold in the conclusions that actually held up when measured — some of which contradict the tidy theory. We start with the collectives and the memory budget (the two things that explain *everything* downstream), then walk through data, tensor, and pipeline parallelism, the ZeRO/FSDP sharding ladder, mixed precision, sequence and expert parallelism, and finally how these compose into "3D parallelism." Where a section leans on ideas from other notes — matrix-multiply throughput on GPUs, or how mixture-of-experts routing works — I link to them rather than repeat.

Companion notes worth having open: [GPU/TPU matmul & FlashAttention](../gpu-tpu-matmul-flashattention.md) for *why* communication-vs-compute ratios decide everything, and [Mixture of Experts](../mixture-of-experts.md) for the routing machinery that expert parallelism distributes.

---

## Table of Contents

- [Setup and Notation](#setup-and-notation)
- [The Collectives: the Vocabulary of Distributed Training](#the-collectives-the-vocabulary-of-distributed-training)
- [Where the Memory Actually Goes](#where-the-memory-actually-goes)
- [The Landscape: What Each Strategy Splits](#the-landscape-what-each-strategy-splits)
- [Data Parallelism](#data-parallelism)
- [Tensor Parallelism](#tensor-parallelism)
- [Pipeline Parallelism](#pipeline-parallelism)
- [ZeRO and FSDP: Sharding the State](#zero-and-fsdp-sharding-the-state)
- [Mixed Precision: Why the Optimizer Is the Real Memory Hog](#mixed-precision-why-the-optimizer-is-the-real-memory-hog)
- [Sequence and Context Parallelism](#sequence-and-context-parallelism)
- [Expert (MoE) Parallelism](#expert-moe-parallelism)
- [Putting It Together: 3D Parallelism](#putting-it-together-3d-parallelism)
- [What the Benchmarks Actually Taught Me](#what-the-benchmarks-actually-taught-me)
- [Takeaways](#takeaways)
- [Sources](#sources)

---

## Setup and Notation

A handful of symbols recur throughout. Defining them once here keeps every later equation readable.

| Symbol | Meaning |
| --- | --- |
| $W$ | **world size** — the number of GPUs (a.k.a. devices/ranks) cooperating. In my benchmarks $W=4$. |
| rank | the integer id $0,1,\dots,W-1$ of a particular GPU. |
| $B$ | the **global batch size** — total examples per optimizer step, across all GPUs. |
| $b = B/W$ | the **per-GPU (local) batch size** under data parallelism. |
| $F$ (or $d$) | the feature / hidden dimension of a layer. |
| $L$ | the number of layers in the model. |
| $N$ | the total number of parameters. |
| $X, W_{\ell}, Y$ | a layer's input activations, weight matrix, and output. |
| $g$ | a gradient tensor (same shape as the parameter it belongs to). |

The other half of the vocabulary is the set of **collective communication operations** — the primitives every strategy is built from. They are important enough to get their own section and figure next.

---

## The Collectives: the Vocabulary of Distributed Training

Before any parallelism makes sense, you need to know the five or six ways GPUs talk to each other. These are **collective operations**: every GPU calls the same function at the same time, and data flows between all of them according to a fixed pattern. They are implemented in libraries like NCCL and are heavily optimized to use the fast interconnect (NVLink within a node, InfiniBand across nodes).

![Four panels showing GPU communication collectives. Panel 1 all-reduce: four GPUs each holding a distinct partial value g0..g3, arrows to four GPUs each now holding the same summed value Sigma. Panel 2 all-gather: four GPUs each holding one shard A,B,C,D, arrows to four GPUs each now holding all four shards ABCD. Panel 3 reduce-scatter: four GPUs with partials, arrows to four GPUs each holding one slice of the sum (SigmaA, SigmaB, SigmaC, SigmaD). Panel 4 broadcast: only GPU0 holds X, arrows to all four GPUs now holding X.](../assets/par_collectives.jpg)

- **all-reduce.** Every GPU starts with its own tensor of the same shape; afterward every GPU holds the element-wise *sum* (or average) of all of them. This is the workhorse of data parallelism — averaging gradients — and it is symmetric: everyone contributes, everyone receives the result.
- **all-gather.** Each GPU starts with a *shard* (a slice) of some tensor; afterward every GPU holds the *full concatenation* of all shards. Used to reassemble a tensor that was split across devices — e.g. re-forming a full activation from column-sharded pieces.
- **reduce-scatter.** The dual of all-gather. Every GPU starts with a full-size tensor; the operation sums them element-wise but then *scatters* the result so each GPU keeps only its $1/W$ slice of the sum. Cheaper than all-reduce when you only need your own slice back.
- **broadcast.** One GPU has the data; afterward all GPUs have a copy. Used to push a freshly-updated parameter shard out to everyone.
- **send / recv.** Point-to-point: one specific GPU sends a tensor to one specific neighbor. This is *not* collective — it is how pipeline parallelism passes activations from stage to stage.
- **all-to-all.** Every GPU sends a different chunk to every other GPU — a full transpose of who-holds-what. This is the characteristic operation of expert parallelism, where tokens must be shuffled to whichever GPU owns their assigned expert.

One identity is worth burning into memory now, because it reappears in the ZeRO section as a small miracle:

$$\text{all-reduce} \;=\; \text{reduce-scatter} \;+\; \text{all-gather}.$$

Read it operationally: to sum tensors across all GPUs and give everyone the full result (all-reduce), you can first reduce-scatter (everyone ends up with the summed slice they own) and then all-gather (everyone collects the other slices). The two-step version moves the *same total volume of data* as a direct all-reduce — and that equality is exactly why a certain kind of memory saving turns out to be "free."

A practical note: because every GPU must call a collective together, the collective *is* a synchronization point. If one GPU reaches the all-reduce late, everyone waits. You rarely need an explicit barrier around a collective — the collective already imposes one.

---

## Where the Memory Actually Goes

It is tempting to think a model's memory footprint is "the parameters." For training, that is wildly wrong, and understanding *why* is the key that unlocks every sharding strategy. Consider one training step of my ~2.4B-parameter model in the standard mixed-precision + Adam setup. Per GPU, the memory splits like this:

![Left: a bar chart of per-GPU memory of one training step for a 2.4B-parameter model in mixed precision with Adam. Bars: fp16 weights 4.5 GB, fp32 master 9.0 GB, Adam momentum 9.0 GB, Adam variance 9.0 GB, fp16 gradients 4.5 GB, activations plus overhead 18.4 GB. A bracket over the three optimizer-related bars is annotated 'Adam states = 3x the fp16 model'. Right: a pie chart of the static state that ZeRO/FSDP can shard — fp16 weights 4.5G, fp32 master 9.0G, Adam m 9.0G, Adam v 9.0G, fp16 grads 4.5G — everything except activations.](../assets/par_memory_breakdown.jpg)

Walk through the pieces, because each later strategy targets a *specific one* of them:

- **fp16 weights (~4.5 GB).** The parameters the forward pass actually multiplies with, in half precision. This is the number people quote as "model size," and it is the *smallest* contributor.
- **fp32 master copy (~9 GB).** Mixed-precision training keeps a full-precision copy of the weights that the optimizer updates, so tiny gradient steps do not vanish into fp16 rounding. Twice the size of the fp16 weights because fp32 is 4 bytes vs 2.
- **Adam momentum + variance (~9 GB + ~9 GB).** Adam maintains two running statistics *per parameter*, both in fp32. Together they are **more than three times the size of the fp16 model.** This single fact is the reason optimizer-state sharding (ZeRO) exists.
- **fp16 gradients (~4.5 GB).** One gradient per parameter, same size as the fp16 weights.
- **activations (~18 GB, and highly variable).** The intermediate tensors saved during the forward pass so the backward pass can compute gradients. This scales with batch size, sequence length, and depth — and it is the one component ZeRO/FSDP *cannot* shard, because it is not part of the model's static state. Sequence/context parallelism and activation checkpointing exist to attack this term.

The headline: for an Adam-trained model, the **optimizer states dwarf the weights**, and the *static* state (everything except activations) — weights, master copy, Adam moments, gradients — is what a sharding strategy can divide across GPUs. That static block is what the right-hand pie above represents, and it is precisely the target of the ZeRO ladder we build up later.

---

## The Landscape: What Each Strategy Splits

With collectives and the memory budget in hand, the whole zoo of parallelism strategies reduces to a single question: *which axis do you cut across?*

![A table-style diagram titled 'What does each parallelism cut across?' with six colored rows. Data Parallel (DP): splits the BATCH — each GPU has the full model, different examples. Tensor Parallel (TP): splits WEIGHTS within a layer — each GPU has a slice of every matrix. Pipeline Parallel (PP): splits LAYERS across depth — each GPU has a contiguous block of layers. ZeRO/FSDP: splits the STATE (optimizer/grad/param) — each GPU has 1/W of optimizer states, grads, params. Sequence/Context (SP/CP): splits the SEQUENCE length — each GPU has a chunk of tokens. Expert Parallel (EP): splits the EXPERTS in a mixture-of-experts layer — each GPU has a subset of experts.](../assets/par_landscape.jpg)

These are not mutually exclusive — real large-scale runs combine several at once (the "3D parallelism" section). But it helps to meet them one at a time, because each has a clean story about *what it saves* and *what it costs in communication*. Roughly:

- **DP** saves *time* (more throughput) but not memory — every GPU still holds the whole model.
- **TP and PP** save *memory* (each GPU holds only part of the model) at the cost of extra intra-step communication.
- **ZeRO/FSDP** saves *memory* by sharding the static state, and cleverly can do so at almost no extra cost when combined with DP.
- **SP/CP** save *activation memory* and unlock very long sequences.
- **EP** lets total parameter count explode while keeping per-token compute fixed.

---

## Data Parallelism

Data parallelism (DP) is the first thing anyone reaches for, and the easiest to understand. **Replicate the entire model on every GPU, hand each GPU a different slice of the batch, and average the gradients before stepping.**

![Data parallelism diagram. A 'Global batch' box of B examples fans out with arrows to four GPUs (0-3). Each GPU holds a 'batch shard' of B/W examples on top of a 'FULL model copy' with identical weights. Below all four, a wide yellow bar reads 'all-reduce gradients, then divide by W -> identical weights everywhere', with arrows from each model copy pointing down into it.](../assets/par_data_parallel.jpg)

Concretely, GPU $r$ takes the slice of the global batch from index $rb$ to $(r{+}1)b$ (with $b=B/W$), runs a full forward and backward pass on its $b$ examples, and produces a local gradient $g_r$ for every parameter. Because each GPU saw different data, these local gradients differ. To keep all replicas identical — which we must, or they would drift into $W$ different models — we average them with an all-reduce:

$$g \;=\; \frac{1}{W}\sum_{r=0}^{W-1} g_r.$$

Here $g_r$ is the gradient computed on GPU $r$'s local batch, and $g$ is the averaged gradient every GPU then uses for its optimizer step. Why is averaging the *correct* thing to do (not, say, summing)? Because the true loss is a mean over the whole batch, and the gradient of a mean is the mean of the per-example gradients. Splitting the batch across GPUs and averaging their gradients yields **exactly** the gradient you would have gotten on one GPU with the full batch — DP is mathematically identical to single-GPU training, just faster. (This exactness holds for plain summed/averaged losses; it breaks subtly for things like batch normalization whose statistics couple examples, which is one reason transformers use layer norm — see [Normalization Layers](./003_NormalisationLayers.md).)

The all-reduce is the *only* communication, and it happens once per step, right after the backward pass. It is also the synchronization point: every GPU blocks until the averaged gradient is ready, then all step in lockstep, so all replicas stay bit-for-bit identical.

**What DP buys and what it doesn't.** DP scales throughput — more GPUs chew through more examples per second — but it does *nothing* for memory: every GPU still stores the full model, full gradients, and full optimizer state. So DP alone cannot train a model that doesn't fit on one GPU; it only makes a model that *does* fit train faster. And even the speedup is conditional: the all-reduce moves a fixed volume of data ($\propto N$, the parameter count) every step, and if the per-GPU batch $b$ is small, the compute per step is too little to hide that communication. This is not a hypothetical — it is the single most surprising thing my benchmarks showed, and it gets its own treatment in [the benchmarks section](#what-the-benchmarks-actually-taught-me). The one-line version: **DP is only worth it once the batch is large enough that compute dominates communication.**

---

## Tensor Parallelism

Tensor parallelism (TP) attacks the memory problem head-on: instead of replicating each weight matrix, **split it across GPUs so each device holds only a slice.** This is *intra-layer* parallelism — a single matrix multiply is carried out cooperatively by all GPUs. The subtlety is that you can split a matrix two ways, and they have different communication signatures.

![Three panels on tensor parallelism. Panel 1 'Column split: Y = X W, W cut by columns': an X block (B x F) times a W matrix split into four colored column blocks, giving local outputs of shape B x F/W, followed by an 'all-gather -> full Y (B x F)' bar. Panel 2 'Row split: W cut by rows, X pre-sharded': four X shards times four W row blocks giving partial sums of shape B x F each, followed by an 'all-reduce -> full Y (B x F)' bar. Panel 3 'Megatron: column THEN row': a vertical stack Column (no comm) -> Row (all-reduce) -> Column (no comm) -> Row (all-reduce), with only the row steps needing an all-reduce.](../assets/par_tensor_column_row.jpg)

### Column parallelism

Take a linear layer $Y = XW$, where $X$ is the input of shape $(B, F)$ and $W$ is $(F, F)$. **Column parallelism** splits $W$ by its columns into $W$ blocks $W_0,\dots,W_{W-1}$, each of shape $(F, F/W)$, and gives block $i$ to GPU $i$. Each GPU has the *full* input $X$ (broadcast to all) and computes a local slice of the output:

$$Y_i \;=\; X\,W_i \quad\text{of shape } (B,\; F/W).$$

Every GPU now holds a different column-slice of the answer. To reconstruct the full $Y$ of shape $(B, F)$, all-gather the slices and concatenate along the feature dimension. Memory-wise this is a clean win: each GPU stores only $1/W$ of the weight matrix (in my benchmark, 25% of the 2.4B parameters per GPU).

The backward pass has a lovely property. Because each GPU owns a *distinct* column block, the gradient with respect to its own weight shard depends only on quantities it already has locally — **there is no gradient synchronization across GPUs for the weights.** In a from-scratch implementation this is expressed as a custom autograd function whose forward is "all-gather + concatenate" and whose backward is simply "keep my own chunk of the incoming gradient" (`chunk(grad, W)[rank]`). Each shard's gradient flows independently. This is the opposite of data parallelism, where gradients must be averaged.

### Row parallelism

**Row parallelism** splits $W$ by its rows: block $i$ has shape $(F/W, F)$, and correspondingly the input $X$ must already be sharded along its feature dimension into pieces $X_i$ of shape $(B, F/W)$. Each GPU computes a *partial* result of the full output shape:

$$Y_i \;=\; X_i\,W_i \quad\text{of shape } (B,\; F),\qquad Y \;=\; \sum_{i=0}^{W-1} Y_i.$$

Each $Y_i$ is a full-size but *incomplete* output — it is the contribution of only $1/W$ of the inner dimension. The true output is their sum, so we finish with an all-reduce. Row parallelism *consumes* a sharded input and *produces* a summed full output; column parallelism *consumes* a full input and *produces* a sharded output. They are duals — which is exactly what makes them chain beautifully.

### The Megatron pattern: column then row

Megatron-LM's insight is to *alternate* column and row parallelism so the sharded output of a column layer feeds directly into a row layer as its (already-sharded) input. A transformer's MLP block is two linear layers back to back; making the first column-parallel and the second row-parallel means:

- the **column** layer needs *no* communication (its output is meant to stay sharded), and
- the **row** layer's all-reduce reassembles the block's final output.

So a two-layer block costs **one all-reduce** instead of the two all-gathers a naive column-only implementation would use per layer. Stacking blocks gives the `column → row → column → row` pattern in the figure, with a collective only after each row layer.

### The counterintuitive part: fewer operations ≠ faster

Here is where my benchmarks corrected my intuition. It is natural to assume the optimized col→row pattern is "half the communication." It is not. Counting the actual bytes moved, the naive column-only version (4 all-gathers across the block) and the optimized col→row version (2 all-reduces) move the **identical total volume** — on the order of $12 \cdot B \cdot F$ elements either way. The optimization reduces the *number of collective calls*, not the *volume* of data. And volume, plus the ability to overlap it with compute, is what determines wall-clock time. Measured, the "optimized" version was only about **6.7% faster** (247.6 ms → 230.8 ms), not the 2× a naive operation-count argument would predict. The lesson generalizes: **communication *count* is not communication *time*.** Fewer, larger messages have slightly less per-call overhead, but if they carry the same bytes, they take almost the same time.

TP's real cost is that it demands a *fast* interconnect — the all-reduces sit on the critical path of every layer, so TP is almost always confined to GPUs within a single node connected by NVLink, never spread across slower network links.

---

## Pipeline Parallelism

Pipeline parallelism (PP) is *inter-layer*: instead of splitting each matrix, **give each GPU a contiguous block of layers** and stream data through them like an assembly line. GPU 0 runs layers 0–$k$, GPU 1 runs layers $k{+}1$–$2k$, and so on. Activations flow forward with point-to-point `send`/`recv`; gradients flow backward the same way. Each GPU stores only its layers' weights, so PP, like TP, cuts model memory — but along depth instead of width.

The catch is the **pipeline bubble.** If you feed a single batch through naively, then while GPU 0 computes layer 0, GPUs 1–3 sit idle waiting for its output; when the data reaches GPU 3, GPUs 0–2 are idle. Only one GPU is ever busy. Utilization is roughly $1/W$, and total time is essentially the *sum* of the per-stage times — you have paid for $W$ GPUs to get single-GPU throughput. In my benchmark the naive pipeline ran at about **0.13× the baseline** — eight times *slower* — which is exactly this bubble.

![Three Gantt-style timelines for pipeline parallelism across four GPUs (rows) over time (horizontal). Top 'Naive (1 batch)': a single forward cell F0 marches diagonally down GPUs 0->3, then a single backward cell B0 marches back up 3->0, with only one GPU busy at any moment (util ~ 1/W). Middle 'GPipe (m=4 micro-batches)': four forward cells F0..F3 fill each GPU staggered, then four backward cells B0..B3, filling most of the grid (bubble = (W-1)/(m+W-1)). Bottom '1F1B': forwards and backwards interleave so each GPU alternates one forward then one backward in steady state, same bubble as GPipe but bounded activation memory.](../assets/par_pipeline_bubble.jpg)

### Shrinking the bubble: micro-batching (GPipe)

The fix is **micro-batching.** Split the batch into $m$ micro-batches and pump them through the pipeline back to back. Once the pipe is *full*, every GPU is working on a different micro-batch simultaneously — the assembly line is finally saturated. The bubble shrinks because the only idle time is the fill-and-drain at the start and end, amortized over $m$ micro-batches. The fraction of time wasted in the bubble is approximately

$$\text{bubble fraction} \;\approx\; \frac{W-1}{m + W - 1},$$

where $W$ is the number of pipeline stages and $m$ the number of micro-batches. Read it: with $m \gg W$ the bubble becomes negligible; with $m=1$ (the naive case) it collapses to $(W-1)/W$ — almost everything wasted. This is the GPipe schedule: **all** forwards for all micro-batches, then **all** backwards.

### 1F1B: same bubble, less memory

GPipe has a memory problem: doing all forwards first means every micro-batch's activations pile up before a single backward frees them. The **1F1B** ("one-forward-one-backward") schedule, from PipeDream, interleaves them — as soon as a micro-batch finishes its forward pass through the last stage, its backward pass begins, freeing activations early. The bubble fraction is the *same* as GPipe, but the peak number of in-flight activation sets is bounded by the pipeline depth rather than by $m$. That is why production frameworks default to 1F1B (and interleaved variants): it buys GPipe's throughput at a fraction of the activation memory.

PP's communication is cheap (only activations at stage boundaries, via point-to-point) and tolerant of slower links, which is why PP is typically the strategy used to span *across nodes*, while TP stays *within* a node.

---

## ZeRO and FSDP: Sharding the State

Plain data parallelism has one glaring waste: every one of the $W$ GPUs stores a *complete, identical* copy of the model's static state — weights, gradients, and optimizer states. That is $W$ redundant copies of the very block the memory section showed to be enormous. **ZeRO** — the **Zero Redundancy Optimizer**, from Microsoft's DeepSpeed — removes exactly that redundancy: it *partitions* the static state across the data-parallel GPUs so each holds only its share, while still letting every GPU compute with the full model. PyTorch's **FSDP** (Fully Sharded Data Parallel) is the same idea, productized; I'll treat FSDP as ZeRO-3 and only sketch its extra wrinkles at the end.

The exposition here follows the framing in the [Ultra-Scale Playbook](https://huggingface.co/spaces/nanotron/ultrascale-playbook), which I found the cleanest way to see *why* each stage saves what it does.

### Memory usage, revisited (in units of Ψ)

Let $\Psi$ be the number of parameters in the model (this is the playbook's notation; it is the same $N$ from the notation table, renamed to match the source). For **mixed-precision training with Adam**, the per-GPU static memory, measured in *bytes per parameter*, breaks down as:

$$\underbrace{2\Psi}_{\text{params (bf16)}} \;+\; \underbrace{2\Psi}_{\text{grads (bf16)}} \;+\; \underbrace{4\Psi}_{\substack{\text{fp32 master}\\\text{copy}}} \;+\; \underbrace{4\Psi + 4\Psi}_{\substack{\text{Adam momentum}\\\text{+ variance (fp32)}}} \;=\; 16\Psi.$$

Reading it term by term: the bf16 weights and bf16 gradients are 2 bytes each per parameter, hence $2\Psi$ apiece. The optimizer keeps three fp32 tensors — the master weight copy and Adam's two moments — at 4 bytes each, giving $4\Psi + 4\Psi + 4\Psi = 12\Psi$. The playbook bundles those last three into a single **optimizer multiplier $k = 12$** (for Adam in mixed precision). So the total is $2\Psi + 2\Psi + k\Psi = 16\Psi$ per GPU (or $20\Psi$ if you also keep fp32 gradients for stable accumulation). The punchline is the same one from the memory section, now in clean units: **the $12\Psi$ of optimizer state is three-quarters of the footprint**, and it is pure training overhead that never enters the forward pass. That is the fat ZeRO trims.

ZeRO peels off the sharded components one at a time. With data-parallel degree $N_d$ (the number of GPUs among which we shard — the same $W$, using the playbook's symbol), the three stages give:

![Stacked bar chart 'ZeRO / FSDP: shard progressively more state, shown at N_d = 8'. Four bars in units of Psi, each stacked as params (bf16) 2 Psi + grads (bf16) 2 Psi + optimizer states (fp32) 12 Psi. Baseline DP: 16 Psi. ZeRO-1 (shard opt states): 4 Psi + 12 Psi / N_d = 5.5 Psi. ZeRO-2 (+ shard grads): 2 Psi + 14 Psi / N_d = 3.75 Psi. ZeRO-3 / FSDP (+ shard params): 16 Psi / N_d = 2 Psi. Note that activations are not shardable and are excluded.](../assets/par_zero_stages.jpg)

### ZeRO-1: partition the optimizer states

**Motivation.** The optimizer states are the biggest block ($12\Psi$) and are only touched *once per step*, during the update — they never participate in the forward or backward pass. So there is no reason for every GPU to hold all of them. ZeRO-1 gives each GPU the fp32 master copy and Adam moments for only $1/N_d$ of the parameters. Per-GPU memory drops to:

$$\underbrace{2\Psi}_{\text{params}} + \underbrace{2\Psi}_{\text{grads}} + \frac{12\Psi}{N_d} \;=\; 4\Psi + \frac{12\Psi}{N_d}.$$

Only the $12\Psi$ term is divided by $N_d$; the full bf16 weights and gradients are still replicated everywhere, because every GPU still needs the whole model to run forward/backward.

**The training step, unrolled.** The subtlety is that each GPU can only *update* the parameter shard it owns the optimizer state for, yet it needs the *full* updated weights for the next forward pass. Here is how one step flows:

![ZeRO-1 training step unrolled across 4 GPUs and 5 stages. Stage 1 Forward: every GPU holds full params. Stage 2 Backward: every GPU holds full grads. Stage 3 Reduce-scatter grads: each GPU keeps only its own 1/Nd slice of the summed gradient. Stage 4 Optimizer step: each GPU updates only its own 1/Nd shard of the fp32 params/optimizer state. Stage 5 All-gather updated params: the shards are gathered so every GPU again holds the full bf16 params. Caption: same total volume on the wire as plain DP's single all-reduce, since all-reduce = reduce-scatter + all-gather, so optimizer sharding is free.](../assets/par_zero1_steps.jpg)

1. **Forward** with the full bf16 weights (present on every GPU).
2. **Backward** produces full gradients on every GPU.
3. **Reduce-scatter** the gradients: this both *averages* them across GPUs and *scatters* the result so each GPU keeps only the $1/N_d$ gradient slice matching the parameters it owns — and frees the rest.
4. **Optimizer step** on that shard: each GPU updates its $1/N_d$ of the fp32 master weights using its Adam moments.
5. **All-gather** the updated bf16 parameter shards so every GPU again holds the full model for the next step.

**Why it's essentially free.** Notice steps 3 and 5 use *reduce-scatter* then *all-gather* — and recall the identity from the collectives section, $\text{all-reduce} = \text{reduce-scatter} + \text{all-gather}$. Plain data parallelism was *already* doing that all-reduce to average gradients; ZeRO-1 just splits it into its two halves and slots the optimizer step in between. The **total communication volume is identical to plain DP** ($2\Psi$ on the wire), so you reclaim a huge chunk of memory at *no extra communication cost*. This is the single most important practical takeaway of the whole ZeRO story, and it is why DP+ZeRO-1 is a near-universal default — my benchmarks even showed it running slightly *faster* than plain DP, because the split collectives overlap well with compute.

### ZeRO-2: add gradient partitioning

**Motivation.** After ZeRO-1, gradients are the next redundant block. But look at step 3 above: the reduce-scatter already leaves each GPU needing only its $1/N_d$ gradient slice for the update — the full gradient is never actually required. So why materialize all $2\Psi$ of it? ZeRO-2 stops storing the full gradient at all: as gradients are produced in the backward pass, they are reduce-scattered and each GPU retains only its slice, releasing the rest immediately. Per-GPU memory becomes:

$$2\Psi + \frac{2\Psi + 12\Psi}{N_d} \;=\; 2\Psi + \frac{14\Psi}{N_d}.$$

Now both the gradient ($2\Psi$) and optimizer ($12\Psi$) blocks are divided by $N_d$; only the bf16 parameters stay replicated.

**Cost.** Crucially, ZeRO-2 uses the *same* collectives as ZeRO-1 — the reduce-scatter simply happens incrementally during the backward pass instead of once at the end — so the communication volume is unchanged ($2\Psi$). ZeRO-2 gives strictly more memory savings than ZeRO-1 for **no real communication overhead**, which is why in practice you almost always prefer ZeRO-2 over ZeRO-1 once you're sharding at all.

### ZeRO-3 / FSDP: add parameter partitioning

**Motivation.** The last redundant block is the bf16 parameters themselves ($2\Psi$). ZeRO-3 shards those too, so each GPU permanently holds only $1/N_d$ of *every* tensor. Per-GPU static memory finally scales fully with $N_d$:

$$\frac{2\Psi + 2\Psi + 12\Psi}{N_d} \;=\; \frac{16\Psi}{N_d}.$$

Add more GPUs, and the per-GPU footprint keeps shrinking — in principle without bound. This is what lets you train models far larger than any single GPU could hold.

**The catch: parameters must be gathered just-in-time.** A GPU can't run layer $\ell$'s matmul with only $1/N_d$ of that layer's weights. So ZeRO-3 **all-gathers each layer's full parameters right before using them, then discards them immediately after.** This happens once in the forward pass and again in the backward pass. The single parameter all-gather of ZeRO-1/2 (done once, at the end) becomes *many* small all-gathers threaded through the whole forward and backward. The communication volume rises accordingly: roughly **$3\Psi$ per step** (two parameter all-gathers — forward and backward — plus one gradient reduce-scatter), versus $2\Psi$ for ZeRO-1/2. That extra ~50% communication is the price of the deepest memory savings.

**Hiding the cost with prefetching.** Done naively, gathering layer $\ell$'s weights and *then* computing would leave the GPU idle during every gather — communication stalling compute. The fix is **prefetching**: all-gather layer $\ell{+}1$'s parameters *while* the GPU is still computing layer $\ell$, so the gather is hidden behind useful work.

![Two timelines contrasting ZeRO-3 communication strategies. Top 'Prefetch': a compute track runs fwd L0, L1, L2, L3 back to back with no gaps, while a comm track gathers L1, L2, L3 underneath and in parallel, so the parameter all-gathers are fully hidden behind compute. Bottom 'Naive': gather L0, then compute fwd L0, then gather L1, then compute fwd L1, and so on serialized, so the GPU stalls during every gather and the total time is much longer.](../assets/par_zero_overlap.jpg)

This overlap works well as long as each layer's compute is long enough to hide its neighbor's gather — which fails if you spread the data-parallel dimension too thin (the shards get so small that gathers dominate). The playbook's rule of thumb is to keep $N_d \lesssim 512$.

**FSDP, briefly.** PyTorch's FSDP *is* ZeRO-3's algorithm, wrapped in an ergonomic API. It groups parameters into "flat" units (typically per transformer block), and around each unit it does the same gather-use-discard dance, with reduce-scatter for gradients. It adds engineering knobs — how coarsely to group parameters (the sharding granularity), how aggressively to prefetch, whether to keep some units unsharded — but the mental model is exactly the ZeRO-3 picture above: **shard everything, gather each unit just-in-time, free it right after.** The one thing to keep straight is that FSDP can get fiddly in the details (mixed-precision policies, gradient-accumulation interactions), so the safe summary is the motivation, not the low-level mechanics.

### The one thing ZeRO cannot shard

Look back at the stacked-bar figure: it is titled *static* memory. ZeRO partitions parameters, gradients, and optimizer states — the model's persistent state. It does **nothing** for **activations**, the intermediate tensors saved during the forward pass, which scale with batch size and sequence length. For long sequences activations can dominate, and no amount of ZeRO sharding touches them. That is precisely the gap the next section — sequence and context parallelism — exists to fill.

---

## Mixed Precision: Why the Optimizer Is the Real Memory Hog

Mixed precision is not itself a parallelism strategy, but it is inseparable from the memory story, and it is *why* ZeRO's savings are so pronounced in practice. The scheme, done from scratch, is:

- Keep the model **weights and gradients in fp16** (half precision) — this halves their memory and lets the hardware's fast fp16 matmul units do the heavy lifting.
- Keep an **fp32 master copy** of the weights that the optimizer actually updates. Gradient steps are tiny; adding a tiny fp32 step to an fp16 weight would round away to nothing, so the update is accumulated in fp32 and only *then* cast back to fp16 for the next forward pass.
- Keep the **Adam states (momentum, variance) in fp32** as well.

The training loop becomes: forward + backward in fp16 → copy the fp16 gradients into fp32 → optimizer step on the fp32 master → cast the updated fp32 weights back to fp16. (A real implementation also does **loss scaling** — multiply the loss by a large constant before backprop so small gradients don't underflow to zero in fp16, then divide it back out — but the memory accounting is unchanged.)

Now the punchline connecting back to ZeRO: everything that lives in **fp32** — the master copy and both Adam moments — is the bulk of the static memory, and it is *pure optimizer overhead* that never touches the forward pass. That is exactly the block ZeRO shards. So the memory win from ZeRO is small if you train in pure fp32 with plain SGD (no optimizer state to shard), but **large** in the realistic mixed-precision + Adam regime, where the shardable fp32 optimizer state is 3× the model. This is why my SGD benchmarks showed disappointing ZeRO savings while the Adam + mixed-precision benchmarks showed a dramatic ~64% reduction in static memory per GPU — the strategy only pays off when there is a lot of optimizer state to divide.

---

## Sequence and Context Parallelism

Every strategy so far splits the model or the batch. But the memory breakdown flagged one component nothing had touched — and ZeRO explicitly *couldn't* touch: **activations**, which scale with batch size *and* sequence length. For long-context training — sequences of 128k tokens and beyond — activations, not weights, become the binding constraint, and attention's cost grows with sequence length (see [Efficient Attention](./006_EffecientAttention.md)). **Sequence parallelism (SP)** and **context parallelism (CP)** split along the *token* axis to attack exactly this.

**Sequence parallelism** is a companion to tensor parallelism. TP shards the compute-heavy matmuls but leaves operations like layer norm and dropout replicated, so each GPU still materializes the *full-length* activation for those. SP shards *those* remaining operations along the sequence dimension too, so no single GPU ever holds the full-length activation. It slots into the existing TP communication with little extra cost. But SP only splits the sequence in the *non-attention* modules — inside a tensor-parallel region the attention block still sees the whole sequence, so beyond a point memory is still bounded by sequence length.

**Context parallelism** removes that last limit: it splits the sequence across GPUs for the *entire* model, attention included. For most modules this is trivial — an MLP or a layer norm processes each token independently, so a GPU can operate on its own chunk with no coordination. Attention is the hard part, because it is fundamentally *all-to-all across positions*: every query must attend to the keys and values of every token, including ones now living on other GPUs. A naive split simply doesn't have the data it needs. Ring Attention is the answer.

### Discovering Ring Attention

Arrange the GPUs in a **ring**. Each GPU holds one contiguous chunk of the sequence — its own queries $Q$, keys $K$, and values $V$. A query only ever stays on its home GPU; it is the $K,V$ blocks that *travel*. Each step of the ring does three things at once:

![Ring Attention diagram. Four GPUs (0-3) arranged in a circle, each labeled as holding its own Q,K,V chunk. Curved clockwise arrows connect them showing K,V blocks being passed to the next GPU. Center text 'each step': 1. send my K,V onward (non-blocking); 2. compute softmax(QKᵀ/√d)·V on the K,V block I currently hold; 3. receive the next K,V block. Caption: after W steps every query has seen every key; online softmax accumulates the partial outputs so no GPU ever stores the full K,V, and the sends overlap the compute.](../assets/par_ring_attention.jpg)

1. **Send** the $K,V$ block it currently holds to the next GPU in the ring — issued *non-blocking*, so the transfer happens in the background.
2. **Compute** local attention, $\mathrm{softmax}(QK^\top/\sqrt{d})\,V$, between its own queries and whichever $K,V$ block it holds *this* step.
3. **Receive** the next $K,V$ block from the previous GPU, and repeat.

After $W$ steps (one full rotation), every query has been multiplied against every key in the sequence — attention is complete, but no GPU ever had to store more than one $K,V$ block at a time. The trick that makes step 2 legal across rounds is **online softmax**: instead of needing all the scores at once to normalize, each GPU keeps a running max and running sum and folds in each new block's contribution as it arrives — the same numerically-stable streaming accumulation that makes FlashAttention work (see [Efficient Attention](./006_EffecientAttention.md)). The payoff of the non-blocking send in step 1 is **overlap**: while one GPU is busy computing on the block it has, the next block is already flying toward it over the interconnect, so the communication hides behind the compute. Context length now scales with the number of GPUs — million-token contexts that no single device could hold.

### Zig-Zag Ring Attention — a balanced compute implementation

Ring Attention as described has a subtle but serious flaw the moment you add a **causal mask** (the standard setup, where token $t$ may only attend to tokens $\le t$). With the sequence handed out *sequentially* — GPU 0 gets tokens $t_0$–$t_1$, GPU 1 gets $t_2$–$t_3$, and so on — the causal mask makes the per-GPU workloads wildly unequal.

![Two side-by-side causal attention matrices, 8 queries by 8 keys, lower-triangular (each query t attends to keys <= t). Left 'Sequential': query rows colored by owning GPU in blocks — GPU0 owns the top two rows (few active cells), down to GPU3 owning the bottom two rows (many active cells). Per-GPU work bars read 3, 7, 11, 15 — a staircase, later GPUs do far more. Right 'Zig-zag': each GPU owns one early row and one late row (GPU0 owns t0 and t7, GPU1 owns t1 and t6, etc.), so every GPU's active-cell count equals 9 — perfectly balanced.](../assets/par_ring_zigzag.jpg)

Why the imbalance? Attention's softmax runs row by row over the score matrix, and under a causal mask row $t$ has only $t{+}1$ active entries — the matrix is lower-triangular. Early tokens sit in short rows (little work); late tokens sit in long rows (much work). So the GPU holding the *last* chunk does far more computation than the GPU holding the *first*, and because the ring proceeds in lock-step, the lightly-loaded GPUs finish early and sit **idle** waiting for the stragglers. In the 8-token, 4-GPU example above, the sequential assignment gives work $[3, 7, 11, 15]$ — the last GPU does five times the first's.

**Zig-Zag Ring Attention** fixes this with nothing more than a smarter token-to-GPU assignment. Instead of contiguous chunks, hand each GPU a **mix of early and late tokens** — pair the first chunk with the last, the second with the second-to-last, and so on (GPU 0 owns $t_0$ *and* $t_7$, GPU 1 owns $t_1$ *and* $t_6$, …). Now every GPU holds one short row and one long row, and the totals equalize: $[9, 9, 9, 9]$ in the figure. The compute is perfectly balanced, the idle time vanishes, and — because attention is all-to-all anyway — the communication pattern is essentially unchanged. In practice the reshuffle is implemented either with an **all-gather** (simpler, but higher peak memory) or an **all-to-all ring exchange** (more memory-efficient, slightly more latency from the extra steps); both keep Ring Attention's overlap of communication with computation.

These are niche relative to DP/TP/PP/ZeRO — you reach for them specifically when *sequence length* is what's breaking you — but they are the standard answer to long-context training.

---

## Expert (MoE) Parallelism

A mixture-of-experts layer replaces one big feed-forward network with many smaller "expert" networks plus a **router** that sends each token to only its top-$k$ experts (usually $k=1$ or $2$). The point is *conditional computation*: total parameter count can grow enormously while the compute *per token* stays fixed, because each token only visits a couple of experts. The routing machinery itself is covered in [Mixture of Experts](../mixture-of-experts.md); here we care only about how it's parallelized.

![Expert (MoE) parallelism diagram. Four tokens on the left feed into a central 'Router (top-k)' box. The router fans out to four experts, each pinned to its own GPU (Expert 0 / GPU 0 through Expert 3 / GPU 3). A bar underneath reads 'all-to-all: route tokens to experts, then gather results back'. A side note reads 'Each GPU stores only its experts' weights -> huge param count, fixed compute per token'.](../assets/par_expert_parallel.jpg)

**Expert parallelism (EP)** places different experts on different GPUs. Since the experts are what make an MoE model huge, sharding them across devices is the natural way to fit the parameter count. The complication is that the router's assignment is *data-dependent*: a token on GPU 0 might be routed to an expert living on GPU 3. So the characteristic communication of EP is **all-to-all** — a global shuffle that sends every token to whichever GPU owns its assigned expert, runs the experts locally, then a second all-to-all to send the results back to where the tokens came from. This all-to-all is EP's defining cost, and it is sensitive to **load balance**: if the router sends too many tokens to one expert, that GPU becomes a straggler while others idle — which is why MoE training adds auxiliary load-balancing losses to keep expert usage even. EP is almost always combined with the other strategies (data parallelism over the non-expert layers, tensor parallelism within large experts), which brings us to composition.

---

## Putting It Together: 3D Parallelism

No single axis is enough for a frontier-scale model. The strategies compose, and a large training run typically uses several at once — often called **3D parallelism** when it stacks data, tensor, and pipeline parallelism (with ZeRO and expert parallelism layered on top). Conceptually the GPUs form a **device mesh**, and each parallelism dimension is one axis of that mesh.

![3D parallelism diagram. Two 'DP replica' blocks side by side, each processing a different batch shard. Within each replica is a grid: rows are pipeline stages (layers L0, L1, L2) with a green 'PP (layers)' arrow pointing down, and columns are tensor shards (shard 0, shard 1) with a purple 'TP (within layer)' arrow pointing across. A caption reads: 'Rule of thumb: TP inside a node (NVLink), PP across nodes, DP outermost; ZeRO shards state within each DP group.'](../assets/par_3d_parallelism.jpg)

The way they compose follows directly from each strategy's communication profile:

- **Tensor parallelism goes innermost, within a node.** Its all-reduces sit on the critical path of every layer and demand the fastest interconnect, so TP groups are confined to GPUs sharing NVLink.
- **Pipeline parallelism spans across nodes.** Its only communication is passing activations at stage boundaries — cheap and infrequent — so it tolerates the slower inter-node network.
- **Data parallelism is outermost.** Each DP replica is an entire TP×PP model copy processing a different batch shard, synchronizing gradients once per step.
- **ZeRO/FSDP shards the static state within each data-parallel group**, reclaiming the redundant optimizer/gradient/parameter memory as described above.
- **Expert parallelism** adds another axis when the model is MoE, distributing experts (typically with its own all-to-all group).

The design problem is a balancing act: TP degree is limited by node size (NVLink domain), PP degree by how cleanly the layers divide and how large you can make $m$ to shrink the bubble, and DP degree by how large a global batch you can tolerate before it hurts convergence. There is no universal setting — the right mesh depends on model shape, cluster topology, and interconnect speeds.

---

## What the Benchmarks Actually Taught Me

The theory above is clean. Running it on real hardware (4×H100, ~2.4B MLP) produced a few conclusions that were sharper — and occasionally contrary — to what the tidy math suggested. These are the ones worth carrying forward.

**1. Memory savings and speedups, at a glance.** Distilling the full benchmark tables into the pattern that matters:

| Strategy | Per-GPU memory vs baseline | Speedup vs baseline | One-line verdict |
| --- | --- | --- | --- |
| Data Parallel | ~1.0× (no saving) | up to ~1.95× (large batch) | Throughput, not memory |
| Tensor Parallel | ~0.3× | ~1.7× | Big memory cut, needs NVLink |
| Pipeline (naive) | ~0.29× | ~0.13× | Memory cut, but bubble kills speed |
| ZeRO-1 alone | small (SGD) → large (Adam) | ~1.0× | Free memory when combined with DP |
| DP + ZeRO-1 (Adam) | ~0.65× | ~1.0–1.3× | The default: memory *and* speed |

The standout is the bottom row: **DP+ZeRO-1 dominates plain DP** — same throughput, but ~35% less static memory — which is the "optimizer sharding for free" identity showing up in the numbers.

**2. Batch size decides whether data parallelism even helps.** This was the most counterintuitive result. At a *small* per-GPU batch, data parallelism was actually **slower than a single GPU** (~0.89×) — the all-reduce moves a fixed volume every step regardless of batch, so with little compute to hide behind, communication dominates and you are worse off than not parallelizing at all. At a *large* per-GPU batch, the same DP setup ran ~1.42× faster, because now there is enough compute per step to overlap the communication.

![A line chart 'Batch size is critical: DP only pays off once compute hides communication'. X-axis: examples per GPU on a log scale (768, 1536, 3072, 6144); Y-axis: speedup vs single GPU. Two rising lines, Data Parallel and DP + ZeRO-1, both starting near or below 1.0 at 768 examples and climbing to ~1.4x at 6144. A dotted horizontal line marks the single-GPU baseline at 1.0. A red shaded 'comm-bound (DP slower!)' region at small batch where DP dips below 1.0, and a green shaded 'compute-bound (DP wins)' region at large batch.](../assets/par_batch_scaling.jpg)

The rule of thumb this crystallized: **parallelism is a compute-vs-communication trade, and you only win when there is enough compute to hide the communication behind.** A GPU sitting half-idle on a tiny batch cannot amortize a fixed communication cost.

**3. Communication *count* is not communication *time*.** As covered in the TP section, the Megatron col→row optimization halves the *number* of collectives but moves the same *volume*, and delivered only ~6.7% speedup — not 2×. Whenever you optimize distributed code, measure bytes-on-the-wire and overlap, not the number of NCCL calls.

**4. Activation memory grows sub-linearly with batch — the "activation puzzle."** Theory says doubling the batch should roughly double activation memory. In practice, an **8× batch increase raised peak memory by only ~3.6%.** The reason is that PyTorch's memory management hides most of it: the caching allocator reuses freed blocks, in-place operations avoid new allocations, activations are released layer by layer as the backward pass consumes them, and gradient buffers are fixed-size. The practical consequence is liberating — you can usually fit a *far* larger batch than a naive "bytes per activation × batch" calculation predicts, which is often the cheapest way to push a comm-bound DP setup into the compute-bound regime where it actually pays off.

---

## Takeaways

- **Parallelism is the art of choosing what to split and paying for it in communication.** Batch (DP), weights-within-a-layer (TP), layers (PP), static state (ZeRO/FSDP), sequence (SP/CP), or experts (EP) — each removes a different bottleneck and incurs a different collective.
- **The optimizer, not the model, is the memory hog.** For Adam + mixed precision, the fp32 master copy plus momentum and variance are ~3× the fp16 weights. This is *the* reason state-sharding exists, and why sharding pays off far more with Adam than with SGD.
- **Data parallelism buys throughput, not memory — and only when the batch is large enough.** At small per-GPU batch it can be *slower* than one GPU, because a fixed-volume all-reduce has no compute to hide behind.
- **Tensor parallelism cuts model memory but lives on the critical path**, so it stays within an NVLink node; the Megatron col→row pattern reduces collective *count* but not *volume*, and thus only modestly the time.
- **Pipeline parallelism is nearly useless naive** (the bubble wastes $(W{-}1)/W$ of the time) and essential once you add micro-batching; 1F1B gives GPipe's throughput at bounded activation memory, and PP's cheap point-to-point communication lets it span nodes.
- **ZeRO/FSDP shards the static state in stages**: ZeRO-1 (optimizer states, per-GPU memory $4\Psi + 12\Psi/N_d$), ZeRO-2 (+ gradients, $2\Psi + 14\Psi/N_d$), ZeRO-3/FSDP (+ parameters, $16\Psi/N_d$). ZeRO-1 and ZeRO-2 cost the *same* communication as plain DP ($2\Psi$) — free — thanks to `all-reduce = reduce-scatter + all-gather`; ZeRO-3 costs ~50% more ($3\Psi$) for just-in-time parameter gathers, hidden by prefetching. None of them shard **activations**.
- **Sequence/context parallelism attacks activation memory** — the one thing ZeRO can't shard — for long contexts. Ring Attention rotates KV blocks around a ring (with online-softmax accumulation and comm/compute overlap); Zig-Zag Ring Attention rebalances the causal-mask workload by giving each GPU a mix of early and late tokens. **Expert parallelism** distributes MoE experts and is defined by its all-to-all shuffle and its load-balancing headache.
- **Real runs compose these into a device mesh** — TP innermost (NVLink), PP across nodes, DP outermost, ZeRO within each DP group — and the right configuration depends on model shape and cluster topology.
- **Measure volume and overlap, not operation counts.** Communication *count* is not communication *time*, and activation memory grows sub-linearly with batch — both mean the theory-predicted numbers are only a starting point.

---

## Sources

- Shoeybi et al. (2019), [*Megatron-LM: Training Multi-Billion Parameter Language Models Using Model Parallelism*](https://arxiv.org/abs/1909.08053) (the column/row tensor-parallel pattern).
- Korthikanti et al. (2022), [*Reducing Activation Recomputation in Large Transformer Models*](https://arxiv.org/abs/2205.05198) (Megatron sequence parallelism).
- Rajbhandari et al. (2019), [*ZeRO: Memory Optimizations Toward Training Trillion Parameter Models*](https://arxiv.org/abs/1910.02054) (the three ZeRO stages and the sharding-for-free idea).
- Nanotron / Hugging Face (2025), [*The Ultra-Scale Playbook: Training LLMs on GPU Clusters*](https://huggingface.co/spaces/nanotron/ultrascale-playbook) (the $\Psi$-based memory accounting, per-stage communication volumes, and prefetching — the framing this note's ZeRO section follows).
- Zhao et al. (2023), [*PyTorch FSDP: Experiences on Scaling Fully Sharded Data Parallel*](https://arxiv.org/abs/2304.11277) (FSDP, ZeRO-3 in production).
- Huang et al. (2018), [*GPipe: Efficient Training of Giant Neural Networks using Pipeline Parallelism*](https://arxiv.org/abs/1811.06965) (micro-batching and the bubble).
- Narayanan et al. (2019), [*PipeDream: Generalized Pipeline Parallelism for DNN Training*](https://arxiv.org/abs/1806.03377) (the 1F1B schedule).
- Liu et al. (2023), [*Ring Attention with Blockwise Transformers for Near-Infinite Context*](https://arxiv.org/abs/2310.01889) (context parallelism).
- Lepikhin et al. (2020), [*GShard*](https://arxiv.org/abs/2006.16668) and Fedus et al. (2021), [*Switch Transformer*](https://arxiv.org/abs/2101.03961) (expert parallelism and all-to-all routing).
- Micikevicius et al. (2017), [*Mixed Precision Training*](https://arxiv.org/abs/1710.03740) (fp16 weights, fp32 master copy, loss scaling).

Companion notes: [003 — Normalization Layers](./003_NormalisationLayers.md) (why layer norm keeps DP exact), [006 — Efficient Attention](./006_EffecientAttention.md) (the sequence-length cost that SP/CP attack), [Mixture of Experts](../mixture-of-experts.md) (the routing that EP distributes), [GPU/TPU matmul & FlashAttention](../gpu-tpu-matmul-flashattention.md) (why compute-vs-communication ratios govern everything).
