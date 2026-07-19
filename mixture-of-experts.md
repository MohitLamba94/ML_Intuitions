# Mixture of Experts (MoE): Training, Load Balancing, Parallelism, and Inference

Standard ("dense") transformers have one uncomfortable property: **every parameter is used for every token.** If you want a smarter model, you make the layers wider or deeper, and the cost of processing a single token grows in lockstep with the parameter count. Quality and compute are chained together.

Mixture of Experts breaks that chain. Inside each transformer block, instead of one big feed-forward network (FFN) that every token flows through, you keep *many* FFNs — the **experts** — and a small **router** that sends each token to just a **few** of them. The model can now hold an enormous number of parameters (all the experts together), yet any single token only pays for the handful it actually visits. This is the whole idea, and almost every term you keep hearing — *active parameters*, *load balancing*, *routing collapse*, *expert parallelism* — is a consequence of it.

This note builds the concept up from the original sparsely-gated layer, then works through the four things that make MoEs distinctive in practice: why they were unstable to train and how load balancing fixes that, why they parallelize so naturally, and how "active vs total parameters" governs both compute and memory at inference time.

---

## Table of Contents

- [The one idea: conditional computation](#the-one-idea-conditional-computation)
- [Setup and Notation](#setup-and-notation)
- [The sparsely-gated MoE layer (Shazeer et al., 2017)](#the-sparsely-gated-moe-layer-shazeer-et-al-2017)
- [Active vs total parameters (Mixtral)](#active-vs-total-parameters-mixtral)
- [Why early MoEs were unstable: routing collapse](#why-early-moes-were-unstable-routing-collapse)
- [Load balancing: the auxiliary loss and its friends](#load-balancing-the-auxiliary-loss-and-its-friends)
- [Switch Transformers: simplifying to top-1](#switch-transformers-simplifying-to-top-1)
- [Parallelism: why MoE is a natural fit](#parallelism-why-moe-is-a-natural-fit)
- [A full forward and backward pass](#a-full-forward-and-backward-pass)
- [Inference and serving](#inference-and-serving)
- [Trade-offs at a glance](#trade-offs-at-a-glance)
- [The modern landscape](#the-modern-landscape)
- [Sources](#sources)

---

## The one idea: conditional computation

Take a normal transformer block. It has two big pieces: an attention sub-layer and a feed-forward sub-layer (the FFN, usually two linear layers with a nonlinearity in between). In a large language model the FFN holds the *majority* of the parameters — often two-thirds or more. It is also where "knowledge" is widely believed to live.

Now ask: does *every* token really need the *entire* FFN? A token completing a Python function and a token in the middle of a French sentence plausibly want very different transformations. **Conditional computation** is the idea that the network should activate different parameters for different inputs, rather than forcing everything through one shared block.

MoE is the concrete realization of this for transformers. Replace the single FFN with:

- **$N$ expert FFNs**, each structurally identical to the FFN it replaced (same shape, different weights), and
- a tiny **router** (also called the gate) that looks at the token and decides which experts should handle it.

Crucially, the router picks only **$k$** experts per token (typically $k=1$ or $k=2$), not all $N$. So a layer might *contain* 8 or 64 or 256 experts, but any given token only ever runs through 1 or 2 of them. The FLOPs per token are set by $k$, while the parameter count is set by $N$. That decoupling is the entire value proposition, and it is worth stating in the two vocabulary terms you will hear constantly:

- **Total parameters** — every expert in every layer, plus the shared parts. This determines how much memory the model occupies.
- **Active parameters** — the parameters a *single token* actually touches: the shared parts plus its $k$ experts. This determines how much compute (and therefore latency) a token costs.

A dense model has active = total. An MoE deliberately makes total ≫ active. Hold on to this distinction; the whole note keeps returning to it.

![Left: a dense feed-forward block where a token flows through one large MLP, so the active parameter count equals the total. Right: a sparse MoE layer where a router scores four experts, selects the top two, and forms the output as a gate-weighted sum of only those two experts, leaving the other two unused with zero FLOPs.](./assets/moe_dense_vs_layer.jpg){ width=100% }

---

## Setup and Notation

These symbols recur throughout the note. Where a concept needs more than a line, it is explained in full where it first appears.

| Symbol | Meaning |
|---|---|
| $x$ | The hidden state of a single token entering the MoE layer (a vector of dimension $d$). This is what gets routed. |
| $d$ | Model (hidden) dimension. |
| $N$ | Number of experts in the layer. |
| $E_i(\cdot)$ | The $i$-th expert — a feed-forward network with its own weights. $E_i(x)$ is its output for token $x$. |
| $k$ | How many experts each token is routed to (the "top-$k$"). Usually 1 or 2. |
| $W_g$ | The router's weight matrix, shape $d \times N$. Maps a token to one score ("logit") per expert. |
| $h(x)$ | Router logits, $h(x) = x\, W_g$ — a length-$N$ vector, one raw score per expert. |
| $G(x)$ | The gate values — a length-$N$ vector of nonnegative weights summing to 1, mostly zeros, nonzero only for the chosen experts. $g_i(x)$ is its $i$-th entry. |
| $C$ | Expert **capacity** — the maximum number of tokens one expert will process in a batch (a fixed buffer size). |
| $f_i$ | Fraction of tokens in a batch that were routed to expert $i$ (a load measurement). |
| $P_i$ | Average router probability assigned to expert $i$ across the batch. |
| $\alpha$ | Coefficient (small) that scales the auxiliary load-balancing loss added to the main training loss. |

---

## The sparsely-gated MoE layer (Shazeer et al., 2017)

The paper that made MoE work for modern deep learning ("Outrageously Large Neural Networks") introduced the **sparsely-gated mixture-of-experts layer**. Everything later — Switch, GShard, Mixtral — is a refinement of this template. Let us build it piece by piece.

**Step 1 — score the experts.** The router is just a linear layer. It takes the token $x$ and produces one score per expert:

$$
h(x) = x\, W_g
$$

Here $h(x)$ is a vector of length $N$; its $i$-th entry $h(x)_i$ is how strongly the router thinks expert $i$ should handle this token. $W_g$ (shape $d \times N$) is the *only* new parameter the router adds — it is tiny compared to a single expert, which is why routing is nearly free.

**Step 2 — keep only the top $k$.** We do not want a soft blend over all $N$ experts; that would defeat the purpose (we'd run every expert). Instead we keep only the $k$ largest scores and treat the rest as if they were $-\infty$:

$$
\tilde h(x)_i =
\begin{cases}
h(x)_i & \text{if } h(x)_i \text{ is among the top } k \text{ scores} \\
-\infty & \text{otherwise}
\end{cases}
$$

Setting the losers to $-\infty$ is a trick: in the next step we apply a softmax, and $e^{-\infty} = 0$, so those experts get exactly zero weight and are never executed.

**Step 3 — turn the surviving scores into gate weights.** Apply a softmax over the kept scores:

$$
G(x) = \mathrm{softmax}\big(\tilde h(x)\big), \qquad
g_i(x) = \frac{e^{\tilde h(x)_i}}{\sum_{j} e^{\tilde h(x)_j}}
$$

Now $G(x)$ is a length-$N$ vector that is zero everywhere except the $k$ chosen experts, where the entries are positive and sum to 1. Each $g_i(x)$ says "how much weight this token gives to expert $i$." Because the loser logits were $-\infty$, the softmax's denominator effectively only sums over the $k$ survivors — so the gate weights are a clean probability distribution over just the chosen experts.

**Step 4 — combine the chosen experts.** The layer's output is the gate-weighted sum of the experts that actually ran:

$$
y = \sum_{i \,\in\, \text{top-}k} g_i(x)\, E_i(x)
$$

Read this carefully, because it is the heart of MoE. The sum runs *only over the $k$ chosen experts* — every other $E_j(x)$ is multiplied by a gate weight of zero, so we never even compute it. For each chosen expert we run the token through its FFN, getting $E_i(x)$, and scale that output by the gate weight $g_i(x)$. A token routed to experts 2 and 3 with weights 0.7 and 0.3 produces $y = 0.7\,E_2(x) + 0.3\,E_3(x)$. This is exactly what the right panel of the figure above shows.

**Why gate the output at all?** The multiplication by $g_i(x)$ does two jobs. First, it lets the model express *confidence* — a token that clearly belongs to one expert gets a peaky gate, an ambiguous token gets a more even split. Second, and more subtly, it is what makes the router *trainable*: the gate weight sits on the forward path, so gradients flow back through $g_i(x)$ into $W_g$, teaching the router which experts help. Without it, the top-$k$ selection (an `argmax`-like operation) would be non-differentiable and the router could not learn.

**Noisy top-$k$ gating.** Shazeer added one more wrinkle. Before taking the top-$k$, they add tunable Gaussian noise to the logits:

$$
h(x)_i = (x\, W_g)_i + \varepsilon_i \cdot \mathrm{softplus}\big((x\, W_{\text{noise}})_i\big), \qquad \varepsilon_i \sim \mathcal{N}(0,1)
$$

The noise term is scaled by a second small learned matrix $W_{\text{noise}}$ (so the model controls *how much* noise each expert gets), and $\mathrm{softplus}$ just keeps that scale positive. Why add noise at all? Early in training the router is essentially random, and whichever experts happen to win the first few rounds start getting all the gradient signal, get better, and win even more — a rich-get-richer spiral. The noise jitters the rankings so that experts near the decision boundary sometimes get selected too, giving the underused ones a chance to receive tokens and improve. It is a load-spreading and exploration device. This is our first hint of the central difficulty in MoE training, which the next sections tackle head-on.

**What this bought.** With this layer Shazeer et al. trained models with up to ~137 billion parameters (astonishing for 2017) on language modeling and translation, while keeping the *per-example* compute modest because only a couple of experts ran per token. That is conditional computation delivering on its promise.

---

## Active vs total parameters (Mixtral)

Mixtral 8x7B (Mistral AI, 2024) is the cleanest concrete example of the active-vs-total distinction, so it is worth grounding the abstraction in its actual numbers.

The name "8x7B" is slightly misleading. It does **not** mean $8 \times 7 = 56$ billion parameters. Here is why. In Mixtral, only the **FFN** sub-layer of each block is turned into a mixture of 8 experts. The attention layers, embeddings, and normalization layers are **shared** — there is exactly one copy, used by every token. So when you replicate "an expert" 8 times, you are only replicating the FFN portion, not the whole 7B model. The arithmetic works out to roughly:

- **Total parameters ≈ 47B** — the shared attention/embeddings *plus* all 8 experts in every layer. This is what sits in memory.
- **Active parameters ≈ 13B** — the shared parts *plus* the 2 experts each token actually uses ($k=2$). This is what a single token computes.

So Mixtral runs at the *speed and compute cost* of a ~13B dense model, but has the *capacity* (and much of the quality) of something far larger. On many benchmarks it matches or beats a 70B dense model while doing a fraction of the per-token math.

![Left: a stacked bar showing Mixtral's total of about 47 billion parameters made of a small shared attention/embedding base plus eight expert blocks, next to a shorter bar for the roughly 13 billion active parameters per token, which is the shared base plus only two experts. Right: two labeled boxes stating that total parameters drive VRAM/memory because every expert must be stored, while active parameters drive compute/latency because they equal the FLOPs of a dense k-expert model.](./assets/moe_active_vs_total.jpg){ width=100% }

The practical rule this establishes — and it is the single most useful thing to remember about MoE — is:

$$
\boxed{\text{TOTAL params} \rightarrow \text{VRAM} \qquad\qquad \text{ACTIVE params} \rightarrow \text{compute / latency}}
$$

You must have enough memory to store *all* the experts, because you never know in advance which ones a batch of tokens will need — so all of them sit in VRAM. But the FLOPs you burn per token, and hence your latency, track only the *active* count. An MoE gives you a large model's quality at a small model's compute, provided you can pay the memory bill. This tension is the theme of the [inference](#inference-and-serving) section.

---

## Why early MoEs were unstable: routing collapse

The elegant picture above hides a real problem, and it is the reason MoEs earned a reputation for being finicky to train. The router and the experts are learned *together, from scratch*, and their learning dynamics create a vicious feedback loop.

Here is the loop. At initialization the router is random, so some experts get slightly more tokens than others by pure chance. Those experts see more data, so they train faster and become more useful. Because they are more useful, the router (which is being trained to send tokens where they help most) learns to send them *even more* tokens. Meanwhile the neglected experts get little data, barely improve, and so the router sends them even less. Within a short time a few experts hog nearly all the traffic and the rest are effectively dead weight.

This failure mode is called **routing collapse** (or expert imbalance / expert starvation). It is bad for two compounding reasons:

1. **Wasted capacity.** You are paying to store dozens of experts but only a handful ever fire. All that parameter budget — the entire reason you built an MoE — is squandered.
2. **A hard capacity wall.** As we will see, each expert has a fixed buffer (its *capacity*). When one expert is flooded, its buffer overflows and the surplus tokens get **dropped** — passed through without being processed by any expert. Dropped tokens get no useful transformation, which hurts quality and destabilizes training further.

![Two bar charts of how tokens spread across eight experts. Left, labeled routing collapse: expert 1 receives about 46 percent of tokens and expert 2 about 28 percent while the remaining experts receive almost none, far from the dashed uniform line at 12.5 percent. Right, labeled balanced routing with an auxiliary loss: every expert receives close to the uniform 12.5 percent, so all capacity is used.](./assets/moe_routing_collapse.jpg){ width=100% }

The noisy gating from Shazeer's paper helps a little by jittering the rankings, but noise alone is not enough at scale. What actually tamed MoE training was adding an explicit *pressure* toward balance — a load-balancing loss — which is the subject of the next section.

---

## Load balancing: the auxiliary loss and its friends

The fix for routing collapse is to add a term to the training objective that *penalizes imbalance*, nudging the router toward spreading tokens evenly across experts. This extra term is called the **auxiliary loss** (or load-balancing loss). It is added to the ordinary language-modeling loss:

$$
\mathcal{L}_{\text{total}} = \mathcal{L}_{\text{LM}} + \alpha \cdot \mathcal{L}_{\text{balance}}
$$

where $\alpha$ is a small coefficient. The whole design challenge is making $\mathcal{L}_{\text{balance}}$ (a) small when load is even, (b) large when it is skewed, and (c) differentiable so gradients can push the router toward balance.

**The Switch Transformers formulation.** The cleanest, most-copied version comes from Switch Transformers. For a batch of $T$ tokens and $N$ experts, define two quantities per expert:

- $f_i$ = the **fraction of tokens** in the batch that were dispatched to expert $i$ (a hard count divided by $T$).
- $P_i$ = the **average router probability** assigned to expert $i$, i.e. the mean of the softmax gate value for expert $i$ over all tokens in the batch.

The loss is:

$$
\mathcal{L}_{\text{balance}} = \alpha \, N \sum_{i=1}^{N} f_i \, P_i
$$

Let us unpack why this simple product does the job. Both $f_i$ and $P_i$ are, ideally, $1/N$ when load is perfectly uniform (every expert gets its equal share of tokens *and* equal average probability). Their sum $\sum_i f_i P_i$ is minimized exactly when both are flat at $1/N$, and it grows whenever the router piles probability and tokens onto a few experts. The factor $N$ just rescales the loss so it stays around 1 regardless of how many experts you have, which keeps $\alpha$ meaningful across model sizes.

**Why the product of two terms, and not just one?** This is the clever part. $f_i$ is a *count* — it involves the discrete top-$k$ selection, so it has no useful gradient (you cannot differentiate "which expert won"). $P_i$ is the *soft* router probability, which is fully differentiable. By multiplying them, the loss uses $f_i$ as a (constant, gradient-free) *weight* that measures the actual observed load, and lets the gradient flow through $P_i$. The effect: for an expert that is currently overloaded (large $f_i$), the loss pushes its router probability $P_i$ *down*; for a starved expert, there is little penalty and its probability is free to rise. Over many steps this steadily equalizes the load. The figure below shows this for the two-expert case — the loss is a bowl with its minimum exactly at the balanced split.

![A U-shaped curve of the auxiliary load-balancing loss for two experts as a function of the fraction of tokens sent to expert 1. The curve reaches its minimum at the balanced 50/50 split, marked in green, and rises to its maximum at the extremes where all tokens go to a single expert, marked in red as collapse.](./assets/moe_load_balancing.jpg){ width=72% }

**Capacity factor and token dropping.** Balance alone is not enough for efficient hardware execution. On a GPU/TPU you want fixed-shape tensors, so each expert is given a fixed **capacity** $C$ — the maximum number of tokens it will accept in a batch. Capacity is set by a *capacity factor*:

$$
C = \text{capacity factor} \times \frac{\text{tokens per batch}}{N}
$$

A capacity factor of 1.0 gives each expert exactly its fair share of buffer; 1.25 gives 25% slack for uneven batches. If more tokens route to an expert than its capacity allows, the overflow tokens are **dropped** — they skip the expert entirely and are carried forward by the residual connection unchanged. This is the throughput-vs-quality knob: a larger capacity factor drops fewer tokens (better quality) but wastes compute and memory on padding (lower throughput); a smaller factor is cheaper but risks dropping more tokens. Good load balancing keeps drop rates low even at modest capacity factors, which is exactly why the auxiliary loss matters so much.

**Router z-loss.** A second, smaller stabilizer often added on top (introduced in the ST-MoE work) is the **router z-loss**, which penalizes the router logits from growing too large in magnitude:

$$
\mathcal{L}_z = \frac{1}{T}\sum_{t=1}^{T}\left(\log \sum_{i=1}^{N} e^{h(x_t)_i}\right)^2
$$

Large logits make the softmax razor-sharp and the numerics fragile (exponentials of big numbers overflow, gradients spike). By keeping the log-sum-exp of the logits small, the z-loss keeps the router well-conditioned. It is cheap and improves training stability noticeably. Together, the load-balancing loss (fixes *what* the router does) and the z-loss (fixes *how confidently* it does it) are the workhorses that turned MoE from "unstable research curiosity" into "trainable at scale."

---

## Switch Transformers: simplifying to top-1

Shazeer's original layer used $k > 1$ (route to at least two experts), partly out of a belief that the router needed to compare at least two options to get a useful training signal. Switch Transformers (Fedus et al., 2021) challenged this and showed that **top-1 routing — sending each token to a single expert — works fine**, provided you add the load-balancing loss and a few stability fixes.

Why does this matter? Top-1 is the cheapest possible MoE:

- **Less compute.** Each token runs through exactly one expert, so active parameters are minimized for a given expert size.
- **Less communication.** As the [parallelism](#parallelism-why-moe-is-a-natural-fit) section explains, tokens must be shipped across devices to reach their expert. With $k=1$ each token is sent to exactly one place, halving the routing traffic compared to $k=2$.
- **Simpler code.** No need to combine multiple expert outputs; the "weighted sum" is just a single scaled expert output.

Switch's recipe to make top-1 stable was: (1) the auxiliary load-balancing loss above; (2) a **capacity factor** with token dropping to keep tensors fixed-shape; and (3) training in reduced precision carefully — they found that computing the *router* in full `float32` (even when the rest of the model is `bfloat16`) removes a major source of instability, because the router's softmax is sensitive to rounding. With these, they scaled to over a **trillion** total parameters while keeping per-token compute low. Switch is the reason "just use top-1 with an aux loss" became the default mental model for MoE. (Mixtral later went back to $k=2$ for a quality boost; the point is that both are viable and the choice is a compute-vs-quality trade-off, not a correctness one.)

---

## Parallelism: why MoE is a natural fit

You keep hearing that MoEs are "great for parallelism." To see why, first recall the three standard ways to split a *dense* model across devices:

- **Data parallelism** — every device holds a full copy of the model and processes a different slice of the batch; gradients are averaged (an all-reduce) each step. Simple, but every device must fit the whole model.
- **Tensor parallelism** — a single big matrix multiply is split across devices (each holds a slice of the weight matrix); results are combined with an all-reduce *within every layer*. Lets you exceed one device's memory, but the per-layer all-reduce is heavy communication.
- **Pipeline parallelism** — different *layers* live on different devices, and micro-batches flow through the pipeline. Memory-efficient, but introduces pipeline "bubbles" (idle time).

(For the hardware reasons behind why communication, not FLOPs, is often the bottleneck, see `gpu-tpu-matmul-flashattention.md`.)

**Expert parallelism (GShard).** MoE unlocks a fourth axis that fits its structure perfectly. Since the experts are *independent* FFNs, you can simply place **different experts on different devices** — e.g. expert 0 on GPU 0, expert 1 on GPU 1, and so on. This is **expert parallelism**, introduced at scale by GShard (Lepikhin et al., 2020).

Why is this such a good fit? Because of the active-vs-total decoupling. Each token only visits $k$ experts, so even though the *total* parameter count grows linearly as you add experts (and devices), the *per-token compute* stays fixed. You can scale a model to hundreds of billions or trillions of parameters by just adding more devices, each holding a few more experts, without any single token doing more work. Contrast this with tensor parallelism, where splitting one matmul finer and finer eventually gives diminishing returns and rising communication. Expert parallelism scales the model by scaling the *number of independent things*, which is exactly what a fleet of accelerators is good at.

**The catch: all-to-all communication.** There is no free lunch. A token's hidden state lives on the device that computed the previous layer, but the expert it needs may live on a *different* device. So each MoE layer requires two collective communication steps:

1. **Dispatch (all-to-all):** every device sends each of its tokens to whichever device holds that token's chosen expert. After this shuffle, each device has gathered exactly the tokens its local experts must process.
2. **Combine (all-to-all):** after the experts run, the outputs are shuffled *back* to each token's original device so the rest of the layer can proceed.

![Four devices, each holding one expert (top) and a batch of local tokens (bottom). A shaded middle band shows two collective operations: an all-to-all dispatch that sends each token up to the device holding its chosen expert, and an all-to-all combine that shuffles the expert outputs back down to each token's home device. Crossing colored arrows illustrate tokens from every device being routed to experts on other devices. A caption notes that adding experts means adding devices while per-token compute stays fixed, and that the two all-to-alls are the main communication cost.](./assets/moe_expert_parallelism.jpg){ width=100% }

An **all-to-all** is a collective where every device simultaneously sends a (different) chunk of data to every other device. It is the natural primitive for "everyone reshuffle your tokens by destination expert," and it is the dominant communication cost of an MoE layer. Note it is a *different* pattern from tensor parallelism's all-reduce: an all-reduce sums the same-shaped tensor across devices, while an all-to-all *permutes* data between them. The two all-to-alls per MoE layer are the price of expert parallelism, and much MoE systems work goes into overlapping them with computation and keeping them cheap.

**Automatic sharding.** GShard's other contribution was engineering: rather than hand-writing all this dispatch/combine logic, it lets you *annotate* tensors with how they should be split, and the compiler (XLA) automatically generates the sharded computation and the all-to-alls. This made training giant MoEs practical without rewriting the model for every device topology. In practice, MoE training combines expert parallelism (for the experts) with data parallelism (for the shared attention layers), and often tensor/pipeline parallelism too — the experts scale on one axis while the dense parts scale on the others.

---

## A full forward and backward pass

Putting the pieces together, here is what one MoE layer does for a batch of tokens during training. Following the chain end to end makes the moving parts concrete:

1. **Attention (shared).** Tokens pass through the ordinary, shared attention sub-layer — no MoE here.
2. **Route.** The router computes logits $h(x) = x\,W_g$ for every token, adds noise (if used), and picks the top-$k$ experts per token.
3. **Capacity check.** Tokens are assigned to their experts' buffers. If an expert is over capacity $C$, the overflow tokens are dropped (carried by the residual).
4. **Dispatch (all-to-all).** Each token is sent to the device holding its chosen expert(s).
5. **Expert compute.** Each expert runs its FFN on the tokens it received — a batch of ordinary matmuls, done in parallel across devices.
6. **Combine (all-to-all).** Expert outputs are shuffled back to each token's home device.
7. **Weighted sum.** Each token forms $y = \sum_{i \in \text{top-}k} g_i(x)\,E_i(x)$ and adds the residual.
8. **Loss.** The main language-modeling loss is computed, and the **auxiliary load-balancing loss** (and z-loss) are added: $\mathcal{L}_{\text{total}} = \mathcal{L}_{\text{LM}} + \alpha\,\mathcal{L}_{\text{balance}} + \beta\,\mathcal{L}_z$.

The **backward pass** mirrors this. Gradients flow through the expert FFNs as usual, and — this is the important part — back through the gate weights $g_i(x)$ into the router matrix $W_g$, teaching the router which choices reduced the loss. The auxiliary loss contributes an *additional* gradient into $W_g$ that pushes toward balanced load. So the router is being trained by two forces at once: "send tokens where they help the LM objective" and "don't overload any single expert." The small $\alpha$ ensures the balancing pressure regularizes routing without overwhelming the primary objective.

---

## Inference and serving

At inference the active-vs-total story returns with a vengeance, because the two numbers now pull in opposite directions.

**Memory is set by total parameters.** To serve the model you must hold *every* expert in memory (VRAM), because any incoming token might route to any of them and you cannot know in advance. Mixtral's ~47B parameters must all be resident even though a token only uses ~13B of them. So an MoE is far more **memory-hungry** than a dense model of equivalent *speed*. This is often the binding constraint: an MoE that runs as fast as a 13B dense model still needs the VRAM of a ~47B model.

**Compute is set by active parameters.** Each generated token only runs through its $k$ experts, so the arithmetic (and thus latency, when compute-bound) matches a much smaller dense model. This is why MoEs are attractive for high-throughput serving *if* you can afford the memory.

**The batching wrinkle.** In dense inference, batching is trivial — every token does the same work. In an MoE, different tokens in a batch route to different experts, so the work is *uneven*: one expert might receive many tokens while another receives none. Efficient serving therefore needs **expert-aware batching / scheduling** — grouping tokens by destination expert, padding to capacity, and keeping expert GPUs busy. Because only a slice of the weights is touched per token but *all* weights must be loaded and reachable, MoE inference tends to be **memory-bandwidth-bound** rather than compute-bound: you spend a lot of time moving expert weights relative to the math you do on them. This is the opposite of the usual dense-model regime and is why MoE inference backends (and techniques like expert offloading to CPU/host memory, or quantizing experts) are an active engineering area. The Hugging Face "MoEs in Transformers" writeup goes into the execution-backend details; the one-line summary is: *active parameters dictate the FLOPs, total parameters dictate the memory, and serving well means managing the gap between them.*

---

## Trade-offs at a glance

Pulling the threads together, here is when MoE helps and what it costs.

**Why MoE wins.**
- **Better quality per unit of training/inference compute.** For a fixed FLOPs budget you can hold many more parameters, and more parameters (used sparsely) generally means better quality. MoEs reach a target quality with less compute than a dense model, or exceed a dense model at equal compute.
- **Scales by adding independent parts.** Growing the model = adding experts = adding devices, with near-constant per-token compute — a clean fit for large accelerator fleets.

**What it costs.**
- **High memory.** Total parameters must all be stored, so VRAM requirements are those of the full (large) model even though speed is that of a small one.
- **Communication overhead.** The two all-to-alls per layer add network cost and complexity; systems work is needed to hide it.
- **Training complexity.** Load balancing, capacity tuning, router precision, and z-loss are all extra knobs; get them wrong and you get routing collapse or dropped tokens.
- **Fine-tuning is harder.** MoEs are prone to **overfitting** during fine-tuning (lots of capacity, comparatively little downstream data) and the router can destabilize. Practical recipes often freeze or lightly regularize the experts and tune the shared/attention parts more.

The Hugging Face "Mixture of Experts Explained" post frames the headline trade-off well: MoEs are **pretraining-efficient** (cheaper to reach a given quality) but carry **inference/serving overhead** (memory and complexity). Whether that trade is worth it depends on whether you are memory-rich and throughput-hungry.

---

## The modern landscape

The template above is stable, but two refinements show up in the strongest recent MoEs and are worth knowing.

**Fine-grained + shared experts (DeepSeekMoE).** Two ideas, usually together:

- *Fine-grained experts:* instead of a few large experts, use *many small* ones and route to more of them (e.g. split each expert into several thinner ones and pick top-6 of 64 rather than top-2 of 8). Smaller experts can specialize more sharply, and having more of them to combine gives the router finer control — more distinct routing combinations for the same active-parameter budget.
- *Shared experts:* designate a small number of experts as **always-on** — every token goes through them in addition to its routed experts. The intuition is a division of labor: the shared experts absorb the *common* knowledge that every token needs (so the specialists don't each have to relearn it redundantly), freeing the routed experts to capture genuinely specialized patterns. This also improves balance, since common knowledge no longer creates pressure to overload one popular "generalist" expert.

This design (used in DeepSeek's MoE models, including DeepSeek-V3) tends to get more specialization and better balance out of the same parameter budget.

**Expert-choice routing.** The standard router is *token-choice*: each token picks its top-$k$ experts. This is what makes load balancing hard — nothing stops many tokens from picking the same expert, hence the auxiliary loss. **Expert-choice routing** flips the direction: each *expert* picks its top-$C$ tokens (up to its capacity). Because every expert selects exactly a fixed number of tokens, **load is perfectly balanced by construction** — no auxiliary loss needed. The trade-off is that a token is no longer guaranteed a fixed number of experts: a "popular" token might be picked by several experts while an "unpopular" one is picked by none (relying on the residual). It is an elegant way to sidestep routing collapse, at the cost of this uneven per-token treatment.

**Where MoE is today.** Sparse MoE has moved from research curiosity to production mainstream. Mixtral demonstrated a clean, open, top-2 MoE at usable scale; DeepSeek-V3 and other frontier models use fine-grained shared-expert MoEs at hundreds of billions of total parameters with only tens of billions active. The recurring theme across all of them is the same one this note started with: **decouple capacity from compute, then spend engineering effort on keeping the router balanced and the communication cheap.**

---

## Sources

- **Outrageously Large Neural Networks: The Sparsely-Gated Mixture-of-Experts Layer** — Shazeer et al., 2017. The original sparsely-gated MoE layer, noisy top-$k$ gating, and conditional computation.
- **GShard: Scaling Giant Models with Conditional Computation and Automatic Sharding** — Lepikhin et al., 2020. Expert parallelism, all-to-all dispatch/combine, automatic sharding.
- **Switch Transformers: Scaling to Trillion Parameter Models with Simple and Efficient Sparsity** — Fedus et al., 2021. Top-1 routing, the auxiliary load-balancing loss, capacity factor, router precision.
- **Mixtral of Experts** — Jiang et al., 2024. A production top-2 MoE; the clearest illustration of active vs total parameters.
- **Mixture of Experts Explained** — Hugging Face blog, Dec 2023. Accessible overview; pretraining efficiency vs inference overhead trade-offs.
- **Mixture of Experts (MoEs) in Transformers** — Hugging Face blog, Feb 2024. Execution abstractions, inference backends, active vs total parameters for compute vs VRAM.
- **ST-MoE: Designing Stable and Transferable Sparse Expert Models** — Zoph et al., 2022. Router z-loss and stability/fine-tuning practices (referenced for the z-loss).
- **DeepSeekMoE / DeepSeek-V3** — for fine-grained and shared experts.
