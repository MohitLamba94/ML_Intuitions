# Kimi K3: A Field Guide to Frontier LLM Training

Most model reports tell you *what* a lab built. The **Kimi K3** technical report (Kimi Team, arXiv:2607.24653, July 2026) is more useful than that: it is a near-complete tour of *how the frontier is built in 2026*. To ship a 2.8-trillion-parameter, natively multimodal, million-token model, the Kimi team had to make a decision at almost every layer of the stack — the attention mechanism, the feed-forward mechanism, how residuals flow, how experts are routed and balanced, which optimizer, which numeric format, how to do RL over hundred-thousand-step agent trajectories, how to keep 51 million sandboxes alive, how to cache a million-token prefix. Each decision cites the paper that introduced the idea. Read end to end, the report is a **literature survey disguised as a model card**, which is exactly why it is worth a long note.

This note is that long note. It walks the whole paper, and wherever Kimi K3 leans on a prior idea — the delta rule, MLA, DeepSeekMoE, Muon, MXFP4, EAGLE-3, and a dozen others — it stops and explains the prior idea first, because you cannot understand *why* Kimi K3 changed something without knowing what it changed. The goal is that after reading this you could hold your own in a conversation with someone who works on frontier training. It is intuition-first; the equations are all here, but each one is unpacked term by term.

A map of the model in one breath: along the **sequence** axis, a hybrid of Kimi Delta Attention (a linear-attention RNN) and periodic Gated MLA (global softmax attention with no positional encoding); along the **depth** axis, Attention Residuals that let a layer attend back over earlier layers instead of just adding to a running sum; along the **width** axis, Stable LatentMoE with 896 experts of which 16 fire per token. Wrapped around that: Muon as the optimizer, MXFP4 quantization-aware training, a nine-way RL-then-distill post-training pipeline, and an infrastructure section that is a systems paper in its own right.

If some of the attention and MoE machinery is new to you, the sibling notes are good warm-ups: [Attention (005)](./training_llms/005_Attention.md) and [Efficient Attention (006)](./training_llms/006_EffecientAttention.md) for the sequence axis, [Mixture of Experts](./mixture-of-experts.md) for the width axis, [Optimisers](./optimisers.md) for Muon's lineage, [Floating Point Precision](./floating_precison.md) for the MXFP4 story, [Gated Attention](./gated-attention.md) for the output-gate idea that recurs everywhere here, and [Scaling Laws (008)](./training_llms/008_ScalingLaws.md) for the efficiency framing.

---

## Table of Contents

- [Setup and Notation](#setup-and-notation)
- [The big picture: what Kimi K3 is](#the-big-picture-what-kimi-k3-is)
- [Three axes of information flow](#three-axes-of-information-flow)
- [Kimi Delta Attention: linear attention that actually remembers](#kimi-delta-attention-linear-attention-that-actually-remembers)
  - [From softmax attention to linear attention](#from-softmax-attention-to-linear-attention)
  - [The delta rule: write by correcting your error](#the-delta-rule-write-by-correcting-your-error)
  - [Gating: let the memory forget](#gating-let-the-memory-forget)
  - [KDA proper: a channel-wise decayed delta rule](#kda-proper-a-channel-wise-decayed-delta-rule)
  - [Lower-bounded decay: the sigmoid that saves the Tensor Cores](#lower-bounded-decay-the-sigmoid-that-saves-the-tensor-cores)
  - [The full-rank output gate](#the-full-rank-output-gate)
- [Gated MLA: the global-attention half of the hybrid](#gated-mla-the-global-attention-half-of-the-hybrid)
  - [What MLA is (DeepSeek-V2)](#what-mla-is-deepseek-v2)
  - [NoPE: no positional encoding at all](#nope-no-positional-encoding-at-all)
  - [The output gate and the FP32 fix](#the-output-gate-and-the-fp32-fix)
- [Attention Residuals: attention over depth](#attention-residuals-attention-over-depth)
  - [The full form](#the-full-form)
  - [The block form (what Kimi K3 actually uses)](#the-block-form-what-kimi-k3-actually-uses)
- [Stable LatentMoE: scaling to 896 experts](#stable-latentmoe-scaling-to-896-experts)
  - [A quick MoE refresher](#a-quick-moe-refresher)
  - [LatentMoE: experts in a compact latent space](#latentmoe-experts-in-a-compact-latent-space)
  - [Fix 1a — Normalized LatentMoE](#fix-1a--normalized-latentmoe)
  - [Fix 1b — SiTU-GLU: a soft-capped activation](#fix-1b--situ-glu-a-soft-capped-activation)
  - [Fix 2 — Quantile Balancing (QB)](#fix-2--quantile-balancing-qb)
  - [Estimating the quantile with a histogram](#estimating-the-quantile-with-a-histogram)
- [Native vision: MoonViT-V2 from scratch](#native-vision-moonvit-v2-from-scratch)
- [Per-Head Muon](#per-head-muon)
- [Pre-training: data, scaling laws, and long context](#pre-training-data-scaling-laws-and-long-context)
  - [Data and rephrasing](#data-and-rephrasing)
  - [Scaling laws: cosine beats WSD](#scaling-laws-cosine-beats-wsd)
  - [The training recipe](#the-training-recipe)
  - [Reaching one million tokens](#reaching-one-million-tokens)
- [Post-training: SFT, RL, and distillation](#post-training-sft-rl-and-distillation)
  - [The three-stage pipeline](#the-three-stage-pipeline)
  - [Reinforcement learning across domains and efforts](#reinforcement-learning-across-domains-and-efforts)
  - [Reasoning-effort RL and the generative reward model](#reasoning-effort-rl-and-the-generative-reward-model)
  - [Multi-Teacher On-Policy Distillation](#multi-teacher-on-policy-distillation)
  - [Deployment-aware post-training: MXFP4 and a draft model](#deployment-aware-post-training-mxfp4-and-a-draft-model)
- [Building RL environments and tasks](#building-rl-environments-and-tasks)
- [Infrastructure I: making KDA fast](#infrastructure-i-making-kda-fast)
  - [KDA Context Parallelism (KCP)](#kda-context-parallelism-kcp)
- [Infrastructure II: training a 2.8T model](#infrastructure-ii-training-a-28t-model)
  - [MoonEP: perfectly balanced expert parallelism](#moonep-perfectly-balanced-expert-parallelism)
  - [Fitting it in memory](#fitting-it-in-memory)
  - [Hiding the vision encoder](#hiding-the-vision-encoder)
- [Infrastructure III: million-token agentic RL](#infrastructure-iii-million-token-agentic-rl)
  - [AgentENV: resumable microVM sandboxes](#agentenv-resumable-microvm-sandboxes)
- [Infrastructure IV: serving the model](#infrastructure-iv-serving-the-model)
  - [KDA-aware prefix caching](#kda-aware-prefix-caching)
  - [Kernels and fleet scheduling](#kernels-and-fleet-scheduling)
- [How good is it? Evaluations](#how-good-is-it-evaluations)
  - [Internal benchmarks](#internal-benchmarks)
  - [Cyber-security evaluation](#cyber-security-evaluation)
  - [Third-party and cost efficiency](#third-party-and-cost-efficiency)
- [Case studies: the model builds its own tools](#case-studies-the-model-builds-its-own-tools)
- [The chat template (XTML)](#the-chat-template-xtml)
- [Takeaways](#takeaways)
- [Sources](#sources)

---

## Setup and Notation

The paper reuses a compact set of symbols across very different subsystems. Each is reintroduced where it does real work, but here they are in one place so nothing below reads like an alien.

| Symbol | Meaning |
| --- | --- |
| $x_t \in \mathbb{R}^d$ | the hidden state (token representation) at sequence position $t$; $d$ is the model width (7,168 in Kimi K3) |
| $q_t, k_t \in \mathbb{R}^{d_k}$ | query and key vectors for one attention head; $d_k$ is the key/query head width |
| $v_t \in \mathbb{R}^{d_v}$ | the value vector for one head; $d_v$ is the value head width |
| $S_t \in \mathbb{R}^{d_k \times d_v}$ | the **recurrent state matrix** of a linear-attention head — a fixed-size memory that replaces the growing KV cache |
| $\alpha_t \in (0,1)^{d_k}$ | the **channel-wise forget gate** (retention factor): how much of each key-channel of the state survives one step |
| $\beta_t \in (0,1)$ | the **delta-rule write strength**: how aggressively the current token overwrites the state |
| $\odot$ | elementwise (Hadamard) product |
| $W_\downarrow, W_\uparrow$ | down- and up-projections between the full width $d$ and the compact latent width $\ell$ used by the routed experts |
| $\sigma(\cdot)$ | the logistic sigmoid, $\sigma(z) = 1/(1+e^{-z})$ |
| $\mathrm{RMSNorm}(\cdot)$ | root-mean-square normalization — rescales a vector by its RMS, no mean-subtraction, one learned gain per channel |
| $E$, $R$ | number of experts, and the expert-parallel world size (number of ranks), in the MoE infrastructure |
| $\pi_\theta$, $\pi_{\text{teacher}}$ | the student (trained) policy and a teacher policy, in the RL/distillation stage |
| $\mathrm{sg}(\cdot)$ | the stop-gradient operator: forward value unchanged, gradient blocked |

Two conventions worth stating up front. First, "**head**" always means one attention head; both attention variants split the width into many heads (96 in Kimi K3). Second, when the paper writes a **chunkwise** form it means the sequence is cut into fixed-length chunks so that computation is *parallel inside a chunk* and *recurrent across chunks* — this is the standard trick that makes linear-attention RNNs run fast on GPUs, and it recurs throughout.

---

## The big picture: what Kimi K3 is

Kimi K3 is a **Mixture-of-Experts (MoE)** transformer with **2.8 trillion total parameters** but only **104 billion activated per token**, meaning any single token's forward pass touches roughly 1 in 27 of the weights. It is **natively multimodal** — text, images, and video share one backbone and one context, with no bolted-on adapter — and it handles a **context window of one million tokens**. The lab positions it as the first *open-weights* model in the "3T-class," released to close the gap between open models (which had mostly plateaued around 1T parameters) and the strongest closed models.

The headline scientific claim is a **≈2.5× improvement in scaling efficiency over Kimi K2**: to reach the same validation loss, Kimi K3 needs about 2.5× less training compute than its predecessor. That gain is not one trick — it is the *sum* of the architectural changes this note is about, plus refined data and training recipes.

The report frames the whole design around **two axes of scaling**. The first, classic axis is **pre-training scale** (bigger model, more data). The second, newer axis is **test-time scale** (reasoning and long-horizon agentic interaction, driven by RL). The authors' thesis is that the open ecosystem had been pushing only the second axis on top of stagnant ~1T foundations, so they push *both*: a 3T-class foundation *and* aggressive RL at 1M context.

To ground the rest of the note, here is how Kimi K3 differs from Kimi K2 concretely. This single table previews almost every design decision explained below.

| | Kimi K2 | Kimi K3 |
| --- | --- | --- |
| Total parameters | 1.04 T | 2.78 T |
| Activated parameters / token | 32.6 B | 104.2 B |
| Layers | 61 | 93 |
| Attention | MLA (all layers) | Hybrid: 69 KDA + 24 Gated MLA |
| Positional encoding | RoPE | **None (NoPE)** |
| FFN / channel mixing | dense-width MoE, SwiGLU | **Stable LatentMoE, SiTU-GLU** |
| Routed experts (active) | 384 (8) | 896 (16) |
| Shared experts | 1 | 2 |
| Latent MoE width | — | 3,584 ($0.5\times$ the 7,168 hidden) |
| MoE hidden per expert | 2,048 | 3,072 |
| Attention heads | 64 | 96 |
| Load balancing | aux-loss-free (sign rule) | **Quantile Balancing** |
| Depth mixing | standard residual | **Attention Residuals** |
| Training context | 128 K | 1 M |
| Vision encoder | SigLIP-initialized | **MoonViT-V2, from scratch** |
| Optimizer | Muon | **Per-Head Muon** |

Everything in bold is a place where Kimi K3 introduced or adopted a new idea, and each gets its own section. Notice the through-line: almost every change is about **information flow** — how information moves along the sequence (KDA/MLA), across depth (AttnRes), and across the width (LatentMoE).

---

## Three axes of information flow

Before diving into mechanisms, it helps to hold the organizing metaphor the paper uses, because the architecture is literally designed around it. A transformer moves information in three directions, and Kimi K3 upgrades each:

- **Along the sequence (token mixing).** How does position $t$ learn about positions $1 \dots t-1$? Classic softmax attention lets every token look at every other token, which is expressive but costs memory and compute that grow with sequence length. Kimi K3 uses mostly **Kimi Delta Attention (KDA)**, a linear-attention RNN whose memory is a *fixed-size* state, and sprinkles in **Gated MLA** every fourth layer to recover full global attention where it matters.
- **Across depth (layer mixing).** How does layer $l$ use what layers $1 \dots l-1$ computed? The standard residual stream just *adds* each layer's output to a running sum. Kimi K3's **Attention Residuals** instead let each layer *attend* over all previous layers' outputs and pick what it needs.
- **Across the width (channel mixing).** How is each token's representation transformed? A feed-forward network mixes channels. Kimi K3 replaces it with **Stable LatentMoE**, a sparse mixture of 896 experts operating in a compressed latent space.

![A schematic of the Kimi K3 architecture. On the left, a vertical stack shows the token/channel pathway: an Embedding at the bottom (fed from the side by a vision pathway of image or video, then MoonViT-V2, then an MLP projector), then a repeating unit of three Kimi Delta Attention layers followed by one Gated MLA layer, each attention layer paired with a Stable LatentMoE feed-forward block on the right; a final Gated MLA layer sits on top before the output. A note reads 93 layers total equals 69 KDA plus 24 MLA, and each LatentMoE has 896 experts with 16 active and 2 shared. On the right, a separate panel illustrates Attention Residuals: a current layer holding a learned pseudo-query attends with softmax weights over the embedding and all preceding block outputs, computing its input as a weighted sum, rather than keeping a single accumulated residual state.](assets/kimik3_architecture.jpg)

The figure is worth staring at: the left column is the ordinary forward stack (with the 3-KDA-then-1-MLA rhythm and a LatentMoE hung off each attention layer), while the right panel is the *depth-wise* attention that AttnRes adds on top. Keep both in mind; the sections below take the three axes one at a time.

---

## Kimi Delta Attention: linear attention that actually remembers

KDA is the workhorse of Kimi K3 — 69 of 93 layers are KDA. To understand it you need the four-step lineage it sits on: linear attention → the delta rule → gated (forgetting) delta networks → Kimi Linear → KDA. We build it up in that order, because each step fixes a concrete flaw in the previous one.

### From softmax attention to linear attention

Standard attention computes, for one head, $\mathrm{softmax}(QK^\top/\sqrt{d_k})\,V$. The trouble is the $QK^\top$ matrix: it is $n \times n$ for a sequence of length $n$, so both the compute and the KV cache you must store grow with $n$. At a million tokens this is ruinous.

**Linear attention** (Katharopoulos et al., 2020, "Transformers are RNNs") removes the softmax. Once the softmax is gone, attention is just $ (\phi(Q)\phi(K)^\top) V $ for some feature map $\phi$, and by associativity you can reassociate it as $ \phi(Q)\,(\phi(K)^\top V) $. The object $\phi(K)^\top V$ is a $d_k \times d_v$ **state matrix** $S$ that does not depend on $n$. Processing the sequence left to right, you maintain

$$S_t = S_{t-1} + k_t v_t^\top, \qquad o_t = S_t^\top q_t.$$

Read this as a memory: at each step you *write* the outer product $k_t v_t^\top$ into $S$ (associating key $k_t$ with value $v_t$), and to *read*, you project the accumulated memory with the current query. This is an RNN with a matrix-valued hidden state, and crucially the state is **fixed-size** — it does not grow with context. That is the whole appeal for long context.

The catch: pure additive writing never forgets and never corrects. Keys interfere, the state saturates, and old information you would rather overwrite just piles up.

### The delta rule: write by correcting your error

The **delta rule** (Schlag et al. 2021, "Linear Transformers Are Secretly Fast Weight Programmers"; Yang et al. 2024, DeltaNet) fixes the "never corrects" problem by borrowing the classic delta learning rule. Instead of blindly adding $k_t v_t^\top$, first *read out* what the memory currently associates with $k_t$, namely $S_{t-1}^\top k_t$, and write only the **difference** between the new target $v_t$ and that current readout, scaled by a write strength $\beta_t$:

$$S_t = S_{t-1} + \beta_t\, k_t\,(v_t - S_{t-1}^\top k_t)^\top = (I - \beta_t k_t k_t^\top)\,S_{t-1} + \beta_t k_t v_t^\top.$$

The two forms are algebraically identical; the right-hand one is the shape KDA uses. Term by term: $(I - \beta_t k_t k_t^\top)$ is an **erase** operator — it removes from the state the component along $k_t$ (as much of it as $\beta_t$ allows) — and $\beta_t k_t v_t^\top$ then **writes** the fresh association. The net effect is "delete the stale value for this key, insert the new one," which is exactly the corrective, non-interfering write that plain linear attention lacked. DeltaNet showed this dramatically improves associative recall.

### Gating: let the memory forget

The delta rule corrects, but it still does not *decay*: information that no key ever overwrites lives forever. Real sequences want controlled forgetting (recency). This is where **gating** enters — the same $(0,1)$ multiply-a-signal idea covered in the [Gated Attention note](./gated-attention.md). Gated Linear Attention (GLA, Yang et al. 2024) and Mamba-2 (Dao & Gu, 2024) multiply the state by a decay each step; Gated DeltaNet (Yang, Kautz, Hatamizadeh, 2025) combines *both* a decay gate and the delta rule. Kimi Linear (Kimi Team, 2025) is the lab's own such architecture, and **KDA is its refinement.**

### KDA proper: a channel-wise decayed delta rule

KDA's update, for one head, is (paper Eq. 1):

$$S_t = \big(I - \beta_t k_t k_t^\top\big)\,\mathrm{Diag}(\alpha_t)\,S_{t-1} + \beta_t k_t v_t^\top, \qquad \tilde o_t = S_t^\top q_t.$$

Compared with the plain delta rule, the one new object is $\mathrm{Diag}(\alpha_t)$, a **channel-wise forget gate**. Where GLA/Mamba-2 typically use a *scalar* decay per head, KDA gives **each of the $d_k$ key-channels its own retention factor** $\alpha_{t,j} \in (0,1)$ — a "fine-grained" gate. Reading the update left to right: first $\mathrm{Diag}(\alpha_t)$ shrinks the old state channel by channel (forgetting), *then* $(I - \beta_t k_t k_t^\top)$ erases the current key's stale association, *then* $\beta_t k_t v_t^\top$ writes the new one. The output $\tilde o_t = S_t^\top q_t$ reads the freshly updated memory with the query.

![A two-panel schematic of Kimi Delta Attention. Panel (a) shows one KDA step operating on the state matrix S of shape d_k by d_v: the previous state S at t minus 1 passes through a channel-wise decay Diag(alpha_t), then an erase operator I minus beta_t k k-transpose, then a write of plus beta_t k v-transpose, producing the new state S_t; a separate arrow reads the output o_t equals S_t-transpose times q_t. A caption notes the channel-wise forget gate alpha_t lies in the interval e-to-the-minus-5 to 1 and the delta-rule write beta_t lies in zero to one. Panel (b) shows the chunkwise form: three chunks each computed with parallel matmuls on Tensor Cores, connected along the bottom by a serial carry of a fixed-size recurrent state from chunk to chunk, illustrating that computation is parallel inside a chunk but recurrent across chunks.](assets/kimik3_kda_recurrence.jpg)

How are the per-head quantities produced from $x_t$? (Paper Eq. 2.) The query, key, and value each pass through a **short convolution** (a small depthwise conv over a few adjacent positions, giving cheap local mixing) followed by a **Swish** nonlinearity; queries and keys are additionally **L2-normalized** (projected onto the unit sphere, which stabilizes the delta rule). The write strength is $\beta_t = \sigma(W_\beta x_t) \in (0,1)$. The decay logit $z_t$ comes from a **low-rank** projection $W_\uparrow^\alpha W_\downarrow^\alpha x_t + b^\alpha$ — low-rank because a full $d_k \times d$ projection per head would be expensive and the decay does not need that much capacity.

### Lower-bounded decay: the sigmoid that saves the Tensor Cores

Here is the first genuinely new KDA contribution, and it is a lovely example of **algorithm–hardware co-design**. To run KDA fast you use the chunkwise form (panel (b) above): inside a chunk you rescale keys by the reciprocal cumulative decay $1/\Gamma$, where $\Gamma$ is a running product of the $\alpha$'s. Because each $\alpha \in (0,1)$, that product can get *tiny*, so its reciprocal can **explode and overflow** in BF16. Kimi Linear worked around this by computing decay in log-space and splitting each chunk into 16-token sub-tiles; the *off-diagonal* tiles could then use dense matrix multiply on Tensor Cores, but the *diagonal* tiles still needed slow, explicit position-by-position math — the intra-chunk bottleneck.

Kimi K3 removes the bottleneck by changing how the decay logit $z_t$ maps to the log-decay $g_t$. Kimi Linear (following GDN and Mamba-2) used an **unbounded** negative-softplus, $g_t = -e^{A}\,\mathrm{softplus}(z_t) \in (-\infty, 0)$. Kimi K3 uses a **scaled sigmoid that is bounded below** (Eq. 5):

$$g_t = g_{\min}\,\sigma\!\big(e^{A}\,z_t\big) \in (g_{\min},\,0), \qquad \alpha_t = \exp(g_t) \in (e^{g_{\min}},\,1),$$

with $g_{\min} = -5$ fixed and $A$ a learnable per-head log-scale (initialized to 0). The consequence is arithmetic, not just aesthetic. With $g_{\min}=-5$, every retention factor satisfies $\alpha_{t,j} > e^{-5} \approx 6.7\times10^{-3}$, so over a 16-token tile the cumulative log-decay stays in $(-80, 0)$, and the reciprocal rescaling factor stays below $e^{80}$ — **inside the BF16 range**. Now *every* causal tile, diagonal included, can use a dense Tensor-Core matmul. The position-pair diagonal path is gone.

![A two-panel figure explaining Kimi K3's lower-bounded decay. Panel (a) plots the per-step log-decay g against the decay logit z: Kimi Linear's negative-softplus curve (blue) heads to negative infinity as z decreases, so the retention factor alpha goes to zero and the reciprocal one-over-decay overflows; Kimi K3's scaled-sigmoid curve (red) flattens against a dashed floor at g-min equals minus five, keeping alpha bounded away from zero. Panel (b) shows a lower-triangular grid of chunk tiles: Kimi Linear computes the diagonal tiles (yellow) with slow position-pair math while off-diagonal tiles (green) use dense Tensor-Core matmuls, whereas Kimi K3's bounded range keeps one-over-decay inside BF16 so every causal tile, diagonal included, uses dense Tensor-Core matmuls.](assets/kimik3_lower_bounded_decay.jpg)

This is a recurring theme in the report: a small change to a *mathematical parameterization* is chosen precisely because of what it does to the *kernel*. The bound $-5$ is not tuned for loss; it is tuned so a reciprocal fits in 16 bits.

### The full-rank output gate

Finally, KDA gates its output (Eq. 6). After RMSNorm-ing the raw readout $\tilde o_t$, it applies a data-dependent, **full-rank** output gate:

$$y_t = W_o\big[\,\sigma(W_g x_t)\,\odot\,\mathrm{RMSNorm}(\tilde o_t)\,\big].$$

This is exactly the head-output sigmoid gate from the [Gated Attention note](./gated-attention.md) — a per-channel $\sigma(W_g x_t)$ that lets the token decide which channels of the attention readout to pass through, injecting a nonlinearity into an otherwise linear read-out path and draining "attention-sink"-style pathologies. Kimi Linear used a cheaper *low-rank* gate; Kimi K3 makes $W_g$ full-rank, spending a few more parameters for a more expressive gate.

---

## Gated MLA: the global-attention half of the hybrid

Linear attention is cheap but lossy: a fixed-size state cannot perfectly recall everything. So Kimi K3 keeps **full global attention** in one of every four layers, using **Multi-head Latent Attention (MLA)**. The rhythm is **3 KDA layers then 1 Gated MLA layer**, repeated, with an extra Gated MLA at the very top so the final layer is always global. KDA supplies cheap, recency-aware, position-sensitive local mixing; MLA supplies unrestricted token-to-token recall. Each covers the other's weakness.

### What MLA is (DeepSeek-V2)

MLA was introduced in DeepSeek-V2 (2024) to shrink the KV cache. Ordinary multi-head attention caches a separate key and value vector *per head per token*; at long context that cache dominates memory. MLA instead compresses each token's keys and values into a single **low-dimensional latent vector** $c_t = W_c x_t$, caches only $c_t$, and reconstructs the per-head keys and values on the fly via learned up-projections during attention. You cache one small vector per token instead of many large ones, while keeping full global attention. Kimi K2 used MLA in every layer; Kimi K3 keeps it in the periodic global layers.

### NoPE: no positional encoding at all

The striking choice: Kimi K3 applies **No Position Encoding (NoPE)** to *all* MLA layers — no RoPE, no ALiBi, nothing. The queries and keys carry no explicit position signal. Why does this work? Because the interleaved **KDA layers are inherently position-aware**: an RNN that processes tokens in order and decays its state encodes recency and order *implicitly*, through the dynamics themselves. With KDA carrying position, the MLA layers are freed to do pure content-based global lookup.

The payoff is enormous for long context. With RoPE, extending a model from 128K to 1M tokens means carefully retuning the rotary frequency base or interpolating angles (the YaRN recipe, discussed in the [Position Embeddings note](./training_llms/002_PositionEmbeddings.md)). With NoPE there is **nothing to retune** — the model extrapolates to 1M tokens with no positional-encoding surgery at all. Removing a component removed an entire class of long-context headaches.

### The output gate and the FP32 fix

Like KDA, MLA in Kimi K3 gets a full-rank channel-wise output gate (Eq. 7), $y_t = W_o[\sigma(W_g x_t) \odot \tilde o_t]$, so both attention families share the same gating parameterization. Separately, the report notes a low-precision detail: flash-attention accumulates a **biased rounding error** in BF16, so during training they keep the attention output in **FP32** (following Qiu & Yao, 2026, "Why Low-Precision Transformer Training Fails"). That doubles the on-chip footprint of the output tile, which they hide by redesigning the kernel to overlap it with KV staging buffers rather than the query tile — a small but telling instance of the report's obsession with kernel-level memory choreography.

---

## Attention Residuals: attention over depth

This is one of the most conceptually elegant ideas in the paper, and it is a clean analogy. In a standard transformer, the **residual stream** works like this: layer $l$ reads the running state $h_{l-1}$, computes something, and adds it back, $h_l = h_{l-1} + f_l(h_{l-1})$. All information from all earlier layers is squeezed into that *single* accumulated vector. The report points out this is exactly a **depth-wise RNN**: one state, updated additively, with the same bottleneck an RNN has over time.

And we already know the fix for an RNN bottleneck over time — it is **attention**. So **Attention Residuals (AttnRes)** (Kimi Team, "Attention Residuals," 2026) do to *depth* what attention did to *sequence*: instead of one accumulated state, keep every layer's output around and let each layer **attend over all of them**, retrieving what it needs with data-dependent weights.

![A two-panel comparison of residual schemes. Panel (a), standard residual, draws a vertical chain of six layers each adding its output to a single running state, labeled a depth-wise RNN bottleneck, with the note that all past layers are compressed into one vector via h_l equals h_{l-1} plus f_l of h_{l-1}. Panel (b), Attention Residuals, shows a current layer holding a learned pseudo-query attending with softmax weights over four earlier block summaries (the embedding b0 and blocks b1, b2, b3); the arrow thickness encodes the attention weight, and the layer input is computed as a weighted sum h_l equals the sum over i of alpha i-to-l times v_i, so each layer selectively retrieves earlier representations instead of reading one accumulated state.](assets/kimik3_attn_residuals.jpg)

### The full form

For each layer $l$, define a learnable **pseudo-query** $q_l = w_l$ (just a learned vector — there is no "token" here, the query is the layer's identity), and let the keys and values be the outputs of all earlier layers (Eq. 8): $k_i = v_i = f_i(h_i)$ for $1 \le i \le l-1$, with $k_0 = v_0 = h_1$ the token embedding (so the embedding is always an available source). The attention weights use a softmax kernel over RMSNorm-ed keys (Eq. 9):

$$\alpha_{i\to l} = \frac{\exp\!\big(q_l^\top \mathrm{RMSNorm}(k_i)\big)}{\sum_{j=0}^{l-1}\exp\!\big(q_l^\top \mathrm{RMSNorm}(k_j)\big)}, \qquad h_l = \sum_{i=0}^{l-1}\alpha_{i\to l}\,v_i.$$

The RMSNorm on keys is there so a layer with large-magnitude output cannot dominate the weights just by being loud — it competes on *direction*, not scale. So $h_l$, the input handed to layer $l$, is no longer "the sum of everything so far" but a **learned, selective mixture** of earlier representations. A late layer can reach past its immediate predecessor straight back to the embedding if that is what helps.

### The block form (what Kimi K3 actually uses)

The full form keeps *all* $L$ layer outputs alive, costing $O(Ld)$ memory and, under pipeline parallelism, cross-stage communication. To cut that, **Block AttnRes** partitions the $L$ layers into $N$ blocks. Within a block, outputs are summed into a single block representation $b_n = \sum_{j \in B_n} f_j(h_j)$; across blocks, attention runs over just the $N$ block-level summaries plus the running intra-block partial sum (Eq. 10). This drops memory and communication from $O(Ld)$ to $O(Nd)$, and — because the block count is small and fixed — it bounds the inference-time state, so the parallel inter-block result can be merged with the sequential intra-block partial sum using **online softmax** (the same streaming-softmax trick FlashAttention uses). Empirically $N \approx 8$ recovers most of the benefit; Kimi K3 uses **8 blocks of 12 layers** (9 counting the embedding as a source). Note this "block" (12 layers) is a coarser grouping than the 4-layer 3-KDA-1-MLA attention rhythm — the two groupings coexist.

---

## Stable LatentMoE: scaling to 896 experts

The width axis. Kimi K3 pushes sparsity hard — **896 routed experts, 16 active per token**, a sparsity ratio of 56 — and most of this section is about the three things that break at that extreme and how each is fixed.

### A quick MoE refresher

If MoE is unfamiliar, read the [Mixture of Experts note](./mixture-of-experts.md); the one-paragraph version: a MoE feed-forward layer holds many "expert" sub-networks; a small **router** scores the experts for each token and sends the token to only its **top-$k$**, so total parameters grow while per-token compute stays fixed. Kimi K3 follows **DeepSeekMoE** (2024) in splitting experts into a couple of always-on **shared experts** (which learn common transformations every token needs) plus many **routed experts** (which specialize). The two persistent headaches of MoE are (1) **load balancing** — routers like to overuse a few experts, starving the rest — and (2) **stability** at scale.

### LatentMoE: experts in a compact latent space

Adding more active experts should broaden specialization, but in vanilla MoE each selected expert receives the full $d$-dimensional token, so communication and expert-weight traffic grow with the number of active experts — expensive. **LatentMoE** (Elango et al., 2026) makes the expansion affordable by **decoupling model width from routed-expert width**. The shared experts keep the full width $d$; the routed experts operate in a compact **latent space** of width $\ell$ (here $\ell = 3{,}584$, half of $d = 7{,}168$). Concretely (Eq. 11), the routed path down-projects the token to $z = W_\downarrow x \in \mathbb{R}^\ell$, dispatches $z$ to the selected experts, aggregates their weighted outputs into $u$, and up-projects back:

$$u = \sum_{i \in T_k(x)} p_i\,E^{\text{routed}}_i(W_\downarrow x), \qquad y = \sum_{j=1}^{N_s} E^{\text{shared}}_j(x) + W_\uparrow\,\mathrm{RMSNorm}(u).$$

Here $T_k(x)$ is the set of selected experts, $p_i$ the router weight for expert $i$, and $N_s = 2$ shared experts. Because the routed experts live in the smaller latent space, you can afford far more of them — this is what makes 896 experts practical. The report notes the flip side: at this sparsity the routed path chains four matmuls ($W_\downarrow$, a gated expert FFN, $W_\uparrow$) into a nearly-linear, **ill-conditioned** composition that, at 2.8T scale, produces **exploding activations**. That, plus balancing ~$10^3$ experts, is the "instability" that "Stable" LatentMoE has to tame. Three fixes follow.

### Fix 1a — Normalized LatentMoE

The aggregated routed representation $u$ has a scale that swings with *which* experts fired and *how* they were weighted. Feeding that variable-scale vector straight into $W_\uparrow$ (and then adding it to the full-width shared branch) is a stability hazard. The fix is the `RMSNorm(u)` you already see in Eq. 11: **normalize the aggregate before up-projecting.** It desensitizes the routed branch to scale swings before it merges with the shared branch, and — a nice bonus — it also improves validation loss and downstream scores, not just stability.

### Fix 1b — SiTU-GLU: a soft-capped activation

The second half of the activation-explosion fix is a new activation function, and understanding it means understanding what came before. A **Gated Linear Unit (GLU)** computes $\sigma(W_g x) \odot W_u x$: a linear "up" branch modulated by a sigmoid gate. **SwiGLU** (Shazeer, 2020) swaps the sigmoid for a **Swish** gate, $\mathrm{Swish}(x) = x\,\sigma(x)$, and became the default FFN in LLMs for its empirical edge. But **both of SwiGLU's factors are unbounded** — the gate $x\,\sigma(x)$ grows like $x$ for large positive $x$, and the up branch $W_u x$ is just linear — so two coincident large coordinates multiply into an **activation outlier**, exactly the overflow risk you do not want in low precision.

Kimi K3 proposes **Sigmoid Tanh Unit GLU (SiTU-GLU)**. The idea is to keep SwiGLU's shape near the origin (where it works) but **smoothly cap** both factors so their product cannot blow up. The tool is a soft cap $\mathrm{softcap}(x,\beta) = \beta\tanh(x/\beta)$, applied to the linear factor of the gate *and* to the up branch (Eq. 12):

$$\mathrm{SiTU\text{-}GLU}(x) = \Big[\beta_1\tanh\!\big(\tfrac{W_g x}{\beta_1}\big)\odot\sigma(W_g x)\Big]\odot\Big[\beta_2\tanh\!\big(\tfrac{W_u x}{\beta_2}\big)\Big],$$

with $\beta_1 = 4$ (gate) and $\beta_2 = 25$ (up). Why this exact shape? Because $\beta\tanh(x/\beta) = x + O(x^3/\beta^2)$ near zero, SiTU-GLU is **first-order identical to SwiGLU around the origin** (so it keeps SwiGLU's useful local behavior and its vanishing negative tail), yet since $|\tanh| < 1$ and $0 < \sigma < 1$, every output coordinate is **bounded**: $\|\mathrm{SiTU\text{-}GLU}(x)\|_\infty \le \beta_1\beta_2 = 100$. It also recovers SwiGLU exactly as $\beta_1,\beta_2 \to \infty$. The report (Appendix B) stresses one subtlety: a *smooth* cap is better than a *hard* clamp because it preserves nonzero gradients away from the saturation boundary, giving cleaner training.

![A three-panel figure comparing GLU, SwiGLU, and SiTU-GLU on a common input range. The left panel shows the gate branch: GLU's sigmoid stays near one, SwiGLU's Swish grows without bound, while SiTU's scaled-tanh-times-sigmoid rises then flattens against a dashed line at beta-one equals four. The middle panel shows the up branch: GLU and SwiGLU use the unbounded linear x, while SiTU uses a scaled tanh that saturates at beta-two equals twenty-five. The right panel shows the product of gate and up, which is one output coordinate: SwiGLU's product blows up quadratically, whereas SiTU-GLU's product saturates against a dashed bound at beta-one times beta-two equals one hundred, so it cannot overflow in low precision.](assets/kimik3_situ_glu.jpg)

The figure makes the trade-off legible: near the origin the red SiTU curves hug the SwiGLU curves, so nothing about the well-behaved regime changes; far out, SiTU flattens while SwiGLU rockets off — that flattening is the overflow insurance.

### Fix 2 — Quantile Balancing (QB)

Now the load-balancing problem, which QB solves with a genuinely clean idea. First, the baseline it improves on: **auxiliary-loss-free balancing** (DeepSeek-V3, 2024). Rather than adding a balancing loss (which fights the language-modeling objective and adds gradient noise), you add a per-expert **bias** $b_j$ to the router score *only for the top-$k$ selection*, not for the mixture weights (Eq. 13):

$$T_i = \mathrm{argtopk}(s_i + b), \qquad p_{i,j} = \frac{s_{i,j}}{\sum_{r\in T_i}s_{i,r}},\ \ j\in T_i,$$

where $s_i = \sigma(W_r x_i)$ are the router scores. Because $b$ steers *which* experts are chosen but is absent from the weights $p_{i,j}$, it rebalances load **without touching the gradient** the router learns from. DeepSeek-V3 updated $b$ by a slow fixed-step **sign** rule, $b_j \leftarrow b_j + \gamma\,\mathrm{sign}(\bar\ell - \ell_j)$: nudge the bias up if expert $j$ is underloaded, down if overloaded. The weakness is $\gamma$: too small and balance lags, too large and the loads oscillate. At **896 experts** that tuning problem gets much worse.

**Quantile Balancing** (Su, 2026; derived in Appendix C) replaces the timid sign step with an *exact jump*. The insight: given a target of $q = mk/n$ tokens per expert (for a batch of $m$ tokens, $n$ experts, top-$k$), the bias that lands expert $j$ at *exactly* its target load is a **quantile of that expert's routing margins**. Concretely, run top-$(k{+}1)$ instead of top-$k$: the first $k$ entries are the routes actually taken, and the $(k{+}1)$-th score $\alpha^{(t)}_i$ is the *cutoff* token $i$'s $k$-th choice had to beat. The margin $s_{i,j} - \alpha^{(t)}_i$ says how comfortably expert $j$ would make token $i$'s cut. Setting expert $j$'s load to $q$ means keeping exactly $q$ of its margins above the threshold, i.e. the threshold is the $(1 - k/n)$-quantile of its margins (Eq. 14):

$$\hat b^{(t+1)}_j \leftarrow -\,\mathrm{quantile}_{1-k/n}\big(s_{:,j} - \alpha^{(t)}\big), \qquad b^{(t+1)} \leftarrow \hat b^{(t+1)} - \mathrm{mean}\big(\hat b^{(t+1)}\big).$$

The mean-subtraction removes a common offset that would not change top-$k$ selection anyway. The beautiful part (Appendix C): this is the *exact coordinate minimizer* of the dual of an optimal balanced-assignment linear program — the same problem BASE Layers and BIP solve — so QB **jumps straight to balance in a few steps and has no learning-rate knob at all**. The sign rule is just a one-step-in-the-right-direction approximation of this jump.

![A three-panel illustration of Quantile Balancing with eight tokens, four experts, and one selected expert per token. Panel (a) shows top-k routing producing an imbalanced load: expert E1 is overheated with load four (shown dark red), E2 has three, E3 has one, and E4 is a dying expert with load zero (faded, dashed). Panel (b) shows, for each expert, a row of its routing margins as bars with a dashed red cutoff placed after the second-largest margin, so exactly q equals two margins stay above the cutoff per expert. Panel (c) shows the routing after QB shifts each expert's bias to that cutoff: every expert now has the balanced load of two tokens.](assets/kimik3_quantile_balancing.jpg)

### Estimating the quantile with a histogram

One practical wrinkle (Appendix D): the quantile in Eq. 14 is over the *whole global batch* — millions of margins spread across data-parallel ranks and gradient-accumulation micro-batches. Gathering them all for an exact quantile every step is infeasible. But you never need the margins themselves, only their **distribution**, and a **histogram** summarizes a distribution at fixed cost. So each rank scatter-adds its local margins into a per-expert histogram of a few hundred bins; one integer **all-reduce** sums the bin counts across ranks; and every rank reads the quantile off the pooled counts. Because counts are *additive*, the pooled histogram is exactly invariant to how tokens were sharded — it represents the true global batch, up to the bin width (with 1,000 bins the error is a few $\times 10^{-3}$). The communication is one small integer all-reduce per layer per step, independent of the token count. A neat, scalable estimator for what looks at first like an intractable global statistic.

---

## Native vision: MoonViT-V2 from scratch

Kimi K3 is multimodal from the first training step, not adapted afterward. Text, image, and video tokens are interleaved in one context and share one backbone under a single next-token-prediction objective. Because rendered outputs and the code that produced them live in the *same token stream*, the model can write code, look at a screenshot of the result, and iterate — the vision-in-the-loop behavior that powers its web-dev and GUI-agent skills.

The vision encoder is **MoonViT-V2**, and the headline decision is that it is trained **entirely from scratch with next-token prediction**, rather than initialized from a contrastively pre-trained encoder like **SigLIP** (which had been standard practice, including in Kimi K2.5). The motivation is **training stability**: a SigLIP-initialized encoder attached to the LLM shows persistently higher gradient norms with frequent spikes during joint optimization, whereas the from-scratch MoonViT-V2 stays stable throughout. Training under the language objective (rather than a contrastive one that favors global semantics over fine-grained text and structure) also shapes the visual features toward what the LLM actually needs. The surprising empirical result: from-scratch **matches** the SigLIP-initialized baseline on vision evaluations — so at this scale, contrastive pre-training is simply *unnecessary* as an initialization.

Architecturally MoonViT-V2 is a 27-layer, ~0.4B-parameter ViT that uses RMSNorm and drops all bias terms (which further stabilizes the from-scratch optimization). Images and videos share parameters: attention is **factorized** into intra-frame *spatial* and inter-frame *temporal* passes, with temporal pooling compressing along time. A **pixel-shuffle** with $2\times2$ downsampling cuts the visual-token count by 4× before projection, keeping images up to $3584\times3584$ affordable inside the 1M-token budget. A lightweight MLP projector maps visual features into the LLM embedding space.

---

## Per-Head Muon

Kimi K3 optimizes its matrix parameters with **Muon** (Jordan et al., 2024), the optimizer whose lineage the [Optimisers note](./optimisers.md) covers. The one-paragraph idea: Muon takes the momentum matrix of a weight and **orthogonalizes** it (via a few **Newton–Schulz** iterations that push its singular values toward 1) before applying the update. Intuitively, plain momentum lets a few dominant directions hog the update; orthogonalizing equalizes the update across directions, which empirically trains faster and more stably at LLM scale (Liu et al., 2025, "Muon is Scalable").

Kimi K3 refines this into **Per-Head Muon** for the attention projections. Instead of orthogonalizing the *whole* $Q$/$K$/$V$ projection matrix as one block, it **partitions the momentum along the head dimension and orthogonalizes each head's block separately**. The reasoning: a full-matrix orthogonalization treats all heads as one coupled block, so heads with larger gradient scale dominate the shared update direction while smaller-scale heads get under-normalized updates. Per-head orthogonalization **equalizes the update scale across heads**, giving more balanced learning dynamics and better stability at scale — and, as a bonus, Newton–Schulz on tall skinny per-head blocks is *cheaper* than on the full matrix, so optimizer overhead drops slightly too.

---

## Pre-training: data, scaling laws, and long context

### Data and rephrasing

The pre-training corpus spans four text domains — **Web Text, Code, Mathematics, Knowledge** — plus a large **vision** corpus (captions, interleaved image–text, OCR, perception, video, and visual-coding data). Each domain is filtered with rule-based heuristics, classifier-based quality scoring, and deduplication, with per-domain sampling rates fixed by ablations on small models. A notable technique carried from Kimi K2 is **rephrasing**: knowledge and math corpora are rewritten with style- and perspective-diverse prompting, generated chunk-by-chunk autoregressively, and verified for fidelity against the source — a data-augmentation move that multiplies the usable signal in scarce high-quality domains. On the vision side, coordinate supervision is given in both absolute and normalized $[0,1]$ formats (for resolution-robust localization), and **programmatic multimodal data** — code paired with its rendered output across SVG, 3D assets, webpages, games, and CAD — is scaled up heavily to teach the code-then-look loop.

### Scaling laws: cosine beats WSD

Because every architectural change also shifts the *optimal training regime*, the team re-ran dedicated **scaling-law** studies (in the sense of the [Scaling Laws note](./training_llms/008_ScalingLaws.md)) to retune batch size, learning rate, tokens-per-parameter, and model shape. Fitting loss-versus-compute curves for both generations yields the report's headline **≈2.5× scaling-efficiency gain** over Kimi K2.

![A log-scale plot of validation loss against training compute in FLOPs, with two fitted power-law curves. The Kimi K3 curve (red) sits entirely below the Kimi K2 curve (blue), meaning lower loss at every compute budget. A horizontal dotted line at a fixed loss of about 2.8 is marked at the point where each curve crosses it, and a green double-headed arrow between the two crossings is labeled 2.5x less compute, illustrating that Kimi K3 reaches the same loss as Kimi K2 with roughly 2.5 times less training compute.](assets/kimik3_scaling_law.jpg)

A methodological point worth remembering: they find **cosine decay consistently beats Warmup-Stable-Decay (WSD)** — but only after a *fair* comparison. Cosine and WSD have *different* optimal peak learning rates and batch sizes, so comparing them under one shared hyperparameter set can rig the result. Running an independent scaling-law search for each schedule, cosine wins on final loss. It is a good reminder that schedule comparisons are only meaningful at each schedule's own optimum.

### The training recipe

Optimization uses **Per-Head Muon** plus the **weight-clipping** mechanism from Kimi K2, with **Quantile Balancing** for MoE load balance. The schedule is **cosine** with a 1% linear warmup; weight decay is **0.1** throughout. Pre-training starts at an **8K** context and later extends to **64K**. From the SFT stage onward they run **quantization-aware training** with MXFP4 weights and MXFP8 activations (detailed in the post-training section).

### Reaching one million tokens

Long context comes from three ingredients. First, **positional encoding is not the bottleneck** because there is none — NoPE means the model extrapolates without RoPE rescaling. Second, **data**: long natural documents and videos are noisy, so a dedicated pipeline does exact + fuzzy dedup (plus perceptual hashing on video frames), quality filtering, and structural validation; genuinely long coherent samples are scarce, so they are **upsampled**, and — crucially — additional long-context data is **synthesized by permuting and concatenating** documents and sub-tasks so that the embedded task can only be solved by attending across the *full* window. Without this, an attention mechanism trained on merely-long-but-local data degenerates into local patterns. Third, a **four-stage progressive curriculum**: the window grows 8K→64K during pre-training and 256K→1M during cooldown, concentrating the expensive long-sequence compute in a small fraction of the budget.

---

## Post-training: SFT, RL, and distillation

Post-training is where a pretrained model becomes a usable agent, and Kimi K3's pipeline is a three-stage paradigm: **SFT → RL → distillation**.

### The three-stage pipeline

**Supervised fine-tuning (SFT)** establishes a high-quality cold-start policy. They synthesize trajectory data with domain-specialized models from earlier Kimi generations, then apply multi-stage verification and human-in-the-loop annotation, broadening coverage of complex agentic tasks. All data is serialized in an XTML-based chat template (described later). SFT already runs under MXFP4/MXFP8 QAT.

**Reinforcement learning (RL)** is where higher-order reasoning and execution come from. Rather than train a separate model per task, they scale RL across **three broad domains** — (i) general tasks (reasoning, vision, knowledge work, search, faithfulness), (ii) general agents (long-horizon assistants, deep research, writing), (iii) coding agents (SWE, kernels, web dev) — **each at three reasoning-effort levels** {low, high, max}. Three domains × three efforts = **nine expert models**. The report shows that scaling RL FLOPs steadily increases the number of tool-call steps the model takes *and* its capability across every axis.

**Multi-Teacher On-Policy Distillation (MOPD)** then consolidates those nine specialists back into one model. This RL-then-distill structure — specialize, then merge — is the pipeline's backbone.

### Reinforcement learning across domains and efforts

Two mechanisms make long-horizon RL tractable. **Partial rollouts** (from Kimi k1.5 and K2.5) attack the long-tail latency of agentic trajectories: you sample $K$ completions for each of $N$ prompts and, instead of waiting for *all* $NK$ trajectories to finish, pause generation once a fraction $\lambda$ complete, so policy optimization proceeds without waiting on stragglers; paused rollouts resume next iteration (enabled by resumable sandboxes, later). Because a single long trajectory then spans multiple iterations, its data becomes **stale** (off-policy); a **per-token regularization** that constrains updates to a local neighborhood lets the algorithm tolerate this extreme off-policy regime without destabilizing.

### Reasoning-effort RL and the generative reward model

To *control* reasoning effort (and stop the model from overthinking), they use **per-problem budget control**. Each problem $x$ gets an initial token budget $b_0(x)$ estimated from the cold-start model; any trajectory whose length exceeds a scaled threshold $\tau\cdot b_0(x)$ has its reward overridden to $-1$. Training follows a curriculum over the multiplier $\tau$: first a **max-budget** variant with large $\tau$ (but still capped to suppress runaway thinking), then $\tau$ is annealed down to produce **high-** and **low-effort** experts. For general tasks the budget counts *thinking* tokens; for agentic tasks it counts *all* output tokens including tool-call arguments.

For non-verifiable tasks (no ground-truth checker) they use an **Agentic Generative Reward Model (GRM)** — a tournament of binary comparisons where the judge must follow a strict protocol: read the output, *generate a rubric*, score each candidate against it, and record scores in a scorepad. To stop the model from reward-hacking by being verbose, a **budget-based verbosity control** automatically loses any comparison whose output exceeds a length threshold.

### Multi-Teacher On-Policy Distillation

MOPD merges the nine experts. On-policy distillation (Lu & Thinking Machines Lab, 2025) means the *student* generates the trajectory and the *teacher* grades it token by token — this keeps training on the student's own distribution (avoiding the exposure bias of learning only from teacher trajectories). For a domain $d$ and sampled effort $e$, the matching teacher $\pi^{(d,e)}_{\text{teacher}}$ supplies a **per-token reward** (Eq. 15):

$$r^d_{\text{opd}}(y_t\mid e,x,y_{<t}) = \mathrm{clip}\!\left(\mathrm{sg}\!\left(\log\frac{\pi^{(d,e)}_{\text{teacher}}(y_t\mid x,y_{<t})}{\pi_\theta(y_t\mid e,x,y_{<t})}\right),\ -R_{\max},\ R_{\max}\right).$$

Unpack it: the reward for emitting token $y_t$ is the **log-ratio of the teacher's to the student's probability** for that token — positive when the teacher likes the token more than the student does (push the student up), negative otherwise. The `sg` (stop-gradient) makes it a fixed reward signal rather than a differentiable loss, so it plugs straight into the RL machinery (including partial rollouts); the `clip` to $\pm R_{\max}$ bounds extreme advantages for stability. It is a dense, per-token teaching signal. They note a finer-grained top-$k$ distillation objective gave no clear benefit here.

### Deployment-aware post-training: MXFP4 and a draft model

Two moves make the trained model cheap to *serve*.

**MXFP4 quantization-aware training.** The MoE expert weights dominate parameter memory, so they are quantized to **MXFP4** — a 4-bit **microscaling** float format (Rouhani et al., 2023) that shares a scale across a small block of values, giving 4-bit storage with far better range than plain INT4 (see the [Floating Point Precision note](./floating_precison.md)). Activations run in **MXFP8**; everything non-expert (attention, latent projections, shared experts, routers) stays higher-precision. Critically, this is **quantization-aware training** run through *all* of SFT and RL, so the model *adapts* to the precision loss, and — because RL rollout and training share the same quantized scheme — there is **no train/inference mismatch**.

**A draft model for speculative decoding.** Kimi K3 pre-trains a **multi-token-prediction (MTP)** layer whose structure mirrors a backbone block. Since the draft model of **EAGLE-3** (Li et al., 2025) is a single decoder layer of just this shape, they fine-tune the MTP layer into an **EAGLE-3-style draft**: the target model is frozen, only the draft layer and a feature-fusion projection train. Following EAGLE-3's training-time-test protocol, the draft is unrolled 7 steps during training, consuming its own earlier outputs (mirroring inference). The draft input fuses **low-, mid-, and high-level features** of the target (outputs of the 1st, 4th, and final AttnRes blocks), concatenated and projected by a matrix initialized to $[\,0\ 0\ I\,]$ so it starts out reading only the high-level feature (what the MTP layer was pretrained on) and *learns* to use the lower levels. Finally, rather than the usual KL surrogate, they directly optimize the **acceptance rate** with the **LK loss** (Eq. 16), $\mathcal{L}_{LK} = -\log\sum_x \min(p(x), q(x))$ — the negative log of the exact speculative-sampling acceptance probability between target $p$ and draft $q$ — because minimizing KL does not actually maximize the rate for a capacity-limited draft.

---

## Building RL environments and tasks

The report is emphatic that RL results live or die on the *environments*. Kimi K3's environment engineering is substantial enough to be its own contribution.

- **Unified white-box RL environment.** Training against one fixed agent harness makes a model overfit to that harness's tool schema, prompts, and protocol. So they build a harness as a set of composable modules (tool interfaces, system prompts, context-management strategies, skills, memories, subagents), from which they can instantiate Kimi Code, Claude Code, Codex, OpenClaw, Hermes, or entirely new harnesses. During RL they *randomize* the harness configuration per task group, forcing cross-harness generalization instead of memorizing one convention.
- **Knowledge-graph-guided task synthesis.** To control both *granularity* and *coverage* of synthesized tasks, agents build a **self-evolving hierarchical knowledge graph** (a DAG from coarse to fine concepts) by recursive web exploration; before adding a node an agent checks the existing graph to avoid duplication, and a branch stops when a concept is "atomic." Sampling nodes (individually or in related groups) yields keyword sets that drive web retrieval of real source materials, from which a synthesis agent builds tasks — surfacing specialized, underrepresented knowledge that random sampling would miss.
- **Verifiable agentic problems.** Multi-step web search with verifiable answers; real professional work (investment banking, data analysis, legal) done in a sandbox over dozens-to-hundreds of steps; and multi-step **visual** reasoning where the agent has a Python interpreter to crop/zoom/transform an image, compute, and read the results (including generated images) as new observations.
- **Kernel optimization tasks.** A large suite from single kernels to fused mega-kernels (sourced from repos like Flash Linear Attention), spanning CUDA, Triton, CuTe DSL, Gluon, ThunderKittens, TileLang, and BF16/FP8/FP4. Reward blends correctness (a PyTorch reference; too-large numerical error → zero reward) with performance (matching an expert kernel earns 0.5, approaching the hardware roofline pushes toward 1). A dedicated **hacking-detection** system penalizes cheats like CUDA-graph replay, input caching, or precision reduction.
- **Personal assistant tasks.** Realistic *mock* implementations of Gmail, Notion, Slack, and Canvas, so agents can act over multiple simulated days across interdependent events, sometimes thousands of tool calls and millions of context tokens per rollout — a "living environment" with modeled event streams and world-state transitions.
- **Autonomous Execution Tasks (AET).** Each task gives only an initial state, a constrained goal, a tool action space, budgets, and an **independent verifier** — no reference trajectory. The agent must decompose, plan, recover from errors, and decide when to stop; reward comes from the verifier's judgment of the *final state*, not the agent's self-report. Reward hacking is blocked by isolating agents from verifiers and pairing a diagnostic **public** verifier with a held-out **hidden** one.
- **Web development tasks.** Websites, games, 3D/WebGL scenes, dataviz, SVGs, full-stack apps, run in containers under diverse scaffolds. Reward = deterministic functional/pixel checks + model judging (source inspection or interacting with the artifact); a build failure or a faked artifact zeroes the reward.

---

## Infrastructure I: making KDA fast

The infrastructure section is a systems paper stapled to a model report, and it exists because Kimi K3 stacks three rarely-combined challenges: a hybrid recurrent-plus-global attention, 3T-scale sparse multimodal training, and million-token agentic RL. We take it in four parts. First, KDA.

KDA's blessing is a fixed-size state $S \in \mathbb{R}^{d_k\times d_v}$ that is cheap to transfer and reuse; its curse is that the state update is **serial**, which fights the GPU's love of wide parallelism. The fix is a different kernel per regime.

- **Training and prefill** use the chunkwise form (parallel within a chunk, serial across chunks). Executed naively the two phases alternate and leave SMs idle during the serial carry. **FlashKDA** (a CUTLASS kernel) overlaps intra-chunk compute with cross-chunk state propagation, decomposing the work into token-parallel stages and a head-parallel recurrence tuned independently; it backs `flash-linear-attention`.
- **Long-context prefill** additionally uses **intra-device context parallelism**: because a segment's state transition can be computed independently of the incoming state and composed exactly afterward, an SM-level planner splits the sequence across a *single rank's* SMs, evaluates the segment transitions in parallel, and merges — no cross-device communication.

### KDA Context Parallelism (KCP)

Across devices, the story is more interesting, and it is where the fixed-size state pays off spectacularly. In **softmax** attention, context parallelism forces ranks to exchange key–value blocks whose size **grows with sequence length** (Ring Attention). In **linear** attention the context is a fixed-size state, so in principle you exchange something tiny. Prior linear-attention CP methods summed each rank's locally-generated state — but that additive trick **fails for KDA**, because KDA's delta rule applies a *token-dependent* matrix $M_t = (I - \beta_t k_t k_t^\top)\mathrm{Diag}(\alpha_t)$ to the incoming state *before* writing, so a segment's effect depends on the state entering it and cannot be computed from a zero-start state alone.

**KCP** (Eq. 17) fixes this by decomposing each segment into two locally-computable fragments: a **cumulative transition** $M$ (the product of the segment's $M_t$'s, which acts on whatever state enters) and a **local state** $\tilde S$ (what the segment generates starting from zero). Both can be computed from local tokens *before* the incoming state is known. Because these rank-level updates **compose associatively** — $S_{[i+1]} = M_{[i+1]} S_{[i]} + \tilde S_{[i+1]}$ — the incoming state of every rank is recovered by a **parallel prefix scan** after a single fixed-size all-gather of the fragments. So KCP synchronizes recurrent state with communication **independent of sequence length**, and achieves linear compute scaling.

![A schematic of KDA Context Parallelism across four ranks. Each rank holds a segment of the sequence and computes two local fixed-size fragments: a cumulative transition matrix M and a local-from-zero state S-tilde. A single all-gather exchanges these fixed-size fragments among all ranks. Then a prefix scan composes them left to right using the recurrence S at i plus one equals M at i plus one times S at i plus S-tilde at i plus one, recovering each rank's incoming state. A note emphasizes that each fragment has fixed size independent of sequence length, so communication does not grow with context.](assets/kimik3_kcp_prefix_scan.jpg)

The prefix-scan picture is the whole idea in one image: instead of shipping ever-larger KV blocks, each rank ships a constant-size $(M, \tilde S)$ pair and the incoming states fall out of an associative scan.

---

## Infrastructure II: training a 2.8T model

The pre-training stack combines the usual parallelism alphabet soup — Pipeline Parallelism with virtual stages, Expert Parallelism, ZeRO-1 data parallelism, Pipeline ZeRO-2 gradient sharding, and Context Parallelism (see the [Parallelism note](./training_llms/007_Parallelism.md)). Native multimodal 3T-scale training then poses three problems: token loads imbalance across expert-parallel ranks, activations/gradients/optimizer-states overflow memory, and the vision encoder's variable compute sits on the critical path. Three subsystems answer them.

### MoonEP: perfectly balanced expert parallelism

In ordinary expert parallelism, tokens route unevenly, so some ranks do more work (throughput loss) and the varying activation shapes fragment memory. **MoonEP** achieves **perfect** load balance with **dynamic redundant experts**: it replicates hot experts on the fly, planning them from the current micro-batch's router output and prefetching before the routed-expert compute (and reducing their gradients back to home ranks in the backward pass). The theoretical heart (Appendix E) is a bound: **at most $E/R$ redundant experts per rank always suffice** to give every rank exactly $S\times K$ tokens (for sequence length $S$, top-$K$, $E$ experts, $R$ ranks), and this bound is essentially tight. So reserving $E/R$ redundant slots per rank *guarantees* a feasible balanced plan every step — training is **never** interrupted for lack of a plan. Contrast prior work (ECHO, UltraEP) that presets a redundant-expert count or a per-rank token cap and must **stop** when no plan fits.

Perfect balance then simplifies everything downstream: (1) **zero-copy communication** — a fused permute/unpermute sends each token directly to its expert-grouped slot on the remote rank, and the buffer is a fixed $S\times K$ instead of the worst-case $S\times K\times R$ DeepEP would need; (2) **static shapes** — every rank always gets exactly $S\times K$ tokens, so the per-layer host↔device synchronization (normally needed to learn the dynamic shapes) *disappears*, killing a pipeline stall; (3) a **workload-aware GEMM scheduler** handles the residual within-rank skew.

### Fitting it in memory

A **unified activation manager** treats every saved-for-backward tensor as having a pluggable storage backend, so recomputation, quantization, and offload/remote-offload become *composable policies* declared by lightweight annotations, decoupled from model code. In practice most activations are block-wise FP8 plus offload; element-wise ops are recomputed. Additional tricks: a **memory-efficient MoE** rewrites a backward gradient (following SonicMoE) so it depends on an intermediate activation rather than the forward output, and recomputes the dispatch in the backward pass (overlapping the resulting communication with GEMM); a **Block-AttnRes-aware** optimization generates the block representation once at the boundary and shares it, wrapping AttnRes in checkpointing so its saved activation equals a standard residual's; **remote activation offload** to *other* PP ranks' memory (via the Mooncake Transfer Engine) balances the uneven activation distribution of interleaved 1F1B pipelines; **Pipeline ZeRO-2** shards gradients across data-parallel ranks and stores them in CPU memory; and a **P2P-based Muon orthogonalization** gathers only the parameter shards each rank owns (rather than an all-gather of the whole buffer), cutting both memory and communication for the Newton–Schulz step.

### Hiding the vision encoder

Two ideas keep the ViT off the critical path. **Dynamic CP in the encoder** splits a large image along its patch dimension across devices (gathering KV across CP ranks for attention) and load-balances multiple large images across sub-CP-groups, so encoder latency and cross-device imbalance both drop. And **encoder-in-pipeline-bubbles**: under interleaved 1F1B, the first micro-batches' forward passes and last micro-batches' backward passes leave *bubbles*; they schedule most ViT forward/backward compute into those bubbles, largely erasing the vision encoder's effective overhead.

---

## Infrastructure III: million-token agentic RL

Scaling agentic RL to 1M-token contexts on a *bounded* GPU budget makes resource efficiency the first-order concern. They keep each experiment within a few hundred GPUs using **co-located** RL (training and rollout on the same devices) plus partial rollouts — which creates a memory contention between the **rollout KV cache** (which must persist for the next iteration) and the memory training needs.

- **External KV cache pool.** At 1M context a prefix-cache miss is catastrophic, and partial rollouts make many long prefill requests arrive together at each iteration's start. So they **decouple prefix retention from GPU residency**: active decoding blocks stay in GPU KV cache, but reusable idle prefixes are **written back to a CPU-DRAM pool** only when evicted from GPU, and prefetched before reuse; KDA states are offloaded and prefetched *together* with their MLA KV blocks so their lifecycles stay aligned. To free DRAM for this pool, training states (weights, optimizer) are offloaded to **NVMe** between iterations.
- **Rollout auto-throttling.** Contexts grow as a trajectory advances, so a fixed concurrency is either too conservative early or triggers KV-cache preemption late. A scheduler watches runtime signals (active/queued request counts, KV-cache utilization) and dynamically dials concurrency up early and down as pressure rises — no manual tuning.
- **Gradient-buffer reuse for reference models.** Forward-only reference models are too big to keep resident, so their weights live in CPU and are materialized on demand into the policy model's FP32 **gradient-buffer** storage — reusing existing GPU memory (safe because the buffers get overwritten when real gradients are computed), streamed chunk by chunk with double-buffered prefetch.

### AgentENV: resumable microVM sandboxes

Agents that explore aggressively (and sometimes try to reward-hack) crashed the team's early container sandboxes with kernel panics and deadlocks — yet complex tasks need a *real* environment where an agent can mount disks, run containers, even launch VMs. So they built **AgentENV**, a sandbox on **Firecracker microVMs** (the lightweight virtualization behind AWS Lambda) for container-beating isolation and fidelity. Its RL-specific superpower is **incremental checkpointing**: only memory pages dirtied since the last checkpoint are saved, giving checkpoint/resume latencies of **133 ms / 49 ms**. On top of that: **Pause/Resume** (a paused sandbox uses zero CPU/memory, and a sandbox can be paused while waiting on model inference — up to **98%** of its lifetime), **Fork** (branch the exact state for side-effect-free reward judging), and **Snapshot** (periodic saves for error recovery). Fast, dense launches (OverlayBD images, a custom ublk driver, P2P transport, copy-on-write memory) hit sub-second launches and a **6.5× memory overcommit**. The scale number is staggering: **51,219,741 sandboxes across 1,505,678 images** over training and evaluation.

---

## Infrastructure IV: serving the model

Serving re-exposes the same tensions from the production side. Three levels of solution.

### KDA-aware prefix caching

The hybrid architecture keeps **two very different caches**: the MLA KV cache grows per token, while the KDA recurrent state is fixed-size, one snapshot per request. A prefix is reusable only if *both* can be restored at the same boundary. Naively, standard block-hash prefix caching requires one block size shared by all layers, and since a KDA state snapshot is affordable only at *sparse* boundaries, that forces a coarse 1024–6144-token block — at which caching is nearly useless (short requests never reuse; chunked prefill exports nothing until it crosses a full block).

Their fix **decouples the two granularities**. They pack KDA states into the *same paged pool* as MLA KV (same byte size, so one allocator/eviction path serves both — and any type-confused access yields obvious garbage, a free sanity check). Then prefix **hashing** runs on fine 512-token hash blocks inside MLA pages, while the **physical** block stays coarse; KDA checkpoints are saved only at a sparse subset of those hash endpoints (typically conversation-turn boundaries). A lookup matches whole physical blocks by chained hash, falls back to hash endpoints inside the first missing block, and then requires a KDA checkpoint at the candidate boundary in *every* KDA group. The hit is the longest boundary satisfying both — a multiple of the 512-token hash block, *not* required to be a multiple of the physical block.

![A schematic of fine-grained prefix caching in a hybrid KDA and MLA model. A 6144-token physical cache block is subdivided into twelve 512-token hash blocks. Cached MLA key-value blocks are shown filled, empty ones light gray; below them, markers show KDA checkpoint status at each hash boundary, with most boundaries lacking a checkpoint and a few sparse persisted checkpoints, one of which is the hit at boundary B equals 2560. On a hit, the system restores the KDA checkpoint at B, copy-on-writes the partial MLA block, and resumes prefill from token B with zero recompute of the tokens before B, deep inside the physical block rather than only at physical-block boundaries.](assets/kimik3_prefix_cache.jpg)

The result: prefix caching for a hybrid KDA–MLA model reaches the **same generality as full-attention caching** — any shared prefix reusable at any 512-token boundary, regardless of request length or scheduling — after handling three concurrency hazards (pin every hit block across all cache groups before allocating; exclude just-written blocks from matching until their copies land; and evict a KDA checkpoint from all groups atomically).

### Kernels and fleet scheduling

Dedicated **decode kernels** matter because KDA *decoding* has a different bottleneck than prefill: the state updates in place every step, which breaks MTP speculative decoding — if verification rejects some drafted tokens, the state has already advanced past the last accepted one. Snapshotting the state per draft position would fix rollback but multiplies state traffic. Their trick (independently proposed as ReplaySSM): cache only the small *projected inputs* of the draft tokens, rebuild the accepted states on-chip, and fuse the short convolution, normalization, gating, recurrence, and output-norm into one kernel — so verification latency grows sub-linearly. **Block AttnRes** kernels use sequence parallelism in prefill (so each token's block representation is materialized on exactly one rank) and side-stream overlap plus fusion in decode. For the sparse **LatentMoE**, small-batch decode is memory-bound, so they build on **WarpDecode** (each warp owns one output neuron, streaming weights from memory) with finer lane teams and an offline weight permutation to cut dequantization overhead.

At the **fleet** level, two policies turn per-instance efficiency into predictable production: **cache-aware affinity scheduling** routes each session to the cluster holding its prefix cache (a 400K-token prefix with a 4K increment makes a cache hit orders of magnitude cheaper than a miss), using consistent hashing to pin each session to a primary *and* a secondary cluster so a single cluster failure's re-prefill cost spreads across the fleet; and **budget-based admission control** gives each request class its own resource budget, so a burst of million-token requests cannot starve the short ones (per-request cost spans three orders of magnitude, which breaks any "average request" capacity model).

---

## How good is it? Evaluations

Kimi K3 is benchmarked across four axes — Reasoning & Knowledge, Coding, Agentic, and Vision — against proprietary frontier models (Claude Fable 5, GPT-5.6 Sol, Claude Opus 4.8, GPT-5.5) and the strongest open model (GLM-5.2), all at maximum reasoning effort. The one-line summary from the report: **Kimi K3 trails only Claude Fable 5 and GPT-5.6 Sol overall, and consistently beats everything else, open or closed.** Rather than reproduce the full 40-row tables, here are the marquee numbers (Kimi K3 at max effort; **bold** = best in that row across the models shown).

| Benchmark (axis) | Kimi K3 | Fable 5 | GPT-5.6 Sol | Opus 4.8 | GLM-5.2 |
| --- | --- | --- | --- | --- | --- |
| GPQA Diamond (reason) | 93.5 | 92.6 | **94.1** | 91.0 | 91.2 |
| HLE-Full, no tools / tools (reason) | 43.5 / 56.0 | **53.3 / 63.0** | 44.5 / 58.0 | 49.8 / 57.9 | – |
| DeepSWE (coding) | 67.5 | 70.0 | **73.0** | 59.0 | 46.2 |
| ProgramBench (coding) | **77.8** | 76.8 | 77.6 | 71.9 | 63.7 |
| Terminal-Bench 2.1 (coding) | 88.3 | 88.0 | **88.8** | 84.6 | 82.7 |
| FrontierSWE (coding) | 81.2 | **86.6** | 71.3 | 66.7 | 67.3 |
| SWE-Marathon (GPU kernels) | **42.0** | 35.0 | 39.0 | 40.0 | 13.0 |
| BrowseComp (agentic) | **91.2** | 88.0 | 90.4 | 84.3 | – |
| MCPMark-Verified (agentic) | **94.5** | 87.4 | 92.9 | 76.4 | – |
| GDPval-AA v2 Elo (agentic) | 1686 | **1747** | 1736 | 1593 | 1510 |
| OmniDocBench (vision) | **91.1** | 89.8 | 85.8 | 87.9 | – |
| Math-Vision, no / tools (vision) | 94.3 / 97.8 | 94.8 / **98.6** | **95.8** / 97.8 | 86.7 / 97.1 | – |
| ZeroBench pass@5, no / tools (vision) | 23.0 / 41.0 | 23.0 / **46.0** | 17.0 / 35.0 | 17.0 / 34.0 | – |

The pattern the report draws out: Kimi K3 is **at or near the frontier** on graduate reasoning (93.5 GPQA) but still **behind on research-level tasks** (HLE-Full, CritPt) — flagged as the main area to improve. In **coding** it is strongest on agentic/long-horizon suites (best on ProgramBench, best on the GPU-kernel SWE-Marathon by 7 points, essentially tied with GPT-5.6 Sol on Terminal-Bench, second on FrontierSWE). In **agentic** it is state-of-the-art on a broad swath (BrowseComp 91.2, MCPMark 94.5, DeepSearchQA, τ³-Banking, Harvey Lab), with the exceptions being the Elo-rated knowledge-work suites led by Fable 5. In **vision** it tops OmniDocBench and is amplified strongly by Python tools (ZeroBench 23→41 pass@5).

### Internal benchmarks

The lab keeps a refreshed in-house suite (coding, general-agent, and conversational) that separates strengths from weaknesses more sharply than public benchmarks. Kimi K3 **leads Swarm Bench and Deep Research Bench by clear margins** (strong at decomposing objectives and coordinating parallel work), is a coding strength (second only to Fable 5 on Kimi Code Bench 2.0, best on Coding Experience), and — on a blind expert-judged **web-dev** benchmark — is preferred over Claude Opus 4.8 by **+31 points overall**, with the biggest gap on 3D/WebGL/Shader tasks (+59). It trails on some behavior/enterprise-collaboration suites (Agent Behavior Bench, MIRA, 24/7 ClawBench).

### Cyber-security evaluation

A notable inclusion: a two-tier cyber evaluation (frontier Anthropic/OpenAI models refuse these tasks, so they are excluded). **Tier 1 (vulnerability discovery, defensive)** — across dozens of deployed systems the model found hundreds of candidate vulnerabilities; of those human-reviewed, ~70% were genuine, including **16 previously-unknown vulnerabilities across six projects** (e.g. a remotely-triggerable Linux-kernel heap out-of-bounds write, and a Dirty-COW-class RDMA privilege-escalation). **Tier 2 (end-to-end exploit development, misuse-relevant)** — on a 36-task suite (each verified solvable by experts, ~540 expert-hours total) it solved **14/36 (38.9%)** vs GLM-5.2's 8/36, but most successes were user-space; kernel exploitation and hardened targets remain a clear gap to human experts. A joint UK AISI / NIST CAISI assessment reached consistent conclusions. The team frames all of this as a **lower bound** on capability.

### Third-party and cost efficiency

Independent evaluators corroborate the standing: **Artificial Analysis** Intelligence Index v4.1 = 57.1 (4th of 580, behind Fable 5 and GPT-5.6 Sol); **Vals AI** index 2nd of 39; **LMArena** — first open model to *top* the WebDev Arena (1,678 Elo, ahead of Fable 5). On **cost efficiency** (score vs per-task dollars across Kimi Code Bench, BrowseComp, GDPval-AA v2, AA-Briefcase) Kimi K3 sits on or near the frontier: on BrowseComp it gets the best score at \$2.03/task — half GPT-5.6 Sol's cost and an order of magnitude below the Claude models at max effort.

---

## Case studies: the model builds its own tools

The report closes with case studies that double as a capability demo, several of which fed back into building Kimi K3 itself.

- **GPU kernel optimization.** Given 24 hours per kernel in a sandbox, Kimi K3 improved all four target kernels — cutting AttnRes latency 283.6→114.4 ms, DSA and KDA by 55% and 74% — matching Fable 5 and beating the other baselines. An early checkpoint was already doing most of the team's own kernel-optimization work during late development.
- **A from-scratch GPU compiler (MiniTriton).** Kimi K3 built a compact Triton-like compiler — a tile-level Python frontend, a warp-level MLIR optimization layer, and a PTX codegen pipeline — plus a dual-mode tensor library with autograd and NCCL distributed primitives. Its matmul path reaches ~90% of the machine roofline, and it trains a GPT end-to-end matching PyTorch to fp32 rounding error. (Open-sourced.)
- **An inference chip (nano-kpu).** As a proof of concept it designed an INT4 inference-chip prototype for a nano model of the same architecture, in a single 48-hour autonomous run with open-source EDA tools: it closes timing at 100 MHz within a 4 mm² budget and simulates >8,700 tokens/s. (RTL open-sourced.)
- **Research and knowledge work.** It reproduced the astrophysics I–Love–Q relations (20+ papers reviewed, 300+ equations, 3,000+ lines of Python, an HTML dashboard) in ~2 hours versus a researcher's 1–2 weeks; built a 42-year AI-ASIC industry website over 120 refinement rounds (2,800+ web searches, 1,100+ terminal queries); and, using its native multimodality, produced a 3Blue1Brown-style motion-graphics explainer of its own architecture and edited its teaser video from 56 clips.

---

## The chat template (XTML)

A small but instructive appendix: Kimi K3's chat template is redesigned around three goals — **extensibility** (new capabilities via new message formats, not template rewrites), **low alignment tax** (learnable from little supervised data), and **decoding friendliness** (simple parsers, grammar-constrained enforcers). The format, **XTML**, is XML-like but replaces angle brackets with three reserved special tokens `[open]`, `[sep]`, `[close]` (plus `[end_of_msg]`), so every structural boundary is an explicit token — no tokenization ambiguity at element edges.

Two ideas are worth carrying away. **Option messages as in-context instructions:** request options (tool declarations, `tool_choice`, `response_format`, and *reasoning effort*) are translated into short natural-language instructions placed in context rather than into special syntax — global options that rarely change go *before* the history (so changing them invalidates the KV cache anyway), one-shot options go *after* (so per-request changes leave the history cache intact). Because a pretrained model already follows instructions, new options cost little or no extra training — the low-alignment-tax principle in action. And **channels** (inspired by OpenAI's Harmony format): an assistant message body splits into `think` (reasoning trace), `response` (user-visible answer), and `tools` (typed, indexed parallel tool calls) — with the generation prefix alone (`[open]think[sep]` vs `[open]response[sep]`) selecting thinking vs instruct mode. Reasoning effort is exposed as a natural-language `thinking-effort` option (low/medium/high/max), aligning directly with the effort-conditioned RL training.

---

## Takeaways

- **The report is a survey.** Kimi K3's value as a *reading* is that almost every subsystem is a named prior idea, tweaked: delta rule → Gated DeltaNet → Kimi Linear → KDA; MLA → NoPE-MLA; residual stream → Attention Residuals; MoE → DeepSeekMoE → LatentMoE → Stable LatentMoE; aux-loss-free balancing → Quantile Balancing; Muon → Per-Head Muon; EAGLE-3, MXFP4, on-policy distillation, Firecracker. Learn the lineages and you learn the field.
- **Information flow is the organizing idea.** Sequence (KDA + Gated MLA), depth (AttnRes), width (LatentMoE) — each axis gets an attention-like, data-dependent upgrade over a dumb accumulate-everything default. The ≈2.5× scaling gain is the sum of these, not one silver bullet.
- **Algorithm–hardware co-design is everywhere.** The clearest example is the lower-bounded decay: a $g_{\min}=-5$ sigmoid chosen so a reciprocal fits in BF16 and every chunk tile can use Tensor Cores. MoonEP's $E/R$ redundant-expert bound (guaranteeing a static, sync-free, zero-copy MoE), KCP's fixed-size prefix scan, and the fused KDA prefix cache are the same spirit: pick the math that makes the kernel and the systems layer clean.
- **Bounding and normalizing tames extreme scale.** SiTU-GLU soft-caps SwiGLU's product at 100; RMSNorm sits before every risky projection (routed-MoE aggregate, AttnRes keys, KDA/MLA readouts); FP32 flash-attention output fixes a low-precision bias. At 2.8T parameters, keeping activations in range *is* the training recipe.
- **Post-training is specialize-then-merge.** Nine RL experts (3 domains × 3 efforts), consolidated by a dense per-token log-ratio distillation reward, with budget controls to prevent overthinking and verbosity-hacking — trained in white-box, randomized, verifier-grounded environments and served under MXFP4 QAT with an EAGLE-3 draft.
- **The systems section is half the paper.** Perfectly balanced expert parallelism, million-token co-located RL with NVMe/CPU state offload, 51M resumable Firecracker sandboxes, and hybrid-cache prefix reuse are what make a 3T-class, 1M-token, agentic model actually trainable and servable. The architecture is necessary; the infrastructure is what ships it.

---

## Sources

- Kimi Team, [Kimi K3: Open Frontier Intelligence](https://arxiv.org/abs/2607.24653) (arXiv:2607.24653, July 2026) — the paper this note covers; weights on [Hugging Face](https://huggingface.co/moonshotai/Kimi-K3).
- Kimi Team, [Kimi Linear: An Expressive, Efficient Attention Architecture](https://arxiv.org/abs/2510.26692) (2025) — the direct ancestor of KDA (chunkwise gated delta rule).
- Yang, Kautz, Hatamizadeh, [Gated Delta Networks: Improving Mamba2 with Delta Rule](https://openreview.net/forum?id=r8H7xhYPwz) (ICLR 2025); Yang et al., [Parallelizing Linear Transformers with the Delta Rule](https://arxiv.org/abs/2406.06484) (NeurIPS 2024) — delta rule + gating lineage.
- Katharopoulos et al., [Transformers are RNNs](https://proceedings.mlr.press/v119/katharopoulos20a.html) (ICML 2020); Dao & Gu, [Transformers are SSMs (SSD / Mamba-2)](https://arxiv.org/abs/2405.21060) (2024) — linear attention and state-space duality.
- DeepSeek-AI, [DeepSeek-V2](https://arxiv.org/abs/2405.04434) (2024, MLA); [DeepSeekMoE](https://arxiv.org/abs/2401.06066) (2024, shared+routed experts); [DeepSeek-V3](https://arxiv.org/abs/2412.19437) (2024, auxiliary-loss-free balancing).
- Elango et al., [LatentMoE](https://arxiv.org/abs/2601.18089) (2026) — routed experts in a compact latent space.
- Shazeer, [GLU Variants Improve Transformer](https://arxiv.org/abs/2002.05202) (2020, SwiGLU); Kimi K3 Appendix B for SiTU-GLU details.
- Kimi Team, "Attention Residuals" (preprint, 2026) — attention over network depth (block form).
- Jordan et al., [Muon optimizer](https://kellerjordan.github.io/posts/muon/) (2024); Liu et al., [Muon is Scalable for LLM Training](https://arxiv.org/abs/2502.16982) (2025).
- Li et al., [EAGLE-3](https://arxiv.org/abs/2503.01840) (2025, speculative-decoding draft); Rouhani et al., [Microscaling Data Formats (MXFP)](https://arxiv.org/abs/2310.10537) (2023); Lu & Thinking Machines Lab, [On-policy distillation](https://thinkingmachines.ai/blog/on-policy-distillation/) (2025).
- Zhao et al., [DeepEP](https://github.com/deepseek-ai/DeepEP) (2025) and Kimi K3's [MoonEP](https://github.com/MoonshotAI/MoonEP); Agache et al., [Firecracker](https://www.usenix.org/conference/nsdi20/presentation/agache) (NSDI 2020); Qin et al., [Mooncake](https://arxiv.org/abs/2407.00079) (2024).
- Qiu et al., [Gated Attention](https://arxiv.org/abs/2505.06708) (2025) — the output-gate idea reused by both KDA and Gated MLA; see the [Gated Attention note](./gated-attention.md).
