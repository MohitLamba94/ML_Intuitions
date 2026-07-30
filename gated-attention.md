# Gated Attention: One Sigmoid That Kills the Attention Sink

Gating is one of the oldest tricks in deep learning: multiply a signal by a learned number in $(0,1)$ so the network can decide, per input, *how much of this to let through*. LSTMs (1997) used it to control memory; Highway Networks used it to let gradients skip layers; modern state-space models (Mamba) and linear-attention variants are gates all the way down. Softmax attention is the one place gating was mostly **absent** — and it turns out that a single, almost trivial gate bolted onto attention fixes several of the transformer's most stubborn pathologies at once.

The paper this note is about — **"Gated Attention for Large Language Models: Non-linearity, Sparsity, and Attention-Sink-Free"** (Qiu et al., 2025), which won a **NeurIPS 2025 Best Paper** award (4 of 5,290 submissions) and an Oral — does something the field rarely bothers to do: it takes one small idea and ablates it *exhaustively*. Thirty variants, 15B-parameter Mixture-of-Experts and 1.7B dense models, 3.5 trillion training tokens, gates in five different positions with every combination of head-specific/shared, elementwise/scalar, multiplicative/additive, sigmoid/SiLU. Out of that sweep comes one clear recommendation: put a **head-specific, elementwise sigmoid gate right after the attention output, before the output projection**. That variant improves loss, lets you train at higher learning rates without diverging, and — most strikingly — makes the notorious **attention sink** and **massive activations** almost vanish, which in turn buys much better long-context extrapolation. This is not a paper-only curiosity: the exact gate ships in **Qwen3-Next**.

This note explains *why* such a tiny change does so much. Two reasons carry the whole story: the gate injects a **non-linearity** into a path that was secretly just one low-rank linear map, and its output is **query-dependent and sparse**, which finally gives a softmax head a way to say "nothing here is relevant" without dumping its leftover probability onto the first token. If you have not yet read the [Attention note (005)](./training_llms/005_Attention.md) or the [Efficient Attention note (006)](./training_llms/006_EffecientAttention.md), skim the first — everything here builds on standard scaled-dot-product attention. The attention-sink and massive-activation story also touches [Normalisation Layers (003)](./training_llms/003_NormalisationLayers.md).

---

## Table of Contents

- [Setup and Notation](#setup-and-notation)
- [A 30-second recap of softmax attention](#a-30-second-recap-of-softmax-attention)
- [The gate: essentially one line of code](#the-gate-essentially-one-line-of-code)
- [Why it helps I — non-linearity in a secretly-linear path](#why-it-helps-i--non-linearity-in-a-secretly-linear-path)
- [Why it helps II — query-dependent sparsity](#why-it-helps-ii--query-dependent-sparsity)
- [The payoff: draining the attention sink and massive activations](#the-payoff-draining-the-attention-sink-and-massive-activations)
- [Long context and Qwen3-Next](#long-context-and-qwen3-next)
- [What the ablations actually say](#what-the-ablations-actually-say)
- [Takeaways](#takeaways)
- [Sources](#sources)

---

## Setup and Notation

A handful of symbols recur throughout. All of them live inside a single attention layer.

| Symbol | Meaning |
| --- | --- |
| $X$ | the layer's input hidden states (already normalised), shape $n \times d_{\text{model}}$, one row per token |
| $n$ | sequence length (number of tokens) |
| $h$ | index of an attention head; there are $H$ heads, each of width $d_{\text{head}}$ |
| $Q, K, V$ | queries, keys, values: $Q = X W_Q$, etc., each head slicing out its own $d_{\text{head}}$ columns |
| $Y_h$ | the scaled-dot-product-attention output of head $h$, shape $n \times d_{\text{head}}$ (before the output projection) |
| $W_O$ | the output projection that mixes all heads back to $d_{\text{model}}$ |
| $W_{\theta,h}$ | the **gate** projection for head $h$: a learned matrix producing the gating logits |
| $\sigma$ | the logistic sigmoid, $\sigma(z) = 1/(1+e^{-z})$, squashing any real number into $(0,1)$ |
| $\odot$ | elementwise (Hadamard) product |

The only genuinely new object is $W_{\theta,h}$ — everything else is standard attention.

---

## A 30-second recap of softmax attention

For one head, attention takes the queries, scores them against the keys, normalises those scores into a probability distribution over positions, and uses that distribution to average the values:

$$Y_h = \mathrm{softmax}\!\left(\frac{Q_h K_h^\top}{\sqrt{d_{\text{head}}}}\right) V_h.$$

Read left to right: $Q_h K_h^\top$ is an $n \times n$ table of "how much does query token $i$ care about key token $j$"; dividing by $\sqrt{d_{\text{head}}}$ keeps those logits from blowing up as the head gets wider; the row-wise `softmax` turns each row into a probability distribution that **must sum to 1**; and multiplying by $V_h$ replaces each token's representation with a weighted average of the value vectors it attended to. All $H$ heads' outputs $Y_h$ are concatenated and passed through $W_O$ to produce the layer output that gets added back to the residual stream.

Two features of this recap matter for what follows, and it is worth flagging them now because they are exactly where the pathologies come from. First, **`softmax` cannot abstain**: because each row sums to 1, a head that finds nothing relevant for the current token still has to put its probability mass *somewhere*. Second, the value-then-output step, $Y_h$ followed by $W_O$, is **purely linear** — there is no non-linearity anywhere between the values and the residual stream. Both of these are about to be fixed by one multiplication.

---

## The gate: essentially one line of code

The recommended gate — call it **G1**, for reasons that become clear in a moment — modifies the head output like this:

$$Y'_h = Y_h \odot \sigma\!\left(X W_{\theta,h}\right).$$

That is the whole method. Let us unpack every piece, because each choice in it was ablated and each one matters:

- $X W_{\theta,h}$ takes the layer input $X$ (the same hidden states that produced the query) and projects it through the head's own gate matrix to get one gating **logit per output channel per token**. Because the input is $X$, the gate is **query-conditioned**: it depends on *what the current token is asking for*.
- $\sigma(\cdot)$ squashes each logit into $(0,1)$. A value near 1 means "let this channel of this head's output through untouched"; a value near 0 means "suppress it."
- $\odot$ applies the gate **elementwise**: every one of the $d_{\text{head}}$ output channels gets its own gate value, not a single scalar for the whole head. This "fine-grained" form beat the scalar version in the sweep.
- The gate is **head-specific** (each head has its own $W_{\theta,h}$) and **multiplicative** (it multiplies the output rather than being added to it). These two facts turn out to be the single most important design choices — more on that in the ablations.
- Crucially, the gate sits **after** scaled-dot-product attention but **before** $W_O$. That placement is what makes it position G1, and it is what gives it both of its superpowers below.

The cost is negligible: $W_{\theta,h}$ is a small projection, and the paper measures **under 2% latency overhead**. For that price you get a consistent quality gain and the stability and long-context benefits described below.

Why "G1"? Because a gate could in principle sit in several places, and the paper tries all of them:

![A schematic of one attention block drawn left to right — input hidden states X, then the Q/K/V projections, then the scaled-dot-product-attention core, then the output projection Wo, then the block output. Small circled-times gate glyphs hang off five points in the pipeline: G1 sits on the arrow between SDPA and Wo and is highlighted in red as the winner, labelled post-SDPA and query-conditioned; G2 sits on the value path and is coloured orange with the note that it is denser and weaker; G3 sits on the key path, G4 on the query path, and G5 after the output projection, all in grey. A caption reads that every gate multiplies its stream by sigmoid of X Wtheta, elementwise and per head.](assets/gated_positions_g1_g5.jpg)

- **G1** — on the SDPA output, before $W_O$ (query-conditioned). **The winner.**
- **G2** — on the value path ($V$). Query-independent, and empirically its gates stay *dense* (rarely near 0), which we will see is why it underperforms.
- **G3, G4** — on the key and query paths.
- **G5** — after the output projection, on the final block output.

G2–G5 all operate on a single representation stream and modulate it by that same stream; only G1's gate is computed from the query-side input $X$ while modulating the *attention result*, so only G1 can ask "given what this token wanted, was the retrieved content actually useful?" That distinction is the seed of both explanations below.

---

## Why it helps I — non-linearity in a secretly-linear path

Here is a fact about multi-head attention that is easy to miss. Within one head, the value vectors are $V_h = X W_V^{(h)}$ (a linear map down to $d_{\text{head}}$ dimensions), and the head's contribution to the output is $Y_h W_O^{(h)}$ (a linear map back up to $d_{\text{model}}$). The attention weights aside, the transformation applied to the *content* of each attended token, from input to residual stream, is $W_V^{(h)} W_O^{(h)}$ — **a product of two linear maps, hence a single linear map**, and one whose rank is capped at $d_{\text{head}}$ because it is squeezed through the narrow head. There is no non-linearity anywhere in this content path.

![Two horizontal pipelines stacked. The top pipeline, labelled 'Standard attention: a linear bottleneck', shows the SDPA output Y passing through a Wv-side map of width d_head and then through Wo to the output, with a red brace underneath noting that the two linear maps compose into one map of rank at most d_head with nothing non-linear between them. The bottom pipeline, labelled 'Gated attention: sigmoid breaks the bottleneck', shows Y passing through an elementwise sigmoid gate and then Wo, with a green caption saying the gate is the only non-linearity between V and Wo and makes the path strictly more expressive. A small inset on the right plots the sigmoid curve rising from 0 to 1.](assets/gated_lowrank_nonlinearity.jpg)

The G1 gate inserts a sigmoid *exactly in the middle of that composed linear map*. Now the content path is $Y_h \odot \sigma(X W_{\theta,h})$ followed by $W_O$ — no longer expressible as one matrix, because the sigmoid bends it. In effect the paper is turning the value-to-output path into a tiny **gated MLP**, the same GLU-style non-linearity ($\text{value} \odot \sigma(\text{gate})$) that made SwiGLU feed-forward blocks so effective, but placed inside attention. This is the "non-linearity upon the low-rank mapping" that the title refers to, and it is the mechanism-agnostic half of the benefit: even ignoring attention sinks entirely, a strictly more expressive head fits the data better. It also explains a small but real finding in the sweep — that the gate helps *more* the narrower the head is, because a narrow head is exactly where the low-rank linear bottleneck bites hardest.

---

## Why it helps II — query-dependent sparsity

The second reason is about *what the gate learns to do*, not just that it adds capacity. Because G1's gate reads the query-side input $X$, it can look at the current token and decide the retrieved attention output is not worth passing on — and the measured gates show it does exactly that, **aggressively**. Across heads the average gate value is roughly **0.116**, with most of the distribution piled up near 0. In other words the gate is **sparse**: for any given token, most heads' outputs are largely switched off, and only a few fire.

![A histogram of gate scores on the interval 0 to 1. The G1 post-SDPA gate distribution, in dark red, is heavily concentrated near 0 with a tall spike below 0.1 and a dashed vertical line marking its mean around 0.12, labelled sparse. A second distribution in light orange, the G2 value gate, is spread broadly across the middle of the range with a mean around 0.45, labelled dense. The title notes that the winning gate is sparse and that most heads stay near 0 and abstain.](assets/gated_gate_sparsity_hist.jpg)

Contrast this with G2, the value-path gate: its gates stay *dense*, spread across the middle of $(0,1)$, and it underperforms. The difference is the query-conditioning. G2 gates the value based on the value itself, which does not know what the query wanted; G1 gates the attention *result* based on the query, so it can genuinely evaluate relevance. Sparsity is therefore not something the paper imposes — it is what a query-conditioned gate *chooses* when trained, and it correlates with the better models.

Why does the ability to be sparse matter so much? Recall from the recap that **softmax cannot abstain**. A head processing a token for which nothing in the context is relevant is still forced to output a convex combination of value vectors — it cannot output "nothing." The gate is the missing escape hatch: when $\sigma(X W_{\theta,h}) \approx 0$, the head's contribution $Y'_h \approx 0$ *regardless of where the attention mass landed*. The head can now do nothing cleanly. Hold onto that sentence, because it is the entire explanation for the next section.

---

## The payoff: draining the attention sink and massive activations

The **attention sink** is one of the most reported oddities in transformers: across almost every layer and head, a large fraction of attention mass piles onto the very first token (often the `BOS` token), even though that token carries no useful information. In the paper's baseline, the first token soaks up on average **46.7%** of all attention weight, with some layers far worse (one layer parks **83%** of its mass on token 0). Closely related are **massive activations** — a few hidden-state entries with enormous magnitude, wildly out of line with everything else — which make training numerically touchy and force conservative learning rates.

The standard explanation ties these together, and it follows directly from the two facts we flagged in the recap. A head that has nothing relevant to attend to still must spend its probability mass; the cheapest place to dump it is a fixed, always-present, low-information token — the first one. To make token 0 a reliable "null sink," the model grows that token's key norm and the corresponding value, which shows up as a massive activation; softmax then amplifies even a small logit advantage into a large probability, so the sink is **self-reinforcing** once it starts. The whole contraption is an emergent workaround for softmax's inability to abstain.

Gating removes the *reason* for the workaround. Once a head can zero its own output with the gate, it no longer needs a garbage token to point at — it just closes the gate and contributes nothing. The paper measures precisely this: first-token attention drops from **46.7% to about 4.8%** globally, and the pathological layers collapse too (the 83% layer falls to ~4%, another from 41% to ~1%). With the sink gone, the massive activations it was propping up shrink as well, and because the gate also **caps how hard any single token can push into the residual stream**, the whole forward pass becomes better-behaved. That numerical calm is what lets gated models tolerate **larger learning rates** and train more stably — a direct, practical dividend of a cleaner activation landscape.

![Two side-by-side attention heatmaps over a 40-token causal sequence, query position on the vertical axis and key position on the horizontal. The left map, baseline softmax attention, has a bright vertical stripe down the first column showing almost all mass concentrated on token 0, and is labelled with an average of about 45% attention on token 0. The right map, gated attention, has no bright first column and instead shows attention following a smooth band along the causal diagonal, labelled with about 3% attention on token 0. A shared title notes that the gate drains the attention sink from 46.7% down to 4.8% of mass on token 0.](assets/gated_attention_sink_heatmap.jpg)

It is worth stressing that this is a *mechanistic* fix, not a regulariser bolted on to suppress a symptom. The sink and the massive activations were both compensations for a missing capability; supply the capability and the compensations dissolve on their own.

---

## Long context and Qwen3-Next

Attention sinks are not just an aesthetic problem — they are a major obstacle to **long-context extrapolation**. When a model that was trained at, say, 32k tokens is asked to run at 128k, the sink behaviour interacts badly with the extended positions and the distribution of attention gets brittle. Because gated attention keeps attention well-distributed instead of collapsed onto token 0, it extrapolates far better.

The paper's numbers show the split clearly. **Within** the trained context window (up to 32k) the gated model is only slightly ahead of the baseline — the expressiveness gain, but nothing dramatic. **Beyond** it, at 64k–128k reached via YaRN-style RoPE interpolation, the gated model **significantly** outperforms the baseline. (Briefly: RoPE encodes position by rotating query/key subvectors by an angle proportional to position; you stretch a model to longer contexts by increasing the RoPE base frequency and/or interpolating those angles — YaRN is one such recipe — so the model sees "slower" rotations and generalises past its training length. The [Position Embeddings note (002)](./training_llms/002_PositionEmbeddings.md) covers RoPE in detail.) The gate does not change RoPE; it just removes the sink pathology that made long-context attention fragile in the first place.

This is why the result matters beyond the paper: the winning **SDPA-output gate is used in Qwen3-Next**, which targets ultra-long contexts (up to ~1M tokens). A NeurIPS Best Paper that is *already* in a shipping production model is a rare and strong validation — the systematic ablation was not academic; it directly informed a real architecture.

---

## What the ablations actually say

The value of this paper is less the gate itself (people had tried attention gates before) than the **clean, large-scale attribution** of which design choices matter and by how much. Distilled from the 30-variant sweep:

- **Head-specific beats shared** — this is the single most impactful choice. When one gate is shared across heads, the model produces larger, less selective gate values and loses the per-head specialisation that makes the sparsity useful. Give every head its own gate.
- **Elementwise beats scalar** — a per-channel gate ($d_{\text{head}}$ values) outperforms a single scalar per head. Finer control helps.
- **Multiplicative beats additive** — modulating by multiplication (a true gate) beats adding a learned bias term.
- **Sigmoid beats SiLU** for the SDPA-output gate specifically. A clean bounded $(0,1)$ gate is what you want here; the unbounded SiLU is worse in this position.
- **Position G1 beats G2–G5** — gating the SDPA output (query-conditioned) is best; the value gate G2 is the notable loser precisely because its gates stay dense and query-blind.

None of these individually is a surprise, but having them measured at 15B/3.5T scale, on the same footing, is what turns folklore into a recommendation you can adopt with confidence.

---

## Takeaways

- **One multiplication does the work.** Replace $Y_h$ with $Y_h \odot \sigma(X W_{\theta,h})$ — a head-specific, elementwise, multiplicative sigmoid gate on the attention output, before $W_O$. Under 2% latency, consistent quality gain.
- **Two mechanisms, not one.** (1) The gate is the **only non-linearity** in an otherwise low-rank linear $V \to W_O$ path, making each head strictly more expressive (a GLU inside attention). (2) It is **query-conditioned and learns to be sparse** (mean gate ≈ 0.116), giving softmax heads the ability to *abstain* that they never had.
- **The sink was a symptom.** Because softmax can't abstain, models learn to dump idle probability on the first token, building an attention sink and the massive activations that support it. Give heads a gate and the sink drains itself: first-token attention **46.7% → ~4.8%**.
- **Stability and length are downstream benefits.** A calmer activation landscape allows **higher learning rates**; a sink-free attention map **extrapolates** to 64k–128k far better than the baseline.
- **It's real.** The SDPA-output gate ships in **Qwen3-Next**; this is a Best Paper whose central recommendation is already in production.

---

## Sources

- Qiu, Zhang, Lin, Xu, Han, Dong, Chen, Yang, Lin, Yang, Zhou et al., [Gated Attention for Large Language Models: Non-linearity, Sparsity, and Attention-Sink-Free](https://arxiv.org/abs/2505.06708) (arXiv:2505.06708, 2025) — the paper; NeurIPS 2025 Best Paper and Oral.
- [OpenReview page](https://openreview.net/forum?id=1b7whO4SfY) — reviews and discussion.
- Official code and models: [qiuzh20/gated_attention](https://github.com/qiuzh20/gated_attention) (GitHub) and [QwQZh/gated_attention](https://huggingface.co/QwQZh/gated_attention) (Hugging Face) — the `Qwen3Attention` implementation with `headwise_attn_output_gate` / `elementwise_attn_output_gate` flags.
- [NeurIPS 2025 Best Paper Review: Qwen's Systematic Exploration of Attention Gating](https://towardsdatascience.com/neurips-2025-best-paper-review-qwens-systematic-exploration-of-attention-gating/) (Towards Data Science) — the clearest walkthrough of the G1–G5 positions and the sparsity argument.
- [NeurIPS 2025 Best Papers TL;DR part 1: Gated Attention](https://virtuslab.com/blog/ai/neur-ips-2025-papers-part-1) (VirtusLab) — the low-rank non-linearity framing and per-layer sink numbers.
- [Announcing the NeurIPS 2025 Best Paper Awards](https://blog.neurips.cc/2025/11/26/announcing-the-neurips-2025-best-paper-awards/) (NeurIPS Blog) — the award citation.
- On attention sinks generally: Xiao et al., [Efficient Streaming Language Models with Attention Sinks (StreamingLLM)](https://arxiv.org/abs/2309.17453) (ICLR 2024), and Gu et al. on massive activations — background for why the sink forms.
