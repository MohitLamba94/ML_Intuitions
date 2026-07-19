# Attention: from a Single Dot Product to Multi-Head, MQA, and GQA

A transformer layer has to let each token *look at* other tokens and pull in whatever it needs from them — the verb needs its subject, the pronoun needs its antecedent, the closing bracket needs its opening one. Attention is the mechanism that does this, and the cleanest way to think about it is as a **soft, differentiable lookup**: every token emits a *query* ("what am I looking for?"), every token also advertises a *key* ("what do I offer?"), and once a query has matched against the keys, each token hands over a *value* ("here is what you get if you pick me"). The output for a token is a weighted blend of the values, where the weights come from how well its query matched each key. Nothing is hard-coded — the queries, keys, and values are all learned linear projections, so the model learns *what to look for* end to end.

This note builds that picture up in layers. We first fix notation, then develop the query/key/value intuition and turn it into the matrix form $\mathrm{softmax}(QK^\top/\sqrt{d_k})V$ — with a diagram, because the jump from "a dot product between two vectors" to "a matrix multiply" is exactly where the picture usually goes fuzzy. Then we explain *why* the $\sqrt{d_k}$ scaling has to be there. Next we build **multi-head attention** and make precise how the *head* axis behaves just like the *batch* axis — which is what lets a GPU do all heads in one shot instead of a slow Python loop. Finally we get to **MQA** and **GQA**, the inference-time variants that shrink the KV cache; for the KV-cache machinery itself we lean on the companion note [Why Hardware Makes Matrix Multiply Fast](../gpu-tpu-matmul-flashattention.md) rather than re-deriving it.

The one cost this note does *not* address is that the $n \times n$ score matrix makes attention **quadratic** in sequence length — the sequel [006 — Efficient Attention](./006_EffecientAttention.md) is entirely about escaping that quadratic curse (sparse, low-rank, linear/kernelized, and hashing attention).

---

## Table of Contents

- [Setup and Notation](#setup-and-notation)
- [The core idea: a soft, differentiable lookup](#the-core-idea-a-soft-differentiable-lookup)
- [Single-head attention as matrix multiplication](#single-head-attention-as-matrix-multiplication)
- [Why divide by √dₖ](#why-divide-by-dₖ)
- [Multi-head attention](#multi-head-attention)
- [The head axis is just another batch axis](#the-head-axis-is-just-another-batch-axis)
- [QK-Norm: bounding attention logits for stable training](#qk-norm-bounding-attention-logits-for-stable-training)
- [MQA and GQA: shrinking the KV cache](#mqa-and-gqa-shrinking-the-kv-cache)
- [Takeaways](#takeaways)
- [Sources](#sources)

---

## Setup and Notation

A handful of symbols recur throughout; each is reintroduced where it first does real work, but here they are in one place.

| Symbol | Meaning |
| --- | --- |
| $n$ | Sequence length — the number of tokens we attend over. |
| $d_{model}$ (also $d$) | The model width — the length of each token's vector as it flows through the network. |
| $X \in \mathbb{R}^{n \times d}$ | The input: one row per token, each row a $d$-dimensional vector. |
| $h$ | Number of attention **heads**. |
| $d_k$ | Per-head dimension of queries and keys. In multi-head attention $d_k = d_{model}/h$. |
| $d_v$ | Per-head dimension of values (almost always $d_v = d_k$). |
| $W^Q, W^K, W^V$ | Learned projection matrices that turn token vectors into queries, keys, and values. |
| $W^O$ | Learned output projection that mixes the concatenated head outputs back to width $d_{model}$. |
| $Q, K, V$ | The query, key, and value matrices: $Q = XW^Q$, etc. $Q,K \in \mathbb{R}^{n \times d_k}$, $V \in \mathbb{R}^{n \times d_v}$. |
| $S = QK^\top$ | The **scores** (or logits): an $n \times n$ matrix of query–key similarities. |
| $A$ | The **attention weights**: $S$ after scaling and a row-wise softmax; each row sums to 1. |
| $B$ | Batch size (number of independent sequences processed together). |
| $g$ | Number of **key/value heads** (for GQA/MQA). $g = h$ is standard multi-head; $g = 1$ is multi-query. |

Convention: a **row** is always one token. So "row $i$ of $Q$" is the query vector for token $i$, and "$A$ is row-wise softmax" means each token's attention weights sum to 1 independently.

---

## The core idea: a soft, differentiable lookup

Imagine a Python dictionary. You have a `query`, you compare it against every stored `key`, and where it matches you retrieve the associated `value`. A normal dictionary match is *hard*: exactly one key matches, you get exactly one value. Attention is the **soft** version: the query is compared against *all* keys, each comparison produces a score, the scores are turned into weights that sum to 1, and the returned value is a **weighted average** of *all* the values. Instead of "return the value at the one matching key," attention returns "a blend of every value, weighted by how well its key matched."

Concretely, for a single token $i$:

1. It produces a query vector $q_i$ (by projecting its own token vector through $W^Q$).
2. Every token $j$ has a key vector $k_j$ and a value vector $v_j$.
3. The score of $i$ attending to $j$ is the **dot product** $q_i \cdot k_j$ — a big number when the query and key point the same way, i.e. when token $j$ is "the kind of thing token $i$ is looking for."
4. Softmax over $j$ turns those scores into weights $a_{ij}$ that are all positive and sum to 1.
5. The output for token $i$ is $\sum_j a_{ij}\, v_j$ — the weighted blend of the values.

This is the picture most people hold comfortably: *one query, dotted against many keys, then a weighted sum of values.* Two things make it "soft" and, crucially, **differentiable**: (a) softmax is smooth, so a small change in a score smoothly changes the weights, and (b) $q, k, v$ are all linear functions of the token vectors, so gradients flow back into $W^Q, W^K, W^V$ and the model *learns* what to look for. A hard argmax lookup would have zero gradient almost everywhere and could never be trained. The next section is nothing more than doing steps 1–5 for **all** tokens $i$ at once — which is exactly where it becomes a matrix multiply.

---

## Single-head attention as matrix multiplication

The per-token story above involves one query vector at a time. But we have $n$ tokens, and every one of them runs the same computation. Rather than loop over tokens, we **stack** all $n$ query vectors as the rows of a matrix $Q$, all $n$ keys as rows of $K$, and all $n$ values as rows of $V$. Now the whole thing collapses into two matrix multiplies with a softmax in between:

$$\mathrm{Attention}(Q, K, V) = \mathrm{softmax}\!\left(\frac{Q K^\top}{\sqrt{d_k}}\right) V.$$

Let us read it left to right, because each piece corresponds exactly to one step from the previous section.

- **$Q K^\top$ — all the dot products at once.** $Q$ is $n \times d_k$ (one query per row) and $K^\top$ is $d_k \times n$ (one key per *column*). Their product $S = QK^\top$ is $n \times n$, and entry $S_{ij}$ is precisely the dot product $q_i \cdot k_j$ — the score of token $i$ attending to token $j$. So a single matrix multiply produces the *entire table* of "who is looking at whom." Row $i$ of $S$ is token $i$'s scores against every key.
- **$\tfrac{1}{\sqrt{d_k}}$ — the scaling.** We divide every score by $\sqrt{d_k}$ before the softmax. Why this exact factor is the subject of the [next section](#why-divide-by-dₖ); for now, read it as "keep the numbers in a sane range."
- **$\mathrm{softmax}(\cdot)$ — done row by row.** The softmax is applied **independently to each row** of the scaled scores. Row $i$ becomes a probability distribution over the $n$ keys: all weights positive, summing to 1. Call the result $A$ (the attention-weight matrix). This is step 4 for every token simultaneously.
- **$\cdots V$ — the weighted average.** $A$ is $n \times n$ and $V$ is $n \times d_v$, so $AV$ is $n \times d_v$. Row $i$ of the output is $\sum_j A_{ij}\, v_j$: the blend of value rows, weighted by token $i$'s attention. That is step 5, again for every token at once.

The figure below shows the shapes flowing through, with a real (causal) attention matrix in the middle so you can see what $A$ actually looks like:

![A left-to-right pipeline of coloured matrix blocks. A blue block Q of shape n by d_k, times a purple block K-transpose of shape d_k by n, equals an n-by-n score matrix shown as a blue heatmap labelled A = softmax(QK-transpose / sqrt(d_k)); the heatmap is lower-triangular (a causal mask), rows sum to 1, with axes "query i" down the side and "key j" along the bottom. That heatmap is then multiplied by a green V block of shape n by d_v to give an orange output block of shape n by d_v. Captions: step 1, every query dotted with every key gives an n-by-n table of similarities; step 2, each output row is a weighted average of the value rows.](../assets/attn_matmul_flow.jpg)

The one mental adjustment worth making explicit: **the dot product you already understand is hiding inside $QK^\top$.** You are not doing anything new when you go from vectors to matrices — you are doing the *same* dot product $n \times n$ times, and matrix multiplication is just the notation (and the hardware primitive) that does all of them in one shot. The $n \times n$ score matrix is the object to anchor on: it is literally the grid of "how much does row-token attend to column-token."

**A one-line note on masking.** In a decoder LLM, token $i$ must not peek at future tokens $j > i$. This is enforced by setting those entries of $S$ to $-\infty$ *before* the softmax (so their weights become exactly 0), which is why the heatmap above is lower-triangular. That single change — the **causal mask** — is what makes the KV cache possible during generation; the details live in [Why Hardware Makes Matrix Multiply Fast, Part 7](../gpu-tpu-matmul-flashattention.md#part-7--the-kv-cache-prefill-decode-and-why-inference-is-memory-bound), and we return to its consequences in the MQA/GQA section.

---

## Why divide by √dₖ

The scaling factor $1/\sqrt{d_k}$ looks like a fussy detail, but leaving it out breaks training, and the reason is a short variance argument.

Each score is a dot product of two $d_k$-dimensional vectors: $q_i \cdot k_j = \sum_{m=1}^{d_k} q_{im} k_{jm}$. Suppose, at initialization, the entries of $q$ and $k$ are roughly independent with mean 0 and variance 1 (a reasonable approximation for freshly initialized projections). Then each product term $q_{im} k_{jm}$ has variance $\approx 1$, and summing $d_k$ independent such terms gives a dot product with variance $\approx d_k$ — so a typical score has magnitude on the order of $\sqrt{d_k}$. As $d_k$ grows, the *scores grow with it.*

Why is a large score bad? Because softmax is scale-sensitive. Feed it logits that are spread far apart and it collapses toward a **near one-hot** distribution — almost all the weight lands on the single largest score, the rest go to nearly 0. That is the **saturated regime**, and in it the gradient of softmax is almost zero: nudging any score barely moves the output, so almost no learning signal flows back. Training stalls before it starts. Dividing by $\sqrt{d_k}$ rescales the score variance back down to $\approx 1$ regardless of head size, keeping softmax in a responsive, well-conditioned regime where gradients are healthy. This is the argument given in Vaswani et al. (2017), §3.2.1.

The figure makes both halves concrete:

![Two panels. Left: a bar chart of attention weights over eight key positions, for the same logit shape scaled up as if d_k were 8, 64, and 512 with no scaling; as d_k grows the distribution collapses from moderately peaked to essentially a single bar at position 0 — near one-hot. Right: a line plot of the average maximum softmax weight (over random unit-variance queries and keys, 8 keys each) versus d_k on a log axis. The "no scaling" orange curve climbs from about 0.5 toward 1.0 as d_k grows, i.e. more and more saturated; the "scaled by 1/sqrt(d_k)" blue curve stays flat around 0.35 regardless of d_k. A dashed line marks the uniform weight 1/8.](../assets/attn_sqrt_dk.jpg)

The left panel takes one fixed "shape" of logits and scales it up the way growing $d_k$ would; the softmax goes from a spread-out distribution to a single spike. The right panel measures peakiness (the average largest weight) across many random draws: without scaling it marches toward 1 (fully saturated) as $d_k$ increases, while with the $1/\sqrt{d_k}$ scaling it holds steady no matter how wide the head is. That flat blue curve is the whole point — the scaling **decouples softmax sharpness from head dimension.**

---

## Multi-head attention

A single attention head produces, for each token, *one* weighted average — one "view" of the context. But a token often needs to track several relationships at once: a verb might want its subject *and* its object, a word might care about both the token two positions back and one twelve positions back. One softmax-weighted average can only emphasize one blend of things at a time. The fix is to run several attention operations in parallel, each free to attend differently, and combine them.

The elegant part is that this costs almost nothing extra, because instead of running $h$ *full-width* attentions, we **split the width across the heads**. With $d_k = d_{model}/h$, each head gets its own small projections $W^Q_i, W^K_i, W^V_i$ that map tokens into a $d_k$-dimensional subspace, runs the exact single-head attention from above in that subspace, and produces an $n \times d_k$ output. We then **concatenate** the $h$ head outputs back into an $n \times d_{model}$ matrix and pass it through one more learned projection $W^O$ that lets the heads' results mix:

$$\mathrm{MultiHead}(X) = \mathrm{Concat}(\text{head}_1, \dots, \text{head}_h)\, W^O, \qquad \text{head}_i = \mathrm{Attention}(XW^Q_i,\, XW^K_i,\, XW^V_i).$$

Read term by term: each $\text{head}_i$ is just the scaled-dot-product attention we already built, but on the narrow $d_k$-wide projection belonging to head $i$; $\mathrm{Concat}$ stacks the $h$ outputs side by side to rebuild the full width; and $W^O$ (shape $d_{model} \times d_{model}$) mixes information across heads so the layer's output isn't just $h$ independent blocks glued together.

The concrete numbers make the "split, don't add" point clear: the original Transformer uses $d_{model} = 512$ and $h = 8$, so $d_k = 512/8 = 64$. Each head is small, and because the total width is unchanged, the total compute is roughly the same as one big head — you get $h$ different views of the context essentially for free.

![A flow diagram. On the left, a token matrix X of shape n by d_model shown as four coloured vertical column-slices; a caption reads "split width into h heads, d_k = d_model/h". Each slice feeds its own attention block (head 1 through head 4, each shape n by d_k, each labelled "own W_i^Q, W_i^K, W_i^V"). The four head outputs feed a "concat" block of shape n by d_model, which feeds a "mix, times W^O" block, producing an "output" block of shape n by d_model. A caption reads: example d_model=512, h=8 gives d_k=64 — total width and compute is preserved, just carved into h subspaces.](../assets/attn_multihead.jpg)

Each head learns to specialize: empirically some heads track syntactic dependencies, some track positional offsets, some copy rare tokens. But you do not design this — you give the model $h$ subspaces and it discovers useful roles for them during training.

---

## The head axis is just another batch axis

Here is the implementation insight that trips people up. In a naive first implementation you might write a Python `for` loop: for each of the $h$ heads, project, compute $QK^\top$, softmax, multiply by $V$, and stash the result. This is correct but **slow on a GPU**, and understanding *why* is the whole point.

A GPU is a throughput machine: it has hundreds of parallel compute units (SMs) that are happiest when handed one big, dense matrix multiply that keeps them all busy. A Python loop does the opposite — it issues $h$ *separate*, *small* GPU operations one after another. Each launch carries fixed overhead, the small per-head matmuls under-fill the hardware (most SMs sit idle), and the ops run sequentially rather than overlapping. You pay $h$ times the launch cost to do $h$ tiny jobs, none of which saturates the chip. (The underlying "the pipe, not the pump, is the limit" model of GPU throughput is developed in [Why Hardware Makes Matrix Multiply Fast, Parts 2–3](../gpu-tpu-matmul-flashattention.md).)

The fix is to notice that **the $h$ heads are completely independent** — head 3's attention never touches head 5's data. That is exactly the property the *batch* axis already has: sequence 3 in a batch never interacts with sequence 5. So we can treat the head axis the same way we treat the batch axis. Concretely, we reshape the projected tensors from

$$(B,\; n,\; d_{model}) \quad\longrightarrow\quad (B,\; n,\; h,\; d_k) \quad\longrightarrow\quad (B,\; h,\; n,\; d_k),$$

reading each step: we start with a batch of $B$ sequences, each $n$ tokens of width $d_{model}$; we **split** the width into $h$ heads of size $d_k$ (this is just a view — no data moves, since $d_{model} = h \cdot d_k$); then we **transpose** so the head axis sits next to the batch axis. Now $(B, h)$ together form a single stack of $B \cdot h$ independent attention problems, and a batched matrix multiply handles all of them in **one** kernel launch. The math done per $(b, h)$ slice is byte-for-byte identical to the loop — we have only changed *how the work is scheduled*, replacing $B \cdot h$ tiny serial jobs with one large parallel one that fills the hardware.

This is the sense in which "the head is an extra dimension you vectorize over": to the matmul kernel, a head is indistinguishable from another batch element. Vectorizing over it turns $h$ sequential operations into a single massive one — the reason real implementations never loop over heads.

---

## QK-Norm: bounding attention logits for stable training

Recall the [$\sqrt{d_k}$ section](#why-divide-by-dₖ): dividing the scores by $\sqrt{d_k}$ fixes their variance to $\approx 1$ *at initialization*, assuming $q$ and $k$ have unit-variance entries. But that assumption only holds at the very start of training. As training proceeds, gradient descent is free to grow the weights $W^Q$ and $W^K$, which grows the norms $\lVert q \rVert$ and $\lVert k \rVert$, which grows the logits $q \cdot k$ — and the fixed $\sqrt{d_k}$ constant does nothing to stop it. So $\sqrt{d_k}$ is a one-time patch, not a running guarantee. **QK-Norm is the running guarantee.**

### The failure mode: logit growth → attention entropy collapse

At large scale this "logit drift" turns into an outright training-stability problem, and the mechanism is worth spelling out because it is the whole reason QK-Norm exists.

1. **Logits grow unboundedly.** Somewhere in training, a few attention logits start drifting to very large magnitudes. In the ViT-22B work (Dehghani et al., 2023) the maximum attention logit was observed climbing past **50,000**.
2. **Softmax saturates into one-hot.** Feed softmax a row whose top logit dwarfs the rest and it returns a near one-hot distribution — essentially all the weight on a single key. This is the same saturation the $\sqrt{d_k}$ section warned about, but now driven by trained weights, not head size.
3. **Attention entropy collapses.** A one-hot distribution has near-zero entropy. When this happens across heads, the model has effectively stopped *mixing* information — every token copies one other token. This "attention entropy collapse" (Zhai et al., 2023, *Stabilizing Transformer Training by Preventing Attention Entropy Collapse*) is empirically correlated with the training going bad.
4. **Training destabilizes.** Saturated softmax has vanishing gradients, huge logits interact badly with low precision (bf16/fp16 can overflow), and the result is **loss spikes and divergence** — the run either stalls or blows up.

![Two panels sharing the story. Left, a log-scale plot of the maximum attention logit magnitude versus training progress: the "no QK-norm" curve climbs exponentially past a dashed line marked "one-hot softmax / fp16 overflow risk" at 50,000 and hits a red X marked "loss spike / divergence", while the "with QK-norm" curve stays flat and bounded near 10. Right, attention entropy as a fraction of its maximum versus training progress: the "no QK-norm" curve starts high (~0.9) and collapses toward ~0.05 (annotated "entropy collapse"), while the "with QK-norm" curve stays steady around 0.66.](../assets/attn_qknorm_stability.jpg)

*Schematic illustration (stylized curves, not measured) of the mechanism reported by Dehghani et al. (2023, ViT-22B) and studied via small-scale proxies by Wortsman et al. (2024).*

### The fix: normalize queries and keys before the dot product

QK-Norm attacks the problem at its root — the *magnitude* of $q$ and $k$ — by normalizing them before they ever meet. There are two closely related forms:

- **L2 / cosine form (Henry et al., 2020 — the original "QKNorm").** L2-normalize each query and key vector along the head dimension so they have unit length, then multiply the dot product by a single **learnable scalar** $g$ instead of dividing by $\sqrt{d_k}$:

$$\mathrm{score}_{ij} = g \cdot \frac{q_i}{\lVert q_i \rVert} \cdot \frac{k_j}{\lVert k_j \rVert} = g \cdot \cos\theta_{ij}.$$

  Reading it: once $q$ and $k$ are unit vectors, their dot product is exactly the **cosine of the angle** between them, which lives in $[-1, 1]$ no matter how large the underlying weights grow. The logit is therefore bounded in $[-g, g]$, and $g$ is learned — so the network can still choose how "sharp" attention is allowed to get, but it can never drift there by accident. This is why the paper frames it as making softmax "less prone to arbitrary saturation without sacrificing expressivity." Henry et al. reported gains averaging about +0.93 BLEU on low-resource translation, but the technique's lasting importance turned out to be *stability at scale*.
- **LayerNorm/RMSNorm form (ViT-22B and most modern LLMs).** Apply a per-head LayerNorm (or RMSNorm) to $q$ and $k$ right after their projections, before $QK^\top$. This does not force exactly-unit vectors but it removes the runaway-magnitude degree of freedom, which is enough to keep logits in a sane range. Dehghani et al. found this let ViT-22B train stably across three orders of magnitude of learning rate — precisely the regime where the unnormalized model diverged.

Either way the effect is the blue curves above: logits stay bounded, entropy stays healthy, and the run does not spike. QK-Norm has consequently become a near-default ingredient in recent large models.

### Two refinements and one counterpoint

The idea has been pushed further in three directions worth knowing, in decreasing order of how established they are:

- **Combining QK-Norm with other stabilizers (Rybakov et al., 2024, *Methods of Improving LLM Training Stability*).** This work looks beyond attention logits at the L2 norm of *all* linear-layer outputs in a block, and finds the QKV, output-projection, and second-FFN layers grow the most under a high learning rate. It shows that pairing QK-LayerNorm with **softmax logit capping** (softly clamping the logits) — or normalizing after the QKV/FFN layers — lets you push the learning rate about **1.5× higher** than QK-Norm alone before divergence, with perplexity gains too. The takeaway: QK-Norm is necessary but not always sufficient; logit growth is one symptom of a broader magnitude-growth problem.
- **Normalizing everything (nGPT, Loshchilov et al., 2024, *Normalized Transformer*).** nGPT takes normalization to its logical extreme: *every* vector — embeddings, attention and MLP weight rows, hidden states — is constrained to unit norm, so all representations live on a hypersphere and each layer is a small rotation on that sphere. QK-Norm falls out as a special case (queries and keys are already unit vectors). The reported payoff is dramatic optimization speed: **4–20× fewer training steps** to reach the same quality, with stable gradients as a built-in consequence of the geometry.
- **A counterpoint — norm actually carries signal (NaLaFormer, Meng et al., 2025, *Norm-Aware Linear Attention*).** It is worth remembering that the query's *norm* is not pure noise. In softmax attention the query norm controls how *spiky* (low-entropy) that token's attention is — a large-norm query attends sharply, a small-norm one attends diffusely. Linear-attention variants (which drop the softmax to get $O(n)$ cost) accidentally throw this norm information away, causing an "entropy gap." NaLaFormer decouples each query/key into **norm** and **direction**, using the direction for matching and the norm to modulate spikiness on purpose. The lesson for QK-Norm: normalizing away the magnitude is a stability win, but the magnitude was doing a job, so aggressive normalization can cost a little expressivity — which is exactly why the practical forms keep a learnable scale $g$ to hand some of that control back.

---

## MQA and GQA: shrinking the KV cache

Everything so far is about *what attention computes*. MQA and GQA change *how much you have to store and move* when a trained model generates text — and to see why that matters, we build on [Why Hardware Makes Matrix Multiply Fast, Part 7](../gpu-tpu-matmul-flashattention.md#part-7--the-kv-cache-prefill-decode-and-why-inference-is-memory-bound), which covers the KV cache in detail. The short version of what that note establishes: when generating one token at a time (**decode**), the causal mask means each past token's key and value never change, so you compute them once and **cache** them; each new step then runs a single query against the whole cached $K, V$. That step does very little arithmetic but must **stream the entire KV cache out of memory**, so decode is **memory-bandwidth-bound** — its speed is set by how many bytes of cache you move per step, not by FLOPs. That note ends by pointing here, because the natural next question is: *can we make the cache smaller?*

The cache size is the lever. In standard multi-head attention you cache a separate $K$ and $V$ **for every one of the $h$ heads**, so the cache grows linearly with $h$. That is a lot of bytes to re-read on every single generated token. The observation behind both variants: you have $h$ *query* heads, but do you really need $h$ distinct *key/value* heads?

- **Multi-Query Attention (MQA)** — Shazeer (2019). Keep all $h$ query heads, but share a **single** key/value head across all of them ($g = 1$). The KV cache shrinks by a factor of $h$, so decode moves far fewer bytes and speeds up dramatically. The cost: collapsing all keys/values into one head throws away representational capacity, which can hurt quality and can make training less stable.
- **Grouped-Query Attention (GQA)** — Ainslie et al. (2023). The middle ground: use $g$ key/value heads with $1 < g < h$, and let each **group** of query heads share one KV head. For example 8 query heads with $g = 2$ means two groups of 4 query heads, each group sharing one $K, V$. This recovers almost all of multi-head's quality while still cutting the cache by roughly $h/g$. A practical bonus: a GQA model can be **uptrained** from an existing multi-head checkpoint with a small amount of extra training (mean-pool the KV heads within each group, then fine-tune), so you don't have to train from scratch.

The two named methods are just the endpoints and middle of one knob — the number of KV heads $g$:

![Three side-by-side diagrams, each with a row of 8 query heads (Q1–Q8) on top and a row of KV heads on the bottom, with lines showing which query heads read which KV head. Left, MHA (g = h): 8 query heads, 8 KV heads, one-to-one. Middle, GQA (1 < g < h): 8 query heads but only 2 KV heads, with Q1–Q4 sharing KV1 and Q5–Q8 sharing KV2 (query heads coloured by their group). Right, MQA (g = 1): all 8 query heads fan into a single shared KV head. Title: fewer KV heads means a smaller KV cache to stream each decode step, MHA to GQA to MQA.](../assets/attn_mha_mqa_gqa.jpg)

So the spectrum is $g = h$ (full multi-head, biggest cache, best quality) $\to$ $1 < g < h$ (GQA) $\to$ $g = 1$ (MQA, smallest cache, fastest, some quality risk). The rest of this section unpacks the three results from the GQA paper (Ainslie et al., 2023) that pin down *why GQA, and why $g$ around 8*, using their T5-Large / T5-XXL experiments (T5-XXL has $h = 64$ heads, so "MHA" here means 64 KV heads).

### The quality–speed tradeoff: GQA gets XXL quality at MQA speed

The headline result is a single quality-vs-speed plot. The authors take a pretrained multi-head T5-XXL checkpoint, convert it to MQA and to GQA-8 (i.e. $g = 8$), *uptrain* each with just 5% of the original pre-training compute, and measure both average task quality and inference time per sample against plain multi-head T5-Large and T5-XXL.

![A scatter plot of average performance (y, roughly 45.7 to 47.4) versus time per sample (x). MHA-XXL sits far right (slowest, ~1.5 relative time) at the top quality ~47.2. MHA-Large is bottom-left-ish (fast, ~0.37) but clearly lower quality ~46.0. MQA-XXL is far left (fastest, ~0.24) at quality ~46.6. GQA-XXL is also far left (~0.28, nearly as fast as MQA) but at quality ~47.1, essentially matching MHA-XXL.](../assets/attn_gqa_perf_vs_time.jpg)

*Figure recreated from Ainslie et al. (2023), "GQA: Training Generalized Multi-Query Transformer Models from Multi-Head Checkpoints" (arXiv:2305.13245), Figure 3, using the paper's data points. Reproduced for educational purposes.*

Read the four points as a story. **MHA-XXL** (top right) is the quality ceiling but also the slowest — its full 64-head KV cache is expensive to stream. **MHA-Large** (bottom) is fast simply because it is a smaller model, but it pays for that in quality. The interesting two are on the far left: **MQA-XXL** is the fastest of all, and lands well above MHA-Large in quality — so even the aggressive one-KV-head variant is a *favorable* trade. **GQA-XXL** is the punchline: it is nearly as fast as MQA yet sits right up at MHA-XXL's quality. You essentially get the big model's accuracy at the small model's latency. That is why GQA, not MQA, became the default.

### You don't retrain from scratch — you *uptrain*

A crucial practical point is that MQA/GQA models are not trained from zero. You start from an existing multi-head checkpoint, **mean-pool** the $h$ key (and value) projection matrices within each group down to $g$ of them, and then continue training for a *small fraction* of the original pre-training budget to let the model adjust. This "uptraining" is what makes converting a released MHA model cheap. The second result measures how much uptraining you actually need:

![A line plot of performance (y) versus uptraining proportion (x, from 0 to 0.1, i.e. 0% to 10% of pre-training compute). A dotted horizontal line marks the original multi-head model's performance (~57.5). The GQA-8 curve (squares) starts high at ~56.7 with zero uptraining and quickly rises to meet the MHA line by ~5%. The MQA curve (triangles) starts much lower at ~53.9 with no uptraining and climbs steeply, reaching ~56.9 at 5% and only approaching MHA near 10%.](../assets/attn_gqa_uptraining.jpg)

*Figure recreated from Ainslie et al. (2023), Figure 5, using the paper's data points. Reproduced for educational purposes.*

Two things jump out. First, **GQA-8 starts far higher** than MQA at zero uptraining (~56.7 vs ~53.9): keeping 8 KV heads instead of collapsing to 1 preserves much more of the original model's structure, so there is simply less to repair. Second, **GQA-8 recovers essentially all of the multi-head quality with very little uptraining** — it is already at the dotted MHA reference line by about 5% — whereas MQA needs noticeably more uptraining to close the gap and still trails slightly. So GQA is not only better at inference; it is *cheaper and safer to convert to*.

### Why $g \approx 8$: the cost of adding groups is not symmetric

If GQA-8 already matches MHA quality, why not push $g$ higher for even more quality headroom? Because the *speed* cost of adding KV heads is highly non-linear. The third result times decoding as a function of the number of groups:

![A line plot of decode time per sample in seconds (y) versus the number of GQA groups g (x, log scale: 1, 4, 8, 16, 32, 64), for input length 2048 and output length 512. A dotted line at ~2.53 s marks full multi-head (64 groups); a dotted line at ~0.49 s marks MQA (1 group). The GQA curve is almost flat and low from g=1 through g=8 (~0.49 to ~0.51 s), a shaded "cheap zone" region, then rises — ~0.59 at 16, ~0.80 at 32, and snapping up to ~2.53 at 64 (which equals MHA).](../assets/attn_gqa_time_vs_groups.jpg)

*Figure recreated from Ainslie et al. (2023), Figure 6, using the paper's data points. Reproduced for educational purposes.*

From $g = 1$ (MQA) up to about $g = 8$ the decode time barely moves — the KV cache is still small enough that streaming it is cheap, so those extra groups are nearly free. Past 8, the curve turns up sharply, and by $g = 64$ (which *is* full multi-head) you are back to the slow ~2.5 s. So the two curves together define the sweet spot: quality is essentially saturated by $g \approx 8$ (previous figure), while speed is still essentially free there (this figure). Choosing $g$ in that low range buys you MHA-level quality while staying on the flat part of the cost curve — exactly why real models land on a handful of KV heads rather than one or all.

**In practice.** GQA has become the default in modern open LLMs precisely because it sits at this sweet spot: **Llama 2 70B, Llama 3, and Mistral 7B all use GQA** (Llama 2 70B, for instance, uses 8 KV heads), while pure MQA appears in models like PaLM and Falcon. In every case the motivation is the same one from the KV-cache note — decode is bottlenecked on memory bandwidth, so the winning move is to move fewer bytes.

---

## Takeaways

- **Attention is a soft, differentiable dictionary lookup.** A query is compared to every key by dot product; softmax turns the scores into weights that sum to 1; the output is the weighted average of the values. Because everything is smooth and learned, the model discovers *what to look for*.
- **The matrix form is just the per-token dot product done all at once.** $QK^\top$ is the $n \times n$ grid of every query-against-every-key dot product; softmax is applied **row by row**; multiplying by $V$ turns each row of weights into a weighted average of value rows. Anchor on that $n \times n$ score matrix.
- **The $\sqrt{d_k}$ scaling keeps softmax responsive.** Dot products of $d_k$-dimensional vectors have variance $\approx d_k$, so unscaled scores grow with head size and drive softmax into a saturated, near-zero-gradient regime. Dividing by $\sqrt{d_k}$ holds the score variance at $\approx 1$ regardless of $d_k$.
- **Heads are parallel views on slices of the width.** With $d_k = d_{model}/h$, each head attends in its own subspace; concatenating and mixing with $W^O$ gives $h$ different context views at roughly the cost of one full-width head.
- **The head axis behaves exactly like the batch axis.** Heads are independent, so reshaping to $(B, h, n, d_k)$ lets one batched matmul do all heads in a single GPU kernel — vectorizing away the slow per-head loop without changing the math.
- **QK-Norm bounds the logits throughout training, not just at init.** $\sqrt{d_k}$ only fixes variance at initialization; as weights grow, logits drift, softmax saturates, attention entropy collapses, and the run diverges. Normalizing $q$ and $k$ (to cosine similarity, or via LayerNorm/RMSNorm) with a learnable scale caps the logits and keeps large-model training stable.
- **MQA and GQA trade KV-cache bytes for a little quality.** Fewer KV heads ($g < h$) means a smaller cache to stream every decode step, which is what actually bottlenecks generation. GQA ($1 < g < h$) is the modern default; MQA ($g = 1$) is the aggressive extreme.
- **The $n \times n$ matrix makes attention quadratic** in sequence length — the motivation for the whole efficient-attention literature, covered in the sequel [006 — Efficient Attention](./006_EffecientAttention.md).

---

## Sources

- Vaswani et al. (2017), [*Attention Is All You Need*](https://arxiv.org/abs/1706.03762) (the original Transformer; §3.2.1 gives the variance argument for the $\sqrt{d_k}$ scaling).
- Shazeer (2019), [*Fast Transformer Decoding: One Write-Head is All You Need*](https://arxiv.org/abs/1911.02150) (introduces Multi-Query Attention — share one KV head across all query heads).
- Ainslie et al. (2023), [*GQA: Training Generalized Multi-Query Transformer Models from Multi-Head Checkpoints*](https://arxiv.org/abs/2305.13245) (Grouped-Query Attention and how to uptrain it from a multi-head checkpoint).
- Touvron et al. (2023), [*Llama 2: Open Foundation and Fine-Tuned Chat Models*](https://arxiv.org/abs/2307.09288) (uses GQA in the larger models).
- Jiang et al. (2023), [*Mistral 7B*](https://arxiv.org/abs/2310.06825) (uses GQA together with sliding-window attention).
- Henry et al. (2020), [*Query-Key Normalization for Transformers*](https://arxiv.org/abs/2010.04245) (the original QKNorm: L2-normalize $q,k$ to cosine attention with a learnable scale, avoiding softmax saturation).
- Dehghani et al. (2023), [*Scaling Vision Transformers to 22 Billion Parameters*](https://openreview.net/pdf?id=Lhyy8H75KA) (attention-logit growth caused divergence at scale; QK-LayerNorm fixed it — the stability motivation).
- Zhai et al. (2023), [*Stabilizing Transformer Training by Preventing Attention Entropy Collapse*](https://arxiv.org/abs/2303.06296) (names and analyzes the entropy-collapse failure mode).
- Wortsman et al. (2024), [*Small-scale proxies for large-scale Transformer training instabilities*](https://arxiv.org/abs/2309.14322) (reproduces logit growth in small models and studies QK-norm / logit clipping fixes).
- Rybakov et al. (2024), [*Methods of Improving LLM Training Stability*](https://arxiv.org/abs/2410.16682) (QK-norm plus softmax capping / extra normalization allows ~1.5× higher learning rate).
- Loshchilov et al. (2024), [*nGPT: Normalized Transformer with Representation Learning on the Hypersphere*](https://arxiv.org/abs/2410.01131) (normalizes all vectors to unit norm; QK-norm as a special case; 4–20× faster convergence).
- Meng et al. (2025), [*NaLaFormer: Norm-Aware Linear Attention*](https://arxiv.org/abs/2506.21137) (query norm controls attention spikiness; restores it in linear attention — a counterpoint on why magnitude matters).
- Jay Alammar, [*The Illustrated Transformer*](https://jalammar.github.io/illustrated-transformer/) (the standard visual walkthrough of multi-head attention and data flow).
- Lilian Weng (2018), [*Attention? Attention!*](https://lilianweng.github.io/posts/2018-06-24-attention/) (a broad technical survey of attention mechanisms and their evolution).
- Sebastian Raschka, [*Understanding Multi-Head, Multi-Query, and Grouped-Query Attention*](https://magazine.sebastianraschka.com/) (a clear, code-oriented explainer of the three variants).
- Companion note: [Why Hardware Makes Matrix Multiply Fast — GPUs, TPUs, and FlashAttention](../gpu-tpu-matmul-flashattention.md) (Part 7 covers the KV cache, prefill vs decode, and why decode is memory-bandwidth-bound — the motivation for MQA/GQA).
