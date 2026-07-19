# Efficient Attention: Escaping the Quadratic Curse

The attention layer we built in [005 — Attention](./005_Attention.md) has one fatal flaw hiding in plain sight: the $n \times n$ score matrix $S = QK^\top$. For a sequence of $n$ tokens, that matrix has $n^2$ entries, so both the compute ($O(n^2 d)$) and, in a naive implementation, the memory ($O(n^2)$) grow **quadratically** with sequence length. Double the context and attention gets four times more expensive; go from 2K tokens to 128K and the cost balloons by a factor of over four thousand. This is *the* reason long context was hard for years, and the reason a whole subfield exists to make attention cheaper.

This note is a map of that subfield. The organizing idea is simple and worth stating up front, because every method below is a variation on it: **the $n \times n$ matrix is the enemy, and there are only a few fundamentally different ways to avoid paying for it.** You can *sparsify* it (compute only a chosen subset of the entries), *low-rank* it (approximate the whole matrix with a much thinner one), *kernelize it away* (remove the softmax so you can re-order the matrix multiplies and never form $n \times n$ at all), or *hash/cluster around it* (only let similar tokens attend to each other). A fifth, orthogonal route — **FlashAttention** — keeps the math *exactly* the same and attacks only the memory cost; we already covered it in detail and will just place it on the map.

We give each family depth in proportion to how much it actually matters in practice. Linear/kernelized attention and sparse attention are load-bearing ideas you should understand well, so they get full treatment. Low-rank methods get a solid section. Hashing (Reformer) gets a medium one. Landmark/memory methods get a brief skim — they are clever but far less influential today, and the note flags them for expansion on request. We close with an honest "so which one won?" that explains why, despite all this ingenuity, production LLMs mostly went a different way.

---

## Table of Contents

- [Setup and Notation](#setup-and-notation)
- [The landscape: five ways to dodge the n×n matrix](#the-landscape-five-ways-to-dodge-the-nn-matrix)
- [Sparse attention](#sparse-attention)
- [Low-rank attention](#low-rank-attention)
- [Linear and kernelized attention](#linear-and-kernelized-attention)
- [Hashing and clustering: Reformer](#hashing-and-clustering-reformer)
- [Landmark and memory-based attention (skim)](#landmark-and-memory-based-attention-skim)
- [So which one actually won?](#so-which-one-actually-won)
- [Takeaways](#takeaways)
- [Sources](#sources)

---

## Setup and Notation

This note builds directly on [005 — Attention](./005_Attention.md); the core symbols ($n, d_{model}, Q, K, V, S, A, h$) are defined there and reused here without repeating their definitions. In particular, recall the one equation everything hangs on:

$$\mathrm{Attention}(Q, K, V) = \mathrm{softmax}\!\left(\frac{Q K^\top}{\sqrt{d_k}}\right) V, \qquad S = QK^\top \in \mathbb{R}^{n \times n}.$$

To keep the complexity discussion clean, we write $d$ for the per-head (or model) width — think of it as $d_k$, the head dimension — and treat it as a **fixed, modest constant** (say 64–128), while $n$ is the thing that grows into the thousands or millions. That single asymmetry, $d \ll n$, is what makes "trade an $n$ for a $d$" the winning move throughout. A few new symbols appear only in this note:

| Symbol | Meaning |
| --- | --- |
| $\phi(\cdot)$ | A **feature map**: a function applied to a query or key vector so that $\phi(q)\cdot\phi(k)$ stands in for the attention similarity. The heart of linear/kernelized attention. |
| $k$ (as a rank) | The **projected length** in low-rank methods — the softmax matrix's $n$ dimension is squeezed down to a small fixed $k \ll n$. (Distinct from a key vector $k_j$; context disambiguates.) |
| $w$ | The **window size** in sparse attention — each token attends to $w$ neighbours on each side. |
| $m$ | Size of an **external memory** in memory-based methods (number of stored $(k,v)$ pairs). |

Throughout, "cost" means the dominant term as $n \to \infty$; constants and the fixed $d$ are dropped in the big-$O$ but noted in prose when they matter in practice.

---

## The landscape: five ways to dodge the n×n matrix

Before diving in, here is the whole territory on one page. The two survey papers this note leans on — *Efficient Transformers: A Survey* (Tay et al., 2020) and *A Survey of Transformers* (Lin et al., 2021) — organize the zoo of "X-former" models into essentially the families below, and it is genuinely helpful to hold the map in your head before touching any single method, because most papers are one specific choice within one of these boxes.

![A landscape diagram. At the top, a grey card "Dense (vanilla) attention" labelled O(n^2 d) compute, O(n^2) memory — "form the full n×n softmax matrix, Transformer (Vaswani 2017)" — with an arrow and the caption "how do we avoid it?" pointing down to five coloured cards. Card 1 "Sparse": compute only a chosen subset of entries, O(n·sqrt(n)) to O(n), examples Sparse Transformer, Longformer, BigBird. Card 2 "Low-rank": the softmax matrix is approximately low-rank, project n→k, O(nk), examples Linformer, Nyströmformer. Card 3 "Linear / kernel": drop softmax, use φ(q)·φ(k), reassociate, O(n d^2), examples Transformers-as-RNN, Performer (FAVOR+). Card 4 "Hashing / cluster": attend only within buckets of similar tokens, O(n log n), examples Reformer, Routing Transformer. Card 5 "Memory": look up a large external store of past (k,v) pairs, O(n·m), example Memorizing Transformer. Below, a dashed blue box "Orthogonal escape route: FlashAttention (exact, no approximation) — same math as dense, still O(n^2) compute but O(n) memory by never writing the n×n matrix to HBM."](../assets/eff_attn_landscape.jpg)

Read the map top to bottom. The grey card is the problem: vanilla attention forms the full matrix and pays $O(n^2)$. The five coloured cards are the five genuinely distinct escapes, and each later section is one card:

- **Sparse** — don't compute every entry of $S$; decide *in advance* (or cheaply) that most query–key pairs are irrelevant and skip them. You keep exact softmax, but only over a chosen subset.
- **Low-rank** — keep dense attention conceptually, but exploit that the softmax matrix is *approximately low-rank*, so you can replace the length-$n$ key/value axis with a compressed length-$k$ one.
- **Linear / kernel** — the deepest trick: replace $\exp(q\cdot k)$ with a factored $\phi(q)\cdot\phi(k)$. Because it factors, matrix-multiply **associativity** lets you contract over $n$ first and never build $n\times n$ at all.
- **Hashing / clustering** — a *learned or data-dependent* form of sparsity: group tokens that are likely to attend strongly (by hashing or clustering), and only attend within a group.
- **Memory** — step outside the current sequence entirely and retrieve relevant $(k,v)$ pairs from a large external store, so effective context far exceeds $n$.

And the dashed blue box is the one that behaves differently from all of them: **FlashAttention** does not approximate anything. It computes the *exact* same attention, and its win is purely in memory traffic — it never materializes the $n \times n$ matrix in slow HBM. It is still $O(n^2)$ in compute but $O(n)$ in memory, which is why it slots *alongside* rather than *inside* the taxonomy. We covered it end to end in the companion note [Why Hardware Makes Matrix Multiply Fast, Part 6](../gpu-tpu-matmul-flashattention.md#part-6--flashattention-same-math-far-less-data-movement), and its cousin the KV cache in Part 7, so here we simply note *where it sits* and move on to the approximate families it doesn't cover.

One honest caveat the surveys stress and we will return to at the end: a better big-$O$ is not the same as a faster or better model. Many of these methods win asymptotically but lose on wall-clock time at practical lengths (their constants are large), or on quality (their approximation costs accuracy). Keep that tension in mind — it is the punchline.

---

## Sparse attention

The most intuitive escape is also the oldest: **most of the $n \times n$ attention matrix is close to useless anyway.** In a long document, a token rarely needs a precise, learned interaction with a token ten thousand positions away; it mostly cares about nearby tokens, plus a handful of special "anchor" positions. So why compute and store all $n^2$ scores? Sparse attention picks a **fixed pattern** of allowed (query, key) pairs — a sparsity mask over $S$ — and computes softmax only over those. If each query is allowed to see $O(\sqrt{n})$ or $O(1)$ keys instead of all $n$, the cost drops from $O(n^2)$ toward $O(n\sqrt{n})$ or even $O(n)$.

The whole design question becomes: **which pattern?** A good pattern must (a) be cheap — ideally a fixed, structured shape a GPU can exploit — and (b) still let information flow between any two tokens, possibly in a couple of hops, so the model doesn't lose the ability to relate distant tokens entirely. The three landmark papers are three increasingly clever answers.

![Five small square attention-mask grids side by side, each with rows = queries and columns = keys, a shaded cell meaning "this query may attend to this key." Panel 1 "Full O(n^2)": every cell shaded (100% of entries). Panel 2 "Sliding window": a shaded diagonal band (27%). Panel 3 "+ Dilated / strided": the band plus regularly spaced off-diagonal stripes (49%). Panel 4 "Longformer (window + global)": the band plus the first two full rows and first two full columns shaded (34%). Panel 5 "BigBird (window + global + random)": the band, a global row/column, plus scattered random cells (33%).](../assets/eff_attn_sparse_patterns.jpg)

**Sparse Transformer** (Child et al., 2019) is the first to make this work at scale, on images and audio. Its key move is to **factorize** the full attention into two (or more) cheaper attention patterns that *compose* to cover everything. Concretely, one head attends to a local window (the recent past), and another attends to a *strided* set of positions (every $\sqrt{n}$-th token). Neither head alone can connect two arbitrary tokens, but *together, in two hops*, any token can reach any other: token $i$ reaches a nearby "hub" via the strided head, and the hub reaches token $j$. This is exactly why the factorization is legal — you have not severed the graph, only made most edges implicit. The cost falls to $O(n\sqrt{n})$, which let them train on sequences of tens of thousands of steps that dense attention could not touch.

**Longformer** (Beltagy et al., 2020) makes the pattern practical for NLP and, crucially, **linear** in $n$. It combines three ingredients, each with a clear job:

- A **sliding window** of width $w$: token $i$ attends to the $w$ tokens on each side. This captures the local context that dominates language, and costs $O(n \cdot w)$ — linear, since $w$ is a small constant.
- **Dilated windows** (optional, borrowed from dilated convolutions): leave gaps in the window so the same number of attended positions covers a wider *receptive field*, letting information propagate further per layer without more compute.
- A few **global tokens**: a handful of special positions (e.g. the `[CLS]` token, or every question token in QA) that attend to *everything* and are attended to *by* everything. These are the long-range highways — any two ordinary tokens can communicate in two hops through a global token.

That combination — cheap local window for the common case, plus a few global highways for long-range needs — is the template almost every practical sparse scheme has used since, including the **sliding-window attention** in Mistral 7B. It is linear, hardware-friendly, and empirically strong on long documents.

**BigBird** (Zaheer et al., 2020) adds one more ingredient and, importantly, a *theoretical justification*. Its pattern is **window + global + random**: on top of Longformer's local window and global tokens, each query also attends to a small number of **randomly chosen** keys. Why random links? Because they turn the attention pattern into an *expander-like graph* — a sparse graph where any two nodes are connected by a short path with high probability. This matters because the authors prove that BigBird's sparse attention is a **universal approximator of sequence functions** and is **Turing-complete**, i.e. it retains the full expressive power of dense attention despite touching only $O(n)$ pairs. That is the reassuring result the whole sparse program needs: you can drop the vast majority of the matrix *without* provably crippling the model. BigBird pushed practical context lengths up 8× and set records on long-document QA and genomics.

The common thread: sparse attention keeps softmax **exact** on the pairs it does compute, and the art is entirely in choosing a pattern that is cheap yet keeps the whole sequence *connected* in a hop or two. The weakness is that the pattern is (mostly) **fixed and content-independent** — it cannot adapt to a document where the important long-range link happens to fall outside the chosen shape. That limitation is exactly what the hashing/clustering methods later try to fix by making the sparsity *data-dependent*.

---

## Low-rank attention

Sparse methods bet that most entries of $S$ are *unnecessary*. Low-rank methods make a different, complementary bet: that the softmax attention matrix $A$, while dense, is **approximately low-rank** — its $n \times n$ worth of numbers really only contains about $k \ll n$ dimensions of information. If that is true, you should be able to represent attention with something much thinner than $n \times n$ and lose almost nothing.

**Linformer** (Wang et al., 2020) is the clean realization of this idea, and it comes with empirical evidence: the authors show that the softmax matrix from a *trained* Transformer has a rapidly decaying spectrum — a handful of singular values carry almost all the mass — so it is well approximated by a low-rank matrix. Their fix is delightfully simple: **project the key and value matrices down along the length axis before attending.** Introduce two learned projection matrices $E, F \in \mathbb{R}^{k \times n}$ and compute $EK$ and $FV$, each of shape $k \times d$ — the *sequence length has been squeezed from $n$ to a small fixed $k$*, while the feature width $d$ is untouched.

![A two-stage diagram. Top: a tall K block (n×d) with an arrow to a short EK block (k×d) labelled "project length n→k", and likewise a tall V block (n×d) to a short FV block (k×d); a note says "E, F ∈ R^{k×n}, learned, shared projections." Bottom: the attention itself — Q (n×d) times (EK)^T (d×k) equals a scores block (n×k) labelled "small! not n×n", times FV (k×d) equals out (n×d); the whole thing tagged O(nk), linear in n for fixed k.](../assets/eff_attn_lowrank.jpg)

Now trace the shapes in the lower half of the figure to see why it works. The scores become $Q(EK)^\top$, of shape $n \times k$ instead of $n \times n$ — the expensive dimension is gone. Softmax runs over this thin $n \times k$ matrix, and multiplying by $FV$ (shape $k \times d$) gives the usual $n \times d$ output. Every step is $O(nk)$, so with $k$ a fixed constant the whole layer is **linear in $n$**. The one modeling assumption doing the work is that a *single shared* set of $k$ projected positions can summarize the keys and values for *every* query — which is exactly the low-rank hypothesis, and which is why Linformer is very strong when it holds (fixed-length, in-distribution sequences) and shakier when it doesn't (it needs $n$ baked in via $E, F$, so it is awkward for autoregressive generation or variable lengths).

**Nyströmformer** (Xiong et al., 2021) reaches the same $O(n)$ target but sidesteps Linformer's need to fix $n$ in the projection weights. It borrows the classical **Nyström method** for approximating a large matrix from a small subset of its rows and columns. The idea in one breath: instead of the full softmax matrix, pick a modest number of representative "landmark" tokens (e.g. by averaging tokens into $k$ groups), compute attention *between all tokens and the landmarks* and *among the landmarks*, and stitch these small pieces back into an approximation of the full $A$. Because it approximates the softmax matrix directly rather than assuming a fixed low-rank projection of $K,V$, it handles variable lengths more gracefully. It is a solid, widely-cited method, but conceptually a refinement of the same "summarize $n$ into $k$ landmarks" theme, so we leave it at the intuition.

The honest limitation of the whole low-rank family: the low-rank assumption is empirical, not guaranteed. On tasks where attention genuinely needs sharp, near-one-hot, high-rank patterns (precise copying, exact retrieval from a long context), squashing the matrix to rank $k$ throws away exactly the information that mattered.

---

## Linear and kernelized attention

This is the conceptual centerpiece of the note, and the most elegant idea in the whole area. It does not sparsify the matrix or approximate it as low-rank — it **removes the reason the matrix exists in the first place**, and by doing so makes attention truly linear in $n$ with a *constant-size* recurrent state that even makes autoregressive decoding cheap.

### Why softmax is the villain

Look again at $\mathrm{softmax}(QK^\top)V$ and ask: *why must we build the $n \times n$ matrix at all?* Matrix multiplication is associative, so if attention were just $QK^\top V$ we could compute it in the order $Q(K^\top V)$ — and $K^\top V$ is only $d \times d$! We would never form anything of size $n \times n$. The **only** thing stopping us is the softmax sitting between $QK^\top$ and $V$: it is a nonlinear, row-wise function that needs the *entire row* of scores (to exponentiate and normalize), so you are forced to materialize each full row of $QK^\top$ before you can apply it. Softmax is precisely the glue that welds the two matmuls into a fixed, non-reorderable order. Remove it and associativity is free.

### The kernel trick: factor the similarity

The similarity between a query and key under softmax is $\exp(q \cdot k)$ — and the problem is that $\exp$ of a dot product does **not** factor into (something about $q$) times (something about $k$). Linear attention's move is to *replace* that similarity with one that **does** factor:

$$\exp(q \cdot k) \;\longrightarrow\; \phi(q) \cdot \phi(k),$$

where $\phi$ is a **feature map** applied separately to each query and each key. The point of insisting on this form is that once similarity is an inner product of separately-transformed vectors, the attention output for query $i$ becomes

$$\text{out}_i \;=\; \frac{\sum_j \big(\phi(q_i)\cdot\phi(k_j)\big)\, v_j}{\sum_j \phi(q_i)\cdot\phi(k_j)} \;=\; \frac{\phi(q_i)^\top \left(\sum_j \phi(k_j)\, v_j^\top\right)}{\phi(q_i)^\top \left(\sum_j \phi(k_j)\right)}.$$

Read the right-hand side carefully, because the entire speedup lives in that rearrangement. In the middle expression each output is a sum over all $j$ of a scalar similarity times a value — the usual $O(n)$-per-query, $O(n^2)$-total attention. But because the similarity factors, we can **pull $\phi(q_i)$ out of the sum over $j$**: the quantity $\sum_j \phi(k_j)\, v_j^\top$ does not depend on $i$ at all. So we compute that once — it is a single $d \times d$ matrix (an outer-product sum over all keys and values) — and then every query just multiplies its $\phi(q_i)$ against this shared matrix. The denominator is the same story with a $d$-vector $\sum_j \phi(k_j)$ for normalization.

![Two rows of coloured matrix blocks. Top row, labelled "Softmax forces this order: first QK^T (the n×n matrix), then ×V": Q (n×d) times K^T (d×n) equals a large red QK^T block (n×n) marked "BIG: grows as n^2", times V (n×d) equals out (n×d); tagged O(n^2 d). Bottom row, labelled "Kernel φ removes softmax ⇒ reassociate: first φ(K)^T V (a small d×d matrix), then φ(Q)× it": φ(Q) (n×d) times φ(K)^T (d×n) times V (n×d), with a bracket showing φ(K)^T·V is computed first to give a small d×d block marked "SMALL: no n!", then φ(Q) times it equals out (n×d); tagged O(n d^2). Caption: "Same three matrices — only the multiplication order changes. Associativity turns the n×n bottleneck into a d×d one, and d ≪ n."](../assets/eff_attn_linear_assoc.jpg)

The figure is the whole idea in one glance. Top row: softmax forces you to build the red $n \times n$ block first — $O(n^2 d)$. Bottom row: with the factored kernel you instead compute $\phi(K)^\top V$ first, which contracts over the length axis $n$ immediately and produces a tiny $d \times d$ matrix that has no $n$ in it at all; then multiplying by $\phi(Q)$ gives the output in $O(n d^2)$. Since $d$ is a fixed constant and $n$ grows, $O(nd^2)$ is **linear in sequence length**. Same three matrices, same associative multiplication — only the *order* changed, and the order was only forbidden by the softmax we removed.

### "Transformers are RNNs": the recurrent view

Katharopoulos et al. (2020), *Transformers are RNNs*, introduced this linear attention and drew out a beautiful consequence for **autoregressive** generation. In a causal model, query $i$ may only attend to keys $j \le i$. Under linear attention that means the shared matrix is a *running prefix sum*: define the state $S_i = \sum_{j \le i} \phi(k_j) v_j^\top$ (a $d \times d$ matrix) and $z_i = \sum_{j\le i}\phi(k_j)$ (a $d$-vector). Then

$$\text{out}_i = \frac{\phi(q_i)^\top S_i}{\phi(q_i)^\top z_i}, \qquad S_i = S_{i-1} + \phi(k_i)\, v_i^\top.$$

This is *exactly* an RNN: a **fixed-size hidden state** $(S_i, z_i)$ that you update with a simple additive rule as each token arrives, then read out with the current query. The consequences are striking. Generating a token costs $O(1)$ regardless of how much history precedes it, and the memory is $O(1)$ too — you keep only the $d \times d$ state, **not** a KV cache that grows with sequence length. Contrast this with standard attention's decode, which must stream the entire growing KV cache every step (the memory-bandwidth bottleneck from [the hardware note, Part 7](../gpu-tpu-matmul-flashattention.md#part-7--the-kv-cache-prefill-decode-and-why-inference-is-memory-bound)). Linear attention turns the quadratic-time, linear-memory decoder into a linear-time, constant-memory one. This equivalence — attention with a factored kernel *is* a linear RNN — is the seed of the entire modern "linear-attention / state-space / RNN-revival" line of work.

### Performer and FAVOR+: approximating softmax, not replacing it

The catch with the simple approach is the feature map $\phi$. Early choices (e.g. $\phi(x) = \mathrm{elu}(x)+1$) make similarity factor but they do **not** reproduce softmax's actual behavior — you get *a* linear attention, just not one that matches the model softmax would have learned, and quality often suffers.

**Performer** (Choromanski et al., 2020) fixes this properly with a mechanism called **FAVOR+** (Fast Attention Via positive Orthogonal Random features). The insight is that the softmax/exponential kernel can be written *exactly* as an expectation over random features: there exists a randomized $\phi$ (built from random projections followed by an exponential nonlinearity) such that $\mathbb{E}[\phi(q)\cdot\phi(k)] = \exp(q\cdot k)$. So instead of *inventing* a factored similarity and hoping it behaves like softmax, Performer *approximates the real softmax kernel* with a factored one, **unbiasedly**, using $r$ random features. The "+" refinements — using *positive* random features (so the approximated attention weights stay positive and the estimator is far more stable than the naive trigonometric version) and making the random projections *orthogonal* (which sharply reduces variance) — are what make it actually work in practice. You get provable, controllable approximation of true softmax attention at $O(n r d)$ cost — linear in $n$ — with a knob $r$ trading accuracy for speed.

A brief pointer for the curious: Schlag et al. (2021), *Linear Transformers Are Secretly Fast Weight Programmers*, reinterprets the $d\times d$ state $S_i$ as an **associative memory** whose "weights" are rewritten by each token via the outer-product update $\phi(k_i)v_i^\top$ — connecting linear attention to the older *fast weight* literature and explaining a known weakness (the additive update never *forgets*, so the memory can saturate), which later gated/delta variants address. Worth knowing the framing exists; the details are a rabbit hole we won't descend here.

The linear/kernel family's promise is the best of any approach on paper: genuinely linear time, constant-memory recurrent decoding. Its historical weakness has been **quality** — on hard tasks that need sharp, selective attention, the finite-rank kernel approximation blurs exactly the crisp patterns that matter — which is why for years it lost to exact attention. That gap has narrowed sharply with modern gated linear-attention and state-space models, but that is a story for a later note.

---

## Hashing and clustering: Reformer

Sparse attention used a *fixed, content-independent* pattern. The obvious improvement is to make the sparsity **data-dependent**: instead of hard-coding which tokens can interact, *discover* which tokens are likely to attend strongly and only compute those. The difficulty is that finding, for each query, the keys with the largest $q\cdot k$ seems to require computing all $n^2$ scores — the very thing we are trying to avoid. Hashing breaks that circularity.

**Reformer** (Kitaev et al., 2020) uses **Locality-Sensitive Hashing (LSH)**. An LSH is a hash function with a special property: vectors that are *close* (high dot product) are likely to land in the *same* bucket, while distant ones usually don't. Since softmax is dominated by the few largest scores anyway (it is exponential — the biggest $q\cdot k$ swamps the rest), Reformer's bet is that **a query only really needs the keys in its own bucket.** So: hash all queries and keys, sort tokens by bucket, and compute exact attention *only within each bucket* (plus a small overlap to neighbouring chunks). With buckets of bounded size this costs $O(n \log n)$ instead of $O(n^2)$.

![Two panels. Left, "LSH hashes nearby q/k into the same bucket (random hyperplanes = the hash)": a scatter of token points in 2-D falling into three coloured clusters (bucket 1, 2, 3), with two dashed lines depicting the random hyperplanes that define the hash. Right, "Attend only within your bucket ⇒ block-diagonal, O(n log n)": an attention matrix, with tokens sorted by bucket, that is block-diagonal — three coloured blocks on the diagonal, everything off-diagonal white (not computed).](../assets/eff_attn_reformer_lsh.jpg)

The figure shows the mechanism: hashing partitions tokens into buckets (left), and once you sort tokens by bucket the attention you actually compute is **block-diagonal** (right) — each token attends only within its coloured block, and the entire off-diagonal is skipped. Reformer pairs this with a second, orthogonal memory trick — **reversible residual layers**, which let you recompute each layer's activations during backprop instead of storing them, cutting activation memory so a very deep model fits on one accelerator. LSH attention was an influential proof that *learned/data-dependent* sparsity is possible, though in practice the hashing overhead and the need for multiple hash rounds (to avoid missing a relevant key that happened to hash elsewhere) make it finicky, and it has been largely superseded by exact-attention approaches.

**Routing Transformer** (Roy et al., 2020) is the natural sibling: instead of hashing, it runs **online $k$-means clustering** on the queries and keys, and lets each query attend only to keys in the same learned cluster. Same principle as Reformer — group similar tokens, attend within groups — but the grouping is *learned by clustering* rather than fixed by a random hash, which can make the buckets more semantically coherent. It also lands around $O(n\sqrt{n})$. It is a clever, moderately-cited method; we note the idea and move on.

---

## Landmark and memory-based attention (skim)

The last family steps outside the current sequence altogether. Rather than making attention *within* the $n$ tokens cheaper, it lets the model reach a **large external memory** of information from far outside the current window — so the *effective* context can be enormous even though attention over the live sequence stays cheap. These methods are clever but, as of today, far less central to mainstream LLMs than the families above, so we keep this to the intuition and flag it for expansion if you want more.

**Memorizing Transformer** (Wu et al., 2022) augments one attention layer with a giant **non-differentiable external memory** of past $(k, v)$ pairs — potentially millions of them, accumulated across a long document or corpus. At each step the model does a **$k$-nearest-neighbour** lookup into this store to retrieve the most relevant past keys/values and attends over them in addition to the local context. Because the memory is a frozen kNN index rather than something you backprop through, it can be huge and cheap to grow, effectively giving the model a read-only long-term memory that extends context far beyond what dense attention could afford. The intuition to keep: *retrieval as attention* — the same soft-lookup mechanism, but the "keys" now live in an external database instead of the current sequence. (This idea is a close relative of retrieval-augmented generation, and of the "landmark" tokens that summarize chunks of far-away context in other schemes.)

If you want these fleshed out — the memory-update mechanics, staleness of stored keys, how kNN attention is interleaved with normal layers — say so and I'll expand this section.

---

## So which one actually won?

Here is the honest, slightly deflating punchline the surveys hint at and practice confirmed: **for mainstream LLMs, mostly none of the approximate methods above won.** Despite years of clever sparse, low-rank, kernel, and hashing schemes with better big-$O$, the dominant recipe in production models (Llama, Mistral, GPT-class) is *exact* attention made fast and memory-light — **FlashAttention** for the compute, the **KV cache** with **GQA/MQA** for decode — often with a light dusting of *sliding-window* sparsity (Mistral) as the one approximate idea that reliably pays off.

Why did exactness win? Three reasons worth internalizing, because they generalize:

- **Quality is unforgiving.** Most approximations blur precisely the sharp, selective attention patterns that hard tasks (exact retrieval, copying, in-context lookup) depend on. A method that is 3× faster but a few points worse on a benchmark is usually a bad trade for a frontier model.
- **Big-$O$ hides the constants.** At the sequence lengths people actually ran for years (a few thousand tokens), a linear method with a large constant and poor hardware utilization is often *slower in wall-clock time* than well-optimized quadratic attention. Asymptotics only pay off past a crossover length many models never reached.
- **FlashAttention moved the goalposts.** Once exact attention became memory-linear and several times faster *without any quality loss* (see [the hardware note, Part 6](../gpu-tpu-matmul-flashattention.md#part-6--flashattention-same-math-far-less-data-movement)), the bar for an approximate method rose brutally: you now have to beat a highly-optimized *exact* baseline, and pay for it in accuracy, to justify the added complexity. Most couldn't.

This does not make the note academic. First, these ideas are the intellectual foundation of what *is* now ascendant: the linear-attention / RNN view (Katharopoulos) directly seeded modern **gated linear attention and state-space models** (Mamba and friends), which are finally competitive on quality — a topic for a later note. Second, as context lengths push toward the millions, the quadratic term reasserts itself and sparse/linear ideas keep returning in new clothes. The right takeaway is not "approximations lost" but "**the bar is exact attention, made hardware-efficient — beat that on quality, not just on FLOPs, or don't bother.**"

---

## Takeaways

- **The $n\times n$ score matrix is the whole problem.** Every efficient-attention method is one of five moves against it: sparsify it, low-rank it, kernelize it away, hash/cluster around it, or (orthogonally) keep it exact but never write it to slow memory (FlashAttention).
- **Sparse attention** keeps softmax exact on a chosen subset of pairs. The winning recipe — local **window** + a few **global** tokens (+ **random** links in BigBird, which is provably still a universal approximator) — keeps the sequence connected in a hop or two at $O(n)$ cost. Sliding-window attention survives in production models.
- **Low-rank attention** (Linformer) exploits that the softmax matrix is empirically near-low-rank, projecting the length-$n$ key/value axis down to a fixed $k$ for $O(nk)$. It shines on fixed-length in-distribution data and struggles when attention genuinely needs to be high-rank.
- **Linear/kernel attention** is the deepest idea: replace $\exp(q\cdot k)$ with a factored $\phi(q)\cdot\phi(k)$ so associativity lets you contract over $n$ first ($O(nd^2)$, linear). It reveals attention as a **linear RNN** with a constant-size state — $O(1)$-memory decoding — and **Performer/FAVOR+** approximates true softmax unbiasedly with random features.
- **Hashing/clustering** (Reformer's LSH, Routing Transformer's $k$-means) makes sparsity *data-dependent*: group similar tokens and attend only within a group, $O(n\log n)$ to $O(n\sqrt n)$.
- **Memory-based** methods (Memorizing Transformer) retrieve from a huge external $(k,v)$ store via kNN — "retrieval as attention" — to extend effective context far beyond $n$.
- **Exact, hardware-efficient attention mostly won.** FlashAttention + KV-cache + GQA + light sliding-window is the production default, because approximations tend to cost quality and their asymptotic wins hide large constants. The approximate ideas live on chiefly as the ancestors of modern linear-attention/state-space models.

---

## Sources

**Surveys (the map):**
- Tay, Dehghani, Bahri, Metzler (2020), [*Efficient Transformers: A Survey*](https://arxiv.org/abs/2009.06732) — the taxonomy this note follows.
- Lin, Wang, Liu, Qiu (2021), [*A Survey of Transformers*](https://arxiv.org/abs/2106.04554) — a broader survey with a complementary categorization.

**Sparse attention:**
- Child, Gray, Radford, Sutskever (2019), [*Generating Long Sequences with Sparse Transformers*](https://arxiv.org/abs/1904.10509) — factorized strided/fixed patterns, $O(n\sqrt n)$.
- Beltagy, Peters, Cohan (2020), [*Longformer: The Long-Document Transformer*](https://arxiv.org/abs/2004.05150) — sliding window + dilation + global tokens, linear.
- Zaheer et al. (2020), [*Big Bird: Transformers for Longer Sequences*](https://arxiv.org/abs/2007.14062) — window + global + random; universal-approximator / Turing-complete proof.

**Low-rank attention:**
- Wang, Li, Khabsa, Fang, Ma (2020), [*Linformer: Self-Attention with Linear Complexity*](https://arxiv.org/abs/2006.04768) — project $K,V$ from $n$ to $k$; the low-rank spectrum evidence.
- Xiong et al. (2021), [*Nyströmformer: A Nyström-Based Algorithm for Approximating Self-Attention*](https://arxiv.org/abs/2102.03902) — Nyström landmark approximation of softmax.

**Linear / kernelized attention:**
- Katharopoulos, Vyas, Pappas, Fleuret (2020), [*Transformers are RNNs: Fast Autoregressive Transformers with Linear Attention*](https://arxiv.org/abs/2006.16236) — the factored-kernel reassociation and the linear-RNN view.
- Choromanski et al. (2020), [*Rethinking Attention with Performers*](https://arxiv.org/abs/2009.14794) — FAVOR+: unbiased softmax approximation via positive orthogonal random features.
- Schlag, Irie, Schmidhuber (2021), [*Linear Transformers Are Secretly Fast Weight Programmers*](https://arxiv.org/abs/2102.11174) — the associative-memory / fast-weight reinterpretation.

**Hashing / clustering:**
- Kitaev, Kaiser, Levskaya (2020), [*Reformer: The Efficient Transformer*](https://arxiv.org/abs/2001.04451) — LSH attention + reversible layers.
- Roy, Saffar, Vaswani, Grangier (2020), [*Efficient Content-Based Sparse Attention with Routing Transformers*](https://arxiv.org/abs/2003.05997) — online $k$-means clustering of tokens.

**Memory-based:**
- Wu, Rabe, Hutchins, Szegedy (2022), [*Memorizing Transformers*](https://arxiv.org/abs/2203.08913) — kNN lookup into a large non-differentiable external memory.

**Blogs:**
- [Hazy Research (Stanford) blog](https://hazyresearch.stanford.edu/blog) — the group behind FlashAttention; excellent posts on efficient/long-context attention and state-space models.
- [main-horse, *Transformer Upgrade* (translation)](https://main-horse.github.io/translations/transformer-upgrade/) — a clear walkthrough of linear-attention variants.

**Companion notes in this repo:**
- [005 — Attention](./005_Attention.md) — builds standard attention, multi-head, MQA/GQA (this note's prerequisite).
- [Why Hardware Makes Matrix Multiply Fast](../gpu-tpu-matmul-flashattention.md) — **FlashAttention (Part 6)** and the **KV cache / prefill vs decode (Part 7)**, the exact-attention escape route referenced throughout.
