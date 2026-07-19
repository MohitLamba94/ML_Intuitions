# Position Embeddings: Teaching a Transformer Where Each Token Is

Self-attention has a strange blind spot: **it does not know the order of its inputs.** If you shuffle the tokens of a sentence, the attention mechanism computes the *same* set of pairwise interactions — just permuted. The math treats a sequence as an unordered *bag* of vectors. But "the cat sat on the mat" and "the mat sat on the cat" mean different things, and word order is most of what makes language language. So something has to inject the notion of *position* back into the model. That something is the **position embedding (PE)**.

This note builds up the idea from scratch. We start with *why* attention is order-blind and what "injecting position" even means. Then we walk the two big design axes: **absolute vs. relative** position (do we tell a token its slot number, or tell each pair how far apart they are?) and **learnable vs. fixed** (do we learn the position codes from data, or compute them from a formula?). We explain the classic **sinusoidal** encoding and its intimidating-looking formula term by term. Finally we spend most of our time on **RoPE (Rotary Position Embedding)** — the method that quietly won, why it combines the best of absolute and relative, how it works in 1D, and how it generalizes to the multi-axis (image/video) and long-context settings used in modern generative models.

---

## Table of Contents

- [Setup and Notation](#setup-and-notation)
- [Why attention is order-blind](#why-attention-is-order-blind)
- [Absolute vs. relative position](#absolute-vs-relative-position)
- [Learnable vs. fixed position embeddings](#learnable-vs-fixed-position-embeddings)
- [Sinusoidal position encoding](#sinusoidal-position-encoding)
- [Rotary Position Embeddings (RoPE)](#rotary-position-embeddings-rope)
- [Takeaways](#takeaways)
- [Sources](#sources)

---

## Setup and Notation

A handful of symbols recur throughout; each is reintroduced where it first does real work.

| Symbol | Meaning |
|---|---|
| $d$ | The model's hidden dimension — the length of each token's vector (e.g. 128, 4096). |
| $m, n$ | Integer **positions** of two tokens in the sequence (e.g. the query is at position $m$, a key at position $n$). |
| $x_m$ | The token embedding at position $m$: a $d$-dimensional vector, *before* any position information is added. |
| $q_m, k_n$ | The **query** vector at position $m$ and the **key** vector at position $n$ (obtained by linear projections of the token embeddings). Attention scores come from dot products $q_m \cdot k_n$. |
| $PE_m$ | The position encoding for position $m$: a $d$-dimensional vector (sinusoidal case). |
| $\theta_i$ | A per-dimension-pair **rotation frequency** in RoPE (also written $\omega_i$). Small index $i$ → fast rotation; large $i$ → slow rotation. |
| $R(m)$ | A rotation matrix that RoPE applies to a vector, parameterized by position $m$. |

---

## Why attention is order-blind

To see the blind spot concretely, recall what attention does. Each token produces a query $q$, a key $k$, and a value $v$. The score between token $m$ and token $n$ is the dot product $q_m \cdot k_n$, these scores are softmaxed into weights, and each token's output is the weighted sum of the values. Crucially, **the queries, keys, and values are computed per token, independently of position** — token $m$'s query depends only on token $m$'s content, not on where $m$ sits. So if I take the exact same tokens and reorder them, every $q_i \cdot k_j$ pair still exists with the same value; only the *labels* move. Attention is **permutation-equivariant**: permute the inputs and the outputs permute identically, but nothing about the *content* of the result changes.

That is fatal for language, where "dog bites man" and "man bites dog" share identical tokens and differ only in order. The fix is to make each token's vector *depend on its position* before (or during) attention, so that the dot product $q_m \cdot k_n$ can "feel" where $m$ and $n$ are. There are two broad philosophies for how to encode that positional dependence — absolute and relative — and we look at both next.

---

## Absolute vs. relative position

![Two side-by-side schematics over the sentence 'The cat sat on the mat'. Left, labeled Absolute PE 'you are at slot m': each word box has a fixed position label above it (pos 0, pos 1, ... pos 5) with a small arrow pointing down into the box; caption reads 'Each token gets a fixed code for its own index.' Right, labeled Relative PE 'how far is key from query': the word 'the' at index 4 is highlighted as the query, and curved arrows point from every other word to it, each labeled with the signed offset -4, -3, -2, -1, +1; caption reads 'Only the offset (query minus key) matters, so the same rule works no matter where the pair sits in the sequence.'](../assets/pe_absolute_vs_relative.jpg)

**Absolute position** answers the question "*where am I?*" with a fixed slot number. Token 0 gets the code for position 0, token 1 the code for position 1, and so on; this code is added to (or otherwise mixed into) the token's embedding. This is simple and it works, but it has two well-known weaknesses. First, it is about *absolute* index, so the model has to *infer* relative distance — the thing that actually matters for grammar — indirectly, by comparing two absolute codes. Second, it **extrapolates poorly**: if the model only ever saw positions 0–2047 during training, position 4096 is a code it has literally never encountered, and behavior degrades.

**Relative position** answers a different, more useful question: "*how far apart are these two tokens?*" Instead of tagging each token with its own index, it makes the attention score between a query at $m$ and a key at $n$ depend on the **offset** $m - n$. The intuition is that grammar is overwhelmingly about *relative* structure — an adjective modifies the noun a couple of tokens away, a verb agrees with a subject some distance back — and that structure is the same whether the phrase appears at the start of the document or 3000 tokens in. Relative encodings bake this **translation invariance** in directly: shift the whole sentence by 10 positions and every offset $m-n$ is unchanged, so the attention pattern is unchanged. This is exactly why relative schemes generalize to longer sequences far better than absolute ones.

The catch is cost and awkwardness. Early relative schemes (e.g. Shaw et al., 2018; the T5 relative bias) inject the offset by adding learned terms *inside* the attention score computation — a bias that depends on $m-n$. That works, but it complicates the attention kernel, adds parameters or lookups indexed by distance, and doesn't compose cleanly with the highly optimized matrix multiplications that make attention fast. So for a while there was a real tension: absolute encodings are cheap and simple but generalize badly; relative encodings generalize well but are fiddly and slower.

**This is the gap RoPE closes.** RoPE is the trick that gives you *relative* behavior — the attention score depends only on the offset $m-n$ — while being applied like an *absolute* encoding (each token is transformed using only its own position $m$, once, before attention). You get translation invariance and good length generalization without any special bias terms inside the attention kernel. We build up to exactly how it pulls this off in the RoPE section.

---

## Learnable vs. fixed position embeddings

Cutting across the absolute/relative axis is a second choice: where do the actual position codes *come from*?

**Learnable (learned) PE.** Treat positions like a second vocabulary. Create a lookup table with one trainable $d$-dimensional row per position (row 0 for position 0, up to some maximum length $L$), and add the row for position $m$ to token $m$'s embedding. These rows are learned by gradient descent alongside everything else. This is what the original BERT and GPT-2 used. It is dead simple and lets the model shape the position codes however the data prefers. The downsides: it adds $L \times d$ parameters, and — more importantly — it has a **hard length ceiling**. There is simply no row for position $L+1$, so the model cannot process a sequence longer than it was built for, and even positions near the maximum are seen rarely and thus poorly trained.

**Fixed (non-learnable) PE.** Compute the position codes from a deterministic formula with no trainable parameters — the sinusoidal encoding below is the canonical example. Nothing to learn, no parameters added, and because it is just a function of $m$, you can in principle evaluate it at *any* position, including ones longer than seen in training. The trade-off is that the code is fixed by the designer's formula rather than discovered from data, so it may not be perfectly matched to the task — though in practice good fixed schemes work about as well as learned ones and generalize better.

The empirical verdict from the original Transformer paper and much follow-up work: learned and fixed absolute encodings perform *comparably* on in-distribution lengths, but fixed/relative-flavored schemes win on **length extrapolation**. Modern LLMs have largely moved to RoPE, which is fixed (no learned position parameters) *and* relative in effect — the best of both columns of this section. So the trend has been: learned-absolute (BERT/GPT-2) → fixed-relative-via-rotation (RoPE, used by Llama, GPT-NeoX, PaLM, and essentially all recent open models).

---

## Sinusoidal position encoding

The original "Attention Is All You Need" Transformer used a **fixed, absolute** encoding built from sines and cosines. Here is the formula that trips everyone up on first read:

$$
PE_{m,\,2i}   = \sin\!\left(\frac{m}{10000^{\,2i/d}}\right), \qquad
PE_{m,\,2i+1} = \cos\!\left(\frac{m}{10000^{\,2i/d}}\right)
$$

Let's decode every piece. $m$ is the token's position (which row of the encoding we want). $d$ is the hidden dimension. The index $i$ runs over **dimension pairs**: $i = 0, 1, 2, \dots, d/2 - 1$. For each pair we fill *two* entries of the $d$-dimensional code — an even slot $2i$ with a sine and the neighboring odd slot $2i+1$ with a cosine. So the encoding is really $d/2$ independent (sine, cosine) pairs stacked together.

The one genuinely mysterious term is the denominator $10000^{2i/d}$, which sets the **frequency** of each pair. Read it as follows. Define the angle fed into the sinusoids as $m \cdot \omega_i$ where the frequency is $\omega_i = 1 / 10000^{2i/d} = 10000^{-2i/d}$. When $i = 0$ the exponent is $0$, so $\omega_0 = 1$ — the fastest oscillation, cycling as position increments by 1. As $i$ grows toward $d/2$, the exponent climbs toward $1$, so $\omega_i$ shrinks toward $1/10000$ — an extremely *slow* oscillation whose wavelength is thousands of positions. In other words, **the pairs form a geometric ladder of frequencies from very fast to very slow.** The base $10000$ just controls how slow the slowest pair gets (its longest wavelength ≈ $2\pi \cdot 10000$); it is a knob chosen so that the slowest wavelength comfortably exceeds any realistic sequence length.

Why a spectrum of frequencies rather than one? Because a single frequency cannot uniquely stamp a long range of positions — a sinusoid repeats. But a *bank* of many frequencies together forms something like a binary/Fourier code for position: the fast pairs flip quickly and pin down *fine* distinctions (position 5 vs. 6), while the slow pairs change gradually and pin down *coarse* distinctions (position 5 vs. 500). Reading all $d/2$ pairs at once identifies the position uniquely, the same way a set of clock hands ticking at different rates jointly tells the exact time.

![A heatmap of the sinusoidal position encoding, with embedding dimension i from 0 to 127 on the horizontal axis and position m from 0 to about 100 on the vertical axis; color runs from blue (negative) through white (zero) to red (positive). The left side (low dimensions) shows a fine, rapidly alternating checkerboard pattern that changes quickly with position; the right side (high dimensions) shows broad vertical bands that change only slowly with position. Annotations point to the left region labeled 'fast dims (distinguish nearby positions)' and the right region labeled 'slow dims (distinguish far-apart positions)'.](../assets/pe_sinusoidal_heatmap.jpg)

The heatmap makes the frequency ladder visual: scanning down a *low-index* column (left), the color flips rapidly with position — fast oscillation; scanning down a *high-index* column (right), the color changes glacially — slow oscillation. Every position (row) therefore has a distinct fingerprint across the columns.

There is one more elegant property, and it is the reason sines and cosines were chosen specifically. For any fixed offset $k$, the encoding at position $m+k$ can be written as a **fixed linear transformation** of the encoding at position $m$ — because rotating the angle by $k\omega_i$ is a linear map on each $(\sin, \cos)$ pair (the angle-addition formulas). Concretely, $\sin(m\omega + k\omega)$ and $\cos(m\omega + k\omega)$ are linear combinations of $\sin(m\omega)$ and $\cos(m\omega)$ with coefficients that depend only on $k$, not on $m$. This means the model can, in principle, learn to attend "3 tokens back" using a single position-independent operation — a first taste of **relative** behavior emerging from an *absolute* sinusoidal code. RoPE takes this insight and makes it the whole mechanism, rather than a lucky side effect.

---

## Rotary Position Embeddings (RoPE)

RoPE (Su et al., 2021) is the encoding that essentially all recent LLMs use. Its guiding idea is the one we teased earlier: **encode position by *rotating* the query and key vectors by an angle proportional to their position, so that when you take the dot product $q_m \cdot k_n$, the two rotations combine into a dependence on only the relative offset $m - n$.** You apply it like an absolute encoding (each vector rotated using its own position), but you get a relative result for free — no bias terms, no special attention kernel.

### 7.1 RoPE in 1D: the core idea

Start with the simplest case: 1D positions, i.e. a plain text sequence where position is a single integer.

Take a token's vector of dimension $d$ (say $d = 128$):

```
token_vector = [d₀, d₁, d₂, d₃, d₄, d₅, ..., d₁₂₆, d₁₂₇]
```

RoPE groups the dimensions into consecutive **pairs** and rotates each pair independently, each with its own frequency:

- Pair 0: $(d_0, d_1)$ — rotated with frequency $\omega_0$
- Pair 1: $(d_2, d_3)$ — rotated with frequency $\omega_1$
- Pair 2: $(d_4, d_5)$ — rotated with frequency $\omega_2$
- …
- Pair 63: $(d_{126}, d_{127})$ — rotated with frequency $\omega_{63}$

Picture each pair $(d_0, d_1)$ as the coordinates of a point on a 2D plane. RoPE rotates that point by an angle that depends on the token's position $m$. For a token at position $m$ we rotate by angle $m\theta$ using the standard 2D rotation matrix:

$$
\begin{bmatrix} d_0' \\ d_1' \end{bmatrix}
=
\begin{bmatrix} \cos(m\theta) & -\sin(m\theta) \\ \sin(m\theta) & \phantom{-}\cos(m\theta) \end{bmatrix}
\begin{bmatrix} d_0 \\ d_1 \end{bmatrix}
$$

![A 2D plot showing a unit circle. A single vector representing the dimension pair (d0, d1) is drawn from the origin at four rotations, labeled m=0, m=1, m=2, m=3, each rotated an additional fixed angle theta counter-clockwise from the previous. A small arc between the m=0 and m=1 vectors is labeled theta. The axes are labeled 'dimension d0' (horizontal) and 'dimension d1' (vertical). Title: 'RoPE on one dimension pair (d0, d1): position m rotates the vector by angle m*theta'.](../assets/rope_rotation.jpg)

The figure shows one pair being rotated by successively larger angles as the position increases: at $m=0$ no rotation, at $m=1$ a rotation of $\theta$, at $m=2$ a rotation of $2\theta$, and so on. The *content* of the pair (its length) is untouched; only its *angle* encodes the position. This is the crucial difference from sinusoidal PE: sinusoidal PE **adds** a position vector to the token, whereas RoPE **rotates** the token's own query/key — it multiplies rather than adds.

**Why rotation gives relative position.** Rotations have a beautiful algebraic property. If you rotate the query at position $m$ by $m\theta$ and the key at position $n$ by $n\theta$, then their dot product depends only on the *difference* of the angles, $(m - n)\theta$. Intuitively, the dot product of two vectors depends on the angle *between* them; rotating both by their respective positions leaves the in-between angle equal to $(m-n)\theta$. So the attention score $q_m \cdot k_n$ automatically becomes a function of the offset $m - n$ — pure relative position — even though each vector was rotated using only its own absolute position. That is the whole trick, and it is why RoPE is described as "absolute rotation, relative effect."

### 7.2 Multi-frequency rotation

A single rotation angle is not enough. If every pair rotated at the same speed $\theta$, we would have exactly one frequency, which (like a single sinusoid) cannot distinguish positions across a long range. So — exactly as in sinusoidal PE — RoPE gives **each pair its own frequency**, forming a geometric ladder from fast to slow:

$$
\omega_i = \frac{1}{10000^{\,2i/d}}, \qquad i = 0, 1, \dots, \tfrac{d}{2}-1
$$

- **Fast frequencies** (pair 0, 1, …) spin a lot per position step, so they resolve *nearby* positions (5 vs. 6).
- **Slow frequencies** (the last pairs) barely move per step, so they resolve *distant* positions (5 vs. 500) without wrapping around.

![Two panels. Left panel titled 'Different dimension pairs rotate at different speeds': four smooth sine curves of sin(m*omega_i) versus position m from 0 to about 63, one for a fast pair (many oscillations), two medium pairs, and the slowest pair (a nearly flat line near zero); a legend lists each pair index and its frequency omega. Right panel titled 'Angle = position times frequency': the same four pairs plotted as accumulated angle m*omega_i (on a log scale) versus position, showing the fast pair's angle rising steeply while the slowest pair's angle stays almost flat, i.e. fast pairs sweep huge angles and slow pairs barely move.](../assets/rope_multifreq.jpg)

Concretely, for a sequence of length 512 with $d/2 = 64$ pairs, the angle applied to each pair at each position is just position × frequency, giving a table of angles:

```
[[0·ω₀,    0·ω₁,   0·ω₂,   ..., 0·ω₆₃],     # position 0
 [1·ω₀,    1·ω₁,   1·ω₂,   ..., 1·ω₆₃],     # position 1
 [2·ω₀,    2·ω₁,   2·ω₂,   ..., 2·ω₆₃],     # position 2
 ...
 [511·ω₀,  511·ω₁, 511·ω₂, ..., 511·ω₆₃]]   # position 511
```

Each *row* is one position; each *column* is one frequency. Row $m$ tells you exactly how much to rotate each of the 64 pairs for a token at position $m$. Fast columns (left) sweep through many full turns across the sequence; slow columns (right) barely budge. Together they give the model both fine and coarse positional resolution — the same Fourier-code intuition as sinusoidal PE, but delivered by rotating the query/key rather than adding a vector.

### 7.3 Extending to multiple axes: multi-axis RoPE

So far position has been a single integer — fine for text. But in **image and video generative models** a token is not in a 1D sequence; it lives in a multi-dimensional grid. A video token, for instance, has a natural 4-dimensional position:

- **Width $w$** — horizontal position of the patch in the frame (which column).
- **Height $h$** — vertical position in the frame (which row).
- **Time $f$** — which frame, typically converted to seconds (e.g. frame index ÷ FPS = 2.5 s).
- **Global index $g$** — which asset/clip this token belongs to (e.g. the 1st vs. 2nd image in a multi-image prompt).

So a token might sit at $(w{=}15,\ h{=}8,\ f{=}2.5,\ g{=}1)$: column 15, row 8, at 2.5 seconds, in the first asset.

**The key idea is refreshingly simple: we do *not* invent a 4D rotation.** We slice the head dimension into four chunks, one per axis, and run an ordinary **1D RoPE independently on each chunk**, then concatenate. Some dimension pairs get rotated according to $w$, others according to $h$, others according to $f$, and the rest according to $g$. Because each axis is just a 1D RoPE, everything from 7.1–7.2 (rotate pairs, geometric frequency ladder, relative-offset property) carries over per axis. The relative property now holds *per axis*: the attention score depends on $(\Delta w, \Delta h, \Delta f, \Delta g)$ — the offsets along each axis independently.

**How many dimensions per axis?** You split $d$ across the axes according to how much positional resolution each one needs. A representative allocation for a video model, given as an illustrative config:

```yaml
rotary_theta: 10000        # base for the frequency ladder
rotary_axes_dim:
  - 96   # width
  - 96   # height
  - 48   # time
  - 16   # global index
rope_spatial_scale_factor:  8.0    # scale for width and height
rope_temporal_scale_factor: 24.0   # scale for time
```

Width and height get the most dimensions (96 each) because high-resolution frames carry the most spatial detail and need the finest positional resolution. Time gets fewer (48), and the global index gets the fewest (16) because a prompt usually contains only a handful of assets — you simply don't need many bits to tell the 1st asset from the 2nd.

**Why the scale factors?** This is the subtle part. The axes live on very different numeric ranges. Spatial indices might run 0–53, while time in seconds might run only 0–1.25 for a short clip. If you fed those raw into RoPE, a one-*frame* step (say 0.083 s) would produce a *tiny* rotation while a one-*pixel* step produced a large one. The model would then have excellent spatial resolution but almost no temporal resolution — consecutive frames would look positionally identical. The scale factors **stretch each axis so that a one-step move along *any* axis produces a comparable-magnitude rotation.** Multiplying time by 24 and space by 8 rebalances them so the model gets even positional resolution across width, height, and time.

### 7.4 NTK scaling: extending to longer contexts

A final practical problem. RoPE is trained with positions up to some maximum (say 2048, or a 32×32 latent grid). What happens at inference when we hand it position 4096, or a 64×64 grid? The rotation angles $m\theta$ grow into ranges the model **never saw during training**, the query/key rotations look unfamiliar, and quality drops. This is the length-extrapolation problem again, now for RoPE specifically.

**NTK scaling** (from "NTK-aware" RoPE interpolation; NTK = Neural Tangent Kernel, the theory that motivated it) is a clean fix. Rather than extrapolating into unseen large angles, it **stretches the frequency ladder** so that larger positions map back into the familiar angle range the model was trained on. The adjustment is applied to the base frequency:

$$
\theta_{\text{scaled}} = \theta \cdot (\text{scale})^{\,D/(D-2)}
$$

Here $\theta$ is the original base ($10000$), $D$ is the number of dimensions on that axis, and `scale` is how much larger the content is than at training time. Reading the formula: increasing the base $\theta$ makes *every* frequency slower (longer wavelengths), so a large position now produces a rotation that "feels like" a smaller, in-distribution position. In plain terms — **if we process content 2× larger than training, set `scale = 2`, which slows the rotations just enough that positions 0–4096 now occupy the angle range the model learned on 0–2048.** The exponent $D/(D-2)$ (close to 1 for large $D$) is the precise correction that keeps the *fast* frequencies mostly intact while stretching the slow ones — you interpolate long-range positions without blurring short-range detail.

For a spatial axis, `scale` is derived from how much bigger the current grid is than the training grid. A common rule is:

$$
\text{scale} = \left(\frac{\max(h, w)}{32}\right)^{3}
$$

read as: (1) look at the current latent size $h \times w$; (2) compare to the reference training size of 32; (3) if the content is larger, raise the ratio to the third power to get the scale.

**Concrete example.** Training used 32×32 latent grids (512×512 pixel images). Now we generate a 64×64 latent grid (1024×1024 image):

- ratio $= \max(h,w)/32 = 64/32 = 2$
- $\text{scale} = 2^3 = 8$
- $\theta_{\text{scaled}} = 10000 \cdot 8^{\,96/94} \approx 10000 \cdot 8.36 \approx 83{,}600$

The much larger base makes position 64 in the 1024px image rotate as if it were around position 8 in training — comfortably inside the learned range — so the model generalizes to the bigger canvas without ever having trained on it.

---

## Takeaways

- **Self-attention is order-blind** (permutation-equivariant): it sees an unordered bag of tokens. Position embeddings exist solely to inject word order back in.
- **Absolute vs. relative:** absolute PE tags each token with its slot index (simple, but extrapolates poorly and only encodes distance indirectly); relative PE makes attention depend on the offset $m-n$ (translation-invariant, generalizes to longer sequences, but classically fiddly and slow).
- **Learnable vs. fixed:** learned tables (BERT/GPT-2) are simple but add parameters and hit a hard length ceiling; fixed formulas (sinusoidal) add no parameters and extend to any length. Modern models pick RoPE, which is fixed *and* relative-in-effect.
- **Sinusoidal PE** stacks $d/2$ (sin, cos) pairs on a **geometric ladder of frequencies** ($\omega_i = 10000^{-2i/d}$): fast pairs resolve nearby positions, slow pairs resolve far-apart ones. Because rotating an angle is linear, position $m+k$ is a fixed linear map of position $m$ — a hint of relative structure.
- **RoPE** encodes position by **rotating** each query/key dimension-pair by angle $m\theta$. Since a dot product depends on the angle *between* vectors, $q_m \cdot k_n$ collapses to a function of the offset $m-n$ — **applied like absolute, relative in effect**, with no attention-kernel changes.
- **Multi-axis RoPE** (images/video) runs an independent 1D RoPE per axis (width, height, time, global index) and concatenates; per-axis **scale factors** equalize the rotation magnitude of a one-step move across axes with different numeric ranges.
- **NTK scaling** extends RoPE to longer contexts by increasing the base $\theta$ ($\theta_{\text{scaled}} = \theta\,(\text{scale})^{D/(D-2)}$), slowing the rotations so unseen large positions map back into the trained angle range.

---

## Sources

- Vaswani et al. (2017), [*Attention Is All You Need*](https://arxiv.org/abs/1706.03762) (the original sinusoidal position encoding and the learned-vs-fixed comparison).
- Shaw, Uszkoreit, Vaswani (2018), [*Self-Attention with Relative Position Representations*](https://arxiv.org/abs/1803.02155) (early learned relative position bias).
- Raffel et al. (2020), [*Exploring the Limits of Transfer Learning with a Unified Text-to-Text Transformer* (T5)](https://arxiv.org/abs/1910.10683) (relative position bias used in T5).
- Su et al. (2021), [*RoFormer: Enhanced Transformer with Rotary Position Embedding*](https://arxiv.org/abs/2104.09864) (the RoPE paper).
- bloc97 / kaiokendev (2023), NTK-aware scaled RoPE, and Peng et al. (2023), [*YaRN: Efficient Context Window Extension of Large Language Models*](https://arxiv.org/abs/2309.00071) (NTK / frequency-scaling for context extension; the $D/(D-2)$ scaling).
