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
  - [7.1 RoPE in 1D: the core idea](#71-rope-in-1d-the-core-idea)
  - [7.2 Multi-frequency rotation](#72-multi-frequency-rotation)
  - [7.3 Extending to multiple axes: multi-axis RoPE](#73-extending-to-multiple-axes-multi-axis-rope)
  - [7.4 Extending context: Position Interpolation and NTK-aware scaling](#74-extending-context-position-interpolation-and-ntk-aware-scaling)
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

The original Transformer used a **fixed, absolute** position encoding: compute a vector $PE_m$ from position $m$, then add it to the token embedding $x_m$. Its formula looks more mysterious than the underlying idea:

$$
PE_{m,\,2i} = \sin(m\omega_i), \qquad
PE_{m,\,2i+1} = \cos(m\omega_i), \qquad
\omega_i = 10000^{-2i/d}
$$

Here $i = 0, 1, \ldots, d/2-1$ indexes a **pair** of dimensions. The even coordinate $2i$ stores the sine, the neighboring odd coordinate $2i+1$ stores the cosine, and $
\omega_i$ is that pair's angular speed. The usual textbook explanation stops after saying that different pairs use different frequencies. The more useful picture is that every pair is a point moving around a unit circle:

$$
\begin{bmatrix}
PE_{m,\,2i} \\
PE_{m,\,2i+1}
\end{bmatrix}
=
\begin{bmatrix}
\sin(m\omega_i) \\
\cos(m\omega_i)
\end{bmatrix}
$$

Because $\sin^2(m\omega_i)+\cos^2(m\omega_i)=1$, this pair always lies on the circle of radius one. Moving from $m$ to $m+1$ advances it by the fixed angle $\omega_i$. Position is therefore encoded as **rotation**, not as a number written directly into one coordinate.

Why use many circles? Picture an old mechanical car odometer. Its rightmost wheel turns fastest: it steps through 0 to 9 and then returns to 0. The tens wheel moves more slowly, the hundreds wheel more slowly still, and so on. If you look only at the rightmost wheel and see a 7, you cannot tell whether the car has travelled 7, 17, or 10,007 miles. But read all the wheels together and their different turning speeds give you the mileage. The fast wheels resolve small changes; the slow wheels keep track of the larger scale.

Sinusoidal encoding is a smooth version of that stack of wheels. Each pair $(\sin(m\omega_i),\cos(m\omega_i))$ records the angle of one wheel, while $m$ plays the role of distance travelled. There are no discrete digits or carry operations: all the wheels rotate continuously, each at its own speed. Since $\omega_i = 10000^{-2i/d}$ decreases geometrically as $i$ grows, low-index pairs are the fast wheels and high-index pairs are the slow wheels. Their periods are

$$
T_i = \frac{2\pi}{\omega_i} = 2\pi\,10000^{2i/d},
$$

where $T_i$ is the number of positions needed for pair $i$ to complete one turn. The first pair has period $T_0=2\pi\approx 6.3$ positions, while the final pairs have periods approaching $2\pi\cdot 10000$. Reading all pairs together gives each position a rich fingerprint: fast pairs carry fine positional detail and slow pairs carry coarse positional detail. Every coordinate also stays between $-1$ and $1$, and the formula can be evaluated beyond the sequence lengths used in training. That last fact means the encoding itself never runs out of rows; it does **not** guarantee that a model will behave well at unseen lengths.

![A heatmap of the sinusoidal position encoding, with embedding dimension i from 0 to 127 on the horizontal axis and position m from 0 to about 100 on the vertical axis; color runs from blue (negative) through white (zero) to red (positive). The left side (low dimensions) shows a fine, rapidly alternating checkerboard pattern that changes quickly with position; the right side (high dimensions) shows broad vertical bands that change only slowly with position. Annotations point to the left region labeled 'fast dims (distinguish nearby positions)' and the right region labeled 'slow dims (distinguish far-apart positions)'.](../assets/pe_sinusoidal_heatmap.jpg)

The heatmap shows these wheels side by side. Down the low-index columns on the left, colors change rapidly because those pairs turn quickly. Down the high-index columns on the right, colors change gradually because those pairs turn slowly. A row is the joint reading of all the wheels at one position.

The multi-speed fingerprint is useful, but it is not the deepest reason for pairing sine with cosine. The crucial property is that **shifting a position becomes a fixed rotation**. Let $k$ be an integer offset. The angle-addition identities give

$$
\begin{aligned}
\sin((m+k)\omega_i)
&= \sin(m\omega_i)\cos(k\omega_i)
 + \cos(m\omega_i)\sin(k\omega_i), \\
\cos((m+k)\omega_i)
&= \cos(m\omega_i)\cos(k\omega_i)
 - \sin(m\omega_i)\sin(k\omega_i).
\end{aligned}
$$

Both new coordinates are linear combinations of the old pair. In matrix form,

$$
\begin{bmatrix}
PE_{m+k,\,2i} \\
PE_{m+k,\,2i+1}
\end{bmatrix}
=
\underbrace{
\begin{bmatrix}
\cos(k\omega_i) & \sin(k\omega_i) \\
-\sin(k\omega_i) & \cos(k\omega_i)
\end{bmatrix}
}_{\text{depends on }k\text{, not }m}
\begin{bmatrix}
PE_{m,\,2i} \\
PE_{m,\,2i+1}
\end{bmatrix}.
$$

This is why the two channels belong together. A sine value alone is ambiguous and cannot express a shift linearly; its matching cosine supplies the missing coordinate on the circle. Together, “move forward by $k$ positions” means “apply this rotation,” and the same rotation works no matter where $m$ starts. It is like advancing a clock hand by twenty minutes: the required turn is fixed even though the hand's starting angle changes.

That gives an absolute encoding a useful hint of relative structure. In principle, a linear layer can learn one transformation for “$k$ positions away” and reuse it throughout the sequence. But the position vector is still **added** to content, so attention sees content and position mixed together; the relative relationship is available only indirectly. RoPE keeps the same shift-as-rotation idea but applies the rotation directly to queries and keys, making the relative offset appear cleanly in their dot product.

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

**Why rotation gives relative position: the 2D calculation.** Focus on dimension pair $i$. Let $q_m^{(i)}$ be that pair from the query at position $m$, and let $k_n^{(i)}$ be the matching pair from the key at position $n$. RoPE uses the pair's frequency $\theta_i$ and the usual rotation matrix

$$
R(\alpha)=
\begin{bmatrix}
\cos\alpha & -\sin\alpha \\
\sin\alpha & \cos\alpha
\end{bmatrix}
$$

to produce $R(m\theta_i)q_m^{(i)}$ and $R(n\theta_i)k_n^{(i)}$. Each vector therefore receives an **absolute** rotation: the query uses its own position $m$, and the key uses its own position $n$.

Now take their dot product, which is this pair's contribution to the attention score:

$$
\begin{aligned}
\left\langle R(m\theta_i)q_m^{(i)},\,R(n\theta_i)k_n^{(i)}\right\rangle
&= \left(q_m^{(i)}\right)^\top R(m\theta_i)^\top R(n\theta_i)k_n^{(i)} \\
&= \left(q_m^{(i)}\right)^\top R(-m\theta_i)R(n\theta_i)k_n^{(i)} \\
&= \left(q_m^{(i)}\right)^\top R((n-m)\theta_i)k_n^{(i)}.
\end{aligned}
$$

The first line rewrites a dot product as matrix multiplication. The second uses $R(\alpha)^\top=R(-\alpha)$: transposing a rotation reverses it. The third uses $R(\alpha)R(\beta)=R(\alpha+\beta)$, so the query's rotation cancels the matching part of the key's rotation:

$$
R(-m\theta_i)R(n\theta_i)=R((n-m)\theta_i).
$$

The absolute positions $m$ and $n$ have disappeared separately. The score sees them only through the signed offset $n-m$. If we define relative position in the opposite direction, as query minus key, the same result is written using $m-n$; changing the convention only reverses the rotation direction.

Geometrically, this says the dot product cares about the angle **between** the rotated query and key, not their two angles measured from the horizontal axis. Shift both tokens forward by the same amount $s$, and the offset remains $(n+s)-(m+s)=n-m$, so their pairwise score is unchanged. RoPE is therefore an **absolute rotation with a relative effect**. Applying this calculation independently to every dimension pair gives the full RoPE attention score.

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

**Putting all 64 rotations into one matrix.** For this example $d=128$, so the 64 pairwise rotations can be assembled into one $128\times128$ matrix. Define $c_i=\cos(m\omega_i)$ and $s_i=\sin(m\omega_i)$ for pair $i$ at position $m$. Then the full RoPE operator is

$$
R_m=
\begin{bmatrix}
c_0 & -s_0 & 0 & 0 & \cdots & 0 & 0 \\
s_0 &  c_0 & 0 & 0 & \cdots & 0 & 0 \\
0 & 0 & c_1 & -s_1 & \cdots & 0 & 0 \\
0 & 0 & s_1 &  c_1 & \cdots & 0 & 0 \\
\vdots & \vdots & \vdots & \vdots & \ddots & \vdots & \vdots \\
0 & 0 & 0 & 0 & \cdots & c_{63} & -s_{63} \\
0 & 0 & 0 & 0 & \cdots & s_{63} &  c_{63}
\end{bmatrix}
=
\mathrm{diag}\!\left(
R(m\omega_0),R(m\omega_1),\ldots,R(m\omega_{63})
\right).
$$

The displayed zeros mean that different pairs never mix. The upper-left $2\times2$ block rotates coordinates $(d_0,d_1)$ by $m\omega_0$, the next block rotates $(d_2,d_3)$ by $m\omega_1$, and the lower-right block rotates $(d_{126},d_{127})$ by $m\omega_{63}$. Thus applying RoPE to an entire query vector is simply

$$
q_m' = R_m q_m,
$$

where $q_m'$ is the position-rotated query. The key at position $n$ similarly becomes $k_n'=R_nk_n$.

The 2D cancellation from the previous section now happens in every diagonal block at once:

$$
\begin{aligned}
(q_m')^\top k_n'
&=q_m^\top R_m^\top R_n k_n \\
&=q_m^\top\mathrm{diag}\!\left(
R((n-m)\omega_0),\ldots,R((n-m)\omega_{63})
\right)k_n.
\end{aligned}
$$

This is the full picture of 1D RoPE for a 128-dimensional head: one absolute position $m$ creates 64 rotation angles, those rotations fill the diagonal of one large matrix, and the query-key dot product reduces all 64 blocks to the same relative offset $n-m$ at different frequencies.

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

### 7.4 Extending context: Position Interpolation and NTK-aware scaling

RoPE can be evaluated at any position, but that does not mean a trained model understands every resulting angle. Suppose training used relative offsets up to $L$, but inference asks for offsets up to

$$
L' = sL,
$$

where $L'$ is the new context length and $s>1$ is the extension factor. For example, extending a 2048-token model to 4096 tokens gives $s=2$. At frequency $\omega_i$, the largest angle seen during training was $L\omega_i$, whereas inference now reaches $L'\omega_i=sL\omega_i$. Some dimension pairs are therefore pushed onto arcs of their circles that training never covered.

The frequency ladder tells us which pairs are most vulnerable. Fast pairs at small $i$ complete many turns within the training window, so larger offsets mostly revisit familiar angles modulo $2\pi$. Slow pairs at large $i$ may have traversed only a small arc during training, so extending the context sends them directly into unseen angular territory. PI and NTK-aware scaling both try to keep those angles familiar, but they make different trade-offs.

**Position Interpolation (PI): squeeze the entire new context into the old one.** The simplest solution is to divide every inference position by $s$:

$$
m \longmapsto \frac{m}{s}.
$$

Since a RoPE angle is $m\omega_i$, scaling the position is equivalent to scaling every frequency:

$$
m\omega_i \longmapsto \frac{m}{s}\omega_i
=m\frac{\omega_i}{s},
\qquad
\omega_i \longmapsto \omega_i^{\mathrm{PI}}=\frac{\omega_i}{s}.
$$

This guarantees that the new endpoint maps to the old one:

$$
L'\omega_i^{\mathrm{PI}}
=sL\frac{\omega_i}{s}
=L\omega_i.
$$

Thus no pair sees an angle beyond its training range. Imagine a map printed for a five-mile trail. To show a ten-mile trail on the same sheet, PI relabels the scale so that every physical mile occupies half as much paper. The whole route fits, but nearby landmarks become crowded together. RoPE has the same problem: dividing **all** frequencies by $s$ compresses even the fast pairs that already handled long offsets safely. Those pairs carry fine local distinctions, so uniform compression can blur the model's sense of nearby tokens or consecutive frames. PI is simple and often works well with a short fine-tune, but can lose local detail when applied training-free.

**NTK-aware scaling: preserve fast pairs and compress slow pairs.** The natural improvement is to leave the high frequencies alone and progressively interpolate only toward the low-frequency end. NTK-aware scaling achieves this by changing the RoPE base rather than scaling every position. Let $B=10000$ be the original base, so

$$
\omega_i=B^{-2i/d}.
$$

For extension factor $s$, replace $B$ with

$$
B'=B\,s^{d/(d-2)}.
$$

The important part is what this base change does to each frequency. Substituting $B'$ into the frequency formula gives

$$
\begin{aligned}
\omega_i^{\mathrm{NTK}}
&=(B')^{-2i/d} \\
&=\left(B\,s^{d/(d-2)}\right)^{-2i/d} \\
&=B^{-2i/d}s^{-2i/(d-2)} \\
&=\omega_i\,s^{-2i/(d-2)}.
\end{aligned}
$$

Here $d$ is the number of dimensions participating in this 1D RoPE axis. The exponent explains the whole method:

- At the fastest pair, $i=0$, the factor is $s^0=1$, so $\omega_0^{\mathrm{NTK}}=\omega_0$. **Local resolution is untouched.**
- At the slowest pair, $i=d/2-1$, the exponent becomes $-1$, so $\omega_{d/2-1}^{\mathrm{NTK}}=\omega_{d/2-1}/s$. **The vulnerable slowest pair receives full PI-style interpolation.**
- Between those endpoints, the scale changes smoothly and geometrically from $1$ to $1/s$.

A ruler gives the right intuition. To measure a longer distance, we would like to preserve the millimeter ticks that resolve nearby points while stretching the coarse kilometer marks that determine the total range. PI shrinks every tick equally. NTK-aware scaling keeps the finest ticks fixed and increasingly stretches the coarser ones. That is why it usually preserves local detail better than PI in training-free context extension.

For the earlier $d=128$, 64-pair example with $s=2$, pair 0 keeps its original frequency, pair 63 is divided by 2, and the pairs between them receive intermediate factors:

$$
\omega_i^{\mathrm{NTK}}=\omega_i\,2^{-2i/126},
\qquad i=0,1,\ldots,63.
$$

The name “NTK-aware” comes from the neural-tangent-kernel argument that motivated the base rescaling. For understanding the mechanism, the practical takeaway is simpler: **PI compresses the whole frequency ladder uniformly; NTK-aware scaling bends the ladder, preserving fast local frequencies while compressing slow global frequencies.**

---

## Takeaways

- **Self-attention is order-blind** (permutation-equivariant): it sees an unordered bag of tokens. Position embeddings exist solely to inject word order back in.
- **Absolute vs. relative:** absolute PE tags each token with its slot index (simple, but extrapolates poorly and only encodes distance indirectly); relative PE makes attention depend on the offset $m-n$ (translation-invariant, generalizes to longer sequences, but classically fiddly and slow).
- **Learnable vs. fixed:** learned tables (BERT/GPT-2) are simple but add parameters and hit a hard length ceiling; fixed formulas (sinusoidal) add no parameters and extend to any length. Modern models pick RoPE, which is fixed *and* relative-in-effect.
- **Sinusoidal PE** stacks $d/2$ (sin, cos) pairs on a **geometric ladder of frequencies** ($\omega_i = 10000^{-2i/d}$): fast pairs resolve nearby positions, slow pairs resolve far-apart ones. Because rotating an angle is linear, position $m+k$ is a fixed linear map of position $m$ — a hint of relative structure.
- **RoPE** encodes position by **rotating** each query/key dimension-pair by angle $m\theta$. Since a dot product depends on the angle *between* vectors, $q_m \cdot k_n$ collapses to a function of the offset $m-n$ — **applied like absolute, relative in effect**, with no attention-kernel changes.
- **Multi-axis RoPE** (images/video) runs an independent 1D RoPE per axis (width, height, time, global index) and concatenates; per-axis **scale factors** equalize the rotation magnitude of a one-step move across axes with different numeric ranges.
- **PI and NTK-aware scaling** extend RoPE beyond its training length. PI divides every frequency by $s$, fitting all angles into the trained range but compressing local detail. NTK-aware scaling changes the base so the fastest pair is untouched, the slowest pair is divided by $s$, and intermediate pairs are scaled smoothly between them.

---

## Sources

- Vaswani et al. (2017), [*Attention Is All You Need*](https://arxiv.org/abs/1706.03762) (the original sinusoidal position encoding and the learned-vs-fixed comparison).
- Shaw, Uszkoreit, Vaswani (2018), [*Self-Attention with Relative Position Representations*](https://arxiv.org/abs/1803.02155) (early learned relative position bias).
- Raffel et al. (2020), [*Exploring the Limits of Transfer Learning with a Unified Text-to-Text Transformer* (T5)](https://arxiv.org/abs/1910.10683) (relative position bias used in T5).
- Su et al. (2021), [*RoFormer: Enhanced Transformer with Rotary Position Embedding*](https://arxiv.org/abs/2104.09864) (the RoPE paper).
- Chen et al. (2023), [*Extending Context Window of Large Language Models via Positional Interpolation*](https://arxiv.org/abs/2306.15595) (Position Interpolation for extending RoPE context lengths).
- bloc97 / kaiokendev (2023), NTK-aware scaled RoPE, and Peng et al. (2023), [*YaRN: Efficient Context Window Extension of Large Language Models*](https://arxiv.org/abs/2309.00071) (NTK / frequency-scaling for context extension; the $D/(D-2)$ scaling).
