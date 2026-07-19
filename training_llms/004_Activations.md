# Activations in the FFN: from ReLU to Gated Units (SwiGLU / GeGLU)

Every transformer block has two halves: an attention sublayer that lets tokens look at each other, and a **feed-forward network (FFN)** that processes each token on its own. The FFN is where most of a transformer's parameters actually live — it is two big matrix multiplies with a nonlinearity in between — so the *choice of nonlinearity* is not a cosmetic detail. For years that nonlinearity was a plain **ReLU** (or its smoother cousin GELU). Then Noam Shazeer showed, in a short 2020 paper, that replacing it with a **gated** unit — SwiGLU or GeGLU — gives a consistent quality bump for essentially no extra cost. Today virtually every frontier decoder-only model (LLaMA, PaLM, Mistral, DeepSeek, Qwen, …) uses SwiGLU in its FFN.

This note builds up to that result assuming you know only the standard ReLU. We first ask what an activation is *for*, then introduce **GELU** and **Swish/SiLU** as smooth, "soft-gating" replacements for ReLU's hard cutoff. We then introduce the **Gated Linear Unit (GLU)** idea — where one signal *multiplies* another — and show how dropping it into the FFN yields the family of variants (Bilinear, ReGLU, GeGLU, SwiGLU). Finally we cover the trick that keeps the parameter and compute budget fixed, look at Shazeer's numbers, and give the honest intuition for *why* it helps.

---

## Table of Contents

- [Setup and Notation](#setup-and-notation)
- [The FFN and why it needs a nonlinearity](#the-ffn-and-why-it-needs-a-nonlinearity)
- [GELU: a smooth, probabilistic ReLU](#gelu-a-smooth-probabilistic-relu)
- [Swish / SiLU: the sigmoid cousin](#swish--silu-the-sigmoid-cousin)
- [Gated Linear Units: let one signal gate another](#gated-linear-units-let-one-signal-gate-another)
- [Putting a GLU inside the FFN: the variants](#putting-a-glu-inside-the-ffn-the-variants)
- [Keeping the budget fixed: the 2/3 trick](#keeping-the-budget-fixed-the-23-trick)
- [The results](#the-results)
- [So why does gating help?](#so-why-does-gating-help)
- [Takeaways](#takeaways)
- [Sources](#sources)

---

## Setup and Notation

A few symbols recur; each is reintroduced where it first does real work.

| Symbol | Meaning |
|---|---|
| $x \in \mathbb{R}^d$ | The activation vector for **one token** entering the FFN. |
| $d$ | The model's hidden dimension (e.g. 768 or 4096). This is the width of the residual stream. |
| $d_{ff}$ | The FFN's **inner** (hidden) width, i.e. how wide it expands before projecting back. Classically $d_{ff} = 4d$. |
| $W, V$ | Learnable **up-projection** matrices, each $\mathbb{R}^{d \times d_{ff}}$, mapping a token from width $d$ up to width $d_{ff}$. |
| $W_2$ | The learnable **down-projection** matrix, $\mathbb{R}^{d_{ff} \times d}$, mapping back from $d_{ff}$ to $d$. |
| $\sigma(z)$ | The logistic sigmoid, $\sigma(z) = \dfrac{1}{1+e^{-z}}$, squashing any real number into $(0,1)$. |
| $\Phi(z)$ | The **cumulative distribution function (CDF)** of a standard normal — the probability that a standard Gaussian draw is $\le z$. It rises smoothly from 0 to 1. |
| $\odot$ | Element-wise (Hadamard) product of two equal-length vectors. |

Throughout, "activation function" means the scalar nonlinearity applied element-wise (ReLU, GELU, …); "the FFN" or "the block" means the whole two/three-matrix sub-network. Bias terms are omitted throughout — modern LLMs mostly drop them (see [003_NormalisationLayers](003_NormalisationLayers.md)) and they don't change the story.

---

## The FFN and why it needs a nonlinearity

The classic transformer FFN takes a token $x$, projects it up to a wider space, applies a nonlinearity, and projects it back down:

$$
\mathrm{FFN}(x) = \mathrm{ReLU}(xW)\,W_2 = \max(0,\,xW)\,W_2 .
$$

Reading it left to right: $xW$ lifts the token from its native width $d$ into a much wider $d_{ff}$-dimensional scratch space (classically $d_{ff}=4d$, so a 4$\times$ expansion). The **ReLU**, $\max(0, \cdot)$, then zeroes out every negative coordinate and passes positives through unchanged. Finally $W_2$ collapses the wide vector back down to width $d$ so it can be added onto the residual stream.

Why is the nonlinearity non-negotiable? Because without it, $xW W_2$ is just a product of two matrices — which is *itself* a single linear map. Stack a hundred such layers and you still have one linear map; the network could not represent anything a single matrix couldn't. **The nonlinearity is the only thing that lets stacked layers compose into something richer than a matrix.** ReLU's specific job is to make each hidden unit a *gated* detector: it fires (passes its value) when its input is positive and stays silent (outputs 0) otherwise. That "silent below zero" behavior is what lets different units specialize to different input patterns.

ReLU works, but it has two known warts. Its output is **exactly zero for all negative inputs**, so a unit that drifts negative gets *zero gradient* and can "die" — stop learning entirely. And it has a **hard kink at 0**: the decision to fire or not is abrupt, with no smooth transition. The activations below are attempts to keep ReLU's spirit while smoothing that kink.

---

## GELU: a smooth, probabilistic ReLU

**GELU** (Gaussian Error Linear Unit, Hendrycks & Gimpel 2016) is the activation used by BERT, GPT-2, GPT-3, and most ViTs. Its definition looks strange at first but has a clean interpretation:

$$
\mathrm{GELU}(x) = x \cdot \Phi(x),
\qquad
\Phi(x) = \tfrac12\Big(1 + \mathrm{erf}\big(x/\sqrt{2}\big)\Big).
$$

Here $\Phi(x)$ is the standard-normal CDF from the notation table — a smooth S-curve that runs from 0 (far negative) to 1 (far positive), passing through 0.5 at $x=0$. So GELU says: **keep a *fraction* $\Phi(x)$ of the input, where the fraction is how "confidently positive" $x$ looks.** Contrast this with ReLU, which multiplies $x$ by a *hard* gate $\mathbf{1}[x>0]$ that is either exactly 0 or exactly 1. GELU replaces that hard 0/1 switch with a soft, probabilistic dial. The picture below makes the comparison direct — left panel, the output curves; right panel, the gate each one applies:

![Left panel: output curves for ReLU (grey, a hard hinge at zero), GELU (blue) and SiLU/Swish (orange dashed) over input from -5 to 5. GELU and SiLU hug ReLU for large positive x and both dip slightly negative around x=-1 before flattening to zero, with an annotation 'small negative dip (non-monotonic)'. Right panel: the gating fraction each applies. The ReLU gate is a hard step from 0 to 1 at x=0 (grey), annotated 'ReLU decides abruptly'; the GELU gate is the smooth normal CDF Phi(x) rising gradually from 0 to 1 (blue). Title: output = x times gate(x); ReLU uses a hard step, GELU/SiLU use a smooth probability.](../assets/act_relu_gelu_silu.jpg)

Two things to read off the figure. First, for large positive $x$ the gate $\Phi(x)\to 1$ and GELU $\to x$, matching ReLU; for large negative $x$ the gate $\to 0$ and the output vanishes, also matching ReLU. GELU only differs *near the origin*, exactly where ReLU's kink lives — it rounds that corner off. Second, notice GELU (and Swish) **dip slightly negative** for small negative inputs before returning to zero: the output is *non-monotonic*. That small negative lobe means a mildly-negative unit still passes a little signal (and a little gradient) instead of being hard-zeroed, which sidesteps ReLU's "dying unit" problem and gives smoother gradients everywhere.

Because the exact $\mathrm{erf}$ is mildly expensive, two approximations are common and worth recognizing: a $\tanh$ form, and the very clean **sigmoid approximation** $\mathrm{GELU}(x) \approx x\,\sigma(1.702\,x)$ — which is almost exactly the Swish function we meet next.

---

## Swish / SiLU: the sigmoid cousin

**Swish** — also called **SiLU** (Sigmoid Linear Unit) — swaps the Gaussian CDF for the logistic sigmoid as the gate:

$$
\mathrm{Swish}_\beta(x) = x \cdot \sigma(\beta x),
\qquad
\mathrm{SiLU}(x) = x \cdot \sigma(x)\ \ (\beta=1).
$$

The gate is now $\sigma(\beta x)$ instead of $\Phi(x)$, but the shape is nearly identical: a smooth S-curve from 0 to 1 with the same small negative dip. The extra knob $\beta$ controls how sharp the gate is — as $\beta\to\infty$ the sigmoid becomes a step and Swish collapses back to ReLU; as $\beta\to 0$ it becomes a soft linear ramp. In practice people fix $\beta=1$ (plain SiLU) so there is nothing extra to learn. For our purposes GELU and Swish are interchangeable smooth activations; the only reason to introduce both is that the two most successful *gated* variants happen to use one each — GeGLU uses GELU, SwiGLU uses Swish.

---

## Gated Linear Units: let one signal gate another

Everything so far applies a *fixed function* to each coordinate. **Gating** is a different idea, from Dauphin et al. (2017): instead of squashing a value with a fixed curve, compute **two** projections of the input and let one *multiply* the other, element-wise. The original Gated Linear Unit is

$$
\mathrm{GLU}(x) = (xW) \odot \sigma(xV) .
$$

Two separate linear maps of the same input $x$: call $xW$ the **value** and $\sigma(xV)$ the **gate**. The gate is squashed into $(0,1)$ by a sigmoid, so each of its coordinates is a soft "open/closed" valve; multiplying element-wise, the value passes through wherever the gate is near 1 and is suppressed wherever the gate is near 0. The crucial difference from a plain activation: **the gate is data-dependent and learned separately from the value.** ReLU decides whether coordinate $i$ fires based only on coordinate $i$'s own value; a GLU can learn a gate for coordinate $i$ that depends on a *different learned projection* of the whole input. It's a multiplicative interaction, which a single element-wise curve can never produce.

You can see why this is expressive: multiplication lets the network represent "this feature matters *only when* that other condition holds," a kind of soft if-then that additive layers approximate only clumsily. This is the same mechanism at the heart of LSTMs (input/forget gates) — Dauphin's contribution was showing it works as a *feed-forward* building block for language modeling.

---

## Putting a GLU inside the FFN: the variants

Shazeer's 2020 paper, *GLU Variants Improve Transformer*, does one thing: it replaces the FFN's "up-project → activate → down-project" with "up-project **two** ways → gate → down-project." Concretely the standard FFN

$$
\mathrm{FFN}(x) = \mathrm{act}(xW)\,W_2
$$

becomes the **gated FFN**

$$
\mathrm{FFN}_{\mathrm{GLU}}(x) = \big(\mathrm{act}(xW) \odot xV\big)\,W_2 .
$$

The structural change is exactly the extra matrix $V$: we now form two up-projections of the token, $xW$ and $xV$; we pass **one** of them ($xW$) through an activation to make the gate, leave the **other** ($xV$) as a plain linear value, multiply them element-wise, and send the product through the down-projection $W_2$. The diagram contrasts the two:

![Two block diagrams side by side. Left, 'Standard FFN (2 matrices)': input x flows up through 'up-proj xW' (d to d_ff), then an 'activation (ReLU/GELU)' box, then 'down-proj times W2' (d_ff to d); labelled max(0, xW) W2. Right, 'Gated FFN / GLU (3 matrices)': input x splits into two parallel up-projections, a 'gate proj xW' on the left feeding an 'activation (GELU/Swish)' box, and a 'value proj xV' on the right left linear; the activated gate and the linear value meet at an element-wise-product node (circle with a dot), whose output goes up through 'down-proj times W2'; labelled (act(xW) element-product xV) W2. A caption notes one branch is squashed by an activation then multiplied into a plain linear copy.](../assets/act_ffn_vs_gated_ffn.jpg)

The family of "GLU variants" is just this template with a different choice of activation on the gate branch — note the original GLU's sigmoid is only one option, and it turns out to be a mediocre one:

| Variant | Gate activation | Formula |
|---|---|---|
| **Bilinear** | none (identity) | $(xW)\odot(xV)\,W_2$ |
| **GLU** | sigmoid | $\sigma(xW)\odot(xV)\,W_2$ |
| **ReGLU** | ReLU | $\max(0,xW)\odot(xV)\,W_2$ |
| **GeGLU** | GELU | $\mathrm{GELU}(xW)\odot(xV)\,W_2$ |
| **SwiGLU** | Swish | $\mathrm{Swish}(xW)\odot(xV)\,W_2$ |

**Bilinear** is the striking edge case: no activation at all, purely two linear projections multiplied. It still has a nonlinearity — the *multiplication itself* is nonlinear in $x$ — and, as we'll see, it already beats plain ReLU. That is the clearest evidence that the *gating* (the multiplicative interaction), not the specific squashing curve, is where most of the benefit comes from. GeGLU and SwiGLU add a good activation on top of the gating and do best of all.

---

## Keeping the budget fixed: the 2/3 trick

There's an obvious objection: the gated FFN has **three** big matrices ($W$, $V$, $W_2$) where the standard one has **two** ($W$, $W_2$). Isn't that just a bigger, more expensive network — so of course it does better? Shazeer heads this off by *shrinking the inner width* so the parameter and FLOP counts match.

The standard FFN's two matrices are each $d \times d_{ff}$, so its parameter count is $2\,d\,d_{ff}$. The gated FFN's three matrices cost $3\,d\,d_{ff}'$ for some new inner width $d_{ff}'$. Setting them equal, $3\,d\,d_{ff}' = 2\,d\,d_{ff}$, gives

$$
d_{ff}' = \tfrac{2}{3}\,d_{ff} .
$$

So you shrink the FFN's hidden dimension to **two-thirds** of its usual size. With the classic $d_{ff}=4d$ this means the gated FFN uses $d_{ff}' = \tfrac{2}{3}\cdot 4d = \tfrac{8}{3}d$ — this is exactly why you see the odd-looking factor $\tfrac{8}{3}$ (and inner widths like 2048 vs 3072, or the famous "multiple-of-256" roundings in LLaMA) in real configs. In Shazeer's experiments $d=768$, so the baselines use $d_{ff}=3072$ and the gated variants use $d_{ff}'=2048$. **After this adjustment every model in the comparison has the same parameter count and the same compute**, so any quality difference is attributable to the *shape* of the FFN, not its size. This is what makes the result a fair, almost free, swap.

---

## The results

With the budget held fixed, Shazeer pre-trained a T5-style model with each FFN and measured heldout **log-perplexity** (lower is better — it's the model's average surprise per token). The gated variants win cleanly:

![Bar chart of heldout log-perplexity for eight FFN variants, lower is better. Non-gated baselines in grey: ReLU 1.997, GELU 1.983, Swish 1.994. Gated GLU variants in orange: GLU 1.982, Bilinear 1.960, ReGLU 1.953, GeGLU 1.942, SwiGLU 1.944. A dashed line marks the ReLU baseline. All gated variants sit below the non-gated ones, with GeGLU and SwiGLU lowest.](../assets/act_glu_perplexity.jpg)

| FFN | Log-perplexity (65k steps) | Gated? |
|---|---|---|
| ReLU (baseline) | 1.997 | no |
| GELU | 1.983 | no |
| Swish | 1.994 | no |
| GLU (sigmoid) | 1.982 | yes |
| Bilinear (no act.) | 1.960 | yes |
| ReGLU | 1.953 | yes |
| **GeGLU** | **1.942** | yes |
| **SwiGLU** | **1.944** | yes |

Read the ledger: swapping ReLU for GELU/Swish buys you a little (2.00 → 1.98). But *every* gated variant beats *every* non-gated one, and the best (**GeGLU / SwiGLU**, ≈1.94) open a clear gap over the ReLU baseline. Even parameter-free **Bilinear** (1.960) beats GELU — again pointing at the gating, not the curve, as the source of the gain. The advantage carries over to fine-tuning: on downstream GLUE/SuperGLUE and SQuAD the gated variants come out ahead too, so this isn't a pre-training-only artifact. Because the compute is identical, the community read this as a near-free win and adopted **SwiGLU** as the default (SwiGLU and GeGLU are within noise of each other; SwiGLU became the convention via LLaMA and PaLM).

---

## So why does gating help?

Honestly, the field does not have a fully rigorous answer — and Shazeer says so himself, in the paper's most-quoted line:

> "We offer no explanation as to why these architectures seem to work; we attribute their success, as all else, to divine benevolence."

Still, the intuition we *can* offer is the multiplicative one from earlier. A standard FFN unit applies a fixed curve to each coordinate independently; a gated unit computes a **data-dependent gate** from one learned projection and uses it to modulate a **separate learned value**. That multiplication is a genuinely new kind of interaction — it lets the network cheaply express conditional, "route this feature through only when that pattern is present" behavior that additive layers reach only with more parameters and depth. Bilinear's success (a pure product, no activation, still beating ReLU) is the cleanest evidence that this multiplicative interaction is doing the heavy lifting. The soft activation on the gate (GELU/Swish) then adds its usual smooth-gradient benefits on top. Combine "smoother than ReLU" with "multiplicative gating for free," verify empirically that quality goes up at fixed cost, and you have exactly why SwiGLU sits in nearly every modern LLM's feed-forward block.

---

## Takeaways

- **The FFN's nonlinearity is what makes depth meaningful** — without it, stacked linear layers collapse to a single matrix. ReLU's warts are its hard kink and dead (zero-gradient) negative region.
- **GELU and Swish/SiLU are smooth ReLUs.** Both have the form $x \cdot \mathrm{gate}(x)$ with a *soft* gate ($\Phi(x)$ or $\sigma(x)$) instead of ReLU's hard 0/1 step; both dip slightly negative, which keeps gradients alive everywhere.
- **A GLU multiplies two learned projections** — a value $xV$ and a squashed gate $\mathrm{act}(xW)$ — giving a data-dependent, multiplicative interaction that a single element-wise curve cannot produce.
- **The gated FFN adds a third matrix; the $\tfrac{2}{3}$ trick shrinks the inner width** ($d_{ff}\to\tfrac{2}{3}d_{ff}$, hence the $\tfrac{8}{3}d$ you see in configs) so parameters and FLOPs stay fixed — making it a fair, near-free swap.
- **Gated variants win at fixed budget.** Every GLU variant beat every non-gated FFN in Shazeer's tests; **GeGLU/SwiGLU** were best (≈1.94 vs ReLU's ≈2.00 log-perplexity), and even activation-free **Bilinear** beat GELU — evidence the *gating* matters more than the curve.
- **SwiGLU is now the default FFN** in most frontier decoder-only LLMs (LLaMA, PaLM, Mistral, DeepSeek, …), with no crisp theory for *why* — "divine benevolence," per the author.

---

## Sources

- Shazeer (2020), [*GLU Variants Improve Transformer*](https://arxiv.org/abs/2002.05202) (the SwiGLU/GeGLU paper; Table 1 perplexities, the $\tfrac{2}{3}$ trick, the divine-benevolence quote).
- Dauphin, Fan, Auli & Grangier (2017), [*Language Modeling with Gated Convolutional Networks*](https://arxiv.org/abs/1612.08083) (the original Gated Linear Unit).
- Hendrycks & Gimpel (2016), [*Gaussian Error Linear Units (GELUs)*](https://arxiv.org/abs/1606.08415) (GELU $=x\,\Phi(x)$ and its sigmoid/tanh approximations).
- Ramachandran, Zoph & Le (2017), [*Searching for Activation Functions*](https://arxiv.org/abs/1710.05941) (Swish $=x\,\sigma(\beta x)$).
- Elfwing, Uchibe & Doya (2017), [*Sigmoid-Weighted Linear Units for Neural Network Function Approximation in Reinforcement Learning*](https://arxiv.org/abs/1702.03118) (SiLU, the $\beta=1$ Swish).
