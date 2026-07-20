# Scaling Laws

> Based on Stanford CS336 (Percy Liang), Lecture 9 — *Scaling Laws: Basics* — plus the original papers it draws on (Hestness 2017, Kaplan 2020, Hoffmann/Chinchilla 2022).

## The one question this whole topic answers

Imagine a friend hands you **ten thousand B200 GPUs for one month** and says: *build a good open-source language model.* You assemble an infra team, you curate a great pretraining dataset — and then you hit the wall that this note is about: **which model do you actually train?**

Wide or deep? How many attention heads? Which nonlinearity? Adam or SGD? A bigger model on less data, or a smaller model on more data? You get essentially **one shot** — the run eats your entire budget — so you cannot just try ten configurations and keep the best. The old, unpleasant way was to tune these knobs directly on the big model (ruinously expensive) or to *cargo-cult* whatever the last famous LM did (but where did *those* choices come from?).

**Scaling laws** are the escape. The core bet is:

> Train a handful of **small, cheap** models, watch how performance changes as you scale, fit a **simple predictive curve**, and **extrapolate** to the big model you can only afford to train once.

That is the entire game. Everything below is either (a) *evidence* that such simple curves exist and are remarkably regular, (b) *why* they take the shape they do, or (c) *how to use them* to make the two big decisions — hyperparameters, and the data-vs-model-size split (the famous Chinchilla result).

---

## Setup and Notation

These symbols recur throughout; skim once and refer back.

| Symbol | Meaning |
|---|---|
| $N$ (also $m$) | **model size** — number of parameters (often *non-embedding* params; this distinction matters later) |
| $D$ (also $n$) | **data size** — number of training tokens (or examples) |
| $C$ | **compute** budget, usually in FLOPs. Rule of thumb for a transformer: $C \approx 6\,N\,D$ (6 FLOPs per parameter per token — forward + backward) |
| $L$ | the **loss** (cross-entropy in nats/token) or, more generally, test **error** |
| $\alpha, \beta$ | **scaling exponents** — the *slopes* of the power laws; the star quantities of this topic |
| $E$ | the **irreducible loss** — the noise/entropy floor no model can beat |
| $B_\mathrm{crit}$ | the **critical batch size** — the batch beyond which returns diminish |

A note on the two naming conventions: the theory/statistics literature (and Percy's Part 1) uses $n$ for data; the LLM papers (Kaplan, Chinchilla) use $N$ for parameters and $D$ for data. I'll use $n$ in the statistical toy examples and switch to $N,D$ once we reach the LLM laws — flagging it each time.

---

## 1. What a scaling law actually *is*

A scaling law is a **simple formula mapping a resource to error**. The simplest is data → error: a function that takes dataset size $n$ and returns the loss you'll achieve. The empirical discovery — first for classical ML, then spectacularly for neural nets — is that this function is a **power law**:

$$L(n) \;\approx\; c \, n^{-\alpha}$$

Here $c$ is a constant and $\alpha > 0$ is the exponent. Why do practitioners get so excited about *power laws* specifically? Because of what they look like when you take logs of both sides:

$$\log L(n) \;=\; \log c \;-\; \alpha \log n .$$

**On a log-log plot, a power law is a straight line** with slope $-\alpha$. That is the single most important picture in this entire topic. A straight line is the easiest thing in the world to *extrapolate*: measure loss at a few small $n$, draw the line, read off where it goes. Power laws are also called **"scale-free"** — the *fractional* drop in loss from doubling data is the same whether you go from 1M→2M tokens or 1B→2B tokens. There's no special scale at which the benefit of data "kicks in" or "runs out"; every doubling buys the same multiplicative improvement.

![A single log-log plot. The x-axis is dataset size n (log scale, spanning several decades); the y-axis is test loss (log scale). A cloud of measured points falls almost perfectly on a straight descending line of slope minus alpha, annotated "L = c times n to the minus alpha" and "straight line on log-log = power law". A faint dashed extension of the line continues to the right into a shaded "extrapolation" region, showing how a few cheap measurements at small n predict the loss at large n. A second inset shows the same data on ordinary linear axes, where it looks like a boring curved decay with no obvious structure, captioned "same data, linear axes — structure hidden".](../assets/scale_powerlaw_loglog.jpg)

Two caveats stated up front, because they recur:

* **It's not a power law forever.** The *expected* shape [Hestness+ 2017] is monotonic and **logistic-like (S-shaped)**, with three regimes: a flat *small-data* region where the model is basically guessing, the clean *power-law* middle, and an *irreducible-error* floor $E$ where the curve flattens because you've hit the entropy of the data itself. The "law" — the straight line on log-log — is the middle stretch.
* **A law is an empirical regularity, not a guarantee.** It can *break* if you apply it blindly outside the range you fit, and it's better read as a **lower bound on what's achievable** — a good recipe can sometimes beat the extrapolation, it just tells you what you can *count on*.
* **It's pure curve-fitting, not a theorem.** There is no golden rule forcing the form to be a power law — scaling laws are *empirical fits*. Theory (how error rates decay) and physics (how limits behave) merely *suggest* candidate functional forms. A sharp corollary Percy stresses: over a **narrow slice of compute, everything looks linear** (Taylor's theorem), so you genuinely cannot tell a power law from an exponential until you span many orders of magnitude — always be a little skeptical of a fit drawn from a short range.

A cultural aside that explains a lot of the field: at the big labs, scaling laws are less a tool than **a paradigm — almost a belief system**. "We believe in the scaling laws" is a real stance, because the whole enterprise of spending millions on one run rests on trusting that a curve fit at small scale will hold at large scale.

![A schematic learning curve on log-log axes: generalization error (y, log) versus training data size (x, log). The curve has three shaded regions left to right. The "small-data region" (grey) is a nearly-flat plateau near a dotted line marked "≈ random / best-guess error", where more data barely helps. The "power-law region" (blue, the middle) is a straight downward line annotated "straight line on log-log, slope = -alpha" — the actual scaling law. The "irreducible-error region" (green) flattens onto a horizontal dashed floor labelled "irreducible error E", the entropy of the data that no amount of data can beat. The overall shape is the monotonic, S-shaped (logistic-like) curve of Hestness et al 2017.](../assets/scale_hestness_regimes.jpg)

---

## 2. A brief history (skim — expand on request)

Scaling is older than deep learning. This is background; I keep it short and flag the one paper that matters most.

* **Learning theory (pre-1990s).** Statisticians long studied *sample complexity* — how error shrinks as $n$ grows — via VC-dimension bounds and density-estimation rates (e.g. Hall 1989). But these are **upper bounds** on worst-case error, not the *actual realized loss* you'd measure; they told us decay should happen, not what curve to expect in practice.
* **1993 — the earliest data-scaling-law paper**, from **Bell Labs (Cortes, Vapnik and colleagues)**: training big classifiers is expensive, so fit classifiers on *small* samples, fit a curve to how their error decays, and use it to *estimate* large-scale performance — almost literally a data scaling law, three decades ago. Then **Banko & Brill 2001**, who showed **log-linear** improvement with data on NLP tasks (the "more data beats better algorithms" result).
* **Kolachina et al 2012** fit explicit power-law functional forms relating data to downstream MT quality.
* **⭐ Hestness et al 2017 — *"Deep Learning Scaling is Predictable, Empirically."*** This is the pivotal one: the first *large-scale neural* study showing predictable power-law scaling across **machine translation, language modeling, and speech**, and hypothesizing the universal learning-curve shape. It was years ahead of its time — it already gestured at what we'd later call *emergence*, at scaling *by compute*, and at the "speed = accuracy" idea. We'll use its findings as the empirical anchor for the theory in §4.

---

## 3. The three questions neural scaling answers

Percy frames the neural part as three practical questions. The whole rest of the note is organized around them:

1. **Data vs performance** — *are there simple rules for how data affects the model?* (§4)
2. **Data vs model size** — *should we spend our budget on more data or a bigger model?* (§7, Chinchilla)
3. **Hyperparameters vs performance** — *how do we set the knobs on the big model without training it repeatedly?* (§6)

A striking empirical fact frames all three: these power-law relationships hold across an astonishing range of phenomena — model size, dataset size, compute — and even in non-standard settings [Kaplan+ 2020]. The regularity is the gift; the questions are how we cash it in.

---

## 4. Data scaling — and *why* it's a power law

We know error should *decrease* monotonically with data. The mystery is why it decreases as a **power law** (a straight line in log-log) rather than, say, exponentially or along some idiosyncratic curve. The answer Percy builds up — and the deepest, most "buried" intuition in this lecture — is:

> **Estimation error decays polynomially in the number of samples, and a polynomial rate $1/n^\alpha$ *is* a power law.** The exponent $\alpha$ is set by how hard the estimation problem is — ultimately, by the *effective dimensionality* of the data.

Let's earn that statement in three steps, from a trivial example to the real claim.

> **Setup assumption.** Throughout this section, keep the **model comfortably larger than the dataset** (rule of thumb ~10× more capacity than data warrants). This keeps us in the clean *power-law* middle of the learning curve and away from the irreducible-error floor — if the model gets small relative to the data, it saturates and you drop into the flat asymptote, which is a different regime. Data scaling laws are a statement about the power-law stretch.

### 4.1 Toy example: estimating a mean

The cleanest possible "learning" problem. Draw $n$ samples $x_1,\dots,x_n \sim N(\mu, \sigma^2)$ — i.i.d. draws from a Gaussian with unknown mean $\mu$ and variance $\sigma^2$. Your "model" is a single number: the estimate of the mean, $\hat\mu = \frac{1}{n}\sum_i x_i$. How wrong are you, in expectation?

Standard result for the variance of a sample mean:

$$\mathbb{E}\big[(\hat\mu - \mu)^2\big] \;=\; \frac{\sigma^2}{n}.$$

Unpack it: each sample contributes independent noise of variance $\sigma^2$; averaging $n$ of them cuts the variance by a factor $n$ (the classic $1/n$ of the law of large numbers). Now take logs — the same move that turned a power law into a line:

$$\log\big(\text{Error}\big) \;=\; -\log n \;+\; 2\log\sigma .$$

**This is already a scaling law.** It's a straight line in log-log with **slope $-1$** and an intercept set by the noise level $\sigma$. Notice the anatomy that will repeat everywhere: the **slope** ($-1$) comes from the *structure of the problem* (averaging independent noise); the **offset** ($2\log\sigma$) comes from *how noisy* the data is. More generally, any polynomial rate $1/n^\alpha$ is a scaling law with slope $-\alpha$.

> **Remember this slope-vs-offset split — it is the single most-repeated empirical lesson in the whole lecture.** Almost *every* intervention you'll try — a better data mixture, SGD → Adam, adding regularization or ensembling, even a fancier architecture — moves the **offset** (a constant-factor win at every scale) while leaving the **slope essentially unchanged**. Slopes are set by the problem / model class; only intercepts move. It's surprising enough that Percy says he's taken aback "every time I see it," yet it holds remarkably often — and it's the reason so many design decisions made cheaply at small scale still hold at large scale. Watch for this refrain in §5 (mixtures, repetition, ensembling) and §6 (optimizer, architecture).

![A log-log plot of mean-squared error versus number of samples n. Simulated points (many independent trials averaged) lie on a straight descending line of slope exactly minus one, labelled "Error = sigma-squared / n, slope = -1". Two parallel lines are drawn for two noise levels sigma, the noisier one shifted vertically upward (higher offset) but with identical slope, annotated "slope set by the problem (averaging), offset set by noise sigma".](../assets/scale_mean_estimation.jpg)

### 4.2 The mystery: classical slope is $-1$, neural slopes are much shallower

Here's the puzzle. Nearly all *classical* estimators — mean estimation, linear/parametric regression, and friends — have this same $1/n$ behavior. So the theory *predicts* a slope of about $-1$: a plot of $\log(\text{error})$ vs $\log n$ should look like $y = -x + C$.

But when Hestness measured real neural scaling laws, the slopes were **nowhere near $-1$** — they were far *shallower*, and they *differed by domain*:

| Domain (Hestness 2017) | measured exponent $\alpha$ |
|---|---|
| Word language models | $\approx 0.066$ |
| Character language models | $\approx 0.094$ |
| Machine translation | $\approx 0.128$ |
| Speech recognition | $\approx 0.30$ |
| Image classification (top-1) | $\approx 0.31$ |

A slope of $-0.07$ instead of $-1$ is a *huge* qualitative difference: it means neural nets need **enormously** more data to squeeze out the same relative error gain than the classical theory would suggest. Why so shallow — and why does the number depend on the task?

![Two side-by-side log-log panels of error versus data. Left, "classical prediction": a steep straight line of slope minus one. Right, "measured neural scaling": several much flatter lines, one per domain, with slopes annotated -0.066 (word LM), -0.128 (MT), -0.30 (speech), -0.31 (vision). A big arrow between the panels reads "why so much shallower, and why domain-dependent?".](../assets/scale_classical_vs_neural.jpg)

### 4.3 The resolution: nonparametric learning has dimension-dependent slopes

The fix is to stop modeling neural nets as *parametric* estimators (a fixed handful of parameters, like a single mean) and treat them as **nonparametric** ones — flexible enough to approximate essentially arbitrary functions. That flexibility changes the scaling dramatically. Here is the toy example that makes it click.

**Setup.** Inputs $x_1,\dots,x_n$ are spread uniformly in the 2D unit square. Each label is a smooth function plus noise, $y_i = f(x_i) + N(0,1)$, and the task is to estimate the function $f$. A simple *nonparametric* recipe: **chop the input space into little boxes and average the $y$'s inside each box.** To keep it balanced, choose the box side length so each box holds a handful of points.

**The counting argument.** In 2D, if we make boxes of side length $n^{-1/4}$, there are about $\big(n^{1/4}\big)^2 = \sqrt{n}$ boxes, so each box catches about $n/\sqrt{n} = \sqrt{n}$ samples. Within a box you're back to *mean estimation*, so that box's error is $\approx 1/(\text{samples in box}) = 1/\sqrt{n}$. Averaged over boxes, the overall error scales as

$$\text{Error} \;\approx\; \frac{1}{\sqrt n} \;=\; n^{-1/2} \quad(\text{in 2D}),$$

plus smaller "smoothness" terms. Generalize the box-counting to $d$ input dimensions and the exponent picks up the dimension directly:

$$\boxed{\;\text{Error} \;=\; n^{-1/d} \;\;\Longrightarrow\;\; \log(\text{Error}) = -\tfrac{1}{d}\log n + C\;}$$

**This is the punchline.** The slope is $-1/d$, set by the **dimensionality of the problem**. High-dimensional data (large $d$) gives a *shallow* slope — exactly the "we need tons of data" behavior we saw in neural nets. This is one face of the **curse of dimensionality**: carving up a high-dimensional space needs exponentially many boxes, so each new sample helps only a little.

![A schematic in three parts. Left, a 2D unit square partitioned into a grid of small boxes, dots scattered inside, each box labelled with its local average — captioned "nonparametric estimate: average y within each box". Middle, an arrow to a formula "each box gets about root-n points, box error about 1/root-n, so total error about n^(-1/2) in 2D, and n^(-1/d) in d dimensions". Right, a log-log plot with several straight lines of slope -1/d for d = 2, 5, 20, showing the slope flattening toward horizontal as d grows, annotated "higher dimension = shallower slope = data-hungry".](../assets/scale_nonparametric.jpg)

### 4.4 The intrinsic-dimensionality theory

This leads to an elegant, if not airtight, theory of neural scaling [Bahri et al 2021]:

1. Scaling laws arise because learning has **polynomial rates** $1/n^\alpha$.
2. The exponent $\alpha$ is tied to the **intrinsic dimension** of the data — not the raw pixel/token count, but the dimension of the low-dimensional manifold the data actually lives on. Natural images and text sit on far lower-dimensional manifolds than their nominal size, so $d$ (and hence the slope) reflects that intrinsic structure.

The story is appealing because it *explains* the domain-dependence: different data types have different intrinsic dimensions, hence different slopes. The honest caveat Percy stresses: **estimators of intrinsic dimension are sketchy**, so this is a compelling intuition rather than a proven law. But as a mental model — *shallow scaling slope ⇔ high effective dimension of the problem* — it's the best available.

### 4.5 Recap of the theory arc

Mean estimation gave us slope $-1$ from averaging noise. Nonparametric box-counting turned that into slope $-1/d$, revealing that *flexibility buys dimension-dependence*. And the intrinsic-dimension hypothesis ties the measured neural exponents back to how complex the data really is. That chain — **polynomial rate → dimension in the exponent → intrinsic dimension of data** — is *why* data scaling is a power law, and why its slope is what it is.

---

## 5. Advanced data scaling (medium depth)

Here's the honest motivation for this section. A plain data scaling law, by itself, **isn't that useful** — it only tells you *how fast your model learns*, which is good for forecasting and little else. The *engineering* value comes from richer questions: what's my optimal **data mixture**? Should I **repeat** data or spend the compute elsewhere? How does the answer change with **scale**? These all ride on the same power-law machinery. I treat the two most useful in a bit of depth and flag the rest for expansion.

### 5.1 Distribution shift: composition changes the *offset*, not the *slope*

What if your test data comes from a different distribution than your training data? A clean finding [Hashimoto 2021]: distribution shift, and data composition more generally, moves the **offset** of the scaling line up, but leaves the **slope** essentially unchanged — the same offset-vs-slope split we saw in §4.1. Intuitively, training on a mismatched or lower-quality mix is like adding a fixed "tax" to your loss at every scale; it doesn't change *how fast* you learn, just *where you start*. The practical upshot: this is exactly why **collecting diverse, on-distribution data matters** — it lowers the whole curve.

### 5.2 Data repetition: when you run out of fresh tokens

In reality data is finite, so you may **repeat** examples across epochs. How much is a repeated token worth? The scaling-under-repetition work (Muennighoff et al) models an **effective data** size $D'$ that is smaller than the naive count once tokens are seen more than once: with $U_D$ *unique* tokens and repetition count $R_D$, repeated tokens contribute with diminishing value, and there's a "constant" $R_D^\star$ marking where repetition stops helping much (empirically, up to ~4 epochs is nearly as good as fresh data; beyond ~40 epochs repetition adds almost nothing). Two consequences flow from this:

* **Data selection should be adaptive to scale.** Since repeated data is worth less, how aggressively you filter/curate should depend on how much compute (and thus how many effective passes) you're planning — a large run exhausts easy data and must reach for more.
* It sharpens the earlier caution: **scaling laws can break** if you extrapolate past the point where you've run out of fresh data.

**Taken to the extreme — the infinite-compute regime.** Recent work (Percy and a co-advised student) asks the limiting question: with *unlimited* compute but *fixed* data, how good can you get? You can't just keep repeating passes or growing the model — both hit diminishing returns — so you reach for other levers, notably **ensembling** models to squeeze more out of finite data. The telling detail, and the recurring refrain: these interventions (regularization, ensembling) improve performance but mostly move the **offset** — the *slopes stay remarkably similar*.

### 5.3 The rest, briefly (expand on request)

* **Data-mixture selection** — using small models to pick the best mixture of sources. Attractive in theory, but *hard in practice*; a strong baseline is simply "train small models on each candidate set and take the best."
* **Quality vs repetition trade-offs** — combining mixture choice with repetition budgeting.

These are active, useful, but more niche; say the word and I'll expand any of them.

### 5.4 Recap: data scaling

Pulling the data half together before we switch to model engineering:

* **Empirically**, there's a remarkably **linear relationship between log-data-size and log-error** — a power law — that **holds across domains and models**.
* **Theoretically**, it looks a lot like classical generalization/sample-complexity bounds; the **mean-estimation example** is the cleanest window into *why* (polynomial decay → power law), and the slope reflects the problem's effective dimension.
* **In practice**, this drives **data collection and curation** decisions — how much data to gather, what to mix, whether to repeat — all with the offset-vs-slope lens.

---

## 6. Using scaling laws to set hyperparameters

Now the payoff for the "10,000 B200s" problem. The **design procedure** is always the same three steps:

1. Train a **few small** models spanning a range of scales.
2. Establish a **scaling law** for the choice you care about (e.g. one curve for Adam, one for SGD).
3. **Pick the option whose extrapolated curve wins** at your target scale — *before* ever training the big model.

Percy runs this through the classic **Kaplan et al 2020** framework across four kinds of knobs.

### 6.1 Architecture — Transformers vs LSTMs, and shape

Rather than spend tens of millions training an LSTM the size of GPT-3, you fit scaling laws for both families on small models and compare slopes. Result [Kaplan+ 2020]: **Transformers dominate** — LSTMs keep up on tokens early in the context but can't match Transformers on later tokens, and the gap widens with scale. A related, almost shocking finding: at a **fixed parameter count $N$, performance depends only very weakly on the *shape*** — depth, width, aspect ratio, number of heads. A $(n_\text{layer}, d_\text{model}) = (6, 4288)$ model lands within ~3% of a $(48, 1600)$ model. What matters is *how many* parameters, not *how you arrange* them (within reason).

This is now the field's default lens on architecture: **"if it doesn't show up in the scaling law, it's not a good intervention."** Every new-architecture paper (Mamba, gated DeltaNet, …) ships a plot of *their model vs. a vanilla transformer* across compute — and you want your line either below or with a *better slope*, since a worse slope means you eventually *lose* at scale. The one I'd single out is **Tay et al 2022** (the T5-scaling study): trained many architecture variants at increasing scale, and it *retroactively predicted the choices we actually use today* — gated linear units scale well (we adopted them), efficient-attention like Performer scales poorly (we didn't), Switch-transformer MoE scales well (we did). That a small-scale study foresaw frontier choices is exactly why people treat scaling laws as paradigmatic.

### 6.2 Optimizer — Adam vs SGD

Straight from the §4 offset-vs-slope lesson: [Hestness+ 2017] found that swapping optimizers (or improving architecture) **shifts the learning curve down** — a better *offset* — but **does not change the exponent**. So a better optimizer is a constant-factor win at every scale, not a change in how you scale. (Their study predates transformers — "RHN" there is *recurrent highway nets*.)

### 6.3 Depth/width and the "value of a parameter"

Digging into shape: going from **1 to 2 layers matters enormously**, but beyond that extra layers show **diminishing returns** below ~$10^7$ params. There's a subtlety worth naming: the *number of layers* is **not scale-invariant** — bigger models genuinely want more layers — but the **aspect ratio** ($d_\text{model}$ per layer) roughly *is*: Kaplan finds the optimal aspect ratio sits near the same value (~100) across model sizes, drifting only slightly. That's what makes "fix the aspect ratio and scale up" a safe recipe — you can plot it and confirm the optimum isn't moving out from under you.

And crucially, **not all parameters are equal** — **embedding-layer** parameters behave differently from the rest. This is why the field distinguishes **non-embedding** parameter counts, and it's the seed of a Kaplan-vs-Chinchilla discrepancy we'll hit in §7. Related and rapidly growing: **scaling laws for Mixtures-of-Experts (MoE)**, where only a subset of parameters is active per token, so "parameters" and "compute" decouple and the effective value of a parameter changes. A neat MoE finding (Apple/MIT): compute-optimal models get **sparser** as they scale, and the *inactive* experts still lower loss — a "parameter" is worth something even when it isn't firing.

This is also where Percy makes a point that reframes the entire topic: **scaling laws aren't magic — they're engineered.** Predictable straight lines across many orders of magnitude *do not happen automatically*; you have to pick the right x-axis (e.g. non-embedding params), set hyperparameters correctly, and stay in the right regime. The parameter-counting choice above is a perfect example — a "reasonable-looking" decision that quietly bends the whole scaling law. Hold onto this; it's the key to the Kaplan-vs-Chinchilla saga in §7.

### 6.4 Critical batch size (deep — subtle and often misunderstood)

Batch size has strong **diminishing returns** past a point. Increasing the batch lets you take *fewer optimizer steps* to reach a target loss, but the two don't trade off one-for-one forever. The definitive treatment is OpenAI's **"An Empirical Model of Large-Batch Training"** [McCandlish, Kaplan & Amodei 2018] — the paper that introduced both the **critical batch size** and the **gradient noise scale** that predicts it. The **critical batch size** $B_\mathrm{crit}$ is the sweet spot separating two regimes:

* **Below $B_\mathrm{crit}$ — the *noise-limited* (variance-limited) regime.** Your gradient estimate is dominated by *noise*: examples disagree, so every extra example you average in genuinely cancels variance and sharpens the step. You get **near-perfect returns** — doubling the batch roughly halves the steps. This is parallelism-for-free.
* **Above $B_\mathrm{crit}$ — the *bias-limited* regime.** Once the batch is large enough to have driven the *variance* below the gradient's **noise scale**, you're no longer held back by noise but by **bias**: SGD only ever sees the *local* slope, and the local descent direction simply doesn't point exactly at the global optimum. Averaging more examples makes your estimate of the *local* gradient cleaner, but that gradient was already the bottleneck — so extra examples per step barely reduce the step count and you're burning compute for little speedup.

The critical batch is the crossover between these two — the largest batch you can use while still (almost) in the perfect-scaling, noise-limited regime.

**The steps–examples tradeoff.** The recipe: pick a target loss, and for each batch size record **(1) the number of optimizer steps $S$** (the *serial* time — how long you wait) and **(2) the total number of examples $E$** processed (the *compute* — how much you pay) needed to hit it. Sweeping batch size, these trace a clean hyperbola:

$$\Big(\frac{S}{S_\text{min}} - 1\Big)\Big(\frac{E}{E_\text{min}} - 1\Big) = 1,$$

whose two asymptotes are the "minimum possible steps" $S_\text{min}$ (achieved with **huge** batches — you can't go faster in wall-clock even with infinite parallelism) and "minimum possible examples" $E_\text{min}$ (achieved with **tiny** batches — you can't be more data-efficient). Choosing the **knee** of this curve balances the two costs — it lands at roughly **2× the minimum steps and 2× the minimum passes**, i.e. exactly at the critical batch. A tidy consequence: the critical batch is the ratio of the two floors, $B_\mathrm{crit} = E_\text{min}/S_\text{min}$.

**Why $B_\mathrm{crit}$ exists — the gradient noise scale.** The paper's key insight is that $B_\mathrm{crit}$ is set by *how noisy the gradient is*. Define the **gradient noise scale**

$$B_\text{noise} \;\approx\; \frac{\mathrm{tr}(\Sigma)}{|G|^2},$$

the ratio of the **trace of the per-example gradient covariance** $\Sigma$ (how much individual-example gradients scatter) to the **squared norm of the true (average) gradient** $G$ (how strong the real signal is). The intuition is exactly signal-to-noise: if examples disagree wildly ($\mathrm{tr}(\Sigma)$ large) relative to the average direction ($|G|^2$), averaging *more* of them per step genuinely helps, so a big batch pays off; if they already agree, extra examples per step are redundant. The central result unifies the two pictures — **$B_\mathrm{crit} \approx B_\text{noise} = E_\text{min}/S_\text{min}$** — so a quantity you can *measure from gradient statistics* predicts the batch beyond which parallelism stops buying speed.

**It grows during training.** Crucially, $B_\text{noise}$ **increases as loss falls**: early on, gradients are large and agree (small noise scale → small useful batch); late in training the signal shrinks while noise persists (large noise scale → large useful batch). This is *why* late-stage training can profitably use enormous batches, and it motivates **adaptive batch schedules** that grow the batch as training matures.

Kaplan's LLM-specific instantiation (from the 2020 scaling paper) packages this same loss-dependence into a formula — the critical batch depends only on the **loss**, not the model size:

$$B_\mathrm{crit}(L) \;=\; \frac{B_\star}{L^{1/\alpha_B}}, \qquad B_\star \approx 2\times10^8 \text{ tokens},\; \alpha_B \approx 0.21 .$$

Read it plainly: **the lower the loss you're targeting, the larger the batch you can profitably use.** Early in training (high loss) gradients agree across examples, so a modest batch suffices; late in training (low loss) the useful signal per example shrinks, so you need to average more to make progress. This is why late-stage training tolerates — and benefits from — very large batches.

![Two panels. Left, the steps-versus-examples trade-off: a hyperbola with x-axis "total examples E to reach target loss" and y-axis "optimizer steps S to reach target loss", both log scale. Horizontal asymptote labelled S_min (huge batch), vertical asymptote labelled E_min (tiny batch), and a marked knee point labelled "critical batch size, about 2x S_min and 2x E_min". Right, a plot of critical batch size versus target loss (loss decreasing to the right), a rising curve annotated "smaller loss target = bigger usable batch, B_crit = B_star / L^(1/0.21)".](../assets/scale_critical_batch.jpg)

### 6.5 Learning rate and initialization — muP (deep)

If you naively reuse a small model's learning rate on a big model, it's **wrong** — the optimal LR *drifts with scale*. The mental picture (for width scaling): the **bigger the model, the smaller the optimal LR**, because more parameters means you're changing more things at once, so each step should move less. A well-known rule of thumb is to scale the LR like **$1/\text{width}$**. That gives **two philosophies** for setting the big model's LR, and both have been used successfully at scale:

1. **Predict the drift (scaling-law philosophy).** The LR optimum moves *predictably* with scale, so measure how the minimum shifts across small models and extrapolate it to the target size.
2. **Kill the drift (reparametrization philosophy) — µP.** **µP (Maximal Update Parametrization)** [Yang et al 2022] and scale-aware schemes [Yao et al 2024] rescale the initialization and per-layer step sizes so the **optimal LR becomes (nearly) invariant to width**. Then you tune *once* on a small model and **transfer** the same hyperparameters up — "µTransfer" — the small→large extrapolation this whole note is about, applied to the optimizer knobs themselves.

Anecdotally the field leans slightly toward the scaling-law philosophy, but both are viable; the advanced lecture treats them in depth.

### 6.6 Caution: downstream can be less predictable

An honest asterisk [Tay et al 2023]: upstream loss (perplexity) scales cleanly and nearly linearly with parameters, but **downstream task metrics** are far shakier — accuracy tends to follow a **sigmoid** in compute (hence the "emergence" look) and can even *reorder models*. Tay's striking example: on perplexity a model they call **NL12** looks best, but the actually-better downstream model was **NL32XL** — worse in perplexity, better on the task. So upstream→downstream transfer is much less certain than the clean perplexity plots suggest.

The practical philosophy Percy gives: **anchor your scaling law on the low-variance quantity (perplexity)** — it's clean, regular, single-run repeatable — and then *separately* establish that it transfers to the capability you care about. Don't skip that second step: "the perplexity is good, it's your problem now" is how pre-training teams hand a subtly broken model to post-training. Scaling laws are strongest on the smooth quantity (loss) and weakest on the thing you ultimately ship (capabilities).

### 6.7 The surprising takeaway

The effect of **optimizer, depth, architecture** on a huge LM can be **predicted before training it** — by fitting laws on small models and extrapolating. That is a genuinely surprising and enormously money-saving fact.

---

## 7. Joint data–model scaling and compute-optimal training (Chinchilla)

We finally reach question #2 — **more data or a bigger model?** — the most-cited, must-know part of this topic. The tension: pour data into a tiny model and most of it is *wasted* (the model saturates); build a giant model and starve it of data and you've *wasted parameters*. Given a fixed compute budget $C \approx 6ND$, how do you split it between $N$ and $D$?

### 7.1 Joint scaling laws

The trick is a single formula for loss as a function of *both* $N$ and $D$. Two influential forms:

**Rosenfeld et al 2020** — an additive form:

$$L(N,D) \;=\; \frac{A}{N^{\alpha}} \;+\; \frac{B}{D^{\beta}} \;+\; E .$$

Each term is one power law; $E$ is the **irreducible floor** (data entropy). Read it as: your loss is the noise floor $E$, *plus* a penalty for having a finite model ($A/N^\alpha$, shrinks as you grow the model), *plus* a penalty for having finite data ($B/D^\beta$, shrinks as you grow data). Whichever term dominates is your current **bottleneck** — and that's the whole insight, because it tells you where to spend.

**Kaplan et al 2020** used a slightly different combined form,

$$L(N,D) \;=\; \Big[\big(N_c/N\big)^{\alpha_N/\alpha_D} + D_c/D\Big]^{\alpha_D},$$

with individual fits $L(N)=(N_c/N)^{0.076}$ and $L(D)=(D_c/D)^{0.095}$. Both forms fit measured joint error surprisingly well. The point is the same: **fit the exponents on small models / small data, then predict everything else** — including the best split.

### 7.2 The compute-optimal frontier — and the famous disagreement

Minimizing $L(N,D)$ subject to the compute constraint $C \approx 6ND$ gives *optimal* allocations $N_\text{opt} \propto C^{a}$ and $D_\text{opt} \propto C^{b}$. Here two landmark papers **sharply disagreed**:

| | $N_\text{opt}$ | $D_\text{opt}$ | prescription |
|---|---|---|---|
| **Kaplan 2020** | $\propto C^{0.73}$ | $\propto C^{0.27}$ | when compute grows, **grow the model much faster than data**; tokens/param *falls* with scale |
| **Chinchilla 2022** | $\propto C^{0.49\text{–}0.50}$ | $\propto C^{0.50\text{–}0.54}$ | grow model and data **equally** — *double the model ⇒ double the tokens* |

That is a big gap. Kaplan says "mostly make it bigger"; Chinchilla says "grow both in lockstep." Since GPT-3-era models were built on Kaplan's advice, they turned out to be **badly under-trained** — too big for the data they saw.

![A log-log plot of optimal model size N (y) versus compute budget C (x). Two straight lines from a common region diverge as C grows: a steep line "Kaplan: N ~ C^0.73" climbing fast, and a shallower line "Chinchilla: N ~ C^0.50". A shaded wedge between them is labelled "the disagreement", and a marker on the steep line notes "GPT-3 sits here: too big for its data (undertrained)". A small secondary axis note reads "Chinchilla: tokens and params grow together".](../assets/scale_kaplan_vs_chinchilla.jpg)

### 7.3 Chinchilla's three fitting methods

Hoffmann et al didn't trust a single method; they estimated the frontier **three independent ways** and checked they agreed (they did, except a wrinkle in Method 3):

* **Method 1 — minimum over training curves.** Train models of several sizes; each produces a loss-vs-compute curve as it trains. The **lower envelope** (min loss achievable at each compute level, across all runs) is itself a power law. Same spirit as Kaplan's compute-frontier figure.
* **Method 2 — IsoFLOP profiles.** Pick a set of **fixed FLOP budgets** (they used nine, from $6\times10^{18}$ to $3\times10^{21}$). For each budget, train several models of *different sizes* (which forces different token counts, since $C\approx6ND$ is fixed) and plot final loss vs model size. Each budget gives a **U-shaped ("convex") curve** with a clear minimum — too small a model underfits, too big a model sees too few tokens. Fit a parabola, read off the optimum, and the optima across budgets trace a power law. This is the cleanest, most visual method.
* **Method 3 — parametric joint fit.** Fit the full $L(N,D)=E+A/N^\alpha+B/D^\beta$ to *all* runs at once via least-squares (Huber loss, L-BFGS). Their fitted constants:

$$E = 1.69,\quad A = 406.4,\quad B = 410.7,\quad \alpha = 0.34,\quad \beta = 0.28 .$$

Because $\alpha \approx \beta$, the model and data terms shrink at similar rates — which is *why* the optimal split comes out roughly 50/50.

![An IsoFLOP figure. The x-axis is model size N (log scale); the y-axis is final training loss. Several U-shaped curves are drawn, one per fixed FLOP budget, the budgets increasing so the curves stack lower and shift rightward. Each U has its minimum marked with a dot. A dashed line threads through all the minima, and an arrow projects it to a star marked "compute-optimal frontier: N_opt ~ C^0.5". Captions: "too small a model underfits (left arm)", "too big a model is starved of tokens (right arm)".](../assets/scale_isoflops.jpg)

### 7.4 Why did Kaplan and Chinchilla differ?

This is the heart of the lecture's message: **the methods barely differ** — Chinchilla didn't do something obviously smarter that makes you slap your forehead — yet the *predictions* diverge sharply. That's the whole point: scaling laws are **engineered**, sensitive to seemingly minor choices, and you have to **respect the process**, not just turn a crank. Two papers dissect *why*:

**Explanation 1 — "Resolving Discrepancies in Compute-Optimal Scaling" [Porian, Yair et al 2024].** They *replicate Kaplan*, then flip three small things one at a time and watch the curve walk over to Chinchilla:

* **Parameter counting.** Kaplan counted **non-embedding** params and *also dropped the last (softmax) layer*. The input embedding and output softmax are **dual** — shapes vocab×hidden and hidden×vocab — so it felt natural to exclude both. But whether you include the last layer materially changes the scaling law's shape, and at Kaplan's small sizes those params are a big fraction, distorting the small-model end.
* **Learning-rate warmup.** Kaplan's smallest models were so small they **hadn't converged before warmup ended** — their LRs were effectively mis-set, so their final losses were unfairly high.
* **Batch size.** Kaplan fixed one large batch that was **suboptimal for the small models**; tuning batch per model fixes it.

Apply all three and you land essentially on Chinchilla — a chain of "minor" decisions producing the entire gap.

**Explanation 2 — Pearce & Song 2024.** Cleverly, they **train no models at all**: they take Chinchilla's *implied* training curves and simulate what Kaplan *would* have measured under his choices. Their diagnosis is slightly different — it's the **low compute scale Kaplan operated at** (very sensitive to small changes) *plus* the mild nonlinearity introduced by dropping non-embedding params — that explains the disagreement.

The meta-lesson Percy attaches: **scaling laws are lower bounds on a recipe.** They say "if I scale *this exact recipe* up, here's what I get." Scale up a recipe with crazy warmup or a bad fixed batch and you'll fit a real — but pessimistic — law. Stay as close to a proper full run as you can.

### 7.5 Fun addendum: Chinchilla's Method 3 was itself flawed

Method 3 was the one loose thread: it *never quite agreed* with Methods 1 & 2, and the disagreement isn't cosmetic. Methods 1 & 2 give **equal exponents** → a *fixed* tokens-to-params ratio (the famous **20**). Method 3's *unequal* exponents imply that asymptotically you'd want **far more tokens than parameters** — a genuinely different scaling conclusion at large compute. So the mismatch mattered.

The epilogue [Epoch AI — Besiroglu et al 2024]: they couldn't get the raw data or code, so they **extracted the data points from the paper's plots** and **re-ran the Method-3 regression**. The original fit had *underfit* its own data; the corrected fit reaches lower loss and lands **right back at the ~20 ratio**, consistent with Methods 1 & 2. The funny punchline: **the Chinchilla authors were more right than they knew** — a fitting bug was the *only* reason Method 3 ever disagreed.

### 7.6 The catch: train-optimal is *not* deployment-optimal

Chinchilla answers "best model for a fixed **training** compute." But in a real product, **most lifetime compute is spent on inference**, not training. If a model will serve billions of queries, it's worth **"over-training"** a *smaller* model on *far more* tokens than Chinchilla-optimal — you pay more upfront to get a model that's cheaper to run forever. The industry drift is stark:

| Model | tokens / parameter |
|---|---|
| GPT-3 | ~2 |
| Chinchilla | ~20 (the "compute-optimal" number) |
| LLaMA-65B | ~22 |
| Llama 2 70B | ~29 |
| Mistral 7B | ~110 |
| Llama 3 70B | ~215 |

Everyone post-Chinchilla trains *well past* 20 tokens/param, precisely because they're optimizing **total cost including inference**, not training cost alone. The more usage you expect, the more over-training pays off.

![A horizontal bar chart of tokens-per-parameter across model generations, log-scaled x-axis. Bars rise from GPT-3 (~2) and Chinchilla (~20), through LLaMA-65B (~22), Llama 2 70B (~29), Mistral 7B (~110), to Llama 3 70B (~215). A vertical dashed line at 20 is labelled "Chinchilla compute-optimal"; everything to its right is shaded and annotated "over-trained: cheaper inference, worth it at deployment scale".](../assets/scale_tokens_per_param.jpg)

### 7.7 IsoFLOPs everywhere

The IsoFLOP recipe is so clean it now shows up far beyond text LMs — **diffusion models** [Gulrajani+ 2023] and **MoEs** [Abnar+ 2025] among others. The method travels because it needs no functional-form assumption: just sweep sizes at fixed budgets and read the minima.

---

## 8. Recap — surprising and useful

* **Data scaling** is a remarkably clean power law: linear in log-log, holding across domains and models. The theory (mean estimation → nonparametric $n^{-1/d}$ → intrinsic dimension) tells us *why*, and ties the slope to the effective dimensionality of the problem.
* **Model/hyperparameter scaling** lets you **dramatically cut costs**: pick the optimizer, architecture, depth, batch size, and even learning rate for a huge model by fitting laws on small ones and extrapolating — optimizer and architecture only shift the *offset*; the *slope* is set by the problem.
* **Compute-optimal (Chinchilla)** resolves data-vs-size: grow both roughly equally (~20 tokens/param at train-optimal), though real deployments **over-train** because inference dominates lifetime cost.
* **Scaling as prediction** is the deepest lesson: scaling laws tell you *which problems can be brute-forced by scale* — and let you spend a fortune on compute with your eyes open rather than shut.

---

### Sources
- Percy Liang, *CS336 Lecture 9: Scaling Laws — Basics* ([slides](https://github.com/stanford-cs336/lectures/blob/main/lecture_09.pdf), [video](https://www.youtube.com/watch?v=Q15rhEWZPQ4)).
- Hestness et al 2017, *Deep Learning Scaling is Predictable, Empirically* ([arXiv:1712.00409](https://arxiv.org/abs/1712.00409)).
- McCandlish, Kaplan, Amodei et al 2018 (OpenAI), *An Empirical Model of Large-Batch Training* — critical batch size & gradient noise scale ([arXiv:1812.06162](https://arxiv.org/abs/1812.06162)).
- Kaplan et al 2020, *Scaling Laws for Neural Language Models* ([arXiv:2001.08361](https://arxiv.org/abs/2001.08361)).
- Hoffmann et al 2022 (Chinchilla), *Training Compute-Optimal Large Language Models* ([arXiv:2203.15556](https://arxiv.org/abs/2203.15556)).
- Also referenced: Cortes/Vapnik et al 1993 (Bell Labs), Banko & Brill 2001, Kolachina+ 2012, Rosenfeld+ 2020, Bahri+ 2021, Hashimoto 2021, Muennighoff+ 2023 (data repetition), Tay+ 2022/2023 (architecture & downstream), Yang+ 2022 (µP), Yao+ 2024; on the Kaplan–Chinchilla gap: Porian/Yair+ 2024 (*Resolving Discrepancies…*), Pearce & Song 2024, and Besiroglu+ 2024 / Epoch AI (Chinchilla replication).
