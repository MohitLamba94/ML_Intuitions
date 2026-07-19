# Normalisation Layers: Keeping Activations Well-Behaved Inside an LLM

A transformer is a tall stack of residual blocks, and each block adds its output back onto a running "residual stream" of activations. Left unchecked, the numbers flowing through that stream tend to **drift and grow** as you go deeper — one layer's slightly-too-large output feeds the next, which amplifies it again, until gradients either explode or vanish and training falls apart. **Normalisation layers are the cheap, boring fix that makes deep networks trainable at all:** at chosen points they rescale the activations back to a sane, roughly unit-scale range, so every layer sees inputs of a predictable magnitude no matter how deep it sits.

This note is about the normalisation choices that modern LLMs have converged on, and *why*. We start with **LayerNorm** and its lean successor **RMSNorm** — the two you actually see in today's models — and explain why RMSNorm won on wall-clock speed even though it barely changes the FLOP count (this is where the memory-bandwidth story and Table 1 of the "Data Movement" paper come in). Then we look at two techniques that were *dominant* in the convolutional era but have all but vanished from LLMs — **BatchNorm** and **Dropout** — and unpack the specific failures that killed them. Finally we tackle the *placement* question — **Pre-Norm vs Post-Norm** — and why every large model now puts the norm *before* the sublayer.

---

## Table of Contents

- [Setup and Notation](#setup-and-notation)
- [Why normalise at all](#why-normalise-at-all)
- [LayerNorm](#layernorm)
- [RMSNorm: drop what you do not need](#rmsnorm-drop-what-you-do-not-need)
- [Why the speedup is real: norms are memory-bound](#why-the-speedup-is-real-norms-are-memory-bound)
- [BatchNorm and why it died in transformers](#batchnorm-and-why-it-died-in-transformers)
- [Why Dropout also faded](#why-dropout-also-faded)
- [Pre-Norm vs Post-Norm](#pre-norm-vs-post-norm)
- [Takeaways](#takeaways)
- [Sources](#sources)

---

## Setup and Notation

A handful of symbols recur throughout; each is reintroduced where it first does real work.

| Symbol | Meaning |
|---|---|
| $x \in \mathbb{R}^d$ | The activation vector for **one token at one layer** — the thing a norm operates on. |
| $d$ | The model's hidden dimension (the number of features in $x$, e.g. 4096). |
| $x_i$ | The $i$-th feature (coordinate) of $x$, for $i = 1 \dots d$. |
| $\mu, \sigma$ | The **mean** and **standard deviation** of $x$'s features, $\mu = \tfrac1d\sum_i x_i$ and $\sigma = \sqrt{\tfrac1d\sum_i (x_i-\mu)^2}$. |
| $g$ | A learnable **gain** (scale) vector, $g \in \mathbb{R}^d$, applied element-wise. Also called $\gamma$. |
| $b$ | A learnable **bias** (shift) vector, $b \in \mathbb{R}^d$. Also called $\beta$. |
| $\epsilon$ | A tiny constant (e.g. $10^{-5}$) added under the square root to avoid dividing by zero. |
| $\odot$ | Element-wise (Hadamard) product. |
| $L$ | The number of layers (depth) of the transformer. |

A recurring theme: an activation tensor in a transformer has (at least) three axes — **batch** (which example), **sequence** (which token), and **feature** (which of the $d$ coordinates). *Which axis a norm computes its statistics over* is the single most important thing distinguishing the methods below.

---

## Why normalise at all

The original pitch for normalisation was "internal covariate shift": as the weights of early layers change during training, the *distribution* of activations they feed to later layers keeps shifting, so later layers are forever chasing a moving target. Normalising the activations to a fixed mean/variance was supposed to pin that distribution down. That story is intuitive but, honestly, only half-right — later work (Santurkar et al., 2018) showed the bigger benefit is that normalisation **smooths the loss landscape**, making gradients more predictable in scale so you can use larger learning rates without diverging. For our purposes the practical takeaway is enough: **normalisation keeps activations at a stable, roughly unit scale, which keeps gradients stable, which lets deep networks train.**

Every norm we discuss has the same two-part shape. First a **standardisation** step: subtract some center and divide by some spread, so the result has a controlled scale. Then an **affine restore** step: multiply by a learnable gain $g$ and add a learnable bias $b$. Why the second step? Forcing every activation to exactly zero-mean, unit-variance is a straitjacket — it throws away scale information the network might actually want. The learnable $g$ and $b$ let the model *undo* the normalisation if that turns out to be useful, so we get the optimisation benefits without permanently crippling what the layer can represent. The methods differ only in **what statistics they use to standardise** and **over which axis they compute them**.

---

## LayerNorm

LayerNorm standardises each token's activation vector using statistics computed **over that vector's own features**:

$$
\mathrm{LayerNorm}(x) = g \odot \frac{x - \mu}{\sqrt{\sigma^2 + \epsilon}} + b,
\qquad
\mu = \frac1d\sum_{i=1}^{d} x_i,
\qquad
\sigma^2 = \frac1d\sum_{i=1}^{d} (x_i - \mu)^2.
$$

Reading it term by term: $\mu$ is the average of the $d$ features of this one token, and subtracting it **re-centers** the vector to zero mean. Dividing by $\sqrt{\sigma^2+\epsilon}$ **re-scales** it to unit variance (the $\epsilon$ just guards the division). At this point every token's vector has mean 0 and variance 1 regardless of what it started as. Then $g$ and $b$ — both length-$d$ and learned — restore a per-feature scale and shift so the model isn't locked into the standardised distribution.

The crucial property is *which axis $\mu$ and $\sigma$ are taken over*: **only the feature axis of a single token.** No other token, and no other example in the batch, enters the calculation. That makes LayerNorm **completely independent of batch size and of the other tokens** — a token is normalised the same whether it arrives alone or in a batch of 1024, at training time or when generating one token at a time. Keep this in mind; it is exactly the property BatchNorm lacks, and it is why LayerNorm was the natural fit for sequence models. LayerNorm gives the network two invariances: it is unaffected if you add a constant to all features (**re-centering** invariance, thanks to subtracting $\mu$) or scale them all (**re-scaling** invariance, thanks to dividing by $\sigma$).

---

## RMSNorm: drop what you do not need

RMSNorm (Zhang & Sennrich, 2019) asks a sharp question: of LayerNorm's two invariances, **do we actually need the re-centering?** Their finding is that we mostly don't — the benefit comes overwhelmingly from **re-scaling** (keeping the vector's magnitude in check), and the mean-subtraction step contributes little. So RMSNorm throws it out and normalises by the **root-mean-square** of the features instead of the standard deviation:

$$
\mathrm{RMSNorm}(x) = g \odot \frac{x}{\mathrm{RMS}(x)},
\qquad
\mathrm{RMS}(x) = \sqrt{\frac1d\sum_{i=1}^{d} x_i^2 + \epsilon}.
$$

The difference from LayerNorm is exactly two deletions. First, there is **no $\mu$**: we divide the raw $x$ by its RMS rather than by the std of the centered $x$ (note $\mathrm{RMS}(x)$ equals $\sigma$ only if the mean is already zero, so this genuinely skips a step, it isn't just algebra). Second, there is **no bias $b$** — RMSNorm keeps only the learnable gain $g$. The figure below traces both pipelines; the greyed, crossed-out boxes are precisely what RMSNorm removes.

![Two horizontal op-flow diagrams. Top row labelled LayerNorm: input x, then a 'subtract mean' box, a 'divide by std' box, a 'scale x g' box, and a 'shift + b' box, connected left to right by arrows. Bottom row labelled RMSNorm: input x, then a greyed-out 'subtract mean' box crossed out in orange with a dashed 'skip mean' arrow curving over it, then a 'divide by RMS' box, a 'scale x g' box, and a greyed-out crossed-out 'shift + b' box. The figure shows that RMSNorm keeps only the re-scaling and gain steps, dropping the mean subtraction and the bias.](../assets/norm_ln_vs_rmsnorm.jpg)

Why this became the modern default:

- **Fewer operations.** No mean to compute, and no subtraction of it across all $d$ features. You still do one reduction (the sum of squares) instead of two (a sum for $\mu$, then a sum of squared deviations), so there's genuinely less work per token.
- **Fewer parameters.** Dropping $b$ removes a whole length-$d$ vector per norm — and a transformer has many norms — so there are fewer numbers to store, load, and move around.
- **Just as good in practice.** Empirically RMSNorm matches LayerNorm's final quality on the tasks that matter, so the removed pieces really were close to dead weight. This is the modern, deflationary explanation for RMSNorm: not that it's cleverer, but that it's **leaner and just as good**.

There's a subtlety worth stating plainly, because it motivates the next section. **RMSNorm barely reduces the FLOP count** — the arithmetic of a norm is trivial next to the giant matrix multiplies around it. Its real payoff is **wall-clock time**: fewer element-wise passes over the activation tensor and fewer parameters to shuttle between memory and the compute units. To see why that saves real time, we need to understand *why norms are slow in the first place* — which has nothing to do with FLOPs. This same logic ("the params/compute tradeoff for bias terms is poor") is why people increasingly drop bias terms from linear layers too, not just from norms.

---

## Why the speedup is real: norms are memory-bound

A GPU has two very different budgets: how fast it can do arithmetic (**compute**, FLOP/s) and how fast it can move data to and from memory (**memory bandwidth**, bytes/s). An operation is **compute-bound** if arithmetic is the bottleneck, and **memory-bound** if the bottleneck is simply reading the inputs and writing the outputs. A norm reads the whole activation tensor, does a pinch of arithmetic per element, and writes the whole tensor back — so it moves a lot of data to do very little math. It is memory-bound.

Ivanov et al. (2020), *"Data Movement Is All You Need,"* measured exactly this for a BERT training iteration by grouping every operator into three classes and reporting each class's share of the total FLOPs versus its share of the wall-clock runtime. (We use only this one table from that paper; its other contributions — operator fusion and so on — are out of scope here.)

| Operator class | % of FLOPs | % of runtime |
|---|---|---|
| Tensor contraction (matmuls) | 99.80 | 61.0 |
| Statistical normalization (softmax, LayerNorm) | 0.17 | 25.5 |
| Element-wise (activations, dropout, bias-add) | 0.03 | 13.5 |

![Grouped bar chart with three operator classes on the x-axis: tensor contraction, statistical normalization, and element-wise. For each class two bars: percent of FLOPs (blue) and percent of runtime (orange). Tensor contraction is 99.8 percent of FLOPs but only 61 percent of runtime. Statistical normalization is 0.17 percent of FLOPs but 25.5 percent of runtime. Element-wise is 0.03 percent of FLOPs but 13.5 percent of runtime. A callout on the normalization bars reads 'tiny compute, large wall-clock cost'. The title notes these operators are memory-bandwidth bound, not compute bound.](../assets/norm_flop_vs_runtime.jpg)

Look at the normalisation row: **0.17% of the arithmetic, but 25.5% of the runtime.** The matmuls do essentially all the math yet take only 61% of the time; roughly 37% of a training step is spent in these memory-bound normalisation and element-wise operators that contribute almost nothing to the FLOP count. This is the whole justification for caring about RMSNorm's leanness. Because a norm's cost is dominated by *touching memory*, not by *computing*, shaving off an extra reduction (the mean) and an extra parameter tensor (the bias $b$) — each of which is another pass over or another read of the activation-sized data — directly cuts the memory traffic, and therefore the wall-clock time, even though the FLOP savings round to zero. RMSNorm's reported **7%–64% runtime reduction** over LayerNorm (varying by model) comes from exactly this ledger, not from doing fundamentally less arithmetic.

---

## BatchNorm and why it died in transformers

In the convolutional-network era, **BatchNorm** was king — nearly every CNN used it. It also standardises to zero mean and unit variance, but over the *opposite* axis: for each feature, it computes the mean and variance **across all the examples in the mini-batch**.

$$
\mathrm{BatchNorm}(x_i) = g_i \, \frac{x_i - \mu^{\text{batch}}_i}{\sqrt{(\sigma^{\text{batch}}_i)^2 + \epsilon}} + b_i,
$$

where $\mu^{\text{batch}}_i$ and $\sigma^{\text{batch}}_i$ are the mean and std of feature $i$ **taken over the batch** (and, for sequences, over the token positions too). Because you can't compute a "batch" statistic when generating one example at a time, BatchNorm keeps a **running average** of these statistics during training and uses those frozen population estimates at inference. The contrast with LayerNorm is the whole story, and the figure makes it visual: LayerNorm reduces along a *row* (one token, all its features); BatchNorm reduces along a *column* (one feature, all samples in the batch).

![Two grids side by side, each with rows labelled 'batch (samples / tokens)' and columns labelled 'feature dimension i'. Left grid, titled LayerNorm / RMSNorm, highlights a single horizontal row in blue, captioned 'normalise across all features of one sample (a row) — batch-independent'. Right grid, titled BatchNorm, highlights a single vertical column in orange, captioned 'normalise one feature across the whole batch (a column) — couples samples'. The figure shows the two methods reduce over orthogonal axes.](../assets/norm_axes_ln_vs_bn.jpg)

That "column" dependence is fatal for language models. To see *why* — and this is worth dwelling on, because it's a nice case study in how the same tool can be great in one domain and broken in another — it helps to recall where BatchNorm came from and then look at what Shen et al. (2020) actually measured when they tried to force it into transformers.

### The context: BatchNorm was built for vision

BatchNorm (Ioffe & Szegedy, 2015) was one of the enabling tricks of the deep-CNN era. Nearly every landmark vision architecture — ResNet, MobileNet, DenseNet — is built with it, because it reliably reduces sensitivity to learning rate and initialisation and smooths the loss landscape, letting people train much deeper networks than before. So when the field moved to NLP and transformers, the obvious question was: why not just use the tool that already works? And the answer, discovered empirically, was that the transformer community quietly standardised on **LayerNorm** instead — *all* transformer variants use LN, not BN — but for years without a crisp explanation of *why* BN should fail here. Shen et al. set out to answer exactly that.

### The problem: swapping BN into a transformer just breaks it

The starting observation is blunt: take a standard transformer, replace its LayerNorms with BatchNorms, and translation quality drops hard — about **1.1 BLEU on IWSLT14 and 1.4 BLEU on WMT14**. Those are large margins in machine translation, and they're not a tuning artefact; BN is simply the wrong tool here. The interesting scientific question is what, specifically, goes wrong.

### The diagnosis: batch statistics on text are wildly unstable

Shen et al. ran a controlled comparison — **ResNet20 on CIFAR-10** (a domain where BN thrives) against a **transformer with BN on IWSLT14** (where it fails) — and tracked the batch statistics through training. Recall that BN faces a built-in tension: during training it normalises using the *current mini-batch's* mean and variance $(\mu_B, \sigma_B)$, but at inference it must use a *running average* of those statistics accumulated over training. This only works if the per-batch statistics are stable enough that their running average is a faithful stand-in. Two measurements told the story:

- **Forward statistics.** They measured the distance between each batch's statistics $(\mu_B, \sigma_B)$ and the running averages. For ResNet on images, the batches hug the running average closely — the statistics are stable and settle down as training proceeds. For the transformer on text, the batch statistics show **very high variance with extreme outliers that never go away**, even late in training. So the running average BN would use at inference is a poor summary of what any given batch actually looks like — the training-time and inference-time normalisations are *inconsistent*.
- **Backward statistics.** They also looked at the gradients BN produces for its mean and variance terms ($g_\mu$ and $g_{\sigma^2}$). On images these gradient outliers vanish early in training; on text they **persist as large outliers throughout**, injecting noise into the updates.

Why is text so different? Image activations across a batch are fairly homogeneous, but a random batch of *text* is not: topics, lengths, rare tokens, and occasional huge outlier activations vary drastically from batch to batch. So the batch statistics genuinely jump around, the running average chases a moving target and never settles, and normalising by it adds noise instead of removing it. That instability, in both the forward statistics and their gradients, is the concrete disease behind the BLEU drop.

### The fix (and why it confirms the diagnosis)

PowerNorm's design is essentially a list of the unstable pieces, removed. Its first change (**PN-V**) is to **relax the zero-mean requirement** and replace the variance with the **quadratic mean** (root-mean-square) — the very quantity RMSNorm also uses. The motivating measurement is that while $\mu_B$ and $\sigma_B$ swing by orders of magnitude across batches, the quadratic mean is *far* more stable. Dropping the mean also removes its troublesome gradient term, cutting the number of volatile statistics from four to two. Its second change is to use **running statistics in the forward pass** (not just at inference) of that quadratic mean, with an approximate-backprop scheme so the backward pass stays tractable and bounded. That PowerNorm has to strip out precisely the mean, the variance, and their gradients to make batch-style normalisation work in a transformer is the clearest possible confirmation that **batch-statistic instability was the root cause**.

### Two more nails in the coffin

Even setting stability aside, BN's column dependence is awkward for sequence models in ways LayerNorm never is. Because a token's normalised value depends on the *other examples in its batch*, BN behaves differently at different batch sizes, gets noisy for small batches, and is fiddly around variable-length sequences and padding. And most damning for LLMs specifically: **autoregressive generation emits one token at a time**, so there is no meaningful batch to average over and you're forced onto the frozen (and, as we saw, unreliable) running statistics. LayerNorm and RMSNorm dodge every one of these problems by never looking outside the single token — which is why the field simply switched to them and left BN in the CNN era.

---

## Why Dropout also faded

**Dropout** — randomly zeroing a fraction of activations during training to prevent overfitting — was the other CNN-era staple that has largely disappeared from LLM pretraining. There are two reasons, and the first is a genuinely interesting interaction with BatchNorm dissected by Li et al. (2019), *"Understanding the Disharmony Between Dropout and Batch Normalization by Variance Shift."*

### The context: two great tricks that hurt when combined

The puzzle Li et al. start from is that Dropout and BatchNorm are each hugely effective on their own, yet **stacking them usually makes things *worse*.** This wasn't new folklore — Ioffe & Szegedy had already noticed it when they introduced BN, and conjectured that BN provides a regularising effect of its own that overlaps with Dropout's. The practical evidence was everywhere: the strongest vision architectures of the era (ResNet, ResNeXt, DenseNet, Wide ResNet) got their best numbers using **BN with no Dropout at all**, effectively abandoning what had been the default regulariser. But nobody had a precise account of *why* the combination misfires — and, more tellingly, why one architecture (Wide ResNet) is the odd exception that actually *benefits* from using both. A good explanation has to cover both the failures and that exception.

### The mechanism: a train/test variance shift

Their answer is a **variance shift**. Dropout's standard "inverted" form doesn't just zero units; to keep the expected activation unchanged it *scales up* the survivors, which **inflates the variance** of each neuron's output during training (dropping with keep-probability $p$ blows the variance up by roughly $1/p$). BatchNorm, sitting downstream, dutifully accumulates its running variance from these inflated, dropout-perturbed activations. Then at **test time dropout is turned off**: no zeroing, no scaling, so the activation variance snaps back down to its true, un-inflated value — while BN keeps normalising with the *inflated* variance it memorised during training. BN now expects one variance and receives another; that mismatch is the "variance shift," and it propagates through the network, producing numerically unstable, more error-prone predictions. This is what their Figure 1 shows: a visible gap between BN's frozen moving-average variance and the real test-time variance.

This same lens explains the Wide ResNet exception. Li et al. show the shift ratio drifts back toward 1 (i.e. the disharmony vanishes) in two regimes: as you stop dropping (keep-probability $\to 1$), and as the **feature width $d \to \infty$**. Wide ResNet's whole idea is very wide layers, so its large $d$ naturally damps the variance shift — which is exactly why it, uniquely, tolerates Dropout-plus-BN. Their proposed fixes follow the same logic: either apply Dropout **only after the last BN** (so no BN downstream can be poisoned by the shift), or use a **variance-stable Dropout variant** ("Uout") that perturbs activations with bounded uniform noise instead of hard zeroing, whose shift ratio approaches 1 far faster. Both recover most of the lost accuracy.

### The bigger reason for LLMs: nothing to regularise against

The variance-shift story explains why you can't casually mix Dropout with BN, but the dominant reason Dropout vanished from large-scale LLM *pretraining* is more mundane: **there's almost nothing to overfit to.** Dropout fights overfitting, and overfitting requires the model to see the same examples many times. Frontier LLMs are trained on enormous corpora for roughly a single pass, so a given example is essentially never seen twice — memorisation isn't the failure mode, and a regulariser aimed at it just costs compute and throughput for no return. So modern pretraining typically sets dropout to zero (it can reappear during fine-tuning on small datasets, where overfitting is real again). Between the BN interaction that made it fragile and the single-epoch regime that made it pointless, Dropout quietly left the LLM stack.

---

## Pre-Norm vs Post-Norm

Once you've settled on *which* norm to use, one placement question remains: relative to a sublayer (attention or FFN) wrapped in a residual connection, does the norm go **after** the residual addition or **before** the sublayer? This is the subject of Xiong et al. (2020), *"On Layer Normalization in the Transformer Architecture,"* and its answer reshaped how everyone builds transformers.

Writing $\text{Sublayer}(\cdot)$ for an attention or feed-forward block, the two designs are:

$$
\textbf{Post-LN:}\quad x_{\text{out}} = \mathrm{LayerNorm}\big(x + \text{Sublayer}(x)\big)
$$

$$
\textbf{Pre-LN:}\quad x_{\text{out}} = x + \text{Sublayer}\big(\mathrm{LayerNorm}(x)\big)
$$

In **Post-LN** — the design of the original 2017 Transformer — the sublayer acts on the raw $x$, the result is added back, and *then* the sum is normalised. The norm sits **on** the residual highway, so every layer re-normalises the accumulated stream. In **Pre-LN** — used by GPT-2 and essentially every large model since — the norm is moved *inside* the branch: you normalise a copy of $x$, feed it to the sublayer, and add the result back onto the **un-normalised** $x$. Crucially, this leaves a clean, unnormalised identity path running straight from input to output (plus one final LayerNorm before the output head). Panel (a) below is Post-LN, panel (b) is Pre-LN.

![Two side-by-side transformer block diagrams. Panel (a), Post-LN: the input x_l goes into Multi-Head Attention, is added back, then passes through a Layer Norm; that feeds an FFN, is added back, then a second Layer Norm produces x_{l+1}. The Layer Norm boxes sit directly on the main vertical residual path. Panel (b), Pre-LN: the input x_l first passes through a Layer Norm, then Multi-Head Attention, then is added back onto the residual path; separately a Layer Norm then FFN then addition; the Layer Norm boxes sit inside the branches while the residual path runs straight through, with a final Layer Norm before the output.](../assets/xiong_prenorm_postnorm_x1.jpg)

*Figure from Xiong et al. (2020), "On Layer Normalization in the Transformer Architecture" (arXiv:2002.04745). Reproduced for educational purposes.*

**Why the placement matters: gradients at initialisation.** The paper's key contribution is showing, both theoretically and empirically, that Post-LN and Pre-LN have very different gradient behaviour *at the start of training*. For Post-LN, at initialisation the gradients flowing into the parameters **near the output layer are large**, and — this is the important part — their scale **does not shrink as you add depth**. Their result for the last feed-forward block's weights is:

$$
\textbf{Post-LN:}\quad \Big\| \frac{\partial \tilde{\mathcal{L}}}{\partial W^{2,L}} \Big\|_F = \mathcal{O}\!\left(d\sqrt{\ln d}\right)
\qquad
\textbf{Pre-LN:}\quad \Big\| \frac{\partial \tilde{\mathcal{L}}}{\partial W^{2,L}} \Big\|_F = \mathcal{O}\!\left(d\sqrt{\tfrac{\ln d}{L}}\right)
$$

Here $\|\cdot\|_F$ is the Frobenius norm (an overall magnitude of the gradient matrix), $\tilde{\mathcal{L}}$ is the loss, $W^{2,L}$ is the second FFN weight matrix in the final ($L$-th) layer, and $L$ is the depth. Read off the difference: the **Post-LN** gradient scale is **independent of $L$**, so it stays large no matter how deep the network — whereas the **Pre-LN** gradient carries a $1/\sqrt{L}$ factor, so it *shrinks* as the model gets deeper. The mechanism behind this is that the LayerNorm sitting *on* the residual path in Post-LN has a Jacobian that scales like $\|J_{\mathrm{LN}}(x)\|_2 = \mathcal{O}(\sqrt{d}/\|x\|_2)$; when the norm is on the highway (Post-LN) it repeatedly rescales the backward signal in a way that leaves large gradients near the output, while Pre-LN's clean identity path lets gradients pass through with a magnitude that is naturally tempered by depth. (We state the results rather than derive them; the derivation is in the paper.)

**Why this makes warm-up necessary — or not.** A big gradient at initialisation is dangerous: apply a normal-sized learning rate to it and the first few updates overshoot and destabilise training. This is the real reason the original Transformer needed **learning-rate warm-up** — starting from a tiny learning rate and ramping it up over thousands of steps — which is precisely a way to survive those large early Post-LN gradients until the network settles. Without warm-up, Post-LN training is fragile: the paper reports it collapsing to a dismal 8.45 BLEU on IWSLT14 where a warmed-up run reaches ~34. Pre-LN's gradients are already well-scaled at initialisation, so **warm-up can simply be removed**, eliminating a finicky hyperparameter (the warm-up length and peak learning rate) that used to require careful tuning.

**The empirical payoff.** With warm-up gone, Pre-LN also just trains *faster and more robustly*: the paper shows Pre-LN reaching the same validation loss in noticeably fewer steps (e.g. on BERT pre-training, hitting a target loss roughly 40% sooner), with less sensitivity to the learning rate. Fewer hyperparameters, faster convergence, more stability — especially as models get very deep — is why **every large modern LLM uses Pre-Norm** (paired with RMSNorm). The one caveat worth knowing for informed conversation: a *carefully tuned* Post-LN can sometimes reach slightly better final quality (Pre-LN's always-open identity path can let deep layers contribute less — a "representation collapse" effect), but at the scale and depth of real LLMs the training stability of Pre-Norm decisively wins, which is why it's the default.

---

## Takeaways

- **Normalisation exists to keep activations at a stable scale** so gradients stay well-behaved and deep residual stacks are trainable; the learnable gain/bias then restore any expressiveness the standardisation removed.
- **LayerNorm normalises per token, across features** — completely independent of batch size and of other tokens — which is exactly what sequence models need.
- **RMSNorm = LayerNorm minus the mean and minus the bias.** Re-centering turns out to be dispensable; keeping only re-scaling matches quality with fewer ops and fewer parameters.
- **RMSNorm's win is wall-clock, not FLOPs.** Norms are memory-bandwidth bound: ~0.17% of a BERT iteration's FLOPs but ~25% of its runtime, so cutting memory passes (an extra reduction, an extra parameter tensor) directly cuts time — hence RMSNorm's 7–64% speedups.
- **BatchNorm died in transformers** because it couples examples in a batch (bad for variable-length sequences and one-at-a-time generation) and because NLP batch statistics fluctuate wildly during training, making its running averages unreliable (PowerNorm's diagnosis).
- **Dropout faded** partly because of its variance-shift disharmony with BatchNorm, but mainly because single-epoch pretraining on massive corpora barely overfits, so the regulariser buys nothing.
- **Pre-Norm beat Post-Norm.** Post-LN has large, depth-independent gradients at init that force learning-rate warm-up; Pre-LN's identity path yields $1/\sqrt{L}$-tempered gradients, so warm-up can be dropped and training is faster and more stable. Modern LLMs use Pre-Norm + RMSNorm.

---

## Sources

- Ba, Kiros & Hinton (2016), [*Layer Normalization*](https://arxiv.org/abs/1607.06450) (the original LayerNorm).
- Zhang & Sennrich (2019), [*Root Mean Square Layer Normalization*](https://arxiv.org/abs/1910.07467) (RMSNorm; re-centering is dispensable, 7–64% runtime reduction).
- Ioffe & Szegedy (2015), [*Batch Normalization*](https://arxiv.org/abs/1502.03167) (the original BatchNorm).
- Shen et al. (2020), [*PowerNorm: Rethinking Batch Normalization in Transformers*](https://arxiv.org/abs/2003.07845) (why batch statistics are unstable for NLP; ICML 2020).
- Li et al. (2019), [*Understanding the Disharmony Between Dropout and Batch Normalization by Variance Shift*](https://arxiv.org/abs/1801.05134) (CVPR 2019).
- Xiong et al. (2020), [*On Layer Normalization in the Transformer Architecture*](https://arxiv.org/abs/2002.04745) (Pre-LN vs Post-LN gradients and warm-up; source of the reproduced figure).
- Ivanov et al. (2020), [*Data Movement Is All You Need: A Case Study on Optimizing Transformers*](https://arxiv.org/abs/2007.00072) (Table 1: operator FLOP vs runtime proportions; norms are memory-bound).
- Santurkar et al. (2018), [*How Does Batch Normalization Help Optimization?*](https://arxiv.org/abs/1805.11604) (normalisation smooths the loss landscape).
