# JEPA: Predicting in Representation Space, Not Pixel Space

Most self-supervised learning (SSL) tries to *reconstruct* — a masked-autoencoder paints back the hidden pixels, a language model predicts the next token. That works, but it forces the model to spend capacity modeling detail it can never actually predict: the exact texture of grass, the precise pixels of a shadow, the individual leaves on a tree. **JEPA — the Joint-Embedding Predictive Architecture** — is Yann LeCun's answer to this. Its one-sentence idea: *don't predict the input, predict an abstract representation of the input.* You run the missing part through an encoder to get a compact embedding, and you train a predictor to hit **that** embedding from the visible context. Everything unpredictable at the pixel level is quietly dropped, because a good encoder throws it away.

This small change of target — from pixels to embeddings — is what LeCun bets on as a road toward machines that build **world models**: internal simulators that can predict "what happens next" well enough to *plan*, the way animals and humans do without being told the pixel-exact future. JEPA is the representation-learning core of that larger vision (laid out in his 2022 position paper), and the same predict-in-latent-space trick later powers **V-JEPA 2**, whose action-conditioned variant was dropped zero-shot onto a real robot arm for pick-and-place.

JEPA sits between the two classic SSL families and borrows the best of each. Like **generative** methods (masked autoencoders, diffusion; see the diffusion notes for "predict the pixels/noise") it uses a predictor and a masking task — but it predicts *representations*, so it never pays the pixel-reconstruction tax. Like **contrastive / joint-embedding** methods (see [Contrastive Losses](ContrastiveLosses.md)) it compares embeddings of two views — but it **predicts** one embedding from the other instead of just pulling positives together, and crucially it needs **no negative pairs**. That last freedom comes with the whole difficulty of JEPA: without negatives or pixel targets, what stops the encoder from cheating and outputting a constant? Avoiding that **collapse** is the thread running through every design choice below.

---

## Table of Contents

- [Setup and Notation](#setup-and-notation)
- [The big idea: predict in latent space, not pixel space](#the-big-idea-predict-in-latent-space-not-pixel-space)
- [The energy-based framing](#the-energy-based-framing)
- [The collapse problem — and the two ways out](#the-collapse-problem--and-the-two-ways-out)
- [VICReg: preventing collapse with three regularizers](#vicreg-preventing-collapse-with-three-regularizers)
- [I-JEPA: the first concrete image architecture](#i-jepa-the-first-concrete-image-architecture)
- [V-JEPA and V-JEPA 2: video, and a world model you can plan with](#v-jepa-and-v-jepa-2-video-and-a-world-model-you-can-plan-with)
- [The founding vision: where JEPA sits in LeCun's stack](#the-founding-vision-where-jepa-sits-in-lecuns-stack)
- [Frontier work (brief)](#frontier-work-brief)
- [Takeaways](#takeaways)
- [Sources](#sources)

---

## Setup and Notation

JEPA reuses a small, fixed cast of symbols across all its variants. Defining them once here keeps the rest of the note readable.

| Symbol | Meaning |
|---|---|
| $x$ | the **context** — the visible part of the input (an image with blocks masked out, the first few video frames) |
| $y$ | the **target** — the part we want to predict (the masked blocks, the future frames) |
| $E_\theta$ | the **context encoder** with weights $\theta$; maps $x$ to its representation. Trained by gradient descent. |
| $E_{\bar\theta}$ | the **target encoder** with weights $\bar\theta$; maps $y$ to its representation. Usually an **EMA** (exponential moving average) copy of $E_\theta$ — see below. |
| $s_x = E_\theta(x)$ | the representation (embedding) of the context |
| $s_y = E_{\bar\theta}(y)$ | the representation of the target — **this is the prediction target**, computed in embedding space, not pixels |
| $P_\phi$ | the **predictor** with weights $\phi$; takes $s_x$ (plus some hint of *where* $y$ is) and outputs a guess $\hat s_y$ |
| $\hat s_y = P_\phi(s_x, \cdot)$ | the predicted target representation |
| $z$ | a **latent variable** the predictor may use to absorb the genuinely unpredictable part of $y$ (energy-based framing) |
| $F(x,y)$ | an **energy** — low when $x$ and $y$ are compatible, high when they are not |

Two terms worth pinning down now, because they carry the whole method:

- **EMA (exponential moving average) target encoder.** After each step we don't train $E_{\bar\theta}$ directly; we nudge its weights a tiny fraction toward the context encoder's: $\bar\theta \leftarrow m\,\bar\theta + (1-m)\,\theta$ with momentum $m$ close to $1$ (e.g. $0.996$). The target encoder is thus a slow, smoothed version of the one being trained. This asymmetry — a fast student, a slow target — is one of the main reasons the representations don't collapse.
- **Stop-gradient.** Gradients flow only through the context path ($E_\theta$, $P_\phi$). The target $s_y$ is treated as a fixed label for that step; we never backprop into $E_{\bar\theta}$ to make the target easier to hit. Without this, the cheapest way to lower the loss would be to make *both* sides constant.

The core JEPA loss is then just a distance in representation space:

$$\mathcal{L}(\theta,\phi) = \big\lVert\, \hat s_y - \mathrm{sg}(s_y) \,\big\rVert_2^2 = \big\lVert\, P_\phi\big(E_\theta(x),\cdot\big) - \mathrm{sg}\big(E_{\bar\theta}(y)\big) \,\big\rVert_2^2,$$

where $\mathrm{sg}(\cdot)$ is the stop-gradient. In words: encode the context, predict the target's embedding, and compare it to the embedding the (slow) target encoder actually produces — measuring error **in the embedding space**, never in pixels. Every JEPA variant is a different choice of what $x$ and $y$ are (image blocks, video tubes, states-and-actions) plus a different trick for keeping $s_x, s_y$ from collapsing.

---

## The big idea: predict in latent space, not pixel space

Why is predicting embeddings so much better than predicting pixels? Because **most of the pixel-level detail of the future is irreducibly unpredictable**, and a pixel loss forces the model to care about it anyway.

Picture masking the right half of a photo of a street. A generative model (top-left below) must output *actual pixels* for that half: it has to commit to exact colors, the exact position of every parked car, the texture of the road. But from the left half you simply cannot know those details — so the pixel loss is dominated by noise you can never reduce, and the model burns capacity hedging over textures instead of learning that "this is a street with cars and buildings." That is the pixel-reconstruction tax.

JEPA sidesteps it. Run the right half through an encoder first; the encoder's job is to keep semantic content ("cars here, sky there") and discard the un-guessable pixel noise. Now the predictor only has to match *that* — a much cleaner, lower-dimensional, genuinely predictable target. The model is rewarded for understanding the scene, not for hallucinating plausible textures.

![Three side-by-side schematics of self-supervised learning. Left, 'Generative': input x goes through an encoder to a latent z, then a decoder, producing reconstructed pixels x-hat, with the loss computed in pixel space, which wastes capacity on unpredictable detail. Middle, 'Joint-Embedding / Contrastive': two augmented views x and x-prime each pass through an encoder to embeddings s and s-prime; a double-headed arrow pulls positive pairs together while negatives are pushed apart, requiring many negative pairs. Right, 'JEPA': a context x passes through the trainable encoder E-theta to embedding s_x and then a predictor P-phi to a predicted embedding s-hat_y; separately a target y passes through an EMA target encoder E-bar-theta (marked stop-gradient) to a target embedding s_y; the predicted and target embeddings are matched in representation space.](assets/jepa_arch_comparison.jpg)

The figure makes the three-way contrast concrete. **Generative** predicts pixels and pays the tax. **Contrastive** (like SimCLR; see [Contrastive Losses](ContrastiveLosses.md)) compares two views' embeddings but relies on a large supply of **negative pairs** to keep the space from collapsing — expensive, and sensitive to how you sample negatives. **JEPA** keeps the embedding-space comparison but replaces "pull positives / push negatives" with "**predict** one embedding from the other," using an asymmetric encoder/predictor and an EMA target instead of negatives. It is, in effect, a non-contrastive way to get the benefits of contrastive learning.

---

## The energy-based framing

LeCun likes to describe JEPA in the language of **energy-based models (EBMs)**, and it is worth a paragraph because it explains *why* the architecture looks the way it does.

Think of a scalar **energy** $F(x,y)$ that measures compatibility: it should be **low** when $y$ is a plausible continuation/completion of $x$, and **high** otherwise. Training carves a landscape with valleys at the correct $(x,y)$ pairs. In JEPA the energy is exactly the prediction error in latent space,

$$F(x,y) = \min_{z}\ \big\lVert\, P_\phi\big(E_\theta(x), z\big) - E_{\bar\theta}(y) \,\big\rVert_2^2 .$$

Two things earn their place here. First, the **latent variable $z$**: some parts of $y$ are genuinely undetermined by $x$ (which way the pedestrian turns), and $z$ is a knob the predictor can set to account for that residual ambiguity — the $\min_z$ says "give the model the benefit of the doubt over things it could not have known." When $y$ is fully determined by $x$, $z$ does nothing and we recover the plain squared error from the Setup section. Second, this framing reframes collapse as a **failure of the energy landscape**: if $F$ is low *everywhere* (because the encoder outputs a constant, so every prediction is trivially perfect), the model is useless. Good SSL must push energy **down** on real pairs while keeping it **high** elsewhere — and the whole trick of non-contrastive methods like JEPA is to raise energy off the data *without* explicit negative samples. (For a gentle tutorial, see Dawid & LeCun's "Introduction to Latent Variable Energy-Based Models.")

---

## The collapse problem — and the two ways out

**Collapse** is JEPA's central hazard. The loss is "make $\hat s_y$ close to $s_y$." The laziest possible solution: make the encoder output the *same vector for every input*. Then $s_y$ is constant, the predictor trivially outputs that constant, the loss is zero — and the representations carry no information whatsoever. Nothing in the raw loss forbids this.

![Two scatter plots of a 2-D embedding space. Left, titled 'Collapse: encoder ignores input, all reps to one point, loss zero but useless', shows every point piled into a tiny cluster at the origin. Right, titled 'Healthy: variance kept up, dims decorrelated, reps spread across the space', shows points spread out in a roughly circular Gaussian cloud filling the plane.](assets/jepa_collapse_energy.jpg)

Every JEPA-style method is, at heart, a way to make the collapsed solution unreachable. There are **two families of fixes**, and JEPA's lineage uses them in different mixes:

1. **Architectural / implicit (the I-JEPA & V-JEPA route).** Break the symmetry so the trivial solution isn't a stable fixed point. Concretely: (a) a **stop-gradient** on the target side, so the model can't lower the loss by degrading the target; (b) an **EMA target encoder**, a slowly-moving teacher the fast student chases — a moving target is much harder to collapse onto; and (c) an **asymmetric predictor** sitting only on the context side. This is the same recipe that keeps BYOL and DINO from collapsing, and it works *empirically* without an explicit anti-collapse loss term. The cost is that it's a bit mysterious — you're relying on training dynamics rather than a term you can point to.

2. **Regularization / explicit (the VICReg route).** Add a loss term that *directly* rewards spread-out, informative embeddings — penalize low per-dimension variance and penalize correlations between dimensions. Collapse becomes expensive because a constant embedding has zero variance, which the term punishes hard. This is transparent and easy to reason about, at the price of extra hyperparameters. It's worth understanding on its own because it makes JEPA's whole objective click — so it gets its own section next.

In practice I-JEPA and V-JEPA lean on route 1, but the *reason* route 1 works is best understood through the lens of route 2: both are ensuring the embeddings keep **variance** (don't collapse to a point) and stay **decorrelated** (don't collapse to a line/subspace).

---

## VICReg: preventing collapse with three regularizers

**VICReg** (Variance–Invariance–Covariance Regularization; Bardes, Ponce & LeCun, 2022) is the cleanest, most explicit answer to collapse, and the companion idea that makes JEPA's training objective intuitive. It takes two views' embeddings and adds up **three terms**, each targeting a different way the model could cheat.

![Three panels illustrating VICReg's terms. Left, 'Variance: keep each dim's std above a floor', overlays two histograms of one embedding dimension's values: a narrow red spike at zero labelled 'collapsed (std to 0)' and a broad green bell labelled 'healthy (std at least 1)'. Middle, 'Invariance: two views of the same input match', shows blue circles (view x) and orange squares (view x-prime) connected by short line segments, with the caption 'minimize distance between paired embeddings'. Right, 'Covariance: decorrelate dims (off-diagonals to 0)', shows a 6-by-6 covariance matrix as a heatmap with a bright diagonal and near-zero off-diagonal entries.](assets/jepa_vicreg_terms.jpg)

Let $Z = [z_1,\dots,z_n]$ be the batch of embeddings from one view (each $z_i \in \mathbb{R}^d$), and $Z'$ the embeddings of the matched other view. The three terms:

- **Invariance** — the term that actually does the learning: pull the two views of the *same* input together,
$$s(Z,Z') = \frac{1}{n}\sum_i \lVert z_i - z_i' \rVert_2^2 .$$
On its own this term *is* the collapse trap — its global minimum is "make everything equal." The other two terms exist to fence it off.

- **Variance** — force each embedding dimension to keep spread, so the batch can't shrink to a point:
$$v(Z) = \frac{1}{d}\sum_{j=1}^{d} \max\!\big(0,\ \gamma - \mathrm{std}(Z_{:,j})\big),$$
where $Z_{:,j}$ is coordinate $j$ across the batch, $\mathrm{std}$ is its standard deviation, and $\gamma$ is a target floor (typically $1$). The hinge is the key: as long as a dimension's std stays above $\gamma$ it pays nothing, but the moment a dimension starts collapsing toward constant (std $\to 0$) it gets pushed back up. This directly outlaws the constant-embedding solution.

- **Covariance** — stop a subtler collapse where dimensions stay lively but all encode the *same* thing (embeddings pile onto a line or low-dimensional subspace). Penalize the **off-diagonal** entries of the batch covariance matrix $\mathrm{Cov}(Z)$:
$$c(Z) = \frac{1}{d}\sum_{i \neq j} \big[\mathrm{Cov}(Z)\big]_{ij}^2 .$$
Driving off-diagonals to zero decorrelates the coordinates, so different dimensions are pushed to carry different information — spreading the reps out to *fill* the space rather than clumping onto a subspace.

The full objective is a weighted sum $\lambda\, s + \mu\, [v(Z)+v(Z')] + \nu\, [c(Z)+c(Z')]$. The mental model: **invariance** teaches (match the views), while **variance + covariance together** are exactly "keep the cloud spread out and non-degenerate" — the two failure modes in the collapse figure. This is why VICReg needs no negative pairs and no EMA teacher: the anti-collapse pressure is written explicitly into the loss. And it's why the architectural tricks in I-JEPA/V-JEPA are doing morally the same job, just implicitly through training dynamics.

---

## I-JEPA: the first concrete image architecture

**I-JEPA** ("Image-based JEPA," Assran et al., CVPR 2023) is the first place the abstract idea became a working image model — and it's the one to understand in detail. The task: from **one large context block** of an image, predict the representations of **several smaller target blocks**, entirely in latent space.

![A schematic of I-JEPA. On the left, an image is shown patchified into a 6-by-6 grid. A large blue-shaded region covering the lower-left of the grid is labelled 'context (visible)'; three separate green-outlined rectangular regions are labelled 'targets (masked)'. On the right, a pipeline: the context patches feed the trainable encoder E-theta, whose output feeds a predictor P-phi that also receives mask tokens, producing predicted embeddings s-hat_y. Separately the full-image target patches feed an EMA target encoder E-bar-theta producing target embeddings s_y. A double-headed arrow between s-hat_y and s_y is labelled 'loss in rep space'.](assets/jepa_ijepa_masking.jpg)

Walking the pipeline (right side of the figure):

1. **Patchify and encode the context.** The image is split into patches (it's a Vision Transformer backbone). A single large **context block** is kept; the rest is dropped. Those context patches go through the trainable encoder $E_\theta$ to give context representations $s_x$.
2. **Encode the targets separately, with the EMA encoder.** Independently, the *full* image is run through the **target encoder** $E_{\bar\theta}$ (the EMA copy), and from its patch-level output we read off the representations of **several target blocks** — these are the $s_y$ we want to hit. Computing targets from the full image, at the representation level, is what makes them semantically rich rather than pixel-exact.
3. **Predict, block by block, using positional mask tokens.** For each target block, the predictor $P_\phi$ takes the context representations plus **mask tokens carrying the target's position** (this is the "$\cdot$" hint in $P_\phi(s_x,\cdot)$ — the predictor must be told *where* it's predicting, otherwise "predict some block" is ill-posed). It outputs $\hat s_y$ for that block.
4. **Loss in representation space, stop-grad on targets.** Minimize $\lVert \hat s_y - \mathrm{sg}(s_y)\rVert_2^2$, backprop through $E_\theta$ and $P_\phi$ only; update $\bar\theta$ by EMA. No VICReg term is needed — the stop-grad + EMA + asymmetric predictor combination (route 1 above) is what holds off collapse.

Two design choices carry the results. **Multi-block masking**: the targets are a *few, reasonably large* blocks and the context is one *large* block, deliberately non-overlapping. If the targets were tiny or adjacent to the context, the model could solve them by low-level interpolation (copy nearby texture) instead of semantic reasoning; large, distant blocks force it to predict *content*. **Predicting representations, not pixels**: this is what separates I-JEPA from a masked autoencoder (MAE). Empirically I-JEPA learns strong **semantic** features — competitive linear-probe and low-shot accuracy — and it trains **faster** than pixel-reconstruction MAE, because it never wastes gradient steps on un-guessable pixel detail. That efficiency-plus-semantics combination is the headline evidence for the whole "predict in latent space" thesis.

---

## V-JEPA and V-JEPA 2: video, and a world model you can plan with

Images are static; the real prize is **time**. **V-JEPA** (Bardes et al., 2024) applies the exact same recipe to **video**: mask out spatio-temporal regions ("tubes" spanning several frames), encode the visible part, and predict the masked regions' representations. Video is the ideal testbed for the thesis, because the future is *even more* unpredictable at the pixel level (motion blur, lighting, exact positions) yet highly predictable at the *representation* level ("the ball keeps moving right"). V-JEPA learns motion-aware features from raw unlabeled video that transfer well to action-recognition tasks — without ever reconstructing a frame.

**V-JEPA 2** (Assran et al., 2025) scales this up and, more importantly, turns the learned predictor into a **world model you can plan with**. The key extension is **V-JEPA 2-AC** ("action-conditioned"): the predictor is trained to answer "if the current latent state is $s_t$ and I take action $a$, what's the next latent state $\hat s_{t+1}$?" Trained on unlabeled robot video, this gives a latent-space simulator — and that is exactly the ingredient needed for planning.

![A schematic of the V-JEPA 2-AC planning loop, titled 'plan by rolling the predictor forward in latent space'. Left to right: an 'observation (video frame)' box feeds the encoder E-theta, which produces a 'state rep s_t'; this feeds a 'predictor P-phi(s_t, a)' which outputs a 'predicted s-hat_{t+1}' box below it. A separate 'goal rep s_goal' box sits above the predictor, and a red double-headed arrow between the goal and the prediction is labelled 'cost = norm of (s-hat_{t+1} minus s_goal)'. A curved feedback arrow loops from the predicted next state back to the observation, captioned 'search over actions a (MPC), execute best, re-plan'.](assets/jepa_worldmodel_loop.jpg)

The planning loop (a **model-predictive control / MPC** scheme, entirely in latent space):

1. **Encode** the current observation to a state representation $s_t$, and encode a goal image to a **goal representation** $s_{\text{goal}}$.
2. **Roll the predictor forward** over candidate action sequences: for each candidate $a$, predict $\hat s_{t+1}(a)$ — and chain it several steps ahead to imagine the trajectory, all without touching pixels or a physics simulator.
3. **Score by a cost** = distance in representation space between the imagined outcome and the goal, $\lVert \hat s_{t+1} - s_{\text{goal}}\rVert$. Search (e.g. sampling / optimization) for the action sequence that minimizes it.
4. **Execute** the first action, observe the real result, and **re-plan** — the standard MPC receding-horizon loop.

The striking result: a V-JEPA 2-AC model trained purely on **unlabeled** robot video was deployed **zero-shot** for pick-and-place on a real arm — no task-specific reward, no demonstrations of the target task. That is the concrete payoff of the whole program: self-supervised representation learning yielding a predictor good enough to *plan* against, which is precisely LeCun's argument for why predicting representations (not pixels) is the road to agents that understand and act.

---

## The founding vision: where JEPA sits in LeCun's stack

JEPA didn't start as an architecture paper — it started as one module inside LeCun's 2022 position paper, **"A Path Towards Autonomous Machine Intelligence."** That essay is part manifesto: it sketches a full cognitive architecture for an agent that can perceive, predict, and plan, and it's worth reading first for the *why* even though it reports no experiments.

The proposed agent has a few interacting modules: a **perception** module (encode the world into state), a **world model** (predict how the state evolves, possibly under actions — this is where JEPA lives), a **cost** module (how far is a predicted state from what we want — the thing planning minimizes), an **actor** (search for actions that lower the cost), and a **configurator** that reconfigures the others for the task at hand. Planning is then "imagine trajectories with the world model, pick the ones the cost module likes" — exactly the V-JEPA 2-AC loop above, generalized.

The essay also argues the world model must be **hierarchical** — **H-JEPA**, a stack of JEPAs predicting at multiple time-scales and levels of abstraction. Low levels predict fine, short-horizon detail; high levels predict coarse, long-horizon outcomes ("I will end up in the kitchen") without committing to every intermediate step. This mirrors how humans plan — coarse goals first, details later — and is JEPA's proposed answer to long-horizon prediction, where predicting every fine detail far ahead is both impossible and unnecessary. Much of this remains a research program rather than a finished system, but it's the frame that makes the individual JEPA papers cohere.

---

## Frontier work (brief)

These are newer or more specialized; each is a tight sketch here and can be expanded on request. Context worth knowing: **LeCun left Meta in November 2025 to found AMI Labs**, so the most recent JEPA-lineage work is increasingly happening there rather than at Meta FAIR.

- **LeJEPA** (Balestriero & LeCun, 2025) — a more theoretically grounded successor that tries to *remove the heuristics* (EMA teacher, stop-gradient, hand-tuned tricks) that make current JEPAs finicky. Its pitch is a provable, scalable objective that drives embeddings toward a chosen **isotropic Gaussian** distribution — collapse-free by construction, with fewer moving parts. This is the current theoretical edge of the family.
- **VL-JEPA** — extends the joint-embedding-predictive idea to **vision-language**, predicting across modalities in representation space rather than generating text or pixels.
- **"A Generalization Theory for JEPA-Based World Models"** (2025) — a formal treatment of *when and why* JEPA world models generalize; the place to look if you want the math-side guarantees rather than the intuition.
- **`awesome-jepa`** (AbdelStark, GitHub) — a curated, actively maintained index of the whole JEPA family sorted by domain (image, video, audio, graphs, physics benchmarks). Good as an ongoing reference once the fundamentals click.

---

## Takeaways

- **The one idea:** predict an *abstract representation* of the missing input, not the input itself. A good encoder discards un-guessable pixel detail, so the predictor learns semantics instead of hallucinating textures.
- **JEPA vs the alternatives:** it keeps the masking/prediction setup of **generative** SSL but skips the pixel-reconstruction tax, and keeps the embedding comparison of **contrastive** SSL but needs **no negative pairs**.
- **The whole difficulty is collapse.** Without pixels or negatives, a constant encoder trivially wins. Two fixes: implicit (stop-grad + EMA teacher + asymmetric predictor, used by I-JEPA/V-JEPA) and explicit (**VICReg**'s variance + invariance + covariance terms). Both amount to "keep the embedding cloud spread out and decorrelated."
- **It scales into a world model.** V-JEPA 2-AC turns the predictor into a latent-space simulator you can plan against with MPC — demonstrated zero-shot on a real robot — which is the concrete evidence for LeCun's larger bet.
- **Mental model to carry:** JEPA is the world-model module of LeCun's autonomous-intelligence stack; H-JEPA (hierarchical, multi-scale) is the proposed route to long-horizon planning.

---

## Sources

**Founding vision (start here for the *why*)**
- Yann LeCun, *A Path Towards Autonomous Machine Intelligence* (2022) — the position paper introducing JEPA and the world-model / cost / actor / configurator architecture.

**Core architectures**
- Assran et al., *Self-Supervised Learning from Images with a Joint-Embedding Predictive Architecture* (I-JEPA), CVPR 2023.
- Bardes et al., *V-JEPA: Latent Video Prediction for Visual Representation Learning*, 2024.
- Assran et al., *V-JEPA 2: Self-Supervised Video Models Enable Understanding, Prediction and Planning*, 2025 (includes the V-JEPA 2-AC zero-shot robot result).

**Companion concepts**
- Bardes, Ponce & LeCun, *VICReg: Variance-Invariance-Covariance Regularization for Self-Supervised Learning*, ICLR 2022 — the explicit anti-collapse objective.
- Dawid & LeCun, *Introduction to Latent Variable Energy-Based Models* (2023) — the EBM framing behind JEPA.

**Talks (LeCun explains the intuition better verbally)**
- Lex Fridman Podcast #416 with Yann LeCun (2024).
- *Objective-Driven AI* (2024) and *Self-Supervised Learning, JEPA, World Models, and the Future of AI* (2025) talks.

**Explainers**
- Yannic Kilcher — paper-explained videos on the JEPA position paper and V-JEPA.
- Rohit Bandaru — *Deep Dive into Yann LeCun's JEPA* (covers I-JEPA, V-JEPA, V-JEPA 2, and the energy-based framing in one place).
- `awesome-jepa` (AbdelStark, GitHub) — curated, domain-sorted index of the JEPA family.

**Frontier (2025–2026)**
- Balestriero & LeCun, *LeJEPA: Provable and Scalable Self-Supervised Learning Without the Heuristics* (2025).
- *VL-JEPA* — vision-language extension.
- *A Generalization Theory for JEPA-Based World Models* (2025).

**Related notes in this repo**
- [Contrastive Losses](ContrastiveLosses.md) — contrastive SSL and why negatives are needed there (the thing JEPA avoids).
