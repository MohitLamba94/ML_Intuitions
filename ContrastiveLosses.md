# Contrastive Losses: From Siamese Pairs to CLIP, SigLIP, and Negative-Free Twins

Almost every "learn a good embedding" system — face recognition, image–text models like CLIP, dense text retrievers, self-supervised pretraining — is trained by some flavour of **contrastive learning**. The recipe is always the same one sentence: *embed things into a vector space so that things that belong together land close, and things that do not land far apart.* Everything else — the dozen loss functions, the architectural tricks, the temperature knobs — is just different ways of turning that sentence into a gradient.

This note is a guided tour of the **contrastive loss zoo**. We start from the shared scaffolding they all sit on (the twin-tower / dual-encoder architecture and a similarity function), then walk through the two big families: the older **margin-based** losses from metric learning (pairwise contrastive, triplet, N-pair) and the modern **InfoNCE family** (SimCLR's NT-Xent, MoCo, supervised contrastive, CLIP, SigLIP), which reframes the whole problem as *classification*. We then cover the **negative-free** twin-tower methods (BYOL, SimSiam, Barlow Twins, VICReg) that drop negatives entirely, and close with the cleanest lens for *why* any of this works — the **alignment and uniformity** view — plus a practical cheat-sheet for choosing a loss.

InfoNCE is, quite literally, a cross-entropy loss over similarity scores, so the softmax/cross-entropy prerequisites live in [`cross_entropy_loss.md`](./cross_entropy_loss.md) and are not re-derived here. A close cousin worth flagging: the preference loss in [`diffusion-dpo.md`](./diffusion-dpo.md) is a sigmoid-of-log-ratio objective — contrastive in spirit (prefer the good sample over the bad one), even though it is not usually filed under this heading.

---

## Table of Contents

- [The shared setup: twin towers and a similarity space](#the-shared-setup-twin-towers-and-a-similarity-space)
- [Setup and notation](#setup-and-notation)
- [Two ways to frame the objective](#two-ways-to-frame-the-objective)
- [Margin-based losses: the metric-learning lineage](#margin-based-losses-the-metric-learning-lineage)
  - [Pairwise contrastive loss (Hadsell, Chopra, LeCun, 2006)](#pairwise-contrastive-loss-hadsell-chopra-lecun-2006)
  - [Triplet loss (FaceNet, 2015)](#triplet-loss-facenet-2015)
  - [Using many negatives: N-pair, lifted-structured, multi-similarity](#using-many-negatives-n-pair-lifted-structured-multi-similarity)
- [The InfoNCE family: contrastive learning as classification](#the-infonce-family-contrastive-learning-as-classification)
  - [InfoNCE and CPC (van den Oord et al., 2018)](#infonce-and-cpc-van-den-oord-et-al-2018)
  - [NT-Xent and SimCLR (Chen et al., 2020)](#nt-xent-and-simclr-chen-et-al-2020)
  - [MoCo: a momentum encoder and a queue (He et al., 2020)](#moco-a-momentum-encoder-and-a-queue-he-et-al-2020)
  - [Supervised contrastive loss (Khosla et al., 2020)](#supervised-contrastive-loss-khosla-et-al-2020)
  - [Decoupled contrastive learning (Yeh et al., 2022)](#decoupled-contrastive-learning-yeh-et-al-2022)
- [Multimodal contrastive learning: CLIP and SigLIP](#multimodal-contrastive-learning-clip-and-siglip)
  - [CLIP: symmetric InfoNCE across two modalities (Radford et al., 2021)](#clip-symmetric-infonce-across-two-modalities-radford-et-al-2021)
  - [SigLIP: a sigmoid loss that drops global normalization (Zhai et al., 2023)](#siglip-a-sigmoid-loss-that-drops-global-normalization-zhai-et-al-2023)
- [Contrastive learning for text embeddings](#contrastive-learning-for-text-embeddings)
- [Negative-free twin towers: avoiding collapse without negatives](#negative-free-twin-towers-avoiding-collapse-without-negatives)
- [Why does contrastive learning work? Alignment and uniformity](#why-does-contrastive-learning-work-alignment-and-uniformity)
- [Choosing a loss: a practical map](#choosing-a-loss-a-practical-map)
- [Sources](#sources)

---

## The shared setup: twin towers and a similarity space

Before any loss, you need an architecture that produces two (or more) embeddings you can compare. That architecture is the **twin-tower** (a.k.a. **dual-encoder** or **Siamese**) network: two encoders that map their inputs into the *same* vector space, where a similarity function then scores how close the outputs are.

![A twin-tower diagram: input A (image/query/view 1) and input B (text/doc/view 2) each feed an encoder (f and g). The two encoders may share weights (Siamese) or be separate (asymmetric). Each produces an embedding z_A and z_B, which feed a similarity block sim(z_A, z_B); matching pairs are pulled together and non-matching pairs pushed apart.](./assets/contrastive_twin_tower.jpg){ width=78% }

Two design choices show up everywhere:

- **Shared vs separate weights.** If the two inputs are the same *kind* of thing — two augmented crops of an image, two sentences — the towers usually **share weights** (the classic Siamese network). If the inputs are different *kinds* — an image and a caption, a query and a document — the towers are **asymmetric**, each with its own parameters, because a pixel encoder and a text encoder simply cannot share weights. Either way they must land in one shared space so a single similarity is meaningful.
- **The similarity function.** Almost always **cosine similarity**: L2-normalize each embedding to unit length and take a dot product, $\mathrm{sim}(u,v) = \frac{u^\top v}{\lVert u\rVert\,\lVert v\rVert}$. Normalizing removes vector *magnitude* from the comparison so only *direction* (angle) matters — this stabilizes training and puts every embedding on the surface of a unit hypersphere, a picture we return to at the end. Some margin-based losses instead use raw **Euclidean distance** $d(u,v)=\lVert u-v\rVert$; the two are monotonically related on the unit sphere, so the intuition transfers.

This one architecture underlies a huge range of systems: **Siamese networks** for signature (Bromley et al., 1993) and face verification (Chopra et al., 2005); **DSSM** and later **two-tower retrieval / recommendation** models (Huang et al., 2013); **Dense Passage Retrieval (DPR)** for open-domain QA (Karpukhin et al., 2020); and **CLIP**'s image tower + text tower (Radford et al., 2021). The loss is what differs, and that is the rest of this note.

## Setup and notation

A few symbols recur throughout. Everything is defined here once so later equations read cleanly.

| Symbol | Meaning |
|---|---|
| $f(\cdot)$, $g(\cdot)$ | The two encoders (towers). $f=g$ for weight-shared Siamese networks. |
| $z = f(x)$ | The embedding of input $x$; usually L2-normalized to unit length. |
| $a$ (anchor) | A reference item we compute the loss around. |
| $x^+$ | A **positive** — an item that *should* be close to the anchor (a matching pair, another view, same class). |
| $x^-$ | A **negative** — an item that *should* be far from the anchor. |
| $\mathrm{sim}(u,v)$ | Similarity score, by default cosine similarity $\dfrac{u^\top v}{\lVert u\rVert\,\lVert v\rVert}\in[-1,1]$. |
| $d(u,v)$ | Euclidean distance $\lVert u - v\rVert$ (used by the margin-based losses). |
| $m$ | A **margin**: how much farther a negative must be than a positive before we stop penalizing. |
| $\tau$ | **Temperature**: a positive scalar that sharpens ($\tau\!\downarrow$) or softens ($\tau\!\uparrow$) the similarity distribution. |
| $N$ | Batch size (number of pairs/items per training step). |
| $\sigma(z)$ | Logistic sigmoid $\dfrac{1}{1+e^{-z}}\in(0,1)$, read as a probability. |

## Two ways to frame the objective

Every contrastive loss is one of two ideas in disguise. Holding both in mind makes the whole zoo feel small.

1. **The pull–push (margin) view.** Treat learning as *geometry*: literally pull positive pairs together and push negative pairs apart, but only until they are "far enough" (past a margin). This is the metric-learning lineage — pairwise contrastive and triplet losses. It is intuitive and local (it looks at one pair or one triplet at a time), but it needs careful *mining* of which negatives to look at, and it wastes signal by considering few comparisons at once.

2. **The classification (InfoNCE) view.** Treat learning as a *multiple-choice quiz*: given an anchor, here is the one correct positive hidden among $K$ negatives — score them all and use cross-entropy to make the model assign the highest probability to the positive. This is the modern workhorse. It naturally uses *many* negatives at once (every other item in the batch), which is exactly what the pull–push view struggled to do, and it turns "learn embeddings" into the well-understood machinery of softmax + cross-entropy.

The historical arc of the field is basically the migration from view (1) to view (2), then a set of engineering tricks (bigger batches, memory queues, momentum encoders) to feed view (2) as many negatives as possible — and finally a small backlash (SigLIP, negative-free methods) that questions whether we need the global softmax, or negatives, at all.

## Margin-based losses: the metric-learning lineage

These are the original deep-metric-learning losses. They operate directly on distances with an explicit margin.

### Pairwise contrastive loss (Hadsell, Chopra, LeCun, 2006)

This is the loss that named the field. You feed the network a **pair** of inputs with a binary label $y$: $y=1$ if they are similar (a positive pair), $y=0$ if dissimilar. With $d = d(z_1, z_2)$ the distance between the two embeddings, the loss per pair is

$$
\mathcal{L} = y\, d^2 \;+\; (1-y)\,\big[\max(0,\; m - d)\big]^2 .
$$

Read it term by term:

- **Positive pairs ($y=1$)** keep only $d^2$: the loss shrinks as the pair gets closer, so gradient descent **pulls matching embeddings together**, ideally to distance zero. There is no margin here — positives are pulled all the way in.
- **Negative pairs ($y=0$)** keep only $\big[\max(0, m-d)\big]^2$: this is a **hinge**. If the two are already farther apart than the margin $m$ (so $d \ge m$), then $m - d \le 0$, the $\max$ clips it to $0$, and there is **no loss and no gradient** — the pair is "far enough," leave it alone. Only when a negative is *too close* ($d < m$) does it get pushed outward, with force growing as it gets closer.

That hinge is the crucial idea and it recurs in every margin loss: **do not waste effort pushing apart things that are already far apart.** Without the margin, the model would try to send negatives infinitely far away, which is both unachievable and destabilizing.

![Left: pairwise contrastive loss. An anchor at the center; green positives have arrows pulling them inward to distance zero; red negatives sitting outside a dashed margin circle of radius m are left alone, while any inside would be pushed out. Right: triplet loss. An anchor a, a positive x+ at distance d(a,x+), and a negative that starts close (grey X, "before") and is pushed out past the positive-plus-margin to become the red X ("after"), so that d(a,x-) exceeds d(a,x+) by margin m.](./assets/contrastive_margin_pushpull.jpg){ width=95% }

### Triplet loss (FaceNet, 2015)

The pairwise loss looks at similar and dissimilar pairs *separately*, so it fixes absolute distances (positives → 0, negatives → beyond $m$). But for most tasks we only care about *relative* distances: for face recognition, a photo of me should be closer to another photo of me than to a photo of you — the absolute number does not matter. **Triplet loss** encodes exactly this relative statement using three inputs at once: an anchor $a$, a positive $x^+$ (same identity), and a negative $x^-$ (different identity):

$$
\mathcal{L} = \max\!\big(0,\; d(a,x^+) - d(a,x^-) + m\big).
$$

The bracket says: we are happy as soon as the negative is farther from the anchor than the positive is, **by at least a margin $m$**. If $d(a,x^-) \ge d(a,x^+) + m$, the argument is negative, the $\max$ zeroes it out, and the triplet contributes nothing. Otherwise we pay a penalty that pulls $x^+$ in and pushes $x^-$ out until the margin is satisfied (the right panel above). Because it constrains only the *ordering* $d(a,x^+) < d(a,x^-)$, it is far more flexible than pinning absolute distances.

> **⚠️ Triplet loss lives and dies by negative mining.** Most random triplets already satisfy the margin, so they give zero gradient and waste the step. FaceNet's key practical contribution was **hard/semi-hard negative mining**: within each batch, actively pick negatives that are *almost* as close as the positive (the hard cases) so nearly every triplet produces useful signal. Too-hard negatives (closer than the positive) can destabilize training, so "semi-hard" — negatives farther than the positive but still inside the margin — is the sweet spot. This mining headache is precisely what the InfoNCE family sidesteps by using *all* in-batch negatives at once.

### Using many negatives: N-pair, lifted-structured, multi-similarity

A triplet compares the positive against **one** negative per step, which is statistically weak — the anchor might be pushed away from that one negative while remaining too close to a hundred others. A short lineage of losses generalizes the triplet to **many negatives at once**, and they form the conceptual bridge to InfoNCE:

- **N-pair loss** (Sohn, 2016) compares the anchor's positive against $N-1$ negatives simultaneously using a softmax over their similarities — this is essentially InfoNCE before the name existed. *(This is the most influential of the three; the others below are close relatives worth a sentence each.)*
- **Lifted structured loss** (Oh Song et al., 2016) uses *all* pairwise distances within a batch rather than pre-chosen tuples, so a single mini-batch yields a dense set of comparisons.
- **Multi-similarity loss** (Wang et al., 2019) weights each pair by how informative it is (jointly using self-, positive-, and negative-similarity), a refined form of hard-pair mining.

These are worth knowing as the "use the whole batch" precursors; they can be expanded on request, but the softmax-over-negatives idea they converge on is exactly what the next section formalizes.

## The InfoNCE family: contrastive learning as classification

This family dominates modern self-supervised and multimodal learning. The switch in mindset — from geometry to a classification quiz — is what unlocked scale.

### InfoNCE and CPC (van den Oord et al., 2018)

**InfoNCE** (Info Noise-Contrastive Estimation), introduced in Contrastive Predictive Coding, is the central loss of the whole family. The setup: for an anchor $a$ we have exactly one positive $x^+$ and a set of $K$ negatives $\{x^-_1,\dots,x^-_K\}$. Treat "which of these $K{+}1$ candidates is the true positive?" as a $(K{+}1)$-way classification problem. The loss for that anchor is

$$
\mathcal{L}_{\text{InfoNCE}} = -\log \frac{\exp\!\big(\mathrm{sim}(a, x^+)/\tau\big)}{\exp\!\big(\mathrm{sim}(a, x^+)/\tau\big) + \sum_{k=1}^{K}\exp\!\big(\mathrm{sim}(a, x^-_k)/\tau\big)} .
$$

Unpack it against the classification view:

- The fraction is a **softmax over similarity scores**: the numerator is the positive's score (exponentiated), the denominator sums the positive plus all negatives. So the fraction *is* the model's predicted probability that $x^+$ is the correct match — and $-\log(\cdot)$ of it is exactly **cross-entropy with the positive as the label**. If you are shaky on why $-\log p_{\text{correct}}$ is the right classification loss, see [`cross_entropy_loss.md`](./cross_entropy_loss.md). Minimizing it drives the positive's score up and the negatives' scores down — the same pull–push, now done against all $K$ negatives jointly.
- **The temperature $\tau$** rescales all similarities before the softmax. Small $\tau$ sharpens the distribution, so the loss focuses hard on the most similar (hardest) negatives — the closer a negative is, the more it is penalized. Large $\tau$ softens it, spreading attention evenly. $\tau$ is one of the most impactful hyperparameters in the whole family; typical values are $0.05$–$0.2$.
- **The "NCE" and mutual-information story.** The name comes from Noise-Contrastive Estimation (Gutmann & Hyvärinen, 2010), a trick for learning a distribution by training a classifier to tell real data from noise, avoiding an intractable normalizing constant. Oord et al. showed this InfoNCE loss is a **lower bound on the mutual information** between the anchor and its positive: minimizing it maximizes (a bound on) how much the two views share. That is the theoretical justification for "make views of the same thing agree." You do not need the bound to use the loss, but it is the reason people call these methods *information*-based.

### NT-Xent and SimCLR (Chen et al., 2020)

**SimCLR** made contrastive learning shockingly simple and strong for images, using a specific InfoNCE instance called **NT-Xent** (Normalized Temperature-scaled Cross-Entropy). The pipeline: take an image, apply two random augmentations (crop, color jitter, blur) to get two **views**; these form a positive pair. Push both through a shared encoder (plus a small projection head), L2-normalize, and compute cosine similarities. The two augmented views of the same image are the positive; **every other image's views in the batch are the negatives** — no mining, no separate negative sampler, just "everyone else in the batch."

![A 5x5 in-batch similarity matrix. Rows are anchors a1..a5, columns are candidates x1..x5. The diagonal cells (the true positive for each anchor) are dark/high and outlined in green; off-diagonal cells are the negatives. Caption: each row is a softmax over candidates and the green diagonal cell is the correct one.](./assets/contrastive_infonce_matrix.jpg){ width=52% }

For a positive pair $(i,j)$ (the two views of one image) inside a batch, the NT-Xent loss is

$$
\mathcal{L}_{i,j} = -\log \frac{\exp\!\big(\mathrm{sim}(z_i, z_j)/\tau\big)}{\sum_{k \ne i}\exp\!\big(\mathrm{sim}(z_i, z_k)/\tau\big)} .
$$

This is InfoNCE with a specific choice of negatives: the denominator runs over *all other embeddings in the batch* ($k \ne i$), so a batch of $N$ images gives $2N-2$ negatives per anchor for free. The picture above is literally this — each row of the similarity matrix is one anchor's softmax, and the loss wants the green (diagonal) positive to dominate its row. SimCLR's headline empirical lesson: contrastive learning is **hungry for negatives**, so it wants **large batches** (SimCLR used up to 4096) and strong augmentations. That appetite for negatives is exactly the pressure the next two methods engineer around.

### MoCo: a momentum encoder and a queue (He et al., 2020)

SimCLR ties the number of negatives to the batch size, and batch size is capped by GPU memory. **MoCo** (Momentum Contrast) decouples the two so you can have *many* negatives with a *modest* batch, using two ideas.

![MoCo schematic. A query x_q goes through the query encoder f_q (updated by SGD) to produce q. A key x_k+ goes through a separate momentum encoder f_k to produce k+. The momentum encoder's weights are an EMA (exponential moving average) of the query encoder's weights, shown by a dashed update arrow. A queue / memory bank holds many previously-computed keys k1-..k6- as negatives; new keys are enqueued and old ones dequeued. The InfoNCE loss compares q against k+ and the many k-.](./assets/contrastive_moco_queue.jpg){ width=82% }

- **A queue of negatives (memory bank).** Instead of drawing negatives only from the current batch, MoCo keeps a **queue** of embeddings computed in *recent* batches. Each step enqueues the new keys and dequeues the oldest, so the pool of negatives ($K$ up to tens of thousands) is far larger than the batch. Negatives are cheap because they are precomputed — you just read them from the queue.
- **A momentum key encoder.** But queued negatives were computed by slightly *older* weights, so they are stale. If the encoder changes fast, the queued keys become inconsistent with the current query encoder and the comparison is meaningless. MoCo fixes this by encoding keys with a **momentum encoder** whose weights are an exponential moving average (EMA) of the query encoder: $\theta_k \leftarrow \eta\,\theta_k + (1-\eta)\,\theta_q$ with $\eta$ close to $1$ (e.g. $0.999$). The key encoder therefore evolves *slowly and smoothly*, keeping the whole queue mutually consistent even though it spans many past steps.

The loss itself is plain InfoNCE ($q$ against $k^+$ and the queued $k^-$). **MoCo v2** (Chen et al., 2020) then borrowed SimCLR's tricks (an MLP projection head, stronger augmentation) to close the gap further. The momentum-encoder idea reappears, negatives removed, in BYOL below.

### Supervised contrastive loss (Khosla et al., 2020)

Everything so far is self-supervised: the only "label" is "this view came from the same source." But if you *have* class labels, an anchor has **many** valid positives — every other image of the same class, not just its own augmentation. **Supervised Contrastive (SupCon)** loss uses them: for an anchor, all same-class samples in the batch are positives, everything else is negative. It sums the InfoNCE-style log-probability over that positive *set* (normalizing by the number of positives). Intuitively it pulls each class into a tight cluster while pushing classes apart — often beating plain cross-entropy classification on robustness and transfer, because it shapes the *whole embedding geometry* rather than just carving decision boundaries. It is the natural bridge between self-supervised contrastive learning and ordinary supervised classification.

### Decoupled contrastive learning (Yeh et al., 2022)

A more technical refinement worth a paragraph. Look again at the InfoNCE denominator: it includes the **positive term** alongside the negatives. Yeh et al. show this creates a **negative–positive coupling (NPC)**: a multiplier in the gradient that shrinks the learning signal exactly when a pair is already easy (the positive is already dominant), which throttles learning and makes the method crave large batches. **Decoupled Contrastive Learning (DCL)** simply **removes the positive term from the denominator**, decoupling the two effects. The result trains efficiently with smaller batches and less hyperparameter fuss; InfoNCE and DCL become equivalent as the number of negatives grows, but DCL is the better choice in the finite-batch regime everyone actually trains in. It is a small change to the formula with an outsized effect on training economics.

## Multimodal contrastive learning: CLIP and SigLIP

Contrastive learning's biggest cultural impact came from applying it *across modalities* — aligning images and text in one space.

### CLIP: symmetric InfoNCE across two modalities (Radford et al., 2021)

**CLIP** (Contrastive Language–Image Pre-training) trains an image encoder and a text encoder on ~400M (image, caption) pairs scraped from the web. Within a batch of $N$ pairs, the matched pair $(\text{image}_i, \text{text}_i)$ is the positive; every mismatched combination $(\text{image}_i, \text{text}_j)$ with $i \ne j$ is a negative. Compute the $N\times N$ matrix of cosine similarities and apply InfoNCE **both ways**.

![A 5x5 CLIP similarity matrix with images as rows and texts as columns. The diagonal cells (true image-text pairs) are high and outlined in green; off-diagonal cells are negatives. Caption: symmetric InfoNCE applies a softmax down each column (text-to-image) and across each row (image-to-text); the diagonal holds the true pairs.](./assets/contrastive_clip_matrix.jpg){ width=54% }

The loss is the average of two InfoNCE terms sharing one **learned** temperature $\tau$:

$$
\mathcal{L}_{\text{CLIP}} = \tfrac{1}{2}\big(\mathcal{L}_{\text{image}\to\text{text}} + \mathcal{L}_{\text{text}\to\text{image}}\big).
$$

- **$\mathcal{L}_{\text{image}\to\text{text}}$** treats each *row* of the matrix as a classification: given image $i$, pick its caption among all $N$ texts (softmax across the row, correct answer on the diagonal).
- **$\mathcal{L}_{\text{text}\to\text{image}}$** does the transpose: given text $i$, pick its image among all $N$ images (softmax down the column).
- **Symmetry matters** because retrieval goes both directions (find the caption for an image *and* find the image for a caption); optimizing only one direction leaves the other weak.
- CLIP **learns** $\tau$ (parameterized in log-space and clamped) rather than fixing it, letting the model choose how sharp the contrast should be.

The payoff is **zero-shot** classification: to classify an image, embed candidate labels as text ("a photo of a dog") and pick the nearest — no task-specific head. **ALIGN** (Jia et al., 2021) showed the same recipe scales to over a billion *noisy* pairs, trading data cleanliness for scale — evidence that the contrastive objective is remarkably robust to noise at scale.

### SigLIP: a sigmoid loss that drops global normalization (Zhai et al., 2023)

CLIP's softmax has a hidden cost. Each entry's loss depends on a **normalization over the whole row/column**, so every device in a distributed run must gather all embeddings to compute the denominator — and quality depends on having a *large global batch* of negatives. **SigLIP** asks: do we need the global softmax at all? It replaces it with an independent **sigmoid** (binary) loss on *every* image–text cell.

![Two 5x5 similarity matrices side by side. Left (CLIP softmax): one row is boxed in red, showing that a cell's loss is normalized against the whole row, so it needs a big global batch. Right (SigLIP sigmoid): every cell is boxed independently (diagonal in green, off-diagonal in grey), showing each pair is judged as an independent yes/no label with no global normalization, which is batch-size friendly.](./assets/contrastive_siglip_vs_softmax.jpg){ width=95% }

For each pair $(i,j)$ with label $t_{ij} = +1$ if it is a true match ($i=j$) and $t_{ij}=-1$ otherwise, the loss is a standard logistic term over the scaled similarity $s_{ij}$:

$$
\mathcal{L}_{\text{SigLIP}} = \frac{1}{N}\sum_{i}\sum_{j}\; \log\!\Big(1 + \exp\big(t_{ij}\,(-s_{ij})\big)\Big).
$$

Each term is just "is this pair a match, yes or no?" scored independently — no softmax, no denominator over other pairs, so **no global normalization**. The consequences:

- **It decouples batch size from the loss.** Because every cell is self-contained, you can compute the loss shard-by-shard without gathering the whole batch, cutting memory and communication. SigLIP trains well at small *and* large batches, and famously reaches near-optimal performance around a batch of 32k where CLIP wants far more.
- **It naturally handles the huge positive/negative imbalance** (one positive per row, $N-1$ negatives) since each pair carries its own label; in practice a learnable bias term offsets the flood of negatives.

SigLIP is the current default when you want CLIP-style alignment without the giant-batch tax.

## Contrastive learning for text embeddings

The same machinery powers modern text retrieval and sentence embeddings; the only real question is *where the positive pairs come from*.

- **DPR (Dense Passage Retrieval)** (Karpukhin et al., 2020) trains a question tower and a passage tower with InfoNCE, using the gold passage as the positive and, cleverly, **in-batch negatives plus "hard" BM25 negatives** (passages that look lexically relevant but are wrong) — the hard negatives are what teach the model fine distinctions.
- **SimCSE** (Gao et al., 2021) has a beautiful trick for *unsupervised* sentence embeddings: pass the **same sentence twice** through the encoder with different **dropout masks**. The two dropout-noised outputs form a positive pair (dropout *is* the data augmentation), and other sentences in the batch are negatives. It is SimCLR's idea with dropout standing in for image crops, and it set a strong baseline for sentence similarity.
- Modern retrievers (**E5**, **GTR**, and similar) scale the same contrastive recipe over large mined or synthetic (query, passage) pair corpora; they are refinements of the DPR/SimCSE idea rather than new losses, so they are only flagged here.

## Negative-free twin towers: avoiding collapse without negatives

All of the above need negatives, and negatives are the annoying part — you need lots of them, they need mining or huge batches, and they carry a subtle risk of "false negatives" (two different images that are actually semantically identical). A striking line of work asks: **can twin towers learn good features with positives only?**

The immediate objection is **representation collapse**: if you only ever say "make the two views agree" with no pressure to spread out, the trivial solution is to map *every* input to the *same* constant vector — perfect agreement, zero information. Negatives were what prevented this (they force spreading). Negative-free methods replace that repulsion with an **architectural or statistical trick** instead:

![Four small schematics of negative-free twin towers. BYOL: an online encoder + predictor on view 1 regresses to a target encoder on view 2, where the target is an EMA copy with stop-gradient. SimSiam: the same encoder on both views, a predictor on one side, and a stop-gradient on the other — no EMA. Barlow Twins: two encoders whose output cross-correlation matrix C is pushed toward the identity matrix. VICReg: two encoders trained with three explicit terms — Variance, Invariance, and Covariance.](./assets/contrastive_negfree_tricks.jpg){ width=88% }

- **BYOL** (Grill et al., 2020) — an **online** network (encoder + a small *predictor* head) is trained to predict the output of a **target** network on the other view. The target is an EMA copy of the online network, and its gradient is **stopped**. The asymmetry (only one side has a predictor) plus the slowly-moving stop-gradient target is enough to avoid collapse — surprisingly, with no negatives at all.
- **SimSiam** (Chen & He, 2021) — strips BYOL down: same encoder on both views, a predictor on one side, and a **stop-gradient** on the other. It shows the **stop-gradient alone** prevents collapse — no EMA target, no momentum, no negatives. This is the cleanest existence proof that the stop-gradient is the load-bearing trick.
- **Barlow Twins** (Zbontar et al., 2021) — takes a *statistical* route. Compute the **cross-correlation matrix** between the two views' embedding dimensions across the batch, and push it toward the **identity**: diagonal $\to 1$ (each feature agrees across views = invariance) and off-diagonal $\to 0$ (different features are decorrelated = no redundancy). Decorrelation is what implicitly prevents collapse, no negatives needed.
- **VICReg** (Bardes et al., 2022) — makes the anti-collapse pressure fully **explicit** with three loss terms: **V**ariance (keep each embedding dimension's variance above a floor across the batch — directly forbids collapse to a constant), **I**nvariance (the two views should match), and **C**ovariance (decorrelate dimensions, à la Barlow). It needs neither negatives, nor a momentum encoder, nor a stop-gradient — just three regularizers.

A close relative is **DINO** (Caron et al., 2021), **self-di**stillation with **no** labels: a student network matches a **centered, sharpened** softmax distribution from an EMA teacher over two views. It is a distillation framing rather than a contrastive one, but it belongs in the same negative-free, twin-tower family and produces famously strong Vision-Transformer features (later scaled up as **DINOv2**).

## Why does contrastive learning work? Alignment and uniformity

The single most useful lens for *understanding* the InfoNCE loss comes from **Wang & Isola (2020)**. They prove that, in the limit of infinite negatives, the InfoNCE loss decomposes into two competing forces on the unit hypersphere:

![Two unit circles. Left, "Alignment": green positive pairs sit as nearby points on the circle, connected by short double-headed arrows, illustrating that positive pairs map to nearby points. Right, "Uniformity": blue points are spread evenly all the way around the circle, illustrating that features spread evenly over the sphere. Title: what the InfoNCE loss optimizes (Wang & Isola, 2020).](./assets/contrastive_align_uniform.jpg){ width=88% }

- **Alignment** — positive pairs should map to **nearby** points. This comes from the numerator of InfoNCE (maximize the positive's similarity). It is the "make agreeing things agree" force.
- **Uniformity** — the embeddings should be **spread out**, covering the hypersphere as evenly as possible, preserving maximal information. This comes from the denominator (push negatives apart). It is precisely the anti-collapse force.

The whole InfoNCE objective is a **tug-of-war between these two**: alignment alone collapses everything to a point (perfectly aligned, useless); uniformity alone scatters everything with no structure. Good representations balance the two. This reframing is clarifying because it names what negative-free methods are secretly doing: BYOL's predictor, Barlow's decorrelation, and VICReg's variance term are all **substitute sources of uniformity** — different mechanisms to spread features out once you have removed the negatives that used to do it. Wang & Isola even show you can *directly* optimize alignment + uniformity as two explicit losses and match contrastive performance, which is strong evidence that these two properties are what actually matter.

## Choosing a loss: a practical map

The table condenses the tour into a decision aid. Rows are ordered roughly by historical arrival.

| Loss / method | Needs negatives? | Frame | Big-batch / memory demand | Typical use |
|---|---|---|---|---|
| Pairwise contrastive (2006) | Yes (pairs) | Pull–push margin | Low | Verification, classic metric learning |
| Triplet (2015) | Yes (1 per step) + mining | Pull–push margin | Low, but mining-heavy | Face recognition, ranking |
| N-pair / lifted / multi-sim (2016–19) | Yes (many) | Softmax over negatives | Medium | Deep metric learning |
| InfoNCE / NT-Xent / SimCLR (2018–20) | Yes (in-batch, many) | Classification | **High** (wants huge batches) | Self-supervised vision, the default |
| MoCo (2020) | Yes (queue) | Classification | Medium (queue, not batch) | Self-supervised with limited memory |
| SupCon (2020) | Yes + labels | Classification | High | Supervised, robust/transferable features |
| CLIP (2021) | Yes (in-batch, both ways) | Symmetric classification | **Very high** | Image–text alignment, zero-shot |
| SigLIP (2023) | Yes, but per-pair | Independent sigmoid | **Low–medium** (no global softmax) | Image–text, batch-size friendly |
| DPR / SimCSE / E5 (2020–21) | Yes (in-batch + hard) | Classification | Medium–high | Text retrieval, sentence embeddings |
| BYOL / SimSiam / Barlow / VICReg (2020–22) | **No** | Agreement + anti-collapse trick | Low–medium | Self-supervised without negatives |

Two rules of thumb to leave with. **If you have negatives and compute, InfoNCE-family losses are the strong default** — and reach for MoCo (queue) or SigLIP (sigmoid) when a giant global batch is impractical. **If negatives are awkward** — hard to define, expensive to mine, or plagued by false negatives — a negative-free method (SimSiam is the simplest, VICReg the most explicit) is a genuinely competitive alternative. And whichever you pick, the alignment/uniformity lens is the one mental model that explains what the gradients are really doing.

## Sources

- **Signature Verification using a Siamese Time Delay Neural Network** — Bromley et al., 1993. Origin of the Siamese twin-tower architecture.
- **Learning a Similarity Metric Discriminatively, with Application to Face Verification** — Chopra, Hadsell, LeCun, 2005. Siamese networks for verification.
- **Dimensionality Reduction by Learning an Invariant Mapping** — Hadsell, Chopra, LeCun, 2006. The original pairwise contrastive loss with margin.
- **Distance Metric Learning for Large Margin Nearest Neighbor Classification (LMNN)** — Weinberger & Saul, 2009. Margin-based metric-learning precursor to triplet loss.
- **Noise-Contrastive Estimation** — Gutmann & Hyvärinen, 2010. The NCE trick underlying InfoNCE.
- **Learning Deep Structured Semantic Models (DSSM)** — Huang et al., 2013. Early two-tower retrieval.
- **FaceNet: A Unified Embedding for Face Recognition and Clustering** — Schroff et al., 2015. Triplet loss + semi-hard negative mining.
- **Deep Metric Learning via Lifted Structured Feature Embedding** — Oh Song et al., 2016. All-pairs batch loss.
- **Improved Deep Metric Learning with Multi-class N-pair Loss Objective** — Sohn, 2016. Triplet generalized to many negatives (proto-InfoNCE).
- **Representation Learning with Contrastive Predictive Coding (CPC)** — van den Oord, Li, Vinyals, 2018. Introduces InfoNCE and the mutual-information view.
- **Multi-Similarity Loss for Deep Metric Learning** — Wang et al., 2019. Informativeness-weighted pair mining.
- **A Simple Framework for Contrastive Learning of Visual Representations (SimCLR)** — Chen et al., 2020. NT-Xent, augmentations, large batches.
- **Momentum Contrast for Unsupervised Visual Representation Learning (MoCo)** — He et al., 2020; **MoCo v2** — Chen et al., 2020. Momentum encoder + queue of negatives.
- **Supervised Contrastive Learning (SupCon)** — Khosla et al., 2020. InfoNCE with label-defined positive sets.
- **Dense Passage Retrieval (DPR)** — Karpukhin et al., 2020. Dual-encoder QA retrieval with in-batch + hard negatives.
- **Understanding Contrastive Representation Learning through Alignment and Uniformity on the Hypersphere** — Wang & Isola, 2020. The alignment/uniformity decomposition.
- **Bootstrap Your Own Latent (BYOL)** — Grill et al., 2020. Negative-free with predictor + EMA target.
- **Exploring Simple Siamese Representation Learning (SimSiam)** — Chen & He, 2021. Stop-gradient alone prevents collapse.
- **Barlow Twins: Self-Supervised Learning via Redundancy Reduction** — Zbontar et al., 2021. Cross-correlation → identity.
- **Learning Transferable Visual Models From Natural Language Supervision (CLIP)** — Radford et al., 2021. Symmetric image–text InfoNCE, zero-shot.
- **Scaling Up Visual and Vision-Language Representation Learning With Noisy Text Supervision (ALIGN)** — Jia et al., 2021. Billion-scale noisy CLIP.
- **SimCSE: Simple Contrastive Learning of Sentence Embeddings** — Gao et al., 2021. Dropout as augmentation for sentence embeddings.
- **Emerging Properties in Self-Supervised Vision Transformers (DINO)** — Caron et al., 2021; **DINOv2** — Oquab et al., 2023. Self-distillation, no labels/negatives.
- **VICReg: Variance-Invariance-Covariance Regularization** — Bardes, Ponce, LeCun, 2022. Explicit anti-collapse regularizers.
- **Decoupled Contrastive Learning (DCL)** — Yeh et al., 2022. Removes the positive from the InfoNCE denominator (NPC effect).
- **Sigmoid Loss for Language Image Pre-Training (SigLIP)** — Zhai et al., 2023. Per-pair sigmoid loss, no global softmax.
