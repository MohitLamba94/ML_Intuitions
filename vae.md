# Variational Autoencoders (VAE), the ELBO, and Discrete VAEs

A **Variational Autoencoder** looks, at a glance, like an ordinary autoencoder — squeeze the data through a bottleneck and reconstruct it — but it is really a **generative model** in disguise. The one change that makes all the difference is that a VAE encodes each input not to a single point but to a *little cloud* (a probability distribution) in latent space, and it is trained so that all those clouds pile up into one clean, known distribution we can sample from. That is what turns a mere compressor into something that can *dream up new data*.

This note builds the VAE from the ground up: why a plain autoencoder cannot generate, how the **ELBO** (Evidence Lower BOund) turns "maximize the likelihood of the data" into a trainable loss, and the **reparameterization trick** that lets gradients flow through a random sampling step. It then covers the **discrete VAE** line — **VQ-VAE** and **DALL-E's dVAE** — where the latent code is a grid of discrete tokens rather than a continuous vector.

It is written as a **precursor to `ddpm_ddim_flow_score.md`**. Two payoffs land at the end and answer a question worth holding onto the whole way through: *why do diffusion models lean on a VAE rather than a normal autoencoder?* The short version — a diffusion model **is** a VAE with a fixed encoder (which is why an ELBO shows up in its training), and latent diffusion (Stable Diffusion) literally diffuses inside a **VAE's** latent space because a plain autoencoder's latent space is too ill-behaved to add noise to. Both are made precise in [§8](#8-why-diffusion-uses-a-vae-not-a-plain-autoencoder).

---

## Table of Contents

- [1. Setup and Notation](#1-setup-and-notation)
- [2. Why not a plain autoencoder?](#2-why-not-a-plain-autoencoder)
- [3. The VAE idea: encode to a distribution](#3-the-vae-idea-encode-to-a-distribution)
- [4. The ELBO, derived](#4-the-elbo-derived)
  - [4.1 The problem: an intractable likelihood](#41-the-problem-an-intractable-likelihood)
  - [4.2 Introduce the encoder and split the log-likelihood](#42-introduce-the-encoder-and-split-the-log-likelihood)
  - [4.3 The two terms, unpacked](#43-the-two-terms-unpacked)
- [5. The reparameterization trick](#5-the-reparameterization-trick)
- [6. Putting it together: loss, training, generation](#6-putting-it-together-loss-training-generation)
  - [6.1 Two failure modes: posterior collapse and the blur, and β-VAE](#61-two-failure-modes-posterior-collapse-and-the-blur-and-β-vae)
- [7. Discrete VAEs](#7-discrete-vaes)
  - [7.1 VQ-VAE: snap the latent to a codebook](#71-vq-vae-snap-the-latent-to-a-codebook)
  - [7.2 DALL-E's discrete VAE (Gumbel-softmax)](#72-dall-es-discrete-vae-gumbel-softmax)
  - [7.3 VQ-VAE-2 and VQGAN, briefly](#73-vq-vae-2-and-vqgan-briefly)
- [8. Why diffusion uses a VAE, not a plain autoencoder](#8-why-diffusion-uses-a-vae-not-a-plain-autoencoder)
  - [8.1 DDPM is a hierarchical VAE with a fixed encoder](#81-ddpm-is-a-hierarchical-vae-with-a-fixed-encoder)
  - [8.2 Latent diffusion needs a well-behaved latent](#82-latent-diffusion-needs-a-well-behaved-latent)
- [9. Summary](#9-summary)
- [Sources](#sources)

---

## 1. Setup and Notation

A generative model has one job: describe the process that could have produced the data, so that we can then run that process to make *new* data. The VAE tells the following story. There is a hidden, low-dimensional **latent** $z$ drawn from a simple **prior**, and each data point $x$ (an image, say) is produced by feeding some $z$ through a decoder. To generate, sample $z$ from the prior and decode; to learn, figure out which decoder makes real data likely. These are the recurring symbols:

| Symbol | Meaning |
|---|---|
| $x$ | A data point — e.g. an image. Lives in a high-dimensional space (all pixels). |
| $z$ | The **latent** code — a low-dimensional vector summarizing $x$ (e.g. 32 or 256 numbers). |
| $p(z)$ | The **prior** over latents. We *choose* it to be simple: a standard Gaussian $\mathcal{N}(0, I)$. |
| $p_\theta(x \mid z)$ | The **decoder** (a.k.a. likelihood / generator): a neural net with parameters $\theta$ that turns a latent $z$ into a distribution over data. |
| $q_\phi(z \mid x)$ | The **encoder** (a.k.a. approximate posterior / inference network): a neural net with parameters $\phi$ that turns a data point $x$ into a distribution over latents. |
| $p_\theta(z \mid x)$ | The **true posterior** — what the ideal encoder *should* output, but which is intractable to compute. $q_\phi$ is our learned stand-in for it. |
| $\theta, \phi$ | Decoder and encoder parameters. Trained jointly. |
| $\mathrm{KL}(q \,\|\, p)$ | Kullback–Leibler divergence: how far distribution $q$ is from $p$. Always $\ge 0$, and $=0$ only when $q=p$. |

Two conventions worth stating once. The prior is fixed to $\mathcal{N}(0, I)$ not because the world is Gaussian but because it is the simplest thing we know how to sample and the decoder is powerful enough to warp that plain Gaussian into whatever shape the data needs. And "posterior" here just means "the distribution over latent $z$ *given* an observed $x$" — the reverse direction from the decoder's "$x$ given $z$." (If terms like *marginal*, *posterior*, or *likelihood* feel shaky, the companion note `ddpm_ddim_flow_score.md` has a short [Probability Primer](./ddpm_ddim_flow_score.md#extra-1--a-probability-primer) built from scratch; this note is self-contained but that is the deeper reference.)

---

## 2. Why not a plain autoencoder?

Start with what a VAE is *not*. An ordinary **autoencoder** (AE) is a pair of networks: an encoder that maps $x$ to a single latent point $z = f(x)$, and a decoder that maps back $\hat{x} = g(z)$. It is trained to minimize reconstruction error alone,

$$
\mathcal{L}_{\text{AE}} = \| x - g(f(x)) \|^2 ,
$$

which reads "push $x$ through the bottleneck and come back to yourself." This is excellent for *compression* and for learning useful features — the bottleneck forces the network to keep only what matters. But it is **useless as a generator**, and it is worth being precise about why, because the reason is exactly what the VAE fixes.

The problem is that the AE loss says **nothing about the shape of the latent space**. The encoder is free to place each training point's code wherever is most convenient for reconstruction. In practice it flings different clusters of data off to far-apart, arbitrarily-scaled islands with large empty gaps between them (left panel below). Now try to *generate*: you would need to pick a new $z$ and decode it. But which $z$? There is no distribution to draw from — you do not know the scale, and if you guess (say sample a Gaussian), you land in one of the empty gaps between islands, a region the decoder was **never trained on**, and out comes garbage. An AE learns a code for every point it *has seen* but leaves the space *between and around* those points undefined.

![Left: a plain autoencoder latent space with five color-coded clusters flung to far-apart islands at arbitrary scale (roughly plus or minus 10) with large empty holes between them; a star marks a random draw that lands in an empty gap and is annotated as decoding to garbage. Right: a VAE latent space where all five classes overlap and pile up against the unit Gaussian prior (dashed 1- and 2-sigma circles around the origin), filling the space with no holes; a star marks a draw from N(0,I) that lands on data and decodes to a valid sample.](./assets/vae_ae_vs_vae_latent.jpg){ width=100% }

*The whole motivation in one picture. **Left:** a plain autoencoder is free to scatter codes anywhere — isolated clusters, arbitrary scale, big holes — so a random $z$ has nowhere sensible to land. **Right:** a VAE adds a term that pulls every code toward a fixed prior $\mathcal{N}(0,I)$, so codes overlap and fill the space with no holes, and sampling the prior lands on data.*

So the fix we need is a *pressure that packs all the codes into one known, gap-free distribution* — one we can then sample from. That pressure is the KL term of the VAE, and the "encode to a distribution" idea is what makes that pressure meaningful.

---

## 3. The VAE idea: encode to a distribution

The VAE makes two changes to the autoencoder, and they work together.

**Change 1 — encode to a distribution, not a point.** Instead of the encoder emitting one latent vector, it emits the parameters of a *Gaussian* over latents:

$$
q_\phi(z \mid x) = \mathcal{N}\big(z;\ \mu_\phi(x),\ \mathrm{diag}(\sigma^2_\phi(x))\big).
$$

Read it literally: the encoder network looks at $x$ and outputs two vectors — a **mean** $\mu_\phi(x)$ (where this input's code sits) and a **spread** $\sigma_\phi(x)$ (how much wiggle room it gets). The actual latent used is then *sampled* from this little Gaussian cloud. Why a cloud instead of a point? Because a point can be tucked into an infinitely thin gap between other points, but a cloud has width — forcing every input to occupy a fuzzy blob means nearby inputs' blobs must **overlap**, and overlapping blobs tile the space continuously instead of leaving holes. The spread is what glues the latent space together.

**Change 2 — pull every cloud toward the prior.** On its own, the encoder would cheat: it would shrink $\sigma_\phi(x) \to 0$ (a cloud of zero width is just a point again) and drift the means far apart, reverting to the useless AE. So we add a term that penalizes each cloud for straying from the prior $\mathcal{N}(0,I)$, measured by $\mathrm{KL}\big(q_\phi(z\mid x)\,\|\,p(z)\big)$. This term *wants* every per-input Gaussian to be the unit Gaussian centered at the origin. It fights the reconstruction term (which wants distinct, confident codes), and the balance they strike is a latent space where codes are as informative as possible **while still collectively filling $\mathcal{N}(0,I)$**. That is exactly the samplable space the right panel of the figure above showed.

The generative recipe then falls out for free: to make a new data point, **sample $z \sim \mathcal{N}(0,I)$ from the prior and run it through the decoder** $p_\theta(x\mid z)$. No encoder needed at generation time — it was only ever scaffolding to shape the latent space during training.

![Schematic of a VAE: an input x feeds an encoder q_phi(z given x), which splits into two heads producing mu_phi(x) and sigma_phi(x); these combine with an external noise draw epsilon from N(0,I) as z = mu + sigma elementwise-times epsilon at a sample node; z feeds a decoder p_theta(x given z) producing a reconstruction x-hat. The KL term under the encoder pulls q_phi toward the prior N(0,I); the reconstruction term sits under the decoder.](./assets/vae_architecture.jpg){ width=100% }

*The VAE as a computation graph. The encoder emits a mean and spread; a latent is sampled from that Gaussian (using the external $\epsilon$ explained in [§5](#5-the-reparameterization-trick)); the decoder reconstructs. Training balances the two labeled terms — reconstruction pulls one way, the KL-to-prior the other.*

Everything so far has been intuition. The next section shows that these two terms are not a heuristic bolt-on — they are exactly what you get when you honestly try to *maximize the likelihood of the data* and hit a wall.

---

## 4. The ELBO, derived

### 4.1 The problem: an intractable likelihood

The principled goal of any generative model is **maximum likelihood**: choose $\theta$ so the model assigns high probability to the real data. For one data point that means maximizing $\log p_\theta(x)$, where the model's probability of $x$ is obtained by considering every latent that could have produced it and adding up their contributions:

$$
p_\theta(x) = \int p_\theta(x \mid z)\, p(z)\, dz .
$$

This is the **marginal likelihood**: for each possible latent $z$, take the prior probability $p(z)$ of drawing it and the decoder's probability $p_\theta(x\mid z)$ of producing $x$ from it, multiply, and integrate over *all* $z$. The trouble is that integral. It ranges over the entire high-dimensional latent space, and $p_\theta(x\mid z)$ is a deep neural network, so there is no closed form and no cheap numerical estimate — for almost every $z$ the decoder produces something nothing like $x$, so the integrand is essentially zero except in a tiny region you do not know in advance. We **cannot compute $\log p_\theta(x)$, so we cannot maximize it directly.**

### 4.2 Introduce the encoder and split the log-likelihood

Here is where the encoder $q_\phi(z\mid x)$ earns its keep. It is our guess at *which $z$ actually explain this particular $x$* — precisely the tiny region where the integrand is not zero. The variational trick is to use $q_\phi$ to rewrite $\log p_\theta(x)$ exactly, with no approximation yet. Start from the fact that $\log p_\theta(x)$ does not depend on $z$, so taking an expectation over $z \sim q_\phi(z\mid x)$ leaves it unchanged, then insert the definition of conditional probability and multiply inside the log by $q_\phi/q_\phi$:

$$
\log p_\theta(x)
= \underbrace{\mathbb{E}_{q_\phi(z\mid x)}\!\left[\log \frac{p_\theta(x, z)}{q_\phi(z\mid x)}\right]}_{\displaystyle \mathcal{L}(\theta,\phi;\,x)\ =\ \text{ELBO}}
\;+\;
\underbrace{\mathrm{KL}\big(q_\phi(z\mid x)\,\|\,p_\theta(z\mid x)\big)}_{\displaystyle \ge\, 0}.
$$

(The one-line algebra: $\log p_\theta(x) = \mathbb{E}_{q_\phi}[\log p_\theta(x)] = \mathbb{E}_{q_\phi}\big[\log \tfrac{p_\theta(x,z)}{p_\theta(z\mid x)}\big] = \mathbb{E}_{q_\phi}\big[\log \tfrac{p_\theta(x,z)}{q_\phi(z\mid x)}\big] + \mathbb{E}_{q_\phi}\big[\log \tfrac{q_\phi(z\mid x)}{p_\theta(z\mid x)}\big]$, and the last expectation is the KL by definition.)

Now read the split. On the left is the thing we *want* but cannot compute. On the right, the second piece is a KL divergence, and **a KL is never negative**. So dropping it can only *decrease* the right side, which gives the central inequality:

$$
\log p_\theta(x) \;\ge\; \mathcal{L}(\theta,\phi;\,x) \;=\; \mathbb{E}_{q_\phi(z\mid x)}\!\left[\log \frac{p_\theta(x, z)}{q_\phi(z\mid x)}\right].
$$

This lower bound $\mathcal{L}$ is the **ELBO — Evidence Lower BOund** ("evidence" is a classical name for the marginal likelihood $p_\theta(x)$). Two things make it the right object to optimize. First, unlike the intractable integral, the ELBO is an **expectation over the encoder's own samples**, which we can estimate by simply drawing a few $z \sim q_\phi(z\mid x)$ and averaging — cheap and differentiable. Second, because it sits *below* the log-likelihood, **pushing the ELBO up pushes the log-likelihood up with it**; and the slack between them is exactly $\mathrm{KL}(q_\phi \,\|\, p_\theta(z\mid x))$, so maximizing the ELBO over $\phi$ *also* drives the encoder $q_\phi$ toward the true posterior. One objective quietly does two good things at once — trains a decent generator *and* a decent encoder.

### 4.3 The two terms, unpacked

The ELBO in the form above is correct but opaque. Expand the joint $p_\theta(x,z) = p_\theta(x\mid z)\,p(z)$ and split the log:

$$
\mathcal{L}(\theta,\phi;\,x)
= \underbrace{\mathbb{E}_{q_\phi(z\mid x)}\big[\log p_\theta(x\mid z)\big]}_{\textbf{reconstruction}}
\;-\;
\underbrace{\mathrm{KL}\big(q_\phi(z\mid x)\,\|\,p(z)\big)}_{\textbf{regularizer}} .
$$

This is the loss the VAE actually optimizes (maximize $\mathcal{L}$, or equivalently minimize $-\mathcal{L}$), and each term is exactly one of the two intuitions from [§3](#3-the-vae-idea-encode-to-a-distribution):

- **Reconstruction, $\mathbb{E}_{q_\phi}[\log p_\theta(x\mid z)]$.** Encode $x$ to its cloud, sample a $z$, and ask how much probability the decoder assigns to the *original* $x$ from that $z$. Maximizing it means "codes must decode back to me." If the decoder models pixels as Gaussians with fixed variance, $\log p_\theta(x\mid z)$ is just $-\tfrac{1}{2}\|x - \hat{x}(z)\|^2$ up to a constant — i.e. the familiar **squared reconstruction error**, now derived rather than assumed.
- **Regularizer, $\mathrm{KL}(q_\phi(z\mid x)\,\|\,p(z))$.** How far this input's Gaussian cloud is from the prior $\mathcal{N}(0,I)$. Subtracting it *penalizes* clouds that drift off or shrink to points, which is the exact pressure that packs all codes into a samplable $\mathcal{N}(0,I)$.

The two terms pull against each other — perfect reconstruction wants sharp, far-flung, near-deterministic codes; a perfect KL wants every code to be the same standard Gaussian blob (carrying zero information about $x$). The VAE lives at their compromise, and that compromise *is* the well-behaved latent space. Because both distributions are Gaussians, the KL even has a **closed form** — for a diagonal $q_\phi = \mathcal{N}(\mu, \mathrm{diag}(\sigma^2))$ in $D$ latent dimensions against $p=\mathcal{N}(0,I)$,

$$
\mathrm{KL}\big(q_\phi(z\mid x)\,\|\,p(z)\big) = \frac{1}{2}\sum_{j=1}^{D}\Big(\sigma_j^2 + \mu_j^2 - 1 - \log \sigma_j^2\Big),
$$

so no sampling is needed for this term at all: feed the encoder's $\mu$ and $\sigma$ straight into this formula. (Reading it: $\mu_j^2$ punishes drifting off-center, $\sigma_j^2 - 1 - \log\sigma_j^2$ is minimized at $\sigma_j = 1$ and punishes both collapsing to a point and ballooning too wide — together they say "be the unit Gaussian.") Only the reconstruction term still needs a sampled $z$, and sampling is exactly where the next problem hides.

---

## 5. The reparameterization trick

There is a snag that would sink the whole scheme if left alone. To estimate the reconstruction term we must **sample** $z \sim q_\phi(z\mid x)$ and then backpropagate the reconstruction error into the encoder parameters $\phi$. But *sampling is not a differentiable operation*: if $z$ pops out of a random number generator whose distribution depends on $\mu_\phi, \sigma_\phi$, there is no derivative $\partial z / \partial \phi$ to hand to backprop. The gradient path from the loss back to the encoder is severed (left panel below).

![Left panel titled naive: sample inside the graph — a box phi feeds a box z drawn from N(mu_phi, sigma_phi squared), which feeds a loss box; a dashed backward arrow from loss is blocked by a red cross at the random node, annotated random node has no usable gradient. Right panel titled reparameterized — a box phi and a separate box epsilon drawn from N(0,I) both feed a deterministic box z = mu_phi + sigma_phi elementwise-times epsilon, which feeds loss; a solid backward gradient arrow flows cleanly from loss back through z to phi, annotated deterministic in phi given epsilon so gradient flows to phi.](./assets/vae_reparam_trick.jpg){ width=100% }

*The reparameterization trick. **Left:** sampling $z$ inside the graph blocks gradients to $\phi$. **Right:** pull the randomness out into an external input $\epsilon$; now $z$ is a deterministic, differentiable function of $\phi$ (given $\epsilon$), and gradients flow.*

The fix is a change of variables that moves the randomness *out of the way of the gradient*. A Gaussian sample can be written as a fixed base noise, scaled and shifted:

$$
z = \mu_\phi(x) + \sigma_\phi(x) \odot \epsilon, \qquad \epsilon \sim \mathcal{N}(0, I),
$$

where $\odot$ is elementwise multiplication and $\epsilon$ is standard Gaussian noise **that does not depend on any parameter**. This produces a $z$ with exactly the right distribution $\mathcal{N}(\mu_\phi, \mathrm{diag}(\sigma_\phi^2))$, but now the only random thing, $\epsilon$, is an *input* fed in from outside. Given a drawn $\epsilon$, the map $\phi \mapsto z$ is a plain differentiable formula, so $\partial z/\partial \mu_\phi = 1$ and $\partial z/\partial \sigma_\phi = \epsilon$ exist and backprop sails straight through to the encoder. We have not changed *what* is being sampled — only *where the randomness enters* — turning "sample from a parameterized distribution" (not differentiable) into "apply a deterministic function to fixed noise" (differentiable). This single trick is what makes end-to-end gradient training of the VAE possible, and it reappears verbatim in diffusion models, where $x_t = \sqrt{\bar\alpha_t}\,x_0 + \sqrt{1-\bar\alpha_t}\,\epsilon$ is the very same move.

---

## 6. Putting it together: loss, training, generation

Collecting the pieces, the VAE minimizes (over a minibatch, averaging the per-example ELBO negated):

$$
\mathcal{L}_{\text{VAE}}(\theta, \phi) = \underbrace{\big\| x - \hat{x}(z) \big\|^2}_{\text{reconstruction, with } z = \mu_\phi(x) + \sigma_\phi(x)\odot\epsilon}
\;+\;
\underbrace{\frac{1}{2}\sum_{j=1}^{D}\big(\sigma_j^2 + \mu_j^2 - 1 - \log \sigma_j^2\big)}_{\text{KL to the prior}} .
$$

The **training loop** is then unremarkable, which is the point — every step is standard: (1) push $x$ through the encoder to get $\mu_\phi(x), \sigma_\phi(x)$; (2) draw $\epsilon \sim \mathcal{N}(0,I)$ and form $z = \mu_\phi + \sigma_\phi \odot \epsilon$; (3) decode to $\hat{x}$ and compute the two terms above; (4) backprop and update $\theta, \phi$ together. **Generation**, after training, throws the encoder away: draw $z \sim \mathcal{N}(0,I)$ and decode. Because training packed the codes into that same $\mathcal{N}(0,I)$, a fresh draw lands on the data manifold and the decoder returns a plausible new sample — and, crucially, the manifold is *continuous*, so you can walk from one code to another and watch the output morph smoothly.

![A 9-by-9 grid of small decoded images arranged by latent coordinates z1 (horizontal) and z2 (vertical), each a blob whose orientation, aspect ratio and curvature change gradually from cell to neighboring cell, so the whole grid morphs smoothly with no abrupt jumps, illustrating a continuous walkable latent manifold.](./assets/vae_prior_sampling_grid.jpg){ width=62% }

*Decoding a grid of latent codes (a conceptual/synthetic decoder here). The key property a good VAE latent has: **nearby codes decode to nearby outputs**, so the space is smooth and walkable rather than a set of disconnected memorized points. This smoothness is precisely what an AE latent lacks — and, foreshadowing [§8](#8-why-diffusion-uses-a-vae-not-a-plain-autoencoder), precisely what lets us later add and remove noise inside such a space.*

### 6.1 Two failure modes: posterior collapse and the blur, and β-VAE

Two things famously go wrong, worth a paragraph each since they recur across the whole family.

**Posterior collapse.** If the decoder is very powerful (e.g. an autoregressive PixelCNN that can model images well on its own), it can reconstruct $x$ while *ignoring* $z$ entirely. The KL term, meanwhile, is happiest when $q_\phi(z\mid x) = p(z)$ for every $x$ — i.e. the encoder outputs the prior regardless of input, carrying **zero information**. The optimizer can satisfy both by having the latent do nothing: reconstruction leans wholly on the decoder, KL goes to zero, and $z$ becomes an unused vestige. The model then generates poorly because sampling $z$ no longer controls the output. Fixes include weakening the decoder, "KL warm-up" (ramp the KL weight up slowly), or free-bits (don't penalize KL below a small floor per dimension). Discrete VAEs ([§7](#7-discrete-vaes)) sidestep it partly by forcing the latent to carry a real, discrete choice.

**The characteristic blur.** VAE samples are famously a little blurry. The clean reason: the reconstruction term is an *expectation over the cloud* $q_\phi$, so the decoder must map a whole neighborhood of $z$'s to plausible outputs, and the squared-error likelihood makes it hedge by outputting the *average* of everything that neighborhood could be — and an average of images is blurry. This is one motivation for VQ-VAE/VQGAN (a discrete latent + a perceptual/adversarial loss instead of pixel-MSE) and, later, for diffusion (which replaces the single-shot decode with many sharpening steps).

**β-VAE.** A one-knob variant that scales the KL term by a coefficient $\beta$: $\ \mathbb{E}_{q_\phi}[\log p_\theta(x\mid z)] - \beta\,\mathrm{KL}(q_\phi\|p)$. Turning $\beta > 1$ squeezes the latent harder toward the factorized prior, which empirically encourages **disentangled** latent dimensions (each coordinate controlling one interpretable factor) at some cost to reconstruction; $\beta < 1$ favors sharper reconstructions at the cost of a less prior-like latent. (Expandable on request — the key idea is just "the reconstruction/regularization trade-off is a dial, not a fixed 1:1.")

---

## 7. Discrete VAEs

So far $z$ has been a continuous vector. But much of what we want to model is naturally **discrete and compositional** — language is tokens, and even images can be described as "a layout of parts drawn from a fixed vocabulary." A discrete latent also plays beautifully with the most powerful sequence models we have (autoregressive transformers), which are built to model distributions over *tokens*. The **discrete VAE** line makes the latent a grid of symbols drawn from a learned **codebook**, and it is the backbone of a huge amount of modern generative modeling — including the latents that latent diffusion often runs on. Two members matter most.

### 7.1 VQ-VAE: snap the latent to a codebook

**VQ-VAE** (Vector Quantized VAE; van den Oord et al., 2017) keeps the encoder–decoder skeleton but inserts a **quantization** step. Maintain a **codebook** $\{e_1, \dots, e_K\}$ of $K$ learnable embedding vectors. The encoder produces a continuous feature map; each spatial location's vector $z_e(x)$ is then **snapped to its nearest codebook entry**, and that nearest code is what the decoder sees:

$$
z_q(x) = e_k, \qquad k = \arg\min_{j} \; \| z_e(x) - e_j \|_2 .
$$

So the latent is no longer a free vector but an **index into a finite vocabulary** — a small image becomes, say, a $32\times32$ grid of integers in $\{1,\dots,K\}$, exactly like a sentence is a grid of word tokens.

![A 2-D plot of continuous encoder outputs (scattered dots) colored by which of seven codebook vectors e1 through e7 (black stars) each is nearest to; thin lines connect every dot to its assigned star, and the background is tiled into seven colored Voronoi regions, one per code, showing that the plane is partitioned so every continuous output snaps to the nearest code.](./assets/vqvae_codebook_snap.jpg){ width=64% }

*Vector quantization. The continuous plane is partitioned into Voronoi cells around the codebook vectors $e_k$; every encoder output is replaced by the nearest $e_k$ before decoding. The latent becomes a discrete choice of "which code," and generation later amounts to modeling a distribution over these code indices.*

Two difficulties, and their fixes, are the whole of VQ-VAE:

- **Gradients through $\arg\min$.** Snapping to the nearest code is a hard, non-differentiable choice, so there is no gradient from decoder back to encoder — the same wall as [§5](#5-the-reparameterization-trick), but this time the reparameterization trick does not apply (the variable is discrete). VQ-VAE uses the **straight-through estimator**: on the forward pass use the quantized $z_q$, but on the backward pass just *copy* the decoder's gradient at $z_q$ straight onto $z_e$ as if quantization were the identity. It is biased but works well in practice.
- **Training the codebook.** The loss adds two terms beyond reconstruction: a **codebook loss** $\|\,\mathrm{sg}[z_e(x)] - e_k\,\|^2$ that pulls the chosen code toward the encoder output, and a **commitment loss** $\beta\,\|\,z_e(x) - \mathrm{sg}[e_k]\,\|^2$ that pulls the encoder output toward its code so it doesn't wander between codes (here $\mathrm{sg}[\cdot]$ is "stop-gradient" — treat as a constant in backprop, so each term trains only one side). Together they keep the codebook and encoder mutually consistent.

The subtle, important part: VQ-VAE trains with a **uniform prior** over codes and does *not* learn to generate on its own. Generation is a **second stage** — after the autoencoder is trained, freeze it, encode all data to code grids, and fit a powerful autoregressive prior (originally a PixelCNN; today a transformer) *over those discrete grids*. To sample: the prior generates a fresh grid of codes, look each up in the codebook, and decode. This "**compress to tokens, then model the tokens**" split is the template that DALL-E and VQGAN-based systems, and much of latent generative modeling, all follow.

### 7.2 DALL-E's discrete VAE (Gumbel-softmax)

The first **DALL-E** (Ramesh et al., 2021) uses a discrete VAE (**dVAE**) in exactly this compress-to-tokens role: it turns a $256\times256$ image into a $32\times32$ grid of tokens from a codebook of $8192$, and a large autoregressive transformer then models the joint sequence of *text tokens followed by image tokens*. Text-to-image generation becomes "continue the sequence": feed the caption's tokens, let the transformer roll out image tokens, and decode them back to pixels with the dVAE decoder.

The one technical twist versus VQ-VAE is *how* it gets gradients through the discrete choice. Rather than nearest-neighbor snapping with a straight-through estimator, DALL-E's dVAE treats the code as a **categorical distribution** over the $8192$ entries and uses the **Gumbel-softmax** relaxation: replace the hard "pick the argmax category" with a *soft*, temperature-controlled weighted average over all codes that is fully differentiable, then anneal the temperature $\tau \to 0$ during training so the soft choice sharpens into a hard one. High temperature = smooth and differentiable but blurry; low temperature = nearly discrete like the real thing. This gives a clean gradient everywhere while converging to genuine discrete tokens — an alternative solution to the same non-differentiable-latent problem. (Medium-depth by design; the Gumbel-softmax mechanics themselves are a rabbit hole, expandable on request.)

### 7.3 VQ-VAE-2 and VQGAN, briefly

Two follow-ups matter mainly because their latents feed later systems. **VQ-VAE-2** (Razavi et al., 2019) stacks quantized latents in a **hierarchy** (a coarse top level for global structure, a finer bottom level for detail), with a separate autoregressive prior per level, sharply improving sample quality. **VQGAN** (Esser et al., 2021) keeps the VQ-VAE codebook but trains the autoencoder with a **perceptual loss plus a GAN discriminator** instead of plain pixel-MSE, which fixes the VAE blur and yields crisp, high-compression latents. This VQGAN autoencoder is exactly the kind of encoder/decoder that **latent diffusion** ([§8.2](#82-latent-diffusion-needs-a-well-behaved-latent)) wraps around its diffusion model — the reason these discrete-VAE notes belong in a diffusion precursor at all.

---

## 8. Why diffusion uses a VAE, not a plain autoencoder

This is the question to carry into `ddpm_ddim_flow_score.md`. It has two answers because the word "use" points at two different things — one *conceptual* (the training objective) and one *architectural* (latent diffusion). Both come straight from the machinery above.

### 8.1 DDPM is a hierarchical VAE with a fixed encoder

The deepest connection: **a diffusion model is a VAE**, just an unusual one. Line the pieces up.

- A VAE has an **encoder** $q_\phi(z\mid x)$ mapping data to latents. DDPM's **forward noising process** $q(x_{1:T}\mid x_0)$ — gradually adding Gaussian noise to a clean image $x_0$ over $T$ steps to produce the noisy $x_1, \dots, x_T$ — plays *exactly* the encoder's role, mapping data to latents. The one twist: this encoder is **fixed and hand-designed** (it has no learnable parameters — just "add a scheduled bit of noise"), whereas a VAE's encoder is learned.
- A VAE has a **decoder** $p_\theta(x\mid z)$. DDPM's learned **reverse process** $p_\theta(x_{t-1}\mid x_t)$ — the denoiser that walks noise back to data — is the decoder.
- A VAE has **one** latent $z$; DDPM has a whole **ladder** of latents $x_1, \dots, x_T$, one per noise level. That makes it a *hierarchical* (deep, $T$-layer) VAE.

Because it is a VAE, its likelihood is intractable for the very reason in [§4.1](#41-the-problem-an-intractable-likelihood) (you would integrate over all latent trajectories), and it is trained by **maximizing an ELBO** — the exact same lower bound, now with $T$ stacked KL terms, one per step. This is why the DDPM derivation is soaked in ELBO and KL language: it inherits them from the VAE. The payoff of *fixing* the encoder instead of learning it is enormous — the intractable posterior a VAE must approximate becomes, in DDPM, a **known Gaussian in closed form**, so the whole ELBO collapses to a simple per-step "predict the added noise" regression. So a reader who understands the VAE ELBO already understands, structurally, where DDPM's training objective comes from; the diffusion note's [§1.5](./ddpm_ddim_flow_score.md#15-training-from-likelihood-to-elbo-to-predict-the-noise) then fills in the specifics.

That answers "why does a probabilistic, VAE-style objective appear in diffusion at all rather than a plain reconstruction loss": because diffusion *is* latent-variable generative modeling, and the ELBO is the honest way to train such a model. A plain autoencoder objective (reconstruction only) would give you a compressor, not a sampler — the same dead end as [§2](#2-why-not-a-plain-autoencoder).

### 8.2 Latent diffusion needs a well-behaved latent

The second, more literal answer concerns **Latent Diffusion Models** (LDM / Stable Diffusion, Rombach et al., 2022). Running diffusion directly on megapixel images is ruinously expensive, so LDM first **compresses** the image to a small latent with an autoencoder, runs the entire diffusion process in that latent space, and decodes only at the end. The autoencoder they use is a **VAE** (a KL-regularized one, or the VQ-regularized VQGAN of [§7.3](#73-vq-vae-2-and-vqgan-briefly)) — *not* a plain autoencoder. Why does the regularization matter so much here?

Because **diffusion's core operation is "add Gaussian noise, then learn to remove it," and that operation is only meaningful in a latent space that is smooth, bounded in scale, and roughly Gaussian** — exactly the properties [§2](#2-why-not-a-plain-autoencoder) and [§6](#6-putting-it-together-loss-training-generation) showed a plain AE latent *lacks* and a VAE latent *has*. Spelled out:

- **You have to be able to sample the endpoint.** Diffusion's reverse process starts from pure noise $\sim \mathcal{N}(0,I)$ and denoises toward data. That only works if the data's latents actually live near $\mathcal{N}(0,I)$. A VAE's KL term guarantees this; a plain AE's latents sit at arbitrary scales in scattered islands, so a Gaussian starting point would decode to nothing.
- **Adding noise must stay on-manifold.** Diffusion assumes that a slightly-noised latent is still a sensible latent the model can denoise. In a plain AE's holey space, nudging a code by Gaussian noise likely drops it into an empty gap the decoder never learned — noise becomes destructive rather than gently corrupting. The VAE's smooth, gap-free, overlapping-clouds latent (the [§6](#6-putting-it-together-loss-training-generation) manifold) is one where "a bit of noise" maps to "a slightly different plausible latent," which is the entire premise diffusion rests on.
- **Scale must be controlled.** The noise schedule (how much noise at each step) is calibrated to a latent of roughly unit variance. The KL-to-$\mathcal{N}(0,I)$ (or the bounded VQ codebook) keeps the latent in that regime; an unregularized AE latent's arbitrary scale would break the schedule.

So latent diffusion uses a VAE for the same reason [§2](#2-why-not-a-plain-autoencoder) needed a VAE to *generate* in the first place: only the regularized latent is a **space you can sample and perturb**. The VAE's decoder also cheaply repairs the fine high-frequency detail that the compression drops, letting the expensive diffusion model spend its capacity on semantics in a small, tame space. In one line: **a plain autoencoder gives you a latent you can reconstruct from but not sample or safely noise; diffusion needs both, and that is exactly what the VAE's KL/VQ regularization buys.**

---

## 9. Summary

| Idea | One-line takeaway |
|---|---|
| Plain autoencoder | Great compressor, useless generator — its latent space has holes and no distribution to sample. |
| VAE core move | Encode to a *Gaussian cloud* $q_\phi(z\mid x)$, not a point, and pull every cloud toward a fixed prior $\mathcal{N}(0,I)$. |
| ELBO | A tractable lower bound on the intractable $\log p_\theta(x)$; maximizing it = reconstruction $-$ KL-to-prior. |
| Reparameterization | Write $z = \mu + \sigma\odot\epsilon$ with $\epsilon\sim\mathcal{N}(0,I)$ so gradients flow through the sampling step. |
| Failure modes | Posterior collapse (latent ignored) and the characteristic blur (averaging over the cloud). |
| VQ-VAE / dVAE | Discrete latent: snap features to a learned codebook (or Gumbel-softmax); model the token grid with a second-stage autoregressive prior. |
| VQGAN | VQ-VAE + perceptual/GAN loss $\Rightarrow$ crisp, compact latents — the ones latent diffusion often uses. |
| Diffusion, conceptually | A DDPM *is* a hierarchical VAE with a **fixed** noising encoder, hence trained by an ELBO. |
| Diffusion, practically | Latent diffusion runs inside a **VAE's** latent because only a smooth, unit-scale, samplable latent can be noised and denoised. |

---

## Sources

- **Kingma & Welling, "Auto-Encoding Variational Bayes" (2013)** — the original VAE: the ELBO, the reparameterization trick, and amortized inference. The must-read foundation.
- **Rezende, Mohamed & Wierstra, "Stochastic Backpropagation..." (2014)** — the reparameterization idea, developed independently and concurrently.
- **Higgins et al., "β-VAE" (2017)** — the KL-weighting knob and its link to disentanglement.
- **van den Oord, Vinyals & Kavukcuoglu, "Neural Discrete Representation Learning" (VQ-VAE, 2017)** — vector quantization, straight-through gradients, two-stage generation.
- **Razavi, van den Oord & Vinyals, "Generating Diverse High-Fidelity Images with VQ-VAE-2" (2019)** — hierarchical discrete latents.
- **Ramesh et al., "Zero-Shot Text-to-Image Generation" (DALL-E, 2021)** — the dVAE with Gumbel-softmax feeding an autoregressive transformer.
- **Esser, Rombach & Ommer, "Taming Transformers..." (VQGAN, 2021)** — VQ + perceptual/adversarial loss for crisp latents.
- **Rombach et al., "High-Resolution Image Synthesis with Latent Diffusion Models" (Stable Diffusion, 2022)** — diffusion inside a VAE/VQGAN latent; the practical answer of [§8.2](#82-latent-diffusion-needs-a-well-behaved-latent).
- Companion note: **`ddpm_ddim_flow_score.md`** — where the ELBO reappears as DDPM's training objective ([§1.5](./ddpm_ddim_flow_score.md#15-training-from-likelihood-to-elbo-to-predict-the-noise)) and the [Probability Primer](./ddpm_ddim_flow_score.md#extra-1--a-probability-primer).
