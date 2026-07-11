# Classifier-Free Guidance (CFG)

**Classifier-Free Guidance** (Ho & Salimans, 2022) is the successor to classifier guidance and the technique that actually powers modern text-to-image and text-to-video models (Imagen, DALL·E 2, Stable Diffusion, SD3, Flux). Its trick is in the name: it gets the *same* steering effect as classifier guidance **without any classifier at all**.

This is a companion to `classifier-guidance-diffusion.md`; read that first. The recap box below is the minimum needed to follow this note.

---

## Recap: What You Need From the Classifier-Guidance Note

> **Classifier guidance in three lines.** A diffusion model denoises a noisy sample $x_t$ ($t$ = diffusion step; its noise prediction $\epsilon_\theta(x_t,t)$ is the *score* $\nabla_{x_t}\log p(x_t)$, up to scale). To steer generation toward a condition $y$, you add a classifier's gradient, giving the **score decomposition**:
> $$\nabla_{x_t}\log p(x_t\mid y) = \underbrace{\nabla_{x_t}\log p(x_t)}_{\text{unconditional}} + \underbrace{\nabla_{x_t}\log p(y\mid x_t)}_{\text{classifier}}$$
> With a guidance scale you sample from $p(x_t\mid y)\,p(y\mid x_t)^{s}$, trading diversity for fidelity ($s>1$ sharpens).
>
> **Its costs:** you must train a *separate, noise-aware* classifier $p_\phi(y\mid x_t)$, and differentiating through it invites *adversarial* gradients (see the CLIP discussion in the main note).
>
> *For the full derivation, guidance scale, and sharpening, see `classifier-guidance-diffusion.md`.*

CFG keeps the good part (the score decomposition) and deletes the classifier.

---

## Table of Contents

- [What's Annoying About Classifier Guidance](#whats-annoying-about-classifier-guidance)
- [The Idea: One Model, Two Modes](#the-idea-one-model-two-modes)
- [The Sampling Rule](#the-sampling-rule)
- [Why It's Still "Guidance": The Implicit Classifier](#why-its-still-guidance-the-implicit-classifier)
- [The Guidance Weight and the Fidelity–Diversity Trade-off](#the-guidance-weight-and-the-fidelitydiversity-trade-off)
- [Cost and Practicalities](#cost-and-practicalities)
- [Why It Became the Default](#why-it-became-the-default)
- [Cheat-Sheet: Classifier Guidance vs CFG](#cheat-sheet-classifier-guidance-vs-cfg)
- [Sources](#sources)

---

## What's Annoying About Classifier Guidance

Classifier guidance works, but it drags along three problems:

1. **A whole extra model.** You must train a *separate* classifier — and not a normal one, but a **noise-aware** classifier that works on noisy $x_t$ at every timestep. That is a second training pipeline per conditioning signal.
2. **Adversarial gradients.** Differentiating through a classifier to raise $p(y\mid x_t)$ is exactly the recipe for an adversarial example: the gradient can find imperceptible perturbations that fool the classifier without genuinely improving the image (recall why Blended Diffusion needed augmentations).
3. **Awkward for rich conditioning.** A classifier over a fixed label set does not naturally handle free-form **text** prompts, which is what we actually want for text-to-image.

CFG removes all three by asking the *generative model itself* to supply the guidance signal.

---

## The Idea: One Model, Two Modes

Train a **single** diffusion network that can run in two modes:

- **Conditional:** $\epsilon_\theta(x_t, t, y)$ — predict the noise given the condition $y$ (e.g. a text embedding).
- **Unconditional:** $\epsilon_\theta(x_t, t)$ — predict the noise with *no* condition.

You get both from one network by **condition dropout** during training: on each example, with some probability $p_{\text{uncond}}$ (typically 10–20%) you replace the real condition $y$ with a special **null token** $\varnothing$ that means "no condition." So the same weights learn to denoise both when told what to make and when told nothing. No second model, no classifier — just occasionally hiding the label.

---

## The Sampling Rule

At sampling time, query the network twice — once conditional, once unconditional — and **extrapolate**. With guidance weight $w \ge 0$:

$$
\hat\epsilon(x_t, y) = (1 + w)\,\epsilon_\theta(x_t, y) - w\,\epsilon_\theta(x_t)
$$

Rewriting makes the intuition obvious:

$$
\hat\epsilon(x_t, y) = \epsilon_\theta(x_t) + (1 + w)\big(\underbrace{\epsilon_\theta(x_t, y) - \epsilon_\theta(x_t)}_{\text{direction the condition adds}}\big)
$$

The difference $\epsilon_\theta(x_t, y) - \epsilon_\theta(x_t)$ is "what changes when you reveal the condition" — the direction pointing toward *more $y$-like* images. CFG takes a step in that direction and then **overshoots** it (by the factor $1+w$): push away from the unconditional prediction, exaggerating whatever the condition wanted.

- $w = 0$: plain conditional sampling, no guidance.
- $w > 0$: stronger adherence to the condition.

(Conventions differ: some papers call $s = 1 + w$ the "guidance scale," so Stable Diffusion's scale of $7.5$ means $w = 6.5$.)

---

## Why It's Still "Guidance": The Implicit Classifier

Here is the payoff that ties CFG back to the whole classifier-guidance story. We never trained a classifier — yet CFG implements *exactly* classifier guidance, because a classifier is hiding inside the two model modes.

By **Bayes' rule**, $p(y \mid x_t) \propto \dfrac{p(x_t \mid y)}{p(x_t)}$. Take $\nabla_{x_t}\log$ (the constant drops out):

$$
\nabla_{x_t}\log p(y \mid x_t) = \nabla_{x_t}\log p(x_t \mid y) - \nabla_{x_t}\log p(x_t)
$$

The classifier gradient equals the **conditional score minus the unconditional score** — and *both of those are things our one model already predicts* (in $\epsilon$-space, that is exactly $\epsilon_\theta(x_t,y) - \epsilon_\theta(x_t)$, up to the score↔$\epsilon$ scale factor). So we can get the classifier gradient for free, without a classifier.

Plug this into the [score decomposition](#recap-what-you-need-from-the-classifier-guidance-note) with a guidance scale:

$$
\nabla_{x_t}\log p(x_t) + s\,\nabla_{x_t}\log p(y\mid x_t)
= \nabla_{x_t}\log p(x_t) + s\big(\nabla_{x_t}\log p(x_t\mid y) - \nabla_{x_t}\log p(x_t)\big)
$$

which rearranges to the CFG combination $(1+w)$-conditional $-\,w$-unconditional with $s = 1+w$. **The "classifier" in classifier-free guidance is the ratio of the model's own conditional and unconditional predictions.** Same math as the main note; the classifier just became implicit.

A direct consequence: CFG samples from $\propto p(x_t\mid y)\,p(y\mid x_t)^{w}$ — the *same* sharpened distribution classifier guidance targeted, so the [temperature/sharpening intuition](classifier-guidance-diffusion.md) carries over verbatim.

---

## The Guidance Weight and the Fidelity–Diversity Trade-off

Because CFG samples from $p(x_t\mid y)\,p(y\mid x_t)^{w}$, the weight $w$ is a **sharpening / temperature knob** (the same mechanism derived in the sharpening appendix of the main note):

- **Higher $w$** → images adhere more strongly to the prompt, look sharper and more "on-topic" (better fidelity / prompt alignment), but **less diverse**.
- **Too high** → over-saturated colors, blown-out contrast, and artifacts, because the sample is pushed off the model's natural image manifold.

This last failure is common enough that there are standard fixes: **dynamic thresholding** (Imagen) and **CFG-rescale**, which rein in the exaggeration at high $w$. Typical practical scales sit around $w \approx 6$–$7$ (scale $s \approx 7$–$8$) for text-to-image.

---

## Cost and Practicalities

- **Two forward passes per step.** Each denoising step runs the network twice (conditional + unconditional), so CFG roughly **doubles sampling compute**. Still far cheaper than training and backpropagating through a separate classifier every step.
- **Rich conditioning is trivial.** Since $y$ is just an input to the network (e.g. a text embedding from a language/vision encoder), CFG handles free-form text prompts naturally — the thing a fixed-label classifier could not.

---

## Why It Became the Default

CFG hits the sweet spot: no separate classifier, no noise-aware-classifier training, no adversarial-gradient gaming, and native support for text conditioning. It is the guidance method behind essentially all modern conditional generators — **Imagen, DALL·E 2, Stable Diffusion** — and it carries directly to **flow-based** models too: the flow-matching systems **Stable Diffusion 3** and **Flux** use CFG (see the "Guidance for Flow-Based Models" section of `classifier-guidance-diffusion.md`, since velocity is a linear function of the score, the same conditional-minus-unconditional combination applies).

---

## Cheat-Sheet: Classifier Guidance vs CFG

| Aspect | Classifier Guidance | Classifier-Free Guidance |
|---|---|---|
| Needs a separate classifier? | **Yes** (noise-aware, per signal) | **No** |
| Extra training pipeline? | Yes — train the classifier | No — just condition dropout during normal training |
| Guidance signal | $\nabla_{x_t}\log p_\phi(y\mid x_t)$ from the classifier | $\epsilon_\theta(x_t,y) - \epsilon_\theta(x_t)$ from the model itself |
| Gradient through a classifier at sampling? | Yes (backprop through classifier) | No (two forward passes, no backprop) |
| Adversarial-gradient risk | Yes | No |
| Conditioning type | Awkward beyond fixed labels | Natural for free-form text |
| Sampling compute | 1 model pass + 1 classifier backprop / step | 2 model passes / step |
| Target distribution | $p(x_t\mid y)\,p(y\mid x_t)^{s}$ | $p(x_t\mid y)\,p(y\mid x_t)^{w}$ (same form) |
| Used in | Beat-GANs, Guided-TTS, Diffusion-LM | Imagen, DALL·E 2, Stable Diffusion, SD3, Flux |

---

## Sources

- CFG: [Ho & Salimans — *Classifier-Free Diffusion Guidance* (arXiv:2207.12598, 2022)](https://arxiv.org/abs/2207.12598)
- Companion note in this repo: `classifier-guidance-diffusion.md` (classifier guidance, the survey, and the flow-based section).
