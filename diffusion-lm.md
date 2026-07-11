# Diffusion-LM: Diffusion for Controllable Text Generation

**Diffusion-LM** (Li et al., Stanford, NeurIPS 2022) takes the diffusion machinery that works so well for images and makes it work for **text**, unlocking fine-grained *controllable* generation via the very same classifier-guidance trick used for images. It is one of the papers that kicked off the current wave of "diffusion language models."

This is a companion to `classifier-guidance-diffusion.md`. It does **not** re-derive classifier guidance — read that note first. The recap box below is the minimum you need.

---

## Recap: What You Need From the Classifier-Guidance Note

> **Classifier guidance in one paragraph.** A diffusion model generates by repeatedly denoising a noisy sample $x_t$ (here $t$ is the diffusion step, large $t$ = more noise). To steer generation toward some attribute $y$, you add the gradient $\nabla_{x_t}\log p_\phi(y \mid x_t)$ — supplied by a **separate, frozen, noise-aware classifier** $p_\phi$ — into each denoising step. The gradient is taken **with respect to the data $x_t$** (not the weights), it flows **through the classifier only** (the diffusion model is untouched), and the whole thing is a **sampling-time knob** requiring no retraining of the generator. The formal statement is the *score decomposition*: conditional score = unconditional score + classifier gradient.
>
> *For the full derivation, the guidance scale, and where gradients flow, see `classifier-guidance-diffusion.md`.*

Diffusion-LM's contribution is (1) making diffusion run on text at all, and (2) using exactly this guidance mechanism — now on **text** — to control attributes like sentiment, syntax, or length.

---

## Table of Contents

- [The Problem: Diffusion Is Continuous, Text Is Discrete](#the-problem-diffusion-is-continuous-text-is-discrete)
- [Key Idea 1: Diffuse in Embedding Space](#key-idea-1-diffuse-in-embedding-space)
- [Key Idea 2: Control via Classifier Guidance](#key-idea-2-control-via-classifier-guidance)
- [Why It Was Notable](#why-it-was-notable)
- [Notable Follow-up Work](#notable-follow-up-work)
- [Products in the Market](#products-in-the-market)
- [Cheat-Sheet: Autoregressive vs Diffusion LM](#cheat-sheet-autoregressive-vs-diffusion-lm)
- [Sources](#sources)

---

## The Problem: Diffusion Is Continuous, Text Is Discrete

Diffusion models are built for **continuous** data. The forward process adds a little **Gaussian noise** at each step and the reverse process removes it — this only makes sense when your data lives in a continuous space (like pixel intensities) where "add a small amount of noise" is meaningful.

Text is **discrete**: a sentence is a sequence of token IDs (word 4021, word 88, …). There is no meaningful "add 0.3 of Gaussian noise to the word *cat*" — you either have the token or you don't. So vanilla diffusion cannot be applied to tokens directly. Bridging this gap is the first thing Diffusion-LM had to solve.

---

## Key Idea 1: Diffuse in Embedding Space

The trick is to move to a continuous space where noise *does* make sense: **word embeddings**.

- **Embed (tokens → vectors):** map each discrete token $w$ to a continuous embedding vector via a learned embedding function. A sentence of length $n$ becomes a sequence of vectors $x_0 \in \mathbb{R}^{n \times d}$ ($d$ = embedding dimension). This $x_0$ is the "clean data" that diffusion operates on — the exact analogue of a clean image.
- **Diffuse:** run ordinary Gaussian diffusion on $x_0$ — add noise going forward, train a network to denoise going backward — just as in the image case.
- **Round (vectors → tokens):** after the reverse process produces a clean embedding sequence, map each vector back to its **nearest token** in the vocabulary. This "rounding" step turns the continuous result back into actual words.

Because the embeddings themselves are **learned end-to-end**, the objective adds two terms on top of the usual diffusion denoising loss: an **embedding** term (that maps tokens into the space) and a **rounding** term (that makes the denoised vectors land close enough to real token embeddings to decode cleanly). Intuitively:

$$
\mathcal{L} \;=\; \underbrace{\mathcal{L}_{\text{diffusion}}}_{\text{denoise the vectors}} \;+\; \underbrace{\mathcal{L}_{\text{embed}} + \mathcal{L}_{\text{round}}}_{\text{tie the continuous space to real tokens}}
$$

where $\mathcal{L}_{\text{diffusion}}$ is the standard denoising objective from the image case and the extra two terms keep the continuous latent honest about the discrete vocabulary it must eventually decode to. (The exact forms are not essential for intuition; the point is that learning the embedding space and the denoiser jointly is what makes the round-trip tokens → vectors → tokens work.)

---

## Key Idea 2: Control via Classifier Guidance

This is where Diffusion-LM connects directly to the [recap above](#recap-what-you-need-from-the-classifier-guidance-note). Once text generation is a continuous denoising process, you can steer it with **exactly the same classifier-guidance mechanism** used for images — now applied to the embedding latent.

You want text with some attribute $y$ — a target sentiment, a part-of-speech pattern, a specific sentence length, or even a full syntactic parse tree. You train a small **classifier on the noisy latents** $p_\phi(y \mid x_t)$, and at each denoising step you nudge the latent by its gradient $\nabla_{x_t}\log p_\phi(y \mid x_t)$. The base Diffusion-LM is **frozen**; control is added purely at sampling time. This is *plug-and-play* controllable generation: one trained diffusion LM plus many small attribute classifiers, mixed and matched without retraining the generator.

Two practical wrinkles worth knowing (details in the paper):

- **Fluency regularization.** Pure guidance can drift into gibberish that satisfies the classifier but reads badly, so they add a term keeping samples near the model's own likely outputs (fluent text).
- **Multiple gradient steps per diffusion step.** Because the control targets are demanding (e.g. matching a parse tree), they take a few gradient updates on the latent at each diffusion step rather than one.

Diffusion-LM demonstrated control on **six** tasks, including hard *structural* ones like matching a target syntactic parse tree and controlling POS sequences — controls that are awkward for left-to-right autoregressive models because the whole sequence is refined jointly.

---

## Why It Was Notable

- **Non-autoregressive.** Unlike a GPT-style model that emits tokens strictly left-to-right, Diffusion-LM refines the **entire sequence in parallel**, iteratively. This is what makes global, structural control (parse trees, length) natural — the model can adjust any position at any step.
- **Fine-grained control without retraining.** It beat prior plug-and-play controllers (PPLM, FUDGE) on complex control tasks while keeping fluency, using the frozen-generator + gradient-guidance recipe.
- **Infilling falls out for free.** Because generation is not left-to-right, filling a gap in the middle of a sentence (conditioning on both left and right context) is natural.

---

## Notable Follow-up Work

Diffusion-LM used **continuous** (embedding-space Gaussian) diffusion. Much of the follow-up wave that made "diffusion LMs" genuinely competitive instead pivoted to **discrete / masked diffusion** — a related but technically distinct lineage that Diffusion-LM helped spark. The short version of that chain:

- **D3PM** — discrete diffusion via corruption in token space; its **absorbing-state** ("mask") variant seeded masked diffusion LMs.
- **SEDD** (Score Entropy Discrete Diffusion) — sharpened the discrete training objective; **ICML 2024 Best Paper**, first discrete diffusion to seriously rival GPT-2-scale autoregressive models.
- **MDLM** (Simple & Effective Masked Diffusion LMs, NeurIPS 2024) — showed the masked-diffusion loss reduces to a clean weighted mixture of masked-language-modeling losses, simplifying training.
- **LLaDA** — scaled masked diffusion to **8B parameters**, roughly matching **LLaMA3-8B** across reasoning benchmarks; a proof that diffusion LMs scale.
- **BD3-LM** (block diffusion) and **Dream 7B** — block-by-block diffusion that restores **KV-caching** and arbitrary-length generation, the practical recipe now used at scale.

The through-line: Diffusion-LM established "diffusion for text + guidance-style control," and the field then found that **masked/discrete** diffusion was the more scalable path. Guidance-style controllability persists across this lineage.

---

## Products in the Market

Text diffusion has crossed from research into shipping products, chiefly because parallel (non-autoregressive) decoding can be **much faster** than token-by-token generation:

- **Inception Labs — Mercury / Mercury Coder.** Billed as the first commercial-scale diffusion LLM (early 2025), reporting **>1000 tokens/sec on a single NVIDIA H100** — several times faster than speed-optimized autoregressive models — with **Mercury Coder** targeting code. Founded by Stanford's Stefano Ermon and Chenlin Meng (the same lab lineage as much of this diffusion work). A later **Mercury 2** extends the approach to reasoning.
- **Google — Gemini Diffusion.** Announced at **Google I/O 2025**, applying diffusion-based text generation at frontier scale, emphasizing very low latency. As of this writing it is positioned as **experimental** rather than a shipped production model.

The commercial pitch is not (yet) about controllability but about **speed and cost**: generating many tokens in parallel and refining them, instead of one-at-a-time autoregression.

---

## Cheat-Sheet: Autoregressive vs Diffusion LM

| Aspect | Autoregressive LM (GPT-style) | Diffusion LM |
|---|---|---|
| Generation order | Strictly left-to-right, one token at a time | Whole sequence refined in parallel, over several denoising steps |
| Underlying space | Discrete tokens directly | Continuous embeddings (Diffusion-LM) *or* discrete/masked states (SEDD, LLaDA) |
| Controllability | Harder for global/structural constraints | Natural via classifier guidance on the latent; global constraints easy |
| Infilling (fill a middle gap) | Awkward (needs special training) | Natural (bidirectional by construction) |
| Speed | One forward pass per token | Many tokens per step → potentially far faster (the commercial draw) |
| Tie to this repo | — | Uses the **classifier guidance** of `classifier-guidance-diffusion.md`, ported to text |

---

## Sources

- Diffusion-LM: [Diffusion-LM Improves Controllable Text Generation (Li et al., NeurIPS 2022)](https://arxiv.org/abs/2205.14217)
- [SEDD — Discrete Diffusion Modeling by Estimating the Ratios of the Data Distribution (ICML 2024 Best Paper)](https://arxiv.org/abs/2310.16834)
- [MDLM — Simple and Effective Masked Diffusion Language Models (NeurIPS 2024)](https://s-sahoo.com/mdlm/)
- [LLaDA — Large Language Diffusion Models](https://arxiv.org/abs/2502.09992)
- [Mercury: Ultra-Fast Language Models Based on Diffusion (Inception Labs)](https://arxiv.org/abs/2506.17298)
- [Inception Labs — Introducing Mercury](https://www.inceptionlabs.ai/blog/introducing-mercury)
