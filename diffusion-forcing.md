# Diffusion Forcing: Next-Token Prediction Meets Full-Sequence Diffusion

**Diffusion Forcing** (Chen, Martí Monsó, Du, Simchowitz, Tedrake, Sitzmann —
arXiv **2407.01392**, NeurIPS **2024**) is one deceptively small idea with large
consequences: when you train a diffusion model over a *sequence* (video frames,
robot actions, time-series steps), give **each position its own, independently
sampled noise level** instead of noising the whole sequence to one shared level.

That single change makes two families of generative models — the **next-token
prediction** of language models and the **full-sequence diffusion** of video
models — fall out as two special cases of the *same* trained network. And because
they are corners of a continuous space of noise-level patterns, you can pick any
point in between **at sampling time, without retraining**: generate autoregressively
for arbitrary lengths, denoise a whole block jointly, or do something new that is
more stable than either.

This note assumes you have seen a basic diffusion model. If not, the one fact you
need is recapped below (fuller treatment in `ddpm_ddim_flow_score.md`). It is
written for someone who has *not* read the paper but wants to follow the logic
end to end and hold a real conversation about it.

---

## Table of Contents

- [Recap: The One Diffusion Fact We Need](#recap-the-one-diffusion-fact-we-need)
- [Two Paradigms and Their Tension](#two-paradigms-and-their-tension)
- [The Key Idea: Independent Per-Token Noise Levels](#the-key-idea-independent-per-token-noise-levels)
- [Setup and Notation](#setup-and-notation)
- [Training](#training)
- [Sampling on a 2D Grid](#sampling-on-a-2d-grid)
- [Guidance and Planning (Briefly)](#guidance-and-planning-briefly)
- [Applications](#applications)
- [Follow-up: History-Guided Video Diffusion (DFoT)](#follow-up-history-guided-video-diffusion-dfot)
- [Sources](#sources)

---

## Recap: The One Diffusion Fact We Need

> **A diffusion model is a denoiser trained across noise levels.** Take a clean
> item $x^{(0)}$, pick a noise level $k \in \{0, 1, \dots, K\}$, and corrupt it:
>
> $$ x^{(k)} = \sqrt{\bar\alpha_k}\, x^{(0)} + \sqrt{1 - \bar\alpha_k}\, \epsilon, \qquad \epsilon \sim \mathcal{N}(0, I) $$
>
> Here $\bar\alpha_k$ is the **signal-retention** at level $k$ (it slides from
> $\bar\alpha_0 = 1$, all signal, down to $\bar\alpha_K \approx 0$, pure noise),
> and $\epsilon$ is a Gaussian noise sample. A network $\epsilon_\theta$ is trained
> to look at $x^{(k)}$ (and $k$) and predict the $\epsilon$ that was added; knowing
> $\epsilon$ lets you step back toward a cleaner sample. To *generate*, start at
> $k=K$ (pure noise) and walk $k$ down to $0$. (Details: `ddpm_ddim_flow_score.md`.)

The only twist in this note is bookkeeping: we reserve $t$ for **position in the
sequence** and use $k$ for the **noise level**. So a single token can be written
$x_t^{(k)}$ — "the token at position $t$, corrupted to noise level $k$."

**Read the noise level as an amount of masking.** At $k=0$ the token is fully
revealed (clean); at $k=K$ it is pure noise, i.e. fully hidden; intermediate $k$
is a *soft, partial* mask. This "noise = masking" reading is the mental model that
makes the rest of the note click, so keep it in mind.

---

## Two Paradigms and Their Tension

Before the idea, the problem it resolves. There are two dominant ways to model a
sequence generatively, and each is good at exactly what the other is bad at.

**Next-token prediction (teacher forcing).** This is how language models work:
factor the joint distribution into a chain $p(x_1)\,p(x_2 \mid x_1)\,p(x_3 \mid
x_{1:2})\cdots$ and predict one token at a time from a **clean** past. "Teacher
forcing" is the training trick behind it — during training you always feed the
*ground-truth* clean history, never the model's own guesses. The upside is
**causality and variable length**: the model is inherently sequential, so you can
roll it forward for as many steps as you like. The downsides are two. First, there
is **no principled way to steer the whole sequence toward a goal** — the model only
ever thinks one step ahead, so "produce a trajectory that reaches this target" is
awkward. Second, **error accumulation**: at sampling time the past is the model's
own (imperfect) output, not the clean data it trained on, so small mistakes feed
back as input and compound, and long rollouts drift off the data manifold.

**Full-sequence diffusion.** This is how video-diffusion models (think Sora-style)
work: treat a whole fixed-length block of frames as one big object and denoise the
*entire block jointly* under a **single shared noise level** $k$ that ticks down
$K \to 0$ for all frames together. Because it models the joint distribution of the
whole block, it is naturally **guidable** — you can bias the whole trajectory
toward a goal with the same score-guidance machinery used for images (see
`classifier-guidance-diffusion.md`). But it is **non-causal and fixed-horizon**:
the block length is baked in at training time, so generating longer than the
training window, or generating in a streaming/online fashion, does not come for
free.

| | Next-token prediction | Full-sequence diffusion |
|---|---|---|
| **Causal?** | Yes (left-to-right) | No (all frames at once) |
| **Horizon** | Variable / unbounded | Fixed at training |
| **Guidance toward a goal** | Hard | Natural (models the joint) |
| **Long-rollout stability** | Poor (errors compound) | N/A (can't exceed the block) |

The tension is clear: one is causal and open-ended but hard to guide and unstable;
the other is guidable but rigid and fixed-length. Diffusion Forcing's claim is that
these are not two different models but **two settings of one knob**.

---

## The Key Idea: Independent Per-Token Noise Levels

Look at what the noise level $k$ *is* in each paradigm, using the "noise = masking"
reading:

- **Full-sequence diffusion**: every position shares **one** level $k_t = k$ that
  decreases in lockstep. All tokens are equally masked at every step.
- **Next-token prediction**: the past is clean ($k_t = 0$, fully revealed) and the
  token being produced is generated from noise ($k_t = K$, fully masked). The mask
  pattern is a hard step: revealed behind you, hidden ahead.

Both are just **patterns of noise levels across positions**. So instead of tying
all positions to one $k$, or hard-coding the clean-past/noisy-future step, let
**each position $t$ carry its own noise level $k_t$**, and — crucially — sample
those levels *independently* during training. A sequence's "state" is now a whole
**vector** $(k_1, k_2, \dots, k_T)$ of noise levels, one per token. Since a noise
level is a soft mask, this vector is a *continuous, per-token generalization of a
BERT-style mask*: not just "masked or not," but "how masked, position by position."

The two paradigms are now recovered as two corners of this vector space:

- set **all $k_t$ equal** and decrease them together $\to$ full-sequence diffusion;
- hold **past $k_t = 0$** and generate the next token from $k_t = K$ $\to$
  autoregressive next-token prediction.

Everything in between — some positions crisp, some hazy, in any pattern you like —
is new territory. The payoff, made precise in the sampling section, is that a model
trained this way can be run as *either* paradigm *or any blend*, chosen at sampling
time without retraining, because it has been forced to cope with every combination
of clean and noisy neighbours.

---

## Setup and Notation

| Symbol | Meaning |
|---|---|
| $x_{1:T}$ | The sequence to model — $T$ tokens $x_1, \dots, x_T$ (e.g. video frames, robot observation-action pairs, time-series steps). |
| $t$ | **Position** in the sequence, $t = 1, \dots, T$. |
| $k$, $k_t$ | **Noise level**, $0 \le k \le K$. $k=0$ clean, $k=K$ pure noise. $k_t$ is the level assigned to position $t$. |
| $x_t^{(k_t)}$ | Token at position $t$, corrupted to its own noise level $k_t$ via the recap equation. |
| $\bar\alpha_k$ | Signal-retention at level $k$ (from $1$ at $k=0$ down to $\approx 0$ at $k=K$). |
| $\epsilon_\theta$ | The learned network; predicts the noise added to a token given its noisy value, its level, and the context. |
| $z_t$ | The model's **hidden state** at position $t$ — a running summary of the (noisy) history $x_{1:t}$, carried causally left to right. |

---

## Training

The training loop is almost boringly close to ordinary diffusion — the one new line
is where the noise levels come from.

1. **Draw a clean sequence** $x_{1:T}$ from the data.
2. **Sample an independent noise level $k_t$ per position**, each uniform over
   $\{0, \dots, K\}$. This is the whole idea: the levels are *not* shared, so one
   training example might have position 3 nearly clean sitting next to position 4
   nearly pure noise.
3. **Corrupt each token to its own level**, $x_t^{(k_t)} = \sqrt{\bar\alpha_{k_t}}\,
   x_t + \sqrt{1 - \bar\alpha_{k_t}}\, \epsilon_t$, with a fresh $\epsilon_t$ each.
4. **Denoise causally.** A causal model (an RNN in the paper; a causal transformer
   works too) reads the noisy tokens left to right, maintaining a hidden state
   $z_t$ that summarizes the noisy history $x_{1:t}^{(k_{1:t})}$, and predicts the
   noise in each token from that state and its level.

The loss is the standard denoising objective, just **summed over positions**:

$$
\mathcal{L} = \mathbb{E}\left[ \sum_{t=1}^{T} \big\| \epsilon_t - \epsilon_\theta\big(x_t^{(k_t)},\, k_t,\, z_{t-1}\big) \big\|^2 \right]
$$

Unpacking it term by term:

- $\epsilon_t$ — the true noise that was mixed into position $t$; the target.
- $\epsilon_\theta(\cdot)$ — the network's guess of that noise. Its three inputs are
  the noisy token $x_t^{(k_t)}$, **its own** level $k_t$ (so the net knows how
  corrupted this token is), and $z_{t-1}$, the causal summary of everything before
  it.
- The sum over $t$ and the expectation over data, levels, and noise mean: on
  average, across every position and every random assignment of noise levels, the
  net should predict each token's noise well.

**Why sample the levels independently — what does it force?** Because the levels
are random and per-position, over training the network sees *every conceivable
pattern* of clean and noisy neighbours: clean-past/noisy-future (the autoregressive
corner), all-equal (the full-sequence corner), and every ragged mix. It is forced
to denoise a token whether its context is crisp or hazy. That is precisely the
competence that lets us, at sampling time, drive the model along **any** path
through the noise-level space — the flexibility in the next section is bought here.

**A word on what this optimizes.** The authors show this objective is a valid
**variational lower bound on the likelihood of the sequence** (and, because of the
per-token levels, of its subsequences) — the same style of bound that justifies the
ordinary diffusion loss (the ELBO), extended to independent per-token noise. We
take the result as given; the derivation is in the paper.

---

## Sampling on a 2D Grid

Here is where the flexibility cashes out. Picture a **2D grid**: columns are
sequence positions $t$, and rows are denoising iterations. Each **cell holds the
current noise level of that token**. Generation is nothing more than a **schedule**
— a prescription for how to lower each column's level from $K$ (pure noise) down to
$0$ (clean) as you move down the rows. The trained network fills in the denoising
at each step; the *schedule* is a free choice you make at inference.

The figure shows three schedules on the same grid. Dark = high noise, light = clean.

![Three sampling schedules on the Diffusion Forcing grid: columns are sequence positions, rows are denoising iterations, colour is each token's current noise level. Left: full-sequence diffusion lowers all columns uniformly. Middle: next-token / autoregressive uses a hard left-to-right front. Right: the Diffusion Forcing pyramid keeps future tokens noisier than the present.](assets/diffusion_forcing_sampling_grid.jpg)

- **Left — full-sequence diffusion.** Every column has the same level in a given
  row; the whole sequence brightens together. This is the "all $k_t$ equal" corner.
- **Middle — next-token / autoregressive.** A hard front sweeps left to right: a
  token is fully denoised (goes light) before the next one starts. Behind the front
  everything is clean, ahead of it everything is pure noise — the classic
  clean-past / noise-future step.
- **Right — Diffusion Forcing's pyramid.** The distinctive middle-ground schedule.
  Within any row, **noise level rises with position**: the present is nearly clean,
  the near future a little hazy, the far future still mostly noise. The whole ramp
  slides down over rows. In words: **commit firmly to the present, stay uncertain
  about the future, and let that uncertainty resolve only as the future becomes the
  present.**

**Why the pyramid is more stable on long rollouts.** In pure autoregression, once a
token is emitted it is treated as clean, hard fact — so an early mistake becomes
certain input for everything after it, and errors compound until the rollout drifts
off the data manifold. The pyramid refuses to over-commit: future tokens are held
at a *nonzero* noise level, i.e. an explicit "I'm not sure yet." A slightly-noisy
view of the future carries the gist without locking in details that might be wrong,
so the model is not forced to build later tokens on possibly-bad early guesses. This
graceful "uncertainty grows with the horizon" behaviour is exactly what unstable
autoregressive rollouts lack.

**Why the horizon can be unbounded.** The model is causal (hidden state $z_t$ flows
left to right), so nothing fixes the length: slide the window forward, keep feeding
past context, and generate indefinitely. The paper rolls out **2000+ frames from a
model trained on only 36–72 frames**, without the sliding-window resets that make
full-sequence video models stutter — the variable-length strength of next-token
prediction, now available in a diffusion model.

---

## Guidance and Planning (Briefly)

Because Diffusion Forcing models the **joint** distribution over the sequence (not
just one-step conditionals) *and* is causal and variable-horizon, it inherits the
best trait of full-sequence diffusion: you can **guide** generation toward a
sequence-level objective. The mechanism is the same score-guidance idea used for
conditional image generation — nudge each denoising step by the gradient of a
reward or goal (see `classifier-guidance-diffusion.md` and
`classifier-free-guidance.md`), only now applied *along the sequence*.

For **planning**, treat a trajectory of observations and actions as the sequence and
guide sampling toward high reward. The paper introduces **Monte Carlo Guidance**:
roll out several possible futures, score them, and steer sampling toward the ones
that look good — made practical precisely because the model can keep the near future
crisp while leaving the far future hazy (the pyramid), so it plans over a flexible
horizon rather than a rigid fixed block. The net effect is a diffusion planner that
is both causal (can react step by step) and goal-directed (can be steered).

---

## Applications

Kept to a line each — the point is that *one* recipe spans domains that usually need
different models:

- **Video prediction** (DMLab, Minecraft): long, temporally-consistent rollouts far
  beyond the training length, where full-sequence baselines diverge.
- **Robot imitation learning**: real manipulation tasks needing memory beyond a
  single observation, handled by the causal hidden state.
- **Diffusion planning** (e.g. maze navigation, decision-making): joint modelling of
  observations and actions with test-time Monte Carlo Guidance toward goals.
- **Time series**: continuous sequential data, same per-token-noise recipe.

---

## Follow-up: History-Guided Video Diffusion (DFoT)

A 2025 follow-up, **History-Guided Video Diffusion** (ICML 2025), builds the
**Diffusion Forcing Transformer (DFoT)** — the same per-token-noise idea in a
transformer backbone aimed squarely at video. Its key move is to treat the amount
of *clean history* the model is conditioned on as an explicit **guidance signal**:
by varying how much past is revealed (recall noise = masking) and applying guidance
on top, it generates longer, more consistent, higher-quality video than a plain
Diffusion Forcing model. It is the natural "scale it up for video" successor to the
original paper.

---

## Sources

- **Diffusion Forcing: Next-token Prediction Meets Full-Sequence Diffusion** — Chen,
  Martí Monsó, Du, Simchowitz, Tedrake, Sitzmann. arXiv **2407.01392**, NeurIPS 2024.
  <https://arxiv.org/abs/2407.01392>
- **Project page** (figures, demos, intuition): <https://boyuan.space/diffusion-forcing/>
- **Official code**: <https://github.com/buoyancy99/diffusion-forcing>
- **History-Guided Video Diffusion (DFoT)**, ICML 2025 — follow-up scaling the idea
  to video: <https://boyuan.space/history-guidance/>
- Companion notes in this repo: `ddpm_ddim_flow_score.md` (diffusion basics, the
  noising equation), `classifier-guidance-diffusion.md` and
  `classifier-free-guidance.md` (the guidance mechanism reused for planning).
