# Training-Free Image Editing in Diffusion and Flow Models

This note is about **editing a real photograph with a frozen, pretrained text-to-image model** — no fine-tuning, no LoRA, no extra training of any kind. You hand the model an image and two captions ("a cat walking" → "a puppy walking"), and it returns the edited image. "Training-free" is the whole point: it works on top of a giant off-the-shelf flow model (like FLUX or Stable Diffusion 3.5) using only clever inference-time tricks.

The note is built around one recent paper that I wanted to understand well:

> **DirectEdit: Step-Level Accurate Inversion for Flow-Based Image Editing** (arXiv [2605.02417](https://arxiv.org/pdf/2605.02417), [ar5iv HTML](https://ar5iv.labs.arxiv.org/html/2605.02417)).

DirectEdit is a good vehicle because it forces you to understand the one thing every training-free editor is secretly fighting: **inversion error**. We'll build up *why* inversion is hard, explain the paper's central Figure 2 panel by panel, then go through DirectEdit itself in detail, and finally place it next to three prior methods it builds on.

**Prerequisites.** This note assumes you already know rectified flow — the linear noise↔image path, the velocity field $v_\theta$, and Euler sampling. If not, read Part 2 of [`ddpm_ddim_flow_score.md`](./ddpm_ddim_flow_score.md) first; the "time runs in opposite directions" warning there (§[Shared Setup](./ddpm_ddim_flow_score.md#shared-setup-and-a-warning-about-time)) is especially worth internalizing. For the source/target-prompt machinery, [`classifier-free-guidance.md`](./classifier-free-guidance.md) covers how a text prompt steers the velocity.

---

## Table of Contents

- [The two-step template: invert, then re-generate](#the-two-step-template-invert-then-re-generate)
- [Setup and Notation](#setup-and-notation)
- [Why inversion is the hard part](#why-inversion-is-the-hard-part)
- [Figure 2, explained panel by panel](#figure-2-explained-panel-by-panel)
- [DirectEdit, in detail](#directedit-in-detail)
- [What the edits look like, and the error curve](#what-the-edits-look-like-and-the-error-curve)
- [Three prior methods, briefly](#three-prior-methods-briefly)
- [Takeaways](#takeaways)
- [Sources](#sources)

---

## The two-step template: invert, then re-generate

Almost every training-free editor follows the same two-step recipe, so it's worth stating once up front.

A generative flow model only knows how to do **one** thing: start from Gaussian noise and walk it to a *new* image along the learned velocity field. It has no built-in notion of "here is an existing photo, change this part." So to edit a *real* image we have to smuggle it into the model's world:

1. **Inversion.** Run the generation process *backward*: take the real image and find the noise sample that the model *would have* turned into that exact image. This noise (plus the intermediate states along the way) is the image's "fingerprint" in the model's coordinates.
2. **Re-generation (editing).** Run the process *forward* again from that noise, but now under a **new prompt**. Because we start from the image's own fingerprint, the output stays close to the original; because the prompt changed, the parts the prompt describes get edited.

The entire difficulty of training-free editing lives in **step 1**. If inversion were exact — if forward-from-the-recovered-noise perfectly reproduced the original image — editing would be trivially faithful. It is *not* exact, and the errors it introduces are what DirectEdit exists to kill. The rest of this note is really a story about making step 1 accurate.

![Two rows of before/after image pairs produced by DirectEdit. Top row (single-turn): a red STOP sign edited to read ICML; a wreath with a red bow gaining a jingle bell; a cat turned into a puppy. Bottom row (multi-turn): a seascape given a storm then a rainbow; a stone window arch with a bird removed then restyled as a cartoon.](./assets/direct_edit_teaser.jpg){ width=100% }

*What training-free flow editing can do. Single-turn edits (top): text swap, object addition, object replacement — the untouched regions stay pixel-faithful. Multi-turn edits (bottom): several edits chained on the same image without the background degrading. Figure from DirectEdit (arXiv 2605.02417).*

---

## Setup and Notation

We stay in the **rectified flow** convention throughout (matching the paper). Time $t$ runs from **noise** at $t=0$ to **image** at $t=1$; generation walks $t: 0 \to 1$, inversion walks $t: 1 \to 0$.

| Symbol | Meaning |
|---|---|
| $Z_0$ | Pure Gaussian noise, $Z_0 \sim \mathcal{N}(0, I)$ (the $t=0$ endpoint). |
| $Z_1$ | A clean image (its latent), the $t=1$ endpoint. |
| $Z_t$ | The interpolated state at time $t$: on the straight noise→image line. |
| $v_\theta(Z_t, \psi)$ | The pretrained **velocity field** — the network. Given a state $Z_t$ and a text prompt $\psi$, it predicts the direction to move. |
| $\sigma_0, \dots, \sigma_T$ | The discrete timesteps the sampler actually visits ($T$ Euler steps, e.g. 30–60). |
| $Z_t^{inv}$ | The state at step $t$ along the **inversion** trajectory (image → noise). |
| $\Delta Z_t$ | The **residual** DirectEdit records during inversion: $Z_{t+1}^{inv} - Z_t^{inv}$. |
| $Z_t^{src}, Z_t^{tar}$ | The two parallel editing branches: **source** (reconstruction) and **target** (edited). |
| $\psi_{src}, \psi_{tar}$ | The source caption ("a cat") and target prompt ("a puppy"). |
| $\mathcal{M}$ | A spatial **mask** marking the region to edit vs. the region to protect. |

The one relation to keep in your head is the linear interpolation that *defines* rectified flow:

$$
Z_t = t\, Z_1 + (1-t)\, Z_0, \qquad t \in [0,1].
$$

In words: every intermediate state is just a weighted blend of the image $Z_1$ and the noise $Z_0$, sliding linearly as $t$ moves. The model learns the velocity $v_\theta \approx Z_1 - Z_0$ that pushes you along this line. Everything below is about how faithfully we can walk *back* along it and then *forward* again.

---

## Why inversion is the hard part

This section unpacks the two passages from the paper's introduction that were confusing, because they contain the entire motivation.

### Passage 1 — inversion-based methods accumulate error

To generate, the sampler takes Euler steps forward along the velocity field:

$$
Z_{t+1} = Z_t + (\sigma_{t+1} - \sigma_t)\, v_\theta(Z_t).
$$

Read it plainly: to get the next state, stand at the current state $Z_t$, ask the network which way to go ($v_\theta(Z_t)$), and take a step of length $(\sigma_{t+1}-\sigma_t)$ in that direction. Simple.

Now we want to go **backward** (image → noise). Ideally, inversion would be the exact inverse of that step:

$$
Z_t^{inv} = Z_{t+1}^{inv} - (\sigma_{t+1} - \sigma_t)\, v_\theta(\boxed{Z_t^{inv}}).
$$

Look carefully at the boxed term. To compute the *previous* state $Z_t^{inv}$, this formula needs the velocity **evaluated at $Z_t^{inv}$ itself** — the very thing we are trying to find. It's circular: the answer appears on both sides. So the exact inverse step is not directly computable.

The standard fix is to cheat: evaluate the velocity at the state we *do* have, the current one $Z_{t+1}^{inv}$, instead of the unknown $Z_t^{inv}$:

$$
Z_t^{inv} = Z_{t+1}^{inv} - (\sigma_{t+1} - \sigma_t)\, v_\theta(\boxed{Z_{t+1}^{inv}}).
$$

This is the sentence in the paper: *"the noisy latent at the current timestep is used to approximate the latent of the subsequent timestep... resulting in a deviation from the forward process."* Because $v_\theta(Z_{t+1}^{inv}) \neq v_\theta(Z_t^{inv})$, every single inversion step is slightly wrong — a small **approximation error**. Over the ~30–60 steps of the trajectory these small errors don't cancel; they **compound**, pushing the reconstruction path steadily away from the true inversion path. That compounding is the "inevitable accumulation of reconstruction errors," and *"existing RF-based training-free methods primarily focus on mitigating approximation errors within the inversion path"* — i.e. they try to make each backward step less wrong (higher-order solvers, per-step correction), rather than removing the mismatch entirely.

The figure below makes the compounding concrete on a 1-D toy. The horizontal axis is **flow time $t$** (matching our notation: $t=0$ is noise $Z_0$, $t=1$ is image $Z_1$), and the vertical axis is a scalar stand-in for the latent state $Z_t$, so the ideal trajectory is just the diagonal $Z_t = t$. Inversion progresses image→noise (right→left, decreasing $t$); reconstruction/generation progresses noise→image (left→right, increasing $t$) — the two directional arrows mark this.

![A two-panel schematic with flow time t on the horizontal axis, running from Z-zero noise at t equals zero on the left to Z-one image at t equals one on the right. Left panel titled Vanilla Euler inversion: a blue diagonal inversion path with a leftward image-to-noise arrow, and a red dashed reconstruction path with a rightward noise-to-image arrow that starts together at the noise corner but falls progressively below, with vertical red bars marking a gap that grows every step and an annotation that it never reaches the image. Right panel titled DirectEdit: the same blue diagonal inversion path with a green reconstruction path lying exactly on top of it, a rightward noise-to-image arrow, magenta arrows marking injected residuals delta-Z, annotated Error approximately zero, paths coincide.](./assets/directedit_path_alignment.jpg){ width=100% }

*Why alignment matters (original schematic). **(a)** Each Euler step of the noise→image reconstruction evaluates $v_\theta$ at the wrong state, leaving a small same-signed gap from the inversion reference; the gaps compound, so the reconstruction drifts and never lands on the image. **(b)** DirectEdit injects the recorded residual $\Delta Z_t$ before each velocity call, so the reconstruction rides exactly on the inversion trajectory.*

### Passage 2 — inversion-free methods still drift

One reaction to all this: *don't invert at all.* That is the idea behind **inversion-free** methods such as **FlowEdit**. Instead of recovering the noise and re-generating, they construct a **direct path from the source image to the edited image**, by interpolating the source latent with fresh random noise and letting the *difference* between the source-prompt and target-prompt velocities carry the image toward the edit.

The paper's verdict — *"although offering marginal improvements over standard Euler methods, these training-free methods either continue to suffer from error accumulation or struggle to guarantee the fidelity of the generated images"* — has a clean intuition. Skipping inversion trades one problem for another:

- The direct path is built with **random noise** injected at editing time, so the trajectory is stochastic and only *approximately* anchored to the source image → residual **error still accumulates** along the direct path.
- Or, if you pin it hard to the source to protect fidelity, you lose editability. You can't easily have both without the exact anchor that a *good* inversion would have given you.

So neither camp is free: inversion-based methods drift because each backward step is approximate; inversion-free methods drift because the direct path is only loosely tied to the source. DirectEdit's bet is that if you fix inversion *properly* — make it step-level exact — you get the fidelity anchor of inversion **without** the accumulation. That is the gap Figure 2 dramatizes.

---

## Figure 2, explained panel by panel

![Figure 2 from the paper: three side-by-side diagrams, each with Gaussian Noise at the top, a green Source Image region at bottom-left and an orange Target Image region at bottom-right. A legend defines source/reconstruction/target latents and the various path and arrow types. Panel (a) Vanilla Euler Inversion shows the reconstruction path diverging from the inversion path with a red Error greater than zero arrow and a Drifted Feature label. Panel (b) Stepwise Correction shows black correction arrows pulling the path back each step but a residual Error greater than zero and still a Drifted Feature. Panel (c) DirectEdit shows the reconstruction path lying on the inversion path with Error equals zero and an Ideal Feature label.](./assets/direct_edit_fig2.jpg){ width=100% }

*Figure 2 from DirectEdit (arXiv 2605.02417). Comparison of inversion methods.*

**Reading the legend first.** Each panel is a little 2-D map. The **Source Image** (green cloud, bottom-left) is the real photo; **Gaussian Noise** (purple cloud, top) is where inversion should end up; the **Target Image** (orange cloud, bottom-right) is the edit we want. The dots are latent states — blue = source latent, magenta = reconstruction latent, grey = target latent. The arrows are the three trajectories plus their errors:

- **grey solid** = the *inversion path* (image → noise);
- **grey dashed** = the *reconstruction path* (noise → image, what you get replaying forward);
- **blue dashed** = the *editing path* (noise → target image under the new prompt);
- **black** = a *correction* applied to a step; **red** = the *reconstruction error*; **green** = *feature interaction* — the source features being injected into the edit.

The key idea to hold onto: **editing quality depends on feeding the target branch clean source features.** If the reconstruction path is off-trajectory, the features it injects are "drifted," and the edit inherits that corruption.

**(a) Vanilla Euler inversion.** The reconstruction (magenta, dashed) does *not* retrace the inversion (grey). At each step there's a red **Error > 0** gap, and — exactly as in the previous section — these compound. Because the reconstruction latents are wrong, the features handed to the editing path (green arrow) are labelled **Drifted Feature**. Result: neither the reconstruction nor the edit is faithful.

**(b) Stepwise correction.** Now a black **Correction** arrow pulls the reconstruction back toward the inversion path *after each step*. This stops the errors from snowballing across the whole trajectory — the global accumulation is mitigated. But look closely: there is *still* a small **Error > 0** at every step, because the correction is applied *after* the erroneous velocity was already computed. The per-step residual never reaches zero, so the injected features are *still* **Drifted** — just less so. Better reconstruction, still-flawed edits.

**(c) DirectEdit.** Here the reconstruction path lies *exactly on* the inversion path — **Error = 0** at the step level, not merely bounded. Since the source features are now clean, the feature interaction arrow is labelled **Ideal Feature**, and the editing path (blue) can lean on them safely. The trick that makes Error = 0 possible without extra network calls is the subject of the next section.

The distinction between (b) and (c) is the crux and easy to miss: (b) *corrects the result of a wrong step*; (c) *never takes the wrong step in the first place*, by feeding $v_\theta$ the right input.

---

## DirectEdit, in detail

### The alignment idea

Everything in the previous panels is a symptom of one root cause: during re-generation, the velocity network is evaluated at states that don't match the inversion trajectory. So DirectEdit sets a single, blunt goal — **make the re-generation states equal the inversion states, step for step:**

$$
Z_t = Z_t^{inv} \quad \text{for every } t.
$$

If that holds, then the network is always evaluated on-trajectory, $v_\theta(Z_t) = v_\theta(Z_t^{inv})$, and there is *no* step-level mismatch to accumulate. The question is how to enforce it cheaply.

### Residual latent injection (the core trick)

The elegant observation is that during inversion we already computed the *entire* correct trajectory $Z_0^{inv}, Z_1^{inv}, \dots$. So we can simply **record how much the latent moved at each inversion step** and replay exactly that movement during generation. Define the per-step **residual**:

$$
\Delta Z_t = Z_{t+1}^{inv} - Z_t^{inv}.
$$

This is just "the vector the latent traveled between consecutive inversion states" — a quantity we get for *free* while inverting, by subtraction. It carries the true, on-trajectory displacement.

Now, during re-generation, before asking the network for a velocity, **add the recorded residual to align the current state onto the inversion trajectory:**

$$
\hat{Z}_t = Z_t + \Delta Z_t,
$$

and take the Euler step using the velocity evaluated at this *aligned* state:

$$
Z_{t+1} = Z_t + (\sigma_{t+1} - \sigma_t)\, v_\theta(\hat{Z}_t).
$$

Walk through why this fixes everything:

- $\hat Z_t$ is, by construction, the latent *shifted to where the inversion trajectory actually was*. So the network sees the same input it saw during inversion → the velocity is the correct, on-trajectory one, not one computed from a drifted state.
- Because we reuse residuals we *already stored*, there are **no additional neural function evaluations (NFEs)** — DirectEdit runs the network the same number of times as plain Euler. This is what separates it from methods that re-invert or iteratively optimize each step (which cost extra forward passes).
- Contrast with stepwise correction (Figure 2b): correction fixes the state *after* a wrong velocity was already used, so a residual error survives. DirectEdit fixes the *input to the velocity*, so the wrong velocity is never computed — the step-level error is zero, not merely reduced.

That is the whole engine. The remaining pieces are about turning accurate reconstruction into a controllable *edit*.

### Dual-branch editing

To edit rather than merely reconstruct, DirectEdit runs **two branches in parallel**, both sharing the same recorded residuals $\Delta Z_t$:

- the **source branch** $Z_t^{src}$, denoised under the original caption $\psi_{src}$ — this faithfully rebuilds the input and is the source of clean features;
- the **target branch** $Z_t^{tar}$, denoised under the new prompt $\psi_{tar}$ — this is the edit.

Both are aligned every step ($\hat Z_t^{src} = Z_t^{src} + \Delta Z_t$ and $\hat Z_t^{tar} = Z_t^{tar} + \Delta Z_t$), so both ride the true trajectory. The source branch stays pinned to the original image, giving the target branch a trustworthy reference to borrow from.

### Attention value injection

How does the target branch "borrow" from the source? Through the transformer's attention. In the early denoising steps — when global layout and identity are being decided — DirectEdit replaces the target branch's **Value** features with the source branch's, while keeping the target's own Query and Key:

$$
\hat{F}_t^{tar} =
\begin{cases}
\mathrm{Attention}(Q_t^{tar},\, K_t^{tar},\, V_t^{src}) & \text{if } t < t_{inj},\\[4pt]
\mathrm{Attention}(Q_t^{tar},\, K_t^{tar},\, V_t^{tar}) & \text{otherwise.}
\end{cases}
$$

The intuition for *why Values*: in attention, the Query/Key pair decides *where to look* (the layout, the correspondence), while the Value carries *what content* gets pulled in. By injecting $V^{src}$, the target branch keeps the source image's actual appearance and details (identity, texture) while still arranging them according to the target prompt's Q/K. And this only runs for the early steps ($t < t_{inj}$): once structure is locked in, control is handed back to the target's own Values so the prompt-specified change ("puppy") can fully materialize. This is exactly the **Ideal Feature** interaction of Figure 2c — it only works because the injected $V^{src}$ comes from a *drift-free* source branch.

### Multi-branch mask and blending

The last piece protects everything the user did *not* ask to change. A multimodal LLM reads the source image and the two prompts and emits an **edit type** $O$ (local / background / global) and a **region** described by points/box $P, Q$. From these a spatial mask $\mathcal{M}$ is built:

$$
\mathcal{M}(O, P, Q) =
\begin{cases}
\mathcal{S}(\mathcal{B}(P,Q)) & \text{if } O = \text{Local},\\
\mathbf{1} - \mathcal{S}(\mathcal{B}(P,Q)) & \text{if } O = \text{Background},\\
\mathbf{1} & \text{if } O = \text{Global},\\
\mathcal{B}(P,Q) & \text{otherwise,}
\end{cases}
$$

where $\mathcal{B}$ is the bounding box and $\mathcal{S}$ is SAM segmentation refining the box into a tight object mask. Read the cases plainly: a **local** edit masks in just the object; a **background** edit inverts that (edit everything *but* the object); a **global** edit (e.g. restyle) uses the whole frame $\mathbf{1}$.

Then at each step the target latent is spliced: **keep the source (reconstruction) latent outside the mask, keep the edited latent inside it:**

$$
Z_{t+1}^{tar} = Z_{t+1}^{src} \odot (\mathbf{1} - \mathcal{M}) + Z_{t+1}^{tar} \odot \mathcal{M},
$$

where $\odot$ is elementwise (per-pixel) multiplication. This is why, in the teaser, removing the bird or swapping "STOP"→"ICML" leaves the surrounding sky and sign pixel-identical: those regions are literally copied from the drift-free source branch.

### Putting it together (algorithm)

**Inversion phase** — walk the image to noise once, recording the true displacement at every step (here the step index runs $0$ = image to $T$ = noise):

$$
\begin{aligned}
&Z_0^{inv} \gets \text{VAE-Encode(source image)} &&\text{\small(image end)}\\
&\textbf{for } t = 0,\dots,T-1: &&\text{\small(walk image}\to\text{noise)}\\
&\quad Z_{t+1}^{inv} \gets Z_t^{inv} - (\sigma_{t+1}-\sigma_t)\, v_\theta(Z_t^{inv},\, \psi_{src})\\
&\quad \Delta Z_t \gets Z_{t+1}^{inv} - Z_t^{inv} &&\text{\small(store residual, free)}
\end{aligned}
$$

**Editing phase** — run two aligned branches from the noise back to the image:

$$
\begin{aligned}
&O, P, Q \gets \text{MLLM}(\text{image},\, \psi_{src},\, \psi_{tar}); \quad \mathcal{M} \gets \text{mask}(O, P, Q)\\
&Z^{src},\, Z^{tar} \gets Z_T^{inv} &&\text{\small(both branches start at the noise)}\\
&\textbf{for } t = T-1,\dots,0: &&\text{\small(walk noise}\to\text{image)}\\
&\quad \hat{Z}^{src} \gets Z^{src} + \Delta Z_t &&\text{\small(align onto inversion trajectory)}\\
&\quad \hat{Z}^{tar} \gets Z^{tar} + \Delta Z_t\\
&\quad v^{tar} \gets v_\theta(\hat{Z}^{tar},\, \psi_{tar}) \ \text{ with } V^{src}\text{ injected if } t < t_{inj}\\
&\quad Z^{src} \gets Z^{src} + (\sigma_{t+1}-\sigma_t)\, v_\theta(\hat{Z}^{src},\, \psi_{src})\\
&\quad Z^{tar} \gets Z^{tar} + (\sigma_{t+1}-\sigma_t)\, v^{tar}\\
&\quad Z^{tar} \gets Z^{src} \odot (\mathbf{1}-\mathcal{M}) + Z^{tar} \odot \mathcal{M} &&\text{\small(protect untouched regions)}\\
&\textbf{return } \text{VAE-Decode}(Z^{tar})
\end{aligned}
$$

Four ideas, in order of importance: **residual injection** (kills step-level error, the whole point), **dual branches** (a clean reference to borrow from), **value injection** (borrow appearance, keep prompt control), **mask blending** (protect the rest). The first is what makes the other three trustworthy.

---

## What the edits look like, and the error curve

The mechanism figure below (a zoom of the DirectEdit geometry) shows the same story as Figure 2c but with the pieces labelled: the recorded **Residual $\Delta Z_t$** is what nudges the state along the inversion path near the noise end, producing the **Ideal Feature** interaction (green) between the source reconstruction and the editing path, with the target velocity $\hat v_{tar}$ (green/pink) driving the blue dashed editing path toward the target image.

![The DirectEdit mechanism schematic: Gaussian Noise at top, Source Image (green) at bottom-left, Target Image (orange) at bottom-right. A magenta arrow labelled Residual delta-Z-t runs along the top of the inversion path; a green arrow labelled Ideal Feature connects the reconstruction and editing latents; target velocity arrows v-hat-tar and another delta-Z-t label sit along the blue dashed editing path descending to the target image.](./assets/direct_edit_x7.jpg){ width=62% }

*The DirectEdit mechanism (arXiv 2605.02417): the recorded residual $\Delta Z_t$ keeps the reconstruction on the inversion path, enabling drift-free ("Ideal") feature interaction between the source and target branches.*

**Numbers.** On PIE-Bench (700 images, 9 edit categories) with FLUX.1-dev as the backbone, DirectEdit reports the best structure preservation (Structure Distance ≈ 17.9), background fidelity (PSNR ≈ 32.6, LPIPS ≈ 35.5) and text alignment (CLIP ≈ 25.4) among the compared training-free methods. The headline diagnostic is the **step-level reconstruction error**: average MSE ≈ **0.0006** for DirectEdit versus ≈ **0.29** for stepwise correction — three orders of magnitude smaller — and all of it at the *same* number of NFEs as vanilla Euler. That gap is the quantitative version of "Error = 0" in Figure 2c.

---

## Three prior methods, briefly

DirectEdit's contribution is clearest against the methods it cites. Each of these is a different answer to the same question — *how do you get a faithful anchor to the real image?* — and DirectEdit's residual trick is best understood as the fourth answer.

### RF-Inversion (rectified stochastic differential equations)

*Rout et al., "Semantic image inversion and editing using rectified stochastic differential equations," arXiv [2410.10792](https://arxiv.org/abs/2410.10792).*

The first *efficient* inversion/editing method for rectified-flow models like FLUX. The problem it confronts is the same one from [Passage 1](#passage-1--inversion-based-methods-accumulate-error): running the reverse RF field to turn an image into noise is inaccurate, and re-generating from that noise doesn't reproduce the image. Earlier fixes optimized the latent or the prompt embedding at inference — slow and fragile. RF-Inversion instead *derives* a better inversion field in closed form, so no per-image optimization is needed.

The idea is to steer inversion with a **controlled vector field** that blends two objectives: (i) *stay faithful to the input image*, via the ordinary reverse RF field $u_t(Y_t)$, and (ii) *stay on the clean-image distribution*, via a conditional field $u_t(Y_t \mid y_1)$ that points toward the target sample $y_1$. That second field is not hand-designed — it is the solution of a **Linear Quadratic Regulator (LQR)** optimal-control problem, the classic "drive a system to a target state at minimum cost" setup, which happens to have a clean closed form for the linear RF dynamics. The controlled ODE is their interpolation of the two:

$$
dY_t = \big[\,u_t(Y_t) + \gamma\,\big(u_t(Y_t \mid y_1) - u_t(Y_t)\big)\,\big]\,dt.
$$

Read the bracket as "the faithful field, plus a $\gamma$-weighted correction toward the controlled target field." The single knob $\gamma \in [0,1]$ sets the trade-off: $\gamma=0$ is pure faithful inversion (the noise exactly reconstructs the input), $\gamma=1$ makes the endpoint a clean standard Gaussian (enabling generation from a corrupted or stroke input), and intermediate values trade content-preservation for editability. The authors also show this controlled ODE has an **equivalent SDE** form — a drift-plus-diffusion version — and argue the stochasticity is what makes intermediate $\gamma$ robust to a bad starting image. Editing is then: invert with a null prompt to get the latent, then run the reverse process under the desired prompt with a second guidance knob for edit strength. **How DirectEdit differs:** RF-Inversion *guides* the inversion path with a control term, which is elegant but still leaves a per-step approximation gap (the control biases the field, it doesn't make the discrete step exact). DirectEdit instead *records and replays* the exact inversion displacements, so there is nothing to approximate — step-level error goes to zero with no control objective and no tuning knob.

### FlowEdit

*Kulikov et al., "FlowEdit: inversion-free text-based editing using pre-trained flow models," ICCV 2025, pp. 19721–19730.*

The representative **inversion-free** method, and the concrete target of [Passage 2](#passage-2--inversion-free-methods-still-drift). Its motivating observation: the standard "invert to noise, then re-generate" detour is wasteful and lossy — you push the image *all the way* to pure noise and back, accumulating error over the full trajectory, just to change one object. Why travel to noise at all if source and target images are *close*?

So FlowEdit never visits noise. It constructs a **direct ODE path from the source image to the edited image**, whose velocity at each point is the *expected difference* between the model's target-prompt and source-prompt velocities, evaluated at coupled interpolation states that share the same injected noise:

$$
v^{\text{FE}}(Z_t, t) = \mathbb{E}\big[\, v_\theta(Z_t^{tar}, t, \psi_{tar}) - v_\theta(Z_t^{src}, t, \psi_{src}) \,\big].
$$

The intuition: the source-prompt velocity is the "flow that would have made the source image," and the target-prompt velocity is the "flow that would make the target"; their *difference* is precisely the drift that carries you from one image to the other, while the shared-noise coupling cancels the parts of each flow that are about generic image structure rather than the edit. Integrating this difference-field from $t{=}0$ to $1$ moves the source latent directly to the edited latent along a *short* path, and averaging over several noise draws (the $\mathbb{E}$) suppresses variance. Because the path never reaches noise, there is far less trajectory over which error can build up. **How DirectEdit differs:** FlowEdit dodges inversion error by dodging inversion — but the direct path is built from *random noise* and is only loosely tied to the source, so it either keeps a residual stochastic error or, if pinned hard to the source, sacrifices editability (the exact tension of Passage 2). DirectEdit keeps a real inversion but makes it *exact*, so it gets a strong, deterministic source anchor **and** low error at once.

### Null-text inversion

*Mokady et al., "Null-text inversion for editing real images using guided diffusion models," CVPR 2023, pp. 6038–6047.*

The classic diffusion-era (pre-flow) editor and the intellectual ancestor of the whole "make inversion faithful so editing can lean on it" program — DirectEdit is doing the flow-era version of the same job, which is why it's worth understanding in a bit more depth.

**The problem it solves.** Prompt-to-Prompt editing works by manipulating a diffusion model's *attention maps along its denoising trajectory*, so to edit a *real* photo you first need that trajectory — i.e. you must invert the image into the sequence of latents the model would have produced. The natural tool is **DDIM inversion**, the deterministic reverse of DDIM sampling. It works acceptably *only when classifier-free guidance is off* (guidance scale $w=1$). But meaningful text-driven editing *requires* strong guidance ($w \gg 1$), and here's the trap: DDIM inversion is a first-order approximation, so it carries a small per-step error, and **classifier-free guidance amplifies that error at every step**. With guidance on, the errors blow up and the reconstruction no longer looks like the input — so you can't even reproduce the original, let alone edit it.

**Idea 1 — pivotal inversion.** Rather than trying to invert perfectly, run the cheap DDIM inversion *once* (with $w=1$) to get a rough but coherent trajectory of latents $z_T^*, z_{T-1}^*, \dots, z_0^*$, and treat this single trajectory as a **pivot** — a reference the real (guided) sampling should hug. Optimizing around one pivot trajectory is far more efficient and stable than optimizing around random noise, because there is a single, consistent target to aim at.

**Idea 2 — optimize only the null-text embedding.** Now we want guided ($w \gg 1$) denoising, started from the pivot's noise $z_T^*$, to actually retrace the pivot. We could force this by tuning the model weights (breaks the model), or the conditional prompt embedding (breaks editability, since the prompt must stay meaningful). Null-text inversion picks the one lever that is otherwise "wasted": the **unconditional / null-text embedding** $\varnothing$. Recall the CFG update combines a conditional and an unconditional prediction,

$$
\tilde{\epsilon}_\theta = \epsilon_\theta(z_t, t, \varnothing) + w\,\big(\epsilon_\theta(z_t, t, c) - \epsilon_\theta(z_t, t, \varnothing)\big),
$$

so $\varnothing$ enters every guided step and has plenty of leverage over the outcome. The method optimizes a **separate null embedding $\varnothing_t$ for each timestep**, minimizing the gap between the guided step and the pivot,

$$
\min_{\varnothing_t}\; \big\| z_{t-1}^* - z_{t-1}(\bar{z}_t, \varnothing_t) \big\|^2,
$$

where $z_{t-1}(\cdot)$ is one guided DDIM step. In words: at each timestep, nudge only the null embedding until the guided reconstruction lands back on the pivot latent. The real prompt $c$ and the model weights stay frozen throughout; only these per-timestep null vectors are learned, and they are then reused at edit time. The payoff is high-fidelity reconstruction of a real image *with guidance on*, which is exactly what Prompt-to-Prompt needs to produce clean, controllable edits — at the cost of a short per-image optimization (a handful of gradient steps per timestep, on the order of a minute).

**How DirectEdit differs.** Null-text inversion and DirectEdit share the same instinct — *pin the guided/reconstruction path to a faithful reference* — but the mechanism could not be more different. Null-text inversion reaches faithfulness by *learning*: a per-timestep optimization loop, specific to the diffusion + DDIM + CFG setting, that costs real wall-clock time per image. DirectEdit reaches it by *bookkeeping*: it records the true inversion displacement $\Delta Z_t$ and adds it back with a single subtraction — no gradients, no per-image optimization, no extra NFEs — and it is built for rectified-flow models. In effect, DirectEdit is what Null-text inversion's goal looks like when the underlying model is a flow and the exact trajectory can simply be *stored* instead of *re-fit*.

---

## Takeaways

- **Training-free editing = invert, then re-generate under a new prompt.** All the difficulty is in making inversion faithful.
- **Inversion is approximate** because the exact backward step needs the velocity at the unknown previous state; using the current state instead introduces a per-step error that **compounds**.
- **Stepwise correction reduces but never zeroes** that error — it fixes the *state after* a wrong velocity, so drifted features keep leaking into edits.
- **DirectEdit zeroes it** by aligning the *input* to the velocity: record the true inversion displacement $\Delta Z_t$ and add it back before each network call. Same NFEs as plain Euler, step-level error ≈ 0.
- The rest — **dual branches, attention Value injection, mask blending** — turns that drift-free reconstruction into a controllable edit that protects everything you didn't ask to change.

---

## Sources

- **DirectEdit: Step-Level Accurate Inversion for Flow-Based Image Editing.** arXiv [2605.02417](https://arxiv.org/pdf/2605.02417) · [ar5iv HTML](https://ar5iv.labs.arxiv.org/html/2605.02417) · [code](https://github.com/Tr1stesse/DirectEdit). (Figures 2, teaser, and the mechanism schematic are from this paper/repo.)
- Rout et al., **Semantic image inversion and editing using rectified stochastic differential equations (RF-Inversion).** arXiv [2410.10792](https://arxiv.org/abs/2410.10792).
- Kulikov et al., **FlowEdit: inversion-free text-based editing using pre-trained flow models.** ICCV 2025, pp. 19721–19730.
- Mokady et al., **Null-text inversion for editing real images using guided diffusion models.** CVPR 2023, pp. 6038–6047.
- Prerequisites in this repo: [`ddpm_ddim_flow_score.md`](./ddpm_ddim_flow_score.md) (rectified flow, velocity, Euler) and [`classifier-free-guidance.md`](./classifier-free-guidance.md) (prompt-conditioned velocity).
