# Wavelets for Image Processing

Suppose you are handed an image and asked a simple question: *where are the edges, and how big are they?* A sharp eyelash is a tiny, high-frequency feature living at one precise spot. The slow shading across a cheek is a low-frequency feature spread over a huge region. A good representation of the image should be able to talk about **both** kinds of structure at once — the fine and the coarse, the local and the global — and, crucially, tell you *where* each one is.

This is exactly what the classical tool, the **Fourier transform**, cannot do well. Fourier tells you *which frequencies* are present in an image but throws away *where* they are: a single bright edge gets smeared across the entire frequency spectrum. **Wavelets** were invented to fix precisely this. The one-line idea, which everything below unpacks, is:

> **Analyze a signal with little wave-shaped "probes" that you can shrink, stretch, and slide.** Small probes pin down fine, high-frequency detail at exact locations; large probes capture slow, low-frequency trends over wide regions. You get *frequency* and *location* together.

This note builds wavelets from zero — assuming you know nothing beyond "an image is a grid of pixel values." We start from why Fourier is not enough, define what a wavelet actually is, work a full example by hand (the Haar wavelet), see how the whole thing becomes a cascade of simple filters, extend it to 2D images, and finish with the three big applications: **compression** (JPEG 2000), **denoising**, and **feature detection**.

---

## Table of Contents

- [Setup and Notation](#setup-and-notation)
- [Why Fourier is not enough](#why-fourier-is-not-enough)
- [The key idea: probe at many scales](#the-key-idea-probe-at-many-scales)
- [What exactly is a wavelet?](#what-exactly-is-a-wavelet)
- [Continuous vs discrete: which one you actually use](#continuous-vs-discrete-which-one-you-actually-use)
- [The Haar wavelet: a full worked example](#the-haar-wavelet-a-full-worked-example)
- [Multiresolution: the filter-bank / pyramid view](#multiresolution-the-filter-bank--pyramid-view)
- [From 1D to 2D images: the subbands](#from-1d-to-2d-images-the-subbands)
- [Wavelet families: Haar, Daubechies, and friends](#wavelet-families-haar-daubechies-and-friends)
- [Application 1: image compression (JPEG 2000)](#application-1-image-compression-jpeg-2000)
- [Application 2: denoising by thresholding](#application-2-denoising-by-thresholding)
- [Application 3: edges, fusion, and other uses](#application-3-edges-fusion-and-other-uses)
- [Where wavelets sit today (vs Fourier and CNNs)](#where-wavelets-sit-today-vs-fourier-and-cnns)
- [Practical guidance and takeaways](#practical-guidance-and-takeaways)
- [Sources](#sources)

---

## Setup and Notation

A handful of symbols recur. Each is introduced again where it first does real work, but here they are in one place.

| Symbol | Meaning |
|---|---|
| $f(t)$ or $x[n]$ | The **signal** — continuous $f(t)$ for the theory, discrete samples $x[n]$ in practice. For an image, replace $t$/$n$ with 2D pixel coordinates. |
| $\psi(t)$ | The **mother wavelet** — the little "template" wave. Zero mean, localized (dies off away from the origin). |
| $\phi(t)$ | The **scaling function** — a companion "bump" that captures the local *average* (the low-frequency part) rather than the wiggle. |
| $s$ | **Scale** (or dilation): how much we stretch ($s>1$) or squeeze ($s<1$) the wavelet. Big scale = low frequency, coarse view. |
| $\tau$ | **Translation**: where along the signal we slide the wavelet. |
| $a_j,\ d_j$ | **Approximation** and **detail** coefficients at level $j$ — the coarse blur and the fine wiggle, respectively. |
| $L,\ H$ | The **low-pass** and **high-pass** filters that implement one wavelet step. |
| $\downarrow 2$ | **Downsampling** by 2: keep every other sample (throw away the redundant half). |
| LL, LH, HL, HH | The four **subbands** of one 2D wavelet step: approximation, horizontal, vertical, and diagonal detail. |

The mental model to hold onto: a wavelet transform never just lists frequencies. It repeatedly splits a signal into a **coarse average** plus the **detail you lost** by coarsening — and it does this at every scale, keeping track of *where* each piece of detail lives.

---

## Why Fourier is not enough

The Fourier transform writes any signal as a sum of **pure sine and cosine waves** of different frequencies. It answers "how much of each frequency is in this signal?" beautifully, and for many tasks (audio equalizers, solving differential equations, steady periodic signals) that is exactly right.

The trouble is that a sine wave is **infinitely long** — it stretches from $-\infty$ to $+\infty$ with the same amplitude everywhere. So the Fourier representation has **perfect frequency resolution but zero location resolution**. If your image has a single sharp edge at one spot, Fourier cannot say "there is a high frequency *here*"; it can only say "there is some high-frequency content *somewhere*," spreading that one edge across countless frequency components. For images — which are almost entirely made of *localized* events (edges, textures, corners, small objects) — this is a terrible fit.

The first patch was the **Short-Time Fourier Transform (STFT)**: chop the signal into short windows and Fourier-transform each window separately. Now you know roughly *when/where* each frequency occurs. But you are forced to pick **one window size for everything**, and that choice is a trap:

- A **short** window localizes *time/position* well but sees only a few oscillations, so it cannot distinguish nearby frequencies — poor frequency resolution.
- A **long** window resolves frequency finely but blurs *where* things happened — poor location resolution.

This is not a failure of cleverness; it is the **uncertainty principle** for signals: you cannot have arbitrarily sharp location *and* arbitrarily sharp frequency at the same time. The product of the two uncertainties has a floor. Any method just gets to *choose how to spend* its fixed budget across the time–frequency plane.

![Three panels tiling a time-frequency plane, x-axis time/position, y-axis frequency. (a) Raw samples: tall thin columns spanning the full frequency height, exact time but no frequency information. (b) STFT/Fourier window: a uniform 4x4 grid of identical rectangles, the same box shape everywhere. (c) Wavelet: a dyadic tiling with one wide short box at the bottom (low frequency), then rows of progressively shorter and more numerous boxes toward the top (high frequency). Every tile in all panels has the same area, the uncertainty limit.](../assets/wavelet_tiling.jpg)

The picture above is the single most important intuition in this whole note. Each method carves the **time–frequency plane** into tiles, and every tile has the *same area* (the uncertainty floor) — they just have different shapes:

- **Raw samples** (a): infinitely tall, zero-width tiles. You know exactly *where* every value is, but nothing about frequency.
- **STFT** (b): the same rectangle everywhere. You committed to one trade-off for all frequencies.
- **Wavelets** (c): **short-and-wide tiles at low frequency, tall-and-thin tiles at high frequency.** This is the wavelet's superpower. For slow trends it spends its budget on frequency precision (you don't need to know exactly *where* a slow trend is). For fast detail it spends its budget on location precision (a sharp edge needs a precise position, and you already know it's high-frequency). This *adaptive* tiling matches how real images and sounds are actually built.

That adaptive tiling — fine in position where the signal is fast, fine in frequency where the signal is slow — is the reason wavelets exist.

---

## The key idea: probe at many scales

Forget formulas for a moment. Here is the whole method as a physical procedure.

Take a little wave-shaped bump — call it a **probe**. Slide it along the signal and, at each position, ask: *how much does the signal locally look like my probe?* (Mathematically "how much it looks like" is just a dot product / correlation: multiply the two, point by point, and sum.) Where the signal wiggles the way the probe wiggles, you get a big number; where it doesn't, you get near zero.

Now do the clever part: **change the size of the probe and repeat.**

- A **tiny** probe only responds to fast, sharp changes — it lights up on fine edges and texture, and it tells you their exact position.
- A **big** probe only responds to slow, broad changes — it lights up on gradients and large structures, but is vague about exact position (which is fine, because broad structures don't have a sharp "location" anyway).

Sweep through a range of probe sizes and, at each size, slide across every position. The output is a stack of "response maps," one per scale. Together they say, for every location and every scale, *how much structure of that scale sits at that spot.* That stack **is** the wavelet transform. Everything else — the specific probe shapes, the fast algorithms, the 2D extension — is engineering on top of this one idea.

Two knobs control the probe, and they are the two fundamental wavelet operations:

- **Scale $s$** — stretch or squeeze the probe. This tunes *which frequency* it responds to.
- **Translation $\tau$** — slide the probe. This tunes *which location* it looks at.

---

## What exactly is a wavelet?

A **wavelet** is the probe: a function $\psi(t)$ (the Greek letter "psi") that satisfies two requirements, both of which come straight from the intuition above.

1. **It is localized** — it is a bump that is large near the origin and decays to (essentially) zero away from it. This is what makes it able to pin down *location*, unlike an infinite sine. The name literally means "small wave."
2. **It has zero mean** — $\int \psi(t)\,dt = 0$, so it wiggles up as much as it dips down. This is the "wave" part: it responds to *changes*, not to constant offsets. A flat gray region produces zero wavelet response; only variation registers. (This is called the **admissibility condition**; zero mean is its practical face.)

Below is a concrete one, the **Morlet wavelet** (a cosine inside a Gaussian envelope), shown as the template and then under the two operations.

![Three panels. (a) Mother wavelet psi(t): an oscillation packed inside a bell-shaped envelope, symmetric about t=0, wiggling above and below zero and dying out on both sides. (b) Scaling: the same shape at s=0.5 (squeezed narrow, more oscillations per unit length, high frequency), s=1 (the mother), and s=2 (stretched wide, slow oscillation, low frequency). (c) Translation: three copies of the mother wavelet centered at tau=-4, 0, and 4, identical in shape but slid to different positions.](../assets/wavelet_mother_scales.jpg)

From one **mother wavelet** $\psi$ you generate an entire family by scaling and translating it:

$$\psi_{s,\tau}(t) = \frac{1}{\sqrt{s}}\,\psi\!\left(\frac{t-\tau}{s}\right).$$

Unpacking each piece:

- $(t-\tau)$ shifts the center of the wavelet to position $\tau$ — that is the **slide**.
- Dividing the argument by $s$ **stretches** the wavelet: a large $s$ makes it wide and slow (low frequency), a small $s$ makes it narrow and fast (high frequency). Panel (b) shows exactly this.
- The front factor $\tfrac{1}{\sqrt{s}}$ is just **energy bookkeeping**: when you stretch a wavelet out, you lower its height by the right amount so that all sizes carry the same total "energy" and their responses are comparable. Without it, big wavelets would automatically produce bigger numbers regardless of the signal.

The wavelet transform coefficient at scale $s$ and position $\tau$ is then the correlation we described in words:

$$W(s,\tau) = \int f(t)\,\psi_{s,\tau}(t)\,dt,$$

i.e. "slide probe $\psi_{s,\tau}$ over signal $f$ and record the overlap." Large $|W(s,\tau)|$ means "there is a lot of scale-$s$ structure right around $\tau$."

**One more character: the scaling function $\phi$.** The wavelet has zero mean, so by design it can *never* represent the overall average level of the signal — it only sees change. We need a partner that captures the leftover **low-frequency "DC" content**: a smooth bump $\phi(t)$ with *nonzero* mean (a local averaging window). In practice the pair travels together: $\phi$ grabs the **coarse average** (the blur), $\psi$ grabs the **detail** (the wiggle you lost by averaging). Keep this pairing in mind — it is the engine of the fast algorithm below, where $\phi$ becomes a low-pass filter and $\psi$ a high-pass filter.

---

## Continuous vs discrete: which one you actually use

There are two flavors, and it is worth knowing which is which.

**Continuous Wavelet Transform (CWT).** Let $s$ and $\tau$ vary *continuously* over a fine grid of scales and positions. You get a lush, highly redundant 2D picture $W(s,\tau)$ called a **scalogram** — great for *analysis and visualization* (spotting transients in an EKG, chirps in a seismic trace, rhythms in audio) because nearby scales overlap and nothing is missed. The cost: it is redundant and expensive, and you generally can't reconstruct the signal cheaply. Think of the CWT as a **microscope for looking**, not a codec.

**Discrete Wavelet Transform (DWT).** Here is the practical workhorse for images. Instead of a fine continuum, sample the scales and positions on a smart, sparse **dyadic grid**: scales go in powers of two ($s = 1, 2, 4, 8,\dots$) and at each scale you translate in steps matched to that scale. This specific sampling is not arbitrary — with the right wavelet it is *exactly enough* to represent the signal with **no redundancy and no loss**: the number of output coefficients equals the number of input pixels, and you can reconstruct the original perfectly. That efficiency (same size, invertible, fast) is why every image codec, denoiser, and multiresolution pipeline uses the DWT. The rest of this note is about the DWT.

---

## The Haar wavelet: a full worked example

The simplest wavelet of all is the **Haar wavelet** (Alfréd Haar, 1909 — it predates the word "wavelet" by decades). It makes every idea above concrete with nothing but **averages and differences**, so let's do one step fully by hand.

Take an 8-sample signal. One Haar step walks through it in **non-overlapping pairs** and, for each pair $(a, b)$, computes two numbers:

- the **average** $\dfrac{a+b}{2}$ — the local coarse value (this is $\phi$, the scaling function, at work), and
- the **difference** $\dfrac{a-b}{2}$ — how much the pair deviates from its own average, i.e. the local detail (this is $\psi$, the wavelet).

![Three panels. (a) An 8-sample signal drawn as a staircase: values 4,6,10,12,8,6,5,5, rising to a peak of 12 then falling. (b) Approximation: 4 samples, the pairwise averages 5,11,7,5, a coarser staircase that tracks the overall shape at half the resolution. (c) Detail: 4 bars showing the pairwise differences/2, small values near zero except where the signal changed sharply, marking the edges.](../assets/wavelet_haar_1d.jpg)

Concretely, with the signal $x = [4, 6, 10, 12, 8, 6, 5, 5]$:

- **Approximation** (averages of each pair): $\left[\tfrac{4+6}{2}, \tfrac{10+12}{2}, \tfrac{8+6}{2}, \tfrac{5+5}{2}\right] = [5, 11, 7, 5]$ — a half-length blurred copy of the signal.
- **Detail** (half-differences): $\left[\tfrac{4-6}{2}, \tfrac{10-12}{2}, \tfrac{8-6}{2}, \tfrac{5-5}{2}\right] = [-1, -1, 1, 0]$ — small everywhere the signal was locally smooth, and only nonzero where a pair disagreed.

Three things to notice, because they generalize to *every* wavelet:

1. **Nothing is lost.** From an average $m$ and a half-difference $d$ you recover the original pair exactly: $a = m + d,\ b = m - d$. So the transform is **perfectly invertible** — 8 numbers in, 8 numbers out (4 approximation + 4 detail), and you can rebuild the input. This is the "no redundancy, no loss" promise of the DWT, visible in the simplest possible case.

2. **The detail is mostly near zero — this is the whole game.** Real signals and images are *locally smooth*: most neighboring values are similar, so most differences are tiny. A representation whose coefficients are overwhelmingly near-zero is called **sparse**, and sparsity is the property every application below exploits — you can drop the small coefficients (compression), or zero out the noise-sized ones (denoising), with almost no visible damage.

3. **The big detail coefficients mark the edges.** The only place the detail spiked is where the signal genuinely jumped. Wavelet detail coefficients are, quite literally, a multi-scale **edge detector**.

Now recurse: apply the *same* averaging-and-differencing step to the 4-sample approximation $[5,11,7,5]$, producing a length-2 approximation and length-2 detail. Repeat again to get down to a single overall average. Each pass doubles the scale (you are now averaging over 2, then 4, then 8 samples) and captures the detail lost at *that* scale. After all passes, the original 8 samples are re-expressed as **one overall average plus detail coefficients at every scale** — coarse detail, medium detail, fine detail — each tagged with where it lives. That is a complete multiresolution wavelet decomposition.

The Haar wavelet is blocky (its probe is a single up-then-down square step), so it is not the smoothest choice for photographs, but every richer wavelet is just a fancier version of this exact average-and-difference dance.

---

## Multiresolution: the filter-bank / pyramid view

The "walk through pairs computing averages and differences" you just did by hand has a much more powerful description that unlocks fast algorithms and connects wavelets to ordinary signal processing. This is **Mallat's multiresolution analysis (1989)**, and it is how the DWT is *actually computed*.

The key realization: computing a local average at every pair is exactly **convolving with a low-pass filter** $L$ (a smoothing/blurring filter) and then keeping every other output; computing a local difference is **convolving with a high-pass filter** $H$ (an edge/change filter) and keeping every other output. "Keeping every other output" is **downsampling by 2** ($\downarrow 2$) — legitimate because after halving the frequency content you need only half as many samples (this is why the transform doesn't grow in size).

So **one wavelet step = pass the signal through two complementary filters and downsample each**:

![Schematic flow diagram. On the left, a box "signal x". Two arrows fan out to an upper branch and a lower branch. Upper branch: "high-pass H" then "down-arrow 2" then "detail D1" (the fine wiggle). Lower branch: "low-pass L" then "down-arrow 2" then "approx A1" (the coarse blur). A further arrow from A1 leads to a box "repeat on A1", with a note that only the approximation is split again.](../assets/wavelet_filterbank.jpg)

Reading the diagram:

- The signal $x$ goes down **two paths**. The **low-pass** path ($L$ then $\downarrow 2$) produces the **approximation** $A_1$ — the coarse, half-resolution blur, exactly the "averages" from the Haar example. The **high-pass** path ($H$ then $\downarrow 2$) produces the **detail** $D_1$ — the "differences," the information $A_1$ threw away.
- The two filters are designed as a matched pair (a *quadrature mirror* pair) so that between them they capture *all* the information: $A_1$ and $D_1$ together let you reconstruct $x$ perfectly by running the diagram in reverse (upsample, filter with the synthesis filters, add). Perfect reconstruction is a *design constraint* on the filters, not a happy accident.
- Then the **recursion**: you feed **only the approximation** $A_1$ back into the same two-filter box, producing $A_2$ (coarser blur) and $D_2$ (coarser detail). Repeat. Because each stage halves the data, the total work is $N + N/2 + N/4 + \dots \approx 2N$ — the whole multi-level transform is **linear time**, even faster than the FFT's $N\log N$.

This one-sided recursion (split the approximation, never re-split the detail) is why wavelets are sometimes called a **pyramid**: each level is half the size of the one below, and the detail coefficients accumulate at every level. It is also the precise meaning of "multiresolution" — you end up with a description of the image at a whole ladder of resolutions simultaneously, coarse to fine.

---

## From 1D to 2D images: the subbands

An image is 2D, so we apply the 1D machine **separably**: first run it along the **rows**, then run it along the **columns** of the result. Each direction splits into low ($L$) and high ($H$), and the two choices combine into **four** outputs per level — the famous **subbands**:

- **LL** — low-pass in both directions: the **approximation**, a clean half-size thumbnail of the image (the "blur").
- **LH** — low along rows, high along columns: responds to **horizontal edges** (intensity changing as you move vertically).
- **HL** — high along rows, low along columns: responds to **vertical edges**.
- **HH** — high in both: responds to **diagonal detail and corners**, where the image changes in both directions at once.

![A 2x3 grid. Top-left: the original grayscale "cameraman" photograph. Top-middle: LL, a visibly identical but half-size blurred thumbnail. Top-right: LH on a dark background with bright horizontal-ish edges (the horizon, shoulders, grass texture) glowing. Bottom-left: HL highlighting vertical structures (the tripod leg, building edges, the coat's vertical seams). Bottom-middle: HH lighting up diagonal edges and fine texture, especially the grass and camera. Bottom-right: the standard two-level pyramid layout, a bright small thumbnail in the top-left corner surrounded by three faint detail subbands, with the top-left quadrant itself subdivided into a second smaller level.](../assets/wavelet_2d_subbands.jpg)

Look at how cleanly the image separates by **orientation**: the tripod leg shows up in **HL** (it's vertical), the horizon and shoulders in **LH** (horizontal), and the grass texture — which changes every which way — scatters across **HH**. Meanwhile **LL** looks just like the original, only smaller. This directional decomposition is enormously useful: it is a built-in, multi-scale, orientation-selective edge analysis of the image, computed in linear time.

Then, as in 1D, you **recurse on LL only**. The second level splits the thumbnail into its own LL/LH/HL/HH, a quarter the size, and so on. Stacking these in the conventional layout gives the **wavelet pyramid** in the bottom-right panel: a small bright approximation in the corner, ringed by detail subbands at each scale. Almost all of those detail pixels are near-black — i.e. near zero — which is the sparsity we are about to cash in.

---

## Wavelet families: Haar, Daubechies, and friends

Haar is the simplest wavelet but rarely the best for photographs, because its blocky square-step probe reconstructs smooth regions with visible staircasing. The freedom in wavelet design is: *what shape should the probe be?* Different shapes trade off smoothness, support width (how localized), and symmetry. The must-know families:

- **Haar (db1).** The square step. Maximally localized (only 2 taps), perfectly symmetric, but discontinuous. Great for teaching, for sharp block-like data, and where speed matters most. It is the shortest **Daubechies** wavelet.
- **Daubechies (db2, db4, db8, …).** Ingrid Daubechies' 1988 family — the workhorses. They are the *shortest* filters that are still smooth and can exactly represent polynomial trends up to a chosen degree (the number of **vanishing moments**). More vanishing moments = smoother wavelet = detail coefficients even sparser on smooth images, at the cost of a wider, less localized probe. `db2`–`db4` are common defaults for natural images.
- **Symlets and Coiflets.** Tweaks of Daubechies for **near-symmetry** (symlets) or extra vanishing moments on the scaling function too (coiflets). Symmetry matters because asymmetric filters shift features slightly and produce lopsided artifacts around edges.
- **Biorthogonal wavelets (e.g. CDF 9/7 and 5/3).** Here the analysis and synthesis wavelets are *different* functions, which buys you **exact symmetry together with smoothness** — impossible for a single orthogonal wavelet beyond Haar. This is why **JPEG 2000** uses biorthogonal CDF **9/7** for lossy compression and the integer **5/3** for lossless. Symmetry avoids phase distortion; that's the whole reason they're chosen for images.

The practical takeaway: **Haar to understand, `db2`/`db4` or `sym4` as sensible defaults, biorthogonal 9/7 when you specifically need symmetry for compression.** Beyond that, wavelet choice is usually a second-order tuning knob.

---

## Application 1: image compression (JPEG 2000)

Compression is the poster application, and by now the mechanism should be obvious. Run a few levels of 2D DWT: you get one small LL thumbnail plus a pile of detail subbands that are **overwhelmingly near zero** (smooth image regions produce tiny details). So:

1. **Transform** the image with the DWT (biorthogonal 9/7).
2. **Quantize** — round coefficients to a coarse grid; the many near-zero ones snap to exactly zero.
3. **Encode** the surviving handful of large coefficients compactly (plus their locations).

To reconstruct, invert the DWT. The figure shows what happens if we keep only the largest few percent of coefficients and zero the rest:

![Four reconstructions of the cameraman image side by side. Keeping 100% of coefficients: pixel-perfect, PSNR ~120 dB. Keeping 10%: visually indistinguishable, PSNR ~35 dB. Keeping 2%: still clearly recognizable with slightly softened texture, PSNR ~28 dB. Keeping 0.5%: blurry but the overall figure, tripod, and horizon are still perfectly readable, PSNR ~23 dB.](../assets/wavelet_compression.jpg)

Keeping just **10%** of the coefficients is visually lossless; even **2%** is very usable and **0.5%** still recognizable. That graceful degradation — quality falling smoothly as you keep fewer coefficients — is exactly what a codec wants.

Why did **JPEG 2000** replace the DWT for the older **JPEG**'s block-DCT? Two wavelet advantages:

- **No block artifacts.** Classic JPEG splits the image into $8\times8$ blocks and transforms each independently, so at low bitrates you see the tell-tale blocky grid. The wavelet transform is applied to the *whole image* at once, so its errors are spread and smooth rather than blocky.
- **Natural quality scalability.** Because the DWT is inherently multiresolution, a JPEG 2000 file can be decoded to a low-res thumbnail by reading only the coarse coefficients, or progressively refined — one file, many resolutions and qualities, no re-encoding. This is prized in medical imaging, digital cinema, and satellite imagery.

(JPEG 2000 never displaced ordinary JPEG on the web for adoption reasons, not technical ones — but the wavelet compression story is the textbook success.)

---

## Application 2: denoising by thresholding

Denoising falls out of the same sparsity, plus one more observation. Consider what the DWT does to a **noisy** image. Real image structure — edges, textures — is *sparse*: it concentrates into a **few large** detail coefficients. Random noise is the opposite — it is spread evenly across the image, so it scatters into **many small** detail coefficients, roughly the same tiny magnitude everywhere.

That difference is the whole trick (Donoho & Johnstone's **wavelet shrinkage**, early 1990s):

1. **Transform** the noisy image with the DWT.
2. **Threshold** the detail coefficients: any coefficient smaller than a threshold $\lambda$ is assumed to be noise and set to zero; the large ones (real edges) are kept. Two variants: **hard thresholding** zeroes the small ones and leaves the rest untouched; **soft thresholding** zeroes the small ones *and* shrinks the survivors toward zero by $\lambda$, which removes the noise riding on top of the real coefficients too and gives smoother results.
3. **Invert** the DWT to get the clean image.

![Three panels. Left: the clean cameraman image. Middle: the same image buried in heavy grainy Gaussian noise, labeled PSNR 17.7 dB. Right: the wavelet soft-thresholded reconstruction, labeled PSNR 23.8 dB, visibly much cleaner with edges and the main figure preserved while the grain is largely gone, at the cost of slightly softened fine texture.](../assets/wavelet_denoise.jpg)

The reconstruction jumps from **17.7 dB** to **23.8 dB** with nothing but "zero the small detail coefficients." Note the tell-tale trade-off: some genuine fine texture (the grass) softens along with the noise, because the smallest real coefficients look just like noise and get culled too. The classic threshold choice is the **universal threshold** $\lambda = \sigma\sqrt{2\ln N}$ ($N$ = number of pixels, $\sigma$ = estimated noise level, itself read off from the median magnitude of the finest detail band, where almost everything *is* noise). This one idea — that signal is sparse and noise is dense in the wavelet domain — powered a decade of state-of-the-art denoisers before deep networks, and it still shows up inside many modern pipelines.

---

## Application 3: edges, fusion, and other uses

Because detail coefficients *are* localized, oriented, multi-scale edge responses, wavelets show up all over classical image processing beyond compression and denoising:

- **Edge and feature detection.** Large detail coefficients across scales mark edges; tracking which edges persist from coarse to fine scales separates true object boundaries (persistent) from noise (dies out quickly). This "multiscale edge" idea underlies a lot of texture analysis and corner/blob detection.
- **Image fusion.** To merge, say, a blurry-but-colorful image with a sharp-but-gray one (or multi-focus / multi-sensor shots), transform both, then combine subbands with a rule like "keep the larger detail coefficient" (sharper source wins the detail) and average the approximations. Invert to get a single image that is sharp *and* colorful. Common in remote sensing (**pan-sharpening**) and medical multi-modal imaging.
- **Watermarking and forensics.** Embedding a watermark in mid-level detail coefficients makes it invisible yet robust to mild compression; conversely, unusual wavelet-coefficient statistics can reveal tampering.
- **Texture classification.** The energy distribution across subbands (how much sits in fine vs coarse, horizontal vs diagonal) is a compact, rotation-aware texture descriptor — a staple feature before deep learning.

None of these need the full theory; they all reduce to "transform, manipulate the coefficients according to what each subband means, invert."

---

## Where wavelets sit today (vs Fourier and CNNs)

It helps to place wavelets among their neighbors:

- **vs Fourier:** Fourier is the right tool for *stationary, periodic* signals and for anything where you genuinely want global frequency content (filtering, convolution theorems). Wavelets win whenever the interesting content is **localized and transient** — which is essentially all natural images. The rule of thumb: *Fourier for what frequencies, wavelets for what frequencies **and where**.*
- **vs learned CNNs:** A single wavelet step is strikingly similar to a convolutional layer — filter, then downsample (a wavelet's $L$/$H$ filters are a fixed, hand-designed version of a conv layer's learned kernels + pooling). The difference is that CNN filters are **learned from data** and stacked with nonlinearities, so on large datasets they outperform fixed wavelets at recognition. But wavelets remain valuable where you have **little or no training data**, need **interpretability and guarantees** (perfect reconstruction, known behavior), or want **cheap, principled multiresolution** structure. Hybrids exist — **scattering transforms** (Mallat, 2012) stack wavelet transforms with nonlinearities to get CNN-like, translation-stable features *without training*, useful in low-data regimes; and wavelet-based downsampling/upsampling blocks appear inside modern networks to reduce aliasing.

The honest summary: wavelets are no longer the state of the art for high-level vision, but they remain a fundamental, efficient, interpretable representation — and understanding them is the cleanest way to build intuition for multiresolution, sparsity, and the time/space–frequency trade-off that pervades all of signal and image processing.

---

## Practical guidance and takeaways

- **The one idea:** analyze with little waves you can shrink, stretch, and slide, so you learn *what frequency* **and** *where* — the adaptive time/space–frequency tiling that Fourier can't give you.
- **The mechanism:** repeatedly split the image into a coarse **approximation** (low-pass + downsample) and the **detail** it lost (high-pass + downsample), recursing on the approximation. In 2D that yields the **LL / LH / HL / HH** subbands — a blur plus horizontal, vertical, and diagonal edges — at every scale.
- **Why it's useful:** natural images are **sparse** in the wavelet domain (mostly near-zero details). Compression keeps the few big coefficients; denoising zeroes the small ones; both are one-liners once you've transformed.
- **Defaults:** `Haar` to learn, `db2`/`db4` or `sym4` for general images, biorthogonal **9/7** (lossy) or **5/3** (lossless) for compression. Use the **CWT/scalogram** for *looking* at signals, the **DWT** for *processing* them.
- **Perspective:** think of a wavelet step as a fixed, invertible cousin of a conv-plus-pool layer. Learned CNNs beat it on recognition given data; wavelets win on interpretability, guarantees, and low-data efficiency.

---

## Sources

- S. Mallat, *A Wavelet Tour of Signal Processing* (3rd ed., 2008) — the standard reference; Chapters 1–7 cover everything here rigorously.
- I. Daubechies, *Ten Lectures on Wavelets* (1992) — the foundational text on compactly supported wavelets.
- S. Mallat, "A theory for multiresolution signal decomposition: the wavelet representation," *IEEE PAMI*, 1989 — the filter-bank / pyramid algorithm.
- D. Donoho & I. Johnstone, "Ideal spatial adaptation by wavelet shrinkage," *Biometrika*, 1994 — wavelet denoising by thresholding.
- ISO/IEC 15444, **JPEG 2000** standard; and the [PyWavelets documentation](https://pywavelets.readthedocs.io/) for hands-on `dwt2` / `wavedec2` experiments (used to make the figures in this note).
- The figures here were generated by `scripts/make_wavelet_figs.py` in this repo.
