# t-SNE: Visualising High-Dimensional Data in 2D

Modern data lives in far more than three dimensions. A word embedding is a 300-dimensional vector; an image passed through a network becomes a 512- or 2048-dimensional feature; a single cell in a genomics experiment is described by 20,000 gene expression levels. We cannot *see* any of this directly — our eyes handle two dimensions comfortably, three with effort. So we constantly want a faithful **2D map**: a scatter plot of dots, one per data point, where dots that were *similar* in the original high-dimensional space end up *near each other* on the page, and dissimilar ones end up apart. If such a map exists, a glance reveals the structure — clusters of similar images, families of related words, cell types — that was invisible in the raw numbers.

**t-SNE** (t-distributed Stochastic Neighbor Embedding, van der Maaten & Hinton, 2008) is the workhorse method for producing exactly this kind of map. It is the tool behind most of those colourful "blobs of clusters" plots you have seen for MNIST digits or single-cell data. Its one guiding idea, which everything below elaborates, is deceptively simple:

> **Preserve *who is near whom*, not the exact distances.** Get the local neighbourhoods right and don't worry about the global geometry.

This note builds t-SNE from the ground up — why the obvious linear method (PCA) is not enough, how t-SNE turns distances into "neighbour probabilities," what the mysterious *perplexity* knob does, why there is a "t" in the name (the *crowding problem*), and — crucially — how to *read* a t-SNE plot without fooling yourself.

---

## Table of Contents

- [Setup and Notation](#setup-and-notation)
- [Why not just use PCA?](#why-not-just-use-pca)
- [Step 1: turn distances into neighbour probabilities](#step-1-turn-distances-into-neighbour-probabilities)
- [Perplexity and the per-point bandwidth](#perplexity-and-the-per-point-bandwidth)
- [Step 2: the same idea in the 2D map, and the cost function](#step-2-the-same-idea-in-the-2d-map-and-the-cost-function)
- [The crowding problem and why Student-t](#the-crowding-problem-and-why-student-t)
- [The gradient as attraction and repulsion](#the-gradient-as-attraction-and-repulsion)
- [Optimisation tricks that actually matter](#optimisation-tricks-that-actually-matter)
- [Making it scale: Barnes-Hut](#making-it-scale-barnes-hut)
- [How to read a t-SNE plot (pitfalls)](#how-to-read-a-t-sne-plot-pitfalls)
- [t-SNE vs UMAP](#t-sne-vs-umap)
- [Practical guidance and takeaways](#practical-guidance-and-takeaways)
- [Sources](#sources)

---

## Setup and Notation

A handful of symbols recur throughout. Each is reintroduced where it first does real work, but here they are in one place.

| Symbol | Meaning |
|---|---|
| $x_i$ | The $i$-th **data point in the original high-dimensional space** (e.g. a 512-dim image feature). This is the input. |
| $y_i$ | The **map point**: where $x_i$ lands in the 2D plot. These are what t-SNE solves for. |
| $N$ | Number of data points. |
| $\|x_i - x_j\|$ | Euclidean distance between two original points. |
| $p_{j\mid i}$ | **Directional neighbour probability**: how likely point $i$ is to "pick" $j$ as a neighbour, in the high-dim space. |
| $p_{ij}$ | **Symmetric joint** neighbour probability in the high-dim space (a tidied-up version of $p_{j\mid i}$). |
| $q_{ij}$ | The corresponding neighbour probability in the **2D map**. t-SNE tries to make $q_{ij}\approx p_{ij}$. |
| $\sigma_i$ | The **bandwidth** (width) of the Gaussian centred on $x_i$ — set per-point via perplexity. |
| Perplexity | A user-chosen knob, roughly the **effective number of neighbours** each point considers (typical 5–50). |
| $\mathrm{KL}(P\,\|\,Q)$ | **Kullback–Leibler divergence** — the cost measuring how far the map's neighbour distribution $Q$ is from the true $P$. |

The mental model to hold onto: t-SNE never tries to match *distances*. It converts distances into *probabilities of being neighbours*, both in the original space ($p$) and in the map ($q$), and then it shuffles the map points $y_i$ around until those two probability pictures agree.

---

## Why not just use PCA?

Before t-SNE, the default tool for "squash high-dim data into 2D" was **PCA** (Principal Component Analysis). PCA finds the two directions along which the data varies the most and projects onto them. It is fast, deterministic, and preserves *large-scale* variance — the overall spread of the cloud.

The problem: PCA is **linear**. It can only rotate and flatten; it cannot bend. So whenever the interesting structure is *curved* — points lying on a twisted sheet or a curled manifold — a flat projection smears together things that were actually far apart along the surface. The classic example is the **swiss roll**: a 2D sheet rolled up like a pastry. Neighbours along the sheet are what matter, but PCA looks straight through the roll and overlaps the outer and inner layers.

![Three panels on the same 1500 points. Left: the original swiss-roll data, a 2D sheet curled up in 3D, coloured by position along the roll from purple to red. Middle: the PCA projection, where the spiral is seen edge-on so far-apart colours (e.g. red and blue) overlap and mix. Right: the t-SNE embedding, which pulls the roll into ordered strips where the colour gradient flows smoothly, showing neighbourhoods were preserved.](../assets/tsne_pca_vs_tsne.jpg)

t-SNE takes the opposite stance. It ignores the global shape entirely and obsesses over the question: *for each point, which handful of other points are its nearest neighbours?* If it can lay out the 2D map so that every point keeps roughly the same close neighbours it had in high-dim, the curved structure unrolls on its own. This is why t-SNE is called a **non-linear** dimensionality reduction / manifold learning method — it can follow bends that PCA cannot.

The trade-off, which we return to in [pitfalls](#how-to-read-a-t-sne-plot-pitfalls), is that by throwing away global geometry, t-SNE gives you a map where *local* neighbourhoods are trustworthy but *global* relationships (how far apart two clusters "really" are) are not.

---

## Step 1: turn distances into neighbour probabilities

t-SNE's first move is to convert the raw high-dim distances into a soft notion of "neighbourliness." Fix a point $x_i$. Place a **Gaussian bell curve** centred on it. Now ask: if I were standing at $x_i$ and reached out to grab a neighbour, with the probability of grabbing $x_j$ falling off like that Gaussian with distance, how likely am I to pick $x_j$? That likelihood is $p_{j\mid i}$:

$$
p_{j\mid i} = \frac{\exp\!\big(-\|x_i - x_j\|^2 / 2\sigma_i^2\big)}{\sum_{k \ne i} \exp\!\big(-\|x_i - x_k\|^2 / 2\sigma_i^2\big)}.
$$

Let us read this term by term, because everything else builds on it:

- **The numerator** $\exp(-\|x_i-x_j\|^2/2\sigma_i^2)$ is a Gaussian evaluated at the distance from $i$ to $j$. Nearby points ($\|x_i-x_j\|$ small) give a value close to 1; distant points give a value near 0. So it is a raw "closeness score" that decays smoothly with distance.
- **$\sigma_i$** is the **width** of that Gaussian — how far the notion of "near" extends around $x_i$. A larger $\sigma_i$ means even fairly distant points still count as neighbours. We will see in the next section why each point gets *its own* $\sigma_i$.
- **The denominator** sums the closeness scores over *all* other points $k$. Dividing by it turns the scores into a proper probability distribution: the $p_{j\mid i}$ over all $j$ sum to 1. In other words, it is a **softmax over negative squared distances** — the same softmax you know from classification, here applied to "how much does $i$ want $j$ as a neighbour."

So for each point $i$ we get a little probability distribution over all the other points, sharply peaked on its true near neighbours. This is the high-dim "ground truth" of neighbourliness that the 2D map will have to reproduce.

One tidy-up step: $p_{j\mid i}$ is **directional** (asymmetric) — because $\sigma_i \ne \sigma_j$, the chance $i$ picks $j$ differs from the chance $j$ picks $i$. That asymmetry is awkward (an outlier point can end up mattering to no one). t-SNE symmetrises into a single **joint** probability

$$
p_{ij} = \frac{p_{j\mid i} + p_{i\mid j}}{2N},
$$

which just averages the two directions and normalises so all the $p_{ij}$ together sum to 1. Intuitively, $p_{ij}$ is now a single symmetric "these two are neighbours" weight for the *pair*. The $1/N$ guarantees every point contributes something, so no point can be completely ignored.

---

## Perplexity and the per-point bandwidth

Why does each point need its own Gaussian width $\sigma_i$? Because real data has **dense and sparse regions**. In a crowded region, a point has many close neighbours, and a small $\sigma_i$ suffices to capture them. In a sparse region, the nearest neighbours are far away; a small $\sigma_i$ would make *every* $p_{j\mid i}$ essentially zero (the point would think it has no neighbours), so it needs a larger $\sigma_i$. A single global width would over-smooth the dense areas and blind the sparse ones.

t-SNE sets each $\sigma_i$ **automatically** so that every point ends up with roughly the *same effective number of neighbours*. That target number is called the **perplexity**. Formally, perplexity is $2^{H(P_i)}$ where $H(P_i)$ is the entropy of the distribution $p_{\cdot\mid i}$; informally, **perplexity ≈ the effective count of neighbours each point pays attention to.** You pick perplexity (say 30), and for each point t-SNE does a quick **binary search** on $\sigma_i$ until that point's distribution has the requested effective neighbour count. Dense points get a small $\sigma_i$, sparse points a large one — but all end up "looking at" about 30 neighbours.

Perplexity is *the* knob of t-SNE, and it genuinely changes the picture. Too small, and the map fixates on tiny local pockets and looks grainy or fragmented. Too large, and each point's neighbourhood blurs across the whole dataset, so distinct clusters get smoothed together and compressed. Typical useful values are **5 to 50**; the default is often 30.

![Three t-SNE embeddings of the same four overlapping 10-dimensional clusters at perplexity 5, 30, and 100. At perplexity 5 the clusters are grainy and locally fragmented; at 30 they form four clean well-separated blobs; at 100 they become compact and more evenly spaced as the neighbourhood smoothing increases.](../assets/tsne_perplexity_sweep.jpg)

The practical lesson, echoed in the pitfalls section: **run t-SNE at a few perplexities** and only trust structure that is stable across them.

---

## Step 2: the same idea in the 2D map, and the cost function

Now the 2D map. We have map points $y_1, \dots, y_N$ (initially random). We define neighbour probabilities $q_{ij}$ among them by *exactly the same "distance → soft neighbour" recipe*, so that $p$ and $q$ are comparable quantities. (The one twist — using a different, heavy-tailed curve for $q$ instead of a Gaussian — is the subject of the [next section](#the-crowding-problem-and-why-student-t); for now just picture the same kind of similarity built from the map distances $\|y_i - y_j\|$.)

The whole optimisation is then: **move the $y_i$ around until the map's neighbour distribution $Q = \{q_{ij}\}$ matches the true distribution $P = \{p_{ij}\}$.** We measure the mismatch with the **Kullback–Leibler (KL) divergence**, which is the standard way to score how far one probability distribution is from another:

$$
\mathcal{C} = \mathrm{KL}(P\,\|\,Q) = \sum_{i \ne j} p_{ij} \, \log \frac{p_{ij}}{q_{ij}}.
$$

Reading it: for every pair $(i,j)$ we compare the true neighbourliness $p_{ij}$ against the map's $q_{ij}$. The term is zero when they agree ($\log 1 = 0$) and grows as they diverge. Summing over all pairs gives one scalar cost $\mathcal{C}$ that we minimise by nudging the $y_i$.

The subtle but *essential* feature of KL is that it is **asymmetric**, and that asymmetry is precisely what makes t-SNE preserve *local* structure. Look at where the penalty is large:

- If two points are **true neighbours** ($p_{ij}$ large) but the map puts them **far apart** ($q_{ij}$ small), then $\log(p_{ij}/q_{ij})$ is large *and* it is weighted by the large $p_{ij}$ — a **big penalty**. So t-SNE fights hard to keep genuine neighbours together.
- If two points are **not** neighbours ($p_{ij}\approx 0$) but the map places them **close** ($q_{ij}$ large), the cost term is weighted by $p_{ij}\approx 0$ — a **tiny penalty**. t-SNE barely cares about accidentally placing strangers near each other.

In short: KL punishes *tearing neighbours apart* far more than it punishes *crushing strangers together*. That is a deliberate design choice — it is exactly why t-SNE guards local neighbourhoods faithfully while being cavalier about global layout.

---

## The crowding problem and why Student-t

Here is the single most important idea in the paper, and the reason for the "t" in the name.

**The crowding problem.** High-dimensional space has an absurd amount of "room." In, say, 50 dimensions you can have *many* points all mutually at a moderate distance from a central point — there is enough space for a large shell of neighbours to coexist without piling on top of each other. When you try to reproduce that in 2D, there simply is not enough area: the 2D plane cannot hold all those moderately-distant neighbours at the *same* moderate distance. If we used a Gaussian for $q$ (as the original "SNE" did), the only way to fit the many true neighbours nearby would be to also pull the moderately-distant points inward, and everything **collapses into a crowded blob** in the middle. The map crushes together.

**The fix: give the map a heavy tail.** Instead of a Gaussian, t-SNE builds $q_{ij}$ from a **Student-t distribution with one degree of freedom** (equivalently a Cauchy distribution):

$$
q_{ij} = \frac{\big(1 + \|y_i - y_j\|^2\big)^{-1}}{\sum_{k \ne l}\big(1 + \|y_k - y_l\|^2\big)^{-1}}.
$$

The numerator $(1+\|y_i-y_j\|^2)^{-1}$ is the Student-t kernel: like the Gaussian it is 1 at zero distance and decays with distance, but it decays **much more slowly** — it has a *heavy tail*. The denominator normalises over all pairs, as before.

Why the heavy tail cures crowding: with a heavy-tailed $q$, a pair that sits at a **moderate map distance still retains a decent $q_{ij}$**. So to represent a given true similarity $p_{ij}$, the map is allowed to place the points **farther apart** than a Gaussian would demand — there is no need to jam them all into the centre. The extra room at moderate distances is exactly what the 2D plane was missing. As a bonus, the heavy tail actively *pushes* dissimilar points well apart, which is why t-SNE produces those characteristic clean gaps between clusters.

![Two similarity-versus-distance curves. The blue Gaussian curve (used for the high-dimensional p) decays quickly and is essentially zero beyond distance 2. The orange Student-t / Cauchy curve (used for the 2D-map q) starts the same near zero distance but has a much heavier tail, retaining noticeable weight at moderate and large distances. A shaded band at moderate distances highlights that the Student-t keeps weight there, so points are not over-attracted and need not collapse into the centre.](../assets/tsne_gaussian_vs_studentt.jpg)

So: **Gaussian for $p$ in the crowded high-dim space, heavy-tailed Student-t for $q$ in the roomy-starved 2D map.** That mismatch is not a bug — it is the whole trick, and it is what "t-distributed SNE" refers to.

---

## The gradient as attraction and repulsion

To minimise the KL cost we use gradient descent on the map points $y_i$. The gradient has a famously clean form:

$$
\frac{\partial \mathcal{C}}{\partial y_i} = 4 \sum_{j \ne i} (p_{ij} - q_{ij})\,\big(1 + \|y_i - y_j\|^2\big)^{-1}\,(y_i - y_j).
$$

We will not derive it (a page of algebra; see the paper), but its *meaning* is the intuition worth keeping. Read each factor:

- $(y_i - y_j)$ is the **direction** from $j$ to $i$ — the axis along which the pair can move apart or together.
- $(p_{ij} - q_{ij})$ is the **sign and strength**: if the true neighbourliness $p_{ij}$ exceeds what the map currently shows ($q_{ij}$), the pair should be **closer** → the force pulls $y_i$ toward $y_j$ (**attraction**). If the map over-connects them ($q_{ij} > p_{ij}$), the force pushes them apart (**repulsion**).
- $(1+\|y_i-y_j\|^2)^{-1}$ is the same heavy-tailed factor, which keeps the forces well-behaved at long range.

This is exactly a system of **springs**. Between every pair of map points sits a spring whose rest length encodes the true similarity: true neighbours are pulled together, mis-placed strangers are shoved apart. The layout you finally see is the **equilibrium** where all the pulls and pushes balance.

![A schematic of one central map point y_i with four surrounding points. Two green points labelled as true neighbours (high p) have arrows pointing inward toward y_i, indicating attraction / pull-in. Two red points that are close in the map but not true neighbours (low p) have arrows pointing outward away from y_i, indicating repulsion / push-out. The caption notes the final layout settles where pulls balance pushes.](../assets/tsne_springs.jpg)

Because this energy landscape is **non-convex** (many local minima) and starts from a random initialisation, there is no single "correct" answer — different random seeds give different-looking maps. That is normal for t-SNE and motivates the tricks below.

---

## Optimisation tricks that actually matter

Naive gradient descent on the KL cost tends to get stuck with clusters overlapping. Two practical tricks, both in the original paper, make t-SNE work in practice:

- **Early exaggeration.** For the first ~250 iterations, multiply all the $p_{ij}$ by a constant (typically 4–12). This *inflates* the attractive forces between true neighbours, so genuine clusters snap together tightly and — because they clump — open up wide empty gaps between different clusters. It gives the optimisation room to move clusters past each other before settling. After the exaggeration phase, $p_{ij}$ is restored to normal and the layout relaxes into its final positions.
- **Momentum and learning rate.** t-SNE uses gradient descent *with momentum* (carry inertia across steps to speed through flat regions and damp oscillation — see [optimisers.md](../optimisers.md)). The learning rate matters: too small and clusters stay a diffuse ball; a common modern heuristic is to set it to about $N/12$ for large datasets.

A third, less flashy but important choice: **initialisation.** Initialising the map with **PCA** (rather than pure random noise) gives more reproducible layouts and preserves a little global structure, and is now a common default.

None of these change the objective — they just help gradient descent find a good minimum of a non-convex cost.

---

## Making it scale: Barnes-Hut

The plain algorithm computes a force between **every pair** of points, costing $O(N^2)$ per iteration — fine for a few thousand points, painful for hundreds of thousands. The standard speed-up is **Barnes–Hut t-SNE**, borrowed from N-body physics simulation. The idea: a far-away *group* of map points exerts almost the same repulsion as if it were a single point at the group's centre of mass. So you build a quad-tree over the 2D map, and for any point you treat distant tree cells as one lumped summary instead of visiting their members individually. This drops the cost to about $O(N \log N)$ and is what makes t-SNE on large datasets tractable. (For very large $N$, **FIt-SNE**, which uses a fast Fourier-based interpolation for the repulsive forces, pushes this even further.)

---

## How to read a t-SNE plot (pitfalls)

This section is as important as the algorithm itself. t-SNE plots are notoriously easy to over-interpret, and the excellent Distill article *"How to Use t-SNE Effectively"* (Wattenberg et al.) is the canonical warning. The core message: **t-SNE preserves local neighbourhoods; almost everything else on the plot is unreliable.** Concretely:

- **Cluster *sizes* are meaningless.** Because each point's bandwidth $\sigma_i$ is rescaled to a fixed perplexity, t-SNE *expands* dense clusters and *contracts* sparse ones. A tight, compact cluster in reality can look the same size as a diffuse one. Do not read "this blob is bigger, so it has more variance."
- **Distances *between* clusters are (mostly) meaningless.** Two clusters sitting side by side in a t-SNE plot are not necessarily more similar than two clusters far apart. The heavy-tailed repulsion and the local focus mean global inter-cluster distances are not faithfully preserved. (Higher perplexity recovers *some* global structure, but never trust it fully.)

![Two panels illustrating the pitfalls. Left, the ground truth (first two of five dimensions): cluster A is a tiny tight blob, cluster B is five times wider, and cluster C sits very far away from both. Right, the t-SNE map of the same data: all three clusters come out as roughly equal-sized blobs spaced at similar intervals, so the true size differences and C's much larger true distance are both lost.](../assets/tsne_pitfalls.jpg)

- **Perplexity changes the picture**, so run several. Structure that only appears at one perplexity is suspect; structure stable across a range is real.
- **Random noise can look like clusters.** At low perplexity, t-SNE can carve pure noise into tidy-looking clumps. Always sanity-check against a known-random baseline before believing small clusters.
- **You may need several runs / enough iterations.** Because the cost is non-convex and randomly initialised, different seeds differ, and a plot that has not converged (too few steps) can show fake pinched or split shapes. Let it run to convergence and compare runs.

The safe way to use t-SNE: treat it as a tool for the question *"do natural groups exist, and which points cluster together?"* — and **not** for *"how big is this group?"* or *"how far apart are these two groups?"*

---

## t-SNE vs UMAP

The main modern alternative is **UMAP** (Uniform Manifold Approximation and Projection). It follows the same broad recipe — build a neighbour graph in high-dim, lay it out in 2D by balancing attraction and repulsion — but with a different mathematical footing (fuzzy simplicial sets / cross-entropy rather than KL of t-distributions). In practice UMAP is usually **faster**, tends to **preserve more of the global structure** (relative cluster placement is a bit more trustworthy), and scales better to very large datasets. t-SNE is often considered to give slightly cleaner *local* separation and remains extremely popular. The pitfalls above — don't over-read cluster size or spacing — apply to UMAP too. *(This comparison is deliberately brief; expand on request.)*

---

## Practical guidance and takeaways

- **Preprocess first.** Standardise your features, and if the data is very high-dimensional, run PCA down to ~50 dimensions *before* t-SNE. This denoises, and hugely speeds up the neighbour computations.
- **Start with perplexity 30**, then sweep a few values (e.g. 5, 30, 50) and trust only stable structure.
- **Initialise with PCA** and let the optimiser run to convergence (enough iterations); use early exaggeration (it is on by default in libraries like scikit-learn).
- **Read only what t-SNE is faithful to:** *which points are neighbours* and *whether distinct clusters exist.* Ignore blob sizes and between-cluster distances.
- **Remember what it is not:** t-SNE is a **visualisation** tool, not a general-purpose feature transform. The 2D coordinates are not meaningful features to feed downstream — they are a picture for human eyes.

The one-line summary: t-SNE converts distances into neighbour probabilities in both the original space (Gaussian, $p$) and the 2D map (heavy-tailed Student-t, $q$), then moves the map points until those two neighbour pictures agree — faithfully preserving *who is near whom* at the deliberate cost of global geometry.

---

## Sources

- **van der Maaten & Hinton (2008), "Visualizing Data using t-SNE"**, *JMLR* — the original paper; introduces the symmetric SNE, the crowding problem, and the Student-t fix. The must-read.
- **Hinton & Roweis (2002), "Stochastic Neighbor Embedding"**, *NeurIPS* — the predecessor SNE that t-SNE improves upon (Gaussian $q$, asymmetric cost).
- **van der Maaten (2014), "Accelerating t-SNE using Tree-Based Algorithms"**, *JMLR* — Barnes–Hut t-SNE, the $O(N\log N)$ speed-up.
- **Wattenberg, Viégas & Johnson (2016), "How to Use t-SNE Effectively"**, *Distill* — the essential guide to reading (and mis-reading) t-SNE plots. https://distill.pub/2016/misread-tsne/
- **McInnes, Healy & Melville (2018), "UMAP"** — the leading modern alternative.
- **scikit-learn documentation**, `sklearn.manifold.TSNE` — practical defaults and API.
