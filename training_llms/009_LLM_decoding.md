# LLM Decoding: from a Probability Distribution to Actual Words

A trained language model does **not** output text. At each step it outputs a
vector of scores — one number per token in the vocabulary — and turns them into
a probability distribution over "what comes next." **Decoding** (also called
*generation* or *sampling*) is the separate, inference-time question of how you
turn that distribution into an actual token, step after step, until a sentence
comes out.

This matters more than beginners expect. The *same* frozen model can sound like
a careful encyclopedia, a repetitive broken record, or a wild hallucinating
poet — purely depending on the decoding knobs you set. None of this touches the
weights; it is all downstream of the softmax. This note walks through the
family of decoding methods you have heard named — greedy, temperature,
sampling, top-k, top-p (nucleus), min-p, beam search, and repetition penalties
— building the intuition for *why* each one exists and what problem it fixes in
the one before it.

The model itself (how those scores are produced) is covered in the companion
notes on [attention](005_Attention.md) and the [tokenizer](001_tokeniser.md)
that defines the vocabulary; here we take the per-step distribution as given
and ask only what to *do* with it.

---

## Table of Contents

- [Setup and Notation](#setup-and-notation)
- [1. The one tension that explains every method](#1-the-one-tension-that-explains-every-method)
- [2. Greedy decoding](#2-greedy-decoding)
- [3. Beam search](#3-beam-search)
  - [3.1 The idea: keep several hypotheses alive](#31-the-idea-keep-several-hypotheses-alive)
  - [3.2 n-gram blocking and length penalty](#32-n-gram-blocking-and-length-penalty)
  - [3.3 Why beam text still feels dull](#33-why-beam-text-still-feels-dull)
- [4. Sampling and temperature](#4-sampling-and-temperature)
  - [4.1 Plain sampling](#41-plain-sampling)
  - [4.2 Temperature](#42-temperature)
- [5. Top-k sampling](#5-top-k-sampling)
- [6. Top-p (nucleus) sampling](#6-top-p-nucleus-sampling)
- [7. Min-p sampling](#7-min-p-sampling)
- [8. Repetition, frequency, and presence penalties](#8-repetition-frequency-and-presence-penalties)
  - [8.1 The order everything is applied in](#81-the-order-everything-is-applied-in)
- [9. A practical cheat-sheet](#9-a-practical-cheat-sheet)
- [10. A note on speculative decoding (different axis)](#10-a-note-on-speculative-decoding-different-axis)
- [Recap](#recap)
- [Sources](#sources)

---

## Setup and Notation

These symbols recur throughout; skim once and refer back.

| Symbol | Meaning |
|---|---|
| $V$ | the **vocabulary** — the set of all tokens the model can emit (see [tokenizer note](001_tokeniser.md)). |
| $w_{1:t-1}$ | the **context** so far — the tokens already generated (plus the prompt). |
| $z_i$ | the **logit** for token $i$ — the raw, unnormalized score the model outputs at the current step. |
| $p_i$ | the **probability** of token $i$, obtained by softmax over the logits: $p_i = \dfrac{e^{z_i}}{\sum_j e^{z_j}}$. |
| $T$ | the **temperature** — a scalar that sharpens ($T<1$) or flattens ($T>1$) the distribution. |
| $k$ | **top-k** — keep only the $k$ highest-probability tokens. |
| $p$ (as a threshold) | **top-p / nucleus** — keep the smallest set of tokens whose probabilities sum to at least $p$. |
| $p_{\max}$ | the probability of the single most likely token at this step (used by min-p). |

Two facts to keep in mind. First, generation is **autoregressive**: you pick a
token, append it to the context, and run the model again — so a bad choice
early poisons everything after it. Second, everything below operates on the
distribution $p$ at a single step; the methods differ only in *how they pick
one token from it*.

---

## 1. The one tension that explains every method

Every decoding method is a different answer to a single trade-off, so it is
worth stating up front — after this, each method is just a point on a slider.

- If you **always take the most probable tokens**, you get text that is
  grammatical but **boring and repetitive**. Worse, it tends to fall into
  loops ("I love this. I love this. I love this…"). It turns out that
  high-quality human writing does *not* consist of the single most likely word
  at every step — humans constantly pick mildly surprising words. A model that
  greedily maximizes probability produces text that is, paradoxically, *too*
  predictable to sound human.

- If you **sample freely from the full distribution**, you inject the surprise
  back in — but the tail of the distribution is enormous. A vocabulary has tens
  of thousands of tokens, and even if each junk token has probability
  $0.0001$, collectively they carry real mass. Sample long enough and you will
  eventually draw one, derail the sentence, and — because generation is
  autoregressive — never recover. The result is **incoherent gibberish**.

So the whole game is: **inject enough randomness to sound natural, but truncate
the garbage tail so you never sample nonsense.** Greedy and beam search sit at
the "maximize probability" extreme; temperature, top-k, top-p, and min-p are
increasingly clever ways to keep the good randomness and cut the bad tail.

---

## 2. Greedy decoding

The simplest possible rule: at each step, take the single highest-probability
token and move on.

$$
w_t = \arg\max_{w \in V} \; P(w \mid w_{1:t-1})
$$

Here $\arg\max$ just means "the token with the largest probability." There is no
randomness — feed the same prompt twice and you get the exact same output. It
is fast (one forward pass, one lookup) and it is what you want when there is a
single correct answer (a factual question, a code completion, a translation).

**Its failure mode is short-sightedness.** Greedy commits to the best token
*right now* without any regard for what it does to *future* steps. Picture the
model choosing the second word after "The":

- "nice" has probability $0.5$, "dog" has $0.4$. Greedy takes "nice."
- But suppose after "dog" the model is very confident the next word is "has"
  ($0.9$), while after "nice" the best continuation ("woman") is only $0.4$.

Greedy's path "The nice woman" has joint probability $0.5 \times 0.4 = 0.2$,
while "The dog has" scores $0.4 \times 0.9 = 0.36$ — a *more* probable sequence
that greedy never finds, because the high-probability word "has" was hidden
behind a token ("dog") that merely looked second-best one step earlier. Greedy
literally cannot see one word ahead.

Combined with the repetition problem from §1, plain greedy is rarely used for
open-ended generation — but it is the baseline every other method is measured
against.

---

## 3. Beam search

### 3.1 The idea: keep several hypotheses alive

Beam search is the direct fix for greedy's short-sightedness. Instead of
committing to one token per step, it keeps the `num_beams` best *partial
sequences* (hypotheses) alive at all times. At each step it expands every
surviving beam by every possible next token, scores all the resulting longer
sequences by their total probability, and keeps only the top `num_beams` of
them. At the end it returns the highest-scoring complete sequence.

Run the "The …" example with `num_beams = 2`. Greedy locked onto "nice" and was
stuck. Beam search keeps *both* "The nice" and "The dog" alive, expands both,
and discovers "The dog has" at $0.36$ — beating greedy's "The nice woman" at
$0.2$. By carrying a few alternatives forward, it uncovers the high-probability
word hidden one step deep.

A useful guarantee: **beam search always finds a sequence at least as probable
as greedy** (greedy is just beam search with `num_beams = 1`). It is *not*
guaranteed to find the globally most probable sequence — that would require
exploring the entire exponential tree — but with a modest beam width it gets
much closer. The cost is proportional: `num_beams` times the compute and
memory of greedy.

### 3.2 n-gram blocking and length penalty

Two practical patches make beam search usable:

- **n-gram blocking (`no_repeat_ngram_size`).** Because beam search still
  chases high probability, it *also* loves to repeat itself. The fix is blunt:
  forbid any n-gram from appearing twice by forcing the probability of any
  token that would complete a repeated n-gram to zero. With
  `no_repeat_ngram_size = 2`, no pair of consecutive tokens can ever recur. The
  hazard is obvious once stated — a named entity like "New York" legitimately
  needs to repeat across a paragraph, and hard blocking will mangle it. Use it
  with care.

- **Length penalty.** The score of a sequence is a product of probabilities
  (or a sum of log-probabilities, all of which are negative). Longer sequences
  therefore accumulate more negative terms and look "worse," so naive beam
  search is biased toward *short* outputs. A length penalty divides the score
  by (roughly) the sequence length raised to some power $\alpha$, canceling
  this bias so beams of different lengths compete fairly.

### 3.3 Why beam text still feels dull

Here is the twist that motivates the entire second half of this note. Even
done well, beam search produces text that human readers find flat and
lifeless. The reason connects back to §1: **real human text is not
high-probability text.** If you plot the per-token probability a model assigns
to *actual human writing*, it swings all over the place — humans routinely
choose words the model considered unlikely. Beam search, by construction,
hugs the high-probability ridge and never takes those human-like risks, so its
output reads as bland and slightly robotic, even when every word is "correct."

The takeaway: for tasks with one right answer (translation, summarization,
constrained code), maximizing probability via beam search is great. For
**open-ended** generation (stories, chat, brainstorming), we need to *stop*
maximizing probability and start sampling — which is the rest of this note.

---

## 4. Sampling and temperature

### 4.1 Plain sampling

The opposite of greedy: instead of taking the top token, **draw one at random
according to its probability**.

$$
w_t \sim P(w \mid w_{1:t-1})
$$

A token with probability $0.5$ is picked about half the time; one with $0.01$
is picked about one time in a hundred. This immediately restores the human-like
surprise greedy and beam search lacked — but, as warned in §1, unrestricted
sampling from the full vocabulary eventually draws a junk token from the long
tail and derails. Every method from here on is plain sampling **plus a way to
tame the distribution first**. The first such knob is temperature.

### 4.2 Temperature

Temperature rescales the logits *before* the softmax, sharpening or flattening
the distribution you sample from:

$$
p_i = \frac{e^{z_i / T}}{\sum_j e^{z_j / T}}
$$

The only change from the plain softmax is dividing each logit by $T$. That
single division has an intuitive effect:

- **$T < 1$ (e.g. 0.5)** divides the logits by a small number, *magnifying* the
  gaps between them. Big logits pull even further ahead, so the distribution
  gets **sharper** — more mass on the top token, more conservative and
  greedy-like sampling. In the limit $T \to 0$, all the mass collapses onto the
  single largest logit and sampling becomes exactly **greedy decoding**.

- **$T > 1$ (e.g. 2.0)** shrinks the gaps, **flattening** the distribution so
  unlikely tokens get a fairer shot — more diverse, more creative, more random.
  As $T$ grows large the distribution approaches uniform (pure chaos).

The figure below shows the exact *same* logits reshaped by three temperatures.
Notice that temperature never changes the *ranking* of tokens — "cat" stays the
favorite everywhere — it only changes how much the favorites dominate.

![Three bar charts of a probability distribution over eight candidate tokens (cat, dog, car, the, sky, run, blue, of), computed from one identical set of logits at temperature 0.5, 1.0, and 2.0. At T=0.5 the top token "cat" holds 0.74 of the mass and the rest are nearly flat on the floor — a sharp, greedy-like distribution. At T=1.0 (the raw model) "cat" is 0.50 and "dog" 0.27, with a gentle tail. At T=2.0 the distribution is visibly flatter — "cat" drops to 0.31 and the lower-ranked tokens all rise, giving unlikely words a real chance of being sampled. The ranking of tokens is identical across all three panels; only the sharpness changes.](../assets/decode_temperature.jpg)

The trade-off is a single dial. Set $T$ **too low** and you recover greedy's
dullness and repetition; set it **too high** and you slide back into gibberish.
In practice people use low temperatures ($\approx 0.0$–$0.3$) for analytical or
factual tasks where you want the safe answer, and temperatures near $0.7$–$1.0$
for creative writing where you want variety. Crucially, temperature *reshapes*
the whole distribution but does **not** truncate the tail — a rare token still
has nonzero probability. That is exactly the job of the next two methods.

---

## 5. Top-k sampling

Top-k is the first true **truncation** method. The recipe is dead simple: keep
only the $k$ most probable tokens, throw the rest away, renormalize the survivors
so they sum to 1, and sample from those.

By deleting everything below rank $k$, you make it *impossible* to sample the
junk tail — no matter how unlucky your random draw, you can only land on one of
the $k$ candidates the model actually liked. With $k = 6$, weird low-rank
options simply cannot be chosen. Setting $k = 1$ recovers greedy again (only the
top token survives).

**Its weakness is that $k$ is a fixed number, and the right number changes from
step to step.** Consider two situations:

- After "The sky is …" the model is *confident*: maybe "blue," "clear," and
  "grey" carry almost all the mass. Here a large $k$ forces you to also keep
  several genuinely bad tokens just to fill the quota of $k$.
- After "She opened the door and saw a …" the model is *uncertain*: dozens of
  nouns are individually plausible. Here a small $k$ chops off perfectly good
  continuations and makes the text needlessly safe.

A single fixed $k$ cannot be right for both a sharp distribution and a flat one.
That mismatch is precisely what top-p fixes.

---

## 6. Top-p (nucleus) sampling

Top-p — also called **nucleus sampling** — keeps the truncation idea but makes
the cutoff **adaptive**. Instead of "keep a fixed number of tokens," the rule is
"keep the smallest set of top tokens whose probabilities add up to at least
$p$" (a common choice is $p = 0.9$ or $0.92$). You then renormalize that set —
the *nucleus* — and sample from it.

Because the cutoff is defined by *cumulative probability* rather than *count*,
the size of the surviving set breathes with the model's confidence:

- When one or two tokens dominate (a sharp step), just those few already reach
  $p$, so the nucleus is tiny and you stay safe.
- When the model is unsure (a flat step), you have to gather many tokens before
  the mass reaches $p$, so the nucleus grows and you keep the diversity.

The figure contrasts the two methods on exactly this. On a **peaked**
distribution, top-k rigidly keeps 4 tokens while top-p keeps only 3 (it stopped
as soon as it had 90% of the mass). On a **flat** distribution, top-k *still*
keeps 4 — throwing away good candidates — while top-p expands to 8 to gather the
same 90%. Same $k$, same $p$, wildly different behavior: top-p adapts, top-k
does not.

![A 2x2 grid of bar charts comparing top-k and top-p truncation. The left column is top-k (k=4) and the right column is top-p (p=0.9); kept tokens are colored green and discarded tokens are greyed out. Top row is a peaked distribution where the model is confident: top-k keeps a fixed 4 tokens, while top-p keeps only 3 because the top three already reach 90% of the probability mass. Bottom row is a flat distribution where the model is unsure: top-k again keeps exactly 4 tokens, discarding several nearly-equal candidates, while top-p expands to keep 8 tokens to accumulate the same 90% mass. The figure shows top-k using a fixed-size pool regardless of shape, whereas top-p grows or shrinks its pool to match the distribution.](../assets/decode_topk_vs_topp.jpg)

In theory nucleus sampling is the more elegant idea, and it is the most common
default in practice today. Empirically top-k and top-p perform comparably well,
and the two are often **combined**: apply top-k first as a hard safety cap on
the absolute number of candidates, then apply top-p within that. Temperature is
usually applied alongside — flatten or sharpen with $T$, then truncate the tail
with top-p.

---

## 7. Min-p sampling

Min-p is a newer truncation rule that targets a specific failure of top-p:
**at high temperature, top-p can still let junk in.** Recall that top-p keeps
adding tokens until the cumulative mass hits $p$. If you crank temperature up,
the distribution flattens, so it takes *many* tokens to reach $p$ — and some of
those extra tokens are genuinely bad. Top-p's fixed mass budget does not know
the difference.

Min-p sets the cutoff *relative to the top token's confidence* instead of by
cumulative mass. It keeps every token whose probability is at least a fraction
of the best token's probability:

$$
V_{\text{min-p}} = \{\, i \;:\; p_i \ge \tau \,\}, \qquad \tau = (\text{min-p}) \cdot p_{\max}
$$

where $p_{\max}$ is the probability of the single most likely token this step
and min-p is a small scalar (e.g. $0.05$–$0.1$). The intuition is a *dynamic
floor*: if the model is very confident ($p_{\max}$ large), the threshold $\tau$
rises and the pool tightens to only near-peer tokens; if the model is unsure
($p_{\max}$ small), $\tau$ drops and more tokens qualify. This lets you push
temperature high for creativity while the min-p floor still snips off the
incoherent tail — the "turn up the heat without losing the plot" trade-off top-p
struggles with. Treat this as a briefer entry than top-k/top-p; it can be
expanded on request.

---

## 8. Repetition, frequency, and presence penalties

All the methods so far act on the probability distribution *shape*. Penalties
are orthogonal: they look at **what has already been generated** and push those
tokens down, directly attacking the "broken record" loops that plague greedy and
low-temperature sampling.

Two flavors are common.

- **Repetition penalty (CTRL-style).** Introduced with the CTRL model, this
  divides the logit of any already-seen token by a factor $\theta > 1$ (a
  typical value is $\theta \approx 1.2$) before the softmax. Dividing a logit
  by $\theta$ shrinks its probability, making tokens you have already used less
  likely to be used again. Set $\theta = 1$ and nothing changes; larger $\theta$
  penalizes harder. (A subtlety: for *negative* logits, dividing by $\theta$
  makes them larger, so implementations multiply-vs-divide by sign — the intent
  is always "make seen tokens less likely.")

- **Frequency and presence penalties (OpenAI-style).** These *subtract* from the
  logit instead of dividing. Conceptually, for each candidate token:

$$
z_i \;\leftarrow\; z_i \;-\; \underbrace{\alpha_{\text{freq}} \cdot c_i}_{\text{frequency penalty}} \;-\; \underbrace{\alpha_{\text{pres}} \cdot \mathbb{1}[c_i > 0]}_{\text{presence penalty}}
$$

  where $c_i$ is the number of times token $i$ has already appeared. The
  **frequency penalty** scales with the *count* $c_i$, so a word used five times
  is punished five times as hard — it fights repetitive loops. The **presence
  penalty** is a flat one-time hit the moment a token appears *at all*
  ($\mathbb{1}[c_i>0]$ is 1 if the token has occurred, else 0), which nudges the
  model to introduce *new* topics rather than dwell on any word it has already
  touched. Both default to 0 (no penalty).

The difference in one line: repetition penalty *scales* logits multiplicatively,
frequency/presence penalties *shift* them additively; frequency cares how
*often* a token appeared, presence only cares *whether* it appeared.

### 8.1 The order everything is applied in

When you set several knobs at once, they are applied in a fixed pipeline, and
the order matters:

1. **Penalties** (repetition / frequency / presence) adjust the raw logits
   first, based on history.
2. **Temperature** rescales those adjusted logits.
3. **Softmax** turns logits into probabilities.
4. **Truncation** (top-k, then top-p, or min-p) cuts the tail.
5. **Sample** one token from the surviving, renormalized distribution.

Penalties come first so that a token you want to suppress is pushed down
*before* the tail-cutting decides what to keep — otherwise you might truncate to
a pool that still contains the very token you were trying to discourage.

---

## 9. A practical cheat-sheet

A rough map of when to reach for what. Treat the parameter values as starting
points, not law — the source blogs all stress there is no universal best
setting, only what works for your use case.

| Method | Randomness | Best for | Main risk |
|---|---|---|---|
| **Greedy** | none | one-right-answer tasks; debugging | repetition, short-sighted |
| **Beam search** | none | translation, summarization, constrained output | bland, biased short, slow |
| **Sampling + temperature** | tunable | general open-ended generation | high $T$ → gibberish |
| **Top-k** | tunable | simple tail-cutting | fixed $k$ ill-fits sharp vs flat steps |
| **Top-p (nucleus)** | tunable | the common default for chat/creative | can leak junk at very high $T$ |
| **Min-p** | tunable | creative writing at high temperature | newer, less battle-tested |
| **+ Repetition / freq / presence penalties** | — | any method stuck in loops | over-penalizing mangles legitimate repeats |

Sensible defaults to start from: for **factual / code** use greedy or a low
temperature ($T \approx 0.2$); for **chat / creative** use temperature
$\approx 0.7$–$1.0$ with **top-p $\approx 0.9$**, adding a mild repetition or
frequency penalty only if you observe loops. For **translation /
summarization**, beam search with 4–5 beams, a length penalty, and modest
n-gram blocking is a strong baseline.

---

## 10. A note on speculative decoding (different axis)

You may hear "speculative decoding" mentioned alongside these methods — it is
worth one paragraph to place it, because it lives on a *different axis*.
Everything above is about **which token to pick** (a *quality/diversity*
choice). Speculative decoding is about **how fast you can produce tokens** (a
*speed* trick) and does not change the output distribution at all. The idea: a
small, cheap "draft" model rapidly guesses several tokens ahead, and the big
"target" model verifies them all in a *single* forward pass via a rejection-
sampling scheme; accepted guesses are kept, the first rejected one is
resampled. Because the verification is provably equivalent to sampling from the
big model directly, you get the **exact same quality** as normal decoding, just
2–3× faster when the GPU would otherwise be underused. So it is not a rival to
top-p or temperature — you still choose one of those; speculative decoding just
makes running it cheaper. It deserves its own note; flagged here only so the
name is not confusing.

---

## Recap

* **Decoding is separate from the model.** The same weights produce very
  different text depending on how you turn each step's distribution into a
  token.
* **One tension drives everything:** maximizing probability is dull and
  repetitive; free sampling is incoherent. Every method balances the two.
* **Greedy** takes the top token — fast, deterministic, short-sighted.
  **Beam search** keeps several hypotheses to find higher-probability
  sequences, but its very probability-hugging makes it read bland.
* **Sampling** restores human-like surprise; **temperature** sharpens or
  flattens the distribution first ($T\to0$ = greedy, $T>1$ = wilder).
* **Top-k** cuts the tail to a fixed number of tokens; **top-p (nucleus)** cuts
  it to an *adaptive* set that breathes with the model's confidence; **min-p**
  floors the cutoff relative to the top token so you can run hot without
  gibberish.
* **Penalties** (repetition, frequency, presence) act on generation history to
  break loops, and are applied *before* temperature and truncation.
* **Speculative decoding** is orthogonal — a speed optimization that leaves the
  output distribution unchanged.

---

### Sources

* Patrick von Platen, *How to generate text: using different decoding methods
  for language generation with Transformers* — Hugging Face blog.
  <https://huggingface.co/blog/how-to-generate>
* Sebastian Raschka, *How do temperature, top-k, and top-p sampling differ?*
  <https://sebastianraschka.com/faq/docs/temperature-topk-topp-sampling.html>
* Aman Chadha, *Token Sampling Methods* — Aman's AI Journal.
  <https://aman.ai/primers/ai/token-sampling/>
* Nguyen et al., *Turning Up the Heat: Min-p Sampling for Creative and Coherent
  LLM Outputs*, arXiv:2407.01082. <https://arxiv.org/abs/2407.01082>
* Keskar et al., *CTRL: A Conditional Transformer Language Model for
  Controllable Generation*, 2019 (repetition penalty).
* Leviathan et al., *Fast Inference from Transformers via Speculative Decoding*,
  arXiv:2211.17192.
