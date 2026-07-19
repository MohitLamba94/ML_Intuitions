# The Tokenizer: Turning Text into Integers an LLM Can Read

A transformer does not read text. It reads a sequence of **integers**, each of which indexes a row of a big lookup table (the embedding matrix) to fetch a vector. So before any language model can train or generate, something has to convert a raw string like `"tokenizing text!"` into a list of integers such as `[30001, 4291, 0]`, and convert integers back into text when the model produces them. That something is the **tokenizer**: a reversible bridge `text ↔ sequence of integer IDs` that sits *before* the model on the way in and *after* it on the way out.

It sounds like plumbing, but the tokenizer quietly decides some of the most important properties of the whole system: how long your sequences are (and therefore how expensive attention is), how large your vocabulary and embedding table are, whether the model can even represent a typo or an emoji or a Chinese character, and how fairly it treats different languages. This note builds the idea up from scratch: why we tokenize at all, the spectrum of granularities (character / byte / word / subword) and their trade-offs, **Byte Pair Encoding (BPE)** in full detail, the compression-ratio vs. vocabulary-size tension (and what "sparsity" means there), and finally whether a tokenizer even needs to be *trained*.

---

## Table of Contents

- [Setup and Notation](#setup-and-notation)
- [Why tokenize at all? Can't we just feed raw text?](#why-tokenize-at-all-cant-we-just-feed-raw-text)
- [The granularity spectrum: character, byte, word, subword](#the-granularity-spectrum-character-byte-word-subword)
- [Byte Pair Encoding (BPE) in detail](#byte-pair-encoding-bpe-in-detail)
- [Compression ratio, vocabulary size, and "sparsity"](#compression-ratio-vocabulary-size-and-sparsity)
- [Do tokenizers need to be trained?](#do-tokenizers-need-to-be-trained)
- [Takeaways](#takeaways)
- [Sources](#sources)

---

## Setup and Notation

A few terms recur throughout; each is explained again where it first does real work.

| Symbol / term | Meaning |
|---|---|
| **corpus** | The large body of raw text used to *build* the tokenizer (and, separately, to train the model). |
| **character** | A single Unicode code point, e.g. `t`, `!`, `世`, `😀`. |
| **byte** | One of 256 possible values (0–255). Text is stored on disk as bytes via an encoding like **UTF-8**; one character is 1–4 bytes. |
| **token** | The atomic unit the tokenizer emits — could be a character, a byte, a subword piece, or a whole word, depending on the scheme. |
| **vocabulary $V$** | The finite set of all distinct tokens the tokenizer knows. $\|V\|$ is the **vocabulary size** (a.k.a. dictionary size). |
| **token ID** | The integer index of a token in $V$ — this is what the model actually consumes. |
| **sequence length $n$** | How many tokens a given piece of text becomes. |
| **compression ratio** | $\dfrac{\text{num\_bytes}}{\text{num\_tokens}}$ — average number of raw UTF-8 bytes packed into one token. |

---

## Why tokenize at all? Can't we just feed raw text?

Here is what "tokenized text" actually looks like — the same passage, with each colored span being one token that the model sees as a single integer:

![A paragraph of English text where every word-piece, space, and punctuation mark is highlighted in an alternating color; each highlighted span is one token, and the same text is also shown as the sequence of integer token IDs it maps to. Illustrates that a language model consumes text as a sequence of integer IDs, not raw characters.](../assets/tokenized_example.jpg)

*Image from Stanford CS336, "Language Modeling from Scratch," Lecture 1 (Percy Liang, Tatsunori Hashimoto et al.). Reproduced for educational purposes.*

So why do we need this step at all — why not train and infer *directly* on raw text?

The blunt answer is that a neural network has no notion of "text." Its first layer is an **embedding lookup**: given an integer $i$, it returns row $i$ of a matrix. That requires the input to already be a sequence of integers drawn from a **fixed, finite** set. Raw text is neither — it is a variable-length stream of Unicode characters drawn from a set of ~150,000 code points (and growing). You *must* commit to some finite alphabet of units and a rule for chopping text into them. **That choice — the unit, plus the map from string to integer IDs — is exactly what "tokenization" is.** There is no "no-tokenizer" option; even feeding raw bytes is a tokenization choice (the unit is "one byte," the vocabulary is the 256 byte values).

Given that we must chop text into *some* fixed set of units, the real question is: **which units?** And the reason this matters so much is **sequence length**. The dominant cost in a transformer is self-attention, whose compute and memory grow **quadratically** with the number of tokens $n$: doubling the sequence length roughly *quadruples* the attention cost. The granularity of your units directly sets $n$ for a given passage — tiny units (characters, bytes) make $n$ huge, coarse units (words) make $n$ small. So tokenization is not cosmetic preprocessing; it sets the price of everything downstream: training cost, inference latency, and how much text fits in a fixed context window. The rest of this note is really about navigating that trade-off well.

---

## The granularity spectrum: character, byte, word, subword

Take one string and split it four different ways. The same 16-character sentence becomes 3 tokens, 5 tokens, or 16 tokens depending on how coarse the unit is:

![The string 'tokenizing text!' shown split four ways on stacked rows. Word-level gives 3 tokens (tokenizing / space / text!); subword/BPE gives 5 tokens (token / izing / space / text / !); character-level gives 16 tokens, one per character; byte-level gives 16 tokens shown as integer byte values 116, 111, 107 and so on. Coarser units give fewer tokens but need a larger vocabulary; finer units give longer sequences but a tiny vocabulary.](../assets/tokenization_granularities.jpg)

The four classic choices, and how they trade off, are below. Keep two competing costs in mind: **vocabulary size $|V|$** (which sets the size of the embedding table and the final softmax, and thus the FLOPs of the input/output layers) and **sequence length $n$** (which sets the quadratic attention cost).

**Word-based.** One token per word (split on whitespace/punctuation). Sequences are *short* (high compression), which is great for attention. But the vocabulary is enormous — English alone has hundreds of thousands of word forms, and once you add other languages, names, numbers, and typos it explodes toward the millions. That fat vocabulary means a giant embedding matrix and a giant output softmax (both scale with $|V|$), which costs parameters and FLOPs. Worse, it is brittle: any word not seen during vocabulary construction is **out-of-vocabulary (OOV)** and collapses to a single `<UNK>` token — so `tokenizing`, `tokenized`, and `tokenizer` share no structure, and a novel word is simply unrepresentable. Word tokenization is essentially obsolete for modern LLMs for these reasons.

**Character-based.** One token per Unicode character. The vocabulary is small (hundreds to a few thousand for common scripts) and there is **no OOV** — any text is representable. But sequences become *very long* (one token per character), which is punishing given attention's $O(n^2)$ cost, and each token carries little meaning on its own, forcing the model to spend capacity re-learning how characters compose into words. You trade a cheap vocabulary for expensive sequences.

**Byte-based.** Go one level below characters: encode text as **UTF-8 bytes** and let each byte be a token. Now the vocabulary is *exactly 256*, fixed forever, and there is *never* any OOV — every possible string, in any language, including emoji and even binary, is just a byte sequence. This is the most robust and universal option. The price is the **longest sequences of all** (a single emoji is 4 bytes = 4 tokens; a Chinese character is ~3 bytes), which again slams into the quadratic attention wall. Byte-level is a fantastic *fallback* (nothing is ever unrepresentable) but wasteful as your only unit.

**Subword-based.** The pragmatic winner, and what essentially every modern LLM uses. Keep frequent whole words as single tokens (`the`, `text`) but break rare words into meaningful *sub*-word pieces (`tokenizing` → `token` + `izing`). The vocabulary is a moderate ~30k–130k, so the embedding/softmax stay affordable; sequences are far shorter than character/byte level; and there is effectively **no OOV**, because in the worst case any unknown chunk backs off to its raw bytes. Subword tokenization is the sweet spot precisely because it lets you *tune* the balance between $|V|$ (input/output-layer cost) and $n$ (attention cost) by choosing the target vocabulary size.

At a glance:

| Scheme | Vocab size $\|V\|$ | Sequence length $n$ | OOV / robustness | Input & output layer FLOPs ($\propto \|V\|$) | Attention FLOPs ($\propto n^2$) |
|---|---|---|---|---|---|
| **Word** | huge (10⁵–10⁶+) | shortest | brittle — `<UNK>` on unseen words | very high | lowest |
| **Character** | small (10²–10³) | long | none (fully representable) | low | high |
| **Byte** | fixed **256** | longest | none — universal | lowest | highest |
| **Subword (BPE)** | moderate (10⁴–10⁵) | short | none (backs off to bytes) | moderate | low |

The recurring pattern: **finer units → smaller vocabulary but longer sequences; coarser units → larger vocabulary but shorter sequences.** Subword tokenization exists to sit at the useful middle of that spectrum — and the standard way to *find* that middle from data is BPE.

---

## Byte Pair Encoding (BPE) in detail

BPE is not a clever mathematical model; it is a **greedy, data-driven compression heuristic**, and that plainness is exactly why it works so robustly. It was invented by Philip Gage in 1994 as a general data-compression trick, adapted for NLP subword tokenization by Sennrich et al. (2016), and then used at byte level by GPT-2 — the recipe modern models still largely follow. The one-line idea: **start from the smallest possible units, then repeatedly glue together whichever adjacent pair occurs most often, until you have as many tokens as you want.** Frequent sequences (common words, common suffixes) naturally get glued into single tokens; rare sequences stay in small pieces.

### Training: building the vocabulary

We *learn* the vocabulary from a corpus. Here is the loop, illustrated on a toy corpus of four words with their counts (`·` marks a word end so merges don't leak across word boundaries):

![BPE training walk-through on the toy corpus low (count 5), lower (2), newest (6), widest (3). Start: every word is split into individual characters. Merge 1: the pair (e,s) is most frequent, so it becomes the new token 'es'. Merge 2: (es,t) merges into 'est'. Merge 3: (l,o) merges into 'lo'. Merge 4: (lo,w) merges into 'low'. The bottom shows the resulting ordered merge list — e+s to es, es+t to est, l+o to lo, lo+w to low — and notes that this ordered list IS the tokenizer.](../assets/bpe_merge_steps.jpg)

**Step 0 — start from bytes (or characters).** Initialize the vocabulary with the 256 raw byte values. Because every possible string is made of bytes, this guarantees the tokenizer can represent *anything* — there is no OOV, ever. Split the corpus so each word is a sequence of these base tokens.

**Step 1 — count adjacent pairs.** Scan the corpus and count how often each *adjacent pair* of current tokens appears (weighted by word frequency). In the toy corpus, `newest·6` and `widest·3` both contain the pair `(e, s)`, giving it count $6+3 = 9$ — the most frequent pair.

**Step 2 — merge the best pair.** Create one new token by concatenating that most-frequent pair (`e`+`s` → `es`), give it the next free ID, and **record the merge rule in an ordered list**. Then rewrite the whole corpus, replacing every occurrence of that adjacent pair with the new token. Now `newest·` is `n e w es t ·`.

**Step 3 — repeat.** Recount pairs on the updated corpus and merge again. Now `(es, t)` is most frequent → `est`. Next `(l, o)` → `lo`, then `(lo, w)` → `low`. Notice the tokens grow: early merges produce 2-character pieces, later merges glue pieces into longer subwords and eventually whole words like `low`. Keep going until you hit your **target vocabulary size** (e.g. 50,000) or no pair repeats.

The crucial output is **not** just the set of tokens — it is the **ordered list of merge rules**: `e+s→es`, then `es+t→est`, then `l+o→lo`, then `lo+w→low`, … . *That ordered list is the tokenizer.* Order matters because later merges depend on earlier ones having already fired (`est` can only form after `es` exists).

### Encoding: applying the tokenizer to new text

To tokenize a *new* string at inference time, you **replay the learned merges, in the order they were learned**:

1. First **pre-tokenize**: split the text into rough chunks (usually words/whitespace/punctuation) using a fixed regex — GPT-2 uses a well-known pattern. This just prevents merges from spanning across, say, a space into the next word, which keeps tokens linguistically sensible and keeps the algorithm fast.
2. Break each chunk into base bytes.
3. Walk the merge list top to bottom; for each rule, replace every matching adjacent pair in the chunk. After all rules, the chunk is a sequence of final tokens.
4. Map tokens to their integer IDs.

Because the base units are bytes, an input the tokenizer has never seen (a weird Unicode symbol, a code snippet) still tokenizes fine — it simply doesn't trigger many merges and stays as smaller pieces or raw bytes. This is why byte-level BPE has **no OOV**. Decoding is the trivial inverse: map IDs back to token strings, concatenate, and interpret the bytes as UTF-8.

A couple of practical notes: **special tokens** like `<|endoftext|>` are added to the vocabulary by hand (not learned) to mark document boundaries and control behavior; and the merges are **greedy and deterministic**, so the same text always tokenizes the same way — which is what makes the map reversible and reproducible. GPT-2, GPT-3, GPT-4, and Llama all use byte-level BPE variants; the differences are mostly the pre-tokenization regex and the vocabulary size.

---

## Compression ratio, vocabulary size, and "sparsity"

A clean way to measure how well a tokenizer packs text is the **compression ratio**:

$$
\text{compression ratio} = \frac{\text{num\_bytes}}{\text{num\_tokens}}
$$

Read it literally: it is the average number of raw UTF-8 bytes that get folded into a single token, measured on some held-out corpus. A byte-level scheme with no merges has a ratio of exactly 1 (one byte per token). A good subword tokenizer for English lands around 4 (roughly four bytes — about four characters — per token). **The larger the compression ratio, the fewer tokens a given passage becomes**, i.e. the shorter the sequence length $n$. And since attention is $O(n^2)$, a higher compression ratio directly means cheaper attention, more text per context window, and faster training and inference. So all else equal, we *want* a high compression ratio.

The natural lever is **vocabulary size**. Give BPE a bigger target vocabulary and it performs more merges, learning longer tokens (whole words, common phrases). Longer tokens swallow more bytes each, so the compression ratio goes up and sequences get shorter. This is why the CS336 lecture notes that *"one could increase compression ratio by increasing vocabulary size."*

But the same line adds the warning: doing so leads to **sparsity** — so what does "sparsity" mean here? The vocabulary is a long tail: a handful of tokens (like `the`, `,`, a space) appear constantly, while most tokens appear rarely. When you push the vocabulary very large, the *new* tokens you add are, by construction, ever-rarer pieces — a specific long word, a niche phrase. Two problems follow:

- **Under-trained embeddings.** Each token has its own embedding row (input side) and its own row in the output softmax projection. Those rows only get a gradient update when that token actually appears in a training batch. A token that shows up a handful of times in the entire corpus gets a handful of updates — so its embedding stays **poorly trained**, close to its random initialization. "Sparsity" is exactly this: a large fraction of the vocabulary is activated so infrequently that its parameters receive sparse, weak learning signal. You've added parameters that the data can't actually teach.
- **Bloated, wasteful input/output layers.** The embedding matrix and the final softmax both scale linearly with $|V|$. A very large vocabulary inflates parameter count and the FLOPs of those two layers — and you're paying that cost for many rare tokens that carry little learned value.

So there is a genuine tension. Small vocabulary → long sequences → expensive quadratic attention, but every token is common and well-trained. Large vocabulary → short sequences → cheap attention, but a fat, partly under-trained (sparse) embedding/softmax and diminishing returns. The compression ratio is what you gain; sparsity and a bloated input/output layer are what you pay. This is why production tokenizers cluster in a **sweet-spot range of roughly 32k–130k tokens** — large enough for good compression, small enough that most tokens are seen often enough to be learned well.

---

## Do tokenizers need to be trained?

"Trained" is doing two different jobs in that question, so it's worth separating them.

**Subword tokenizers (BPE, WordPiece, Unigram) are "trained" — but not the way the model is.** Building a BPE vocabulary means running the merge loop over a corpus: pure frequency counting, no labels, no gradient descent, no neural network. It is a cheap, one-time, *unsupervised* pass that happens **before** and **separately from** the LLM's training. Once built, the tokenizer is **frozen** — the exact same merge list and vocabulary are reused for all model pre-training, fine-tuning, and inference. (It has to be frozen: token ID 8,417 must mean the same string forever, or the model's learned embeddings become meaningless.) So yes, they need a training pass, but it is a lightweight statistical fit, not the giant gradient-based optimization that trains the model itself.

**Fixed tokenizers need no training at all.** Character-level and, especially, **byte-level** tokenization have nothing to learn — the vocabulary is simply the 256 byte values, defined by the UTF-8 standard. You can tokenize with zero corpus and zero fitting. The cost, as we saw, is longer sequences.

**And do we need a separate tokenizer at all?** Increasingly, maybe not. A line of **tokenizer-free** models feeds raw bytes directly to the network and lets the model itself learn to group them:

- **ByT5** and **CANINE** operate on characters/bytes with no learned vocabulary.
- **MambaByte** runs a state-space model over raw bytes, sidestepping attention's quadratic cost so long byte sequences become affordable.
- **Byte Latent Transformer (BLT)** (Meta, 2024) dynamically groups bytes into variable-length **patches** — spending more compute where the next byte is hard to predict and less where it's easy — and reports matching Llama-3-class quality while using up to ~50% fewer inference FLOPs.

These aren't yet the default, but they're strong evidence that a *separately trained* subword tokenizer is a **pragmatic, dominant convention rather than a fundamental requirement**. Its whole reason for existing is to shorten sequences enough to make attention affordable while keeping the vocabulary trainable; if a different architecture removes the quadratic bottleneck or learns the grouping end-to-end, the external tokenizer can fade. For now, though, byte-level BPE in the ~32k–130k range remains the workhorse — a small, boring, frozen preprocessing step that quietly shapes the cost and behavior of everything the model does.

---

## Takeaways

- A model consumes **integer IDs**, not text; the tokenizer is the reversible `text ↔ IDs` map, and choosing it is unavoidable (even "raw bytes" is a choice).
- Granularity trades **vocabulary size** (input/output-layer cost, $\propto |V|$) against **sequence length** (attention cost, $\propto n^2$). Word = short seqs but huge/brittle vocab; character/byte = tiny/robust vocab but long seqs; **subword is the tunable middle.**
- **BPE** learns subwords by greedily, repeatedly merging the most frequent adjacent pair from a byte-level start; its output is an **ordered merge list**, which *is* the tokenizer, replayed in order to encode new text — with no OOV thanks to the byte fallback.
- **Compression ratio $=\frac{\text{bytes}}{\text{tokens}}$** rises with vocabulary size (shorter sequences, cheaper attention), but too-large vocabularies cause **sparsity** — rarely-seen tokens whose embeddings are under-trained and a bloated softmax — which is why real vocabularies sit around 32k–130k.
- Subword tokenizers need a cheap, frozen, one-time **statistical** training pass (not gradient training); byte/char schemes need none; and tokenizer-free byte models (ByT5, MambaByte, BLT) show the external tokenizer is a strong default, not a law of nature.

---

## Sources

- [Stanford CS336, *Language Modeling from Scratch* — Lecture 1: Overview and Tokenization](https://stanford-cs336.github.io/spring2025-lectures/) (source of the tokenized-example image and the compression-ratio / sparsity framing).
- [Hugging Face LLM Course, Chapter 6 — *Byte-Pair Encoding tokenization*](https://huggingface.co/learn/llm-course/en/chapter6/5) (step-by-step BPE training and encoding).
- Sennrich, Haddow, Birch (2016), [*Neural Machine Translation of Rare Words with Subword Units*](https://arxiv.org/abs/1508.07909) (BPE for NLP).
- Gage (1994), *A New Algorithm for Data Compression* (the original BPE compression algorithm).
- Radford et al. (2019), [*Language Models are Unsupervised Multitask Learners* (GPT-2)](https://cdn.openai.com/better-language-models/language_models_are_unsupervised_multitask_learners.pdf) (byte-level BPE and pre-tokenization regex).
- Xue et al. (2022), [*ByT5: Towards a Token-Free Future with Pre-trained Byte-to-Byte Models*](https://arxiv.org/abs/2105.13626).
- Wang et al. (2024), [*MambaByte: Token-free Selective State Space Model*](https://arxiv.org/abs/2401.13660).
- Pagnoni et al. (2024), [*Byte Latent Transformer: Patches Scale Better Than Tokens*](https://arxiv.org/abs/2412.09871).
