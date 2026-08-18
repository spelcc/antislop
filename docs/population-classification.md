# Human / LLM population classification

`antislop nearest` and `antislop classify` answer two different questions:

- **nearest**: which stored writing populations resemble this document under several transparent representations?
- **classify**: given a labeled calibration set, which broad class (Human or LLM) is better supported, and how well calibrated was that decision rule on unseen documents?

Neither command identifies a literal author or proves which model produced a text.

## Candidate populations

A candidate manifest can mix human corpora and LLM models. Every candidate has a fingerprint, a class (`human` or `llm`), and optionally a style profile. The production French experiment used:

Human populations:

- Arthur editorial;
- Europarl;
- GlobalVoices;
- Gutenberg FR;
- News Commentary;
- Wikisource FR.

LLM populations:

- 28 Compar:IA model fingerprints, each built from 300 cleaned French responses.

FreCDo is deliberately excluded from this **long-form population** experiment: only about eight documents survive the >=350-word requirement, which is not enough for a stable long-form candidate fingerprint. FreCDo remains useful inside broad lexical/n-gram baselines.

All 34 fair fingerprints use the same contract:

- fingerprint schema 4;
- recurrence threshold `min_documents=5`;
- `wordfreq-large-fr-v3` lexical reference;
- no accepted-author guard;
- up to 120 words, 40 bigrams and 40 trigrams.

Wikisource is retained as a distinct human control but remains sparse: only 18 documents are available for its candidate signature in this split, so its candidate-specific signal should not be interpreted alone.

## Three independent nearest axes

### Rank distance

A single document is fingerprinted with `min_documents=1` against the same human reference baseline. Its ordered 120-word / 40-bigram / 40-trigram feature list is compared with every candidate using normalized rank distance. Lower is closer.

Rank distance is useful for global fingerprint shape, but it is not intrinsically a Human/LLM score. In the current representation it is frequently LLM-heavy even for genuine human texts, which is exactly why the calibrated classifier does not treat the nearest rank as a verdict.

### Document signal

Every candidate fingerprint is applied directly to the document. The resulting signal reflects candidate-specific patterns that actually occur in the target, normalized per 1,000 tokens. Higher is stronger.

### Style distance

A topic-light style profile compares sentence rhythm (paragraph rhythm is excluded), punctuation, pronouns, function words and sentence starters. Lower is closer.

All three axes remain visible. They are never collapsed into an undocumented hand-written score.

## Calibration features

`calibrate` derives five features from the complete candidate comparison:

1. `distance_margin_llm` = best Human distance - best LLM distance;
2. `signal_margin_llm` = best LLM signal - best Human signal;
3. `style_margin_llm` = best Human style distance - best LLM style distance;
4. `top5_distance_llm_fraction`;
5. `top5_signal_llm_fraction`.

Positive margins point toward LLM under their respective metric. The features are standardized on the **train split only**, then passed to a deterministic L2-regularized logistic regression. Class weights are balanced, so the fitted prior is 50/50 even when the supplied train split is numerically imbalanced.

The classifier JSON stores feature means/scales, weights, intercept, candidate identities and holdout evaluation. It also stores SHA-256 identities for the exact baseline, candidate fingerprints and optional style profiles. `classify` refuses a different baseline or candidate artifact instead of silently applying an obsolete calibration.

## Actionable CI diagnostics

`classify --ci --min-human-probability 0.70` exits with code `2` when the calibrated Human probability is below the editorial threshold. Classification report schema **v2** adds machine-readable correction evidence instead of asking a writer to infer edits from a document-level probability.

`fixes[]` is built from the five strongest LLM document-signal populations and up to five Human signal populations. For each source sentence, antislop compares local multi-word fingerprint signal, keeps only sentences where the LLM-side average exceeds the Human-side average, and reports:

- exact `start_line` / `end_line` and sentence text;
- local LLM signal, Human signal and margin;
- overlapping n-grams merged into maximal bad phrases rather than printing six adjacent trigrams;
- supporting LLM candidate labels and aggregate signal;
- `high` or `medium` evidence confidence;
- an edit instruction that explicitly preserves facts, names, numbers, citations and necessary domain terminology.

Single-model bigrams are deliberately excluded from actionable fixes because they are too easy to confound with topic or tokenization accidents. A bigram must recur in at least two top LLM fingerprints; a trigram-or-longer phrase may be retained with one model and is labeled `medium` unless cross-model support makes it stronger.

`style_fixes[]` is narrower by design. Antislop does **not** tell a writer to add/remove arbitrary function words merely because a regression coefficient likes them. It only localizes style when the nearest Human profile shows a defensible sentence-rhythm deviation. For example, if the document has substantially fewer long sentences and a high short-sentence rate, source lines stacking multiple short sentences are reported with a suggestion to combine/subordinate related claims where editorially natural.

The CLI uses a line-preserving Markdown/Markdoc cleaner for classification diagnostics. Removed frontmatter, components, code, references and plumbing are masked without changing source line count, so `L68` points to line 68 of the original source. Numeric IRZ reference links are treated as plumbing and do not become phantom one-token sentences.

For automation, an editing agent should change only a small number of top-priority passages, rerun the exact classifier bundle, keep changes that improve the Human probability without damaging factual content, and repeat until the configured gate passes. The diagnostic does not estimate the probability gain of an individual edit; those effects remain empirical and must be checked by rerunning classification.

## Leakage-free French benchmark

The final experiment separates candidate-signature data, classifier training data and classifier test data.

### Human data

The candidate/reference baseline contains only the signature partitions. None of the calibration documents is present in that baseline or in a candidate signature.

| Human source | Candidate signature | Calibration train | Calibration test |
| --- | ---: | ---: | ---: |
| Arthur editorial | 102 | 13 | 13 |
| Europarl | 322 | 25 used | 25 used |
| GlobalVoices | 418 | 38 | 38 |
| Gutenberg FR | 77 | 20 | 20 |
| News Commentary | 417 | 38 | 38 |
| Wikisource FR | 18 | 6 | 6 |
| **Total used per calibration split** | | **140** | **140** |

The train-only human reference baseline contains 1,354 candidate-partition documents and about 1.40M normalized tokens.

### LLM data

Candidate fingerprints use the clean Compar:IA corpus:

- 28 models;
- 300 documents/model;
- 8,400 documents;
- 5.03M cleaned prose words.

Calibration uses **different Compar:IA prompts**, with zero prompt overlap with those 8,400 candidate documents:

- 5 documents/model for classifier train = 140;
- 5 documents/model for classifier test = 140.

The small per-model calibration count is forced by the strict long-prose filter after excluding the 300 signature documents: Claude 3.5 Sonnet v2 is the limiting model. Per-model accuracy percentages therefore have very high variance and are reported only as diagnostics, not as model rankings.

### Final holdout

The classifier trains on 280 documents (140 Human + 140 LLM) and is evaluated on a separate balanced 280-document holdout.

| Metric | Result |
| --- | ---: |
| Accuracy | **96.79%** |
| ROC AUC | **0.9947** |
| Brier score | **0.0253** |
| Log loss | **0.1098** |
| Expected Calibration Error (10 bins) | **0.0568** |
| Human -> Human | 136 / 140 |
| Human -> LLM | 4 / 140 |
| LLM -> Human | 5 / 140 |
| LLM -> LLM | 135 / 140 |

Human holdout diagnostics:

| Source | n | Accuracy | Mean predicted LLM probability |
| --- | ---: | ---: | ---: |
| Arthur | 13 | 100% | 0.052 |
| Europarl | 25 | 100% | 0.026 |
| GlobalVoices | 38 | 97.4% | 0.085 |
| Gutenberg | 20 | 100% | 0.025 |
| News Commentary | 38 | 92.1% | 0.153 |
| Wikisource | 6 | 100% | 0.005 |

The model-side groups contain only five test documents each. One error therefore means 80%; those per-model figures are too small to support meaningful comparative claims.

## Bill Gates IRZ case study

Using the fair 34-population candidate set and the balanced classifier, the published French Bill Gates article scores:

```text
Human 70.6%
LLM   29.4%
```

This result uses the classifier's 50/50 prior. It should be read as a calibrated population-comparison result, not a literal posterior probability of authorship.

Best evidence by axis:

| Axis | Best Human | Best LLM |
| --- | --- | --- |
| Rank distance | Arthur 0.4978 (#20 globally) | Kimi-K2 0.4894 (#1) |
| Document signal | Wikisource 17.9728 (#2) | GLM-5 20.0457 (#1) |
| Style distance | Arthur 0.5029 (#2) | GPT-5.4 0.4507 (#1) |

The raw rank-distance top five are all LLM candidates, while three of the top five document signals are human populations. On the calibration distribution, that signal composition is strongly Human-like and outweighs the LLM-heavy rank-distance and style evidence.

The exact logistic contributions toward the LLM class for this article are:

```text
intercept                       -0.1938
distance_margin_llm             -0.0002
signal_margin_llm               -0.2445
style_margin_llm                +0.4490
top5_distance_llm_fraction      +0.4300
top5_signal_llm_fraction        -1.3017
```

This is why exposing the evidence and feature contributions matters: “the nearest candidate is an LLM” is not equivalent to “the calibrated classifier says LLM.”

## Limits

The benchmark is intentionally stronger than a naive in-sample detector test, but it is still a bounded experiment:

- all LLM calibration data comes from Compar:IA and therefore shares collection conditions;
- human corpora differ in genre, period and translation history;
- some human source formatting has damaged paragraph boundaries, which is why paragraph rhythm is excluded;
- prompts are not perfectly matched between every LLM model and every human document;
- unseen future model families, deliberate humanization, translation, heavy editing and mixed human/LLM documents can shift the distribution;
- the 50/50 calibration prior is an evaluation convention, not the prevalence of LLM writing in any real deployment population.

For these reasons `classify` should be treated as calibrated forensic evidence inside the supplied population universe, not a universal AI detector.
