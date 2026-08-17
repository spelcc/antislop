# antislop

Deterministic writing analysis in Rust.

`antislop` measures lexical diversity, repeated structures, corpus over-representation and n-gram fingerprints. It is inspired by the corpus-comparison method in *Slop Forensics* (arXiv:2510.15061v2), but it does **not** claim to detect whether a text was written by AI.

The useful question is narrower: which patterns in this text are unusually common relative to a chosen baseline?

## What is deterministic

Given the same files, configuration and binary version, the output is the same:

- token counts and lexical diversity;
- Root-TTR;
- MATTR with a 500-token window;
- HD-D with a sample size of 42;
- Distinct-1, Distinct-2 and Distinct-3;
- word, bigram and trigram frequencies;
- document frequency for each pattern;
- target/baseline over-representation ratios;
- simple structural rules such as repeated `not X, but Y` forms;
- a localized fingerprint signal showing exactly which patterns contributed.

There is no LLM call and no hidden judge.

## Build

```bash
cargo build --release
cargo test
```

## Analyze one text

```bash
antislop analyze article.txt --language fr
cat article.txt | antislop analyze --language fr
```

## Build corpus profiles

A profile treats each file as one document. Directory inputs are scanned recursively.

```bash
antislop profile corpus/human/ --language fr -o human.json
antislop profile corpus/llm/ --language fr -o llm.json
```

## Build a fingerprint

Patterns must occur in at least three target documents by default. The default fingerprint keeps 120 words, 40 bigrams and 40 trigrams, matching the 200-pattern shape used in the paper. The recurrence threshold avoids treating a one-off accident as a corpus fingerprint.

```bash
antislop fingerprint \
  --target llm.json \
  --baseline human.json \
  --min-documents 3 \
  --word-limit 120 \
  --bigram-limit 40 \
  --trigram-limit 40 \
  -o fingerprint.json
```

The ratio for a pattern is:

```text
frequency(target) / frequency(baseline)
```

A `null` ratio means that the pattern appeared in the target corpus but not in the baseline corpus. It is kept separate instead of pretending division by zero is a meaningful giant number.

## Analyze against a fingerprint

```bash
antislop analyze article.txt \
  --language fr \
  --fingerprint fingerprint.json
```

`fingerprint_signal_per_1000_tokens` is a comparison signal, not an AI probability. It sums only finite over-representation ratios. Patterns absent from the baseline remain visible as `ratio: null` and are counted separately in `zero_baseline_occurrences` instead of receiving an invented infinite weight. Each hit includes its pattern, occurrence count, ratio and contribution.

## Corpus design matters

Do not compare a French design essay to a generic English web corpus and then act surprised when language exists. Use comparable corpora: same language, broadly similar genre and ideally similar length/date distribution.

Useful baselines can include:

- human editorial corpus;
- one publication's accepted corpus;
- one author's corpus;
- generated corpus from a specific model or workflow.

This makes it possible to measure `slop`, self-duplication and house-style repetition as separate phenomena instead of forcing them into one mystical score.

## Method notes

The project deliberately separates three phases:

1. `profile`: observable corpus counts and diversity metrics;
2. `fingerprint`: relative over-representation between two corpora;
3. `analyze`: deterministic matches in a new text.

The original research implementation is available at `sam-paech/slop-forensics`. `antislop` is an independent Rust implementation focused on reusable local/CI analysis, not a line-by-line port.

## License

MIT.
