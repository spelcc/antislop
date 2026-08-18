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
- content-word unigram frequencies and literal adjacent bigram/trigram frequencies;
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


## Author style profiles

The `style` commands are separate from the LLM/human fingerprint. They measure whether a document sits near the statistical center of an author's or publication's prose without treating the result as an authorship probability.

Build a profile from one document per file:

```bash
antislop style profile corpus-arthur-fr/ \
  --language fr \
  -o arthur-fr.style.json
```

Corpus directories may contain metadata JSON files; style profiling only reads `.txt`, `.md`, `.markdown` and `.mdoc` documents. Markdown/Markdoc inputs are cleaned automatically before measurement.

Compare a new article:

```bash
antislop style compare article.mdoc \
  --profile arthur-fr.style.json
```

Or emit machine-readable JSON:

```bash
antislop style compare article.mdoc \
  --profile arthur-fr.style.json \
  --json
```

The profile currently measures five topic-light families:

- `rhythm`: sentence length and short/long sentence rates; paragraph rhythm is intentionally excluded because many corpora do not preserve original paragraph boundaries;
- `punctuation`: commas, semicolons, colons, questions, exclamations, parentheses, dashes and ellipses;
- `pronoun`: first-person singular/plural, `on` in French, and second-person usage;
- `function_word`: fixed high-frequency connective/function-word rates;
- `starter`: selected sentence-opening function words/connectors.

Each metric stores the corpus mean, median, median absolute deviation, p10/p90 and non-zero document rate. Comparison uses robust standardized deviations. Sparse function-word/starter metrics that occur in fewer than 10% of profile documents do not affect the global distance.

`overall_distance` is an equal-weight average of the five group distances. Lower means the document is closer to the center of the profile. `within_profile_band_ratio` reports how many scored metrics remain within two robust standard deviations. Neither value is an authorship probability, an AI detector, or proof that one person wrote a document.

For a useful author profile, prefer at least ~50,000 words across many documents; 100,000+ words is substantially better. Keep one source document per file so document-level distributions remain available.

## Sentence-level lint and CI

`analyze` gives a document-wide report. `lint` localizes the same deterministic evidence sentence by sentence so an editor can change the exact passage that carries the signal.

```bash
antislop lint article.txt --language fr
antislop lint article.txt --language fr --fingerprint slop-fr.json --json
```

Each finding contains the original sentence, sentence index, UTF-8 byte span, start/end line, structural rules and high-confidence fingerprint hits. Unigrams remain document-level diagnostics because single content words are usually topical. Bigrams contribute to the document signal but do not create local warnings by themselves. Local corpus warnings are reserved for trigrams and recovered longer phrases; structural rules remain independent. This makes the JSON suitable for GitHub annotations and editor integrations without turning every technical noun into a red flag.

CI thresholds are opt-in and independent:

```bash
antislop lint article.txt \
  --language fr \
  --fingerprint slop-fr.json \
  --max-structural-hits 2 \
  --max-document-signal 18 \
  --max-sentence-signal 8
```

Exit code `0` means the configured thresholds passed. Exit code `2` means the analysis succeeded but at least one configured threshold was exceeded. Do not copy threshold numbers from this example into a project: calibrate them against that project's accepted corpus first.

## How much corpus is useful?

For editorial fingerprinting, corpus size is better expressed in words than files. Practical starting points per language and per corpus are:

- below 25,000 words: exploratory only;
- around 50,000 words: minimum useful baseline;
- 100,000 words: solid working corpus;
- 250,000+ words: comfortable for rarer bigram/trigram estimates.

These are engineering recommendations, not hard statistical guarantees. Distribution matters as much as size. For LLM-vs-human fingerprints, keep language, genre and approximate register comparable. For an author-style corpus, varied subjects are useful because topic vocabulary changes while recurring stylistic habits remain visible.

The two sides do not need identical word counts because frequencies are normalized, but severely unbalanced corpora make zero-baseline patterns less trustworthy. Prefer roughly comparable sizes or use resampling before treating rare patterns as stable.

## Build corpus profiles

A profile treats each file as one document. Directory inputs are scanned recursively. Schema v3 corpus profiles keep content-filtered unigrams, literal adjacent bigrams/trigrams, prompt recurrence, optional model metadata, and a second stopword-stripped discovery representation used for exact phrase recovery.

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

For a publication or author-specific workflow, an optional accepted-prose guard can remove patterns that are also characteristic of the house style:

```bash
antislop fingerprint \
  --target llm.json \
  --baseline human.json \
  --guard accepted-author.json \
  --min-guard-ratio 2 \
  -o fingerprint.json
```

With a guard, a pattern present in accepted prose must still be at least `min-guard-ratio` times more frequent in the target corpus. Patterns absent from the guard remain eligible. The guard is a false-positive control, not evidence of authorship.

The ratio for a pattern is:

```text
frequency(target) / frequency(baseline)
```

A `null` ratio means that the pattern appeared in the target corpus but not in the baseline corpus. It is kept separate instead of pretending division by zero is a meaningful giant number. Fingerprint ranking assigns zero-baseline patterns finite evidence and also rewards recurrence across target documents, so a rare baseline absence cannot automatically monopolize the fingerprint.

## Analyze against a fingerprint

```bash
antislop analyze article.txt \
  --language fr \
  --fingerprint fingerprint.json
```

`fingerprint_signal_per_1000_tokens` is a comparison signal, not an AI probability. In schema-v4 fingerprints, French unigrams that survive both the human-baseline and `wordfreq` filters contribute low-weight document evidence but never create local warnings. Legacy schema-v3 unigram scoring remains zero for compatibility. Bigrams have reduced document-level weight. Trigrams and recovered longer phrases carry the strongest local evidence. Finite ratios are log-scaled and capped; zero-baseline phrase evidence receives a finite conservative weight rather than infinity. Each hit includes its pattern, n-gram width, signal class, source, recurrence support, ratio and contribution.

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

## Model-aware consensus fingerprints

For multi-model generated corpora, prefer a metadata manifest and build one fingerprint per model before deriving a cross-model consensus:

```bash
antislop profile \
  --manifest corpus-llm/index.json \
  --recover-phrases \
  --language fr \
  -o llm-fr.profile.json

antislop fingerprint \
  --target llm-fr.profile.json \
  --baseline human-fr.profile.json \
  --guard accepted-author.profile.json \
  --min-model-documents 2 \
  --min-models 2 \
  --models-output-dir model-fingerprints/ \
  -o slop-fr.json
```

The profile records `model_id`, inferred/explicit `family`, `prompt_id` and domains. Consensus entries expose model/family support. French lexical candidates are checked against the bundled `wordfreq` large-FR reference; literal n-grams and stopword-stripped discovery n-grams are kept separately; discovery trigrams can recover exact longer surface phrases.

Explore ranked fingerprint similarity with:

```bash
antislop cluster model-fingerprints/*.json arthur.json \
  -o cluster.json \
  --newick-output cluster.nwk
```

See [`docs/model-consensus-forensics.md`](docs/model-consensus-forensics.md) for methodology, caveats and the fixed holdout before/after benchmark.


## Rebuilding the French Compar:IA corpus

The repository includes a reproducible recipe for a clean, model-balanced French corpus sourced from the French Ministry of Culture's public Compar:IA Parquet:

```bash
uv run --with duckdb python tools/build_comparia_corpus.py \
  --recipe corpora/comparia-fr-editorial.json \
  --output /path/outside/git/comparia-fr-editorial-300
```

The checked-in recipe currently targets **28 models × 300 documents = 8,400 documents**. Each final response contains at least 350 cleaned prose words. Opening prompts are globally unique across models and responses are deduplicated before and after normalization. The recipe requires knowledge/editorial categories and rejects creative/lifestyle categories such as Arts, Entertainment, Food, Shopping and Personal Development.

On the Compar:IA resource identified by ETag `888b0bccc9a12948985365697653e494` (Last-Modified 2026-06-03), the recipe produces **5,030,124 cleaned words** with a median document length of 488 words. The final target is 300/model rather than the larger raw availability because code removal is intentionally applied before accepting a document: for example, only 331 Claude 3.5 Sonnet v2 responses remain at >=350 prose words after stripping code/markup.

Use `--dry-run` to inspect current source availability without writing the corpus. Raw corpus files and generated profiles remain outside Git; only the recipe and builder are versioned.

## Comparing a document with human and LLM populations

`nearest` can still rank one document against a directory of model fingerprints:

```bash
antislop nearest article.mdoc \
  --baseline human-fr.profile.json \
  --models-dir fingerprints/models \
  --language fr \
  --top 10
```

For a symmetric human/LLM comparison, use a candidate manifest instead. Every population has the same fingerprint representation and may also carry a style profile:

```json
{
  "schema_version": 1,
  "language": "fr",
  "candidates": [
    {
      "label": "arthur",
      "class": "human",
      "fingerprint": "arthur.fingerprint.json",
      "style_profile": "arthur.style.json"
    },
    {
      "label": "gpt-5.4",
      "class": "llm",
      "fingerprint": "gpt-5.4.fingerprint.json",
      "style_profile": "gpt-5.4.style.json"
    }
  ]
}
```

Then:

```bash
antislop nearest article.mdoc \
  --baseline human-train.profile.json \
  --candidates candidates.json \
  --language fr \
  --metric rank-distance
```

Three independent rankings are exposed:

- `rank-distance`: similarity between ordered 120-word / 40-bigram / 40-trigram fingerprints, lower is closer;
- `document-signal`: candidate-specific over-represented patterns actually present in the document, higher is stronger;
- `style-distance`: topic-light sentence rhythm, punctuation, pronouns, function words and sentence starters, lower is closer.

Candidate fingerprints must use the same language, fingerprint schema, recurrence threshold, lexical reference and guard configuration. `nearest` automatically strips Markdown/MDOC plumbing. The ranking remains descriptive similarity, not authorship attribution.

## Calibrated human-vs-LLM classification

A probability should not be invented from ranks. `calibrate` instead fits a deterministic balanced logistic regression from labeled documents that are separate from the candidate fingerprints:

```bash
antislop calibrate \
  --documents calibration.json \
  --baseline human-train.profile.json \
  --candidates candidates.json \
  --language fr \
  -o classifier.json
```

The calibration manifest explicitly marks `train` and `test` documents. The classifier uses five auditable features: best human-vs-LLM distance margin, signal margin, style margin, LLM fraction in the five nearest rank-distance candidates, and LLM fraction in the five strongest document-signal candidates.

Apply it with:

```bash
antislop classify article.mdoc \
  --classifier classifier.json \
  --baseline human-train.profile.json \
  --candidates candidates.json \
  --language fr
```

For CI, require at least 70% calibrated Human probability and print actionable correction guidance:

```bash
antislop classify article.mdoc \
  --classifier classifier.json \
  --baseline human-train.profile.json \
  --candidates candidates.json \
  --language fr \
  --ci --min-human-probability 0.70
```

A failing gate exits with code `2`. The output lists the LLM-leaning classifier features and then pinpoints source lines to rewrite. `Priority passages` include the exact sentence, merged bad phrase(s), local LLM-vs-Human signal margin, supporting LLM fingerprints, a `high`/`medium` evidence label, and a constrained rewrite instruction. Weak one-model bigrams are excluded. When the nearest Human style profile shows a defensible rhythm deviation, `Style hotspots` also identify exact source lines with short-sentence stacks. The JSON report schema v2 exposes the same data as `fixes[]` and `style_fixes[]` for automated editing loops.

`classify` reports Human/LLM probabilities, holdout accuracy/AUC/Brier/ECE, the best human and LLM evidence on each axis, and each feature's contribution to the logistic score. The fitted prior is deliberately **50% Human / 50% LLM**. A result such as `Human 70%` therefore means “70% under this calibrated balanced comparison,” not “70% real-world probability that a human wrote it.”

The current leakage-free French benchmark uses 34 populations (6 human corpora + 28 Compar:IA models), 280 calibration-train documents and 280 independent test documents. It reaches **96.8% accuracy, ROC AUC 0.9947, Brier 0.0253 and ECE 0.0568**. See [`docs/population-classification.md`](docs/population-classification.md) for the exact split, source-level results and caveats. English corpus sources and the separation between training and external validation are documented in [`docs/english-corpus-sources.md`](docs/english-corpus-sources.md).
