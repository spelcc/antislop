# Model-aware forensic fingerprints

This document describes the model-aware fingerprint pipeline added after the first IRZ sentence-lint implementation.

The goal remains deliberately narrower than AI detection: find linguistic patterns that are unusually recurrent in a generated corpus relative to a human baseline, while exposing enough provenance to audit why a pattern was selected.

## Corpus metadata

`antislop profile` accepts either ordinary files/directories or a JSON manifest:

```bash
antislop profile \
  --manifest corpus/index.json \
  --recover-phrases \
  --language fr \
  -o llm-fr.profile.json
```

A manifest entry can contain:

```json
{
  "file": "documents/claude-001.txt",
  "model_id": "claude-4-5-sonnet",
  "family": "claude",
  "prompt_id": "prompt-001",
  "domains": ["technology"]
}
```

For compatibility with the Compar:IA corpus preparation, `model`, `prompt_sha256` and `categories` are accepted as aliases. When `family` is absent, common model families are inferred deterministically from `model_id`.

Profiles store document frequency and prompt frequency separately and build an independent slice for every model.

## Per-model fingerprints and consensus

A corpus with model metadata produces both a consensus and optional per-model fingerprints:

```bash
antislop fingerprint \
  --target llm-fr.profile.json \
  --baseline human-fr.profile.json \
  --guard arthur-editorial.profile.json \
  --min-model-documents 2 \
  --min-models 2 \
  --models-output-dir fingerprints/models \
  -o fingerprints/consensus.json
```

A pattern first has to survive inside each model profile. The consensus then requires support from at least `--min-models` distinct models. Each consensus entry records its model IDs, model frequency, families and family frequency.

`--min-models 2` is the default. On the current Compar:IA calibration corpus, requiring three models was measurably too conservative because there are only ten sampled documents per model.

## Two n-gram representations

The profile deliberately stores two representations:

1. **literal n-grams**, built from adjacent normalized tokens, for exact linting;
2. **discovery n-grams**, built after French stopword removal, for finding recurrent content skeletons.

This restores the useful part of the Slop Forensics preprocessing without displaying malformed skip-grams to editors. Discovery keeps byte offsets into the original text, allowing an exact surface form to be recovered later.

Example:

```text
surface:   important de noter et de vérifier
skeleton:  important noter vérifier
```

## Long phrase recovery

Long phrase recovery follows the reference toolkit's conservative path: candidates start from a **three-content-word discovery trigram**. The recovered surface must contain 4–16 literal tokens, have balanced delimiters, and satisfy model-consensus recurrence.

The current 200-document training benchmark produces no long phrase that survives all of those constraints. That is a corpus-size result, not a disabled feature. The unit/integration tests cover phrase recovery explicitly. A larger per-model generation corpus is required before long consensus phrases should be used editorially.

## French lexical reference

French lexical entries are additionally checked against the `wordfreq` French large model through the Rust `wordfreq` / `wordfreq-model` packages. A unigram must be sufficiently over-represented relative to both the project human baseline and the general French lexical reference to rank highly.

Unigrams are still low-weight **document evidence**. They never create sentence warnings by themselves. This avoids turning topic vocabulary such as `API` or `cloud` into an editorial alarm merely because the human comparison corpus contains little of it.

The embedded lexical model is reported in the fingerprint as `wordfreq-large-fr-v3`.

## Sentence warnings

The evidence channels are intentionally asymmetric:

- lexical unigram: low-weight document evidence only;
- literal bigram: document evidence only;
- literal trigram: can create a local warning;
- recovered phrase (4+ tokens): can create a local warning;
- structural regex: independent local evidence, never an authorship claim.

## French structural forms

French structural lint covers several contrast templates independently, including:

- `pas seulement X, mais (aussi) Y`;
- `non pas X, mais Y`;
- `il ne s'agit pas de X mais de Y`;
- `ce n'est pas/plus simplement X, mais Y`;
- generic `ce n'est pas X, mais/c'est Y` with an explicit clause separator;
- `ne se contente pas de ...`;
- `pas tant X que Y`;
- `la vraie question / le vrai sujet`.

These forms also occur in genuine human prose. In the final calibration, at least one structural form appears in 18/128 Arthur editorial documents (14.1%) and 60/618 GlobalVoices documents (9.7%). They are localized editorial observations, not corpus-level AI evidence. Structural findings therefore remain separate from the fingerprint score.

## Rank distance and clustering

`antislop cluster` compares ordered fingerprint features and builds a deterministic average-linkage hierarchy:

```bash
antislop cluster \
  fingerprints/models/*.json \
  fingerprints/arthur.json \
  -o cluster.json \
  --newick-output cluster.nwk
```

Rank distance uses the paper-shaped top 200 features only:

- 120 words;
- 40 bigrams;
- 40 trigrams.

Missing features receive the rank after the longest list. The reported distance is a normalized rank-footrule-style distance. It is deterministic and useful for exploratory comparison, but it is not claimed to be an exact reimplementation of the paper's phylogenetic procedure.

With the current IRZ calibration corpus, Arthur's nearest ranked fingerprint is `gpt-oss-120b` at distance 0.486, but only 1/28 model fingerprints has a nearest neighbor from the same inferred family. The Arthur neighbor is therefore exploratory, not an authorship or lineage result. That family-clustering failure is itself informative: ten unique prompts per model are insufficient for a reliable model phylogeny, especially because the corpus was intentionally constructed with globally unique prompts to avoid topic leakage in the consensus fingerprint. The clustering feature is implemented, but family-level conclusions need a larger and prompt-matched generation corpus.

## Compar:IA reproducible corpus recipe

`corpora/comparia-fr-editorial.json` and `tools/build_comparia_corpus.py` provide the reproducible source pipeline used for the larger French LLM corpus. The builder queries the official Compar:IA Parquet directly with DuckDB instead of vendoring the 2.5 GB source file.

The strict recipe requires French, at least one knowledge/editorial category, no creative/lifestyle category, globally unique prompts, and exact response deduplication. It strips fenced/inline code and markup plumbing before enforcing the final 350-word prose minimum.

A live build against the resource with ETag `888b0bccc9a12948985365697653e494` produced:

- 28 models;
- 300 documents/model;
- 8,400 documents total;
- 5,030,124 cleaned words;
- median 488 words/document;
- 8,400 unique prompt IDs;
- 8,400 unique cleaned response hashes.

Raw availability is substantially larger, but a 300/model target is deliberately used for a balanced clean-prose corpus. Code-heavy models lose many nominally long answers after cleanup; Claude 3.5 Sonnet v2 is the limiting model with 331 eligible clean responses in the current source snapshot. Counting code tokens merely to hit a larger corpus target would undermine the stylistic comparison.

## Holdout benchmark

The before/after comparison fixes the data split:

- training target: 200 Compar:IA responses from 20 models;
- LLM holdout: 80 responses from 8 entirely unseen models;
- Arthur guard training: 102 accepted editorial texts;
- Arthur holdout: 26 accepted texts unseen by the guard;
- human baseline: 10,508 documents / 2,883,406 normalized tokens.

The old implementation and the model-aware implementation are evaluated on exactly the same holdout files.

| Metric | Before | Model-aware |
| --- | ---: | ---: |
| document-signal AUC, unseen LLM vs Arthur holdout | 0.8228 | **0.8385** |
| median LLM document signal | 24.51 | **48.84** |
| median Arthur document signal | 0.00 | 8.89 |
| LLM TPR at signal >= 40 | 37.5% | **57.5%** |
| Arthur FPR at signal >= 40 | 0.0% | **0.0%** |
| documents with local corpus warning, LLM | **65.0%** | 40.0% |
| documents with local corpus warning, Arthur | 23.1% | **19.2%** |

The result is a deliberate tradeoff: local warnings become less sensitive and somewhat more precise, while the document-level discrimination improves. No threshold is promoted to a CI failure gate from this benchmark alone.

The published Bill Gates IRZ article remains `0/368` through the actual IRZ Markdoc-masking bridge with the full 28-model consensus fingerprint.

## Relationship to the reference work

The implementation is inspired by Sam Paech's *Slop Forensics* / *Antislop* work, including per-model profiles, recurrence across prompts, stopword-stripped n-gram discovery, exact phrase recovery and ranked model comparisons. This project remains an independent Rust implementation optimized for deterministic local/editorial CI rather than generation-time backtracking or fine-tuning.

Reference implementation: `https://github.com/sam-paech/slop-forensics`

Paper: `https://arxiv.org/abs/2510.15061`

French lexical reference: `https://github.com/kampersanda/wordfreq-rs`, based on Robyn Speer's `wordfreq` data/model work. The upstream model release carries the dataset/model credits; `wordfreq-model` source code is MIT OR Apache-2.0.
