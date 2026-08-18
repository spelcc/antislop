#!/usr/bin/env python3
"""Build a deterministic, model-balanced French editorial corpus from Compar:IA.

DuckDB is intentionally an execution-time dependency instead of a project dependency:

    uv run --with duckdb python tools/build_comparia_corpus.py \
      --recipe corpora/comparia-fr-editorial.json \
      --output /path/to/corpus
"""
from __future__ import annotations

import argparse
import collections
import dataclasses
import hashlib
import json
import re
import sys
import unicodedata
import urllib.request
from pathlib import Path
from typing import Iterable, Sequence

WORD_RE = re.compile(r"[^\W\d_]+(?:[’'][^\W\d_]+)?", re.UNICODE)
FENCED_CODE_RE = re.compile(r"```.*?```|~~~.*?~~~", re.DOTALL)
INLINE_CODE_RE = re.compile(r"`[^`\n]+`")
IMAGE_RE = re.compile(r"!\[[^\]]*\]\([^)]*\)")
LINK_RE = re.compile(r"\[([^\]]+)\]\([^)]*\)")
URL_RE = re.compile(r"https?://\S+")
HTML_RE = re.compile(r"<[^>]+>")
MULTISPACE_RE = re.compile(r"[ \t]+")
MANY_BLANKS_RE = re.compile(r"\n{3,}")


@dataclasses.dataclass(frozen=True)
class Candidate:
    source_id: int
    side: str
    model: str
    prompt_id: str
    response_id: str
    categories: tuple[str, ...]
    raw_words: int


@dataclasses.dataclass(frozen=True)
class SelectedDocument:
    candidate: Candidate
    text: str
    clean_words: int
    sha256: str


def sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()



def clean_response(value: str) -> str:
    text = unicodedata.normalize("NFC", value).replace("\r\n", "\n").replace("\r", "\n")
    text = FENCED_CODE_RE.sub("\n", text)
    text = INLINE_CODE_RE.sub(" ", text)
    text = IMAGE_RE.sub(" ", text)
    text = LINK_RE.sub(r"\1", text)
    text = URL_RE.sub(" ", text)
    text = HTML_RE.sub(" ", text)
    lines = []
    for line in text.splitlines():
        line = re.sub(r"^\s{0,3}(?:#{1,6}\s+|[-*+]\s+|>\s*|\d+[.)]\s+)", "", line)
        line = line.replace("**", "").replace("__", "")
        lines.append(MULTISPACE_RE.sub(" ", line).strip())
    return MANY_BLANKS_RE.sub("\n\n", "\n".join(lines)).strip()


def count_words(value: str) -> int:
    return len(WORD_RE.findall(value))


def sql_quote(value: str) -> str:
    return "'" + value.replace("'", "''") + "'"


def deterministic_key(model: str, prompt_id: str) -> str:
    return sha256_text(model + "\0" + prompt_id)


def dedupe_model_prompts(candidates: Iterable[Candidate]) -> dict[str, list[Candidate]]:
    by_model: dict[str, dict[str, Candidate]] = collections.defaultdict(dict)
    for candidate in candidates:
        previous = by_model[candidate.model].get(candidate.prompt_id)
        if previous is None or (
            candidate.response_id,
            candidate.source_id,
            candidate.side,
        ) < (
            previous.response_id,
            previous.source_id,
            previous.side,
        ):
            by_model[candidate.model][candidate.prompt_id] = candidate
    return {model: list(rows.values()) for model, rows in by_model.items()}


def select_candidate_reserves(
    candidates: Iterable[Candidate],
    models: Sequence[str],
    target_per_model: int,
    reserve_per_model: int,
    allowed_categories: Sequence[str],
    diversity_window: int,
) -> dict[str, list[Candidate]]:
    by_model = dedupe_model_prompts(candidates)
    missing = [model for model in models if model not in by_model]
    if missing:
        raise ValueError(f"models with no eligible candidates: {', '.join(missing)}")

    requested = target_per_model + reserve_per_model
    used_prompts: set[str] = set()
    used_responses: set[str] = set()
    selected: dict[str, list[Candidate]] = {}
    allowed = set(allowed_categories)

    # Rarest models get first choice so abundant models cannot consume their prompts.
    order = sorted(models, key=lambda model: (len(by_model[model]), model))
    for model in order:
        remaining = sorted(
            by_model[model], key=lambda row: deterministic_key(model, row.prompt_id)
        )
        chosen: list[Candidate] = []
        category_counts: collections.Counter[str] = collections.Counter()

        while len(chosen) < requested:
            pool: list[Candidate] = []
            for row in remaining:
                if row.prompt_id in used_prompts or row.response_id in used_responses:
                    continue
                pool.append(row)
                if len(pool) >= diversity_window:
                    break
            if not pool:
                break

            def diversity_score(row: Candidate) -> tuple[int, str]:
                matching = [cat for cat in row.categories if cat in allowed]
                scarcity = min((category_counts[cat] for cat in matching), default=10**9)
                return scarcity, deterministic_key(model, row.prompt_id)

            pick = min(pool, key=diversity_score)
            chosen.append(pick)
            used_prompts.add(pick.prompt_id)
            used_responses.add(pick.response_id)
            for category in pick.categories:
                if category in allowed:
                    category_counts[category] += 1
            remaining.remove(pick)

        if len(chosen) < target_per_model:
            raise ValueError(
                f"{model}: only {len(chosen)} globally unique candidates remain; "
                f"need at least {target_per_model}"
            )
        selected[model] = chosen
    return selected


def finalize_documents(
    selected_candidates: dict[str, list[Candidate]],
    responses: dict[tuple[int, str], str],
    models: Sequence[str],
    target_per_model: int,
    minimum_clean_words: int,
) -> dict[str, list[SelectedDocument]]:
    used_clean_hashes: set[str] = set()
    output: dict[str, list[SelectedDocument]] = {}
    for model in models:
        documents: list[SelectedDocument] = []
        for candidate in selected_candidates[model]:
            response = responses.get((candidate.source_id, candidate.side))
            if response is None:
                continue
            text = clean_response(response)
            words = count_words(text)
            if words < minimum_clean_words:
                continue
            digest = sha256_text(text)
            if digest in used_clean_hashes:
                continue
            used_clean_hashes.add(digest)
            documents.append(SelectedDocument(candidate, text, words, digest))
            if len(documents) == target_per_model:
                break
        if len(documents) != target_per_model:
            raise ValueError(
                f"{model}: only {len(documents)} clean documents remain after normalization; "
                f"need {target_per_model}. Increase reserve_per_model or relax the recipe explicitly."
            )
        output[model] = documents
    return output


def query_candidates(recipe: dict) -> tuple[object, list[Candidate]]:
    try:
        import duckdb  # type: ignore
    except ImportError as exc:  # pragma: no cover - exercised by CLI environment
        raise RuntimeError(
            "DuckDB is required. Run with `uv run --with duckdb python ...`."
        ) from exc

    source = recipe["source"]["parquet_url"]
    models = recipe["models"]
    allowed = recipe["allowed_categories"]
    excluded = recipe["excluded_categories"]
    language = recipe["language"]
    minimum = int(recipe["minimum_clean_words"])
    model_sql = ",".join(sql_quote(model) for model in models)
    allowed_sql = ",".join(sql_quote(category) for category in allowed)
    excluded_sql = ",".join(sql_quote(category) for category in excluded)

    connection = duckdb.connect()
    sql = f"""
    WITH source AS (
      SELECT id, 'a' AS side, opening_msg, categories, languages, model_a_name AS model,
             list_last(list_filter(conversation_a, x -> x.role = 'assistant')).content AS response
      FROM read_parquet({sql_quote(source)})
      UNION ALL
      SELECT id, 'b' AS side, opening_msg, categories, languages, model_b_name AS model,
             list_last(list_filter(conversation_b, x -> x.role = 'assistant')).content AS response
      FROM read_parquet({sql_quote(source)})
    )
    SELECT id, side, model,
           sha256(lower(trim(regexp_replace(opening_msg, '\\s+', ' ', 'g')))) AS prompt_id,
           sha256(trim(response)) AS response_id,
           categories,
           array_length(regexp_extract_all(response, '[[:alpha:]]+(?:[’''][[:alpha:]]+)?')) AS raw_words
    FROM source
    WHERE model IN ({model_sql})
      AND list_contains(languages, {sql_quote(language)})
      AND list_has_any(categories, [{allowed_sql}])
      AND NOT list_has_any(categories, [{excluded_sql}])
      AND response IS NOT NULL
      AND array_length(regexp_extract_all(response, '[[:alpha:]]+(?:[’''][[:alpha:]]+)?')) >= {minimum}
    """
    rows = connection.execute(sql).fetchall()
    candidates = [
        Candidate(
            source_id=int(row[0]),
            side=str(row[1]),
            model=str(row[2]),
            prompt_id=str(row[3]),
            response_id=str(row[4]),
            categories=tuple(row[5] or ()),
            raw_words=int(row[6]),
        )
        for row in rows
    ]
    return connection, candidates


def fetch_responses(connection: object, source_url: str, rows: Iterable[Candidate]) -> dict[tuple[int, str], str]:
    connection.execute("CREATE TEMP TABLE selected_comparia(id BIGINT, side VARCHAR)")
    unique = sorted({(row.source_id, row.side) for row in rows})
    connection.executemany("INSERT INTO selected_comparia VALUES (?, ?)", unique)
    sql = f"""
    WITH source AS (
      SELECT id, 'a' AS side,
             list_last(list_filter(conversation_a, x -> x.role = 'assistant')).content AS response
      FROM read_parquet({sql_quote(source_url)})
      UNION ALL
      SELECT id, 'b' AS side,
             list_last(list_filter(conversation_b, x -> x.role = 'assistant')).content AS response
      FROM read_parquet({sql_quote(source_url)})
    )
    SELECT source.id, source.side, source.response
    FROM source
    JOIN selected_comparia USING (id, side)
    """
    return {(int(row[0]), str(row[1])): str(row[2]) for row in connection.execute(sql).fetchall()}


def source_headers(url: str) -> dict[str, str]:
    try:
        request = urllib.request.Request(url, method="HEAD")
        with urllib.request.urlopen(request, timeout=30) as response:
            return {
                key: value
                for key in ("ETag", "Last-Modified", "Content-Length")
                if (value := response.headers.get(key)) is not None
            }
    except Exception:
        return {}


def write_corpus(output: Path, recipe: dict, documents: dict[str, list[SelectedDocument]]) -> None:
    if output.exists() and any(output.iterdir()):
        raise ValueError(f"output directory is not empty: {output}")
    output.mkdir(parents=True, exist_ok=True)
    document_dir = output / "documents"
    document_dir.mkdir()

    index: list[dict] = []
    category_totals: collections.Counter[str] = collections.Counter()
    total_words = 0
    for model in recipe["models"]:
        safe_model = re.sub(r"[^A-Za-z0-9._-]+", "-", model).strip("-")
        for ordinal, document in enumerate(documents[model], 1):
            filename = f"{safe_model}-{ordinal:04d}.txt"
            (document_dir / filename).write_text(document.text + "\n", encoding="utf-8")
            total_words += document.clean_words
            category_totals.update(document.candidate.categories)
            index.append(
                {
                    "file": filename,
                    "model": model,
                    "source_id": document.candidate.source_id,
                    "source_side": document.candidate.side,
                    "prompt_id": document.candidate.prompt_id,
                    "categories": list(document.candidate.categories),
                    "words": document.clean_words,
                    "sha256": document.sha256,
                }
            )

    recipe_bytes = json.dumps(recipe, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
    manifest = {
        "schema_version": 1,
        "name": recipe["name"],
        "language": recipe["language"],
        "source": recipe["source"],
        "source_headers": source_headers(recipe["source"]["parquet_url"]),
        "recipe_sha256": hashlib.sha256(recipe_bytes).hexdigest(),
        "documents": len(index),
        "words": total_words,
        "models": len(recipe["models"]),
        "documents_per_model": recipe["documents_per_model"],
        "minimum_clean_words": recipe["minimum_clean_words"],
        "allowed_categories": recipe["allowed_categories"],
        "excluded_categories": recipe["excluded_categories"],
        "category_occurrences": dict(sorted(category_totals.items())),
        "constraints": [
            "French conversations only",
            "editorial/knowledge categories required",
            "creative/lifestyle categories excluded",
            "globally unique opening prompts across models",
            "exact raw response deduplication before selection",
            "exact cleaned response deduplication after normalization",
            "fenced/inline code and markup plumbing removed",
        ],
        "model_names": recipe["models"],
    }
    (output / "index.json").write_text(json.dumps(index, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    (output / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def availability_report(candidates: Iterable[Candidate], models: Sequence[str]) -> dict[str, int]:
    by_model = dedupe_model_prompts(candidates)
    return {model: len(by_model.get(model, ())) for model in models}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--recipe", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    recipe = json.loads(args.recipe.read_text(encoding="utf-8"))
    connection, candidates = query_candidates(recipe)
    availability = availability_report(candidates, recipe["models"])
    print("Compar:IA eligible distinct prompts after category/language/length filtering:")
    for model in sorted(recipe["models"], key=lambda item: (availability[item], item)):
        print(f"  {model:38s} {availability[model]:6d}")

    selected = select_candidate_reserves(
        candidates,
        recipe["models"],
        int(recipe["documents_per_model"]),
        int(recipe["reserve_per_model"]),
        recipe["allowed_categories"],
        int(recipe.get("category_diversity_window", 256)),
    )
    print(
        f"Selected {sum(len(rows) for rows in selected.values())} pre-clean candidates; "
        f"target {recipe['documents_per_model']}/model with up to {recipe['reserve_per_model']} reserves/model."
    )
    if args.dry_run:
        return 0
    if args.output is None:
        raise ValueError("--output is required unless --dry-run is used")

    responses = fetch_responses(
        connection,
        recipe["source"]["parquet_url"],
        (row for rows in selected.values() for row in rows),
    )
    documents = finalize_documents(
        selected,
        responses,
        recipe["models"],
        int(recipe["documents_per_model"]),
        int(recipe["minimum_clean_words"]),
    )
    write_corpus(args.output, recipe, documents)
    total = sum(len(rows) for rows in documents.values())
    total_words = sum(doc.clean_words for rows in documents.values() for doc in rows)
    print(f"Wrote {total} documents / {total_words} clean words to {args.output}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (RuntimeError, ValueError) as error:
        print(f"error: {error}", file=sys.stderr)
        raise SystemExit(2)
