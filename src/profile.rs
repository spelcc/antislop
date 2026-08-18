use crate::discovery::{
    PhraseVariantAccumulator, PhraseVariantStats, collect_phrase_variants, discovery_occurrences,
    finalize_phrase_variants,
};
use crate::metadata::{DocumentMetadata, LoadedDocument};
use crate::metrics::{LexicalMetrics, lexical_metrics};
use crate::tokenize::{Language, content_tokens, tokenize};
use serde::{Deserialize, Serialize};
use std::collections::{BTreeMap, BTreeSet};

const PROFILE_SCHEMA_VERSION: u32 = 3;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DocumentProfile {
    pub id: String,
    pub lexical: LexicalMetrics,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub model_id: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub family: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub prompt_id: Option<String>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub domains: Vec<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ProfileSlice {
    pub document_count: usize,
    pub token_count: usize,
    pub content_token_count: usize,
    pub words: BTreeMap<String, PatternStats>,
    pub bigrams: BTreeMap<String, PatternStats>,
    pub trigrams: BTreeMap<String, PatternStats>,
    pub discovery_bigrams: BTreeMap<String, PatternStats>,
    pub discovery_trigrams: BTreeMap<String, PatternStats>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelProfile {
    pub family: String,
    #[serde(flatten)]
    pub slice: ProfileSlice,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CorpusProfile {
    pub schema_version: u32,
    pub language: Language,
    pub document_count: usize,
    pub token_count: usize,
    pub content_token_count: usize,
    pub documents: Vec<DocumentProfile>,
    pub words: BTreeMap<String, PatternStats>,
    pub bigrams: BTreeMap<String, PatternStats>,
    pub trigrams: BTreeMap<String, PatternStats>,
    pub discovery_bigrams: BTreeMap<String, PatternStats>,
    pub discovery_trigrams: BTreeMap<String, PatternStats>,
    #[serde(default, skip_serializing_if = "BTreeMap::is_empty")]
    pub phrase_variants: BTreeMap<String, BTreeMap<String, PhraseVariantStats>>,
    #[serde(default, skip_serializing_if = "BTreeMap::is_empty")]
    pub model_profiles: BTreeMap<String, ModelProfile>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PatternStats {
    pub count: usize,
    pub document_frequency: usize,
    #[serde(default)]
    pub prompt_frequency: usize,
    pub relative_frequency: f64,
}

#[derive(Debug, Clone, Copy, Default)]
pub struct ProfileOptions {
    pub recover_phrases: bool,
}

#[derive(Default)]
struct PatternAccumulator {
    counts: BTreeMap<String, usize>,
    documents: BTreeMap<String, BTreeSet<usize>>,
    prompts: BTreeMap<String, BTreeSet<String>>,
}

#[derive(Default)]
struct SliceAccumulator {
    document_count: usize,
    token_count: usize,
    content_token_count: usize,
    words: PatternAccumulator,
    bigrams: PatternAccumulator,
    trigrams: PatternAccumulator,
    discovery_bigrams: PatternAccumulator,
    discovery_trigrams: PatternAccumulator,
}

pub fn profile_documents(language: Language, documents: &[(String, String)]) -> CorpusProfile {
    let loaded: Vec<_> = documents
        .iter()
        .map(|(id, text)| LoadedDocument {
            id: id.clone(),
            text: text.clone(),
            metadata: DocumentMetadata::default(),
        })
        .collect();
    profile_loaded_documents(language, &loaded, ProfileOptions::default())
}

pub fn profile_loaded_documents(
    language: Language,
    documents: &[LoadedDocument],
    options: ProfileOptions,
) -> CorpusProfile {
    let mut aggregate = SliceAccumulator::default();
    let mut model_accumulators = BTreeMap::<String, (String, SliceAccumulator)>::new();
    let mut document_profiles = Vec::with_capacity(documents.len());
    let mut phrase_variants = BTreeMap::<String, BTreeMap<String, PhraseVariantAccumulator>>::new();

    for (doc_index, document) in documents.iter().enumerate() {
        let raw_tokens = tokenize(&document.text);
        document_profiles.push(DocumentProfile {
            id: document.id.clone(),
            lexical: lexical_metrics(&raw_tokens),
            model_id: document.metadata.model_id.clone(),
            family: document.metadata.family.clone(),
            prompt_id: document.metadata.prompt_id.clone(),
            domains: document.metadata.domains.clone(),
        });
        add_document(
            &mut aggregate,
            doc_index,
            &document.text,
            language,
            document.metadata.prompt_id.as_deref(),
        );
        if let Some(model_id) = document.metadata.model_id.as_deref() {
            let family = document
                .metadata
                .family
                .clone()
                .unwrap_or_else(|| model_id.to_string());
            let (_, accumulator) = model_accumulators
                .entry(model_id.to_string())
                .or_insert_with(|| (family, SliceAccumulator::default()));
            add_document(
                accumulator,
                doc_index,
                &document.text,
                language,
                document.metadata.prompt_id.as_deref(),
            );
        }
        if options.recover_phrases {
            collect_phrase_variants(
                &document.text,
                language,
                doc_index,
                document.metadata.prompt_id.as_deref(),
                document.metadata.model_id.as_deref(),
                document.metadata.family.as_deref(),
                &mut phrase_variants,
            );
        }
    }

    let aggregate = finalize_slice(aggregate);
    let model_profiles = model_accumulators
        .into_iter()
        .map(|(model_id, (family, accumulator))| {
            (
                model_id,
                ModelProfile {
                    family,
                    slice: finalize_slice(accumulator),
                },
            )
        })
        .collect();

    CorpusProfile {
        schema_version: PROFILE_SCHEMA_VERSION,
        language,
        document_count: aggregate.document_count,
        token_count: aggregate.token_count,
        content_token_count: aggregate.content_token_count,
        documents: document_profiles,
        words: aggregate.words,
        bigrams: aggregate.bigrams,
        trigrams: aggregate.trigrams,
        discovery_bigrams: aggregate.discovery_bigrams,
        discovery_trigrams: aggregate.discovery_trigrams,
        phrase_variants: if options.recover_phrases {
            finalize_phrase_variants(phrase_variants)
        } else {
            BTreeMap::new()
        },
        model_profiles,
    }
}

fn add_document(
    accumulator: &mut SliceAccumulator,
    doc_index: usize,
    text: &str,
    language: Language,
    prompt_id: Option<&str>,
) {
    let raw_tokens = tokenize(text);
    let content = content_tokens(text, language);
    accumulator.document_count += 1;
    accumulator.token_count += raw_tokens.len();
    accumulator.content_token_count += content.len();
    collect_tokens(&content, 1, doc_index, prompt_id, &mut accumulator.words);
    collect_tokens(
        &raw_tokens,
        2,
        doc_index,
        prompt_id,
        &mut accumulator.bigrams,
    );
    collect_tokens(
        &raw_tokens,
        3,
        doc_index,
        prompt_id,
        &mut accumulator.trigrams,
    );
    for (pattern, _, _) in discovery_occurrences(text, language, 2) {
        collect_pattern(
            pattern,
            doc_index,
            prompt_id,
            &mut accumulator.discovery_bigrams,
        );
    }
    for (pattern, _, _) in discovery_occurrences(text, language, 3) {
        collect_pattern(
            pattern,
            doc_index,
            prompt_id,
            &mut accumulator.discovery_trigrams,
        );
    }
}

fn collect_tokens(
    tokens: &[String],
    n: usize,
    doc_index: usize,
    prompt_id: Option<&str>,
    accumulator: &mut PatternAccumulator,
) {
    if tokens.len() < n {
        return;
    }
    for window in tokens.windows(n) {
        collect_pattern(window.join(" "), doc_index, prompt_id, accumulator);
    }
}

fn collect_pattern(
    pattern: String,
    doc_index: usize,
    prompt_id: Option<&str>,
    accumulator: &mut PatternAccumulator,
) {
    *accumulator.counts.entry(pattern.clone()).or_default() += 1;
    accumulator
        .documents
        .entry(pattern.clone())
        .or_default()
        .insert(doc_index);
    if let Some(prompt) = prompt_id {
        accumulator
            .prompts
            .entry(pattern)
            .or_default()
            .insert(prompt.to_string());
    }
}

fn finalize_slice(accumulator: SliceAccumulator) -> ProfileSlice {
    ProfileSlice {
        document_count: accumulator.document_count,
        token_count: accumulator.token_count,
        content_token_count: accumulator.content_token_count,
        words: finalize_patterns(accumulator.words),
        bigrams: finalize_patterns(accumulator.bigrams),
        trigrams: finalize_patterns(accumulator.trigrams),
        discovery_bigrams: finalize_patterns(accumulator.discovery_bigrams),
        discovery_trigrams: finalize_patterns(accumulator.discovery_trigrams),
    }
}

fn finalize_patterns(accumulator: PatternAccumulator) -> BTreeMap<String, PatternStats> {
    let denominator: usize = accumulator.counts.values().sum();
    accumulator
        .counts
        .into_iter()
        .map(|(pattern, count)| {
            let relative_frequency = if denominator == 0 {
                0.0
            } else {
                count as f64 / denominator as f64
            };
            let stat = PatternStats {
                count,
                document_frequency: accumulator.documents.get(&pattern).map_or(0, BTreeSet::len),
                prompt_frequency: accumulator.prompts.get(&pattern).map_or(0, BTreeSet::len),
                relative_frequency,
            };
            (pattern, stat)
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn tracks_document_and_prompt_frequency() {
        let docs = vec![
            LoadedDocument {
                id: "a".into(),
                text: "robot étrange robot".into(),
                metadata: DocumentMetadata {
                    prompt_id: Some("p1".into()),
                    ..Default::default()
                },
            },
            LoadedDocument {
                id: "b".into(),
                text: "robot utile".into(),
                metadata: DocumentMetadata {
                    prompt_id: Some("p2".into()),
                    ..Default::default()
                },
            },
        ];
        let profile = profile_loaded_documents(Language::Fr, &docs, ProfileOptions::default());
        assert_eq!(profile.words["robot"].count, 3);
        assert_eq!(profile.words["robot"].document_frequency, 2);
        assert_eq!(profile.words["robot"].prompt_frequency, 2);
    }

    #[test]
    fn builds_per_model_profiles_and_discovery_ngrams() {
        let docs = vec![
            LoadedDocument {
                id: "a".into(),
                text: "Il est important de noter ce point.".into(),
                metadata: DocumentMetadata {
                    model_id: Some("claude-x".into()),
                    family: Some("claude".into()),
                    prompt_id: Some("p1".into()),
                    domains: vec!["culture".into()],
                },
            },
            LoadedDocument {
                id: "b".into(),
                text: "Il reste important de noter ce détail.".into(),
                metadata: DocumentMetadata {
                    model_id: Some("gpt-x".into()),
                    family: Some("gpt".into()),
                    prompt_id: Some("p2".into()),
                    domains: vec!["culture".into()],
                },
            },
        ];
        let profile = profile_loaded_documents(
            Language::Fr,
            &docs,
            ProfileOptions {
                recover_phrases: true,
            },
        );
        assert_eq!(profile.model_profiles.len(), 2);
        assert_eq!(profile.documents[0].prompt_id.as_deref(), Some("p1"));
        assert_eq!(
            profile.discovery_bigrams["important noter"].document_frequency,
            2
        );
        assert!(profile.phrase_variants["important noter"].contains_key("important de noter"));
    }

    #[test]
    fn literal_phrase_ngrams_keep_stopwords_and_adjacency() {
        let docs = vec![(
            "a".into(),
            "Une analyse à partir de données solides.".into(),
        )];
        let profile = profile_documents(Language::Fr, &docs);
        assert!(profile.trigrams.contains_key("partir de données"));
        assert!(!profile.trigrams.contains_key("à partir données"));
        assert!(
            profile
                .discovery_trigrams
                .contains_key("analyse partir données")
        );
    }
}
