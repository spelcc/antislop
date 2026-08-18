use crate::metrics::{LexicalMetrics, lexical_metrics};
use crate::tokenize::{Language, content_tokens, tokenize};
use serde::{Deserialize, Serialize};
use std::collections::{BTreeMap, BTreeSet};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DocumentProfile {
    pub id: String,
    pub lexical: LexicalMetrics,
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
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PatternStats {
    pub count: usize,
    pub document_frequency: usize,
    pub relative_frequency: f64,
}

pub fn profile_documents(language: Language, documents: &[(String, String)]) -> CorpusProfile {
    let mut raw_word_counts = BTreeMap::<String, usize>::new();
    let mut bigram_counts = BTreeMap::<String, usize>::new();
    let mut trigram_counts = BTreeMap::<String, usize>::new();
    let mut word_docs = BTreeMap::<String, BTreeSet<usize>>::new();
    let mut bigram_docs = BTreeMap::<String, BTreeSet<usize>>::new();
    let mut trigram_docs = BTreeMap::<String, BTreeSet<usize>>::new();
    let mut document_profiles = Vec::new();
    let mut token_count = 0;
    let mut content_token_count = 0;

    for (doc_index, (id, text)) in documents.iter().enumerate() {
        let raw_tokens = tokenize(text);
        let content = content_tokens(text, language);
        token_count += raw_tokens.len();
        content_token_count += content.len();
        document_profiles.push(DocumentProfile {
            id: id.clone(),
            lexical: lexical_metrics(&raw_tokens),
        });
        // Single words are topic-sensitive, so profile only content words there.
        collect(&content, 1, doc_index, &mut raw_word_counts, &mut word_docs);
        // Multi-word evidence must preserve literal adjacency and function words.
        collect(
            &raw_tokens,
            2,
            doc_index,
            &mut bigram_counts,
            &mut bigram_docs,
        );
        collect(
            &raw_tokens,
            3,
            doc_index,
            &mut trigram_counts,
            &mut trigram_docs,
        );
    }

    CorpusProfile {
        schema_version: 2,
        language,
        document_count: documents.len(),
        token_count,
        content_token_count,
        documents: document_profiles,
        words: finalize(raw_word_counts, word_docs),
        bigrams: finalize(bigram_counts, bigram_docs),
        trigrams: finalize(trigram_counts, trigram_docs),
    }
}

fn collect(
    tokens: &[String],
    n: usize,
    doc_index: usize,
    counts: &mut BTreeMap<String, usize>,
    docs: &mut BTreeMap<String, BTreeSet<usize>>,
) {
    if tokens.len() < n {
        return;
    }
    for window in tokens.windows(n) {
        let pattern = window.join(" ");
        *counts.entry(pattern.clone()).or_default() += 1;
        docs.entry(pattern).or_default().insert(doc_index);
    }
}

fn finalize(
    counts: BTreeMap<String, usize>,
    docs: BTreeMap<String, BTreeSet<usize>>,
) -> BTreeMap<String, PatternStats> {
    let denominator: usize = counts.values().sum();
    counts
        .into_iter()
        .map(|(pattern, count)| {
            let relative_frequency = if denominator == 0 {
                0.0
            } else {
                count as f64 / denominator as f64
            };
            let stat = PatternStats {
                count,
                document_frequency: docs.get(&pattern).map_or(0, BTreeSet::len),
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
    fn tracks_document_frequency() {
        let docs = vec![
            ("a".into(), "robot étrange robot".into()),
            ("b".into(), "robot utile".into()),
        ];
        let profile = profile_documents(Language::Fr, &docs);
        assert_eq!(profile.words["robot"].count, 3);
        assert_eq!(profile.words["robot"].document_frequency, 2);
    }

    #[test]
    fn phrase_ngrams_keep_stopwords_and_literal_adjacency() {
        let docs = vec![(
            "a".into(),
            "Une analyse à partir de données solides.".into(),
        )];
        let profile = profile_documents(Language::Fr, &docs);
        assert!(profile.trigrams.contains_key("partir de données"));
        assert!(!profile.trigrams.contains_key("à partir données"));
    }
}
