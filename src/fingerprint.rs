use crate::profile::{CorpusProfile, PatternStats};
use serde::{Deserialize, Serialize};
use std::collections::{BTreeMap, BTreeSet};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FingerprintEntry {
    pub pattern: String,
    pub n: u8,
    pub target_frequency: f64,
    pub baseline_frequency: f64,
    pub ratio: Option<f64>,
    pub target_document_frequency: usize,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Fingerprint {
    pub schema_version: u32,
    pub language: crate::tokenize::Language,
    pub min_documents: usize,
    pub words: Vec<FingerprintEntry>,
    pub bigrams: Vec<FingerprintEntry>,
    pub trigrams: Vec<FingerprintEntry>,
}

pub fn build_fingerprint(
    target: &CorpusProfile,
    baseline: &CorpusProfile,
    min_documents: usize,
    word_limit: usize,
    bigram_limit: usize,
    trigram_limit: usize,
) -> Fingerprint {
    assert_eq!(
        target.language, baseline.language,
        "target and baseline languages must match"
    );
    Fingerprint {
        schema_version: 1,
        language: target.language,
        min_documents,
        words: compare(&target.words, &baseline.words, 1, min_documents, word_limit),
        bigrams: compare(
            &target.bigrams,
            &baseline.bigrams,
            2,
            min_documents,
            bigram_limit,
        ),
        trigrams: compare(
            &target.trigrams,
            &baseline.trigrams,
            3,
            min_documents,
            trigram_limit,
        ),
    }
}

fn compare(
    target: &BTreeMap<String, PatternStats>,
    baseline: &BTreeMap<String, PatternStats>,
    n: u8,
    min_documents: usize,
    limit: usize,
) -> Vec<FingerprintEntry> {
    let keys: BTreeSet<_> = target.keys().chain(baseline.keys()).cloned().collect();
    let mut entries: Vec<_> = keys
        .into_iter()
        .filter_map(|pattern| {
            let target_stats = target.get(&pattern)?;
            if target_stats.document_frequency < min_documents {
                return None;
            }
            let baseline_frequency = baseline.get(&pattern).map_or(0.0, |s| s.relative_frequency);
            let ratio = if baseline_frequency == 0.0 {
                None
            } else {
                Some(target_stats.relative_frequency / baseline_frequency)
            };
            Some(FingerprintEntry {
                pattern,
                n,
                target_frequency: target_stats.relative_frequency,
                baseline_frequency,
                ratio,
                target_document_frequency: target_stats.document_frequency,
            })
        })
        .collect();

    entries.sort_by(|a, b| {
        let score_a = a.ratio.unwrap_or(f64::INFINITY);
        let score_b = b.ratio.unwrap_or(f64::INFINITY);
        score_b
            .total_cmp(&score_a)
            .then_with(|| b.target_frequency.total_cmp(&a.target_frequency))
            .then_with(|| a.pattern.cmp(&b.pattern))
    });
    entries.truncate(limit);
    entries
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{Language, profile_documents};

    #[test]
    fn ranks_overrepresented_pattern_first() {
        let target = profile_documents(
            Language::En,
            &[
                ("1".into(), "delve deeply delve".into()),
                ("2".into(), "delve carefully".into()),
                ("3".into(), "delve again".into()),
            ],
        );
        let baseline = profile_documents(
            Language::En,
            &[
                ("1".into(), "write clearly".into()),
                ("2".into(), "write simply".into()),
                ("3".into(), "delve once".into()),
            ],
        );
        let fp = build_fingerprint(&target, &baseline, 3, 10, 10, 10);
        assert_eq!(fp.words[0].pattern, "delve");
        assert!(fp.words[0].ratio.unwrap() > 1.0);
    }
}
