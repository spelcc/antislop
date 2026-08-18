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
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub guard_frequency: Option<f64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub guard_ratio: Option<f64>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Fingerprint {
    pub schema_version: u32,
    pub language: crate::tokenize::Language,
    pub min_documents: usize,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub guard_profile_documents: Option<usize>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub min_guard_ratio: Option<f64>,
    pub words: Vec<FingerprintEntry>,
    pub bigrams: Vec<FingerprintEntry>,
    pub trigrams: Vec<FingerprintEntry>,
}

#[derive(Debug, Clone, Copy)]
pub struct FingerprintOptions {
    pub min_documents: usize,
    pub word_limit: usize,
    pub bigram_limit: usize,
    pub trigram_limit: usize,
    pub min_guard_ratio: f64,
}

impl Default for FingerprintOptions {
    fn default() -> Self {
        Self {
            min_documents: 3,
            word_limit: 120,
            bigram_limit: 40,
            trigram_limit: 40,
            min_guard_ratio: 2.0,
        }
    }
}

pub fn build_fingerprint(
    target: &CorpusProfile,
    baseline: &CorpusProfile,
    min_documents: usize,
    word_limit: usize,
    bigram_limit: usize,
    trigram_limit: usize,
) -> Fingerprint {
    build_fingerprint_with_guard(
        target,
        baseline,
        None,
        FingerprintOptions {
            min_documents,
            word_limit,
            bigram_limit,
            trigram_limit,
            min_guard_ratio: 2.0,
        },
    )
}

pub fn build_fingerprint_with_guard(
    target: &CorpusProfile,
    baseline: &CorpusProfile,
    guard: Option<&CorpusProfile>,
    options: FingerprintOptions,
) -> Fingerprint {
    assert_eq!(
        target.language, baseline.language,
        "target and baseline languages must match"
    );
    let FingerprintOptions {
        min_documents,
        word_limit,
        bigram_limit,
        trigram_limit,
        min_guard_ratio,
    } = options;
    if let Some(guard) = guard {
        assert_eq!(
            target.language, guard.language,
            "target and guard languages must match"
        );
        assert_eq!(
            target.schema_version, guard.schema_version,
            "target and guard profile schemas must match"
        );
    }
    Fingerprint {
        schema_version: 3,
        language: target.language,
        min_documents,
        guard_profile_documents: guard.map(|profile| profile.document_count),
        min_guard_ratio: guard.map(|_| min_guard_ratio),
        words: compare(
            &target.words,
            &baseline.words,
            guard.map(|profile| &profile.words),
            min_guard_ratio,
            1,
            min_documents,
            word_limit,
        ),
        bigrams: compare(
            &target.bigrams,
            &baseline.bigrams,
            guard.map(|profile| &profile.bigrams),
            min_guard_ratio,
            2,
            min_documents,
            bigram_limit,
        ),
        trigrams: compare(
            &target.trigrams,
            &baseline.trigrams,
            guard.map(|profile| &profile.trigrams),
            min_guard_ratio,
            3,
            min_documents,
            trigram_limit,
        ),
    }
}

fn compare(
    target: &BTreeMap<String, PatternStats>,
    baseline: &BTreeMap<String, PatternStats>,
    guard: Option<&BTreeMap<String, PatternStats>>,
    min_guard_ratio: f64,
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
            let guard_frequency = guard.map(|patterns| {
                patterns
                    .get(&pattern)
                    .map_or(0.0, |stats| stats.relative_frequency)
            });
            let guard_ratio = guard_frequency.and_then(|frequency| {
                (frequency > 0.0).then_some(target_stats.relative_frequency / frequency)
            });
            if guard_frequency.is_some() && guard_ratio.is_some_and(|value| value < min_guard_ratio)
            {
                return None;
            }
            Some(FingerprintEntry {
                pattern,
                n,
                target_frequency: target_stats.relative_frequency,
                baseline_frequency,
                ratio,
                target_document_frequency: target_stats.document_frequency,
                guard_frequency,
                guard_ratio,
            })
        })
        .collect();

    entries.sort_by(|a, b| {
        let score_a = ranking_score(a);
        let score_b = ranking_score(b);
        score_b
            .total_cmp(&score_a)
            .then_with(|| {
                b.target_document_frequency
                    .cmp(&a.target_document_frequency)
            })
            .then_with(|| b.target_frequency.total_cmp(&a.target_frequency))
            .then_with(|| a.pattern.cmp(&b.pattern))
    });
    entries.truncate(limit);
    entries
}

fn ranking_score(entry: &FingerprintEntry) -> f64 {
    let ratio_signal = match entry.ratio {
        Some(ratio) if ratio > 1.0 => ratio.log2().min(8.0),
        Some(_) => 0.0,
        // Zero observations in a finite baseline are useful but not infinite evidence.
        None => match entry.n {
            1 => 1.0,
            2 => 2.0,
            3 => 3.0,
            _ => 0.0,
        },
    };
    ratio_signal * (entry.target_document_frequency as f64).ln_1p()
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

    #[test]
    fn rare_zero_baseline_pattern_does_not_automatically_rank_first() {
        let mut target = BTreeMap::new();
        target.insert(
            "rare phrase".to_string(),
            PatternStats {
                count: 3,
                document_frequency: 3,
                relative_frequency: 0.001,
            },
        );
        target.insert(
            "recurrent phrase".to_string(),
            PatternStats {
                count: 20,
                document_frequency: 12,
                relative_frequency: 0.01,
            },
        );
        let mut baseline = BTreeMap::new();
        baseline.insert(
            "recurrent phrase".to_string(),
            PatternStats {
                count: 5,
                document_frequency: 4,
                relative_frequency: 0.002,
            },
        );
        let entries = compare(&target, &baseline, None, 1.0, 2, 3, 10);
        assert_eq!(entries[0].pattern, "recurrent phrase");
        assert_eq!(entries[1].pattern, "rare phrase");
        assert!(entries[1].ratio.is_none());
    }

    #[test]
    fn guard_filters_patterns_common_in_accepted_prose() {
        let target = profile_documents(
            Language::Fr,
            &[
                ("1".into(), "vous avez une idée claire".into()),
                ("2".into(), "vous avez une autre idée".into()),
                ("3".into(), "vous avez encore une idée".into()),
            ],
        );
        let baseline = profile_documents(
            Language::Fr,
            &[
                ("1".into(), "une idée claire existe".into()),
                ("2".into(), "une autre idée existe".into()),
                ("3".into(), "encore une idée existe".into()),
            ],
        );
        let guard = profile_documents(
            Language::Fr,
            &[
                ("1".into(), "vous avez une idée".into()),
                ("2".into(), "vous avez raison".into()),
                ("3".into(), "vous avez le choix".into()),
            ],
        );
        let fp = build_fingerprint_with_guard(
            &target,
            &baseline,
            Some(&guard),
            FingerprintOptions {
                min_documents: 3,
                word_limit: 20,
                bigram_limit: 20,
                trigram_limit: 20,
                min_guard_ratio: 2.0,
            },
        );
        assert!(!fp.bigrams.iter().any(|entry| entry.pattern == "vous avez"));
        assert_eq!(fp.guard_profile_documents, Some(3));
        assert_eq!(fp.min_guard_ratio, Some(2.0));
    }
}
