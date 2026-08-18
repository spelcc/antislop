use crate::fingerprint::{Fingerprint, FingerprintEntry, FingerprintSource};
use crate::metrics::{LexicalMetrics, lexical_metrics};
use crate::tokenize::{Language, content_tokens, tokenize};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum PatternSignalClass {
    /// Topic-sensitive single-word evidence. Retained for diagnostics, never scored as slop.
    Lexical,
    /// Multi-word recurrence used as stylistic/slop evidence.
    Phrase,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PatternHit {
    pub pattern: String,
    pub n: u8,
    pub count: usize,
    pub ratio: Option<f64>,
    pub signal_class: PatternSignalClass,
    pub source: FingerprintSource,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub discovery_pattern: Option<String>,
    pub model_frequency: usize,
    pub family_frequency: usize,
    pub weighted_signal: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StructuralHit {
    pub rule: String,
    pub count: usize,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Analysis {
    pub schema_version: u32,
    pub language: Language,
    pub lexical: LexicalMetrics,
    pub fingerprint_hits: Vec<PatternHit>,
    pub fingerprint_signal_per_1000_tokens: f64,
    pub zero_baseline_occurrences: usize,
    pub structural_hits: Vec<StructuralHit>,
}

pub fn analyze_text(text: &str, language: Language, fingerprint: Option<&Fingerprint>) -> Analysis {
    let raw_tokens = tokenize(text);
    let content = content_tokens(text, language);
    let mut hits =
        fingerprint.map_or_else(Vec::new, |fp| score_fingerprint(&raw_tokens, &content, fp));
    hits.sort_by(|a, b| {
        b.weighted_signal
            .total_cmp(&a.weighted_signal)
            .then_with(|| a.pattern.cmp(&b.pattern))
    });
    let signal: f64 = hits.iter().map(|h| h.weighted_signal).sum();
    let zero_baseline_occurrences = hits
        .iter()
        .filter(|h| h.ratio.is_none() && h.weighted_signal > 0.0)
        .map(|h| h.count)
        .sum();
    let per_1000 = if raw_tokens.is_empty() {
        0.0
    } else {
        signal * 1000.0 / raw_tokens.len() as f64
    };

    Analysis {
        schema_version: 1,
        language,
        lexical: lexical_metrics(&raw_tokens),
        fingerprint_hits: hits,
        fingerprint_signal_per_1000_tokens: round4(per_1000),
        zero_baseline_occurrences,
        structural_hits: crate::structural::structural_rules(text, language),
    }
}

fn score_fingerprint(
    raw_tokens: &[String],
    content_tokens: &[String],
    fingerprint: &Fingerprint,
) -> Vec<PatternHit> {
    let mut counts_by_n = HashMap::<u8, HashMap<String, usize>>::new();
    let mut words = HashMap::new();
    for token in content_tokens {
        *words.entry(token.clone()).or_default() += 1;
    }
    counts_by_n.insert(1, words);

    let widths: std::collections::BTreeSet<u8> = fingerprint
        .bigrams
        .iter()
        .chain(&fingerprint.trigrams)
        .chain(&fingerprint.phrases)
        .map(|entry| entry.n)
        .filter(|n| *n >= 2)
        .collect();
    for n in widths {
        let width = n as usize;
        let mut counts = HashMap::new();
        if raw_tokens.len() >= width {
            for window in raw_tokens.windows(width) {
                *counts.entry(window.join(" ")).or_default() += 1;
            }
        }
        counts_by_n.insert(n, counts);
    }

    fingerprint
        .words
        .iter()
        .chain(&fingerprint.bigrams)
        .chain(&fingerprint.trigrams)
        .chain(&fingerprint.phrases)
        .filter_map(|entry| hit_for(entry, &counts_by_n, fingerprint.schema_version))
        .collect()
}

fn hit_for(
    entry: &FingerprintEntry,
    counts: &HashMap<u8, HashMap<String, usize>>,
    fingerprint_schema: u32,
) -> Option<PatternHit> {
    let lookup = if entry.n == 1 {
        entry.pattern.clone()
    } else {
        tokenize(&entry.pattern).join(" ")
    };
    let count = *counts.get(&entry.n)?.get(&lookup)?;
    let signal_class = if entry.n == 1 {
        PatternSignalClass::Lexical
    } else {
        PatternSignalClass::Phrase
    };
    let weight = if entry.n == 1 {
        if fingerprint_schema >= 4 && entry.source == FingerprintSource::Lexical {
            0.25
        } else {
            0.0
        }
    } else {
        match entry.source {
            FingerprintSource::Lexical => 0.25,
            FingerprintSource::RecoveredPhrase => {
                1.25 + ((entry.n.saturating_sub(3)) as f64 * 0.08).min(0.5)
            }
            FingerprintSource::LiteralNgram => match entry.n {
                2 => 0.5,
                3 => 1.0,
                _ => 1.0,
            },
        }
    };
    let baseline_signal = match entry.ratio {
        Some(ratio) if ratio > 1.0 => ratio.log2().min(6.0),
        Some(_) => 0.0,
        None => match entry.n {
            1 => 1.0,
            2 => 1.0,
            _ => 2.5,
        },
    };
    let ratio_signal = if fingerprint_schema >= 4 && entry.source == FingerprintSource::Lexical {
        let reference_signal = entry
            .reference_ratio
            .filter(|ratio| *ratio > 1.0)
            .map_or(0.0, |ratio| ratio.log2().min(6.0));
        (baseline_signal + reference_signal) / 2.0
    } else {
        baseline_signal
    };
    Some(PatternHit {
        pattern: entry.pattern.clone(),
        n: entry.n,
        count,
        ratio: entry.ratio,
        signal_class,
        source: entry.source,
        discovery_pattern: entry.discovery_pattern.clone(),
        model_frequency: entry.model_frequency,
        family_frequency: entry.family_frequency,
        weighted_signal: round4(count as f64 * weight * ratio_signal),
    })
}

fn round4(value: f64) -> f64 {
    (value * 10_000.0).round() / 10_000.0
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn legacy_fingerprint_unigrams_keep_zero_weight() {
        let fingerprint: Fingerprint = serde_json::from_str(
            r#"{
              "schema_version":3,
              "language":"fr",
              "min_documents":3,
              "words":[{"pattern":"api","n":1,"target_frequency":0.01,"baseline_frequency":0.0,"ratio":null,"target_document_frequency":10}],
              "bigrams":[],"trigrams":[]
            }"#,
        )
        .unwrap();
        let analysis = analyze_text("Une API centrale.", Language::Fr, Some(&fingerprint));
        assert_eq!(analysis.fingerprint_hits[0].weighted_signal, 0.0);
    }

    #[test]
    fn finds_french_false_contrast() {
        let analysis = analyze_text(
            "Ce n'est pas une mode, c'est un problème. Pas seulement un bug, mais une habitude.",
            Language::Fr,
            None,
        );
        assert_eq!(
            analysis
                .structural_hits
                .iter()
                .map(|h| h.count)
                .sum::<usize>(),
            2
        );
    }
}
