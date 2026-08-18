use crate::fingerprint::{Fingerprint, FingerprintEntry};
use crate::metrics::{LexicalMetrics, lexical_metrics};
use crate::tokenize::{Language, content_tokens, tokenize};
use regex::Regex;
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
        structural_hits: structural_rules(text, language),
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
    for n in 2..=3u8 {
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
        .filter_map(|entry| hit_for(entry, &counts_by_n))
        .collect()
}

fn hit_for(
    entry: &FingerprintEntry,
    counts: &HashMap<u8, HashMap<String, usize>>,
) -> Option<PatternHit> {
    let count = *counts.get(&entry.n)?.get(&entry.pattern)?;
    let signal_class = if entry.n == 1 {
        PatternSignalClass::Lexical
    } else {
        PatternSignalClass::Phrase
    };
    let ngram_weight = match entry.n {
        1 => 0.0,
        2 => 0.65,
        3 => 1.0,
        _ => 0.0,
    };
    let ratio_signal = match entry.ratio {
        Some(ratio) if ratio > 1.0 => ratio.log2().min(6.0),
        Some(_) => 0.0,
        // Missing from a finite human baseline is evidence, but not infinite evidence.
        None => match entry.n {
            1 => 0.0,
            // A missing bigram is weak evidence: ordinary two-word combinations are sparse
            // even in multi-million-token baselines. Trigram absence is more informative.
            2 => 1.0,
            3 => 2.5,
            _ => 0.0,
        },
    };
    Some(PatternHit {
        pattern: entry.pattern.clone(),
        n: entry.n,
        count,
        ratio: entry.ratio,
        signal_class,
        weighted_signal: round4(count as f64 * ngram_weight * ratio_signal),
    })
}

fn structural_rules(text: &str, language: Language) -> Vec<StructuralHit> {
    let patterns: &[(&str, &str)] = match language {
        Language::En => &[
            ("not_x_but_y", r"(?i)\bnot\b[^.!?;:]{1,100}\bbut\b"),
            ("not_only_but", r"(?i)\bnot only\b[^.!?;:]{1,100}\bbut\b"),
            (
                "the_real_question",
                r"(?i)\bthe real (?:question|issue|story)\b",
            ),
        ],
        Language::Fr => &[
            (
                "not_x_but_y",
                r"(?i)\bce n['’]est pas\b[^.!?;:]{1,100}\b(?:c['’]est|mais)\b",
            ),
            (
                "not_only_but",
                r"(?i)\bpas seulement\b[^.!?;:]{1,100}\bmais\b",
            ),
            (
                "the_real_question",
                r"(?i)\bla vraie (?:question|histoire)\b|\ble vrai sujet\b",
            ),
        ],
    };

    patterns
        .iter()
        .filter_map(|(rule, pattern)| {
            let regex = Regex::new(pattern).expect("built-in structural regex must compile");
            let count = regex
                .find_iter(text)
                .filter(|matched| {
                    if *rule != "not_x_but_y" {
                        return true;
                    }
                    let value = matched.as_str().to_lowercase();
                    !value.contains("not only") && !value.contains("pas seulement")
                })
                .count();
            (count > 0).then(|| StructuralHit {
                rule: (*rule).to_string(),
                count,
            })
        })
        .collect()
}

fn round4(value: f64) -> f64 {
    (value * 10_000.0).round() / 10_000.0
}

#[cfg(test)]
mod tests {
    use super::*;

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
