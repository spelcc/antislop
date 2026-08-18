use crate::analyze::{Analysis, PatternHit, PatternSignalClass, StructuralHit, analyze_text};
use crate::fingerprint::Fingerprint;
use crate::tokenize::Language;
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct SentenceSpan {
    pub index: usize,
    pub start_byte: usize,
    pub end_byte: usize,
    pub start_line: usize,
    pub end_line: usize,
    pub text: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SentenceFinding {
    pub sentence_index: usize,
    pub start_byte: usize,
    pub end_byte: usize,
    pub start_line: usize,
    pub end_line: usize,
    pub text: String,
    pub token_count: usize,
    pub fingerprint_signal: f64,
    pub fingerprint_signal_per_1000_tokens: f64,
    pub zero_baseline_occurrences: usize,
    pub fingerprint_hits: Vec<PatternHit>,
    pub structural_hits: Vec<StructuralHit>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LintReport {
    pub schema_version: u32,
    pub language: Language,
    pub sentence_count: usize,
    pub flagged_sentence_count: usize,
    pub document: Analysis,
    pub findings: Vec<SentenceFinding>,
}

#[derive(Debug, Clone, Default)]
pub struct LintThresholds {
    pub max_document_signal: Option<f64>,
    pub max_sentence_signal: Option<f64>,
    pub max_structural_hits: Option<usize>,
    pub max_flagged_sentences: Option<usize>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct LintViolation {
    pub metric: String,
    pub actual: f64,
    pub limit: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LintOutcome {
    #[serde(flatten)]
    pub report: LintReport,
    pub passed: bool,
    pub violations: Vec<LintViolation>,
}

pub fn lint_text(
    text: &str,
    language: Language,
    fingerprint: Option<&Fingerprint>,
    thresholds: &LintThresholds,
) -> LintOutcome {
    let document = analyze_text(text, language, fingerprint);
    let sentences = split_sentences(text);
    let findings: Vec<_> = sentences
        .iter()
        .filter_map(|sentence| finding_for(sentence, language, fingerprint))
        .collect();

    let report = LintReport {
        schema_version: 1,
        language,
        sentence_count: sentences.len(),
        flagged_sentence_count: findings.len(),
        document,
        findings,
    };
    let violations = evaluate_thresholds(&report, thresholds);

    LintOutcome {
        passed: violations.is_empty(),
        report,
        violations,
    }
}

pub fn split_sentences(text: &str) -> Vec<SentenceSpan> {
    let mut ranges = Vec::new();
    let mut start = 0;
    let mut chars = text.char_indices().peekable();

    while let Some((index, ch)) = chars.next() {
        let end = index + ch.len_utf8();
        let punctuation_boundary =
            matches!(ch, '.' | '!' | '?') && is_sentence_terminator(text, index, end, ch);
        let paragraph_boundary = ch == '\n' && chars.peek().is_some_and(|(_, next)| *next == '\n');

        if punctuation_boundary || paragraph_boundary {
            push_trimmed_range(text, start, end, &mut ranges);
            start = end;
        }
    }
    push_trimmed_range(text, start, text.len(), &mut ranges);

    ranges
        .into_iter()
        .enumerate()
        .map(|(offset, (start_byte, end_byte))| SentenceSpan {
            index: offset + 1,
            start_byte,
            end_byte,
            start_line: line_at(text, start_byte),
            end_line: line_at(text, end_byte),
            text: text[start_byte..end_byte].to_string(),
        })
        .collect()
}

fn finding_for(
    sentence: &SentenceSpan,
    language: Language,
    fingerprint: Option<&Fingerprint>,
) -> Option<SentenceFinding> {
    let analysis = analyze_text(&sentence.text, language, fingerprint);
    // A single content word is overwhelmingly topic-sensitive. Keep unigram hits in the
    // document analysis for diagnostics, but never create a sentence warning from them.
    let fingerprint_hits: Vec<_> = analysis
        .fingerprint_hits
        .into_iter()
        .filter(|hit| {
            hit.signal_class == PatternSignalClass::Phrase
                && hit.n >= 3
                && hit.weighted_signal >= 2.5
        })
        .collect();
    let fingerprint_signal = round4(fingerprint_hits.iter().map(|hit| hit.weighted_signal).sum());

    if fingerprint_hits.is_empty() && analysis.structural_hits.is_empty() {
        return None;
    }

    Some(SentenceFinding {
        sentence_index: sentence.index,
        start_byte: sentence.start_byte,
        end_byte: sentence.end_byte,
        start_line: sentence.start_line,
        end_line: sentence.end_line,
        text: sentence.text.clone(),
        token_count: analysis.lexical.tokens,
        fingerprint_signal,
        fingerprint_signal_per_1000_tokens: analysis.fingerprint_signal_per_1000_tokens,
        zero_baseline_occurrences: analysis.zero_baseline_occurrences,
        fingerprint_hits,
        structural_hits: analysis.structural_hits,
    })
}

fn evaluate_thresholds(report: &LintReport, thresholds: &LintThresholds) -> Vec<LintViolation> {
    let mut violations = Vec::new();
    add_violation(
        &mut violations,
        "document_signal_per_1000_tokens",
        report.document.fingerprint_signal_per_1000_tokens,
        thresholds.max_document_signal,
    );
    let max_sentence_signal = report
        .findings
        .iter()
        .map(|finding| finding.fingerprint_signal)
        .fold(0.0, f64::max);
    add_violation(
        &mut violations,
        "max_sentence_signal",
        max_sentence_signal,
        thresholds.max_sentence_signal,
    );
    let structural_hits = report
        .findings
        .iter()
        .flat_map(|finding| &finding.structural_hits)
        .map(|hit| hit.count)
        .sum::<usize>() as f64;
    add_violation(
        &mut violations,
        "structural_hits",
        structural_hits,
        thresholds.max_structural_hits.map(|value| value as f64),
    );
    add_violation(
        &mut violations,
        "flagged_sentences",
        report.flagged_sentence_count as f64,
        thresholds.max_flagged_sentences.map(|value| value as f64),
    );
    violations
}

fn add_violation(
    violations: &mut Vec<LintViolation>,
    metric: &str,
    actual: f64,
    limit: Option<f64>,
) {
    if let Some(limit) = limit
        && actual > limit
    {
        violations.push(LintViolation {
            metric: metric.to_string(),
            actual: round4(actual),
            limit: round4(limit),
        });
    }
}

fn is_sentence_terminator(text: &str, start: usize, end: usize, ch: char) -> bool {
    if ch != '.' {
        return true;
    }

    let previous = text[..start].chars().next_back();
    let next = text[end..].chars().next();
    if previous.is_some_and(|value| value.is_ascii_digit())
        && next.is_some_and(|value| value.is_ascii_digit())
    {
        return false;
    }

    let before = text[..start]
        .split_whitespace()
        .next_back()
        .unwrap_or_default()
        .trim_matches(|value: char| !value.is_alphanumeric())
        .to_lowercase();
    !matches!(
        before.as_str(),
        "m" | "mme" | "mr" | "dr" | "pr" | "vs" | "fig" | "no" | "e.g" | "i.e"
    )
}

fn push_trimmed_range(text: &str, start: usize, end: usize, ranges: &mut Vec<(usize, usize)>) {
    let slice = &text[start..end];
    let leading = slice.len() - slice.trim_start().len();
    let trailing = slice.len() - slice.trim_end().len();
    let trimmed_start = start + leading;
    let trimmed_end = end.saturating_sub(trailing);
    if trimmed_start < trimmed_end {
        ranges.push((trimmed_start, trimmed_end));
    }
}

fn line_at(text: &str, byte: usize) -> usize {
    text[..byte.min(text.len())]
        .bytes()
        .filter(|value| *value == b'\n')
        .count()
        + 1
}

fn round4(value: f64) -> f64 {
    (value * 10_000.0).round() / 10_000.0
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn sentence_spans_keep_original_lines() {
        let spans = split_sentences("Une phrase.\nDeuxième phrase !\n\nTroisième paragraphe.");
        assert_eq!(spans.len(), 3);
        assert_eq!(spans[1].start_line, 2);
        assert_eq!(spans[2].start_line, 4);
    }

    #[test]
    fn decimal_does_not_split_sentence() {
        let spans = split_sentences("La version 3.8 existe. Puis elle change.");
        assert_eq!(spans.len(), 2);
    }

    #[test]
    fn unicode_punctuation_keeps_valid_byte_boundaries() {
        let spans = split_sentences("Il répond « vraiment ». Puis il écrit “encore”.");
        assert_eq!(spans.len(), 2);
        assert_eq!(spans[0].text, "Il répond « vraiment ».");
        assert_eq!(spans[1].start_line, 1);
    }

    #[test]
    fn unigram_only_fingerprint_does_not_flag_a_sentence() {
        let fingerprint = Fingerprint {
            schema_version: 4,
            language: Language::Fr,
            min_documents: 3,
            min_models: 0,
            target_model_count: 0,
            target_label: None,
            target_model_id: None,
            target_family: None,
            lexical_reference: None,
            guard_profile_documents: None,
            min_guard_ratio: None,
            words: vec![crate::fingerprint::FingerprintEntry {
                pattern: "api".to_string(),
                n: 1,
                target_frequency: 0.01,
                baseline_frequency: 0.0,
                ratio: None,
                target_document_frequency: 10,
                target_prompt_frequency: 10,
                model_frequency: 0,
                family_frequency: 0,
                model_ids: vec![],
                families: vec![],
                guard_frequency: None,
                guard_ratio: None,
                reference_frequency: None,
                reference_ratio: None,
                discovery_pattern: None,
                source: crate::fingerprint::FingerprintSource::Lexical,
            }],
            bigrams: vec![],
            trigrams: vec![],
            phrases: vec![],
        };
        let outcome = lint_text(
            "Cette API ouvre le marché.",
            Language::Fr,
            Some(&fingerprint),
            &LintThresholds::default(),
        );
        assert_eq!(outcome.report.flagged_sentence_count, 0);
        assert_eq!(outcome.report.document.fingerprint_hits.len(), 1);
        assert_eq!(
            outcome.report.document.fingerprint_hits[0].signal_class,
            PatternSignalClass::Lexical
        );
        assert!(outcome.report.document.fingerprint_hits[0].weighted_signal > 0.0);
        assert!(outcome.report.document.fingerprint_hits[0].weighted_signal < 0.5);
    }

    #[test]
    fn absent_bigram_is_document_evidence_but_not_a_sentence_warning() {
        let fingerprint = Fingerprint {
            schema_version: 2,
            language: Language::Fr,
            min_documents: 3,
            min_models: 0,
            target_model_count: 0,
            target_label: None,
            target_model_id: None,
            target_family: None,
            lexical_reference: None,
            guard_profile_documents: None,
            min_guard_ratio: None,
            words: vec![],
            bigrams: vec![crate::fingerprint::FingerprintEntry {
                pattern: "exemples concrets".to_string(),
                n: 2,
                target_frequency: 0.002,
                baseline_frequency: 0.0,
                ratio: None,
                target_document_frequency: 12,
                target_prompt_frequency: 12,
                model_frequency: 0,
                family_frequency: 0,
                model_ids: vec![],
                families: vec![],
                guard_frequency: None,
                guard_ratio: None,
                reference_frequency: None,
                reference_ratio: None,
                discovery_pattern: None,
                source: crate::fingerprint::FingerprintSource::LiteralNgram,
            }],
            trigrams: vec![],
            phrases: vec![],
        };
        let outcome = lint_text(
            "Voici des exemples concrets.",
            Language::Fr,
            Some(&fingerprint),
            &LintThresholds::default(),
        );
        assert_eq!(outcome.report.flagged_sentence_count, 0);
        assert_eq!(
            outcome.report.document.fingerprint_hits[0].weighted_signal,
            0.5
        );
    }

    #[test]
    fn absent_trigram_is_finite_and_can_flag_a_sentence() {
        let fingerprint = Fingerprint {
            schema_version: 2,
            language: Language::Fr,
            min_documents: 3,
            min_models: 0,
            target_model_count: 0,
            target_label: None,
            target_model_id: None,
            target_family: None,
            lexical_reference: None,
            guard_profile_documents: None,
            min_guard_ratio: None,
            words: vec![],
            bigrams: vec![],
            trigrams: vec![crate::fingerprint::FingerprintEntry {
                pattern: "bien sûr voici".to_string(),
                n: 3,
                target_frequency: 0.001,
                baseline_frequency: 0.0,
                ratio: None,
                target_document_frequency: 8,
                target_prompt_frequency: 8,
                model_frequency: 0,
                family_frequency: 0,
                model_ids: vec![],
                families: vec![],
                guard_frequency: None,
                guard_ratio: None,
                reference_frequency: None,
                reference_ratio: None,
                discovery_pattern: None,
                source: crate::fingerprint::FingerprintSource::LiteralNgram,
            }],
            phrases: vec![],
        };
        let outcome = lint_text(
            "Bien sûr voici trois exemples.",
            Language::Fr,
            Some(&fingerprint),
            &LintThresholds::default(),
        );
        assert_eq!(outcome.report.flagged_sentence_count, 1);
        let hit = &outcome.report.findings[0].fingerprint_hits[0];
        assert_eq!(hit.signal_class, PatternSignalClass::Phrase);
        assert_eq!(hit.weighted_signal, 2.5);
        assert!(hit.weighted_signal.is_finite());
    }
}
