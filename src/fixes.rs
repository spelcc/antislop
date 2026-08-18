use crate::analyze::analyze_text;
use crate::lint::split_sentences;
use crate::nearest::{CandidateClass, NearestCandidate, NearestMatch};
use crate::style::{StyleProfile, compare_style};
use crate::tokenize::{Language, tokenize};
use serde::{Deserialize, Serialize};
use std::collections::{BTreeMap, BTreeSet};

const LLM_REFERENCE_COUNT: usize = 5;
const HUMAN_REFERENCE_COUNT: usize = 5;
const MAX_FIXES: usize = 12;
const MAX_PATTERNS_PER_FIX: usize = 6;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ClassificationFixPattern {
    pub pattern: String,
    pub n: u8,
    pub occurrences: usize,
    pub aggregate_signal: f64,
    pub candidate_count: usize,
    pub confidence: String,
    pub candidates: Vec<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ClassificationFix {
    pub priority: usize,
    pub sentence_index: usize,
    pub start_line: usize,
    pub end_line: usize,
    pub text: String,
    pub llm_signal: f64,
    pub human_signal: f64,
    pub llm_signal_margin: f64,
    pub patterns: Vec<ClassificationFixPattern>,
    pub instruction: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ClassificationStyleFix {
    pub priority: usize,
    pub start_line: usize,
    pub end_line: usize,
    pub text: String,
    pub metric: String,
    pub document_value: f64,
    pub human_median: f64,
    pub instruction: String,
}

#[derive(Debug, Default)]
struct PatternAccumulator {
    n: u8,
    occurrences: usize,
    aggregate_signal: f64,
    candidates: BTreeSet<String>,
}

pub fn build_classification_fixes(
    text: &str,
    language: Language,
    nearest: &[NearestMatch],
    candidates: &BTreeMap<String, NearestCandidate>,
) -> Vec<ClassificationFix> {
    let llm_labels = top_signal_labels(nearest, CandidateClass::Llm, LLM_REFERENCE_COUNT);
    let human_labels = top_signal_labels(nearest, CandidateClass::Human, HUMAN_REFERENCE_COUNT);
    if llm_labels.is_empty() {
        return Vec::new();
    }

    let mut fixes = Vec::new();
    for sentence in split_sentences(text) {
        let mut patterns = BTreeMap::<String, PatternAccumulator>::new();
        let llm_signal = mean_candidate_signal(
            &sentence.text,
            language,
            &llm_labels,
            candidates,
            Some(&mut patterns),
        );
        if llm_signal <= 0.0 || patterns.is_empty() {
            continue;
        }
        let human_signal =
            mean_candidate_signal(&sentence.text, language, &human_labels, candidates, None);
        let margin = llm_signal - human_signal;
        if margin <= 0.0 {
            continue;
        }

        let mut pattern_rows = merge_overlapping_patterns(&sentence.text, patterns);
        pattern_rows.sort_by(|left, right| {
            right
                .aggregate_signal
                .total_cmp(&left.aggregate_signal)
                .then_with(|| right.candidate_count.cmp(&left.candidate_count))
                .then_with(|| right.n.cmp(&left.n))
                .then_with(|| left.pattern.cmp(&right.pattern))
        });
        pattern_rows.truncate(MAX_PATTERNS_PER_FIX);
        if pattern_rows.is_empty() {
            continue;
        }
        let instruction = instruction_for(&pattern_rows);

        fixes.push(ClassificationFix {
            priority: 0,
            sentence_index: sentence.index,
            start_line: sentence.start_line,
            end_line: sentence.end_line,
            text: sentence.text,
            llm_signal: round4(llm_signal),
            human_signal: round4(human_signal),
            llm_signal_margin: round4(margin),
            patterns: pattern_rows,
            instruction,
        });
    }

    fixes.sort_by(|left, right| {
        right
            .llm_signal_margin
            .total_cmp(&left.llm_signal_margin)
            .then_with(|| right.llm_signal.total_cmp(&left.llm_signal))
            .then_with(|| left.start_line.cmp(&right.start_line))
    });
    fixes.truncate(MAX_FIXES);
    for (index, fix) in fixes.iter_mut().enumerate() {
        fix.priority = index + 1;
    }
    fixes
}

fn top_signal_labels(nearest: &[NearestMatch], class: CandidateClass, limit: usize) -> Vec<String> {
    let mut rows: Vec<_> = nearest.iter().filter(|item| item.class == class).collect();
    rows.sort_by(|left, right| {
        left.document_signal_position
            .cmp(&right.document_signal_position)
            .then_with(|| left.label.cmp(&right.label))
    });
    rows.into_iter()
        .take(limit)
        .map(|item| item.label.clone())
        .collect()
}

fn mean_candidate_signal(
    text: &str,
    language: Language,
    labels: &[String],
    candidates: &BTreeMap<String, NearestCandidate>,
    mut patterns: Option<&mut BTreeMap<String, PatternAccumulator>>,
) -> f64 {
    if labels.is_empty() {
        return 0.0;
    }
    let mut total = 0.0;
    let mut measured = 0usize;
    for label in labels {
        let Some(candidate) = candidates.get(label) else {
            continue;
        };
        let analysis = analyze_text(text, language, Some(&candidate.fingerprint));
        let mut candidate_signal = 0.0;
        for hit in analysis
            .fingerprint_hits
            .into_iter()
            .filter(|hit| hit.n >= 2 && hit.weighted_signal > 0.0)
        {
            candidate_signal += hit.weighted_signal;
            if let Some(patterns) = patterns.as_deref_mut() {
                let entry = patterns.entry(hit.pattern.clone()).or_default();
                entry.n = hit.n;
                entry.occurrences = entry.occurrences.max(hit.count);
                entry.aggregate_signal += hit.weighted_signal;
                entry.candidates.insert(label.clone());
            }
        }
        total += candidate_signal;
        measured += 1;
    }
    if measured == 0 {
        0.0
    } else {
        total / measured as f64
    }
}

#[derive(Debug)]
struct PositionedPattern {
    start: usize,
    end: usize,
    occurrences: usize,
    aggregate_signal: f64,
    candidates: BTreeSet<String>,
    strong: bool,
}

fn merge_overlapping_patterns(
    sentence: &str,
    patterns: BTreeMap<String, PatternAccumulator>,
) -> Vec<ClassificationFixPattern> {
    let sentence_tokens = tokenize(sentence);
    let mut positioned = Vec::new();
    let mut unpositioned = Vec::new();
    for (pattern, value) in patterns {
        let pattern_tokens = tokenize(&pattern);
        let start = if pattern_tokens.is_empty() || pattern_tokens.len() > sentence_tokens.len() {
            None
        } else {
            sentence_tokens
                .windows(pattern_tokens.len())
                .position(|window| window == pattern_tokens.as_slice())
        };
        if let Some(start) = start {
            positioned.push(PositionedPattern {
                start,
                end: start + pattern_tokens.len(),
                occurrences: value.occurrences,
                aggregate_signal: value.aggregate_signal,
                strong: value.n >= 3 || value.candidates.len() >= 2,
                candidates: value.candidates,
            });
        } else if value.n >= 3 || value.candidates.len() >= 2 {
            let candidate_count = value.candidates.len();
            unpositioned.push(ClassificationFixPattern {
                pattern,
                n: value.n,
                occurrences: value.occurrences,
                aggregate_signal: round4(value.aggregate_signal),
                candidate_count,
                confidence: confidence_label(value.n, candidate_count).into(),
                candidates: value.candidates.into_iter().collect(),
            });
        }
    }
    positioned.sort_by_key(|item| (item.start, item.end));

    let mut merged: Vec<PositionedPattern> = Vec::new();
    for item in positioned {
        if let Some(last) = merged.last_mut()
            && item.start < last.end
        {
            last.end = last.end.max(item.end);
            last.occurrences = last.occurrences.max(item.occurrences);
            last.aggregate_signal += item.aggregate_signal;
            last.strong |= item.strong;
            last.candidates.extend(item.candidates);
            continue;
        }
        merged.push(item);
    }

    let mut output: Vec<_> = merged
        .into_iter()
        .filter(|item| item.strong)
        .map(|item| {
            let n = (item.end - item.start).min(u8::MAX as usize) as u8;
            let candidate_count = item.candidates.len();
            ClassificationFixPattern {
                pattern: sentence_tokens[item.start..item.end].join(" "),
                n,
                occurrences: item.occurrences,
                aggregate_signal: round4(item.aggregate_signal),
                candidate_count,
                confidence: confidence_label(n, candidate_count).into(),
                candidates: item.candidates.into_iter().collect(),
            }
        })
        .collect();
    output.extend(unpositioned);
    output
}

fn confidence_label(n: u8, candidate_count: usize) -> &'static str {
    if candidate_count >= 3 || (candidate_count >= 2 && n >= 3) {
        "high"
    } else {
        "medium"
    }
}

pub fn build_style_fixes(text: &str, profile: &StyleProfile) -> Vec<ClassificationStyleFix> {
    let comparison = compare_style(text, profile);
    let rhythm_problem = comparison.top_deviations.iter().find(|deviation| {
        (deviation.metric == "rhythm.short_sentence_rate" && deviation.robust_z >= 1.5)
            || (deviation.metric == "rhythm.mean_sentence_tokens" && deviation.robust_z <= -1.5)
            || (deviation.metric == "rhythm.long_sentence_rate" && deviation.robust_z <= -1.5)
    });
    let Some(deviation) = rhythm_problem else {
        return Vec::new();
    };

    let sentences = split_sentences(text);
    let mut by_line = BTreeMap::<usize, Vec<_>>::new();
    for sentence in &sentences {
        let tokens = tokenize(&sentence.text).len();
        if tokens <= 12 {
            by_line
                .entry(sentence.start_line)
                .or_default()
                .push(sentence);
        }
    }
    let mut fixes = Vec::new();
    for (line, rows) in by_line {
        if rows.len() < 2 {
            continue;
        }
        let text = rows
            .iter()
            .map(|sentence| sentence.text.trim())
            .collect::<Vec<_>>()
            .join(" ");
        fixes.push(ClassificationStyleFix {
            priority: 0,
            start_line: line,
            end_line: rows.iter().map(|sentence| sentence.end_line).max().unwrap_or(line),
            text,
            metric: deviation.metric.clone(),
            document_value: deviation.value,
            human_median: deviation.profile_median,
            instruction: format!(
                "This source line stacks {} short sentences while the nearest Human profile uses longer sentence structure overall. If the claims belong together, combine or subordinate them; do not pad the prose merely to raise the score.",
                rows.len()
            ),
        });
    }
    fixes.sort_by(|left, right| {
        right
            .text
            .len()
            .cmp(&left.text.len())
            .then_with(|| left.start_line.cmp(&right.start_line))
    });
    fixes.truncate(6);
    for (index, fix) in fixes.iter_mut().enumerate() {
        fix.priority = index + 1;
    }
    fixes
}

fn instruction_for(patterns: &[ClassificationFixPattern]) -> String {
    let names: Vec<_> = patterns
        .iter()
        .take(3)
        .map(|pattern| format!("“{}”", pattern.pattern))
        .collect();
    if names.is_empty() {
        return "Rewrite the sentence around its concrete claim. Change the sentence structure rather than swapping synonyms; preserve facts, names, numbers and citations.".into();
    }
    format!(
        "Rewrite the sentence around its concrete claim. Remove or structurally rework {} rather than swapping synonyms; preserve facts, names, numbers and citations. Do not replace necessary domain terminology just to chase the detector; if the phrase is technically required, leave it and address another hotspot.",
        names.join(", ")
    )
}

fn round4(value: f64) -> f64 {
    (value * 10_000.0).round() / 10_000.0
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::style::build_style_profile;

    #[test]
    fn weak_single_model_bigram_is_not_actionable() {
        let mut candidates = BTreeSet::new();
        candidates.insert("model-x".to_string());
        let mut patterns = BTreeMap::new();
        patterns.insert(
            "the project".to_string(),
            PatternAccumulator {
                n: 2,
                occurrences: 1,
                aggregate_signal: 2.0,
                candidates,
            },
        );
        assert!(merge_overlapping_patterns("The project ships.", patterns).is_empty());
    }

    #[test]
    fn style_fix_pinpoints_short_sentence_stack() {
        let human_docs = vec![
            (
                "h1".to_string(),
                "A careful editor links the first concrete observation to the second one because the causal relation matters to the reader.".to_string(),
            ),
            (
                "h2".to_string(),
                "The field report keeps related details together while still varying its sentence structure enough to avoid a sequence of clipped assertions.".to_string(),
            ),
            (
                "h3".to_string(),
                "When the evidence changes direction, the writer explains the transition inside a developed sentence rather than breaking every claim into a separate beat.".to_string(),
            ),
        ];
        let profile = build_style_profile(Language::En, &human_docs).unwrap();
        let target = "One short claim. Another short claim.\nA separate sentence remains here.";
        let fixes = build_style_fixes(target, &profile);
        assert!(!fixes.is_empty());
        assert_eq!(fixes[0].start_line, 1);
        assert!(fixes[0].text.contains("One short claim."));
        assert!(fixes[0].instruction.contains("combine or subordinate"));
    }
}
