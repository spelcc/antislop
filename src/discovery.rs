use crate::tokenize::{Language, normalize, normalize_content_word, stopword_set};
use serde::{Deserialize, Serialize};
use std::collections::{BTreeMap, BTreeSet};

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ContentTokenSpan {
    pub token: String,
    pub start_byte: usize,
    pub end_byte: usize,
}

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct PhraseVariantStats {
    pub count: usize,
    pub document_frequency: usize,
    pub prompt_frequency: usize,
    pub model_frequency: usize,
    pub family_frequency: usize,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub model_ids: Vec<String>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub families: Vec<String>,
}

#[derive(Debug, Clone, Default)]
pub struct PhraseVariantAccumulator {
    pub count: usize,
    pub documents: BTreeSet<usize>,
    pub prompts: BTreeSet<String>,
    pub models: BTreeSet<String>,
    pub families: BTreeSet<String>,
}

pub fn content_token_spans(text: &str, language: Language) -> Vec<ContentTokenSpan> {
    let stops = stopword_set(language);
    let mut result = Vec::new();
    let mut start: Option<usize> = None;
    let mut previous_was_letter = false;
    for (byte, ch) in text.char_indices() {
        let apostrophe = matches!(ch, '\'' | '’' | 'ʼ');
        let allowed = ch.is_alphabetic() || (apostrophe && previous_was_letter);
        if allowed {
            start.get_or_insert(byte);
            previous_was_letter = ch.is_alphabetic();
        } else {
            if let Some(token_start) = start.take() {
                push_content_span(text, token_start, byte, language, &stops, &mut result);
            }
            previous_was_letter = false;
        }
    }
    if let Some(token_start) = start {
        push_content_span(text, token_start, text.len(), language, &stops, &mut result);
    }
    result
}

fn push_content_span(
    text: &str,
    start: usize,
    end: usize,
    language: Language,
    stops: &std::collections::HashSet<&'static str>,
    output: &mut Vec<ContentTokenSpan>,
) {
    let raw = &text[start..end];
    let normalized = normalize_content_word(normalize(raw), language);
    if normalized.is_empty() || stops.contains(normalized.as_str()) {
        return;
    }
    output.push(ContentTokenSpan {
        token: normalized,
        start_byte: start,
        end_byte: end,
    });
}

pub fn discovery_occurrences(
    text: &str,
    language: Language,
    n: usize,
) -> Vec<(String, String, usize)> {
    let spans = content_token_spans(text, language);
    if spans.len() < n {
        return Vec::new();
    }
    spans
        .windows(n)
        .filter_map(|window| {
            let start = window.first()?.start_byte;
            let end = window.last()?.end_byte;
            let exact = normalize_exact_phrase(&text[start..end]);
            if exact.is_empty() || has_sentence_end_in_middle(&exact) {
                return None;
            }
            let raw_token_count = crate::tokenize::tokenize(&exact).len();
            Some((
                window
                    .iter()
                    .map(|span| span.token.as_str())
                    .collect::<Vec<_>>()
                    .join(" "),
                exact,
                raw_token_count,
            ))
        })
        .collect()
}

pub fn collect_phrase_variants(
    text: &str,
    language: Language,
    doc_index: usize,
    prompt_id: Option<&str>,
    model_id: Option<&str>,
    family: Option<&str>,
    output: &mut BTreeMap<String, BTreeMap<String, PhraseVariantAccumulator>>,
) {
    for n in [2usize, 3usize] {
        for (skeleton, exact, _) in discovery_occurrences(text, language, n) {
            let entry = output
                .entry(skeleton)
                .or_default()
                .entry(exact)
                .or_default();
            entry.count += 1;
            entry.documents.insert(doc_index);
            if let Some(prompt) = prompt_id {
                entry.prompts.insert(prompt.to_string());
            }
            if let Some(model) = model_id {
                entry.models.insert(model.to_string());
            }
            if let Some(family) = family {
                entry.families.insert(family.to_string());
            }
        }
    }
}

pub fn finalize_phrase_variants(
    values: BTreeMap<String, BTreeMap<String, PhraseVariantAccumulator>>,
) -> BTreeMap<String, BTreeMap<String, PhraseVariantStats>> {
    values
        .into_iter()
        .map(|(skeleton, variants)| {
            let variants = variants
                .into_iter()
                .map(|(phrase, stats)| {
                    (
                        phrase,
                        PhraseVariantStats {
                            count: stats.count,
                            document_frequency: stats.documents.len(),
                            prompt_frequency: stats.prompts.len(),
                            model_frequency: stats.models.len(),
                            family_frequency: stats.families.len(),
                            model_ids: stats.models.into_iter().collect(),
                            families: stats.families.into_iter().collect(),
                        },
                    )
                })
                .collect();
            (skeleton, variants)
        })
        .collect()
}

fn normalize_exact_phrase(value: &str) -> String {
    value.split_whitespace().collect::<Vec<_>>().join(" ")
}

fn has_sentence_end_in_middle(value: &str) -> bool {
    let trimmed = value.trim();
    if trimmed.len() < 2 {
        return false;
    }
    trimmed[..trimmed.len() - trimmed.chars().next_back().unwrap().len_utf8()]
        .chars()
        .any(|ch| matches!(ch, '.' | '?' | '!'))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn cleaned_skeleton_recovers_exact_phrase_with_stopwords() {
        let found = discovery_occurrences(
            "Il est important de noter que ce point compte.",
            Language::Fr,
            2,
        );
        assert!(found.iter().any(|(skeleton, phrase, tokens)| {
            skeleton == "important noter" && phrase == "important de noter" && *tokens == 3
        }));
    }
}
