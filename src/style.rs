use crate::lint::split_sentences;
use crate::tokenize::{Language, tokenize};
use serde::{Deserialize, Serialize};
use std::collections::{BTreeMap, HashMap};

const PROFILE_SCHEMA_VERSION: u32 = 2;
const TOP_DEVIATIONS: usize = 15;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StyleMetricSummary {
    pub mean: f64,
    pub median: f64,
    pub mad: f64,
    pub p10: f64,
    pub p90: f64,
    pub nonzero_rate: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StyleProfile {
    pub schema_version: u32,
    pub language: Language,
    pub document_count: usize,
    pub token_count: usize,
    pub metrics: BTreeMap<String, StyleMetricSummary>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StyleDocumentMetrics {
    pub tokens: usize,
    pub sentences: usize,
    pub paragraphs: usize,
    pub metrics: BTreeMap<String, f64>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StyleDeviation {
    pub metric: String,
    pub group: String,
    pub value: f64,
    pub profile_median: f64,
    pub profile_p10: f64,
    pub profile_p90: f64,
    pub robust_z: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StyleComparison {
    pub schema_version: u32,
    pub language: Language,
    pub profile_documents: usize,
    pub document: StyleDocumentMetrics,
    /// Average group-wise robust distance. Lower means closer to the profile.
    /// This is not a probability or authorship score.
    pub overall_distance: f64,
    pub within_profile_band_ratio: f64,
    pub groups: BTreeMap<String, f64>,
    pub top_deviations: Vec<StyleDeviation>,
}

pub fn build_style_profile(
    language: Language,
    documents: &[(String, String)],
) -> Result<StyleProfile, String> {
    if documents.len() < 3 {
        return Err("style profile requires at least 3 documents".to_string());
    }

    let observations: Vec<_> = documents
        .iter()
        .map(|(_, text)| style_metrics(text, language))
        .collect();
    let token_count = observations.iter().map(|metrics| metrics.tokens).sum();
    let mut by_metric = BTreeMap::<String, Vec<f64>>::new();
    for observation in &observations {
        for (metric, value) in &observation.metrics {
            by_metric.entry(metric.clone()).or_default().push(*value);
        }
    }

    let metrics = by_metric
        .into_iter()
        .map(|(metric, values)| (metric, summarize(&values)))
        .collect();

    Ok(StyleProfile {
        schema_version: PROFILE_SCHEMA_VERSION,
        language,
        document_count: documents.len(),
        token_count,
        metrics,
    })
}

pub fn compare_style(text: &str, profile: &StyleProfile) -> StyleComparison {
    let document = style_metrics(text, profile.language);
    let mut deviations = Vec::new();
    let mut grouped = BTreeMap::<String, Vec<f64>>::new();
    let mut within = 0usize;
    let mut scored = 0usize;

    for (metric, summary) in &profile.metrics {
        if metric.contains("paragraph") {
            continue;
        }
        let value = *document.metrics.get(metric).unwrap_or(&0.0);
        if !metric_is_informative(metric, summary) {
            continue;
        }
        let scale = metric_scale(metric, summary);
        if scale <= f64::EPSILON {
            continue;
        }
        let robust_z = ((value - summary.median) / scale).clamp(-8.0, 8.0);
        let group = metric_group(metric).to_string();
        grouped
            .entry(group.clone())
            .or_default()
            .push(robust_z.abs());
        scored += 1;
        if robust_z.abs() <= 2.0 {
            within += 1;
        }
        deviations.push(StyleDeviation {
            metric: metric.clone(),
            group,
            value: round4(value),
            profile_median: summary.median,
            profile_p10: summary.p10,
            profile_p90: summary.p90,
            robust_z: round4(robust_z),
        });
    }

    let groups: BTreeMap<_, _> = grouped
        .into_iter()
        .map(|(group, values)| (group, round4(mean(&values))))
        .collect();
    let overall_distance = if groups.is_empty() {
        0.0
    } else {
        groups.values().sum::<f64>() / groups.len() as f64
    };
    deviations.sort_by(|a, b| {
        b.robust_z
            .abs()
            .total_cmp(&a.robust_z.abs())
            .then_with(|| a.metric.cmp(&b.metric))
    });
    deviations.truncate(TOP_DEVIATIONS);

    StyleComparison {
        schema_version: PROFILE_SCHEMA_VERSION,
        language: profile.language,
        profile_documents: profile.document_count,
        document,
        overall_distance: round4(overall_distance),
        within_profile_band_ratio: round4(if scored == 0 {
            0.0
        } else {
            within as f64 / scored as f64
        }),
        groups,
        top_deviations: deviations,
    }
}

pub fn style_metrics(text: &str, language: Language) -> StyleDocumentMetrics {
    let tokens = tokenize(text);
    let sentences = split_sentences(text);
    let paragraph_texts = paragraphs(text);
    let sentence_lengths: Vec<f64> = sentences
        .iter()
        .map(|sentence| tokenize(&sentence.text).len() as f64)
        .filter(|length| *length > 0.0)
        .collect();
    let mut metrics = BTreeMap::new();
    metrics.insert(
        "rhythm.mean_sentence_tokens".to_string(),
        mean(&sentence_lengths),
    );
    metrics.insert(
        "rhythm.median_sentence_tokens".to_string(),
        median(&sentence_lengths),
    );
    metrics.insert(
        "rhythm.short_sentence_rate".to_string(),
        fraction(&sentence_lengths, |length| length <= 8.0),
    );
    metrics.insert(
        "rhythm.long_sentence_rate".to_string(),
        fraction(&sentence_lengths, |length| length >= 30.0),
    );
    let token_denominator = tokens.len().max(1) as f64;
    for (name, count) in punctuation_counts(text) {
        metrics.insert(
            format!("punctuation.{name}_per_1000"),
            count as f64 * 1000.0 / token_denominator,
        );
    }

    add_pronoun_metrics(&mut metrics, &tokens, language, token_denominator);
    add_function_word_metrics(&mut metrics, &tokens, language, token_denominator);
    add_starter_metrics(&mut metrics, &sentences, language);

    for value in metrics.values_mut() {
        *value = round4(*value);
    }

    StyleDocumentMetrics {
        tokens: tokens.len(),
        sentences: sentence_lengths.len(),
        paragraphs: paragraph_texts.len(),
        metrics,
    }
}

fn summarize(values: &[f64]) -> StyleMetricSummary {
    let mean_value = mean(values);
    let median_value = median(values);
    let deviations: Vec<_> = values
        .iter()
        .map(|value| (value - median_value).abs())
        .collect();
    StyleMetricSummary {
        mean: round4(mean_value),
        median: round4(median_value),
        mad: round4(median(&deviations)),
        p10: round4(percentile(values, 0.10)),
        p90: round4(percentile(values, 0.90)),
        nonzero_rate: round4(
            values
                .iter()
                .filter(|value| value.abs() > f64::EPSILON)
                .count() as f64
                / values.len().max(1) as f64,
        ),
    }
}

fn metric_is_informative(metric: &str, summary: &StyleMetricSummary) -> bool {
    match metric_group(metric) {
        "function_word" | "starter" => summary.nonzero_rate >= 0.10,
        _ => true,
    }
}

fn metric_scale(metric: &str, summary: &StyleMetricSummary) -> f64 {
    let robust_sigma = summary.mad * 1.4826;
    let percentile_sigma = (summary.p90 - summary.p10).abs() / 2.563;
    robust_sigma.max(percentile_sigma).max(metric_floor(metric))
}

fn metric_floor(metric: &str) -> f64 {
    if metric.contains("_per_1000") {
        0.5
    } else if metric.contains("_per_100_sentences") {
        1.0
    } else if metric.ends_with("_rate") {
        0.03
    } else if metric.contains("paragraph_tokens") {
        2.0
    } else if metric.contains("sentence_tokens") {
        1.0
    } else {
        0.1
    }
}

fn metric_group(metric: &str) -> &str {
    metric.split('.').next().unwrap_or("other")
}

fn punctuation_counts(text: &str) -> BTreeMap<&'static str, usize> {
    let mut counts = BTreeMap::new();
    counts.insert("comma", text.matches(',').count());
    counts.insert("semicolon", text.matches(';').count());
    counts.insert("colon", text.matches(':').count());
    counts.insert("question", text.matches('?').count());
    counts.insert("exclamation", text.matches('!').count());
    counts.insert(
        "parenthesis",
        text.matches('(').count() + text.matches(')').count(),
    );
    counts.insert(
        "dash",
        text.chars().filter(|ch| matches!(ch, '—' | '–')).count(),
    );
    counts.insert(
        "ellipsis",
        text.matches('…').count() + text.matches("...").count(),
    );
    counts
}

fn add_pronoun_metrics(
    metrics: &mut BTreeMap<String, f64>,
    tokens: &[String],
    language: Language,
    denominator: f64,
) {
    let groups: &[(&str, &[&str])] = match language {
        Language::Fr => &[
            ("first_singular", &["je", "me", "moi", "mon", "ma", "mes"]),
            ("first_plural", &["nous", "notre", "nos"]),
            ("on", &["on"]),
            (
                "second_person",
                &[
                    "tu", "te", "toi", "ton", "ta", "tes", "vous", "votre", "vos",
                ],
            ),
        ],
        Language::En => &[
            ("first_singular", &["i", "me", "my", "mine"]),
            ("first_plural", &["we", "us", "our", "ours"]),
            ("on", &[]),
            ("second_person", &["you", "your", "yours"]),
        ],
    };

    for (name, words) in groups {
        let count = tokens
            .iter()
            .filter(|token| {
                words.contains(&token.as_str())
                    || (*name == "first_singular"
                        && language == Language::Fr
                        && token.starts_with("j'"))
            })
            .count();
        metrics.insert(
            format!("pronoun.{name}_per_1000"),
            count as f64 * 1000.0 / denominator,
        );
    }
}

fn add_function_word_metrics(
    metrics: &mut BTreeMap<String, f64>,
    tokens: &[String],
    language: Language,
    denominator: f64,
) {
    let counts = token_counts(tokens);
    for word in function_words(language) {
        let count = *counts.get(*word).unwrap_or(&0);
        metrics.insert(
            format!("function_word.{word}_per_1000"),
            count as f64 * 1000.0 / denominator,
        );
    }
}

fn add_starter_metrics(
    metrics: &mut BTreeMap<String, f64>,
    sentences: &[crate::lint::SentenceSpan],
    language: Language,
) {
    let starters = starter_words(language);
    let mut counts = HashMap::<&str, usize>::new();
    for sentence in sentences {
        let first = tokenize(&sentence.text).into_iter().next();
        if let Some(first) = first {
            for starter in starters {
                if first == *starter {
                    *counts.entry(starter).or_default() += 1;
                }
            }
        }
    }
    let denominator = sentences.len().max(1) as f64;
    for starter in starters {
        metrics.insert(
            format!("starter.{starter}_per_100_sentences"),
            *counts.get(starter).unwrap_or(&0) as f64 * 100.0 / denominator,
        );
    }
}

fn function_words(language: Language) -> &'static [&'static str] {
    match language {
        Language::Fr => &[
            "alors", "aussi", "avec", "car", "ce", "cela", "comme", "dans", "de", "donc", "encore",
            "en", "et", "mais", "même", "ou", "par", "parce", "pas", "pour", "puis", "quand",
            "que", "qui", "sans", "si", "sur", "un", "une", "ça",
        ],
        Language::En => &[
            "also", "and", "as", "because", "but", "by", "for", "from", "however", "if", "in",
            "not", "of", "on", "or", "so", "still", "that", "then", "this", "though", "to", "when",
            "which", "with", "without", "yet",
        ],
    }
}

fn starter_words(language: Language) -> &'static [&'static str] {
    match language {
        Language::Fr => &[
            "alors", "aussi", "car", "comme", "donc", "en", "mais", "même", "pour", "puis",
            "quand", "sans", "si",
        ],
        Language::En => &[
            "also", "and", "as", "because", "but", "for", "however", "if", "so", "still", "then",
            "when", "with", "without", "yet",
        ],
    }
}

fn token_counts(tokens: &[String]) -> HashMap<&str, usize> {
    let mut counts = HashMap::new();
    for token in tokens {
        *counts.entry(token.as_str()).or_default() += 1;
    }
    counts
}

fn paragraphs(text: &str) -> Vec<String> {
    let mut result = Vec::new();
    let mut current = Vec::new();
    for line in text.lines() {
        if line.trim().is_empty() {
            if !current.is_empty() {
                result.push(current.join(" "));
                current.clear();
            }
        } else {
            current.push(line.trim());
        }
    }
    if !current.is_empty() {
        result.push(current.join(" "));
    }
    result
}

fn fraction(values: &[f64], predicate: impl Fn(f64) -> bool) -> f64 {
    if values.is_empty() {
        return 0.0;
    }
    values.iter().filter(|value| predicate(**value)).count() as f64 / values.len() as f64
}

fn mean(values: &[f64]) -> f64 {
    if values.is_empty() {
        0.0
    } else {
        values.iter().sum::<f64>() / values.len() as f64
    }
}

fn median(values: &[f64]) -> f64 {
    percentile(values, 0.5)
}

fn percentile(values: &[f64], percentile: f64) -> f64 {
    if values.is_empty() {
        return 0.0;
    }
    let mut sorted = values.to_vec();
    sorted.sort_by(f64::total_cmp);
    if sorted.len() == 1 {
        return sorted[0];
    }
    let position = percentile.clamp(0.0, 1.0) * (sorted.len() - 1) as f64;
    let lower = position.floor() as usize;
    let upper = position.ceil() as usize;
    if lower == upper {
        sorted[lower]
    } else {
        let weight = position - lower as f64;
        sorted[lower] * (1.0 - weight) + sorted[upper] * weight
    }
}

fn round4(value: f64) -> f64 {
    (value * 10_000.0).round() / 10_000.0
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn captures_rhythm_and_function_words() {
        let metrics = style_metrics("Mais je teste. Puis je recommence ?", Language::Fr);
        assert_eq!(metrics.sentences, 2);
        assert!(metrics.metrics["function_word.mais_per_1000"] > 0.0);
        assert!(metrics.metrics["starter.mais_per_100_sentences"] > 0.0);
        assert!(metrics.metrics["pronoun.first_singular_per_1000"] > 0.0);
        assert!(
            !metrics
                .metrics
                .keys()
                .any(|metric| metric.contains("paragraph"))
        );
    }

    #[test]
    fn matching_text_is_closer_than_alien_text() {
        let docs: Vec<_> = (0..4)
            .map(|index| {
                (
                    index.to_string(),
                    "Mais je teste. Puis je regarde. On recommence ?".to_string(),
                )
            })
            .collect();
        let profile = build_style_profile(Language::Fr, &docs).unwrap();
        let close = compare_style("Mais je regarde. Puis je teste. On recommence ?", &profile);
        let far = compare_style(
            "Toutefois cette proposition excessivement institutionnelle implique une transformation méthodologique considérable ; elle demeure néanmoins complexe.",
            &profile,
        );
        assert!(close.overall_distance < far.overall_distance);
    }
}
