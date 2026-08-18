use crate::discovery::PhraseVariantStats;
use crate::lexical::LexicalReference;
use crate::profile::{CorpusProfile, ModelProfile, PatternStats};
use crate::tokenize::Language;
use serde::{Deserialize, Serialize};
use std::collections::{BTreeMap, BTreeSet, HashMap};

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum FingerprintSource {
    Lexical,
    #[default]
    LiteralNgram,
    RecoveredPhrase,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FingerprintEntry {
    pub pattern: String,
    pub n: u8,
    pub target_frequency: f64,
    pub baseline_frequency: f64,
    pub ratio: Option<f64>,
    pub target_document_frequency: usize,
    #[serde(default)]
    pub target_prompt_frequency: usize,
    #[serde(default)]
    pub model_frequency: usize,
    #[serde(default)]
    pub family_frequency: usize,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub model_ids: Vec<String>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub families: Vec<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub guard_frequency: Option<f64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub guard_ratio: Option<f64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub reference_frequency: Option<f64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub reference_ratio: Option<f64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub discovery_pattern: Option<String>,
    #[serde(default)]
    pub source: FingerprintSource,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Fingerprint {
    pub schema_version: u32,
    pub language: Language,
    pub min_documents: usize,
    #[serde(default)]
    pub min_models: usize,
    #[serde(default)]
    pub target_model_count: usize,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub target_label: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub target_model_id: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub target_family: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub lexical_reference: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub guard_profile_documents: Option<usize>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub min_guard_ratio: Option<f64>,
    pub words: Vec<FingerprintEntry>,
    pub bigrams: Vec<FingerprintEntry>,
    pub trigrams: Vec<FingerprintEntry>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub phrases: Vec<FingerprintEntry>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FingerprintSuite {
    pub schema_version: u32,
    pub consensus: Fingerprint,
    pub models: BTreeMap<String, Fingerprint>,
}

#[derive(Debug, Clone, Copy)]
pub struct FingerprintOptions {
    pub min_documents: usize,
    pub min_model_documents: usize,
    pub min_models: usize,
    pub word_limit: usize,
    pub bigram_limit: usize,
    pub trigram_limit: usize,
    pub phrase_limit: usize,
    pub min_guard_ratio: f64,
    pub min_wordfreq_ratio: f64,
    pub use_wordfreq: bool,
}

impl Default for FingerprintOptions {
    fn default() -> Self {
        Self {
            min_documents: 3,
            min_model_documents: 2,
            min_models: 2,
            word_limit: 120,
            bigram_limit: 40,
            trigram_limit: 40,
            phrase_limit: 100,
            min_guard_ratio: 2.0,
            min_wordfreq_ratio: 3.0,
            use_wordfreq: true,
        }
    }
}

#[derive(Debug, Clone, Default)]
struct Support {
    models: BTreeSet<String>,
    families: BTreeSet<String>,
}

pub fn build_fingerprint(
    target: &CorpusProfile,
    baseline: &CorpusProfile,
    min_documents: usize,
    word_limit: usize,
    bigram_limit: usize,
    trigram_limit: usize,
) -> Fingerprint {
    let options = FingerprintOptions {
        min_documents,
        word_limit,
        bigram_limit,
        trigram_limit,
        min_models: 1,
        use_wordfreq: false,
        ..Default::default()
    };
    build_fingerprint_suite(target, baseline, None, options, None)
        .expect("legacy fingerprint construction should not fail")
        .consensus
}

pub fn build_fingerprint_with_guard(
    target: &CorpusProfile,
    baseline: &CorpusProfile,
    guard: Option<&CorpusProfile>,
    options: FingerprintOptions,
) -> Fingerprint {
    build_fingerprint_suite(target, baseline, guard, options, None)
        .expect("fingerprint construction should not fail")
        .consensus
}

pub fn build_fingerprint_suite(
    target: &CorpusProfile,
    baseline: &CorpusProfile,
    guard: Option<&CorpusProfile>,
    options: FingerprintOptions,
    label: Option<String>,
) -> Result<FingerprintSuite, String> {
    validate_profiles(target, baseline, guard)?;
    let lexical_reference = if options.use_wordfreq {
        LexicalReference::load(target.language)?
    } else {
        None
    };

    let models =
        build_model_fingerprints(target, baseline, guard, options, lexical_reference.as_ref());
    let effective_min_models = if models.is_empty() {
        1
    } else {
        options.min_models.max(1)
    };
    let word_support = support_from_models(&models, |fp| &fp.words, target);
    let bigram_support = support_from_models(&models, |fp| &fp.bigrams, target);
    let trigram_support = support_from_models(&models, |fp| &fp.trigrams, target);

    let discovery_support_bi = discovery_support(target, baseline, guard, options, 2);
    let discovery_support_tri = discovery_support(target, baseline, guard, options, 3);

    let consensus = Fingerprint {
        schema_version: 4,
        language: target.language,
        min_documents: options.min_documents,
        min_models: effective_min_models,
        target_model_count: target.model_profiles.len(),
        target_label: label,
        target_model_id: None,
        target_family: None,
        lexical_reference: lexical_reference
            .as_ref()
            .map(|reference| reference.source().to_string()),
        guard_profile_documents: guard.map(|profile| profile.document_count),
        min_guard_ratio: guard.map(|_| options.min_guard_ratio),
        words: compare(
            &target.words,
            &baseline.words,
            guard.map(|profile| &profile.words),
            1,
            options.min_documents,
            options.word_limit,
            options.min_guard_ratio,
            (!models.is_empty()).then_some(&word_support),
            effective_min_models,
            lexical_reference.as_ref(),
            options.min_wordfreq_ratio,
            FingerprintSource::Lexical,
        ),
        bigrams: compare(
            &target.bigrams,
            &baseline.bigrams,
            guard.map(|profile| &profile.bigrams),
            2,
            options.min_documents,
            options.bigram_limit,
            options.min_guard_ratio,
            (!models.is_empty()).then_some(&bigram_support),
            effective_min_models,
            None,
            0.0,
            FingerprintSource::LiteralNgram,
        ),
        trigrams: compare(
            &target.trigrams,
            &baseline.trigrams,
            guard.map(|profile| &profile.trigrams),
            3,
            options.min_documents,
            options.trigram_limit,
            options.min_guard_ratio,
            (!models.is_empty()).then_some(&trigram_support),
            effective_min_models,
            None,
            0.0,
            FingerprintSource::LiteralNgram,
        ),
        phrases: build_recovered_phrases(
            target,
            baseline,
            guard,
            options,
            &discovery_support_bi,
            &discovery_support_tri,
            effective_min_models,
        ),
    };

    Ok(FingerprintSuite {
        schema_version: 1,
        consensus,
        models,
    })
}

fn validate_profiles(
    target: &CorpusProfile,
    baseline: &CorpusProfile,
    guard: Option<&CorpusProfile>,
) -> Result<(), String> {
    if target.language != baseline.language {
        return Err("target and baseline profile languages do not match".into());
    }
    if target.schema_version != baseline.schema_version {
        return Err(
            "target and baseline profile schemas do not match; rebuild both profiles".into(),
        );
    }
    if let Some(guard) = guard {
        if guard.language != target.language {
            return Err("guard profile language does not match target".into());
        }
        if guard.schema_version != target.schema_version {
            return Err(
                "guard and target profile schemas do not match; rebuild both profiles".into(),
            );
        }
    }
    Ok(())
}

fn build_model_fingerprints(
    target: &CorpusProfile,
    baseline: &CorpusProfile,
    guard: Option<&CorpusProfile>,
    options: FingerprintOptions,
    lexical_reference: Option<&LexicalReference>,
) -> BTreeMap<String, Fingerprint> {
    target
        .model_profiles
        .iter()
        .map(|(model_id, model)| {
            let fingerprint =
                fingerprint_for_slice(model_id, model, baseline, guard, options, lexical_reference);
            (model_id.clone(), fingerprint)
        })
        .collect()
}

fn fingerprint_for_slice(
    model_id: &str,
    model: &ModelProfile,
    baseline: &CorpusProfile,
    guard: Option<&CorpusProfile>,
    options: FingerprintOptions,
    lexical_reference: Option<&LexicalReference>,
) -> Fingerprint {
    Fingerprint {
        schema_version: 4,
        language: baseline.language,
        min_documents: options.min_model_documents,
        min_models: 1,
        target_model_count: 1,
        target_label: Some(model_id.to_string()),
        target_model_id: Some(model_id.to_string()),
        target_family: Some(model.family.clone()),
        lexical_reference: lexical_reference.map(|reference| reference.source().to_string()),
        guard_profile_documents: guard.map(|profile| profile.document_count),
        min_guard_ratio: guard.map(|_| options.min_guard_ratio),
        words: compare(
            &model.slice.words,
            &baseline.words,
            guard.map(|profile| &profile.words),
            1,
            options.min_model_documents,
            options.word_limit,
            options.min_guard_ratio,
            None,
            1,
            lexical_reference,
            options.min_wordfreq_ratio,
            FingerprintSource::Lexical,
        ),
        bigrams: compare(
            &model.slice.bigrams,
            &baseline.bigrams,
            guard.map(|profile| &profile.bigrams),
            2,
            options.min_model_documents,
            options.bigram_limit,
            options.min_guard_ratio,
            None,
            1,
            None,
            0.0,
            FingerprintSource::LiteralNgram,
        ),
        trigrams: compare(
            &model.slice.trigrams,
            &baseline.trigrams,
            guard.map(|profile| &profile.trigrams),
            3,
            options.min_model_documents,
            options.trigram_limit,
            options.min_guard_ratio,
            None,
            1,
            None,
            0.0,
            FingerprintSource::LiteralNgram,
        ),
        phrases: Vec::new(),
    }
}

#[allow(clippy::too_many_arguments)]
fn compare(
    target: &BTreeMap<String, PatternStats>,
    baseline: &BTreeMap<String, PatternStats>,
    guard: Option<&BTreeMap<String, PatternStats>>,
    n: u8,
    min_documents: usize,
    limit: usize,
    min_guard_ratio: f64,
    support: Option<&HashMap<String, Support>>,
    min_models: usize,
    lexical_reference: Option<&LexicalReference>,
    min_wordfreq_ratio: f64,
    source: FingerprintSource,
) -> Vec<FingerprintEntry> {
    let keys: BTreeSet<_> = target.keys().chain(baseline.keys()).cloned().collect();
    let mut entries: Vec<_> = keys
        .into_iter()
        .filter_map(|pattern| {
            let target_stats = target.get(&pattern)?;
            let recurrence = if target_stats.prompt_frequency > 0 {
                target_stats.prompt_frequency
            } else {
                target_stats.document_frequency
            };
            if recurrence < min_documents {
                return None;
            }
            let pattern_support = support.and_then(|map| map.get(&pattern));
            if support.is_some()
                && pattern_support.map_or(0, |value| value.models.len()) < min_models
            {
                return None;
            }
            entry_for(
                pattern,
                target_stats,
                baseline,
                guard,
                n,
                min_guard_ratio,
                pattern_support,
                lexical_reference,
                min_wordfreq_ratio,
                source,
            )
        })
        .collect();
    entries.sort_by(|a, b| {
        ranking_score(b)
            .total_cmp(&ranking_score(a))
            .then_with(|| b.model_frequency.cmp(&a.model_frequency))
            .then_with(|| b.target_prompt_frequency.cmp(&a.target_prompt_frequency))
            .then_with(|| a.pattern.cmp(&b.pattern))
    });
    entries.truncate(limit);
    entries
}

#[allow(clippy::too_many_arguments)]
fn entry_for(
    pattern: String,
    target_stats: &PatternStats,
    baseline: &BTreeMap<String, PatternStats>,
    guard: Option<&BTreeMap<String, PatternStats>>,
    n: u8,
    min_guard_ratio: f64,
    support: Option<&Support>,
    lexical_reference: Option<&LexicalReference>,
    min_wordfreq_ratio: f64,
    source: FingerprintSource,
) -> Option<FingerprintEntry> {
    let baseline_frequency = baseline
        .get(&pattern)
        .map_or(0.0, |stats| stats.relative_frequency);
    let ratio =
        (baseline_frequency > 0.0).then_some(target_stats.relative_frequency / baseline_frequency);
    let guard_frequency = guard.map(|patterns| {
        patterns
            .get(&pattern)
            .map_or(0.0, |stats| stats.relative_frequency)
    });
    let guard_ratio = guard_frequency.and_then(|frequency| {
        (frequency > 0.0).then_some(target_stats.relative_frequency / frequency)
    });
    if guard_frequency.is_some() && guard_ratio.is_some_and(|value| value < min_guard_ratio) {
        return None;
    }
    let reference_frequency = lexical_reference.map(|reference| reference.frequency(&pattern));
    let reference_ratio = reference_frequency.and_then(|frequency| {
        (frequency > 0.0).then_some(target_stats.relative_frequency / frequency)
    });
    if lexical_reference.is_some()
        && reference_frequency.is_some_and(|frequency| frequency > 0.0)
        && reference_ratio.is_some_and(|value| value < min_wordfreq_ratio)
    {
        return None;
    }
    let (model_frequency, family_frequency, model_ids, families) = support.map_or_else(
        || (0, 0, Vec::new(), Vec::new()),
        |support| {
            (
                support.models.len(),
                support.families.len(),
                support.models.iter().cloned().collect(),
                support.families.iter().cloned().collect(),
            )
        },
    );
    Some(FingerprintEntry {
        pattern,
        n,
        target_frequency: target_stats.relative_frequency,
        baseline_frequency,
        ratio,
        target_document_frequency: target_stats.document_frequency,
        target_prompt_frequency: target_stats.prompt_frequency,
        model_frequency,
        family_frequency,
        model_ids,
        families,
        guard_frequency,
        guard_ratio,
        reference_frequency,
        reference_ratio,
        discovery_pattern: None,
        source,
    })
}

fn support_from_models(
    models: &BTreeMap<String, Fingerprint>,
    entries: impl Fn(&Fingerprint) -> &Vec<FingerprintEntry>,
    target: &CorpusProfile,
) -> HashMap<String, Support> {
    let mut support = HashMap::<String, Support>::new();
    for (model_id, fingerprint) in models {
        let family = target
            .model_profiles
            .get(model_id)
            .map(|profile| profile.family.clone())
            .unwrap_or_else(|| model_id.clone());
        for entry in entries(fingerprint) {
            let item = support.entry(entry.pattern.clone()).or_default();
            item.models.insert(model_id.clone());
            item.families.insert(family.clone());
        }
    }
    support
}

fn discovery_support(
    target: &CorpusProfile,
    baseline: &CorpusProfile,
    guard: Option<&CorpusProfile>,
    options: FingerprintOptions,
    n: u8,
) -> HashMap<String, Support> {
    let mut support = HashMap::<String, Support>::new();
    let baseline_patterns = if n == 2 {
        &baseline.discovery_bigrams
    } else {
        &baseline.discovery_trigrams
    };
    let guard_patterns = guard.map(|profile| {
        if n == 2 {
            &profile.discovery_bigrams
        } else {
            &profile.discovery_trigrams
        }
    });
    let limit = options.phrase_limit.max(40) * 2;
    for (model_id, model) in &target.model_profiles {
        let model_patterns = if n == 2 {
            &model.slice.discovery_bigrams
        } else {
            &model.slice.discovery_trigrams
        };
        let entries = compare(
            model_patterns,
            baseline_patterns,
            guard_patterns,
            n,
            options.min_model_documents,
            limit,
            options.min_guard_ratio,
            None,
            1,
            None,
            0.0,
            FingerprintSource::LiteralNgram,
        );
        for entry in entries {
            let item = support.entry(entry.pattern).or_default();
            item.models.insert(model_id.clone());
            item.families.insert(model.family.clone());
        }
    }
    support
}

fn build_recovered_phrases(
    target: &CorpusProfile,
    baseline: &CorpusProfile,
    guard: Option<&CorpusProfile>,
    options: FingerprintOptions,
    _bigram_support: &HashMap<String, Support>,
    trigram_support: &HashMap<String, Support>,
    min_models: usize,
) -> Vec<FingerprintEntry> {
    // Match the reference toolkit's phrase-recovery path: discover with three
    // content words, then recover the exact longer surface substring.
    let skeletons = compare(
        &target.discovery_trigrams,
        &baseline.discovery_trigrams,
        guard.map(|profile| &profile.discovery_trigrams),
        3,
        options.min_documents,
        options.phrase_limit * 3,
        options.min_guard_ratio,
        Some(trigram_support),
        min_models,
        None,
        0.0,
        FingerprintSource::LiteralNgram,
    );

    let mut phrases = Vec::new();
    for skeleton in skeletons {
        let Some(variants) = target.phrase_variants.get(&skeleton.pattern) else {
            continue;
        };
        for (phrase, stats) in variants {
            add_phrase_variant(&mut phrases, phrase, stats, &skeleton, min_models);
        }
    }
    phrases.sort_by(|a, b| {
        phrase_ranking_score(b)
            .total_cmp(&phrase_ranking_score(a))
            .then_with(|| a.pattern.cmp(&b.pattern))
    });
    let mut seen = BTreeSet::new();
    phrases.retain(|entry| seen.insert(entry.pattern.to_lowercase()));
    phrases.truncate(options.phrase_limit);
    phrases
}

fn add_phrase_variant(
    output: &mut Vec<FingerprintEntry>,
    phrase: &str,
    stats: &PhraseVariantStats,
    skeleton: &FingerprintEntry,
    min_models: usize,
) {
    let token_count = crate::tokenize::tokenize(phrase).len();
    if !(4..=16).contains(&token_count)
        || stats.model_frequency < min_models
        || !balanced_delimiters(phrase)
    {
        return;
    }
    output.push(FingerprintEntry {
        pattern: phrase.to_string(),
        n: token_count as u8,
        target_frequency: skeleton.target_frequency,
        baseline_frequency: skeleton.baseline_frequency,
        ratio: skeleton.ratio,
        target_document_frequency: stats.document_frequency,
        target_prompt_frequency: stats.prompt_frequency,
        model_frequency: stats.model_frequency,
        family_frequency: stats.family_frequency,
        model_ids: stats.model_ids.clone(),
        families: stats.families.clone(),
        guard_frequency: skeleton.guard_frequency,
        guard_ratio: skeleton.guard_ratio,
        reference_frequency: None,
        reference_ratio: None,
        discovery_pattern: Some(skeleton.pattern.clone()),
        source: FingerprintSource::RecoveredPhrase,
    });
}

fn balanced_delimiters(phrase: &str) -> bool {
    phrase.matches('(').count() == phrase.matches(')').count()
        && phrase.matches('[').count() == phrase.matches(']').count()
        && phrase.matches('{').count() == phrase.matches('}').count()
}

fn ranking_score(entry: &FingerprintEntry) -> f64 {
    let ratio_signal = match entry.ratio {
        Some(ratio) if ratio > 1.0 => ratio.log2().min(8.0),
        Some(_) => 0.0,
        None => match entry.n {
            1 => 1.0,
            2 => 2.0,
            _ => 3.0,
        },
    };
    let model_bonus = if entry.model_frequency > 0 {
        (entry.model_frequency as f64).ln_1p()
    } else {
        1.0
    };
    let reference_bonus = entry
        .reference_ratio
        .filter(|ratio| *ratio > 1.0)
        .map_or(1.0, |ratio| ratio.log2().clamp(1.0, 6.0));
    ratio_signal
        * (entry
            .target_prompt_frequency
            .max(entry.target_document_frequency) as f64)
            .ln_1p()
        * model_bonus
        * reference_bonus
}

fn phrase_ranking_score(entry: &FingerprintEntry) -> f64 {
    ranking_score(entry) * (entry.n as f64).ln_1p()
}
