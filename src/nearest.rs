use crate::analyze::analyze_text;
use crate::cluster::normalized_rank_distance;
use crate::fingerprint::{Fingerprint, FingerprintOptions, build_fingerprint_suite};
use crate::profile::{CorpusProfile, profile_documents};
use crate::tokenize::Language;
use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;
use std::fs;
use std::path::{Path, PathBuf};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum NearestMetric {
    RankDistance,
    DocumentSignal,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NearestMatch {
    pub label: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub model_id: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub family: Option<String>,
    pub rank_distance: f64,
    pub rank_distance_position: usize,
    pub document_signal_per_1000_tokens: f64,
    pub document_signal_position: usize,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NearestReport {
    pub schema_version: u32,
    pub language: Language,
    pub metric: String,
    pub candidate_count: usize,
    pub target_fingerprint: Fingerprint,
    pub matches: Vec<NearestMatch>,
}

pub fn load_fingerprint_directory(
    directory: &Path,
) -> Result<BTreeMap<String, Fingerprint>, String> {
    if !directory.is_dir() {
        return Err(format!(
            "models directory not found: {}",
            directory.display()
        ));
    }
    let mut files = Vec::new();
    collect_json_files(directory, &mut files)?;
    files.sort();
    if files.is_empty() {
        return Err(format!(
            "models directory contains no JSON fingerprints: {}",
            directory.display()
        ));
    }
    let mut fingerprints = BTreeMap::new();
    for path in files {
        let raw = fs::read_to_string(&path)
            .map_err(|error| format!("failed to read {}: {error}", path.display()))?;
        let fingerprint: Fingerprint = serde_json::from_str(&raw)
            .map_err(|error| format!("invalid fingerprint {}: {error}", path.display()))?;
        let label = fingerprint
            .target_model_id
            .clone()
            .or_else(|| fingerprint.target_label.clone())
            .unwrap_or_else(|| {
                path.file_stem()
                    .and_then(|value| value.to_str())
                    .unwrap_or("fingerprint")
                    .to_string()
            });
        if fingerprints.insert(label.clone(), fingerprint).is_some() {
            return Err(format!("duplicate model fingerprint label: {label}"));
        }
    }
    Ok(fingerprints)
}

fn collect_json_files(directory: &Path, files: &mut Vec<PathBuf>) -> Result<(), String> {
    let entries = fs::read_dir(directory)
        .map_err(|error| format!("failed to read {}: {error}", directory.display()))?;
    for entry in entries {
        let path = entry
            .map_err(|error| format!("failed to read directory entry: {error}"))?
            .path();
        if path.is_dir() {
            collect_json_files(&path, files)?;
        } else if path.extension().and_then(|value| value.to_str()) == Some("json") {
            files.push(path);
        }
    }
    Ok(())
}

pub fn nearest_fingerprints(
    text: &str,
    language: Language,
    baseline: &CorpusProfile,
    guard: Option<&CorpusProfile>,
    candidates: &BTreeMap<String, Fingerprint>,
    metric: NearestMetric,
    top: usize,
) -> Result<NearestReport, String> {
    if candidates.is_empty() {
        return Err("nearest requires at least one candidate fingerprint".into());
    }
    if baseline.language != language {
        return Err("baseline language does not match requested language".into());
    }
    if let Some(guard) = guard
        && guard.language != language
    {
        return Err("guard language does not match requested language".into());
    }
    let first = candidates
        .values()
        .next()
        .expect("non-empty candidates checked above");
    let uses_wordfreq = first.lexical_reference.is_some();
    let uses_guard = first.guard_profile_documents.is_some();
    for (label, fingerprint) in candidates {
        if fingerprint.language != language {
            return Err(format!(
                "candidate fingerprint {label} uses {:?}, expected {:?}",
                fingerprint.language, language
            ));
        }
        if fingerprint.lexical_reference.is_some() != uses_wordfreq {
            return Err(
                "candidate fingerprints disagree on lexical reference configuration".into(),
            );
        }
        if fingerprint.guard_profile_documents.is_some() != uses_guard {
            return Err("candidate fingerprints disagree on guard configuration".into());
        }
    }
    if uses_guard != guard.is_some() {
        return Err(if uses_guard {
            "candidate fingerprints were built with a guard; pass the matching --guard profile"
                .into()
        } else {
            "candidate fingerprints were built without a guard; omit --guard for a comparable target fingerprint"
                .into()
        });
    }

    let target_profile = profile_documents(
        language,
        &[("nearest-target".to_string(), text.to_string())],
    );
    let target_fingerprint = build_fingerprint_suite(
        &target_profile,
        baseline,
        guard,
        FingerprintOptions {
            min_documents: 1,
            min_model_documents: 1,
            min_models: 1,
            word_limit: 120,
            bigram_limit: 40,
            trigram_limit: 40,
            phrase_limit: 0,
            min_guard_ratio: first.min_guard_ratio.unwrap_or(2.0),
            use_wordfreq: uses_wordfreq,
            ..Default::default()
        },
        Some("target".to_string()),
    )?
    .consensus;

    let mut matches: Vec<_> = candidates
        .iter()
        .map(|(label, fingerprint)| {
            let analysis = analyze_text(text, language, Some(fingerprint));
            NearestMatch {
                label: label.clone(),
                model_id: fingerprint.target_model_id.clone(),
                family: fingerprint.target_family.clone(),
                rank_distance: normalized_rank_distance(&target_fingerprint, fingerprint),
                rank_distance_position: 0,
                document_signal_per_1000_tokens: analysis.fingerprint_signal_per_1000_tokens,
                document_signal_position: 0,
            }
        })
        .collect();

    let mut distance_order: Vec<_> = (0..matches.len()).collect();
    distance_order.sort_by(|&left, &right| {
        matches[left]
            .rank_distance
            .total_cmp(&matches[right].rank_distance)
            .then_with(|| matches[left].label.cmp(&matches[right].label))
    });
    for (position, index) in distance_order.into_iter().enumerate() {
        matches[index].rank_distance_position = position + 1;
    }

    let mut signal_order: Vec<_> = (0..matches.len()).collect();
    signal_order.sort_by(|&left, &right| {
        matches[right]
            .document_signal_per_1000_tokens
            .total_cmp(&matches[left].document_signal_per_1000_tokens)
            .then_with(|| matches[left].label.cmp(&matches[right].label))
    });
    for (position, index) in signal_order.into_iter().enumerate() {
        matches[index].document_signal_position = position + 1;
    }

    match metric {
        NearestMetric::RankDistance => matches.sort_by(|left, right| {
            left.rank_distance_position
                .cmp(&right.rank_distance_position)
        }),
        NearestMetric::DocumentSignal => matches.sort_by(|left, right| {
            left.document_signal_position
                .cmp(&right.document_signal_position)
        }),
    }
    matches.truncate(top.min(matches.len()));

    Ok(NearestReport {
        schema_version: 1,
        language,
        metric: match metric {
            NearestMetric::RankDistance => "rank_distance",
            NearestMetric::DocumentSignal => "document_signal",
        }
        .to_string(),
        candidate_count: candidates.len(),
        target_fingerprint,
        matches,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::fingerprint::build_fingerprint;

    fn fingerprint_for(text: &str, baseline: &CorpusProfile, label: &str) -> Fingerprint {
        let target = profile_documents(
            Language::Fr,
            &[
                ("1".into(), text.into()),
                ("2".into(), text.into()),
                ("3".into(), text.into()),
            ],
        );
        let mut fingerprint = build_fingerprint(&target, baseline, 1, 120, 40, 40);
        fingerprint.target_label = Some(label.into());
        fingerprint.target_model_id = Some(label.into());
        fingerprint
    }

    #[test]
    fn ranks_matching_fingerprint_closer() {
        let baseline = profile_documents(
            Language::Fr,
            &[
                (
                    "h1".into(),
                    "Une prose humaine ordinaire décrit le monde.".into(),
                ),
                ("h2".into(), "Le texte humain reste sobre et précis.".into()),
                (
                    "h3".into(),
                    "Cette phrase humaine demeure naturelle.".into(),
                ),
            ],
        );
        let matching = fingerprint_for(
            "Il est important de noter ce résultat. Il est important de noter ce résultat.",
            &baseline,
            "matching",
        );
        let alien = fingerprint_for(
            "Galaxie turquoise mécanique banquise. Galaxie turquoise mécanique banquise.",
            &baseline,
            "alien",
        );
        let candidates = BTreeMap::from([
            ("matching".to_string(), matching),
            ("alien".to_string(), alien),
        ]);
        let report = nearest_fingerprints(
            "Il est important de noter ce résultat et important de noter cette différence.",
            Language::Fr,
            &baseline,
            None,
            &candidates,
            NearestMetric::RankDistance,
            2,
        )
        .unwrap();
        assert_eq!(report.matches[0].label, "matching");
        assert_eq!(report.matches[0].rank_distance_position, 1);
        assert_eq!(report.candidate_count, 2);
    }
}
