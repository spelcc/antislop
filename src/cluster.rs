use crate::fingerprint::Fingerprint;
use serde::{Deserialize, Serialize};
use std::collections::{BTreeMap, BTreeSet, HashMap};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RankDistance {
    pub a: String,
    pub b: String,
    pub distance: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NearestNeighbor {
    pub label: String,
    pub neighbor: String,
    pub distance: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ClusterReport {
    pub schema_version: u32,
    pub labels: Vec<String>,
    pub distances: Vec<RankDistance>,
    pub nearest: Vec<NearestNeighbor>,
    pub newick: String,
}

#[derive(Debug, Clone)]
struct ClusterNode {
    labels: BTreeSet<String>,
    newick: String,
}

pub fn rank_features(fingerprint: &Fingerprint) -> Vec<String> {
    fingerprint
        .words
        .iter()
        .take(120)
        .map(|entry| format!("w:{}", entry.pattern))
        .chain(
            fingerprint
                .bigrams
                .iter()
                .take(40)
                .map(|entry| format!("b:{}", entry.pattern)),
        )
        .chain(
            fingerprint
                .trigrams
                .iter()
                .take(40)
                .map(|entry| format!("t:{}", entry.pattern)),
        )
        .collect()
}

pub fn normalized_rank_distance(a: &Fingerprint, b: &Fingerprint) -> f64 {
    let a_features = rank_features(a);
    let b_features = rank_features(b);
    let a_rank: HashMap<_, _> = a_features
        .iter()
        .enumerate()
        .map(|(index, value)| (value.as_str(), index + 1))
        .collect();
    let b_rank: HashMap<_, _> = b_features
        .iter()
        .enumerate()
        .map(|(index, value)| (value.as_str(), index + 1))
        .collect();
    let union: BTreeSet<_> = a_rank.keys().chain(b_rank.keys()).copied().collect();
    if union.is_empty() {
        return 0.0;
    }
    let missing_rank = a_features.len().max(b_features.len()) + 1;
    let total: usize = union
        .iter()
        .map(|feature| {
            let left = *a_rank.get(feature).unwrap_or(&missing_rank);
            let right = *b_rank.get(feature).unwrap_or(&missing_rank);
            left.abs_diff(right)
        })
        .sum();
    let maximum = union.len() * missing_rank.max(1);
    round4(total as f64 / maximum as f64)
}

pub fn cluster_fingerprints(items: &BTreeMap<String, Fingerprint>) -> ClusterReport {
    let labels: Vec<_> = items.keys().cloned().collect();
    let mut pairwise = BTreeMap::<(String, String), f64>::new();
    let mut distances = Vec::new();
    for left_index in 0..labels.len() {
        for right_index in left_index + 1..labels.len() {
            let a = &labels[left_index];
            let b = &labels[right_index];
            let distance = normalized_rank_distance(&items[a], &items[b]);
            pairwise.insert(pair_key(a, b), distance);
            distances.push(RankDistance {
                a: a.clone(),
                b: b.clone(),
                distance,
            });
        }
    }

    let nearest = labels
        .iter()
        .filter_map(|label| {
            labels
                .iter()
                .filter(|other| *other != label)
                .map(|other| (other, distance_for(label, other, &pairwise)))
                .min_by(|(a_label, a_distance), (b_label, b_distance)| {
                    a_distance
                        .total_cmp(b_distance)
                        .then_with(|| a_label.cmp(b_label))
                })
                .map(|(neighbor, distance)| NearestNeighbor {
                    label: label.clone(),
                    neighbor: neighbor.clone(),
                    distance,
                })
        })
        .collect();

    let mut clusters: Vec<_> = labels
        .iter()
        .map(|label| ClusterNode {
            labels: BTreeSet::from([label.clone()]),
            newick: escape_newick(label),
        })
        .collect();
    while clusters.len() > 1 {
        let mut best: Option<(usize, usize, f64, String)> = None;
        for left in 0..clusters.len() {
            for right in left + 1..clusters.len() {
                let distance = average_linkage(&clusters[left], &clusters[right], &pairwise);
                let tie = format!("{}|{}", clusters[left].newick, clusters[right].newick);
                if best.as_ref().is_none_or(|(_, _, best_distance, best_tie)| {
                    distance < *best_distance || (distance == *best_distance && tie < *best_tie)
                }) {
                    best = Some((left, right, distance, tie));
                }
            }
        }
        let (left, right, distance, _) = best.expect("at least two clusters remain");
        let right_node = clusters.remove(right);
        let left_node = clusters.remove(left);
        let labels = left_node
            .labels
            .union(&right_node.labels)
            .cloned()
            .collect();
        clusters.push(ClusterNode {
            labels,
            newick: format!(
                "({}:{:.4},{}:{:.4})",
                left_node.newick,
                distance / 2.0,
                right_node.newick,
                distance / 2.0
            ),
        });
        clusters.sort_by(|a, b| a.newick.cmp(&b.newick));
    }

    ClusterReport {
        schema_version: 1,
        labels,
        distances,
        nearest,
        newick: clusters
            .pop()
            .map_or_else(|| ";".to_string(), |cluster| format!("{};", cluster.newick)),
    }
}

fn average_linkage(
    left: &ClusterNode,
    right: &ClusterNode,
    pairwise: &BTreeMap<(String, String), f64>,
) -> f64 {
    let mut total = 0.0;
    let mut count = 0usize;
    for a in &left.labels {
        for b in &right.labels {
            total += distance_for(a, b, pairwise);
            count += 1;
        }
    }
    if count == 0 {
        0.0
    } else {
        total / count as f64
    }
}

fn distance_for(a: &str, b: &str, pairwise: &BTreeMap<(String, String), f64>) -> f64 {
    if a == b {
        0.0
    } else {
        *pairwise.get(&pair_key(a, b)).unwrap_or(&1.0)
    }
}

fn pair_key(a: &str, b: &str) -> (String, String) {
    if a <= b {
        (a.to_string(), b.to_string())
    } else {
        (b.to_string(), a.to_string())
    }
}

fn escape_newick(label: &str) -> String {
    label
        .chars()
        .map(|ch| {
            if ch.is_ascii_alphanumeric() || matches!(ch, '-' | '_' | '.') {
                ch
            } else {
                '_'
            }
        })
        .collect()
}

fn round4(value: f64) -> f64 {
    (value * 10_000.0).round() / 10_000.0
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::fingerprint::{FingerprintEntry, FingerprintSource};
    use crate::tokenize::Language;

    fn fp(label: &str, words: &[&str]) -> Fingerprint {
        Fingerprint {
            schema_version: 4,
            language: Language::Fr,
            min_documents: 1,
            min_models: 1,
            target_model_count: 1,
            target_label: Some(label.into()),
            target_model_id: None,
            target_family: None,
            lexical_reference: None,
            guard_profile_documents: None,
            min_guard_ratio: None,
            words: words
                .iter()
                .map(|word| FingerprintEntry {
                    pattern: (*word).into(),
                    n: 1,
                    target_frequency: 0.01,
                    baseline_frequency: 0.001,
                    ratio: Some(10.0),
                    target_document_frequency: 3,
                    target_prompt_frequency: 3,
                    model_frequency: 1,
                    family_frequency: 1,
                    model_ids: vec![],
                    families: vec![],
                    guard_frequency: None,
                    guard_ratio: None,
                    reference_frequency: None,
                    reference_ratio: None,
                    discovery_pattern: None,
                    source: FingerprintSource::Lexical,
                })
                .collect(),
            bigrams: vec![],
            trigrams: vec![],
            phrases: vec![],
        }
    }

    #[test]
    fn identical_rankings_have_zero_distance() {
        assert_eq!(
            normalized_rank_distance(&fp("a", &["x", "y"]), &fp("b", &["x", "y"])),
            0.0
        );
    }

    #[test]
    fn clustering_places_similar_rankings_as_nearest_neighbors() {
        let items = BTreeMap::from([
            ("arthur".into(), fp("arthur", &["x", "y", "z"])),
            ("model-a".into(), fp("model-a", &["x", "y", "q"])),
            ("model-b".into(), fp("model-b", &["r", "s", "t"])),
        ]);
        let report = cluster_fingerprints(&items);
        let arthur = report
            .nearest
            .iter()
            .find(|item| item.label == "arthur")
            .unwrap();
        assert_eq!(arthur.neighbor, "model-a");
        assert!(report.newick.ends_with(';'));
    }
}
