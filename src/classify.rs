use crate::nearest::{
    CandidateClass, NearestCandidate, NearestMatch, NearestMetric, nearest_candidates,
};
use crate::profile::CorpusProfile;
use crate::tokenize::Language;
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::collections::BTreeMap;
use std::fs;
use std::path::{Path, PathBuf};

const FEATURE_NAMES: [&str; 5] = [
    "distance_margin_llm",
    "signal_margin_llm",
    "style_margin_llm",
    "top5_distance_llm_fraction",
    "top5_signal_llm_fraction",
];

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum CalibrationSplit {
    Train,
    Test,
}

#[derive(Debug, Clone)]
pub struct CalibrationDocument {
    pub id: String,
    pub class: CandidateClass,
    pub split: CalibrationSplit,
    pub text: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ClassificationFeatures {
    pub distance_margin_llm: f64,
    pub signal_margin_llm: f64,
    pub style_margin_llm: f64,
    pub top5_distance_llm_fraction: f64,
    pub top5_signal_llm_fraction: f64,
}

impl ClassificationFeatures {
    fn as_array(&self) -> [f64; 5] {
        [
            self.distance_margin_llm,
            self.signal_margin_llm,
            self.style_margin_llm,
            self.top5_distance_llm_fraction,
            self.top5_signal_llm_fraction,
        ]
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CandidateIdentity {
    pub label: String,
    pub class: CandidateClass,
    pub fingerprint_sha256: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub style_profile_sha256: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ClassifierEvaluation {
    pub documents: usize,
    pub human_documents: usize,
    pub llm_documents: usize,
    pub accuracy: f64,
    pub roc_auc: f64,
    pub brier_score: f64,
    pub log_loss: f64,
    #[serde(default)]
    pub expected_calibration_error: f64,
    pub true_human_predicted_human: usize,
    pub true_human_predicted_llm: usize,
    pub true_llm_predicted_human: usize,
    pub true_llm_predicted_llm: usize,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CalibratedClassifier {
    pub schema_version: u32,
    pub language: Language,
    pub positive_class: CandidateClass,
    pub prior_llm: f64,
    pub baseline_sha256: String,
    pub feature_names: Vec<String>,
    pub feature_means: Vec<f64>,
    pub feature_scales: Vec<f64>,
    pub weights: Vec<f64>,
    pub intercept: f64,
    pub candidates: Vec<CandidateIdentity>,
    pub train_documents: usize,
    pub test_documents: usize,
    pub evaluation: ClassifierEvaluation,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CalibrationReport {
    pub schema_version: u32,
    pub train_documents: usize,
    pub test_documents: usize,
    pub evaluation: ClassifierEvaluation,
    pub classifier: CalibratedClassifier,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EvidencePoint {
    pub label: String,
    pub value: f64,
    pub global_rank: usize,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ClassificationEvidence {
    pub best_human_distance: EvidencePoint,
    pub best_llm_distance: EvidencePoint,
    pub best_human_signal: EvidencePoint,
    pub best_llm_signal: EvidencePoint,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub best_human_style: Option<EvidencePoint>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub best_llm_style: Option<EvidencePoint>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ClassificationReport {
    pub schema_version: u32,
    pub language: Language,
    pub predicted_class: CandidateClass,
    pub human_probability: f64,
    pub llm_probability: f64,
    pub features: ClassificationFeatures,
    pub feature_contributions_to_llm_logit: BTreeMap<String, f64>,
    pub intercept_contribution: f64,
    pub llm_logit: f64,
    pub evidence: ClassificationEvidence,
    pub nearest: Vec<NearestMatch>,
    pub calibration: ClassifierEvaluation,
    pub calibration_prior_llm: f64,
}

#[derive(Debug, Deserialize)]
struct CalibrationManifest {
    #[serde(default)]
    language: Option<Language>,
    documents: Vec<CalibrationManifestDocument>,
}

#[derive(Debug, Deserialize)]
struct CalibrationManifestDocument {
    file: String,
    class: CandidateClass,
    split: CalibrationSplit,
}

pub fn load_calibration_manifest(
    path: &Path,
    language: Language,
) -> Result<Vec<CalibrationDocument>, String> {
    let raw = fs::read_to_string(path).map_err(|error| {
        format!(
            "failed to read calibration manifest {}: {error}",
            path.display()
        )
    })?;
    let manifest: CalibrationManifest = serde_json::from_str(&raw)
        .map_err(|error| format!("invalid calibration manifest {}: {error}", path.display()))?;
    if let Some(manifest_language) = manifest.language
        && manifest_language != language
    {
        return Err(format!(
            "calibration manifest language {:?} does not match {:?}",
            manifest_language, language
        ));
    }
    let base = path.parent().unwrap_or_else(|| Path::new("."));
    let mut documents = Vec::new();
    for document in manifest.documents {
        let supplied = Path::new(&document.file);
        let resolved: PathBuf = if supplied.is_absolute() {
            supplied.to_path_buf()
        } else {
            base.join(supplied)
        };
        let text = fs::read_to_string(&resolved).map_err(|error| {
            format!(
                "failed to read calibration document {}: {error}",
                resolved.display()
            )
        })?;
        documents.push(CalibrationDocument {
            id: document.file,
            class: document.class,
            split: document.split,
            text,
        });
    }
    Ok(documents)
}

pub fn classification_features(matches: &[NearestMatch]) -> Result<ClassificationFeatures, String> {
    let humans: Vec<_> = matches
        .iter()
        .filter(|entry| entry.class == CandidateClass::Human)
        .collect();
    let llms: Vec<_> = matches
        .iter()
        .filter(|entry| entry.class == CandidateClass::Llm)
        .collect();
    if humans.is_empty() || llms.is_empty() {
        return Err("classification requires at least one human and one LLM candidate".into());
    }

    let best_human_distance = humans
        .iter()
        .map(|entry| entry.rank_distance)
        .fold(f64::INFINITY, f64::min);
    let best_llm_distance = llms
        .iter()
        .map(|entry| entry.rank_distance)
        .fold(f64::INFINITY, f64::min);
    let best_human_signal = humans
        .iter()
        .map(|entry| entry.document_signal_per_1000_tokens)
        .fold(f64::NEG_INFINITY, f64::max);
    let best_llm_signal = llms
        .iter()
        .map(|entry| entry.document_signal_per_1000_tokens)
        .fold(f64::NEG_INFINITY, f64::max);

    let human_style = humans
        .iter()
        .filter_map(|entry| entry.style_distance)
        .fold(f64::INFINITY, f64::min);
    let llm_style = llms
        .iter()
        .filter_map(|entry| entry.style_distance)
        .fold(f64::INFINITY, f64::min);
    let style_margin = if human_style.is_finite() && llm_style.is_finite() {
        human_style - llm_style
    } else {
        0.0
    };

    let mut by_distance: Vec<_> = matches.iter().collect();
    by_distance.sort_by(|left, right| {
        left.rank_distance
            .total_cmp(&right.rank_distance)
            .then_with(|| left.label.cmp(&right.label))
    });
    let mut by_signal: Vec<_> = matches.iter().collect();
    by_signal.sort_by(|left, right| {
        right
            .document_signal_per_1000_tokens
            .total_cmp(&left.document_signal_per_1000_tokens)
            .then_with(|| left.label.cmp(&right.label))
    });

    Ok(ClassificationFeatures {
        distance_margin_llm: best_human_distance - best_llm_distance,
        signal_margin_llm: best_llm_signal - best_human_signal,
        style_margin_llm: style_margin,
        top5_distance_llm_fraction: top_class_fraction(&by_distance, CandidateClass::Llm, 5),
        top5_signal_llm_fraction: top_class_fraction(&by_signal, CandidateClass::Llm, 5),
    })
}

fn top_class_fraction(matches: &[&NearestMatch], class: CandidateClass, limit: usize) -> f64 {
    let count = matches.len().min(limit);
    if count == 0 {
        return 0.0;
    }
    matches
        .iter()
        .take(count)
        .filter(|entry| entry.class == class)
        .count() as f64
        / count as f64
}

pub fn calibrate_classifier(
    documents: &[CalibrationDocument],
    language: Language,
    baseline: &CorpusProfile,
    candidates: &BTreeMap<String, NearestCandidate>,
    epochs: usize,
    l2: f64,
) -> Result<CalibrationReport, String> {
    if epochs == 0 {
        return Err("calibration epochs must be positive".into());
    }
    if !(0.0..=1.0).contains(&l2) {
        return Err("calibration L2 must be between 0 and 1".into());
    }
    validate_calibration_splits(documents)?;
    let mut observations = Vec::with_capacity(documents.len());
    for document in documents {
        let report = nearest_candidates(
            &document.text,
            language,
            baseline,
            None,
            candidates,
            NearestMetric::RankDistance,
            candidates.len(),
        )?;
        observations.push((
            document.split,
            document.class,
            classification_features(&report.matches)?,
        ));
    }

    let train: Vec<_> = observations
        .iter()
        .filter(|(split, _, _)| *split == CalibrationSplit::Train)
        .collect();
    let test: Vec<_> = observations
        .iter()
        .filter(|(split, _, _)| *split == CalibrationSplit::Test)
        .collect();
    let (means, scales) = standardization(&train);
    let (weights, intercept) = fit_logistic(&train, &means, &scales, epochs, l2);
    let evaluation = evaluate(&test, &means, &scales, &weights, intercept);
    let identities = candidate_identities(candidates)?;
    let classifier = CalibratedClassifier {
        schema_version: 1,
        language,
        positive_class: CandidateClass::Llm,
        prior_llm: 0.5,
        baseline_sha256: stable_sha256(baseline)?,
        feature_names: FEATURE_NAMES
            .iter()
            .map(|name| (*name).to_string())
            .collect(),
        feature_means: means.to_vec(),
        feature_scales: scales.to_vec(),
        weights: weights.to_vec(),
        intercept,
        candidates: identities,
        train_documents: train.len(),
        test_documents: test.len(),
        evaluation: evaluation.clone(),
    };
    Ok(CalibrationReport {
        schema_version: 1,
        train_documents: train.len(),
        test_documents: test.len(),
        evaluation,
        classifier,
    })
}

fn validate_calibration_splits(documents: &[CalibrationDocument]) -> Result<(), String> {
    for split in [CalibrationSplit::Train, CalibrationSplit::Test] {
        for class in [CandidateClass::Human, CandidateClass::Llm] {
            let count = documents
                .iter()
                .filter(|document| document.split == split && document.class == class)
                .count();
            if count == 0 {
                return Err(format!(
                    "calibration split {:?} contains no {:?} documents",
                    split, class
                ));
            }
        }
    }
    Ok(())
}

fn standardization(
    observations: &[&(CalibrationSplit, CandidateClass, ClassificationFeatures)],
) -> ([f64; 5], [f64; 5]) {
    let mut means = [0.0; 5];
    for (_, _, features) in observations {
        for (index, value) in features.as_array().iter().enumerate() {
            means[index] += value;
        }
    }
    for mean in &mut means {
        *mean /= observations.len() as f64;
    }
    let mut scales = [0.0; 5];
    for (_, _, features) in observations {
        for (index, value) in features.as_array().iter().enumerate() {
            scales[index] += (value - means[index]).powi(2);
        }
    }
    for scale in &mut scales {
        *scale = (*scale / observations.len() as f64).sqrt();
        if *scale < 1e-9 {
            *scale = 1.0;
        }
    }
    (means, scales)
}

fn fit_logistic(
    observations: &[&(CalibrationSplit, CandidateClass, ClassificationFeatures)],
    means: &[f64; 5],
    scales: &[f64; 5],
    epochs: usize,
    l2: f64,
) -> ([f64; 5], f64) {
    let human_count = observations
        .iter()
        .filter(|(_, class, _)| *class == CandidateClass::Human)
        .count() as f64;
    let llm_count = observations.len() as f64 - human_count;
    let mut weights = [0.0; 5];
    let mut intercept = 0.0;
    let learning_rate = 0.05;
    for _ in 0..epochs {
        let mut grad = [0.0; 5];
        let mut grad_intercept = 0.0;
        let mut total_weight = 0.0;
        for (_, class, features) in observations {
            let x = standardized(features, means, scales);
            let y = if *class == CandidateClass::Llm {
                1.0
            } else {
                0.0
            };
            let class_weight = if y > 0.5 {
                0.5 / llm_count
            } else {
                0.5 / human_count
            };
            let prediction = sigmoid(intercept + dot(&weights, &x));
            let error = (prediction - y) * class_weight;
            for index in 0..5 {
                grad[index] += error * x[index];
            }
            grad_intercept += error;
            total_weight += class_weight;
        }
        for index in 0..5 {
            let penalty = l2 * weights[index];
            weights[index] -= learning_rate * (grad[index] / total_weight + penalty);
        }
        intercept -= learning_rate * grad_intercept / total_weight;
    }
    (weights, intercept)
}

fn standardized(
    features: &ClassificationFeatures,
    means: &[f64; 5],
    scales: &[f64; 5],
) -> [f64; 5] {
    let raw = features.as_array();
    std::array::from_fn(|index| (raw[index] - means[index]) / scales[index])
}

fn dot(left: &[f64; 5], right: &[f64; 5]) -> f64 {
    (0..5).map(|index| left[index] * right[index]).sum()
}

fn sigmoid(value: f64) -> f64 {
    if value >= 0.0 {
        1.0 / (1.0 + (-value).exp())
    } else {
        let exp = value.exp();
        exp / (1.0 + exp)
    }
}

fn evaluate(
    observations: &[&(CalibrationSplit, CandidateClass, ClassificationFeatures)],
    means: &[f64; 5],
    scales: &[f64; 5],
    weights: &[f64; 5],
    intercept: f64,
) -> ClassifierEvaluation {
    let mut pairs = Vec::with_capacity(observations.len());
    let mut correct = 0usize;
    let mut brier = 0.0;
    let mut log_loss = 0.0;
    let mut hh = 0;
    let mut hl = 0;
    let mut lh = 0;
    let mut ll = 0;
    for (_, class, features) in observations {
        let x = standardized(features, means, scales);
        let probability = sigmoid(intercept + dot(weights, &x)).clamp(1e-12, 1.0 - 1e-12);
        let actual = if *class == CandidateClass::Llm {
            1.0
        } else {
            0.0
        };
        let predicted_llm = probability >= 0.5;
        correct += usize::from(predicted_llm == (actual > 0.5));
        brier += (probability - actual).powi(2);
        log_loss += -(actual * probability.ln() + (1.0 - actual) * (1.0 - probability).ln());
        pairs.push((probability, actual));
        match (*class, predicted_llm) {
            (CandidateClass::Human, false) => hh += 1,
            (CandidateClass::Human, true) => hl += 1,
            (CandidateClass::Llm, false) => lh += 1,
            (CandidateClass::Llm, true) => ll += 1,
        }
    }
    let count = observations.len().max(1) as f64;
    ClassifierEvaluation {
        documents: observations.len(),
        human_documents: observations
            .iter()
            .filter(|(_, class, _)| *class == CandidateClass::Human)
            .count(),
        llm_documents: observations
            .iter()
            .filter(|(_, class, _)| *class == CandidateClass::Llm)
            .count(),
        accuracy: round4(correct as f64 / count),
        roc_auc: round4(roc_auc(&pairs)),
        brier_score: round4(brier / count),
        log_loss: round4(log_loss / count),
        expected_calibration_error: round4(expected_calibration_error(&pairs, 10)),
        true_human_predicted_human: hh,
        true_human_predicted_llm: hl,
        true_llm_predicted_human: lh,
        true_llm_predicted_llm: ll,
    }
}

fn expected_calibration_error(pairs: &[(f64, f64)], bins: usize) -> f64 {
    if pairs.is_empty() || bins == 0 {
        return 0.0;
    }
    let mut total = 0.0;
    for bin in 0..bins {
        let low = bin as f64 / bins as f64;
        let high = (bin + 1) as f64 / bins as f64;
        let values: Vec<_> = pairs
            .iter()
            .filter(|(probability, _)| {
                *probability >= low
                    && (*probability < high || (bin + 1 == bins && *probability <= 1.0))
            })
            .collect();
        if values.is_empty() {
            continue;
        }
        let confidence = values
            .iter()
            .map(|(probability, _)| *probability)
            .sum::<f64>()
            / values.len() as f64;
        let observed = values.iter().map(|(_, actual)| *actual).sum::<f64>() / values.len() as f64;
        total += values.len() as f64 / pairs.len() as f64 * (confidence - observed).abs();
    }
    total
}

fn roc_auc(pairs: &[(f64, f64)]) -> f64 {
    let positives: Vec<_> = pairs.iter().filter(|(_, y)| *y > 0.5).collect();
    let negatives: Vec<_> = pairs.iter().filter(|(_, y)| *y <= 0.5).collect();
    if positives.is_empty() || negatives.is_empty() {
        return 0.5;
    }
    let mut wins = 0.0;
    for positive in &positives {
        for negative in &negatives {
            wins += if positive.0 > negative.0 {
                1.0
            } else if positive.0 == negative.0 {
                0.5
            } else {
                0.0
            };
        }
    }
    wins / (positives.len() * negatives.len()) as f64
}

pub fn classify_text(
    text: &str,
    language: Language,
    baseline: &CorpusProfile,
    candidates: &BTreeMap<String, NearestCandidate>,
    classifier: &CalibratedClassifier,
    top: usize,
) -> Result<ClassificationReport, String> {
    validate_classifier(classifier, language, baseline, candidates)?;
    let nearest = nearest_candidates(
        text,
        language,
        baseline,
        None,
        candidates,
        NearestMetric::RankDistance,
        candidates.len(),
    )?;
    let features = classification_features(&nearest.matches)?;
    let (llm_probability, llm_logit, contributions) =
        probability_with_contributions(&features, classifier)?;
    let evidence = classification_evidence(&nearest.matches)?;
    let predicted_class = if llm_probability >= 0.5 {
        CandidateClass::Llm
    } else {
        CandidateClass::Human
    };
    Ok(ClassificationReport {
        schema_version: 1,
        language,
        predicted_class,
        human_probability: round4(1.0 - llm_probability),
        llm_probability: round4(llm_probability),
        features,
        feature_contributions_to_llm_logit: contributions,
        intercept_contribution: round4(classifier.intercept),
        llm_logit: round4(llm_logit),
        evidence,
        nearest: nearest.matches.into_iter().take(top).collect(),
        calibration: classifier.evaluation.clone(),
        calibration_prior_llm: classifier.prior_llm,
    })
}

fn probability_with_contributions(
    features: &ClassificationFeatures,
    classifier: &CalibratedClassifier,
) -> Result<(f64, f64, BTreeMap<String, f64>), String> {
    if classifier.feature_means.len() != 5
        || classifier.feature_scales.len() != 5
        || classifier.weights.len() != 5
        || classifier.feature_names.len() != 5
    {
        return Err("classifier feature vectors do not match expected schema".into());
    }
    let raw = features.as_array();
    let mut logit = classifier.intercept;
    let mut contributions = BTreeMap::new();
    for (index, value) in raw.iter().enumerate() {
        let contribution = classifier.weights[index]
            * ((*value - classifier.feature_means[index]) / classifier.feature_scales[index]);
        logit += contribution;
        contributions.insert(
            classifier.feature_names[index].clone(),
            round4(contribution),
        );
    }
    Ok((sigmoid(logit), logit, contributions))
}

fn classification_evidence(matches: &[NearestMatch]) -> Result<ClassificationEvidence, String> {
    fn best_distance(matches: &[NearestMatch], class: CandidateClass) -> Option<EvidencePoint> {
        matches
            .iter()
            .filter(|entry| entry.class == class)
            .min_by(|left, right| left.rank_distance.total_cmp(&right.rank_distance))
            .map(|entry| EvidencePoint {
                label: entry.label.clone(),
                value: entry.rank_distance,
                global_rank: entry.rank_distance_position,
            })
    }
    fn best_signal(matches: &[NearestMatch], class: CandidateClass) -> Option<EvidencePoint> {
        matches
            .iter()
            .filter(|entry| entry.class == class)
            .max_by(|left, right| {
                left.document_signal_per_1000_tokens
                    .total_cmp(&right.document_signal_per_1000_tokens)
            })
            .map(|entry| EvidencePoint {
                label: entry.label.clone(),
                value: entry.document_signal_per_1000_tokens,
                global_rank: entry.document_signal_position,
            })
    }
    fn best_style(matches: &[NearestMatch], class: CandidateClass) -> Option<EvidencePoint> {
        matches
            .iter()
            .filter(|entry| entry.class == class && entry.style_distance.is_some())
            .min_by(|left, right| {
                left.style_distance
                    .unwrap_or(f64::INFINITY)
                    .total_cmp(&right.style_distance.unwrap_or(f64::INFINITY))
            })
            .map(|entry| EvidencePoint {
                label: entry.label.clone(),
                value: entry.style_distance.unwrap_or(f64::INFINITY),
                global_rank: entry.style_distance_position.unwrap_or(usize::MAX),
            })
    }
    Ok(ClassificationEvidence {
        best_human_distance: best_distance(matches, CandidateClass::Human)
            .ok_or_else(|| "classification requires a human distance candidate".to_string())?,
        best_llm_distance: best_distance(matches, CandidateClass::Llm)
            .ok_or_else(|| "classification requires an LLM distance candidate".to_string())?,
        best_human_signal: best_signal(matches, CandidateClass::Human)
            .ok_or_else(|| "classification requires a human signal candidate".to_string())?,
        best_llm_signal: best_signal(matches, CandidateClass::Llm)
            .ok_or_else(|| "classification requires an LLM signal candidate".to_string())?,
        best_human_style: best_style(matches, CandidateClass::Human),
        best_llm_style: best_style(matches, CandidateClass::Llm),
    })
}

fn stable_sha256<T: Serialize>(value: &T) -> Result<String, String> {
    let bytes = serde_json::to_vec(value)
        .map_err(|error| format!("failed to serialize calibration artifact: {error}"))?;
    let mut hasher = Sha256::new();
    hasher.update(bytes);
    Ok(format!("{:x}", hasher.finalize()))
}

fn candidate_identities(
    candidates: &BTreeMap<String, NearestCandidate>,
) -> Result<Vec<CandidateIdentity>, String> {
    candidates
        .values()
        .map(|candidate| {
            Ok(CandidateIdentity {
                label: candidate.label.clone(),
                class: candidate.class,
                fingerprint_sha256: stable_sha256(&candidate.fingerprint)?,
                style_profile_sha256: candidate
                    .style_profile
                    .as_ref()
                    .map(stable_sha256)
                    .transpose()?,
            })
        })
        .collect()
}

fn validate_classifier(
    classifier: &CalibratedClassifier,
    language: Language,
    baseline: &CorpusProfile,
    candidates: &BTreeMap<String, NearestCandidate>,
) -> Result<(), String> {
    if classifier.language != language {
        return Err("classifier language does not match requested language".into());
    }
    if classifier.baseline_sha256 != stable_sha256(baseline)? {
        return Err("classifier baseline does not match --baseline profile".into());
    }
    let current = candidate_identities(candidates)?;
    if classifier.candidates.len() != current.len()
        || classifier
            .candidates
            .iter()
            .zip(&current)
            .any(|(left, right)| {
                left.label != right.label
                    || left.class != right.class
                    || left.fingerprint_sha256 != right.fingerprint_sha256
                    || left.style_profile_sha256 != right.style_profile_sha256
            })
    {
        return Err("classifier candidate artifacts do not match --candidates manifest".into());
    }
    Ok(())
}

fn round4(value: f64) -> f64 {
    (value * 10_000.0).round() / 10_000.0
}
