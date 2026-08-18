mod cli_output;

use antislop::{
    CalibratedClassifier, FingerprintOptions, Language, LintThresholds, NearestMetric,
    ProfileOptions, analyze_text, build_fingerprint_suite, build_style_profile,
    calibrate_classifier, classify_text, clean_prose, cluster_fingerprints, compare_style,
    lint_text, load_calibration_manifest, load_candidate_manifest, load_fingerprint_directory,
    load_manifest, nearest_candidates, profile_documents, profile_loaded_documents,
};
use clap::{Parser, Subcommand, ValueEnum};
use cli_output::{print_classification, print_lint, print_nearest, print_style_comparison};
use serde::Serialize;
use std::collections::BTreeMap;
use std::fs;
use std::io::{self, Read};
use std::path::{Path, PathBuf};

#[derive(Parser)]
#[command(
    name = "antislop",
    version,
    about = "Deterministic writing analysis and corpus fingerprinting"
)]
struct Cli {
    #[command(subcommand)]
    command: Command,
}

#[derive(Subcommand)]
enum Command {
    /// Analyze one text from a file or stdin.
    Analyze {
        #[arg(value_name = "FILE")]
        input: Option<PathBuf>,
        #[arg(long, value_enum, default_value = "en")]
        language: Language,
        #[arg(long)]
        fingerprint: Option<PathBuf>,
    },
    /// Localize deterministic findings sentence by sentence, with optional CI thresholds.
    Lint {
        #[arg(value_name = "FILE")]
        input: Option<PathBuf>,
        #[arg(long, value_enum, default_value = "en")]
        language: Language,
        #[arg(long)]
        fingerprint: Option<PathBuf>,
        #[arg(long)]
        json: bool,
        #[arg(long)]
        max_document_signal: Option<f64>,
        #[arg(long)]
        max_sentence_signal: Option<f64>,
        #[arg(long)]
        max_structural_hits: Option<usize>,
        #[arg(long)]
        max_flagged_sentences: Option<usize>,
    },
    /// Fit a balanced human-vs-LLM logistic classifier from labeled train/test documents.
    Calibrate {
        /// Labeled train/test document manifest.
        #[arg(long)]
        documents: PathBuf,
        /// Train-only human reference profile used by every candidate fingerprint.
        #[arg(long)]
        baseline: PathBuf,
        /// Human/LLM population candidate manifest.
        #[arg(long)]
        candidates: PathBuf,
        #[arg(long, value_enum, default_value = "en")]
        language: Language,
        #[arg(long, default_value_t = 2000)]
        epochs: usize,
        #[arg(long, default_value_t = 0.01)]
        l2: f64,
        #[arg(short, long)]
        output: PathBuf,
    },
    /// Classify one document with a previously calibrated human-vs-LLM model.
    Classify {
        #[arg(value_name = "FILE")]
        input: Option<PathBuf>,
        /// Classifier JSON produced by `antislop calibrate`.
        #[arg(long)]
        classifier: PathBuf,
        /// Exact baseline profile used during calibration.
        #[arg(long)]
        baseline: PathBuf,
        /// Exact candidate manifest/artifacts used during calibration.
        #[arg(long)]
        candidates: PathBuf,
        #[arg(long, value_enum, default_value = "en")]
        language: Language,
        #[arg(long, default_value_t = 10)]
        top: usize,
        /// Fail with exit code 2 when Human probability is below this threshold.
        #[arg(long)]
        min_human_probability: Option<f64>,
        /// CI mode: defaults the Human threshold to 0.70 and prints correction guidance.
        #[arg(long)]
        ci: bool,
        #[arg(long)]
        json: bool,
    },
    /// Build a deterministic corpus profile from files or a directory.
    Profile {
        /// Files/directories to profile. Omit when using --manifest.
        inputs: Vec<PathBuf>,
        /// JSON manifest carrying model_id/family/prompt_id/domain metadata.
        #[arg(long)]
        manifest: Option<PathBuf>,
        /// Store exact phrase variants for stopword-stripped discovery n-grams.
        #[arg(long)]
        recover_phrases: bool,
        #[arg(long, value_enum, default_value = "en")]
        language: Language,
        #[arg(short, long)]
        output: PathBuf,
    },
    /// Build or use a topic-light stylistic profile.
    Style {
        #[command(subcommand)]
        command: StyleCommand,
    },
    /// Rank which stored human/LLM population fingerprints are closest to one document.
    Nearest {
        #[arg(value_name = "FILE")]
        input: Option<PathBuf>,
        /// Human/baseline corpus profile used to build the document fingerprint.
        #[arg(long)]
        baseline: PathBuf,
        /// Directory containing one fingerprint JSON per model (legacy all-LLM mode).
        #[arg(long)]
        models_dir: Option<PathBuf>,
        /// Manifest mixing human and LLM candidate fingerprints, with optional style profiles.
        #[arg(long)]
        candidates: Option<PathBuf>,
        /// Optional accepted-prose guard profile used when fingerprinting the document.
        #[arg(long)]
        guard: Option<PathBuf>,
        #[arg(long, value_enum, default_value = "rank-distance")]
        metric: NearestCliMetric,
        #[arg(long, default_value_t = 10)]
        top: usize,
        #[arg(long, value_enum, default_value = "en")]
        language: Language,
        #[arg(long)]
        json: bool,
    },
    /// Compare ranked fingerprints and build an average-linkage cluster tree.
    Cluster {
        #[arg(required = true)]
        fingerprints: Vec<PathBuf>,
        #[arg(short, long)]
        output: PathBuf,
        #[arg(long)]
        newick_output: Option<PathBuf>,
    },
    /// Compare a target corpus profile against a human/baseline profile.
    Fingerprint {
        #[arg(long)]
        target: PathBuf,
        #[arg(long)]
        baseline: PathBuf,
        /// Optional accepted-prose profile. Patterns common here are filtered from the fingerprint.
        #[arg(long)]
        guard: Option<PathBuf>,
        /// Minimum target/guard frequency ratio for guarded patterns.
        #[arg(long, default_value_t = 2.0)]
        min_guard_ratio: f64,
        /// Minimum independent prompts/documents for a pattern inside one model.
        #[arg(long, default_value_t = 2)]
        min_model_documents: usize,
        /// Minimum number of distinct models supporting a consensus pattern.
        #[arg(long, default_value_t = 2)]
        min_models: usize,
        /// Maximum recovered long phrases in the consensus fingerprint.
        #[arg(long, default_value_t = 100)]
        phrase_limit: usize,
        /// Minimum target/wordfreq ratio for lexical entries.
        #[arg(long, default_value_t = 3.0)]
        min_wordfreq_ratio: f64,
        /// Disable the French wordfreq lexical reference channel.
        #[arg(long)]
        no_wordfreq: bool,
        /// Optional directory for one fingerprint JSON per target model.
        #[arg(long)]
        models_output_dir: Option<PathBuf>,
        /// Optional target label used by rank-distance/clustering.
        #[arg(long)]
        label: Option<String>,
        #[arg(short, long)]
        output: PathBuf,
        #[arg(long, default_value_t = 3)]
        min_documents: usize,
        #[arg(long, default_value_t = 120)]
        word_limit: usize,
        #[arg(long, default_value_t = 40)]
        bigram_limit: usize,
        #[arg(long, default_value_t = 40)]
        trigram_limit: usize,
    },
}

#[derive(Debug, Clone, Copy, ValueEnum)]
enum NearestCliMetric {
    RankDistance,
    DocumentSignal,
    StyleDistance,
}

impl From<NearestCliMetric> for NearestMetric {
    fn from(value: NearestCliMetric) -> Self {
        match value {
            NearestCliMetric::RankDistance => NearestMetric::RankDistance,
            NearestCliMetric::DocumentSignal => NearestMetric::DocumentSignal,
            NearestCliMetric::StyleDistance => NearestMetric::StyleDistance,
        }
    }
}

#[derive(Subcommand)]
enum StyleCommand {
    /// Build a stylistic profile from one document per file.
    Profile {
        #[arg(required = true)]
        inputs: Vec<PathBuf>,
        #[arg(long, value_enum, default_value = "en")]
        language: Language,
        #[arg(short, long)]
        output: PathBuf,
    },
    /// Compare one document with a previously built stylistic profile.
    Compare {
        #[arg(value_name = "FILE")]
        input: Option<PathBuf>,
        #[arg(long)]
        profile: PathBuf,
        #[arg(long)]
        json: bool,
    },
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    match Cli::parse().command {
        Command::Analyze {
            input,
            language,
            fingerprint,
        } => {
            let text = read_text(input.as_deref())?;
            let fp: Option<antislop::Fingerprint> =
                fingerprint.as_deref().map(read_json).transpose()?;
            if let Some(fp) = &fp
                && fp.language != language
            {
                return Err("fingerprint language does not match --language".into());
            }
            print_json(&analyze_text(&text, language, fp.as_ref()))?;
        }
        Command::Lint {
            input,
            language,
            fingerprint,
            json,
            max_document_signal,
            max_sentence_signal,
            max_structural_hits,
            max_flagged_sentences,
        } => {
            let text = read_text(input.as_deref())?;
            let fp: Option<antislop::Fingerprint> =
                fingerprint.as_deref().map(read_json).transpose()?;
            if let Some(fp) = &fp
                && fp.language != language
            {
                return Err("fingerprint language does not match --language".into());
            }
            let thresholds = LintThresholds {
                max_document_signal,
                max_sentence_signal,
                max_structural_hits,
                max_flagged_sentences,
            };
            let outcome = lint_text(&text, language, fp.as_ref(), &thresholds);
            if json {
                print_json(&outcome)?;
            } else {
                print_lint(&outcome);
            }
            if !outcome.passed {
                std::process::exit(2);
            }
        }
        Command::Calibrate {
            documents,
            baseline,
            candidates,
            language,
            epochs,
            l2,
            output,
        } => {
            let documents =
                load_calibration_manifest(&documents, language).map_err(io::Error::other)?;
            let baseline: antislop::CorpusProfile = read_json(&baseline)?;
            let candidates =
                load_candidate_manifest(&candidates, language).map_err(io::Error::other)?;
            let report =
                calibrate_classifier(&documents, language, &baseline, &candidates, epochs, l2)
                    .map_err(io::Error::other)?;
            write_json(&output, &report.classifier)?;
            print_json(&report)?;
        }
        Command::Classify {
            input,
            classifier,
            baseline,
            candidates,
            language,
            top,
            min_human_probability,
            ci,
            json,
        } => {
            if top == 0 {
                return Err("--top must be positive".into());
            }
            let raw = read_text(input.as_deref())?;
            let text = antislop::clean_prose_preserving_lines(&raw);
            if text.trim().is_empty() {
                return Err("classify input contains no prose after cleaning".into());
            }
            let classifier: CalibratedClassifier = read_json(&classifier)?;
            let baseline: antislop::CorpusProfile = read_json(&baseline)?;
            let candidates =
                load_candidate_manifest(&candidates, language).map_err(io::Error::other)?;
            let report = classify_text(&text, language, &baseline, &candidates, &classifier, top)
                .map_err(io::Error::other)?;
            let threshold = min_human_probability.or(ci.then_some(0.70));
            if let Some(limit) = threshold
                && !(0.0..=1.0).contains(&limit)
            {
                return Err("--min-human-probability must be between 0 and 1".into());
            }
            if json {
                print_json(&report)?;
            } else {
                print_classification(&report);
                if let Some(limit) = threshold {
                    cli_output::print_classification_gate(&report, limit);
                }
            }
            if threshold.is_some_and(|limit| report.human_probability < limit) {
                std::process::exit(2);
            }
        }
        Command::Profile {
            inputs,
            manifest,
            recover_phrases,
            language,
            output,
        } => {
            if manifest.is_none() && inputs.is_empty() {
                return Err("profile requires input files/directories or --manifest".into());
            }
            if manifest.is_some() && !inputs.is_empty() {
                return Err(
                    "profile accepts either positional inputs or --manifest, not both".into(),
                );
            }
            let profile = if let Some(manifest) = manifest {
                let documents = load_manifest(&manifest, language).map_err(io::Error::other)?;
                profile_loaded_documents(language, &documents, ProfileOptions { recover_phrases })
            } else if recover_phrases {
                let files = collect_files(&inputs)?;
                let documents: Vec<_> = files
                    .iter()
                    .map(|path| {
                        Ok(antislop::LoadedDocument {
                            id: path.display().to_string(),
                            text: fs::read_to_string(path)?,
                            metadata: antislop::DocumentMetadata::default(),
                        })
                    })
                    .collect::<Result<_, io::Error>>()?;
                profile_loaded_documents(
                    language,
                    &documents,
                    ProfileOptions {
                        recover_phrases: true,
                    },
                )
            } else {
                let files = collect_files(&inputs)?;
                let documents: Vec<_> = files
                    .iter()
                    .map(|path| Ok((path.display().to_string(), fs::read_to_string(path)?)))
                    .collect::<Result<_, io::Error>>()?;
                profile_documents(language, &documents)
            };
            write_json(&output, &profile)?;
        }
        Command::Style { command } => match command {
            StyleCommand::Profile {
                inputs,
                language,
                output,
            } => {
                let files = collect_style_files(&inputs)?;
                let documents: Vec<_> = files
                    .iter()
                    .map(|path| {
                        let raw = fs::read_to_string(path)?;
                        Ok((path.display().to_string(), clean_prose(&raw)))
                    })
                    .collect::<Result<_, io::Error>>()?;
                let profile =
                    build_style_profile(language, &documents).map_err(io::Error::other)?;
                write_json(&output, &profile)?;
            }
            StyleCommand::Compare {
                input,
                profile,
                json,
            } => {
                let raw = read_text(input.as_deref())?;
                let text = clean_prose(&raw);
                let profile: antislop::StyleProfile = read_json(&profile)?;
                let comparison = compare_style(&text, &profile);
                if json {
                    print_json(&comparison)?;
                } else {
                    print_style_comparison(&comparison);
                }
            }
        },
        Command::Nearest {
            input,
            baseline,
            models_dir,
            candidates,
            guard,
            metric,
            top,
            language,
            json,
        } => {
            if top == 0 {
                return Err("--top must be positive".into());
            }
            let raw = read_text(input.as_deref())?;
            let text = clean_prose(&raw);
            if text.trim().is_empty() {
                return Err("nearest input contains no prose after cleaning".into());
            }
            let baseline: antislop::CorpusProfile = read_json(&baseline)?;
            let guard: Option<antislop::CorpusProfile> =
                guard.as_deref().map(read_json).transpose()?;
            let candidate_set = match (models_dir, candidates) {
                (Some(directory), None) => {
                    load_fingerprint_directory(&directory).map_err(io::Error::other)?
                }
                (None, Some(manifest)) => {
                    load_candidate_manifest(&manifest, language).map_err(io::Error::other)?
                }
                (Some(_), Some(_)) => {
                    return Err(
                        "nearest accepts either --models-dir or --candidates, not both".into(),
                    );
                }
                (None, None) => {
                    return Err("nearest requires --models-dir or --candidates".into());
                }
            };
            let report = nearest_candidates(
                &text,
                language,
                &baseline,
                guard.as_ref(),
                &candidate_set,
                metric.into(),
                top,
            )
            .map_err(io::Error::other)?;
            if json {
                print_json(&report)?;
            } else {
                print_nearest(&report);
            }
        }
        Command::Cluster {
            fingerprints,
            output,
            newick_output,
        } => {
            let mut items = BTreeMap::new();
            for path in fingerprints {
                let fingerprint: antislop::Fingerprint = read_json(&path)?;
                let label = fingerprint
                    .target_label
                    .clone()
                    .or_else(|| fingerprint.target_model_id.clone())
                    .unwrap_or_else(|| {
                        path.file_stem()
                            .and_then(|value| value.to_str())
                            .unwrap_or("fingerprint")
                            .to_string()
                    });
                if items.insert(label.clone(), fingerprint).is_some() {
                    return Err(format!("duplicate cluster label: {label}").into());
                }
            }
            if items.len() < 2 {
                return Err("cluster requires at least two fingerprints".into());
            }
            let report = cluster_fingerprints(&items);
            write_json(&output, &report)?;
            if let Some(path) = newick_output {
                fs::write(
                    path,
                    format!(
                        "{}
",
                        report.newick
                    ),
                )?;
            }
        }
        Command::Fingerprint {
            target,
            baseline,
            guard,
            min_guard_ratio,
            min_model_documents,
            min_models,
            phrase_limit,
            min_wordfreq_ratio,
            no_wordfreq,
            models_output_dir,
            label,
            output,
            min_documents,
            word_limit,
            bigram_limit,
            trigram_limit,
        } => {
            let target: antislop::CorpusProfile = read_json(&target)?;
            let baseline: antislop::CorpusProfile = read_json(&baseline)?;
            let guard: Option<antislop::CorpusProfile> =
                guard.as_deref().map(read_json).transpose()?;
            if min_guard_ratio <= 1.0 {
                return Err("--min-guard-ratio must be greater than 1".into());
            }
            if min_models == 0 || min_model_documents == 0 {
                return Err("--min-models and --min-model-documents must be positive".into());
            }
            let suite = build_fingerprint_suite(
                &target,
                &baseline,
                guard.as_ref(),
                FingerprintOptions {
                    min_documents,
                    min_model_documents,
                    min_models,
                    word_limit,
                    bigram_limit,
                    trigram_limit,
                    phrase_limit,
                    min_guard_ratio,
                    min_wordfreq_ratio,
                    use_wordfreq: !no_wordfreq,
                },
                label,
            )
            .map_err(io::Error::other)?;
            write_json(&output, &suite.consensus)?;
            if let Some(directory) = models_output_dir {
                fs::create_dir_all(&directory)?;
                for (model_id, fingerprint) in &suite.models {
                    let filename = format!("{}.json", safe_filename(model_id));
                    write_json(&directory.join(filename), fingerprint)?;
                }
            }
        }
    }
    Ok(())
}

fn safe_filename(value: &str) -> String {
    value
        .chars()
        .map(|ch| {
            if ch.is_ascii_alphanumeric() || matches!(ch, '-' | '_' | '.') {
                ch
            } else {
                '-'
            }
        })
        .collect()
}

fn read_text(path: Option<&Path>) -> io::Result<String> {
    match path {
        Some(path) => fs::read_to_string(path),
        None => {
            let mut text = String::new();
            io::stdin().read_to_string(&mut text)?;
            Ok(text)
        }
    }
}

fn collect_style_files(inputs: &[PathBuf]) -> io::Result<Vec<PathBuf>> {
    let files = collect_files(inputs)?;
    let supported: Vec<_> = files
        .into_iter()
        .filter(|path| {
            matches!(
                path.extension()
                    .and_then(|value| value.to_str())
                    .map(str::to_ascii_lowercase)
                    .as_deref(),
                Some("txt" | "md" | "markdown" | "mdoc")
            )
        })
        .collect();
    if supported.is_empty() {
        return Err(io::Error::new(
            io::ErrorKind::InvalidInput,
            "style profile inputs contain no .txt/.md/.markdown/.mdoc documents",
        ));
    }
    Ok(supported)
}

fn collect_files(inputs: &[PathBuf]) -> io::Result<Vec<PathBuf>> {
    let mut files = Vec::new();
    for input in inputs {
        if input.is_file() {
            files.push(input.clone());
        } else if input.is_dir() {
            collect_directory(input, &mut files)?;
        } else {
            return Err(io::Error::new(
                io::ErrorKind::NotFound,
                format!("input not found: {}", input.display()),
            ));
        }
    }
    files.sort();
    files.dedup();
    Ok(files)
}

fn collect_directory(directory: &Path, files: &mut Vec<PathBuf>) -> io::Result<()> {
    for entry in fs::read_dir(directory)? {
        let path = entry?.path();
        if path.is_dir() {
            collect_directory(&path, files)?;
        } else if path.is_file() {
            files.push(path);
        }
    }
    Ok(())
}

fn read_json<T: serde::de::DeserializeOwned>(path: &Path) -> Result<T, Box<dyn std::error::Error>> {
    Ok(serde_json::from_str(&fs::read_to_string(path)?)?)
}

fn write_json(path: &Path, value: &impl Serialize) -> Result<(), Box<dyn std::error::Error>> {
    fs::write(path, format!("{}\n", serde_json::to_string_pretty(value)?))?;
    Ok(())
}

fn print_json(value: &impl Serialize) -> Result<(), serde_json::Error> {
    println!("{}", serde_json::to_string_pretty(value)?);
    Ok(())
}
