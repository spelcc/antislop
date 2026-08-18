use antislop::{
    FingerprintOptions, Language, LintOutcome, LintThresholds, StyleComparison, analyze_text,
    build_fingerprint_with_guard, build_style_profile, clean_prose, compare_style, lint_text,
    profile_documents,
};
use clap::{Parser, Subcommand};
use serde::Serialize;
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
    /// Build a deterministic corpus profile from files or a directory.
    Profile {
        #[arg(required = true)]
        inputs: Vec<PathBuf>,
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
        Command::Profile {
            inputs,
            language,
            output,
        } => {
            let files = collect_files(&inputs)?;
            let documents: Vec<_> = files
                .iter()
                .map(|path| Ok((path.display().to_string(), fs::read_to_string(path)?)))
                .collect::<Result<_, io::Error>>()?;
            write_json(&output, &profile_documents(language, &documents))?;
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
        Command::Fingerprint {
            target,
            baseline,
            guard,
            min_guard_ratio,
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
            if target.language != baseline.language {
                return Err("target and baseline profile languages do not match".into());
            }
            if target.schema_version != baseline.schema_version {
                return Err("target and baseline profile schemas do not match; rebuild both profiles with the same antislop version".into());
            }
            if let Some(guard) = &guard {
                if guard.language != target.language {
                    return Err("guard profile language does not match target".into());
                }
                if guard.schema_version != target.schema_version {
                    return Err("guard and target profile schemas do not match; rebuild both profiles with the same antislop version".into());
                }
                if min_guard_ratio <= 1.0 {
                    return Err("--min-guard-ratio must be greater than 1".into());
                }
            }
            write_json(
                &output,
                &build_fingerprint_with_guard(
                    &target,
                    &baseline,
                    guard.as_ref(),
                    FingerprintOptions {
                        min_documents,
                        word_limit,
                        bigram_limit,
                        trigram_limit,
                        min_guard_ratio,
                    },
                ),
            )?;
        }
    }
    Ok(())
}

fn print_style_comparison(comparison: &StyleComparison) {
    println!(
        "Style distance {:.4} | {:.1}% of metrics within the profile band | {} profile documents",
        comparison.overall_distance,
        comparison.within_profile_band_ratio * 100.0,
        comparison.profile_documents,
    );
    println!(
        "
Group distances (lower is closer):"
    );
    for (group, distance) in &comparison.groups {
        println!("  {group}: {distance:.4}");
    }
    if !comparison.top_deviations.is_empty() {
        println!(
            "
Largest deviations:"
        );
        for deviation in &comparison.top_deviations {
            let direction = if deviation.robust_z >= 0.0 {
                "above"
            } else {
                "below"
            };
            println!(
                "  {}: {:.4} vs median {:.4} ({direction}, z={:.2})",
                deviation.metric,
                deviation.value,
                deviation.profile_median,
                deviation.robust_z.abs(),
            );
        }
    }
    println!(
        "
Distance is descriptive, not an authorship probability."
    );
}

fn print_lint(outcome: &LintOutcome) {
    println!(
        "{} sentences, {} flagged | document signal {:.4}/1000 tokens",
        outcome.report.sentence_count,
        outcome.report.flagged_sentence_count,
        outcome.report.document.fingerprint_signal_per_1000_tokens
    );
    for finding in &outcome.report.findings {
        let lines = if finding.start_line == finding.end_line {
            format!("L{}", finding.start_line)
        } else {
            format!("L{}-{}", finding.start_line, finding.end_line)
        };
        println!("\n{lines}  {}", finding.text.replace('\n', " "));
        for hit in &finding.structural_hits {
            println!("  structure: {} x{}", hit.rule, hit.count);
        }
        for hit in &finding.fingerprint_hits {
            match hit.ratio {
                Some(ratio) => println!(
                    "  fingerprint: {:?} {} | {:.2}x baseline | signal {:.4}",
                    hit.n, hit.pattern, ratio, hit.weighted_signal
                ),
                None => println!(
                    "  fingerprint: {:?} {} | absent from baseline",
                    hit.n, hit.pattern
                ),
            }
        }
    }
    if outcome.report.findings.is_empty() {
        println!("No sentence-level findings.");
    }
    for violation in &outcome.violations {
        eprintln!(
            "CI threshold exceeded: {} = {} > {}",
            violation.metric, violation.actual, violation.limit
        );
    }
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
