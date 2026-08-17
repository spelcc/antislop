use antislop::{Language, analyze_text, build_fingerprint, profile_documents};
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
    /// Build a deterministic corpus profile from files or a directory.
    Profile {
        #[arg(required = true)]
        inputs: Vec<PathBuf>,
        #[arg(long, value_enum, default_value = "en")]
        language: Language,
        #[arg(short, long)]
        output: PathBuf,
    },
    /// Compare a target corpus profile against a human/baseline profile.
    Fingerprint {
        #[arg(long)]
        target: PathBuf,
        #[arg(long)]
        baseline: PathBuf,
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
        Command::Fingerprint {
            target,
            baseline,
            output,
            min_documents,
            word_limit,
            bigram_limit,
            trigram_limit,
        } => {
            let target: antislop::CorpusProfile = read_json(&target)?;
            let baseline: antislop::CorpusProfile = read_json(&baseline)?;
            if target.language != baseline.language {
                return Err("target and baseline profile languages do not match".into());
            }
            write_json(
                &output,
                &build_fingerprint(
                    &target,
                    &baseline,
                    min_documents,
                    word_limit,
                    bigram_limit,
                    trigram_limit,
                ),
            )?;
        }
    }
    Ok(())
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
