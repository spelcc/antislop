use crate::tokenize::Language;
use serde::{Deserialize, Serialize};
use std::fs;
use std::path::{Path, PathBuf};

#[derive(Debug, Clone, Default, Serialize, Deserialize, PartialEq, Eq)]
pub struct DocumentMetadata {
    #[serde(default, alias = "model")]
    pub model_id: Option<String>,
    #[serde(default)]
    pub family: Option<String>,
    #[serde(default, alias = "prompt_sha256")]
    pub prompt_id: Option<String>,
    #[serde(default, alias = "categories")]
    pub domains: Vec<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ManifestDocument {
    pub file: String,
    #[serde(flatten)]
    pub metadata: DocumentMetadata,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CorpusManifest {
    #[serde(default)]
    pub schema_version: Option<u32>,
    #[serde(default)]
    pub language: Option<Language>,
    pub documents: Vec<ManifestDocument>,
}

#[derive(Debug, Deserialize)]
#[serde(untagged)]
enum ManifestInput {
    Object(CorpusManifest),
    Array(Vec<ManifestDocument>),
}

#[derive(Debug, Clone)]
pub struct LoadedDocument {
    pub id: String,
    pub text: String,
    pub metadata: DocumentMetadata,
}

pub fn load_manifest(path: &Path, language: Language) -> Result<Vec<LoadedDocument>, String> {
    let raw = fs::read_to_string(path)
        .map_err(|error| format!("failed to read manifest {}: {error}", path.display()))?;
    let input: ManifestInput = serde_json::from_str(&raw)
        .map_err(|error| format!("invalid manifest {}: {error}", path.display()))?;
    let manifest = match input {
        ManifestInput::Object(manifest) => manifest,
        ManifestInput::Array(documents) => CorpusManifest {
            schema_version: None,
            language: None,
            documents,
        },
    };
    if let Some(manifest_language) = manifest.language
        && manifest_language != language
    {
        return Err(format!(
            "manifest language {:?} does not match --language {:?}",
            manifest_language, language
        ));
    }
    let base = path.parent().unwrap_or_else(|| Path::new("."));
    manifest
        .documents
        .into_iter()
        .map(|mut document| {
            if document.metadata.family.is_none() {
                document.metadata.family = document
                    .metadata
                    .model_id
                    .as_deref()
                    .map(infer_model_family);
            }
            let resolved = resolve_manifest_file(base, &document.file);
            let text = fs::read_to_string(&resolved).map_err(|error| {
                format!(
                    "failed to read manifest document {}: {error}",
                    resolved.display()
                )
            })?;
            Ok(LoadedDocument {
                id: document.file,
                text,
                metadata: document.metadata,
            })
        })
        .collect()
}

fn resolve_manifest_file(base: &Path, file: &str) -> PathBuf {
    let supplied = Path::new(file);
    if supplied.is_absolute() {
        return supplied.to_path_buf();
    }
    let direct = base.join(supplied);
    if direct.exists() {
        return direct;
    }
    let documents = base.join("documents").join(supplied);
    if documents.exists() {
        return documents;
    }
    direct
}

pub fn infer_model_family(model: &str) -> String {
    let normalized = model.to_ascii_lowercase();
    for (needle, family) in [
        ("claude", "claude"),
        ("gemini", "gemini"),
        ("gemma", "gemma"),
        ("deepseek", "deepseek"),
        ("qwen", "qwen"),
        ("mistral", "mistral"),
        ("mixtral", "mistral"),
        ("llama", "llama"),
        ("grok", "grok"),
        ("glm", "glm"),
        ("kimi", "kimi"),
        ("command", "command"),
        ("aya", "aya"),
        ("apertus", "apertus"),
        ("phi", "phi"),
        ("gpt-oss", "gpt-oss"),
        ("gpt", "gpt"),
    ] {
        if normalized.contains(needle) {
            return family.to_string();
        }
    }
    normalized
        .split(['/', ':'])
        .next()
        .unwrap_or(&normalized)
        .to_string()
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::tempdir;

    #[test]
    fn array_manifest_accepts_comparia_aliases_and_infers_family() {
        let dir = tempdir().unwrap();
        fs::create_dir(dir.path().join("documents")).unwrap();
        fs::write(
            dir.path().join("documents/a.txt"),
            "Texte humainement lisible.",
        )
        .unwrap();
        fs::write(
            dir.path().join("index.json"),
            r#"[{"file":"a.txt","model":"claude-4-5-sonnet","prompt_sha256":"p1","categories":["Culture"]}]"#,
        )
        .unwrap();
        let docs = load_manifest(&dir.path().join("index.json"), Language::Fr).unwrap();
        assert_eq!(docs.len(), 1);
        assert_eq!(
            docs[0].metadata.model_id.as_deref(),
            Some("claude-4-5-sonnet")
        );
        assert_eq!(docs[0].metadata.family.as_deref(), Some("claude"));
        assert_eq!(docs[0].metadata.prompt_id.as_deref(), Some("p1"));
    }
}
