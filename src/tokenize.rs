use clap::ValueEnum;
use serde::{Deserialize, Serialize};
use std::collections::HashSet;
use unicode_normalization::UnicodeNormalization;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, ValueEnum)]
#[serde(rename_all = "lowercase")]
pub enum Language {
    En,
    Fr,
}

pub fn normalize(text: &str) -> String {
    text.nfkc()
        .flat_map(char::to_lowercase)
        .map(|c| match c {
            '\u{2018}' | '\u{2019}' | '\u{02bc}' => '\'',
            '\u{201c}' | '\u{201d}' => '"',
            _ => c,
        })
        .collect()
}

pub fn tokenize(text: &str) -> Vec<String> {
    let normalized = normalize(text);
    let mut tokens = Vec::new();
    let mut current = String::new();

    for ch in normalized.chars() {
        if ch.is_alphabetic() || (ch == '\'' && !current.is_empty()) {
            current.push(ch);
        } else if !current.is_empty() {
            let token = current.trim_matches('\'').to_owned();
            if !token.is_empty() {
                tokens.push(token);
            }
            current.clear();
        }
    }

    if !current.is_empty() {
        let token = current.trim_matches('\'').to_owned();
        if !token.is_empty() {
            tokens.push(token);
        }
    }

    tokens
}

pub fn content_tokens(text: &str, language: Language) -> Vec<String> {
    let stops = stopwords(language);
    tokenize(text)
        .into_iter()
        .map(|token| normalize_content_token(token, language))
        .filter(|token| !token.is_empty() && !stops.contains(token.as_str()))
        .collect()
}

fn normalize_content_token(token: String, language: Language) -> String {
    if language != Language::Fr {
        return token;
    }
    for prefix in ["l'", "d'", "j'", "qu'", "n'", "s'", "c'", "t'", "m'"] {
        if let Some(rest) = token.strip_prefix(prefix)
            && !rest.is_empty()
        {
            return rest.to_string();
        }
    }
    token
}

fn stopwords(language: Language) -> HashSet<&'static str> {
    let words: &[&str] = match language {
        Language::En => &[
            "a", "an", "and", "are", "as", "at", "be", "been", "but", "by", "for", "from", "had",
            "has", "have", "he", "her", "his", "i", "if", "in", "is", "it", "its", "me", "my",
            "not", "of", "on", "or", "our", "she", "so", "that", "the", "their", "them", "they",
            "this", "to", "was", "we", "were", "which", "who", "will", "with", "you", "your",
        ],
        Language::Fr => &[
            "a", "ai", "au", "aux", "avec", "ce", "ces", "cette", "dans", "de", "des", "du",
            "elle", "en", "est", "et", "eux", "il", "ils", "je", "la", "le", "les", "leur", "lui",
            "mais", "me", "mes", "moi", "mon", "ne", "nos", "notre", "nous", "on", "ou", "par",
            "pas", "pour", "que", "qui", "sa", "se", "ses", "son", "sur", "ta", "te", "tes", "toi",
            "ton", "tu", "un", "une", "vos", "votre", "vous", "y",
        ],
    };
    words.iter().copied().collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn tokenizes_unicode_and_apostrophes() {
        assert_eq!(
            tokenize("L’IA casse déjà l'écran."),
            vec!["l'ia", "casse", "déjà", "l'écran"]
        );
    }

    #[test]
    fn strips_french_clitics() {
        assert_eq!(
            content_tokens("L’IA n'est pas l'objet", Language::Fr),
            vec!["ia", "objet"]
        );
    }

    #[test]
    fn filters_stopwords() {
        assert_eq!(
            content_tokens("Le robot est dans le garage", Language::Fr),
            vec!["robot", "garage"]
        );
    }
}
