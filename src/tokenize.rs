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
    let stops = stopword_set(language);
    tokenize(text)
        .into_iter()
        .map(|token| normalize_content_word(token, language))
        .filter(|token| !token.is_empty() && !stops.contains(token.as_str()))
        .collect()
}

pub(crate) fn normalize_content_word(token: String, language: Language) -> String {
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

pub(crate) fn stopword_set(language: Language) -> HashSet<&'static str> {
    let words: &[&str] = match language {
        Language::En => &[
            "a", "an", "and", "are", "as", "at", "be", "been", "but", "by", "for", "from", "had",
            "has", "have", "he", "her", "his", "i", "if", "in", "is", "it", "its", "me", "my",
            "not", "of", "on", "or", "our", "she", "so", "that", "the", "their", "them", "they",
            "this", "to", "was", "we", "were", "which", "who", "will", "with", "you", "your",
        ],
        Language::Fr => &[
            "a", "ai", "ainsi", "après", "au", "aucun", "aucune", "aux", "avant", "avec", "à",
            "ce", "ces", "cet", "cette", "chez", "comme", "dans", "de", "des", "donc", "dont",
            "du", "elle", "elles", "en", "encore", "est", "et", "eux", "fait", "faire", "il",
            "ils", "je", "la", "le", "les", "leur", "leurs", "lui", "là", "mais", "me", "mes",
            "moi", "mon", "même", "ne", "nos", "notre", "nous", "on", "ont", "ou", "où", "par",
            "pas", "plus", "pour", "quand", "que", "quel", "quelle", "quelles", "quels", "qui",
            "sa", "sans", "se", "ses", "si", "son", "sont", "sous", "sur", "ta", "te", "tes",
            "toi", "ton", "tous", "tout", "toute", "toutes", "très", "tu", "un", "une", "vos",
            "votre", "vous", "y", "été", "être",
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
