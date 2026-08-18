use crate::tokenize::Language;
use wordfreq::WordFreq;
use wordfreq_model::{ModelKind, load_wordfreq};

pub struct LexicalReference {
    source: &'static str,
    model: WordFreq,
}

impl LexicalReference {
    pub fn load(language: Language) -> Result<Option<Self>, String> {
        match language {
            Language::Fr => load_wordfreq(ModelKind::LargeFr)
                .map(|model| {
                    Some(Self {
                        source: "wordfreq-large-fr-v3",
                        model,
                    })
                })
                .map_err(|error| format!("failed to load French wordfreq model: {error}")),
            Language::En => Ok(None),
        }
    }

    pub fn frequency(&self, word: &str) -> f64 {
        self.model.word_frequency(word).into()
    }

    pub fn source(&self) -> &'static str {
        self.source
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn french_reference_distinguishes_common_french_spelling() {
        let reference = LexicalReference::load(Language::Fr).unwrap().unwrap();
        assert!(reference.frequency("café") > reference.frequency("cafe"));
        assert!(reference.frequency("café") > 0.0);
    }
}
