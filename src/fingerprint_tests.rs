use crate::Language;
use crate::fingerprint::*;
use crate::metadata::{DocumentMetadata, LoadedDocument};
use crate::profile::{ProfileOptions, profile_documents, profile_loaded_documents};

fn metadata_doc(id: &str, model: &str, family: &str, prompt: &str, text: &str) -> LoadedDocument {
    LoadedDocument {
        id: id.into(),
        text: text.into(),
        metadata: DocumentMetadata {
            model_id: Some(model.into()),
            family: Some(family.into()),
            prompt_id: Some(prompt.into()),
            domains: vec!["test".into()],
        },
    }
}

#[test]
fn consensus_requires_support_from_multiple_models() {
    let target_docs = vec![
        metadata_doc(
            "a1",
            "m1",
            "f1",
            "p1",
            "Formule spéciale unique formule spéciale unique.",
        ),
        metadata_doc(
            "a2",
            "m1",
            "f1",
            "p2",
            "Formule spéciale unique revient encore.",
        ),
        metadata_doc(
            "b1",
            "m2",
            "f2",
            "p3",
            "Formule spéciale commune formule spéciale commune.",
        ),
        metadata_doc(
            "b2",
            "m2",
            "f2",
            "p4",
            "Formule spéciale commune revient encore.",
        ),
        metadata_doc(
            "c1",
            "m3",
            "f3",
            "p5",
            "Formule spéciale commune formule spéciale commune.",
        ),
        metadata_doc(
            "c2",
            "m3",
            "f3",
            "p6",
            "Formule spéciale commune revient toujours.",
        ),
    ];
    let target = profile_loaded_documents(Language::Fr, &target_docs, ProfileOptions::default());
    let baseline = profile_documents(
        Language::Fr,
        &[
            ("h1".into(), "Texte humain ordinaire et distinct.".into()),
            ("h2".into(), "Autre texte humain ordinaire.".into()),
            ("h3".into(), "Encore une prose humaine simple.".into()),
        ],
    );
    let suite = build_fingerprint_suite(
        &target,
        &baseline,
        None,
        FingerprintOptions {
            min_documents: 2,
            min_model_documents: 2,
            min_models: 2,
            use_wordfreq: false,
            ..Default::default()
        },
        None,
    )
    .unwrap();
    assert_eq!(suite.models.len(), 3);
    assert!(
        suite
            .consensus
            .bigrams
            .iter()
            .any(|entry| { entry.pattern == "spéciale commune" && entry.model_frequency == 2 })
    );
    assert!(
        !suite
            .consensus
            .bigrams
            .iter()
            .any(|entry| entry.pattern == "spéciale unique")
    );
}

#[test]
fn recovered_phrase_keeps_stopwords_and_model_support() {
    let target_docs = vec![
        metadata_doc(
            "a1",
            "m1",
            "f1",
            "p1",
            "Il est important de noter et de vérifier ce point.",
        ),
        metadata_doc(
            "a2",
            "m1",
            "f1",
            "p2",
            "Il reste important de noter et de vérifier ce détail.",
        ),
        metadata_doc(
            "b1",
            "m2",
            "f2",
            "p3",
            "Il est important de noter et de vérifier ce fait.",
        ),
        metadata_doc(
            "b2",
            "m2",
            "f2",
            "p4",
            "Il paraît important de noter et de vérifier ce cas.",
        ),
    ];
    let target = profile_loaded_documents(
        Language::Fr,
        &target_docs,
        ProfileOptions {
            recover_phrases: true,
        },
    );
    let baseline = profile_documents(
        Language::Fr,
        &[
            ("h1".into(), "Texte humain ordinaire.".into()),
            ("h2".into(), "Autre prose naturelle.".into()),
            ("h3".into(), "Dernier texte humain.".into()),
        ],
    );
    let suite = build_fingerprint_suite(
        &target,
        &baseline,
        None,
        FingerprintOptions {
            min_documents: 2,
            min_model_documents: 2,
            min_models: 2,
            phrase_limit: 20,
            use_wordfreq: false,
            ..Default::default()
        },
        None,
    )
    .unwrap();
    let phrase = suite
        .consensus
        .phrases
        .iter()
        .find(|entry| entry.pattern == "important de noter et de vérifier")
        .unwrap();
    assert_eq!(
        phrase.discovery_pattern.as_deref(),
        Some("important noter vérifier")
    );
    assert_eq!(phrase.model_frequency, 2);
    assert_eq!(phrase.source, FingerprintSource::RecoveredPhrase);
}

#[test]
fn guard_filters_patterns_common_in_accepted_prose() {
    let target = profile_documents(
        Language::Fr,
        &[
            ("1".into(), "vous avez une idée claire".into()),
            ("2".into(), "vous avez une autre idée".into()),
            ("3".into(), "vous avez encore une idée".into()),
        ],
    );
    let baseline = profile_documents(
        Language::Fr,
        &[
            ("1".into(), "une idée claire existe".into()),
            ("2".into(), "une autre idée existe".into()),
            ("3".into(), "encore une idée existe".into()),
        ],
    );
    let guard = profile_documents(
        Language::Fr,
        &[
            ("1".into(), "vous avez une idée".into()),
            ("2".into(), "vous avez raison".into()),
            ("3".into(), "vous avez le choix".into()),
        ],
    );
    let fp = build_fingerprint_with_guard(
        &target,
        &baseline,
        Some(&guard),
        FingerprintOptions {
            min_documents: 3,
            min_models: 1,
            use_wordfreq: false,
            word_limit: 20,
            bigram_limit: 20,
            trigram_limit: 20,
            ..Default::default()
        },
    );
    assert!(!fp.bigrams.iter().any(|entry| entry.pattern == "vous avez"));
    assert_eq!(fp.guard_profile_documents, Some(3));
}
