pub mod analyze;
pub mod cluster;
pub mod discovery;
pub mod fingerprint;
pub mod lexical;
pub mod lint;
pub mod metadata;
pub mod metrics;
pub mod profile;
pub mod prose;
pub mod structural;
pub mod style;
pub mod tokenize;

pub use analyze::{Analysis, PatternHit, PatternSignalClass, analyze_text};
pub use cluster::{ClusterReport, cluster_fingerprints, normalized_rank_distance, rank_features};
pub use fingerprint::{
    Fingerprint, FingerprintEntry, FingerprintOptions, FingerprintSource, FingerprintSuite,
    build_fingerprint, build_fingerprint_suite, build_fingerprint_with_guard,
};
pub use lint::{
    LintOutcome, LintThresholds, SentenceFinding, SentenceSpan, lint_text, split_sentences,
};
pub use metadata::{DocumentMetadata, LoadedDocument, load_manifest};
pub use profile::{
    CorpusProfile, DocumentProfile, ModelProfile, PatternStats, ProfileOptions, ProfileSlice,
    profile_documents, profile_loaded_documents,
};
pub use prose::clean_prose;
pub use style::{
    StyleComparison, StyleDocumentMetrics, StyleMetricSummary, StyleProfile, build_style_profile,
    compare_style, style_metrics,
};
pub use tokenize::Language;

#[cfg(test)]
mod fingerprint_tests;
