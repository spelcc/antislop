pub mod analyze;
pub mod fingerprint;
pub mod lint;
pub mod metrics;
pub mod profile;
pub mod prose;
pub mod style;
pub mod tokenize;

pub use analyze::{Analysis, PatternHit, PatternSignalClass, analyze_text};
pub use fingerprint::{
    Fingerprint, FingerprintEntry, FingerprintOptions, build_fingerprint,
    build_fingerprint_with_guard,
};
pub use lint::{
    LintOutcome, LintThresholds, SentenceFinding, SentenceSpan, lint_text, split_sentences,
};
pub use profile::{CorpusProfile, DocumentProfile, profile_documents};
pub use prose::clean_prose;
pub use style::{
    StyleComparison, StyleDocumentMetrics, StyleMetricSummary, StyleProfile, build_style_profile,
    compare_style, style_metrics,
};
pub use tokenize::Language;
