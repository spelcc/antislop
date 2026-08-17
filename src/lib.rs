pub mod analyze;
pub mod fingerprint;
pub mod metrics;
pub mod profile;
pub mod tokenize;

pub use analyze::{Analysis, PatternHit, analyze_text};
pub use fingerprint::{Fingerprint, FingerprintEntry, build_fingerprint};
pub use profile::{CorpusProfile, DocumentProfile, profile_documents};
pub use tokenize::Language;
