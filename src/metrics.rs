use serde::{Deserialize, Serialize};
use std::collections::{HashMap, HashSet};

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct LexicalMetrics {
    pub tokens: usize,
    pub unique_tokens: usize,
    pub root_ttr: f64,
    pub mattr_500: f64,
    pub hdd_42: f64,
    pub distinct_1: f64,
    pub distinct_2: f64,
    pub distinct_3: f64,
}

pub fn lexical_metrics(tokens: &[String]) -> LexicalMetrics {
    let unique_tokens = tokens.iter().collect::<HashSet<_>>().len();
    LexicalMetrics {
        tokens: tokens.len(),
        unique_tokens,
        root_ttr: round4(if tokens.is_empty() {
            0.0
        } else {
            unique_tokens as f64 / (tokens.len() as f64).sqrt()
        }),
        mattr_500: round4(mattr(tokens, 500)),
        hdd_42: round4(hdd(tokens, 42)),
        distinct_1: round4(distinct_n(tokens, 1)),
        distinct_2: round4(distinct_n(tokens, 2)),
        distinct_3: round4(distinct_n(tokens, 3)),
    }
}

pub fn mattr(tokens: &[String], window: usize) -> f64 {
    if tokens.is_empty() || window == 0 {
        return 0.0;
    }
    let window = window.min(tokens.len());
    let windows = tokens.len() - window + 1;
    let sum: f64 = tokens
        .windows(window)
        .map(|slice| slice.iter().collect::<HashSet<_>>().len() as f64 / window as f64)
        .sum();
    sum / windows as f64
}

pub fn distinct_n(tokens: &[String], n: usize) -> f64 {
    if n == 0 || tokens.len() < n {
        return 0.0;
    }
    let total = tokens.len() - n + 1;
    let unique = tokens
        .windows(n)
        .map(|w| w.join("\u{1f}"))
        .collect::<HashSet<_>>()
        .len();
    unique as f64 / total as f64
}

pub fn hdd(tokens: &[String], sample_size: usize) -> f64 {
    if tokens.is_empty() || sample_size == 0 {
        return 0.0;
    }
    let n = tokens.len();
    let sample = sample_size.min(n);
    let mut counts: HashMap<&str, usize> = HashMap::new();
    for token in tokens {
        *counts.entry(token).or_default() += 1;
    }

    let denominator = log_choose(n, sample);
    let expected_types: f64 = counts
        .values()
        .map(|&frequency| {
            let missing = if n - frequency < sample {
                0.0
            } else {
                (log_choose(n - frequency, sample) - denominator).exp()
            };
            1.0 - missing
        })
        .sum();
    expected_types / sample as f64
}

fn log_choose(n: usize, k: usize) -> f64 {
    if k > n {
        return f64::NEG_INFINITY;
    }
    let k = k.min(n - k);
    (0..k)
        .map(|i| ((n - i) as f64).ln() - ((i + 1) as f64).ln())
        .sum()
}

fn round4(value: f64) -> f64 {
    (value * 10_000.0).round() / 10_000.0
}

#[cfg(test)]
mod tests {
    use super::*;

    fn words(input: &[&str]) -> Vec<String> {
        input.iter().map(|s| s.to_string()).collect()
    }

    #[test]
    fn distinct_is_one_for_unique_sequence() {
        let tokens = words(&["a", "b", "c", "d"]);
        assert_eq!(distinct_n(&tokens, 1), 1.0);
        assert_eq!(distinct_n(&tokens, 2), 1.0);
    }

    #[test]
    fn repeated_vocabulary_reduces_metrics() {
        let rich = words(&["a", "b", "c", "d", "e", "f"]);
        let flat = words(&["a", "a", "a", "a", "a", "a"]);
        assert!(mattr(&rich, 4) > mattr(&flat, 4));
        assert!(hdd(&rich, 4) > hdd(&flat, 4));
    }
}
