use antislop::{ClassificationReport, LintOutcome, NearestReport, StyleComparison};

pub(crate) fn print_classification(report: &ClassificationReport) {
    let predicted = match report.predicted_class {
        antislop::CandidateClass::Human => "human",
        antislop::CandidateClass::Llm => "llm",
    };
    println!(
        "Classification: {predicted} | human {:.1}% | LLM {:.1}% | calibration prior LLM {:.0}%",
        report.human_probability * 100.0,
        report.llm_probability * 100.0,
        report.calibration_prior_llm * 100.0,
    );
    println!(
        "Calibration holdout: accuracy {:.1}% | AUC {:.4} | Brier {:.4} | ECE {:.4} | {} documents",
        report.calibration.accuracy * 100.0,
        report.calibration.roc_auc,
        report.calibration.brier_score,
        report.calibration.expected_calibration_error,
        report.calibration.documents,
    );
    println!("\nBest evidence by class:");
    println!(
        "  distance human: {} {:.4} (#{} global) | LLM: {} {:.4} (#{} global)",
        report.evidence.best_human_distance.label,
        report.evidence.best_human_distance.value,
        report.evidence.best_human_distance.global_rank,
        report.evidence.best_llm_distance.label,
        report.evidence.best_llm_distance.value,
        report.evidence.best_llm_distance.global_rank,
    );
    println!(
        "  signal   human: {} {:.4} (#{} global) | LLM: {} {:.4} (#{} global)",
        report.evidence.best_human_signal.label,
        report.evidence.best_human_signal.value,
        report.evidence.best_human_signal.global_rank,
        report.evidence.best_llm_signal.label,
        report.evidence.best_llm_signal.value,
        report.evidence.best_llm_signal.global_rank,
    );
    if let (Some(human), Some(llm)) = (
        &report.evidence.best_human_style,
        &report.evidence.best_llm_style,
    ) {
        println!(
            "  style    human: {} {:.4} (#{} global) | LLM: {} {:.4} (#{} global)",
            human.label, human.value, human.global_rank, llm.label, llm.value, llm.global_rank,
        );
    }
    println!("\nFeature contributions to LLM logit:");
    println!("  intercept: {:+.4}", report.intercept_contribution);
    for (name, contribution) in &report.feature_contributions_to_llm_logit {
        println!("  {name}: {contribution:+.4}");
    }
    println!("\nNearest candidates by rank distance:");
    for (index, item) in report.nearest.iter().enumerate() {
        let class = match item.class {
            antislop::CandidateClass::Human => "human",
            antislop::CandidateClass::Llm => "llm",
        };
        println!(
            "  {:>2}. {:<32} {:<5} distance {:.4} | signal {:.4} | style {}",
            index + 1,
            item.label,
            class,
            item.rank_distance,
            item.document_signal_per_1000_tokens,
            item.style_distance
                .map_or_else(|| "-".into(), |value| format!("{value:.4}")),
        );
    }
    println!(
        "\nProbability uses the supplied train/test calibration with a 50/50 human-vs-LLM prior; it is not a real-world prevalence estimate."
    );
}

pub(crate) fn print_classification_gate(report: &ClassificationReport, min_human: f64) {
    let passed = report.human_probability >= min_human;
    println!(
        "\nCI {}: Human probability {:.1}% (required >= {:.1}%)",
        if passed { "PASS" } else { "FAIL" },
        report.human_probability * 100.0,
        min_human * 100.0,
    );
    if passed {
        return;
    }

    println!("\nHow to pass:");
    println!(
        "  1. Raise Human probability to at least {:.1}% and rerun the exact classifier bundle.",
        min_human * 100.0
    );

    let mut llm_features: Vec<_> = report
        .feature_contributions_to_llm_logit
        .iter()
        .filter(|(_, contribution)| **contribution > 0.0)
        .collect();
    llm_features.sort_by(|left, right| right.1.total_cmp(left.1));
    for (index, (feature, contribution)) in llm_features.iter().take(3).enumerate() {
        let advice = match feature.as_str() {
            "top5_signal_llm_fraction" => {
                "rewrite recurring model-specific phrases so Human candidate signals re-enter the top five"
            }
            "style_margin_llm" => {
                "vary sentence rhythm, punctuation, function words and sentence openings toward the nearest Human style profile"
            }
            "top5_distance_llm_fraction" => {
                "reduce the concentration of globally LLM-like ranked n-grams across the document"
            }
            "signal_margin_llm" => {
                "remove or rephrase high-weight phrases that are over-represented in the strongest LLM fingerprint"
            }
            "distance_margin_llm" => {
                "move the document fingerprint closer to a Human population by replacing recurrent stock n-grams"
            }
            _ => "reduce this LLM-leaning feature",
        };
        println!(
            "  {}. {} ({:+.4} LLM logit): {}.",
            index + 2,
            feature,
            contribution,
            advice
        );
    }

    println!("\nPriority passages to rewrite:");
    if report.fixes.is_empty() {
        println!(
            "  No sentence-level LLM-over-Human phrase hotspot was isolated. Focus on the document-wide style/rank criteria above."
        );
    } else {
        for fix in report.fixes.iter().take(10) {
            let lines = if fix.start_line == fix.end_line {
                format!("L{}", fix.start_line)
            } else {
                format!("L{}-{}", fix.start_line, fix.end_line)
            };
            println!(
                "\n  {}. {} | local LLM {:.4} vs Human {:.4} | margin +{:.4}",
                fix.priority, lines, fix.llm_signal, fix.human_signal, fix.llm_signal_margin
            );
            println!("     {}", fix.text.replace('\n', " "));
            println!("     Bad phrases:");
            for pattern in &fix.patterns {
                println!(
                    "       - [{}] “{}” x{} | {}-gram | signal {:.4} | {} top LLM fingerprint(s): {}",
                    pattern.confidence,
                    pattern.pattern,
                    pattern.occurrences,
                    pattern.n,
                    pattern.aggregate_signal,
                    pattern.candidate_count,
                    pattern.candidates.join(", "),
                );
            }
            println!("     Fix: {}", fix.instruction);
        }
    }
    println!("\nStyle hotspots:");
    if report.style_fixes.is_empty() {
        println!(
            "  No line-level rhythm hotspot can be justified from the nearest Human style profile; keep the document-wide style criterion as context only."
        );
    } else {
        for fix in &report.style_fixes {
            let lines = if fix.start_line == fix.end_line {
                format!("L{}", fix.start_line)
            } else {
                format!("L{}-{}", fix.start_line, fix.end_line)
            };
            println!(
                "\n  {}. {} | {}: document {:.4}, Human median {:.4}",
                fix.priority, lines, fix.metric, fix.document_value, fix.human_median
            );
            println!("     {}", fix.text.replace('\n', " "));
            println!("     Fix: {}", fix.instruction);
        }
    }

    println!(
        "\nNearest Human references: distance={} (#{}), signal={} (#{}), style={}.",
        report.evidence.best_human_distance.label,
        report.evidence.best_human_distance.global_rank,
        report.evidence.best_human_signal.label,
        report.evidence.best_human_signal.global_rank,
        report
            .evidence
            .best_human_style
            .as_ref()
            .map(|value| value.label.as_str())
            .unwrap_or("n/a"),
    );
}

pub(crate) fn print_nearest(report: &NearestReport) {
    let metric = match report.metric.as_str() {
        "rank_distance" => "rank distance (lower is closer)",
        "document_signal" => "document signal (higher matches more candidate-specific patterns)",
        "style_distance" => "style distance (lower is closer)",
        _ => "unknown metric",
    };
    println!(
        "Nearest fingerprints | {} candidates | {metric}",
        report.candidate_count
    );
    println!(
        "{:<4} {:<32} {:<6} {:>9} {:>8} {:>9} {:>8} {:>9} {:>8}",
        "#", "candidate", "class", "distance", "d-rank", "signal", "s-rank", "style", "st-rank"
    );
    for (index, item) in report.matches.iter().enumerate() {
        let class = match item.class {
            antislop::CandidateClass::Human => "human",
            antislop::CandidateClass::Llm => "llm",
        };
        let style = item
            .style_distance
            .map_or_else(|| "-".to_string(), |value| format!("{value:.4}"));
        let style_rank = item
            .style_distance_position
            .map_or_else(|| "-".to_string(), |value| value.to_string());
        println!(
            "{:<4} {:<32} {:<6} {:>9.4} {:>8} {:>9.4} {:>8} {:>9} {:>8}",
            index + 1,
            item.model_id.as_deref().unwrap_or(&item.label),
            class,
            item.rank_distance,
            item.rank_distance_position,
            item.document_signal_per_1000_tokens,
            item.document_signal_position,
            style,
            style_rank,
        );
    }
    println!(
        "
Nearest fingerprint is descriptive similarity, not model attribution or authorship probability."
    );
}

pub(crate) fn print_style_comparison(comparison: &StyleComparison) {
    println!(
        "Style distance {:.4} | {:.1}% of metrics within the profile band | {} profile documents",
        comparison.overall_distance,
        comparison.within_profile_band_ratio * 100.0,
        comparison.profile_documents,
    );
    println!(
        "
Group distances (lower is closer):"
    );
    for (group, distance) in &comparison.groups {
        println!("  {group}: {distance:.4}");
    }
    if !comparison.top_deviations.is_empty() {
        println!(
            "
Largest deviations:"
        );
        for deviation in &comparison.top_deviations {
            let direction = if deviation.robust_z >= 0.0 {
                "above"
            } else {
                "below"
            };
            println!(
                "  {}: {:.4} vs median {:.4} ({direction}, z={:.2})",
                deviation.metric,
                deviation.value,
                deviation.profile_median,
                deviation.robust_z.abs(),
            );
        }
    }
    println!(
        "
Distance is descriptive, not an authorship probability."
    );
}

pub(crate) fn print_lint(outcome: &LintOutcome) {
    println!(
        "{} sentences, {} flagged | document signal {:.4}/1000 tokens",
        outcome.report.sentence_count,
        outcome.report.flagged_sentence_count,
        outcome.report.document.fingerprint_signal_per_1000_tokens
    );
    for finding in &outcome.report.findings {
        let lines = if finding.start_line == finding.end_line {
            format!("L{}", finding.start_line)
        } else {
            format!("L{}-{}", finding.start_line, finding.end_line)
        };
        println!("\n{lines}  {}", finding.text.replace('\n', " "));
        for hit in &finding.structural_hits {
            println!("  structure: {} x{}", hit.rule, hit.count);
        }
        for hit in &finding.fingerprint_hits {
            match hit.ratio {
                Some(ratio) => println!(
                    "  fingerprint: {:?} {} | {:.2}x baseline | signal {:.4}",
                    hit.n, hit.pattern, ratio, hit.weighted_signal
                ),
                None => println!(
                    "  fingerprint: {:?} {} | absent from baseline",
                    hit.n, hit.pattern
                ),
            }
        }
    }
    if outcome.report.findings.is_empty() {
        println!("No sentence-level findings.");
    }
    for violation in &outcome.violations {
        eprintln!(
            "CI threshold exceeded: {} = {} > {}",
            violation.metric, violation.actual, violation.limit
        );
    }
}
