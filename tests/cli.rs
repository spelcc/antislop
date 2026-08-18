use std::fs;
use std::io::Write;
use std::process::{Command, Stdio};
use tempfile::tempdir;

#[test]
fn analyze_stdin_emits_json() {
    let mut child = Command::new(env!("CARGO_BIN_EXE_antislop"))
        .args(["analyze", "--language", "fr"])
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .spawn()
        .unwrap();
    child
        .stdin
        .as_mut()
        .unwrap()
        .write_all("Un robot casse un robot.".as_bytes())
        .unwrap();
    let output = child.wait_with_output().unwrap();
    assert!(output.status.success());
    let value: serde_json::Value = serde_json::from_slice(&output.stdout).unwrap();
    assert_eq!(value["language"], "fr");
    assert_eq!(value["lexical"]["tokens"], 5);
}

#[test]
fn profile_then_fingerprint_then_analyze() {
    let dir = tempdir().unwrap();
    let target_dir = dir.path().join("target-corpus");
    let base_dir = dir.path().join("base-corpus");
    fs::create_dir(&target_dir).unwrap();
    fs::create_dir(&base_dir).unwrap();
    for i in 0..3 {
        fs::write(
            target_dir.join(format!("{i}.txt")),
            "delve deeply into craft",
        )
        .unwrap();
        fs::write(
            base_dir.join(format!("{i}.txt")),
            "write clearly about craft",
        )
        .unwrap();
    }
    let target_json = dir.path().join("target.json");
    let base_json = dir.path().join("base.json");
    let fp_json = dir.path().join("fingerprint.json");

    for (input, output) in [(&target_dir, &target_json), (&base_dir, &base_json)] {
        let status = Command::new(env!("CARGO_BIN_EXE_antislop"))
            .args([
                "profile",
                input.to_str().unwrap(),
                "--language",
                "en",
                "-o",
                output.to_str().unwrap(),
            ])
            .status()
            .unwrap();
        assert!(status.success());
    }
    let status = Command::new(env!("CARGO_BIN_EXE_antislop"))
        .args([
            "fingerprint",
            "--target",
            target_json.to_str().unwrap(),
            "--baseline",
            base_json.to_str().unwrap(),
            "-o",
            fp_json.to_str().unwrap(),
        ])
        .status()
        .unwrap();
    assert!(status.success());

    let article = dir.path().join("article.txt");
    fs::write(&article, "We delve deeply into the object.").unwrap();
    let output = Command::new(env!("CARGO_BIN_EXE_antislop"))
        .args([
            "analyze",
            article.to_str().unwrap(),
            "--language",
            "en",
            "--fingerprint",
            fp_json.to_str().unwrap(),
        ])
        .output()
        .unwrap();
    assert!(output.status.success());
    let value: serde_json::Value = serde_json::from_slice(&output.stdout).unwrap();
    assert!(
        value["fingerprint_hits"]
            .as_array()
            .unwrap()
            .iter()
            .any(|hit| hit["pattern"] == "delve")
    );
}

#[test]
fn lint_json_localizes_findings_by_sentence_and_line() {
    let dir = tempdir().unwrap();
    let article = dir.path().join("article.txt");
    fs::write(
        &article,
        "Phrase neutre.\nCe n'est pas seulement un objet, mais un abonnement.\nUne autre phrase.",
    )
    .unwrap();

    let output = Command::new(env!("CARGO_BIN_EXE_antislop"))
        .args([
            "lint",
            article.to_str().unwrap(),
            "--language",
            "fr",
            "--json",
        ])
        .output()
        .unwrap();

    assert!(output.status.success());
    let value: serde_json::Value = serde_json::from_slice(&output.stdout).unwrap();
    assert_eq!(value["sentence_count"], 3);
    assert_eq!(value["flagged_sentence_count"], 1);
    assert_eq!(value["findings"][0]["start_line"], 2);
    assert_eq!(value["findings"][0]["end_line"], 2);
    assert_eq!(
        value["findings"][0]["structural_hits"][0]["rule"],
        "not_only_but"
    );
}

#[test]
fn lint_exits_two_when_ci_threshold_is_exceeded() {
    let dir = tempdir().unwrap();
    let article = dir.path().join("article.txt");
    fs::write(
        &article,
        "Ce n'est pas seulement un objet, mais un abonnement.",
    )
    .unwrap();

    let output = Command::new(env!("CARGO_BIN_EXE_antislop"))
        .args([
            "lint",
            article.to_str().unwrap(),
            "--language",
            "fr",
            "--json",
            "--max-structural-hits",
            "0",
        ])
        .output()
        .unwrap();

    assert_eq!(output.status.code(), Some(2));
    let value: serde_json::Value = serde_json::from_slice(&output.stdout).unwrap();
    assert_eq!(value["passed"], false);
    assert_eq!(value["violations"][0]["metric"], "structural_hits");
}

#[test]
fn style_profile_then_compare_ranks_matching_prose_closer() {
    let dir = tempdir().unwrap();
    let corpus = dir.path().join("arthur");
    fs::create_dir(&corpus).unwrap();
    for (index, text) in [
        "Mais je regarde. Je coupe. Puis je recommence ? On voit vite le problème.",
        "Mais je teste. Je garde le détail. Puis je change. On comprend pourquoi.",
        "Je prends l'objet. Je regarde encore. Mais ça casse. Alors je recommence.",
        "On essaie. Puis on compare. Mais je préfère le détail concret. Je continue.",
        "Je regarde la machine. Mais je garde une question. Puis je teste encore.",
    ]
    .iter()
    .enumerate()
    {
        fs::write(corpus.join(format!("{index}.txt")), text).unwrap();
    }
    fs::write(corpus.join("manifest.json"), r#"{"author":"Arthur"}"#).unwrap();
    let profile = dir.path().join("arthur.style.json");
    let status = Command::new(env!("CARGO_BIN_EXE_antislop"))
        .args([
            "style",
            "profile",
            corpus.to_str().unwrap(),
            "--language",
            "fr",
            "-o",
            profile.to_str().unwrap(),
        ])
        .status()
        .unwrap();
    assert!(status.success());

    let close = dir.path().join("close.txt");
    let far = dir.path().join("far.txt");
    fs::write(
        &close,
        "Mais je regarde. Puis je teste. On comprend le problème. Je recommence ?",
    )
    .unwrap();
    fs::write(
        &far,
        "Toutefois, la conceptualisation méthodologique de cette infrastructure implique une reconfiguration substantielle des modalités organisationnelles ; cette transformation demeure néanmoins subordonnée à plusieurs considérations institutionnelles complexes.",
    )
    .unwrap();

    let compare = |file: &std::path::Path| {
        let output = Command::new(env!("CARGO_BIN_EXE_antislop"))
            .args([
                "style",
                "compare",
                file.to_str().unwrap(),
                "--profile",
                profile.to_str().unwrap(),
                "--json",
            ])
            .output()
            .unwrap();
        assert!(output.status.success());
        serde_json::from_slice::<serde_json::Value>(&output.stdout).unwrap()
    };

    let close_report = compare(&close);
    let far_report = compare(&far);
    assert_eq!(close_report["language"], "fr");
    assert_eq!(close_report["profile_documents"], 5);
    assert!(
        close_report["overall_distance"].as_f64().unwrap()
            < far_report["overall_distance"].as_f64().unwrap()
    );
    assert!(close_report["groups"]["rhythm"].as_f64().is_some());
    assert!(close_report["top_deviations"].as_array().is_some());
}

#[test]
fn style_compare_auto_strips_markdown_frontmatter_code_links_and_images() {
    let dir = tempdir().unwrap();
    let corpus = dir.path().join("corpus");
    fs::create_dir(&corpus).unwrap();
    for index in 0..4 {
        fs::write(
            corpus.join(format!("{index}.txt")),
            "Je regarde l'objet. Puis je teste le mécanisme. Mais je garde le détail.",
        )
        .unwrap();
    }
    let profile = dir.path().join("profile.json");
    assert!(
        Command::new(env!("CARGO_BIN_EXE_antislop"))
            .args([
                "style",
                "profile",
                corpus.to_str().unwrap(),
                "--language",
                "fr",
                "-o",
                profile.to_str().unwrap(),
            ])
            .status()
            .unwrap()
            .success()
    );

    let article = dir.path().join("article.mdoc");
    fs::write(
        &article,
        "---\ntitle: Un titre qui ne doit pas compter\nlocale: fr\n---\nJe regarde [l'objet](https://example.com).\n\n![Une image artificielle](https://example.com/a.jpg)\n\n```js\nconst slop = 'ne compte pas';\n```\nPuis je teste le mécanisme.",
    )
    .unwrap();
    let output = Command::new(env!("CARGO_BIN_EXE_antislop"))
        .args([
            "style",
            "compare",
            article.to_str().unwrap(),
            "--profile",
            profile.to_str().unwrap(),
            "--json",
        ])
        .output()
        .unwrap();
    assert!(output.status.success());
    let report: serde_json::Value = serde_json::from_slice(&output.stdout).unwrap();
    assert_eq!(report["document"]["tokens"], 8);
}

#[test]
fn fingerprint_guard_filters_accepted_phrase() {
    let dir = tempdir().unwrap();
    let target_dir = dir.path().join("target");
    let base_dir = dir.path().join("base");
    let guard_dir = dir.path().join("guard");
    fs::create_dir(&target_dir).unwrap();
    fs::create_dir(&base_dir).unwrap();
    fs::create_dir(&guard_dir).unwrap();
    for i in 0..3 {
        fs::write(
            target_dir.join(format!("{i}.txt")),
            "vous avez une idée claire",
        )
        .unwrap();
        fs::write(base_dir.join(format!("{i}.txt")), "une idée claire existe").unwrap();
        fs::write(
            guard_dir.join(format!("{i}.txt")),
            "vous avez une idée humaine",
        )
        .unwrap();
    }
    let target = dir.path().join("target.json");
    let base = dir.path().join("base.json");
    let guard = dir.path().join("guard.json");
    for (input, output) in [
        (&target_dir, &target),
        (&base_dir, &base),
        (&guard_dir, &guard),
    ] {
        assert!(
            Command::new(env!("CARGO_BIN_EXE_antislop"))
                .args([
                    "profile",
                    input.to_str().unwrap(),
                    "--language",
                    "fr",
                    "-o",
                    output.to_str().unwrap()
                ])
                .status()
                .unwrap()
                .success()
        );
    }
    let fp = dir.path().join("fp.json");
    let status = Command::new(env!("CARGO_BIN_EXE_antislop"))
        .args([
            "fingerprint",
            "--target",
            target.to_str().unwrap(),
            "--baseline",
            base.to_str().unwrap(),
            "--guard",
            guard.to_str().unwrap(),
            "--min-guard-ratio",
            "2",
            "--min-documents",
            "3",
            "--bigram-limit",
            "20",
            "--trigram-limit",
            "20",
            "-o",
            fp.to_str().unwrap(),
        ])
        .status()
        .unwrap();
    assert!(status.success());
    let value: serde_json::Value = serde_json::from_str(&fs::read_to_string(fp).unwrap()).unwrap();
    assert_eq!(value["guard_profile_documents"], 3);
    assert!(
        !value["bigrams"]
            .as_array()
            .unwrap()
            .iter()
            .any(|entry| entry["pattern"] == "vous avez")
    );
}
