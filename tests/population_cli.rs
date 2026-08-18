use std::fs;
use std::process::Command;
use tempfile::tempdir;

#[test]
fn nearest_candidate_manifest_ranks_human_and_llm_with_style() {
    let dir = tempdir().unwrap();
    let baseline_dir = dir.path().join("baseline");
    let human_dir = dir.path().join("human");
    let llm_dir = dir.path().join("llm");
    for path in [&baseline_dir, &human_dir, &llm_dir] {
        fs::create_dir(path).unwrap();
    }
    for index in 0..4 {
        fs::write(
            baseline_dir.join(format!("{index}.txt")),
            "Une prose de référence présente un fait concret avec précision.",
        )
        .unwrap();
        fs::write(
            human_dir.join(format!("{index}.txt")),
            "Je regarde l'objet. Puis je vérifie le détail. Mais je garde une question.",
        )
        .unwrap();
        fs::write(
            llm_dir.join(format!("{index}.txt")),
            "Il est important de noter ce résultat. En résumé, voici les éléments essentiels.",
        )
        .unwrap();
    }

    let build_profile = |corpus: &std::path::Path, output: &std::path::Path| {
        assert!(
            Command::new(env!("CARGO_BIN_EXE_antislop"))
                .args([
                    "profile",
                    corpus.to_str().unwrap(),
                    "--language",
                    "fr",
                    "-o",
                    output.to_str().unwrap(),
                ])
                .status()
                .unwrap()
                .success()
        );
    };
    let baseline = dir.path().join("baseline.json");
    build_profile(&baseline_dir, &baseline);

    let candidates_dir = dir.path().join("candidate-files");
    fs::create_dir(&candidates_dir).unwrap();
    for (label, class, corpus) in [
        ("human-editorial", "human", &human_dir),
        ("model-x", "llm", &llm_dir),
    ] {
        let profile = candidates_dir.join(format!("{label}.profile.json"));
        build_profile(corpus, &profile);
        let fingerprint = candidates_dir.join(format!("{label}.fingerprint.json"));
        assert!(
            Command::new(env!("CARGO_BIN_EXE_antislop"))
                .args([
                    "fingerprint",
                    "--target",
                    profile.to_str().unwrap(),
                    "--baseline",
                    baseline.to_str().unwrap(),
                    "--min-documents",
                    "1",
                    "--no-wordfreq",
                    "--label",
                    label,
                    "-o",
                    fingerprint.to_str().unwrap(),
                ])
                .status()
                .unwrap()
                .success()
        );
        let style = candidates_dir.join(format!("{label}.style.json"));
        assert!(
            Command::new(env!("CARGO_BIN_EXE_antislop"))
                .args([
                    "style",
                    "profile",
                    corpus.to_str().unwrap(),
                    "--language",
                    "fr",
                    "-o",
                    style.to_str().unwrap(),
                ])
                .status()
                .unwrap()
                .success()
        );
        assert!(matches!(class, "human" | "llm"));
    }

    let manifest = dir.path().join("candidates.json");
    fs::write(
        &manifest,
        serde_json::to_string_pretty(&serde_json::json!({
            "schema_version": 1,
            "language": "fr",
            "candidates": [
                {
                    "label": "human-editorial",
                    "class": "human",
                    "fingerprint": "candidate-files/human-editorial.fingerprint.json",
                    "style_profile": "candidate-files/human-editorial.style.json"
                },
                {
                    "label": "model-x",
                    "class": "llm",
                    "fingerprint": "candidate-files/model-x.fingerprint.json",
                    "style_profile": "candidate-files/model-x.style.json"
                }
            ]
        }))
        .unwrap(),
    )
    .unwrap();

    let article = dir.path().join("article.txt");
    fs::write(
        &article,
        "Je regarde l'objet. Puis je vérifie le détail. Mais je garde cette question.",
    )
    .unwrap();
    let output = Command::new(env!("CARGO_BIN_EXE_antislop"))
        .args([
            "nearest",
            article.to_str().unwrap(),
            "--baseline",
            baseline.to_str().unwrap(),
            "--candidates",
            manifest.to_str().unwrap(),
            "--language",
            "fr",
            "--top",
            "2",
            "--json",
        ])
        .output()
        .unwrap();
    assert!(
        output.status.success(),
        "{}",
        String::from_utf8_lossy(&output.stderr)
    );
    let value: serde_json::Value = serde_json::from_slice(&output.stdout).unwrap();
    assert_eq!(value["matches"][0]["label"], "human-editorial");
    assert_eq!(value["matches"][0]["class"], "human");
    assert_eq!(value["matches"][1]["class"], "llm");
    assert!(value["matches"][0]["style_distance"].as_f64().is_some());
    assert_eq!(value["matches"][0]["style_distance_position"], 1);
}

#[test]
fn calibrate_then_classify_returns_empirical_human_probability() {
    let dir = tempdir().unwrap();
    let baseline_dir = dir.path().join("baseline");
    let human_dir = dir.path().join("human-candidate");
    let llm_dir = dir.path().join("llm-candidate");
    for path in [&baseline_dir, &human_dir, &llm_dir] {
        fs::create_dir(path).unwrap();
    }
    for index in 0..5 {
        fs::write(
            baseline_dir.join(format!("{index}.txt")),
            "Le texte de référence décrit un objet concret avec une phrase simple et précise.",
        )
        .unwrap();
        fs::write(
            human_dir.join(format!("{index}.txt")),
            "Je regarde l'objet. Puis je teste le détail. Mais je garde une question concrète.",
        )
        .unwrap();
        fs::write(
            llm_dir.join(format!("{index}.txt")),
            "Il est important de noter plusieurs éléments. En résumé, voici les points essentiels à considérer.",
        )
        .unwrap();
    }

    let run_profile = |corpus: &std::path::Path, output: &std::path::Path| {
        assert!(
            Command::new(env!("CARGO_BIN_EXE_antislop"))
                .args([
                    "profile",
                    corpus.to_str().unwrap(),
                    "--language",
                    "fr",
                    "-o",
                    output.to_str().unwrap(),
                ])
                .status()
                .unwrap()
                .success()
        );
    };
    let baseline = dir.path().join("baseline.json");
    run_profile(&baseline_dir, &baseline);
    let candidate_files = dir.path().join("candidate-files");
    fs::create_dir(&candidate_files).unwrap();
    for (label, corpus) in [("human-editorial", &human_dir), ("model-x", &llm_dir)] {
        let profile = candidate_files.join(format!("{label}.profile.json"));
        run_profile(corpus, &profile);
        assert!(
            Command::new(env!("CARGO_BIN_EXE_antislop"))
                .args([
                    "fingerprint",
                    "--target",
                    profile.to_str().unwrap(),
                    "--baseline",
                    baseline.to_str().unwrap(),
                    "--min-documents",
                    "1",
                    "--no-wordfreq",
                    "--label",
                    label,
                    "-o",
                    candidate_files
                        .join(format!("{label}.fingerprint.json"))
                        .to_str()
                        .unwrap(),
                ])
                .status()
                .unwrap()
                .success()
        );
        assert!(
            Command::new(env!("CARGO_BIN_EXE_antislop"))
                .args([
                    "style",
                    "profile",
                    corpus.to_str().unwrap(),
                    "--language",
                    "fr",
                    "-o",
                    candidate_files
                        .join(format!("{label}.style.json"))
                        .to_str()
                        .unwrap(),
                ])
                .status()
                .unwrap()
                .success()
        );
    }
    let candidates = dir.path().join("candidates.json");
    fs::write(
        &candidates,
        serde_json::to_string_pretty(&serde_json::json!({
            "schema_version": 1,
            "language": "fr",
            "candidates": [
                {"label":"human-editorial","class":"human","fingerprint":"candidate-files/human-editorial.fingerprint.json","style_profile":"candidate-files/human-editorial.style.json"},
                {"label":"model-x","class":"llm","fingerprint":"candidate-files/model-x.fingerprint.json","style_profile":"candidate-files/model-x.style.json"}
            ]
        }))
        .unwrap(),
    )
    .unwrap();

    let calibration_dir = dir.path().join("calibration");
    fs::create_dir(&calibration_dir).unwrap();
    let mut documents = Vec::new();
    for (class, prefix, sentence) in [
        (
            "human",
            "h",
            "Je regarde cette machine. Puis je vérifie le détail concret. Mais je garde une question.",
        ),
        (
            "llm",
            "l",
            "Il est important de noter ce résultat. En résumé, voici les principaux éléments à considérer.",
        ),
    ] {
        for index in 0..6 {
            let file = format!("{prefix}{index}.txt");
            fs::write(
                calibration_dir.join(&file),
                format!("{sentence} Exemple numéro {index}."),
            )
            .unwrap();
            documents.push(serde_json::json!({
                "file": format!("calibration/{file}"),
                "class": class,
                "split": if index < 4 { "train" } else { "test" }
            }));
        }
    }
    let calibration = dir.path().join("calibration.json");
    fs::write(
        &calibration,
        serde_json::to_string_pretty(&serde_json::json!({
            "schema_version": 1,
            "language": "fr",
            "documents": documents
        }))
        .unwrap(),
    )
    .unwrap();

    let classifier = dir.path().join("classifier.json");
    let calibration_output = Command::new(env!("CARGO_BIN_EXE_antislop"))
        .args([
            "calibrate",
            "--documents",
            calibration.to_str().unwrap(),
            "--baseline",
            baseline.to_str().unwrap(),
            "--candidates",
            candidates.to_str().unwrap(),
            "--language",
            "fr",
            "-o",
            classifier.to_str().unwrap(),
        ])
        .output()
        .unwrap();
    assert!(
        calibration_output.status.success(),
        "{}",
        String::from_utf8_lossy(&calibration_output.stderr)
    );
    let calibration_report: serde_json::Value =
        serde_json::from_slice(&calibration_output.stdout).unwrap();
    assert_eq!(calibration_report["train_documents"], 8);
    assert_eq!(calibration_report["test_documents"], 4);
    assert!(
        calibration_report["evaluation"]["accuracy"]
            .as_f64()
            .unwrap()
            >= 0.75
    );
    assert!(
        calibration_report["evaluation"]["expected_calibration_error"]
            .as_f64()
            .unwrap()
            <= 0.5
    );

    let article = dir.path().join("article.txt");
    fs::write(
        &article,
        "Je regarde la machine. Puis je vérifie ce détail. Mais je garde une question concrète.",
    )
    .unwrap();
    let output = Command::new(env!("CARGO_BIN_EXE_antislop"))
        .args([
            "classify",
            article.to_str().unwrap(),
            "--classifier",
            classifier.to_str().unwrap(),
            "--baseline",
            baseline.to_str().unwrap(),
            "--candidates",
            candidates.to_str().unwrap(),
            "--language",
            "fr",
            "--json",
        ])
        .output()
        .unwrap();
    assert!(
        output.status.success(),
        "{}",
        String::from_utf8_lossy(&output.stderr)
    );
    let value: serde_json::Value = serde_json::from_slice(&output.stdout).unwrap();
    assert_eq!(value["predicted_class"], "human");
    assert!(value["human_probability"].as_f64().unwrap() > 0.5);
    assert!(value["llm_probability"].as_f64().unwrap() < 0.5);
    assert_eq!(value["nearest"][0]["class"], "human");

    let changed_baseline = dir.path().join("changed-baseline.json");
    let mut changed: serde_json::Value =
        serde_json::from_str(&fs::read_to_string(&baseline).unwrap()).unwrap();
    changed["token_count"] = serde_json::json!(changed["token_count"].as_u64().unwrap() + 1);
    fs::write(
        &changed_baseline,
        serde_json::to_string_pretty(&changed).unwrap(),
    )
    .unwrap();
    let mismatch = Command::new(env!("CARGO_BIN_EXE_antislop"))
        .args([
            "classify",
            article.to_str().unwrap(),
            "--classifier",
            classifier.to_str().unwrap(),
            "--baseline",
            changed_baseline.to_str().unwrap(),
            "--candidates",
            candidates.to_str().unwrap(),
            "--language",
            "fr",
            "--json",
        ])
        .output()
        .unwrap();
    assert!(!mismatch.status.success());
    assert!(String::from_utf8_lossy(&mismatch.stderr).contains("baseline"));
}

#[test]
fn classify_ci_fails_below_human_threshold_and_explains_how_to_pass() {
    let dir = tempdir().unwrap();
    let baseline_dir = dir.path().join("baseline");
    let human_dir = dir.path().join("human-candidate");
    let llm_dir = dir.path().join("llm-candidate");
    for path in [&baseline_dir, &human_dir, &llm_dir] {
        fs::create_dir(path).unwrap();
    }
    for index in 0..6 {
        fs::write(
            baseline_dir.join(format!("{index}.txt")),
            "A careful human editor describes a concrete object with varied sentences and specific observations.",
        )
        .unwrap();
        fs::write(
            human_dir.join(format!("{index}.txt")),
            "I test the object. Then I check one awkward detail. The result changes what I do next.",
        )
        .unwrap();
        fs::write(
            llm_dir.join(format!("{index}.txt")),
            "It is important to note this result. In summary, here are the key considerations and essential points.",
        )
        .unwrap();
    }

    let profile = |corpus: &std::path::Path, output: &std::path::Path| {
        assert!(
            Command::new(env!("CARGO_BIN_EXE_antislop"))
                .args([
                    "profile",
                    corpus.to_str().unwrap(),
                    "--language",
                    "en",
                    "-o",
                    output.to_str().unwrap()
                ])
                .status()
                .unwrap()
                .success()
        );
    };
    let baseline = dir.path().join("baseline.json");
    profile(&baseline_dir, &baseline);
    let files = dir.path().join("candidates");
    fs::create_dir(&files).unwrap();
    for (label, corpus) in [("human-editorial", &human_dir), ("model-x", &llm_dir)] {
        let p = files.join(format!("{label}.profile.json"));
        profile(corpus, &p);
        assert!(
            Command::new(env!("CARGO_BIN_EXE_antislop"))
                .args([
                    "fingerprint",
                    "--target",
                    p.to_str().unwrap(),
                    "--baseline",
                    baseline.to_str().unwrap(),
                    "--min-documents",
                    "1",
                    "--no-wordfreq",
                    "--label",
                    label,
                    "-o",
                    files
                        .join(format!("{label}.fingerprint.json"))
                        .to_str()
                        .unwrap()
                ])
                .status()
                .unwrap()
                .success()
        );
        assert!(
            Command::new(env!("CARGO_BIN_EXE_antislop"))
                .args([
                    "style",
                    "profile",
                    corpus.to_str().unwrap(),
                    "--language",
                    "en",
                    "-o",
                    files.join(format!("{label}.style.json")).to_str().unwrap()
                ])
                .status()
                .unwrap()
                .success()
        );
    }
    let candidates = dir.path().join("candidates.json");
    fs::write(&candidates, serde_json::to_string_pretty(&serde_json::json!({
        "schema_version": 1, "language": "en", "candidates": [
            {"label":"human-editorial","class":"human","fingerprint":"candidates/human-editorial.fingerprint.json","style_profile":"candidates/human-editorial.style.json"},
            {"label":"model-x","class":"llm","fingerprint":"candidates/model-x.fingerprint.json","style_profile":"candidates/model-x.style.json"}
        ]
    })).unwrap()).unwrap();

    let calibration_dir = dir.path().join("calibration");
    fs::create_dir(&calibration_dir).unwrap();
    let mut docs = Vec::new();
    for (class, prefix, text) in [
        (
            "human",
            "h",
            "I test the object. Then I inspect one odd detail. The result changes my next step.",
        ),
        (
            "llm",
            "l",
            "It is important to note this result. In summary, here are the key considerations and essential points.",
        ),
    ] {
        for index in 0..8 {
            let file = format!("{prefix}{index}.txt");
            fs::write(
                calibration_dir.join(&file),
                format!("{text} Example {index}."),
            )
            .unwrap();
            docs.push(serde_json::json!({"file":format!("calibration/{file}"),"class":class,"split":if index < 5 {"train"} else {"test"}}));
        }
    }
    let calibration = dir.path().join("calibration.json");
    fs::write(
        &calibration,
        serde_json::to_string_pretty(
            &serde_json::json!({"schema_version":1,"language":"en","documents":docs}),
        )
        .unwrap(),
    )
    .unwrap();
    let classifier = dir.path().join("classifier.json");
    assert!(
        Command::new(env!("CARGO_BIN_EXE_antislop"))
            .args([
                "calibrate",
                "--documents",
                calibration.to_str().unwrap(),
                "--baseline",
                baseline.to_str().unwrap(),
                "--candidates",
                candidates.to_str().unwrap(),
                "--language",
                "en",
                "-o",
                classifier.to_str().unwrap()
            ])
            .status()
            .unwrap()
            .success()
    );

    let article = dir.path().join("article.txt");
    fs::write(
        &article,
        "A concrete opening sentence.\nIt is important to note this result.\nIn summary, here are the key considerations and essential points to consider.",
    )
    .unwrap();
    let output = Command::new(env!("CARGO_BIN_EXE_antislop"))
        .args([
            "classify",
            article.to_str().unwrap(),
            "--classifier",
            classifier.to_str().unwrap(),
            "--baseline",
            baseline.to_str().unwrap(),
            "--candidates",
            candidates.to_str().unwrap(),
            "--language",
            "en",
            "--min-human-probability",
            "0.70",
            "--ci",
        ])
        .output()
        .unwrap();
    assert_eq!(output.status.code(), Some(2));
    let stdout = String::from_utf8_lossy(&output.stdout);
    assert!(stdout.contains("CI FAIL"), "{stdout}");
    assert!(stdout.contains("Human probability"), "{stdout}");
    assert!(stdout.contains("How to pass"), "{stdout}");
    assert!(stdout.contains("Priority passages to rewrite"), "{stdout}");
    assert!(stdout.contains("L2"), "{stdout}");
    assert!(
        stdout.contains("It is important to note this result."),
        "{stdout}"
    );

    let json_output = Command::new(env!("CARGO_BIN_EXE_antislop"))
        .args([
            "classify",
            article.to_str().unwrap(),
            "--classifier",
            classifier.to_str().unwrap(),
            "--baseline",
            baseline.to_str().unwrap(),
            "--candidates",
            candidates.to_str().unwrap(),
            "--language",
            "en",
            "--min-human-probability",
            "0.70",
            "--ci",
            "--json",
        ])
        .output()
        .unwrap();
    assert_eq!(json_output.status.code(), Some(2));
    let value: serde_json::Value = serde_json::from_slice(&json_output.stdout).unwrap();
    let fixes = value["fixes"]
        .as_array()
        .expect("classification JSON exposes fixes[]");
    assert!(!fixes.is_empty());
    let line_two = fixes
        .iter()
        .find(|fix| fix["start_line"] == 2)
        .expect("line 2 is pinpointed even when another sentence has higher priority");
    assert!(
        line_two["patterns"]
            .as_array()
            .is_some_and(|patterns| !patterns.is_empty())
    );
}
