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
