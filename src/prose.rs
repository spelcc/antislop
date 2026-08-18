use regex::Regex;

/// Remove common Markdown/Markdoc/HTML plumbing while retaining visible prose.
/// This is intentionally conservative: headings, list items, blockquotes and link labels stay text.
pub fn clean_prose(input: &str) -> String {
    let normalized = input.replace("\r\n", "\n").replace('\r', "\n");
    let mut text = strip_frontmatter(&normalized);
    text = regex_replace(&text, r"(?s)```.*?```|~~~.*?~~~", "\n");
    text = regex_replace(&text, r"(?is)<(?:pre|code)\b[^>]*>.*?</(?:pre|code)>", " ");
    text = regex_replace(&text, r"`[^`\n]+`", " ");
    text = regex_replace(&text, r"!\[[^\]]*\]\([^)]*\)", " ");
    text = regex_replace(&text, r"(?m)^\s*!\[[^\]]*\]:\s*\S+.*$", " ");
    text = regex_replace(&text, r"(?i)<img\b[^>]*>", " ");
    text = regex_replace(&text, r"(?s)\{%.*?%\}", "\n");
    text = regex_replace(
        &text,
        r"(?ims)^#{1,6}\s+(?:références|references)\s*$.*\z",
        "\n",
    );

    let links = Regex::new(r"\[([^\]]+)\]\([^)]*\)").expect("built-in link regex must compile");
    text = links.replace_all(&text, "$1").into_owned();
    text = regex_replace(&text, r"(?m)^\s*\[[^\]]+\]:\s*\S+.*$", " ");
    text = regex_replace(&text, r"<https?://[^>]+>", " ");
    text = regex_replace(&text, r"https?://[^\s)]+", " ");
    text = regex_replace(&text, r"<[^>]+>", " ");

    text = regex_replace(&text, r"(?m)^\s{0,3}#{1,6}\s*", "");
    text = regex_replace(&text, r"(?m)^\s*>\s?", "");
    text = regex_replace(&text, r"(?m)^\s*(?:[-*+]\s+|\d+[.)]\s+)", "");
    text = regex_replace(&text, r"(?m)^\s*(?:---+|___+|\*\*\*+)\s*$", "");
    text = text.replace("**", "").replace("__", "");

    normalize_whitespace(&text)
}

fn strip_frontmatter(input: &str) -> String {
    if !input.starts_with("---\n") {
        return input.to_string();
    }
    let rest = &input[4..];
    if let Some(end) = rest.find("\n---\n") {
        return rest[end + 5..].to_string();
    }
    input.to_string()
}

fn regex_replace(input: &str, pattern: &str, replacement: &str) -> String {
    Regex::new(pattern)
        .expect("built-in prose regex must compile")
        .replace_all(input, replacement)
        .into_owned()
}

fn normalize_whitespace(input: &str) -> String {
    let mut lines = Vec::new();
    for line in input.lines() {
        let line = line.split_whitespace().collect::<Vec<_>>().join(" ");
        lines.push(line);
    }
    let mut output = lines.join("\n");
    let blank_runs = Regex::new(r"\n{3,}").expect("built-in blank-line regex must compile");
    output = blank_runs.replace_all(&output, "\n\n").into_owned();
    output.trim().to_string()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn keeps_visible_link_text_and_removes_plumbing() {
        let cleaned = clean_prose(
            "---\ntitle: Secret\n---\n# Titre\nJe lis [ce texte](https://example.com).\n\n![Alt](x.jpg)\n```rs\nlet x = 1;\n```\n\n## Références\nUne bibliographie étrangère.",
        );
        assert_eq!(cleaned, "Titre\nJe lis ce texte.");
    }
}
