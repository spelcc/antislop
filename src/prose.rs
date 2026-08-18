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

    text = regex_replace(&text, r"\[\d+\]\(#ref-\d+\)", " ");
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

/// Remove Markdown/Markdoc/HTML plumbing while preserving the original line count.
/// Use this for diagnostics that must point back to source lines.
pub fn clean_prose_preserving_lines(input: &str) -> String {
    let normalized = input.replace("\r\n", "\n").replace('\r', "\n");
    let mut text = mask_frontmatter(&normalized);
    text = regex_mask(&text, r"(?s)```.*?```|~~~.*?~~~");
    text = regex_mask(&text, r"(?is)<(?:pre|code)\b[^>]*>.*?</(?:pre|code)>");
    text = regex_replace(&text, r"`[^`\n]+`", " ");
    text = regex_replace(&text, r"!\[[^\]]*\]\([^)]*\)", " ");
    text = regex_replace(&text, r"(?m)^[ \t]*!\[[^\]]*\]:[ \t]*\S+.*$", " ");
    text = regex_replace(&text, r"(?i)<img\b[^>]*>", " ");
    text = regex_mask(&text, r"(?s)\{%.*?%\}");
    text = regex_mask(&text, r"(?ims)^#{1,6}\s+(?:références|references)\s*$.*\z");

    text = regex_replace(&text, r"\[\d+\]\(#ref-\d+\)", " ");
    let links = Regex::new(r"\[([^\]]+)\]\([^)]*\)").expect("built-in link regex must compile");
    text = links.replace_all(&text, "$1").into_owned();
    text = regex_replace(&text, r"(?m)^[ \t]*\[[^\]]+\]:[ \t]*\S+.*$", " ");
    text = regex_replace(&text, r"<https?://[^>]+>", " ");
    text = regex_replace(&text, r"https?://[^\s)]+", " ");
    text = regex_replace(&text, r"<[^>]+>", " ");

    text = regex_replace(&text, r"(?m)^[ \t]{0,3}#{1,6}[ \t]*", "");
    text = regex_replace(&text, r"(?m)^[ \t]*>[ \t]?", "");
    text = regex_replace(&text, r"(?m)^[ \t]*(?:[-*+][ \t]+|\d+[.)][ \t]+)", "");
    text = regex_replace(&text, r"(?m)^[ \t]*(?:---+|___+|\*\*\*+)[ \t]*$", "");
    text = text.replace("**", "").replace("__", "");

    normalize_whitespace_preserving_lines(&text)
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

fn mask_frontmatter(input: &str) -> String {
    if !input.starts_with("---\n") {
        return input.to_string();
    }
    let rest = &input[4..];
    let Some(end) = rest.find("\n---\n") else {
        return input.to_string();
    };
    let masked_end = 4 + end + 5;
    format!(
        "{}{}",
        mask_non_newlines(&input[..masked_end]),
        &input[masked_end..]
    )
}

fn regex_mask(input: &str, pattern: &str) -> String {
    Regex::new(pattern)
        .expect("built-in prose regex must compile")
        .replace_all(input, |captures: &regex::Captures<'_>| {
            mask_non_newlines(&captures[0])
        })
        .into_owned()
}

fn mask_non_newlines(value: &str) -> String {
    value
        .chars()
        .map(|ch| if ch == '\n' { '\n' } else { ' ' })
        .collect()
}

fn normalize_whitespace_preserving_lines(input: &str) -> String {
    input
        .split('\n')
        .map(|line| line.split_whitespace().collect::<Vec<_>>().join(" "))
        .collect::<Vec<_>>()
        .join("\n")
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

    #[test]
    fn preserving_cleaner_keeps_source_line_numbers() {
        let source = "---\ntitle: Test\nlocale: en\n---\n# Heading\n\n{% component foo=\"bar\" /%}\nIt is important to note this result.\n\n## References\nHidden reference.";
        let cleaned = clean_prose_preserving_lines(source);
        assert_eq!(cleaned.split('\n').count(), source.split('\n').count());
        assert_eq!(
            cleaned.split('\n').nth(7),
            Some("It is important to note this result.")
        );
        assert!(!cleaned.contains("Hidden reference"));
    }
}
