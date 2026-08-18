use crate::analyze::StructuralHit;
use crate::tokenize::Language;
use regex::Regex;

pub fn structural_rules(text: &str, language: Language) -> Vec<StructuralHit> {
    let patterns: &[(&str, &str)] = match language {
        Language::En => &[
            (
                "not_only_but",
                r"(?i)\bnot only\b[^.!?;:]{1,120}\bbut(?: also)?\b",
            ),
            ("not_x_but_y", r"(?i)\bnot\b[^.!?;:]{1,120}\bbut\b"),
            (
                "the_real_question",
                r"(?i)\bthe real (?:question|issue|story)\b",
            ),
        ],
        Language::Fr => &[
            (
                "fr_pas_seulement_mais",
                r"(?i)\bpas seulement\b[^.!?;:]{1,140}\bmais(?:\s+(?:aussi|également))?\b",
            ),
            ("fr_non_pas_mais", r"(?i)\bnon pas\b[^.!?;:]{1,140}\bmais\b"),
            (
                "fr_il_ne_s_agit_pas_mais",
                r"(?i)\bil ne s['’]agit (?:pas|plus) (?:de|d['’])[^.!?;:]{1,140}\bmais\s+(?:de|d['’])",
            ),
            (
                "fr_pas_simplement_mais",
                r"(?i)\b(?:ce|ça|cela) n['’]est (?:pas|plus) (?:simplement|seulement|juste)\b[^.!?;:]{1,140}\b(?:mais|c['’]est|plutôt)\b",
            ),
            (
                "fr_ce_n_est_pas_mais",
                r"(?i)\b(?:ce|ça|cela) n['’]est (?:pas|plus)\b[^.!?;:]{1,120}(?:,|;|:)\s*(?:mais|c['’]est|plutôt)\b",
            ),
            (
                "fr_ne_se_contente_pas",
                r"(?i)\b(?:il|elle|on|cela|ça) ne se contente pas (?:de|d['’])[^.!?;:]{1,140}\b(?:il|elle|cela|ça)\b",
            ),
            ("fr_pas_tant_que", r"(?i)\bpas tant\b[^.!?;:]{1,120}\bque\b"),
            (
                "the_real_question",
                r"(?i)\bla vraie (?:question|histoire)\b|\ble vrai sujet\b",
            ),
        ],
    };

    patterns
        .iter()
        .filter_map(|(rule, pattern)| {
            let regex = Regex::new(pattern).expect("built-in structural regex must compile");
            let count = regex
                .find_iter(text)
                .filter(|matched| !is_shadowed_generic(rule, matched.as_str()))
                .count();
            (count > 0).then(|| StructuralHit {
                rule: (*rule).to_string(),
                count,
            })
        })
        .collect()
}

fn is_shadowed_generic(rule: &str, value: &str) -> bool {
    let lowered = value.to_lowercase();
    match rule {
        "not_x_but_y" => lowered.contains("not only"),
        "fr_ce_n_est_pas_mais" => {
            lowered.contains("pas seulement")
                || lowered.contains("plus seulement")
                || lowered.contains("pas simplement")
                || lowered.contains("plus simplement")
                || lowered.contains("pas juste")
                || lowered.contains("plus juste")
        }
        _ => false,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn covers_major_french_false_contrast_forms_without_generic_double_count() {
        let text = concat!(
            "Ce n'est pas un gadget, c'est une plateforme. ",
            "Ce n'est plus simplement une interface mais un marché. ",
            "Il ne s'agit pas de vitesse mais de contrôle. ",
            "Pas seulement un outil, mais aussi une habitude. ",
            "Non pas une panne, mais une dépendance."
        );
        let hits = structural_rules(text, Language::Fr);
        let total: usize = hits.iter().map(|hit| hit.count).sum();
        assert_eq!(total, 5);
        assert!(
            hits.iter()
                .any(|hit| hit.rule == "fr_il_ne_s_agit_pas_mais")
        );
        assert!(hits.iter().any(|hit| hit.rule == "fr_pas_simplement_mais"));
    }

    #[test]
    fn ordinary_negation_is_not_a_false_contrast() {
        let hits = structural_rules(
            "Ce n'est pas disponible aujourd'hui. Il ne s'agit pas de notre dossier.",
            Language::Fr,
        );
        assert!(hits.is_empty());
    }
}
