import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from build_comparia_corpus import Candidate, clean_response, count_words, finalize_documents, select_candidate_reserves


class CompariaBuilderTests(unittest.TestCase):
    def candidate(self, source_id, model, prompt, response, category="Technology"):
        return Candidate(source_id, "a", model, prompt, response, (category,), 400)

    def test_selection_is_deterministic_and_globally_unique(self):
        rows = [
            self.candidate(1, "rare", "shared", "r1"),
            self.candidate(2, "rare", "rare-2", "r2"),
            self.candidate(3, "common", "shared", "c1"),
            self.candidate(4, "common", "common-2", "c2"),
            self.candidate(5, "common", "common-3", "c3"),
        ]
        first = select_candidate_reserves(rows, ["common", "rare"], 1, 1, ["Technology"], 16)
        second = select_candidate_reserves(rows, ["common", "rare"], 1, 1, ["Technology"], 16)
        self.assertEqual(first, second)
        prompts = [row.prompt_id for values in first.values() for row in values]
        responses = [row.response_id for values in first.values() for row in values]
        self.assertEqual(len(prompts), len(set(prompts)))
        self.assertEqual(len(responses), len(set(responses)))

    def test_finalize_uses_reserve_when_cleaned_text_is_too_short(self):
        rows = {
            "m": [
                self.candidate(1, "m", "p1", "r1"),
                self.candidate(2, "m", "p2", "r2"),
            ]
        }
        responses = {
            (1, "a"): "```python\n" + "mot " * 500 + "\n```",
            (2, "a"): "mot " * 400,
        }
        result = finalize_documents(rows, responses, ["m"], 1, 350)
        self.assertEqual(result["m"][0].candidate.source_id, 2)
        self.assertEqual(result["m"][0].clean_words, 400)

    def test_clean_response_removes_markup_but_preserves_link_label(self):
        text = "# Titre\n\nVoir [la documentation](https://example.com) et `code`."
        cleaned = clean_response(text)
        self.assertIn("Titre", cleaned)
        self.assertIn("la documentation", cleaned)
        self.assertNotIn("https://", cleaned)
        self.assertNotIn("code", cleaned)
        self.assertEqual(count_words(cleaned), 5)


if __name__ == "__main__":
    unittest.main()
