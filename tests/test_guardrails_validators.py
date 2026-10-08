"""Offline tests for redaction, content-preserving JSON repair, and Guard FIX."""

import importlib
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from guardrails.validators import FailResult, PassResult
from utils.guardrails_validators import JSONFormatter, PIIDetector, parse_json


class GuardrailsValidatorTests(unittest.TestCase):
    """Exercise real custom validators; only synthetic PII is used."""

    def setUp(self):
        self.pii = PIIDetector()
        self.formatter = JSONFormatter()

    def repaired_json(self, text):
        result = self.formatter.validate(text, {})
        self.assertIsInstance(result, FailResult)
        return parse_json(result.fix_value)

    def test_pii_supports_all_four_types(self):
        fixtures = {
            "EMAIL": "user@example.com",
            "PHONE": "555-123-4567",
            "SSN": "123-45-6789",
            "CREDIT_CARD": "4532 1234 5678 9010",
        }
        for pii_type, value in fixtures.items():
            with self.subTest(pii_type=pii_type):
                result = self.pii.validate(value, {})
                self.assertIsInstance(result, FailResult)
                self.assertEqual(result.fix_value, f"[{pii_type}_REDACTED]")

    def test_phone_redacts_full_span(self):
        for value in ("(555) 867-5309", "+1 (555) 867-5309", "5558675309", "555.867.5309"):
            with self.subTest(value=value):
                self.assertEqual(self.pii.validate(value, {}).fix_value, "[PHONE_REDACTED]")

    def test_mixed_pii_is_fully_removed(self):
        value = "user@example.com | +1 (555) 867-5309 | 123-45-6789 | 4532-1234-5678-9010"
        output = self.pii.validate(value, {}).fix_value
        self.assertEqual(output, "[EMAIL_REDACTED] | [PHONE_REDACTED] | [SSN_REDACTED] | [CREDIT_CARD_REDACTED]")
        self.assertIsInstance(self.pii.validate(output, {}), PassResult)

    def test_card_is_not_partially_redacted_as_phone(self):
        self.assertEqual(self.pii.validate("4532123456789010", {}).fix_value, "[CREDIT_CARD_REDACTED]")

    def test_clean_text_is_preserved_by_guard(self):
        module = importlib.import_module("04_guardrails_validator")
        value = "This text contains no sensitive information."
        self.assertEqual(module.make_pii_guard().validate(value, num_reasks=0).validated_output, value)

    def test_repeated_email_is_redacted_everywhere(self):
        result = self.pii.validate("user@example.com and user@example.com", {})
        self.assertEqual(result.fix_value, "[EMAIL_REDACTED] and [EMAIL_REDACTED]")

    def test_valid_json_is_preserved(self):
        value = '{"note": "don\'t change ,} or ,]", "items": [1, 2]}'
        self.assertIsInstance(self.formatter.validate(value, {}), PassResult)
        module = importlib.import_module("04_guardrails_validator")
        self.assertEqual(module.make_json_guard().validate(value, num_reasks=0).validated_output, value)

    def test_fences_quotes_and_trailing_commas_are_repaired_together(self):
        fence = chr(96) * 3
        value = fence + "json\n{'name': 'Alice', 'items': [1, 2,],}\n" + fence
        self.assertEqual(self.repaired_json(value), {"name": "Alice", "items": [1, 2]})

    def test_string_delimiters_and_apostrophes_are_not_modified(self):
        value = "{\"note\": \"don't change ,} or ,]\", \"items\": [1,],}"
        self.assertEqual(self.repaired_json(value), {"note": "don't change ,} or ,]", "items": [1]})

    def test_single_quoted_escaped_apostrophe_and_double_quotes(self):
        value = "{'name': 'O\\'Neil', 'note': 'He said \"hi\".'}"
        self.assertEqual(self.repaired_json(value), {"name": "O'Neil", "note": 'He said "hi".'})

    def test_unrecoverable_inputs_return_json_error_without_raw_data(self):
        for value in ("not json", "{]", "{'x': 'unterminated}", '{"value": NaN}', '{"value": Infinity}'):
            with self.subTest(value=value):
                self.assertEqual(self.repaired_json(value), {"error": "Unable to parse JSON"})

    def test_guard_fix_returns_redacted_and_repaired_outputs(self):
        module = importlib.import_module("04_guardrails_validator")
        pii_output = module.make_pii_guard().validate("Email: user@example.com", num_reasks=0).validated_output
        json_output = module.make_json_guard().validate("{'ok': 1,}", num_reasks=0).validated_output
        self.assertEqual(pii_output, "Email: [EMAIL_REDACTED]")
        self.assertEqual(parse_json(json_output), {"ok": 1})


if __name__ == "__main__":
    unittest.main()
