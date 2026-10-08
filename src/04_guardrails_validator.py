"""Step 4: demonstrate custom PII and JSON validators with Guardrails FIX.

Run from the repository root:
    .venv/bin/python src/04_guardrails_validator.py --demo pii
    .venv/bin/python src/04_guardrails_validator.py --demo json

All demo PII is synthetic. No LLM or provider API is required.
"""

from __future__ import annotations

import argparse

from guardrails import Guard
from guardrails.types.on_fail import OnFailAction
from guardrails.validators import PassResult

from utils.guardrails_validators import JSONFormatter, PIIDetector, parse_json


def make_pii_guard() -> Guard:
    """Apply FIX in the validator constructor and disable metrics telemetry."""
    guard = Guard().use(PIIDetector(on_fail=OnFailAction.FIX))
    guard.configure(allow_metrics_collection=False)
    return guard


def make_json_guard() -> Guard:
    """Create the JSON repair guard without enabling LLM reasks."""
    guard = Guard().use(JSONFormatter(on_fail=OnFailAction.FIX))
    guard.configure(allow_metrics_collection=False)
    return guard


def demo_pii_guard() -> None:
    """Check six cases, including all four PII types, mixed PII, and clean input."""
    guard = make_pii_guard()
    cases = [
        ("Email", "Contact John at john.doe@example.com for details.",
         "Contact John at [EMAIL_REDACTED] for details."),
        ("Phone", "Call our support line at (555) 867-5309.",
         "Call our support line at [PHONE_REDACTED]."),
        ("SSN", "Patient SSN is 123-45-6789 on file.",
         "Patient SSN is [SSN_REDACTED] on file."),
        ("Credit Card", "Payment made with card 4532 1234 5678 9010.",
         "Payment made with card [CREDIT_CARD_REDACTED]."),
        ("Multi-PII", "Email: alice@example.com, Phone: 555-123-4567",
         "Email: [EMAIL_REDACTED], Phone: [PHONE_REDACTED]"),
        ("Clean", "No sensitive information in this text.",
         "No sensitive information in this text."),
    ]
    print("PII DETECTION DEMO | synthetic inputs | on_fail=FIX", flush=True)
    for label, text, expected in cases:
        outcome = guard.validate(text, num_reasks=0)
        output = outcome.validated_output
        print(f"\n[{label}]\nInput: {text}\nOutput: {output}", flush=True)
        if output != expected:
            raise AssertionError(f"{label}: redaction output differs from expected")
        if not isinstance(PIIDetector().validate(output, {}), PassResult):
            raise AssertionError(f"{label}: output still contains a PII pattern")
        print(f"CHECK PASS | Guardrails validation_passed={outcome.validation_passed}", flush=True)
    print(f"\nPII SUMMARY: passed={len(cases)} | failed=0", flush=True)


def demo_json_guard() -> None:
    """Verify repaired/fallback JSON and unchanged content in tricky string values."""
    guard = make_json_guard()
    fence = chr(96) * 3
    cases = [
        ("Valid JSON", '{"name": "Alice", "age": 30}', {"name": "Alice", "age": 30}),
        ("Markdown fences", fence + 'json\n{"name": "Bob"}\n' + fence, {"name": "Bob"}),
        ("Single quotes", "{'name': 'Charlie', 'score': 95}", {"name": "Charlie", "score": 95}),
        ("Trailing comma", '{"key": "value",}', {"key": "value"}),
        ("Truly invalid", "This is not JSON at all: ??? {]", {"error": "Unable to parse JSON"}),
        ("Apostrophe preserved", "{\"note\": \"don't change this\",}", {"note": "don't change this"}),
        ("Comma inside string", '{"note": "keep ,} and ,]", "items": [1, 2,],}',
         {"note": "keep ,} and ,]", "items": [1, 2]}),
    ]
    print("JSON REPAIR DEMO | on_fail=FIX | no raw input in fallback", flush=True)
    for label, text, expected in cases:
        outcome = guard.validate(text, num_reasks=0)
        output = outcome.validated_output
        print(f"\n[{label}]\nInput: {text}\nOutput: {output}", flush=True)
        if not isinstance(output, str) or parse_json(output) != expected:
            raise AssertionError(f"{label}: output is invalid or changes the expected content")
        if label == "Valid JSON" and output != text:
            raise AssertionError("Valid JSON should be preserved")
        print(f"CHECK PASS | Guardrails validation_passed={outcome.validation_passed}", flush=True)
    print(f"\nJSON SUMMARY: passed={len(cases)} | failed=0", flush=True)


def main(*, demo: str = "all") -> None:
    """Run both demos by default; keep CLI parsing separate for run_all.py."""
    if demo not in ("all", "pii", "json"):
        raise ValueError("demo must be all, pii, or json")
    if demo in ("all", "pii"):
        demo_pii_guard()
    if demo in ("all", "json"):
        demo_json_guard()
    print("\nSTEP 4 PASSED", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--demo", choices=("all", "pii", "json"), default="all")
    main(demo=parser.parse_args().demo)
