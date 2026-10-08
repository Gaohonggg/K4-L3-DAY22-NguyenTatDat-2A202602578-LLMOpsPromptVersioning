"""Custom string validators for regex PII redaction and conservative JSON repair."""

from __future__ import annotations

import ast
import json
import re
from typing import Any

from guardrails.validators import FailResult, PassResult, Validator, register_validator


@register_validator(name="custom/pii-detector", data_type="string")
class PIIDetector(Validator):
    """Redact emails, US-format phone numbers/SSNs, and 16-digit card-like numbers.

    Patterns detect syntactic candidates rather than checking whether an account
    exists. Longer identifiers are processed first to avoid partial phone matches.
    """

    PII_PATTERNS = {
        "CREDIT_CARD": r"(?<!\w)(?:\d{4}[ -]?){3}\d{4}(?!\w)",
        "SSN": r"(?<!\w)\d{3}-\d{2}-\d{4}(?!\w)",
        "PHONE": r"(?<!\w)(?:\+?1[ .-]?)?(?:\(\d{3}\)|\d{3})[ .-]?\d{3}[ .-]?\d{4}(?!\w)",
        "EMAIL": r"(?<![\w.+%-])[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b",
    }

    def validate(self, value: str, metadata: dict[str, Any]) -> PassResult | FailResult:
        """Return a fix on detection; preserve clean input without modification."""
        redacted = value
        counts = {}
        for pii_type, pattern in self.PII_PATTERNS.items():
            redacted, count = re.subn(pattern, f"[{pii_type}_REDACTED]", redacted)
            if count:
                counts[pii_type] = count
        if counts:
            return FailResult(
                error_message=f"PII detected: {counts}",
                fix_value=redacted,
            )
        return PassResult()


def parse_json(text: str) -> Any:
    """Parse standards-compatible JSON, rejecting NaN and Infinity extensions."""
    def reject_constant(value: str) -> None:
        raise ValueError(f"Non-JSON numeric constant: {value}")

    return json.loads(text, parse_constant=reject_constant)


@register_validator(name="custom/json-formatter", data_type="string")
class JSONFormatter(Validator):
    """Repair fences, single-quoted string tokens, and structural trailing commas.

    Repairs preserve text inside double-quoted strings. Unrecoverable input is
    replaced with a JSON error object without reproducing the raw input.
    """

    @staticmethod
    def _normalize_quotes(text: str) -> str:
        """Convert complete single-quoted tokens without changing other strings."""
        parts = []
        index = 0
        while index < len(text):
            quote = text[index]
            if quote not in ("'", '"'):
                parts.append(quote)
                index += 1
                continue
            start = index
            index += 1
            while index < len(text):
                if text[index] == "\\":
                    index += 2
                elif text[index] == quote:
                    index += 1
                    break
                else:
                    index += 1
            else:
                raise ValueError("Unterminated string token")
            token = text[start:index]
            if quote == "'":
                value = ast.literal_eval(token)
                if not isinstance(value, str):
                    raise ValueError("Expected a string token")
                token = json.dumps(value, ensure_ascii=False)
            parts.append(token)
        return "".join(parts)

    @staticmethod
    def _remove_trailing_commas(text: str) -> str:
        """Remove commas before closing brackets only when outside a string."""
        parts = []
        in_string = False
        escaped = False
        for index, character in enumerate(text):
            if in_string:
                parts.append(character)
                if escaped:
                    escaped = False
                elif character == "\\":
                    escaped = True
                elif character == '"':
                    in_string = False
                continue
            if character == '"':
                in_string = True
            elif character == ",":
                remaining = text[index + 1:].lstrip()
                if remaining.startswith(("}", "]")):
                    continue
            parts.append(character)
        return "".join(parts)

    @classmethod
    def _repair(cls, text: str) -> str:
        """Strip enclosing fences, normalize quotes, then repair structural commas."""
        text = text.strip()
        fenced = re.fullmatch(r"```(?:json)?\s*\n?(.*?)\s*```", text, flags=re.DOTALL | re.IGNORECASE)
        if fenced:
            text = fenced.group(1).strip()
        return cls._remove_trailing_commas(cls._normalize_quotes(text))

    def validate(self, value: str, metadata: dict[str, Any]) -> PassResult | FailResult:
        """Preserve valid JSON; use FailResult.fix_value for repairs or fallback."""
        try:
            parse_json(value)
            return PassResult()
        except (json.JSONDecodeError, ValueError):
            pass
        try:
            parsed = parse_json(self._repair(value))
            return FailResult(
                error_message="Invalid JSON repaired",
                fix_value=json.dumps(parsed, indent=2, ensure_ascii=False, allow_nan=False),
            )
        except (json.JSONDecodeError, ValueError, SyntaxError):
            return FailResult(
                error_message="Unable to repair JSON",
                fix_value=json.dumps({"error": "Unable to parse JSON"}),
            )
