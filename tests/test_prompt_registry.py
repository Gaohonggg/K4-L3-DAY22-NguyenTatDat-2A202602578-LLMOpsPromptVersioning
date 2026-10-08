"""Offline checks for stable routing and mandatory Hub-backed prompt loading."""

import sys
import unittest
from pathlib import Path
from unittest.mock import Mock

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from utils.prompt_registry import (
    LOCAL_PROMPTS,
    PROMPT_V1_NAME,
    PROMPT_V2_NAME,
    get_prompt_version,
    pull_prompts_from_hub,
)


class PromptRegistryTests(unittest.TestCase):
    """Test lab invariants without making LLM or LangSmith requests."""

    def test_known_md5_assignments(self):
        # Known MD5 digests: a ends in 1 (odd), abc ends in 2 (even).
        self.assertEqual(get_prompt_version("a"), PROMPT_V2_NAME)
        self.assertEqual(get_prompt_version("abc"), PROMPT_V1_NAME)

    def test_batch_routes_to_both_versions_and_repeats_identically(self):
        ids = [f"req-{index:04d}" for index in range(50)]
        first = {request_id: get_prompt_version(request_id) for request_id in ids}
        repeated = {request_id: get_prompt_version(request_id) for request_id in reversed(ids)}
        self.assertEqual(first, repeated)
        self.assertEqual(set(first.values()), {PROMPT_V1_NAME, PROMPT_V2_NAME})

    def test_empty_request_id_is_rejected(self):
        for value in ("", " ", None):
            with self.subTest(value=value), self.assertRaises(ValueError):
                get_prompt_version(value)

    def test_returns_remote_templates_with_commit_metadata(self):
        remote = {
            name: prompt.model_copy(update={"metadata": {"lc_hub_commit_hash": "abc123"}})
            for name, prompt in LOCAL_PROMPTS.items()
        }
        client = Mock()
        client.pull_prompt.side_effect = lambda name, **kwargs: remote[name]
        loaded = pull_prompts_from_hub(client)
        for name in LOCAL_PROMPTS:
            self.assertIs(loaded[name], remote[name])
            client.pull_prompt.assert_any_call(name, include_model=False, skip_cache=True)

    def test_hub_failure_is_not_replaced_with_local_fallback(self):
        client = Mock()
        client.pull_prompt.side_effect = ConnectionError("Hub unavailable")
        with self.assertRaises(ConnectionError):
            pull_prompts_from_hub(client)

    def test_wrong_remote_content_is_rejected(self):
        client = Mock()
        client.pull_prompt.return_value = LOCAL_PROMPTS[PROMPT_V2_NAME]
        with self.assertRaises(ValueError):
            pull_prompts_from_hub(client)

    def test_missing_commit_metadata_is_rejected(self):
        client = Mock()
        client.pull_prompt.return_value = LOCAL_PROMPTS[PROMPT_V1_NAME]
        with self.assertRaises(ValueError):
            pull_prompts_from_hub(client)


if __name__ == "__main__":
    unittest.main()
