"""Validated model-provider configuration tests."""

from __future__ import annotations

import unittest
from unittest.mock import MagicMock

from model_defaults import (
    DEFAULT_PRODUCTION_MODEL,
    ModelConfigurationError,
    create_production_chat_model,
    get_production_model_settings,
)


class ProductionModelSettingsTests(unittest.TestCase):
    def test_project_settings_configure_openai_compatible_model(self) -> None:
        secret = "relay-secret-value"
        settings = get_production_model_settings(
            {
                "CODE_REVIEW_LLM_PROVIDER": "openai_compatible",
                "CODE_REVIEW_LLM_API_KEY": secret,
                "CODE_REVIEW_LLM_MODEL": "gpt-5.6-sol",
                "CODE_REVIEW_LLM_BASE_URL": "https://relay.example/v1/",
                "CODE_REVIEW_LLM_TIMEOUT_SECONDS": "300",
                "CODE_REVIEW_LLM_REASONING_EFFORT": "none",
                "CODE_REVIEW_LLM_MAX_COMPLETION_TOKENS": "8192",
            },
            require_api_key=True,
        )

        self.assertEqual(settings.provider, "openai_compatible")
        self.assertEqual(settings.model, "gpt-5.6-sol")
        self.assertEqual(settings.base_url, "https://relay.example/v1")
        self.assertEqual(settings.timeout_seconds, 300)
        self.assertEqual(settings.reasoning_effort, "none")
        self.assertEqual(settings.max_completion_tokens, 8192)
        self.assertNotIn(secret, repr(settings))

    def test_legacy_project_settings_beat_standard_openai_environment(self) -> None:
        settings = get_production_model_settings(
            {
                "PDE_FRONTIER_LLM_PROVIDER": "openai_compatible",
                "PDE_FRONTIER_LLM_API_KEY": "relay-key",
                "PDE_FRONTIER_LLM_MODEL": "gpt-5.6-sol",
                "PDE_FRONTIER_LLM_BASE_URL": "https://relay.example/v1",
                "OPENAI_API_KEY": "stale-direct-key",
            },
            require_api_key=True,
        )

        self.assertEqual(settings.api_key, "relay-key")
        self.assertEqual(settings.model, "gpt-5.6-sol")

    def test_model_factory_passes_compatible_provider_options(self) -> None:
        settings = get_production_model_settings(
            {
                "CODE_REVIEW_LLM_PROVIDER": "openai_compatible",
                "CODE_REVIEW_LLM_API_KEY": "relay-key",
                "CODE_REVIEW_LLM_MODEL": "gpt-5.6-sol",
                "CODE_REVIEW_LLM_BASE_URL": "https://relay.example/v1",
                "CODE_REVIEW_LLM_REASONING_EFFORT": "none",
                "CODE_REVIEW_LLM_MAX_COMPLETION_TOKENS": "8192",
                "CODE_REVIEW_LLM_TIMEOUT_SECONDS": "300",
            }
        )
        model_class = MagicMock()

        create_production_chat_model(
            chat_model_class=model_class,
            settings=settings,
        )

        arguments = model_class.call_args.kwargs
        self.assertEqual(arguments["model"], "gpt-5.6-sol")
        self.assertEqual(arguments["base_url"], "https://relay.example/v1")
        self.assertEqual(arguments["timeout"], 300)
        self.assertEqual(arguments["reasoning_effort"], "none")
        self.assertEqual(arguments["max_completion_tokens"], 8192)
        self.assertFalse(arguments["use_responses_api"])
        self.assertEqual(arguments["api_key"].get_secret_value(), "relay-key")

    def test_defaults_and_invalid_compatible_url_are_explicit(self) -> None:
        defaults = get_production_model_settings({})
        self.assertEqual(defaults.model, DEFAULT_PRODUCTION_MODEL)
        self.assertEqual(defaults.provider, "openai")
        with self.assertRaises(ModelConfigurationError):
            get_production_model_settings(
                {
                    "CODE_REVIEW_LLM_PROVIDER": "openai_compatible",
                    "CODE_REVIEW_LLM_BASE_URL": "[https://relay.example/v1]",
                }
            )


if __name__ == "__main__":
    unittest.main()
