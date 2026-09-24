"""LLM callback usage accounting and incomplete telemetry tests."""

from __future__ import annotations

import unittest
from typing import TypedDict
from uuid import uuid4

from langchain_core.language_models.fake_chat_models import FakeMessagesListChatModel
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.outputs import ChatGeneration, LLMResult
from langgraph.graph import END, START, StateGraph

from evaluation.runner import RepoGraphRun, _usage_updates
from evaluation.telemetry import EvaluationTelemetryCollector


def response(input_tokens: int = 10, output_tokens: int = 4) -> LLMResult:
    return LLMResult(
        generations=[
            [
                ChatGeneration(
                    message=AIMessage(
                        content="done",
                        usage_metadata={
                            "input_tokens": input_tokens,
                            "output_tokens": output_tokens,
                            "total_tokens": input_tokens + output_tokens,
                        },
                        response_metadata={"model_name": "resolved-model"},
                    )
                )
            ]
        ]
    )


class UsageTelemetryTests(unittest.TestCase):
    def test_langgraph_propagates_callback_to_nested_model_once(self) -> None:
        class State(TypedDict):
            done: bool

        collector = EvaluationTelemetryCollector()
        model = FakeMessagesListChatModel(
            responses=[
                AIMessage(
                    content="done",
                    usage_metadata={
                        "input_tokens": 7,
                        "output_tokens": 2,
                        "total_tokens": 9,
                    },
                )
            ]
        )

        def invoke_model(_state: State) -> dict[str, bool]:
            model.invoke([HumanMessage(content="work")])
            return {"done": True}

        workflow = StateGraph(State)
        workflow.add_node("model", invoke_model)
        workflow.add_edge(START, "model")
        workflow.add_edge("model", END)
        final = workflow.compile().invoke(
            {"done": False},
            config={"callbacks": [collector]},
        )
        self.assertTrue(final["done"])
        summary = collector.summary()
        self.assertEqual(summary.calls, 1)
        self.assertEqual(summary.total_tokens, 9)
        self.assertFalse(summary.incomplete)

    def test_one_llm_call_is_counted_once(self) -> None:
        collector = EvaluationTelemetryCollector()
        run_id = uuid4()
        collector.on_chat_model_start({"kwargs": {"model": "requested"}}, [], run_id=run_id)
        collector.on_llm_end(response(), run_id=run_id)
        summary = collector.summary()
        self.assertEqual(summary.calls, 1)
        self.assertEqual(summary.total_tokens, 14)
        self.assertFalse(summary.incomplete)
        self.assertEqual(summary.models, ["requested", "resolved-model"])

    def test_nested_duplicate_callback_does_not_double_count(self) -> None:
        collector = EvaluationTelemetryCollector()
        run_id = uuid4()
        collector.on_chat_model_start({}, [], run_id=run_id)
        collector.on_llm_start({}, [], run_id=run_id)
        collector.on_llm_end(response(), run_id=run_id)
        collector.on_llm_end(response(), run_id=run_id)
        self.assertEqual(collector.summary().calls, 1)
        self.assertEqual(collector.summary().input_tokens, 10)

    def test_retry_and_correction_calls_count_separately(self) -> None:
        collector = EvaluationTelemetryCollector()
        for _ in range(4):
            run_id = uuid4()
            collector.on_llm_start({}, [], run_id=run_id)
            collector.on_llm_end(response(2, 1), run_id=run_id)
        summary = collector.summary()
        self.assertEqual(summary.calls, 4)
        self.assertEqual(summary.input_tokens, 8)
        self.assertEqual(summary.output_tokens, 4)

    def test_missing_usage_is_incomplete_and_never_estimated(self) -> None:
        collector = EvaluationTelemetryCollector()
        known = uuid4()
        missing = uuid4()
        collector.on_llm_start({}, ["a very long prompt"], run_id=known)
        collector.on_llm_end(response(3, 2), run_id=known)
        collector.on_llm_start({}, ["another prompt"], run_id=missing)
        collector.on_llm_end(LLMResult(generations=[]), run_id=missing)
        summary = collector.summary()
        self.assertEqual(summary.calls, 2)
        self.assertEqual(summary.total_tokens, 5)
        self.assertTrue(summary.incomplete)
        self.assertTrue(any("unavailable" in warning for warning in summary.warnings))

    def test_error_and_missing_completion_are_incomplete(self) -> None:
        collector = EvaluationTelemetryCollector()
        failed = uuid4()
        pending = uuid4()
        collector.on_llm_start({}, [], run_id=failed)
        collector.on_llm_error(RuntimeError("secret"), run_id=failed)
        collector.on_llm_start({}, [], run_id=pending)
        summary = collector.summary()
        self.assertEqual(summary.calls, 2)
        self.assertIsNone(summary.input_tokens)
        self.assertTrue(summary.incomplete)
        self.assertNotIn("secret", " ".join(summary.warnings))

    def test_usage_summary_propagates_to_result_fields(self) -> None:
        collector = EvaluationTelemetryCollector()
        run_id = uuid4()
        collector.on_llm_start({}, [], run_id=run_id)
        collector.on_llm_end(response(), run_id=run_id)
        updates = _usage_updates(RepoGraphRun(llm_usage=collector.summary()))
        self.assertEqual(updates["llm_calls"], 1)
        self.assertEqual(updates["total_tokens"], 14)
        self.assertEqual(updates["resolved_model_names"], ["resolved-model"])


if __name__ == "__main__":
    unittest.main()
