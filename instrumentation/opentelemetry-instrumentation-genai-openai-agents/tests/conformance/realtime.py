"""Realtime response conformance using simulated WebSocket server events."""

from __future__ import annotations

import asyncio
from typing import Any
from unittest.mock import AsyncMock, patch

from agents import FunctionTool
from agents.realtime.model_events import RealtimeModelToolCallEvent

from opentelemetry.instrumentation.genai.openai_agents import (
    OpenAIAgentsInstrumentor,
)
from opentelemetry.sdk._logs import LoggerProvider
from opentelemetry.sdk.metrics import MeterProvider
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.test.weaver_live_check import LiveCheckReport
from opentelemetry.test_util_genai.conformance import (
    ExpectedViolation,
    Scenario,
)
from opentelemetry.test_util_genai.instrumentor import instrument

from ..test_realtime import _event, _session


class RealtimeScenario(Scenario):
    """The pinned registry predates semantic-conventions-genai#394.

    Session correlation and lifecycle events are asserted below until the
    shared registry can advance alongside the repo-wide metric migration.
    """

    expected_violations = (
        ExpectedViolation(
            "missing_attribute",
            "Attribute 'gen_ai.realtime_session.id' does not exist in the registry.",
        ),
        ExpectedViolation(
            "missing_event",
            "Event 'gen_ai.client.realtime_session.started' does not exist in the registry.",
        ),
        ExpectedViolation(
            "missing_event",
            "Event 'gen_ai.client.realtime_session.ended' does not exist in the registry.",
        ),
    )
    expected_spans = {
        "realtime_inference": 1,
        "execute_tool": 1,
    }
    expected_metrics = (
        "gen_ai.client.operation.duration",
        "gen_ai.client.token.usage",
        "gen_ai.client.operation.time_to_first_chunk",
        "gen_ai.execute_tool.duration",
    )

    def run(
        self,
        *,
        tracer_provider: TracerProvider,
        meter_provider: MeterProvider,
        logger_provider: LoggerProvider,
        vcr: Any,
    ) -> None:
        async def get_weather(context: Any, arguments: str) -> str:
            return "sunny"

        with instrument(
            OpenAIAgentsInstrumentor(),
            tracer_provider=tracer_provider,
            meter_provider=meter_provider,
            logger_provider=logger_provider,
        ):
            session = _session(
                tools=[
                    FunctionTool(
                        name="get_weather",
                        description="Weather",
                        params_json_schema={},
                        on_invoke_tool=get_weather,
                    )
                ]
            )

            async def run_session() -> None:
                events = [
                    _event(
                        "session.created",
                        session={
                            "id": "sess_test",
                            "model": "gpt-realtime",
                            "output_modalities": ["audio"],
                        },
                    ),
                    _event("response.created", response={"id": "resp_test"}),
                    _event(
                        "response.output_audio.delta", response_id="resp_test"
                    ),
                    _event(
                        "response.function_call_arguments.done",
                        response_id="resp_test",
                        call_id="call_test",
                        name="get_weather",
                        arguments='{"city":"Paris"}',
                    ),
                    _event(
                        "response.done",
                        response={
                            "id": "resp_test",
                            "status": "completed",
                            "usage": {
                                "input_tokens": 10,
                                "output_tokens": 5,
                                "input_token_details": {
                                    "audio_tokens": 8,
                                    "text_tokens": 2,
                                    "cached_tokens": 4,
                                    "cached_tokens_details": {
                                        "audio_tokens": 3,
                                        "text_tokens": 1,
                                    },
                                },
                                "output_token_details": {
                                    "audio_tokens": 4,
                                    "text_tokens": 1,
                                },
                            },
                        },
                    ),
                    _event(
                        "conversation.item.added",
                        item={
                            "type": "function_call_output",
                            "call_id": "call_test",
                            "output": "sunny",
                        },
                    ),
                ]
                for event in events:
                    await session.model._emit_event(event)
                    if event.data["type"] == "response.done":
                        with patch.object(
                            session.model, "send_event", new=AsyncMock()
                        ):
                            await session._handle_tool_call(
                                RealtimeModelToolCallEvent(
                                    call_id="call_test",
                                    name="get_weather",
                                    arguments='{"city":"Paris"}',
                                    id="item_test",
                                )
                            )
                await session.close()

            asyncio.run(run_session())

    def validate(self, report: LiveCheckReport) -> None:
        super().validate(report)
        response = next(
            sample["span"]
            for sample in report["samples"]
            if "span" in sample
            and sample["span"]["name"] == "realtime_inference gpt-realtime"
        )
        attributes = {
            attr["name"]: attr["value"] for attr in response["attributes"]
        }
        assert attributes["gen_ai.realtime_session.id"] == "sess_test"
        assert "gen_ai.conversation.id" not in attributes
        assert attributes["gen_ai.output.type"] == "speech"
        for name, value in {
            "gen_ai.usage.audio.input_tokens": 8,
            "gen_ai.usage.text.input_tokens": 2,
            "gen_ai.usage.audio.output_tokens": 4,
            "gen_ai.usage.text.output_tokens": 1,
            "gen_ai.usage.cache_read.input_tokens": 4,
            "gen_ai.usage.audio.cache_read.input_tokens": 3,
            "gen_ai.usage.text.cache_read.input_tokens": 1,
        }.items():
            assert attributes[name] == value
            assert type(attributes[name]) is int
        events = [
            sample["log"] for sample in report["samples"] if "log" in sample
        ]
        assert [event["event_name"] for event in events] == [
            "gen_ai.client.realtime_session.started",
            "gen_ai.client.realtime_session.ended",
        ]
        for event in events:
            assert event["severity_number"] == 9
            attributes = {
                attr["name"]: attr["value"] for attr in event["attributes"]
            }
            assert attributes == {
                "gen_ai.provider.name": "openai",
                "gen_ai.request.model": "gpt-realtime",
                "gen_ai.realtime_session.id": "sess_test",
                "server.address": "api.openai.com",
                "server.port": 443,
            }
