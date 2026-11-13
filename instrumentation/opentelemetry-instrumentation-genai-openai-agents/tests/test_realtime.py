from __future__ import annotations

import asyncio
from types import SimpleNamespace
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from opentelemetry._logs import LogRecord, SeverityNumber
from opentelemetry.context import get_current
from opentelemetry.instrumentation.genai.openai_agents import (
    OpenAIAgentsInstrumentor,
)
from opentelemetry.instrumentation.genai.openai_agents.realtime import (
    RealtimeInstrumentation,
    RealtimeTelemetryHandler,
    _token_count,
)
from opentelemetry.instrumentation.utils import suppress_instrumentation
from opentelemetry.sdk._logs.export import InMemoryLogRecordExporter
from opentelemetry.test_util_genai.instrumentor import instrument
from opentelemetry.trace import SpanKind, get_current_span
from opentelemetry.util.genai.handler import TelemetryHandler


def _event(event_type: str, **data: Any) -> SimpleNamespace:
    return SimpleNamespace(
        type="raw_server_event", data={"type": event_type, **data}
    )


@pytest.mark.parametrize(
    "totals", [{"input_tokens": 12, "output_tokens": 8}, {}]
)
def test_realtime_session_response(
    tracer_provider,
    meter_provider,
    logger_provider,
    metric_reader,
    span_exporter,
    log_exporter,
    totals,
) -> None:
    handler = RealtimeTelemetryHandler(
        TelemetryHandler(
            tracer_provider=tracer_provider,
            meter_provider=meter_provider,
            logger_provider=logger_provider,
        ),
        "voice_agent",
    )
    original_context = get_current()
    handler.handle_event(
        _event(
            "session.created",
            session={"id": "sess_1", "model": "gpt-realtime"},
        )
    )
    handler.handle_event(_event("response.created", response={"id": "resp_1"}))
    handler.handle_event(
        _event("response.output_audio.delta", response_id="resp_1")
    )
    handler.handle_event(
        _event(
            "response.done",
            response={
                "id": "resp_1",
                "status": "completed",
                "usage": {
                    **totals,
                    "input_token_details": {
                        "audio_tokens": 10,
                        "text_tokens": 2,
                        "cached_tokens": 5,
                        "cached_tokens_details": {
                            "audio_tokens": 4,
                            "text_tokens": 1,
                        },
                    },
                    "output_token_details": {
                        "audio_tokens": 6,
                        "text_tokens": 2,
                    },
                },
            },
        )
    )
    assert get_current() is original_context
    handler.cleanup()
    (response,) = span_exporter.get_finished_spans()
    assert response.parent is None
    assert response.kind is SpanKind.CLIENT
    assert response.name == "realtime_inference gpt-realtime"
    assert response.attributes["gen_ai.operation.name"] == "realtime_inference"
    assert response.attributes["gen_ai.realtime_session.id"] == "sess_1"
    assert type(response.attributes["gen_ai.realtime_session.id"]) is str
    assert "gen_ai.conversation.id" not in response.attributes
    assert response.attributes["gen_ai.usage.input_tokens"] == 12
    assert response.attributes["gen_ai.usage.output_tokens"] == 8
    assert response.attributes["gen_ai.usage.audio.input_tokens"] == 10
    assert response.attributes["gen_ai.usage.audio.output_tokens"] == 6
    assert response.attributes["gen_ai.usage.text.input_tokens"] == 2
    assert response.attributes["gen_ai.usage.text.output_tokens"] == 2
    assert response.attributes["gen_ai.usage.cache_read.input_tokens"] == 5
    assert (
        response.attributes["gen_ai.usage.audio.cache_read.input_tokens"] == 4
    )
    assert (
        response.attributes["gen_ai.usage.text.cache_read.input_tokens"] == 1
    )
    assert type(response.attributes["gen_ai.usage.audio.input_tokens"]) is int
    assert type(response.attributes["gen_ai.usage.audio.output_tokens"]) is int
    assert response.attributes["gen_ai.response.finish_reasons"] == (
        "completed",
    )
    handler.cleanup()
    assert len(span_exporter.get_finished_spans()) == 1
    started, ended = [
        entry.log_record for entry in log_exporter.get_finished_logs()
    ]
    assert started.event_name == "gen_ai.client.realtime_session.started"
    assert ended.event_name == "gen_ai.client.realtime_session.ended"
    for event in (started, ended):
        assert event.severity_number is SeverityNumber.INFO
        assert event.attributes == {
            "gen_ai.provider.name": "openai",
            "gen_ai.request.model": "gpt-realtime",
            "gen_ai.realtime_session.id": "sess_1",
        }
    metrics = {
        metric.name: metric
        for resource in metric_reader.get_metrics_data().resource_metrics
        for scope in resource.scope_metrics
        for metric in scope.metrics
    }
    assert "gen_ai.client.operation.duration" in metrics
    assert "gen_ai.client.operation.time_to_first_chunk" in metrics
    assert "gen_ai.invoke_agent.duration" not in metrics
    usage = metrics["gen_ai.client.token.usage"].data.data_points
    assert {
        point.attributes["gen_ai.token.type"]: point.sum for point in usage
    } == {"input": 12, "output": 8}
    assert all(
        point.attributes["gen_ai.operation.name"] == "realtime_inference"
        for point in usage
    )
    assert all(
        "gen_ai.realtime_session.id" not in point.attributes for point in usage
    )


@pytest.mark.parametrize(
    "session_config,response_config,expected",
    [
        ({}, {}, None),
        ({"output_modalities": ["audio"]}, {}, "speech"),
        ({"modalities": ["text", "audio"]}, {}, "speech"),
        ({"output_modalities": ["text"]}, {}, "text"),
        (
            {"output_modalities": ["audio"]},
            {"output_modalities": ["text"]},
            "text",
        ),
        ({}, {"modalities": ["audio"]}, "speech"),
    ],
)
def test_realtime_output_type(
    tracer_provider, span_exporter, session_config, response_config, expected
) -> None:
    handler = RealtimeTelemetryHandler(
        TelemetryHandler(tracer_provider=tracer_provider), "voice_agent"
    )
    handler.handle_event(_event("session.created", session={"id": "sess_1"}))
    handler.handle_event(_event("session.updated", session=session_config))
    handler.handle_event(
        _event("session.updated", session={"model": "gpt-realtime"})
    )
    handler.handle_event(
        _event(
            "response.created", response={"id": "resp_1", **response_config}
        )
    )
    handler.handle_event(
        _event("response.output_audio.delta", response_id="resp_1")
    )
    handler.handle_event(
        _event(
            "response.done", response={"id": "resp_1", "status": "completed"}
        )
    )
    handler.cleanup()
    (response,) = span_exporter.get_finished_spans()
    assert response.attributes.get("gen_ai.output.type") == expected


@pytest.mark.parametrize(
    "response_config,expected",
    [
        ({}, "conv_1"),
        ({"conversation_id": "conv_2"}, "conv_2"),
        ({"conversation_id": None}, None),
    ],
)
def test_realtime_conversation_id(
    tracer_provider, span_exporter, response_config, expected
) -> None:
    handler = RealtimeTelemetryHandler(
        TelemetryHandler(tracer_provider=tracer_provider), "voice_agent"
    )
    handler.handle_event(_event("session.created", session={"id": "sess_1"}))
    handler.handle_event(
        _event("conversation.created", conversation={"id": "conv_1"})
    )
    handler.handle_event(
        _event(
            "response.created", response={"id": "resp_1", **response_config}
        )
    )
    handler.handle_event(
        _event(
            "response.done", response={"id": "resp_1", "status": "completed"}
        )
    )
    handler.cleanup()
    (response,) = span_exporter.get_finished_spans()
    assert response.attributes.get("gen_ai.conversation.id") == expected
    assert response.attributes["gen_ai.realtime_session.id"] == "sess_1"


def _session_events(exporter: InMemoryLogRecordExporter) -> list[LogRecord]:
    return [
        entry.log_record
        for entry in exporter.get_finished_logs()
        if entry.log_record.event_name
        in {
            "gen_ai.client.realtime_session.started",
            "gen_ai.client.realtime_session.ended",
        }
    ]


def test_realtime_cross_task_cleanup(tracer_provider, span_exporter) -> None:
    handler = RealtimeTelemetryHandler(
        TelemetryHandler(tracer_provider=tracer_provider), "voice_agent"
    )

    async def run() -> None:
        original_context = get_current()

        async def receive() -> None:
            handler.handle_event(
                _event("session.created", session={"id": "sess_1"})
            )
            handler.handle_event(
                _event("response.created", response={"id": "resp_1"})
            )
            assert get_current() is original_context

        await asyncio.create_task(receive())
        handler.cleanup(ConnectionError("disconnected"))
        assert get_current() is original_context

    asyncio.run(run())
    spans = span_exporter.get_finished_spans()
    assert len(spans) == 1
    assert all(
        span.attributes["error.type"] == "ConnectionError" for span in spans
    )


@pytest.mark.parametrize("capture", [False, True])
def test_realtime_tool_capture(
    tracer_provider, span_exporter, monkeypatch, capture
) -> None:
    if capture:
        monkeypatch.setenv(
            "OTEL_INSTRUMENTATION_GENAI_CAPTURE_MESSAGE_CONTENT", "SPAN_ONLY"
        )
    else:
        monkeypatch.delenv(
            "OTEL_INSTRUMENTATION_GENAI_CAPTURE_MESSAGE_CONTENT", raising=False
        )
    handler = RealtimeTelemetryHandler(
        TelemetryHandler(tracer_provider=tracer_provider), "voice_agent"
    )
    handler.handle_event(_event("session.created", session={"id": "sess_1"}))
    handler.handle_event(_event("response.created", response={"id": "resp_1"}))
    handler.handle_event(
        _event(
            "response.function_call_arguments.done",
            response_id="resp_1",
            call_id="call_1",
            name="weather",
            arguments='{"city":"Paris"}',
        )
    )
    tool_invocation = handler.start_tool(
        SimpleNamespace(
            call_id="call_1", name="weather", arguments='{"city":"Paris"}'
        )
    )
    tool_context = tool_invocation.context
    handler.handle_event(
        _event(
            "response.done", response={"id": "resp_1", "status": "completed"}
        )
    )
    handler.finish_tool("call_1", "sunny")
    handler.cleanup()
    response, tool = span_exporter.get_finished_spans()
    assert tool.parent.span_id == response.context.span_id
    assert response.parent is None
    assert (
        get_current_span(tool_context).get_span_context().span_id
        == tool.context.span_id
    )
    assert tool.attributes["gen_ai.tool.call.id"] == "call_1"
    assert tool.attributes["gen_ai.operation.name"] == "execute_tool"
    if capture:
        assert (
            tool.attributes["gen_ai.tool.call.arguments"] == '{"city":"Paris"}'
        )
        assert tool.attributes["gen_ai.tool.call.result"] == "sunny"
    else:
        assert "gen_ai.tool.call.arguments" not in tool.attributes
        assert "gen_ai.tool.call.result" not in tool.attributes


@pytest.mark.parametrize(
    "status,reason,expected",
    [
        ("completed", None, "completed"),
        ("failed", None, "error"),
        ("incomplete", None, "error"),
        ("incomplete", "max_output_tokens", "max_output_tokens"),
        ("cancelled", None, "interrupted"),
        ("cancelled", "turn_detected", "turn_detected"),
        ("cancelled", "client_cancelled", "client_cancelled"),
    ],
)
def test_realtime_response_status(
    tracer_provider,
    logger_provider,
    span_exporter,
    log_exporter,
    status,
    reason,
    expected,
) -> None:
    handler = RealtimeTelemetryHandler(
        TelemetryHandler(
            tracer_provider=tracer_provider, logger_provider=logger_provider
        ),
        "voice_agent",
    )
    handler.handle_event(_event("session.created", session={"id": "sess_1"}))
    handler.handle_event(_event("response.created", response={"id": "resp_1"}))
    handler.handle_event(
        _event(
            "response.done",
            response={
                "id": "resp_1",
                "status": status,
                "status_details": {"reason": reason},
            },
        )
    )
    handler.cleanup()
    (response,) = span_exporter.get_finished_spans()
    assert response.attributes["gen_ai.response.finish_reasons"] == (expected,)
    assert ("error.type" in response.attributes) == (status == "failed")
    ended = log_exporter.get_finished_logs()[-1].log_record
    assert ended.severity_number is SeverityNumber.INFO
    assert "error.type" not in ended.attributes


@pytest.mark.parametrize(
    "error", [None, ConnectionError("closed"), asyncio.CancelledError()]
)
def test_realtime_unfinished_response(
    tracer_provider, logger_provider, span_exporter, log_exporter, error
) -> None:
    handler = RealtimeTelemetryHandler(
        TelemetryHandler(
            tracer_provider=tracer_provider, logger_provider=logger_provider
        ),
        "voice_agent",
    )
    handler.handle_event(_event("session.created", session={"id": "sess_1"}))
    handler.handle_event(_event("response.created", response={"id": "resp_1"}))
    if error is None:
        handler.handle_event(SimpleNamespace(type="end_of_stream"))
    else:
        handler.handle_event(
            SimpleNamespace(type="exception", exception=error)
        )
    handler.cleanup()
    handler.handle_event(
        _event(
            "response.done", response={"id": "resp_1", "status": "completed"}
        )
    )
    (response,) = span_exporter.get_finished_spans()
    assert response.attributes["gen_ai.response.finish_reasons"] == ("error",)
    assert ("error.type" in response.attributes) == (error is not None)
    assert len(_session_events(log_exporter)) == 2
    ended = _session_events(log_exporter)[-1]
    assert ("error.type" in ended.attributes) == (error is not None)
    assert ended.severity_number is (
        SeverityNumber.INFO if error is None else SeverityNumber.WARN
    )


def test_realtime_optional_dependencies() -> None:
    realtime = RealtimeInstrumentation(MagicMock())
    with patch(
        "opentelemetry.instrumentation.genai.openai_agents.realtime.import_module",
        side_effect=ImportError,
    ):
        realtime.instrument()
    realtime.uninstrument()


def test_realtime_tool_can_invoke_model(
    tracer_provider, span_exporter
) -> None:
    telemetry = TelemetryHandler(tracer_provider=tracer_provider)
    handler = RealtimeTelemetryHandler(telemetry, "voice_agent")
    handler.handle_event(_event("session.created", session={"id": "sess_1"}))
    handler.handle_event(_event("response.created", response={"id": "resp_1"}))
    handler.handle_event(
        _event(
            "response.function_call_arguments.done",
            response_id="resp_1",
            call_id="call_1",
            name="weather",
            arguments="{}",
        )
    )
    tool_invocation = handler.start_tool(
        SimpleNamespace(call_id="call_1", name="weather", arguments="{}")
    )
    tool_context = tool_invocation.context
    child = telemetry.inference(
        "openai",
        request_model="gpt-4o-mini",
        context=tool_context,
        _attach_to_context=False,
    )
    child.stop()
    handler.cleanup()
    child_span, tool_span, response_span = span_exporter.get_finished_spans()
    assert child_span.name == "chat gpt-4o-mini"
    assert child_span.parent.span_id == tool_span.context.span_id
    assert tool_span.parent.span_id == response_span.context.span_id


@pytest.mark.parametrize(
    "usage,expected",
    [
        ({}, None),
        ({"input_tokens": 0}, 0),
        ({"input_token_details": {"audio_tokens": 10, "text_tokens": 2}}, 12),
        ({"input_tokens": 7, "input_token_details": {"audio_tokens": 10}}, 7),
    ],
)
def test_realtime_token_fallback(usage, expected) -> None:
    assert _token_count(usage, "input") == expected


def _session(
    *,
    tools: list[Any] | None = None,
    model_config: dict[str, Any] | None = None,
) -> Any:
    from agents.realtime import RealtimeAgent
    from agents.realtime.openai_realtime import OpenAIRealtimeWebSocketModel
    from agents.realtime.session import RealtimeSession

    return RealtimeSession(
        model=OpenAIRealtimeWebSocketModel(),
        agent=RealtimeAgent(name="voice_agent", tools=tools or []),
        context=None,
        model_config=model_config,
    )


@pytest.mark.parametrize(
    "error", [None, ValueError("tool failed"), asyncio.CancelledError()]
)
@pytest.mark.parametrize("capture", [False, True])
def test_realtime_sdk_tool_execution(
    tracer_provider,
    meter_provider,
    logger_provider,
    span_exporter,
    error,
    capture,
) -> None:
    from agents import FunctionTool
    from agents.realtime.model_events import RealtimeModelToolCallEvent

    result = {"forecast": "sunny"}
    inside_context = None
    invocation_started = None

    async def invoke(context: Any, arguments: str) -> Any:
        nonlocal inside_context, invocation_started
        inside_context = get_current_span()
        invocation_started = inside_context.start_time
        assert inside_context.is_recording()
        assert context.tool_call_id == "call_1"
        assert arguments == '{"city":"Paris"}'
        child = TelemetryHandler(tracer_provider=tracer_provider).inference(
            "openai", request_model="nested-model"
        )
        child.stop()
        if error is not None:
            raise error
        return result

    tool = FunctionTool(
        name="weather",
        description="Weather",
        params_json_schema={},
        on_invoke_tool=invoke,
    )
    original = tool.on_invoke_tool
    with instrument(
        OpenAIAgentsInstrumentor(),
        tracer_provider=tracer_provider,
        meter_provider=meter_provider,
        logger_provider=logger_provider,
        content_capture="SPAN_ONLY" if capture else "NO_CONTENT",
    ):
        session = _session(tools=[tool])

        async def send_result(event: Any) -> None:
            assert event.tool_call.call_id == "call_1"
            assert not inside_context.is_recording()

        async def run() -> None:
            original_context = get_current()
            await session.model._emit_event(
                _event("session.created", session={"id": "sess_1"})
            )
            await session.model._emit_event(
                _event("response.created", response={"id": "resp_1"})
            )
            await session.model._emit_event(
                _event(
                    "response.function_call_arguments.done",
                    response_id="resp_1",
                    call_id="call_1",
                    name="weather",
                    arguments='{"city":"Paris"}',
                )
            )
            await session.model._emit_event(
                _event(
                    "response.done",
                    response={"id": "resp_1", "status": "completed"},
                )
            )
            (response,) = span_exporter.get_finished_spans()
            assert invocation_started is None
            event = RealtimeModelToolCallEvent(
                call_id="call_1",
                name="weather",
                arguments='{"city":"Paris"}',
                id="item_1",
            )
            with patch.object(
                session.model, "send_event", side_effect=send_result
            ):
                if error is None:
                    await session._handle_tool_call(event)
                else:
                    with pytest.raises(type(error)) as raised:
                        await session._handle_tool_call(event)
                    assert raised.value is error
            assert invocation_started >= response.end_time
            assert get_current() is original_context
            await session.close()

        asyncio.run(run())
    assert tool.on_invoke_tool is original
    response, child, tool_span = span_exporter.get_finished_spans()
    assert tool_span.parent.span_id == response.context.span_id
    assert child.parent.span_id == tool_span.context.span_id
    assert tool_span.attributes["gen_ai.tool.call.id"] == "call_1"
    assert ("error.type" in tool_span.attributes) == (error is not None)
    assert ("gen_ai.tool.call.arguments" in tool_span.attributes) == capture
    assert ("gen_ai.tool.call.result" in tool_span.attributes) == (
        capture and error is None
    )


def test_realtime_shared_tool_concurrency(
    tracer_provider, meter_provider, logger_provider, span_exporter
) -> None:
    from agents import FunctionTool
    from agents.realtime.model_events import RealtimeModelToolCallEvent

    contexts = []
    result = {"forecast": "sunny"}

    async def run() -> None:
        both_started = asyncio.Event()

        async def invoke(context: Any, arguments: str) -> Any:
            if context.tool_call_id != "call_shared":
                assert not get_current_span().is_recording()
                return result
            current = get_current_span()
            contexts.append(current)
            if len(contexts) == 2:
                both_started.set()
            await asyncio.wait_for(both_started.wait(), timeout=5)
            assert get_current_span() is current
            assert current.is_recording()
            return result

        tool = FunctionTool(
            name="weather",
            description="Weather",
            params_json_schema={},
            on_invoke_tool=invoke,
        )
        with instrument(
            OpenAIAgentsInstrumentor(),
            tracer_provider=tracer_provider,
            meter_provider=meter_provider,
            logger_provider=logger_provider,
        ):
            sessions = [_session(tools=[tool]), _session(tools=[tool])]
            for session in sessions:
                await session.model._emit_event(
                    _event("session.created", session={"id": "sess_1"})
                )
            event = RealtimeModelToolCallEvent(
                call_id="call_shared",
                name="weather",
                arguments="{}",
                id="item_1",
            )
            with (
                patch.object(sessions[0].model, "send_event", new=AsyncMock()),
                patch.object(sessions[1].model, "send_event", new=AsyncMock()),
            ):
                await asyncio.gather(
                    *(session._handle_tool_call(event) for session in sessions)
                )
                assert (
                    await tool.on_invoke_tool(
                        SimpleNamespace(tool_call_id="ordinary"), "{}"
                    )
                    is result
                )
                with suppress_instrumentation():
                    await sessions[0]._handle_tool_call(
                        RealtimeModelToolCallEvent(
                            call_id="suppressed",
                            name="weather",
                            arguments="{}",
                            id="item_2",
                        )
                    )
            for session in sessions:
                await session.close()
        assert tool.on_invoke_tool is invoke

    asyncio.run(run())
    spans = span_exporter.get_finished_spans()
    assert len(spans) == 2
    tool_spans = [
        span
        for span in spans
        if span.attributes["gen_ai.operation.name"] == "execute_tool"
    ]
    assert len(tool_spans) == 2
    assert all(span.parent is None for span in tool_spans)


@pytest.mark.parametrize(
    "url,address,port",
    [
        (None, "api.openai.com", 443),
        (
            "wss://realtime.example:8443/v1/realtime?key=private",
            "realtime.example",
            8443,
        ),
        ("ws://localhost/v1/realtime", "localhost", 80),
    ],
)
def test_realtime_sdk_lifecycle(
    tracer_provider,
    meter_provider,
    logger_provider,
    span_exporter,
    log_exporter,
    url,
    address,
    port,
) -> None:
    from agents.realtime.session import RealtimeSession

    original_init = RealtimeSession.__init__
    with instrument(
        OpenAIAgentsInstrumentor(),
        tracer_provider=tracer_provider,
        meter_provider=meter_provider,
        logger_provider=logger_provider,
    ):
        session = _session(
            model_config={"url": url} if url is not None else None
        )

        async def run() -> None:
            original_context = get_current()
            await session.model._emit_event(
                _event(
                    "session.created",
                    session={"id": "sess_1", "model": "gpt-realtime"},
                )
            )
            await session.model._emit_event(
                _event("response.created", response={"id": "resp_1"})
            )
            await session.model._emit_event(
                _event(
                    "response.done",
                    response={
                        "id": "resp_1",
                        "status": "completed",
                        "usage": {"input_tokens": 5, "output_tokens": 3},
                    },
                )
            )
            await session.close()
            assert get_current() is original_context

        asyncio.run(run())
    assert RealtimeSession.__init__ is original_init
    assert len(span_exporter.get_finished_spans()) == 1
    (response,) = span_exporter.get_finished_spans()
    records = [entry.log_record for entry in log_exporter.get_finished_logs()]
    for telemetry in [response, *records]:
        assert telemetry.attributes["server.address"] == address
        assert telemetry.attributes["server.port"] == port
        assert type(telemetry.attributes["server.address"]) is str
        assert type(telemetry.attributes["server.port"]) is int
    assert [
        entry.log_record.event_name
        for entry in log_exporter.get_finished_logs()
    ] == [
        "gen_ai.client.realtime_session.started",
        "gen_ai.client.realtime_session.ended",
    ]


@pytest.mark.parametrize(
    "error",
    [ConnectionError("lost"), asyncio.CancelledError(), ValueError("caller")],
)
def test_realtime_sdk_errors(
    tracer_provider,
    meter_provider,
    logger_provider,
    span_exporter,
    log_exporter,
    error,
) -> None:
    with instrument(
        OpenAIAgentsInstrumentor(),
        tracer_provider=tracer_provider,
        meter_provider=meter_provider,
        logger_provider=logger_provider,
    ):
        session = _session()

        async def run() -> None:
            async def connect(*args: Any, **kwargs: Any) -> None:
                await session.model._emit_event(
                    _event("session.created", session={"id": "sess_1"})
                )
                await session.model._emit_event(
                    _event("response.created", response={"id": "resp_1"})
                )

            with patch.object(session.model, "connect", side_effect=connect):
                with pytest.raises(type(error)) as raised:
                    async with session:
                        if not isinstance(error, ConnectionError):
                            raise error
                        await session.model._emit_event(
                            SimpleNamespace(type="exception", exception=error)
                        )
                        async for _event_value in session:
                            pass
                assert raised.value is error

        asyncio.run(run())
    spans = span_exporter.get_finished_spans()
    assert len(spans) == 1
    expected_type = (
        "asyncio.exceptions.CancelledError"
        if isinstance(error, asyncio.CancelledError)
        else type(error).__name__
    )
    assert all(
        span.attributes["error.type"] == expected_type for span in spans
    )
    assert len(_session_events(log_exporter)) == 2
    ended = _session_events(log_exporter)[-1]
    assert ended.severity_number is SeverityNumber.WARN
    assert ended.attributes["error.type"] == expected_type


def test_realtime_connect_failure(
    tracer_provider,
    meter_provider,
    logger_provider,
    span_exporter,
    log_exporter,
) -> None:
    error = ConnectionError("unavailable")
    with instrument(
        OpenAIAgentsInstrumentor(),
        tracer_provider=tracer_provider,
        meter_provider=meter_provider,
        logger_provider=logger_provider,
    ):
        session = _session()

        async def run() -> None:
            with patch.object(
                session.model, "connect", new=AsyncMock(side_effect=error)
            ):
                with pytest.raises(ConnectionError) as raised:
                    async with session:
                        pass
                assert raised.value is error

        asyncio.run(run())
    assert not span_exporter.get_finished_spans()
    assert not log_exporter.get_finished_logs()


def test_realtime_suppression(
    tracer_provider,
    meter_provider,
    logger_provider,
    span_exporter,
    log_exporter,
) -> None:
    with instrument(
        OpenAIAgentsInstrumentor(),
        tracer_provider=tracer_provider,
        meter_provider=meter_provider,
        logger_provider=logger_provider,
    ):
        with suppress_instrumentation():
            session = _session()
            asyncio.run(
                session.model._emit_event(
                    _event("session.created", session={"id": "sess_1"})
                )
            )
            asyncio.run(session.close())
    assert not span_exporter.get_finished_spans()
    assert not log_exporter.get_finished_logs()


def test_realtime_uninstrument_active_session(
    tracer_provider,
    meter_provider,
    logger_provider,
    span_exporter,
    log_exporter,
) -> None:
    with instrument(
        OpenAIAgentsInstrumentor(),
        tracer_provider=tracer_provider,
        meter_provider=meter_provider,
        logger_provider=logger_provider,
    ):
        session = _session()
        asyncio.run(
            session.model._emit_event(
                _event("session.created", session={"id": "sess_1"})
            )
        )
    assert not span_exporter.get_finished_spans()
    assert len(log_exporter.get_finished_logs()) == 2
    asyncio.run(
        session.model._emit_event(
            _event("response.created", response={"id": "late"})
        )
    )
    asyncio.run(session.close())
    assert not span_exporter.get_finished_spans()
    assert len(log_exporter.get_finished_logs()) == 2


@pytest.mark.parametrize("session_id", [None, "sess_1"])
@pytest.mark.parametrize("capture", ["NO_CONTENT", "SPAN_AND_EVENT"])
def test_realtime_session_events_with_recoverable_error(
    tracer_provider,
    logger_provider,
    span_exporter,
    log_exporter,
    monkeypatch,
    session_id,
    capture,
) -> None:
    monkeypatch.setenv(
        "OTEL_INSTRUMENTATION_GENAI_CAPTURE_MESSAGE_CONTENT", capture
    )
    monkeypatch.setenv("OTEL_INSTRUMENTATION_GENAI_EMIT_EVENT", "true")
    tracer = tracer_provider.get_tracer(__name__)
    with tracer.start_as_current_span("application") as parent:
        handler = RealtimeTelemetryHandler(
            TelemetryHandler(
                tracer_provider=tracer_provider,
                logger_provider=logger_provider,
            ),
            "voice_agent",
        )
        created = _event(
            "session.created",
            session={"id": session_id, "model": "gpt-realtime"},
        )
        handler.handle_event(created)
        handler.handle_event(created)
        handler.handle_event(
            _event("error", error={"type": "invalid_request_error"})
        )
        handler.handle_event(
            _event("response.created", response={"id": "resp_1"})
        )
        handler.handle_event(
            _event(
                "response.done",
                response={"id": "resp_1", "status": "completed"},
            )
        )
        handler.cleanup()
        handler.cleanup(ConnectionError("too late"))
    response, _ = span_exporter.get_finished_spans()
    assert response.parent.span_id == parent.get_span_context().span_id
    assert response.attributes.get("gen_ai.realtime_session.id") == session_id
    records = [entry.log_record for entry in log_exporter.get_finished_logs()]
    assert [record.event_name for record in records] == [
        "gen_ai.client.realtime_session.started",
        "gen_ai.client.realtime_session.ended",
    ]
    for record in records:
        assert record.severity_number is SeverityNumber.INFO
        assert record.trace_id == parent.get_span_context().trace_id
        assert record.span_id == parent.get_span_context().span_id
        assert (
            record.attributes.get("gen_ai.realtime_session.id") == session_id
        )
        assert "error.type" not in record.attributes
        assert "gen_ai.conversation.id" not in record.attributes
