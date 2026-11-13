"""Translate OpenAI Agents realtime model events into GenAI invocations."""

from __future__ import annotations

import logging
from collections.abc import Callable
from contextvars import ContextVar
from importlib import import_module
from typing import Any, cast
from urllib.parse import urlparse
from weakref import ReferenceType, WeakKeyDictionary, ref

from wrapt import FunctionWrapper, wrap_function_wrapper

from opentelemetry.context import Context, attach, detach, get_current
from opentelemetry.instrumentation.utils import (
    is_instrumentation_enabled,
    unwrap,
)
from opentelemetry.semconv._incubating.attributes import (
    gen_ai_attributes as GenAI,
)
from opentelemetry.trace import set_span_in_context
from opentelemetry.util.genai.handler import TelemetryHandler
from opentelemetry.util.genai.invocation import (
    RealtimeInferenceInvocation,
    ToolInvocation,
)
from opentelemetry.util.genai.types import Error
from opentelemetry.util.genai.utils import (
    object_to_any_value,
    tool_arguments_to_any_value,
)

logger = logging.getLogger(__name__)


class RealtimeTelemetryHandler:
    def __init__(
        self,
        handler: TelemetryHandler,
        agent_name: str,
        *,
        server_address: str | None = None,
        server_port: int | None = None,
    ) -> None:
        self._handler: TelemetryHandler = handler
        self._agent_name: str = agent_name
        self._server_address: str | None = server_address
        self._server_port: int | None = server_port
        self._root_context: Context = get_current()
        self._session_id: str | None = None
        self._started: bool = False
        self._responses: dict[str, RealtimeInferenceInvocation] = {}
        self._tools: dict[str, ToolInvocation] = {}
        self._tool_contexts: dict[str, Context] = {}
        self._model: str | None = None
        self._conversation_id: str | None = None
        self._output_type: str | None = None
        self._closed: bool = False

    def start(self) -> None:
        if not self._started and not self._closed:
            self._started = True
            self._handler.emit_realtime_session_started(
                GenAI.GenAiProviderNameValues.OPENAI.value,
                request_model=self._model,
                realtime_session_id=self._session_id,
                server_address=self._server_address,
                server_port=self._server_port,
                context=self._root_context,
            )

    @property
    def context(self) -> Context:
        return self._root_context

    def cleanup(self, error: BaseException | Error | None = None) -> None:
        if self._closed:
            return
        self._closed = True
        for response in self._responses.values():
            response.finish_reasons = ["error"]
        invocations = [*self._tools.values(), *self._responses.values()]
        self._tools.clear()
        self._tool_contexts.clear()
        self._responses.clear()
        for invocation in invocations:
            if error is None:
                invocation.stop()
            else:
                invocation.fail(error)
        if self._started:
            self._handler.emit_realtime_session_ended(
                GenAI.GenAiProviderNameValues.OPENAI.value,
                request_model=self._model,
                realtime_session_id=self._session_id,
                server_address=self._server_address,
                server_port=self._server_port,
                error=error,
                context=self._root_context,
            )

    def handle_event(self, event: Any) -> Context:
        if self._closed:
            return self._root_context
        if event.type == "exception":
            self.cleanup(event.exception)
            return self.context
        if event.type == "end_of_stream":
            self.cleanup()
            return self.context
        if event.type != "raw_server_event":
            return self.context
        data = event.data
        event_type = data.get("type")
        if event_type == "session.created":
            session = data["session"]
            self._session_id = session.get("id")
            self._model = session.get("model")
            self._output_type = _output_type(session)
            self.start()
        elif event_type == "session.updated":
            self._model = data["session"].get("model", self._model)
            self._output_type = _output_type(
                data["session"], self._output_type
            )
        elif event_type == "conversation.created":
            self._conversation_id = data["conversation"].get("id")
        elif event_type == "response.created":
            response = data["response"]
            response_id = response.get("id")
            if response_id and response_id not in self._responses:
                invocation = self._handler.realtime_inference(
                    GenAI.GenAiProviderNameValues.OPENAI.value,
                    request_model=self._model,
                    realtime_session_id=self._session_id,
                    server_address=self._server_address,
                    server_port=self._server_port,
                    conversation_id=response.get(
                        "conversation_id", self._conversation_id
                    ),
                    context=self.context,
                    _attach_to_context=False,
                )
                invocation.response_id = response_id
                invocation.output_type = _output_type(
                    response, self._output_type
                )
                self._responses[response_id] = invocation
                return invocation.context
        elif event_type == "response.done":
            response = data["response"]
            invocation = self._responses.pop(response.get("id"), None)
            if invocation is not None:
                invocation.conversation_id = response.get(
                    "conversation_id", invocation.conversation_id
                )
                invocation.output_type = _output_type(
                    response, invocation.output_type
                )
                invocation.response_model_name = invocation.request_model
                usage: dict[str, Any] = response.get("usage") or {}
                invocation.input_tokens = _token_count(usage, "input")
                invocation.output_tokens = _token_count(usage, "output")
                input_details: dict[str, Any] = (
                    usage.get("input_token_details") or {}
                )
                output_details: dict[str, Any] = (
                    usage.get("output_token_details") or {}
                )
                invocation.audio_input_tokens = input_details.get(
                    "audio_tokens"
                )
                invocation.audio_output_tokens = output_details.get(
                    "audio_tokens"
                )
                invocation.text_input_tokens = input_details.get("text_tokens")
                invocation.text_output_tokens = output_details.get(
                    "text_tokens"
                )
                invocation.cache_read_input_tokens = input_details.get(
                    "cached_tokens"
                )
                cached_details: dict[str, int] = (
                    input_details.get("cached_tokens_details") or {}
                )
                invocation.audio_cache_read_input_tokens = cached_details.get(
                    "audio_tokens"
                )
                invocation.text_cache_read_input_tokens = cached_details.get(
                    "text_tokens"
                )
                details: dict[str, Any] = response.get("status_details") or {}
                reason = details.get("reason")
                if not reason:
                    reason = {
                        "completed": "completed",
                        "cancelled": "interrupted",
                    }.get(response.get("status"), "error")
                invocation.finish_reasons = [reason]
                if response.get("status") == "failed":
                    provider_error: dict[str, Any] = details.get("error") or {}
                    invocation.fail(
                        Error(
                            type=provider_error.get("type")
                            or response["status"],
                            message=None,
                        )
                    )
                else:
                    invocation.stop()
        elif event_type == "response.function_call_arguments.done":
            call_id = data["call_id"]
            response = self._responses.get(data.get("response_id"))
            self._tool_contexts[call_id] = (
                set_span_in_context(response.span, self.context)
                if response is not None
                else self.context
            )
        elif event_type in {
            "conversation.item.added",
            "conversation.item.created",
        }:
            item = data["item"]
            if item.get("type") == "function_call_output":
                self._tool_contexts.pop(item.get("call_id"), None)
        elif event_type in {
            "response.output_audio.delta",
            "response.audio.delta",
            "response.text.delta",
            "response.output_text.delta",
        }:
            invocation = self._responses.get(data.get("response_id"))
            if invocation is not None:
                invocation.record_stream_chunk()
                return invocation.context
        return self.context

    def start_tool(self, event: Any) -> ToolInvocation | None:
        if self._closed:
            return None
        tool = self._handler.tool(
            name=event.name,
            tool_type="function",
            agent_name=self._agent_name,
            context=self._tool_contexts.pop(event.call_id, self.context),
            _attach_to_context=False,
        )
        tool.tool_call_id = event.call_id
        if tool.should_capture_content:
            tool.arguments = tool_arguments_to_any_value(event.arguments)
        self._tools[event.call_id] = tool
        return tool

    def finish_tool(
        self,
        call_id: str,
        result: Any = None,
        error: BaseException | None = None,
    ) -> None:
        tool = self._tools.pop(call_id, None)
        if tool is None:
            return
        if error is not None:
            tool.fail(error)
        else:
            if tool.should_capture_content:
                tool.tool_result = object_to_any_value(result)
            tool.stop()


def _output_type(
    config: dict[str, Any], default: str | None = None
) -> str | None:
    modalities = config.get("output_modalities", config.get("modalities"))
    if modalities is None:
        return default
    if "audio" in modalities:
        return GenAI.GenAiOutputTypeValues.SPEECH.value
    if "text" in modalities:
        return GenAI.GenAiOutputTypeValues.TEXT.value
    return None


def _token_count(usage: dict[str, Any], direction: str) -> int | None:
    count = usage.get(f"{direction}_tokens")
    if isinstance(count, int):
        return count
    details = usage.get(f"{direction}_token_details")
    if isinstance(details, dict):
        counts = cast("dict[str, Any]", details)
        return (counts.get("audio_tokens") or 0) + (
            counts.get("text_tokens") or 0
        )
    return None


class RealtimeInstrumentation:
    def __init__(self, handler: TelemetryHandler) -> None:
        self._handler: TelemetryHandler = handler
        self._handlers: WeakKeyDictionary[Any, RealtimeTelemetryHandler] = (
            WeakKeyDictionary()
        )
        self._patches: list[tuple[Any, str]] = []
        self._tool_patches: dict[
            int, tuple[ReferenceType[Any], ReferenceType[Any]]
        ] = {}
        self._tool_call: ContextVar[
            tuple[RealtimeTelemetryHandler, Any] | None
        ] = ContextVar("realtime_tool_call", default=None)
        self._model_class: type[Any] | None = None

    def instrument(self) -> None:
        try:
            session_module = import_module("agents.realtime.session")
            model_module = import_module("agents.realtime.openai_realtime")
        except ImportError:
            logger.debug("Optional realtime dependencies are not installed")
            return
        self._model_class = getattr(
            model_module, "OpenAIRealtimeWebSocketModel", None
        )
        for module, owner, name, wrapper in (
            (session_module, "RealtimeSession", "__init__", self._init),
            (session_module, "RealtimeSession", "__aenter__", self._enter),
            (session_module, "RealtimeSession", "__aexit__", self._exit),
            (session_module, "RealtimeSession", "close", self._close),
            (
                session_module,
                "RealtimeSession",
                "_handle_tool_call",
                self._dispatch_tool,
            ),
            (
                session_module,
                "RealtimeSession",
                "_put_event",
                self._session_event,
            ),
            (
                model_module,
                "OpenAIRealtimeWebSocketModel",
                "_emit_event",
                self._emit,
            ),
        ):
            owner_class = getattr(module, owner, None)
            if owner_class is not None and hasattr(owner_class, name):
                wrap_function_wrapper(module, f"{owner}.{name}", wrapper)
                self._patches.append((owner_class, name))

    def uninstrument(self) -> None:
        for owner, name in reversed(self._patches):
            unwrap(owner, name)
        self._patches.clear()
        for tool_ref, wrapper_ref in list(self._tool_patches.values()):
            tool = tool_ref()
            if tool is not None and tool.on_invoke_tool is wrapper_ref():
                unwrap(tool, "on_invoke_tool")
        self._tool_patches.clear()
        for handler in list(self._handlers.values()):
            self._cleanup(handler)
        self._handlers.clear()

    def _init(
        self,
        wrapped: Callable[..., Any],
        instance: Any,
        args: Any,
        kwargs: Any,
    ) -> Any:
        result = wrapped(*args, **kwargs)
        if (
            is_instrumentation_enabled()
            and self._model_class is not None
            and isinstance(instance.model, self._model_class)
        ):
            try:
                connection_url: str = instance._model_config.get(
                    "url", "wss://api.openai.com/v1/realtime"
                )
                url = urlparse(connection_url)
                self._handlers[instance.model] = RealtimeTelemetryHandler(
                    self._handler,
                    instance._current_agent.name,
                    server_address=url.hostname,
                    server_port=url.port
                    if url.port is not None
                    else (443 if url.scheme == "wss" else 80),
                )
            except Exception:
                logger.debug(
                    "Could not attach realtime telemetry", exc_info=True
                )
        return result

    async def _enter(
        self,
        wrapped: Callable[..., Any],
        instance: Any,
        args: Any,
        kwargs: Any,
    ) -> Any:
        handler = self._handlers.get(instance.model)
        try:
            return await wrapped(*args, **kwargs)
        except BaseException as error:
            self._cleanup(handler, error)
            raise

    async def _exit(
        self,
        wrapped: Callable[..., Any],
        instance: Any,
        args: Any,
        kwargs: Any,
    ) -> Any:
        error = args[1] if len(args) > 1 else kwargs.get("_exc_val")
        if isinstance(error, BaseException):
            self._cleanup(self._handlers.get(instance.model), error)
        return await wrapped(*args, **kwargs)

    async def _close(
        self,
        wrapped: Callable[..., Any],
        instance: Any,
        args: Any,
        kwargs: Any,
    ) -> Any:
        handler = self._handlers.pop(instance.model, None)
        try:
            result = await wrapped(*args, **kwargs)
        except BaseException as error:
            self._cleanup(handler, error)
            raise
        else:
            self._cleanup(handler)
            return result

    async def _dispatch_tool(
        self,
        wrapped: Callable[..., Any],
        instance: Any,
        args: Any,
        kwargs: Any,
    ) -> Any:
        handler = self._handlers.get(instance.model)
        if handler is None or not is_instrumentation_enabled():
            return await wrapped(*args, **kwargs)
        event = args[0] if args else kwargs["event"]
        token = self._tool_call.set((handler, event))
        try:
            return await wrapped(*args, **kwargs)
        finally:
            self._tool_call.reset(token)

    async def _session_event(
        self,
        wrapped: Callable[..., Any],
        instance: Any,
        args: Any,
        kwargs: Any,
    ) -> Any:
        if self._tool_call.get() is not None and is_instrumentation_enabled():
            try:
                event = args[0] if args else kwargs["event"]
                if event.type == "tool_start":
                    tool = event.tool
                    tool_id = id(tool)
                    if tool_id not in self._tool_patches:
                        wrapper = FunctionWrapper(
                            tool.on_invoke_tool, self._invoke_tool
                        )
                        tool.on_invoke_tool = wrapper
                        self._tool_patches[tool_id] = (
                            ref(
                                tool,
                                lambda _: self._tool_patches.pop(
                                    tool_id, None
                                ),
                            ),
                            ref(wrapper),
                        )
            except Exception:
                logger.debug("Could not wrap realtime tool", exc_info=True)
        return await wrapped(*args, **kwargs)

    async def _invoke_tool(
        self,
        wrapped: Callable[..., Any],
        instance: Any,
        args: Any,
        kwargs: Any,
    ) -> Any:
        active_call = self._tool_call.get()
        if active_call is None or not is_instrumentation_enabled():
            return await wrapped(*args, **kwargs)
        handler, event = active_call
        tool = None
        try:
            tool = handler.start_tool(event)
        except Exception:
            logger.debug("Could not start realtime tool", exc_info=True)
        if tool is None:
            return await wrapped(*args, **kwargs)
        context_token = attach(tool.context)
        call_token = self._tool_call.set(None)
        result = None
        error = None
        try:
            result = await wrapped(*args, **kwargs)
            return result
        except BaseException as exc:
            error = exc
            raise
        finally:
            self._tool_call.reset(call_token)
            detach(context_token)
            try:
                handler.finish_tool(event.call_id, result, error)
            except Exception:
                logger.debug("Could not finalize realtime tool", exc_info=True)

    async def _emit(
        self,
        wrapped: Callable[..., Any],
        instance: Any,
        args: Any,
        kwargs: Any,
    ) -> Any:
        handler = self._handlers.get(instance)
        if handler is None or not is_instrumentation_enabled():
            return await wrapped(*args, **kwargs)
        context = handler.context
        try:
            context = handler.handle_event(
                args[0] if args else kwargs["event"]
            )
        except Exception:
            logger.debug("Could not record realtime event", exc_info=True)
        token = attach(context)
        try:
            return await wrapped(*args, **kwargs)
        except BaseException as error:
            self._cleanup(handler, error)
            raise
        finally:
            detach(token)

    @staticmethod
    def _cleanup(
        handler: RealtimeTelemetryHandler | None,
        error: BaseException | None = None,
    ) -> None:
        if handler is not None:
            try:
                handler.cleanup(error)
            except Exception:
                logger.debug(
                    "Could not finalize realtime telemetry", exc_info=True
                )
