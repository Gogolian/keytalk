"""keytalk - use an LLM on one machine from another over Bluetooth LE.

The HOST machine runs an LLM (e.g. an Ollama server) and exposes it through a
custom BLE GATT service.  The CONSUMER machine connects over BLE, writes prompt
chunks, and reads the streamed response chunks - no IP network involved.

Public building blocks:

* :mod:`keytalk.protocol` - transport-agnostic framing/chunking.
* :mod:`keytalk.transport` - the ``Transport`` interface and the in-memory loopback.
* :mod:`keytalk.backends` - the ``LLMBackend`` interface and its implementations
  (Ollama, LM Studio, OpenRouter, llama.cpp, plus test fakes).
* :mod:`keytalk.host` - ``HostService``: prompt frames -> LLM -> response frames.
* :mod:`keytalk.consumer` - ``ConsumerClient``: prompt -> frames -> reply.
* :mod:`keytalk.server` - the Ollama-compatible HTTP bridge (``--serve``).
* :mod:`keytalk.ble` - real radio adapters (optional ``bleak``/``bless`` deps).
"""

from __future__ import annotations

from .backends import (
    DummyFileBackend,
    EchoBackend,
    LMStudioBackend,
    LMStudioError,
    LlamaCppBackend,
    LlamaCppError,
    LLMBackend,
    OllamaBackend,
    OllamaError,
    OpenRouterBackend,
    OpenRouterError,
    StaticBackend,
    TokenStream,
    messages_to_prompt,
)
from .consumer import ConsumerClient, RemoteError, ResponseStream
from .host import HostService
from .server import (
    OllamaBridgeServer,
    PromptStreamer,
    build_prompt_from_messages,
)
from .toolcalls import (
    ToolCallExtractor,
    normalize_tool_calls,
    parse_text_tool_calls,
)
from .protocol import (
    CompleteMessage,
    Flags,
    Frame,
    FrameStreamEncoder,
    MessageType,
    ProtocolError,
    Reassembler,
    chunk_message,
    max_payload_for_mtu,
)
from .transport import InMemoryTransport, Transport, TransportClosed, create_loopback

__version__ = "0.1.0"

__all__ = [
    "__version__",
    # protocol
    "Frame",
    "Flags",
    "MessageType",
    "CompleteMessage",
    "ProtocolError",
    "Reassembler",
    "FrameStreamEncoder",
    "chunk_message",
    "max_payload_for_mtu",
    # transport
    "Transport",
    "TransportClosed",
    "InMemoryTransport",
    "create_loopback",
    # backends
    "LLMBackend",
    "EchoBackend",
    "StaticBackend",
    "DummyFileBackend",
    "OllamaBackend",
    "OllamaError",
    "LMStudioBackend",
    "LMStudioError",
    "OpenRouterBackend",
    "OpenRouterError",
    "LlamaCppBackend",
    "LlamaCppError",
    "TokenStream",
    "messages_to_prompt",
    # endpoints
    "HostService",
    "ConsumerClient",
    "ResponseStream",
    "RemoteError",
    # ollama-compatible HTTP bridge
    "OllamaBridgeServer",
    "PromptStreamer",
    "build_prompt_from_messages",
    # tool-call plumbing
    "parse_text_tool_calls",
    "normalize_tool_calls",
    "ToolCallExtractor",
]
