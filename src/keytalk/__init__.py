"""keytalk - use an LLM on one machine from another over Bluetooth LE.

The HOST machine runs an LLM (e.g. an Ollama server) and exposes it through a
custom BLE GATT service.  The CONSUMER machine connects over BLE, writes prompt
chunks, and reads the streamed response chunks - no IP network involved.

Public building blocks:

* :class:`~keytalk.protocol` - transport-agnostic framing/chunking.
* :class:`~keytalk.transport.Transport` and the in-memory loopback for tests.
* :class:`~keytalk.backends.LLMBackend` (Ollama + test fakes).
* :class:`~keytalk.host.HostService` and
  :class:`~keytalk.consumer.ConsumerClient` - the two endpoints.
* :mod:`keytalk.ble` - real radio adapters (optional ``bleak``/``bless`` deps).
"""

from __future__ import annotations

from .backends import (
    EchoBackend,
    LLMBackend,
    OllamaBackend,
    OllamaError,
    StaticBackend,
)
from .consumer import ConsumerClient, RemoteError
from .host import HostService
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
    "OllamaBackend",
    "OllamaError",
    # endpoints
    "HostService",
    "ConsumerClient",
    "RemoteError",
]
