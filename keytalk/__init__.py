"""keytalk package."""

from .bridge import (
    OllamaClient,
    PromptRequest,
    PromptResponse,
    handle_prompt_lines,
)
from .protocol import (
    DEFAULT_CHUNK_SIZE,
    KeytalkMessage,
    ProtocolError,
    decode_lines,
    encode_text,
)

__all__ = [
    "DEFAULT_CHUNK_SIZE",
    "KeytalkMessage",
    "OllamaClient",
    "PromptRequest",
    "PromptResponse",
    "ProtocolError",
    "decode_lines",
    "encode_text",
    "handle_prompt_lines",
]
