"""Tool-call plumbing for the chat bridge.

Two jobs live here:

* **Rendering** tool definitions, assistant tool calls and tool results into a
  plain transcript (for prompt-only backends and prompt-mode requests), and
* **Parsing** the plaintext tool-call syntax many models emit when they are
  asked for a tool call but the runtime has no structured tool channel (e.g.
  ``<function=search>{"query": "x"}</function>``) back into OpenAI-style
  ``tool_calls`` objects.

Without the parser a bridged reply shows the raw ``<function>...`` markup as
assistant text and the client never executes the tool; with it the reply is
re-framed as ``message.tool_calls`` (Ollama) / ``delta.tool_calls`` (OpenAI)
exactly as a local, tool-aware server would return it.

Recognized plaintext call shapes:

``<function=name>ARGS</function>``
    Llama-3 / functionary-style; the closing tag may be omitted when the
    response ends with the call.
``function name(args)`` / ``function name(args)</function>``
    Functionary-v1-style, on its own line; ``args`` may be a JSON object or
    Python-style keyword arguments (``query="x", k=3``).
``to=name {json}`` / ``to=name [{json}]``
    Qwen-style, on its own line.
``name{json}``
    Granite/Hermes-style bare calls on their own line - only accepted when
    ``name`` is one of the tools actually offered, so prose can never be
    mistaken for an invocation.

``ARGS`` is a JSON object; wrappers such as ``{"call": {...}}`` or
``{"Body": {...}}`` are unwrapped.  Argument JSON may span several lines.
"""

from __future__ import annotations

import ast
import json
import re
from typing import AsyncIterator, Dict, List, Optional, Sequence, Tuple

__all__ = [
    "parse_text_tool_calls",
    "normalize_tool_calls",
    "tool_call_dicts",
    "tool_calls_to_prompt_lines",
    "tool_calls_to_ollama",
    "tools_to_prompt_section",
    "render_tool_result",
    "ToolCallFilter",
    "ToolCallExtractor",
]

#: JSON object wrappers some templates put around the argument object.
_ARG_WRAPPERS = ("call", "Call", "Body", "Parameters", "parameters", "arguments")

_NAME = r"[A-Za-z0-9_.\-]+"
_TAG_START = re.compile(r"<function=(?P<name>%s)>" % _NAME)
_TO_START = re.compile(r"^[ \t]*to=(?P<name>%s)[ \t]*(?=[\[{])" % _NAME, re.MULTILINE)
_BARE_START = re.compile(r"^[ \t]*(?P<name>%s)[ \t]*(?=\{)" % _NAME, re.MULTILINE)
_FUNC_START = re.compile(
    r"^[ \t]*function[ \t]+(?P<name>%s)[ \t]*\(" % _NAME, re.MULTILINE
)
_TAG_CLOSE = "</function>"
#: After a call's argument JSON only whitespace may follow on that line.
_AFTER_JSON = re.compile(r"[ \t]*(?:\n|$)")

_decoder = json.JSONDecoder()


def _tool_names(tools: Optional[Sequence[dict]]) -> Optional[set]:
    """The set of declared tool names (``None`` when no tools were offered)."""

    if tools is None:
        return None
    names = set()
    for tool in tools:
        if not isinstance(tool, dict):
            continue
        function = tool.get("function") if isinstance(tool.get("function"), dict) else tool
        name = function.get("name")
        if name:
            names.add(str(name))
    return names


def _parse_arguments(raw: str) -> Tuple[str, Optional[object]]:
    """Return ``(arguments_string, args_value)`` for a call's raw argument text.

    The ``arguments`` field of an OpenAI tool call is a JSON *string*; the
    parsed value is returned separately for rendering/validation.  Unparseable
    text is kept verbatim (some clients repair it themselves) with a ``None``
    value.
    """

    raw = raw.strip()
    if not raw:
        return "{}", {}
    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError:
        parsed = _parse_kwargs(raw)
        if parsed is None:
            return raw, None
    if isinstance(parsed, dict):
        # Unwrap {"call": {...}} / {"Body": {...}} style envelopes.
        if len(parsed) == 1:
            key = next(iter(parsed))
            if key in _ARG_WRAPPERS:
                inner = parsed[key]
                if isinstance(inner, str):
                    return _parse_arguments(inner)
                if isinstance(inner, dict):
                    return json.dumps(inner, separators=(",", ":")), inner
        return json.dumps(parsed, separators=(",", ":")), parsed
    if isinstance(parsed, list) and parsed and isinstance(parsed[0], dict):
        # ``to=name [{...}]`` sends a single-element argument array.
        only = parsed[0]
        return json.dumps(only, separators=(",", ":")), only
    return raw, None


def _parse_kwargs(raw: str) -> Optional[dict]:
    """Parse functionary-style keyword arguments (``query="x", k=3``).

    Values may be any literal Python/JSON value (strings, numbers, lists,
    dicts); returns ``None`` when the text is not plain keyword arguments.
    """

    try:
        tree = ast.parse("call(" + raw + "\n)", mode="eval")
    except SyntaxError:
        return None
    call = tree.body
    if not isinstance(call, ast.Call) or call.args:
        return None
    out: dict = {}
    for keyword in call.keywords:
        if keyword.arg is None:
            return None
        try:
            out[keyword.arg] = ast.literal_eval(keyword.value)
        except ValueError:
            return None
    return out


def _find_close_paren(text: str, open_index: int) -> Optional[int]:
    """Index of the parenthesis matching ``text[open_index]``, or ``None``.

    Quotes (and backslash escapes inside them) are skipped so parentheses in
    argument strings do not confuse the balance; an unterminated call reports
    ``None`` so streaming code can wait for the rest of the text.
    """

    depth = 0
    quote: Optional[str] = None
    index = open_index
    while index < len(text):
        char = text[index]
        if quote is not None:
            if char == "\\":
                index += 1
            elif char == quote:
                quote = None
        elif char in "\"'":
            quote = char
        elif char == "(":
            depth += 1
        elif char == ")":
            depth -= 1
            if depth == 0:
                return index
        index += 1
    return None


def tool_call_dicts(name: str, raw_args: str, index: int = 0) -> Dict[str, object]:
    """Build one OpenAI-style tool-call object from a plaintext invocation."""

    arguments, _ = _parse_arguments(raw_args)
    return {
        "id": f"call_{index}_{name}",
        "type": "function",
        "function": {"name": name, "arguments": arguments},
    }


def normalize_tool_calls(calls: Sequence[object]) -> List[Dict[str, object]]:
    """Coerce tool calls from any source into strict OpenAI shape.

    Accepts OpenAI-style calls (streamed deltas already merged by the backend),
    plain ``{"name", "arguments"}`` dicts and Ollama-style
    ``{"function": {...}}`` wrappers.  ``arguments`` is always a JSON string,
    as the OpenAI API requires.
    """

    out: List[Dict[str, object]] = []
    for call in calls or []:
        if not isinstance(call, dict):
            continue
        function = (
            call.get("function")
            if isinstance(call.get("function"), dict)
            else call
        )
        name = function.get("name") or call.get("name")
        if not name:
            continue
        arguments = function.get("arguments", "{}")
        if isinstance(arguments, (dict, list)):
            arguments = json.dumps(arguments, separators=(",", ":"))
        elif not isinstance(arguments, str):
            arguments = "{}"
        out.append(
            {
                "id": str(call.get("id") or f"call_{len(out)}_{name}"),
                "type": "function",
                "function": {"name": str(name), "arguments": arguments},
            }
        )
    return out


def tool_calls_to_ollama(calls: Sequence[object]) -> List[Dict[str, object]]:
    """Render tool calls in Ollama's ``message.tool_calls`` shape.

    Ollama models arguments as a JSON *object* (not a string) and correlates
    tool results by name, so the OpenAI ``id`` is dropped.
    """

    out: List[Dict[str, object]] = []
    for call in normalize_tool_calls(calls):
        raw = call["function"]["arguments"]  # type: ignore[index]
        try:
            arguments = json.loads(raw)
        except (ValueError, TypeError):
            arguments = {}
        if not isinstance(arguments, dict):
            arguments = {}
        out.append(
            {"function": {"name": call["function"]["name"], "arguments": arguments}}  # type: ignore[index]
        )
    return out


def _json_end(text: str, start: int) -> Optional[int]:
    """Index just past the complete JSON value at ``start``, or ``None``."""

    while start < len(text) and text[start] in " \t\n\r":
        start += 1
    if start >= len(text) or text[start] not in "{[":
        return None
    try:
        _, end = _decoder.raw_decode(text, start)
    except ValueError:
        return None
    return end


def _iter_calls(text: str, known: Optional[set]):
    """Yield ``(start, end, call)`` for every complete call found in ``text``."""

    pos = 0
    index = 0
    while pos < len(text):
        best = None  # (start, style, match)
        for style, pattern, needs_known in (
            ("tag", _TAG_START, False),
            ("to", _TO_START, False),
            ("func", _FUNC_START, False),
            ("bare", _BARE_START, True),
        ):
            if needs_known:
                # Bare ``name{json}`` is only trusted for declared tool names.
                if known is None:
                    continue
                match = next(
                    (m for m in pattern.finditer(text, pos) if m["name"] in known),
                    None,
                )
            else:
                match = pattern.search(text, pos)
            if match is not None and (best is None or match.start() < best[0]):
                best = (match.start(), style, match)
        if best is None:
            return
        start, style, match = best
        result = _resolve_call(text, start, style, match, index)
        if result is not None:
            end, call = result
            yield start, end, call
            index += 1
            pos = max(end, start + 1)
        else:
            # Unfinished or not really a call: skip past its start marker and
            # keep scanning (the text stays part of the content).
            pos = start + 1


def _resolve_call(
    text: str, start: int, style: str, match: "re.Match[str]", index: int,
    *, require_close: bool = False,
) -> Optional[Tuple[int, Dict[str, object]]]:
    """Resolve one call match into ``(end, call)``; ``None`` if incomplete.

    ``require_close`` keeps ``<function=...>`` calls open until their closing
    tag arrives (streaming: the tag may still be on the wire); without it a
    truncated call whose argument JSON is complete is accepted as-is.
    """

    name = match["name"]
    if style == "tag":
        args_start = match.end()
        close = text.find(_TAG_CLOSE, args_start)
        if close != -1:
            end = close + len(_TAG_CLOSE)
            return end, tool_call_dicts(name, text[args_start:close], index)
        if require_close:
            return None
        # Truncated reply without the closing tag: accept once the argument
        # JSON is complete (common when generation stops right after the call).
        end = _json_end(text, args_start)
        if end is not None:
            return end, tool_call_dicts(name, text[args_start:end], index)
        return None
    if style == "func":
        close = _find_close_paren(text, match.end() - 1)
        if close is None:
            return None  # arguments not complete yet
        end = close + 1
        rest = text[end:]
        if rest.startswith(_TAG_CLOSE):
            end += len(_TAG_CLOSE)
        elif require_close and (not rest or _TAG_CLOSE.startswith(rest)):
            return None  # a closing </function> tag may still be on the wire
        return end, tool_call_dicts(name, text[match.end() : close], index)
    if style == "to":
        args_start = match.end()
    else:  # bare
        brace = text.find("{", match.end())
        if brace < 0:
            return None
        args_start = brace
    end = _json_end(text, args_start)
    if end is None:
        return None
    if _AFTER_JSON.match(text, end) is None:
        return None  # prose continues on the same line: not a call after all
    return end, tool_call_dicts(name, text[args_start:end], index)


def parse_text_tool_calls(
    text: str, tools: Optional[Sequence[dict]] = None
) -> Tuple[str, List[Dict[str, object]]]:
    """Split ``text`` into plain content and structured tool calls.

    ``tools`` is the request's tool definition list (if any): bare ``name{json}``
    calls are only accepted for names it declares, so prose can never be
    misread as an invocation.  Tagged forms (``<function=...>`` / ``to=...``)
    are unambiguous and accepted regardless.

    Returns ``(content, tool_calls)`` where ``tool_calls`` is a list of
    OpenAI-style ``{"id", "type", "function": {"name", "arguments"}}`` dicts;
    ``content`` is the leftover text with the call markup removed (it may be
    empty - a tool-calling turn usually is).
    """

    if not text:
        return text, []
    known = _tool_names(tools)
    parts: List[str] = []
    calls: List[Dict[str, object]] = []
    cursor = 0
    for start, end, call in _iter_calls(text, known):
        if start < cursor:
            continue
        parts.append(text[cursor:start])
        cursor = _swallow_line_break(text, start, end)
        calls.append(call)
    parts.append(text[cursor:])
    content = "".join(parts)
    content = re.sub(r"\n[ \t]+\n", "\n\n", content)
    return content.strip(), calls


def _swallow_line_break(text: str, start: int, end: int) -> int:
    """Advance ``end`` over one newline when the call sat on its own line(s).

    Removing an own-line call would otherwise leave a blank line behind.
    """

    at_line_start = start > 0 and text[start - 1] == "\n"
    if at_line_start and end < len(text) and text[end] == "\n":
        return end + 1
    return end


class ToolCallFilter:
    """Streaming text filter that withholds plaintext tool calls.

    Feed response pieces to :meth:`feed` and get back the prose that is safe to
    emit right now; text that might still turn into a tool call (an unterminated
    ``<function=...>`` block, a ``name{`` whose JSON is not complete yet, ...)
    is held back until it resolves.  At end of stream call :meth:`finish` to get
    the remaining prose plus the :attr:`calls` that were extracted.
    """

    #: How far a trailing partial call marker may reach back into the buffer.
    _MAX_GUARD = 256

    def __init__(self, tools: Optional[Sequence[dict]] = None) -> None:
        self._tools = list(tools) if tools else None
        self._known = _tool_names(self._tools)
        self._buf = ""
        self._last_out = ""
        self.calls: List[Dict[str, object]] = []

    def feed(self, text: str) -> str:
        """Consume a text piece and return the prose safe to emit now."""

        self._buf += text
        return self._drain()

    def finish(self) -> str:
        """Flush the tail: returns remaining prose, harvesting trailing calls."""

        content, calls = parse_text_tool_calls(self._buf, self._tools)
        self.calls.extend(calls)
        self._buf = ""
        return content

    def _drain(self) -> str:
        out: List[str] = []
        while self._buf:
            start = self._next_call_start(self._buf)
            if start is None:
                guard = self._guard_len(self._buf)
                out.append(self._buf[: len(self._buf) - guard])
                self._buf = self._buf[len(self._buf) - guard :]
                break
            out.append(self._buf[:start])
            resolved = self._complete_at(self._buf, start)
            if resolved is None:
                # An unfinished call: withhold from here until more arrives.
                self._buf = self._buf[start:]
                break
            end, call = resolved
            self.calls.append(call)
            # Was the call on its own line?  (The newline before it may have
            # been emitted by a previous feed.)
            prev = self._buf[start - 1] if start > 0 else self._last_out[-1:]
            if prev == "\n" and end < len(self._buf) and self._buf[end] == "\n":
                end += 1
            self._buf = self._buf[end:]
        text = "".join(out)
        if text:
            self._last_out = text
        return text

    def _next_call_start(self, buf: str) -> Optional[int]:
        best = None
        for pattern, needs_known in (
            (_TAG_START, False),
            (_TO_START, False),
            (_FUNC_START, False),
            (_BARE_START, True),
        ):
            if needs_known:
                if self._known is None:
                    continue
                match = next(
                    (m for m in pattern.finditer(buf) if m["name"] in self._known),
                    None,
                )
            else:
                match = pattern.search(buf)
            if match is not None and (best is None or match.start() < best):
                best = match.start()
        return best

    def _complete_at(self, buf: str, start: int):
        for style, pattern in (
            ("tag", _TAG_START),
            ("to", _TO_START),
            ("func", _FUNC_START),
            ("bare", _BARE_START),
        ):
            match = pattern.search(buf, start)
            if match is None or match.start() != start:
                continue
            if style == "bare" and (
                self._known is None or match["name"] not in self._known
            ):
                continue
            return _resolve_call(
                buf, start, style, match, len(self.calls), require_close=True
            )
        return None

    def _guard_len(self, buf: str) -> int:
        """Length of the trailing slice that might still become a call start.

        Returned as a length counted from the *end* of the buffer (0 = flush
        everything)."""

        tail = buf[-self._MAX_GUARD :]
        line_start = tail.rfind("\n") + 1
        line = tail[line_start:]
        if self._is_partial_marker(line):
            return len(tail) - line_start  # withhold the whole tail line
        return 0

    def _is_partial_marker(self, line: str) -> bool:
        stripped = line.strip()
        if not stripped:
            return False
        # A trailing "<..." (anywhere in the line) could still grow into a
        # tagged call start or closing tag - those are not line-anchored.
        if re.search(r"<[/A-Za-z0-9_.=\-]*$", line):
            return True
        head = line.lstrip()
        # Over-hold anything that may be growing into a ``to=...`` or
        # ``function name(...)`` call start, or into a bare tool name: only the
        # trailing incomplete line is affected (bounded by _MAX_GUARD) and it
        # flushes as soon as its newline arrives.
        if head.startswith(("to", "function", "f")):
            return True
        if self._known:
            word = stripped.split("{", 1)[0].strip()
            if word and any(name.startswith(word) for name in self._known):
                return True
        return False


class ToolCallExtractor:
    """Async-iterator wrapper that strips plaintext calls from a text stream.

    Iterating yields only the clean prose pieces; once the source is exhausted
    :attr:`tool_calls` holds the extracted calls (merged with any structured
    ``tool_calls`` the source reported through ``source.tool_calls``).
    """

    def __init__(self, source: AsyncIterator[str], tools: Optional[Sequence[dict]] = None):
        self._source = source
        self._filter = ToolCallFilter(tools)
        self._parsed: List[Dict[str, object]] = []
        self._gen = self._run()

    async def _run(self) -> AsyncIterator[str]:
        async for piece in self._source:
            text = self._filter.feed(piece)
            if text:
                yield text
        tail = self._filter.finish()
        self._parsed = list(self._filter.calls)
        if tail:
            yield tail

    def __aiter__(self) -> "ToolCallExtractor":
        return self

    async def __anext__(self) -> str:
        return await self._gen.__anext__()

    async def aclose(self) -> None:
        aclose = getattr(self._gen, "aclose", None)
        if aclose is not None:
            await aclose()
        close = getattr(self._source, "aclose", None)
        if close is not None:
            await close()

    @property
    def tool_calls(self) -> List[Dict[str, object]]:
        """All tool calls seen: parsed from text plus the source's own."""

        own = getattr(self._source, "tool_calls", None) or []
        return list(own) + list(self._parsed)

    @property
    def timings(self) -> Optional[dict]:
        """Generation stats reported by the source stream, once available."""

        value = getattr(self._source, "timings", None)
        return value if isinstance(value, dict) else None


# ---------------------------------------------------------------------------
# Rendering (prompt-side)
# ---------------------------------------------------------------------------


def tools_to_prompt_section(tools: Sequence[dict]) -> str:
    """Render an OpenAI tool list as prompt instructions for text-only models.

    The instructions teach the same plaintext call syntax
    :func:`parse_text_tool_calls` understands, so a model without a structured
    tool channel can still make calls the bridge will re-frame for the client.
    """

    lines: List[str] = [
        "You can call tools to help the user.  To call one, output a single "
        "line of the exact form",
        '    <function=tool_name>{"arg": "value"}</function>',
        "with the arguments as a JSON object matching the tool's parameters.  "
        "You may call several tools, one per line.  Only call a tool when it "
        "is needed; otherwise just answer.",
        "",
        "Available tools:",
    ]
    for tool in tools:
        if not isinstance(tool, dict):
            continue
        function = tool.get("function") if isinstance(tool.get("function"), dict) else tool
        name = str(function.get("name", ""))
        if not name:
            continue
        description = str(function.get("description", "")).strip()
        parameters = function.get("parameters")
        lines.append(f"- {name}: {description}".rstrip())
        if isinstance(parameters, dict):
            lines.append(
                f"  parameters: {json.dumps(parameters, separators=(',', ':'))}"
            )
    return "\n".join(lines)


def tool_calls_to_prompt_lines(tool_calls: Sequence[dict]) -> List[str]:
    """Render assistant ``tool_calls`` as transcript lines (prompt fallback)."""

    lines: List[str] = []
    for call in tool_calls or []:
        if not isinstance(call, dict):
            continue
        function = call.get("function") if isinstance(call.get("function"), dict) else call
        name = str(function.get("name", ""))
        arguments = function.get("arguments", "{}")
        if not isinstance(arguments, str):
            arguments = json.dumps(arguments, separators=(",", ":"))
        lines.append(f"<function={name}>{arguments}</function>")
    return lines


def render_tool_result(message: dict) -> str:
    """Render a ``role: "tool"`` result message as a transcript line."""

    content = message.get("content", "")
    if content is None:
        content = ""
    if not isinstance(content, str):
        content = json.dumps(content, separators=(",", ":"))
    call_id = str(message.get("tool_call_id") or message.get("name") or "tool")
    return f"Tool result ({call_id}): {content}"
