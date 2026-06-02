"""BLE peripheral transport for the host side (built on ``bless``).

The host advertises the keytalk GATT service with two characteristics:

* PROMPT (write)    - the consumer writes prompt frames here; each write is
  forwarded to the registered receive callback.
* RESPONSE (notify) - the host pushes response frames here as notifications.

``bless`` provides a cross-platform GATT *server* on top of the same OS
backends as ``bleak`` (CoreBluetooth on macOS, BlueZ on Linux).  It is imported
lazily so keytalk works without it installed.
"""

from __future__ import annotations

import asyncio
from typing import Optional

from ..transport import Transport, TransportClosed
from .constants import (
    PROMPT_CHAR_UUID,
    RESPONSE_CHAR_UUID,
    SERVICE_NAME,
    SERVICE_UUID,
)

__all__ = ["BlessPeripheralTransport"]


def _import_bless():
    try:
        import bless  # noqa: F401

        return bless
    except ImportError as exc:  # pragma: no cover - requires missing dep
        raise RuntimeError(
            "the 'bless' package is required for the host BLE transport; "
            "install it with `pip install keytalk[host]`"
        ) from exc


class BlessPeripheralTransport(Transport):
    """Host-side transport that exposes the keytalk GATT service."""

    def __init__(
        self,
        name: str = SERVICE_NAME,
        *,
        service_uuid: str = SERVICE_UUID,
        prompt_char: str = PROMPT_CHAR_UUID,
        response_char: str = RESPONSE_CHAR_UUID,
    ) -> None:
        super().__init__()
        self._name = name
        self._service_uuid = service_uuid
        self._prompt_char = prompt_char
        self._response_char = response_char
        self._server = None  # type: ignore[assignment]
        self._loop: Optional[asyncio.AbstractEventLoop] = None

    async def start(self) -> None:
        _import_bless()
        from bless import (  # type: ignore[import-not-found]
            BlessServer,
            BlessGATTCharacteristicProperties,
            GATTAttributePermissions,
        )

        self._loop = asyncio.get_running_loop()
        server = BlessServer(name=self._name, loop=self._loop)

        # When the consumer writes the prompt characteristic, forward the bytes.
        def _write_request(characteristic, value, **_kwargs) -> None:
            data = bytes(value)
            assert self._loop is not None
            asyncio.run_coroutine_threadsafe(self._dispatch(data), self._loop)

        server.write_request_func = _write_request

        await server.add_new_service(self._service_uuid)

        prompt_props = (
            BlessGATTCharacteristicProperties.write
            | BlessGATTCharacteristicProperties.write_without_response
        )
        prompt_perms = GATTAttributePermissions.writeable
        await server.add_new_characteristic(
            self._service_uuid,
            self._prompt_char,
            prompt_props,
            None,
            prompt_perms,
        )

        response_props = (
            BlessGATTCharacteristicProperties.read
            | BlessGATTCharacteristicProperties.notify
        )
        response_perms = GATTAttributePermissions.readable
        await server.add_new_characteristic(
            self._service_uuid,
            self._response_char,
            response_props,
            b"",
            response_perms,
        )

        await server.start()
        self._server = server

    async def send(self, frame: bytes) -> None:
        if self._server is None:
            raise TransportClosed("BLE peripheral is not running")
        char = self._server.get_characteristic(self._response_char)
        char.value = bytearray(frame)
        # Notify subscribed consumers of the new value.
        self._server.update_value(self._service_uuid, self._response_char)

    async def close(self) -> None:
        server = self._server
        self._server = None
        if server is not None:
            await server.stop()
