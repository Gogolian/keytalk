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

import json
import asyncio
import logging
from typing import List, Optional

from ..transport import Transport, TransportClosed
from .constants import (
    CAPS_CHAR_UUID,
    L2CAP_PSM_CHAR_UUID,
    PROMPT_CHAR_UUID,
    RESPONSE_CHAR_UUID,
    SERVICE_NAME,
    SERVICE_UUID,
)

__all__ = ["BlessPeripheralTransport"]

logger = logging.getLogger("keytalk.ble.peripheral")


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
        caps_char: str = CAPS_CHAR_UUID,
        l2cap_psm_char: str = L2CAP_PSM_CHAR_UUID,
        supported_modes: Optional[List[str]] = None,
        notify_interval: float = 0.02,
        l2cap_psm: Optional[int] = None,
    ) -> None:
        super().__init__()
        self._name = name
        self._service_uuid = service_uuid
        self._prompt_char = prompt_char
        self._response_char = response_char
        self._caps_char = caps_char
        self._l2cap_psm_char = l2cap_psm_char
        # Modes to advertise in the CAPS characteristic; always includes legacy.
        self._supported_modes: List[str] = supported_modes if supported_modes is not None else ["legacy"]
        self._server = None  # type: ignore[assignment]
        self._loop: Optional[asyncio.AbstractEventLoop] = None
        self._notify_interval = notify_interval
        self._send_lock = asyncio.Lock()
        # PSM advertised when l2cap_coc is in supported_modes; None means absent.
        self._l2cap_psm: Optional[int] = l2cap_psm

    async def start(self) -> None:
        _import_bless()
        from bless import (  # type: ignore[import-not-found]
            BlessServer,
            GATTCharacteristicProperties,
            GATTAttributePermissions,
        )

        self._loop = asyncio.get_running_loop()
        server = BlessServer(name=self._name, loop=self._loop)

        # Track connections
        def _on_read(characteristic, **_kwargs) -> None:
            logger.debug("Consumer read from characteristic %s", characteristic)

        # When the consumer writes the prompt characteristic, forward the bytes.
        def _write_request(characteristic, value, **_kwargs) -> None:
            data = bytes(value)
            logger.debug("✓ Consumer sent %d bytes", len(data))
            assert self._loop is not None
            asyncio.run_coroutine_threadsafe(self._dispatch(data), self._loop)

        server.read_request_func = _on_read
        server.write_request_func = _write_request

        await server.add_new_service(self._service_uuid)

        prompt_props = (
            GATTCharacteristicProperties.write
            | GATTCharacteristicProperties.write_without_response
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
            GATTCharacteristicProperties.read
            | GATTCharacteristicProperties.notify
        )
        response_perms = GATTAttributePermissions.readable
        await server.add_new_characteristic(
            self._service_uuid,
            self._response_char,
            response_props,
            None,
            response_perms,
        )

        # CAPS characteristic: read-only, static list of supported modes.
        caps_value = bytearray(json.dumps(self._supported_modes).encode())
        caps_props = GATTCharacteristicProperties.read
        caps_perms = GATTAttributePermissions.readable
        await server.add_new_characteristic(
            self._service_uuid,
            self._caps_char,
            caps_props,
            caps_value,
            caps_perms,
        )

        # L2CAP PSM characteristic: present only when l2cap_coc is supported.
        if "l2cap_coc" in self._supported_modes and self._l2cap_psm is not None:
            import struct as _struct
            psm_value = bytearray(_struct.pack("<H", self._l2cap_psm))
            psm_props = GATTCharacteristicProperties.read
            psm_perms = GATTAttributePermissions.readable
            await server.add_new_characteristic(
                self._service_uuid,
                self._l2cap_psm_char,
                psm_props,
                psm_value,
                psm_perms,
            )
            logger.info("Advertising L2CAP PSM=%d via GATT characteristic", self._l2cap_psm)

        await server.start()
        self._server = server
        logger.info("BLE peripheral started, advertising as '%s'", self._name)

    async def send(self, frame: bytes) -> None:
        if self._server is None:
            raise TransportClosed("BLE peripheral is not running")
        # Serialize and pace notifications: writing char.value and calling
        # update_value() back-to-back can overwrite a value before the OS has
        # transmitted the previous notification, silently dropping frames.
        async with self._send_lock:
            char = self._server.get_characteristic(self._response_char)
            char.value = bytearray(frame)
            # Notify subscribed consumers of the new value.
            self._server.update_value(self._service_uuid, self._response_char)
            logger.debug(
                "Sent %d bytes to consumer via response characteristic", len(frame)
            )
            if self._notify_interval > 0:
                await asyncio.sleep(self._notify_interval)

    async def close(self) -> None:
        server = self._server
        self._server = None
        if server is not None:
            await server.stop()
