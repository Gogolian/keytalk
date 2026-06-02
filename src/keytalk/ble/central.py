"""BLE central transport for the consumer side (built on ``bleak``).

The consumer connects to the host, subscribes to the RESPONSE characteristic for
notifications, and writes prompt frames to the PROMPT characteristic.  ``bleak``
is cross-platform (CoreBluetooth on macOS, BlueZ on Linux, WinRT on Windows).

The import of ``bleak`` is deferred to construction/use so the rest of keytalk
works without it installed.
"""

from __future__ import annotations

from typing import Optional

from ..transport import Transport, TransportClosed
from .constants import PROMPT_CHAR_UUID, RESPONSE_CHAR_UUID, SERVICE_UUID

__all__ = ["BleakCentralTransport", "discover_hosts"]


def _import_bleak():
    try:
        import bleak  # noqa: F401

        return bleak
    except ImportError as exc:  # pragma: no cover - requires missing dep
        raise RuntimeError(
            "the 'bleak' package is required for the consumer BLE transport; "
            "install it with `pip install keytalk[consumer]`"
        ) from exc


async def discover_hosts(timeout: float = 5.0):
    """Return advertising devices that expose the keytalk service.

    Returns a list of ``bleak`` ``BLEDevice`` objects.  Useful for a CLI that
    lets the user pick which host to connect to.
    """

    bleak = _import_bleak()
    from bleak import BleakScanner

    devices = await BleakScanner.discover(
        timeout=timeout, service_uuids=[SERVICE_UUID]
    )
    return list(devices)


class BleakCentralTransport(Transport):
    """Consumer-side transport that speaks to the host over BLE GATT."""

    def __init__(
        self,
        address: str,
        *,
        service_uuid: str = SERVICE_UUID,
        prompt_char: str = PROMPT_CHAR_UUID,
        response_char: str = RESPONSE_CHAR_UUID,
        write_with_response: bool = True,
    ) -> None:
        super().__init__()
        self._address = address
        self._service_uuid = service_uuid
        self._prompt_char = prompt_char
        self._response_char = response_char
        self._write_with_response = write_with_response
        self._client = None  # type: ignore[assignment]

    async def start(self) -> None:
        _import_bleak()
        from bleak import BleakClient

        self._client = BleakClient(self._address)
        await self._client.connect()

        def _notification_handler(_sender: object, data: bytearray) -> None:
            # bleak invokes this from the event loop; schedule dispatch so an
            # async callback can run.
            import asyncio

            asyncio.ensure_future(self._dispatch(bytes(data)))

        await self._client.start_notify(self._response_char, _notification_handler)

    async def send(self, frame: bytes) -> None:
        if self._client is None or not self._client.is_connected:
            raise TransportClosed("BLE central is not connected")
        await self._client.write_gatt_char(
            self._prompt_char, frame, response=self._write_with_response
        )

    async def close(self) -> None:
        client = self._client
        self._client = None
        if client is not None and client.is_connected:
            try:
                await client.stop_notify(self._response_char)
            except Exception:  # pragma: no cover - best effort on teardown
                pass
            await client.disconnect()
