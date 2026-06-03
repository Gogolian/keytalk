"""BLE central transport for the consumer side (built on ``bleak``).

The consumer connects to the host, subscribes to the RESPONSE characteristic for
notifications, and writes prompt frames to the PROMPT characteristic.  ``bleak``
is cross-platform (CoreBluetooth on macOS, BlueZ on Linux, WinRT on Windows).

The import of ``bleak`` is deferred to construction/use so the rest of keytalk
works without it installed.
"""

from __future__ import annotations

import logging
from typing import Optional

from ..transport import Transport, TransportClosed
from .constants import PROMPT_CHAR_UUID, RESPONSE_CHAR_UUID, SERVICE_UUID

__all__ = ["BleakCentralTransport", "discover_hosts"]

logger = logging.getLogger("keytalk.ble.central")


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
        self._prompt_char_obj = None  # type: ignore[assignment]
        self._response_char_obj = None  # type: ignore[assignment]

    async def start(self) -> None:
        _import_bleak()
        from bleak import BleakClient

        logger.info("Connecting to BLE host at %s...", self._address)
        self._client = BleakClient(self._address)
        await self._client.connect()

        # Resolve service and characteristics by iterating directly to avoid
        # ambiguity when multiple items share the same UUID.
        # Find all services matching our UUID
        matching_services = [s for s in self._client.services if s.uuid == self._service_uuid]
        if not matching_services:
            raise RuntimeError(f"Service {self._service_uuid} not found on device")
        if len(matching_services) > 1:
            logger.warning(
                "Found %d services with UUID %s (likely a stale macOS GATT "
                "cache); using the one with the highest handle",
                len(matching_services), self._service_uuid,
            )

        # macOS CoreBluetooth caches GATT tables and may surface stale services
        # and characteristics from previous host runs alongside the live ones.
        # Writing to a stale characteristic gets ACK'd by the OS cache but never
        # reaches the running host process. The live attributes always have the
        # highest handles, so prefer those.
        service = max(matching_services, key=lambda s: s.handle)

        prompt_chars = [c for c in service.characteristics if c.uuid == self._prompt_char]
        response_chars = [c for c in service.characteristics if c.uuid == self._response_char]

        if not prompt_chars:
            raise RuntimeError(f"Prompt characteristic {self._prompt_char} not found")
        if not response_chars:
            raise RuntimeError(f"Response characteristic {self._response_char} not found")

        if len(prompt_chars) > 1 or len(response_chars) > 1:
            logger.warning(
                "Duplicate characteristics detected (prompt=%d, response=%d); "
                "selecting the freshest by handle. If prompts still don't reach "
                "the host, clear the macOS BLE cache (toggle Bluetooth off/on or "
                "reset the host advertisement).",
                len(prompt_chars), len(response_chars),
            )

        # Prefer a writable prompt characteristic; among candidates pick the
        # highest handle (the live one).
        def _writable(c) -> bool:
            props = getattr(c, "properties", []) or []
            return "write" in props or "write-without-response" in props

        writable_prompts = [c for c in prompt_chars if _writable(c)] or prompt_chars
        self._prompt_char_obj = max(writable_prompts, key=lambda c: c.handle)
        self._response_char_obj = max(response_chars, key=lambda c: c.handle)

        logger.info(
            "Using prompt char handle=%s, response char handle=%s",
            self._prompt_char_obj.handle, self._response_char_obj.handle,
        )
        logger.info("✓ Connected to host")

        def _notification_handler(_sender: object, data: bytearray) -> None:
            # bleak invokes this from the event loop; schedule dispatch so an
            # async callback can run.
            import asyncio
            logger.debug("Received %d bytes from host", len(data))

            asyncio.ensure_future(self._dispatch(bytes(data)))

        logger.debug("Setting up notifications for responses...")
        await self._client.start_notify(self._response_char_obj, _notification_handler)
        logger.info("✓ Ready to send prompts")

    async def send(self, frame: bytes) -> None:
        if self._client is None or not self._client.is_connected:
            raise TransportClosed("BLE central is not connected")
        logger.debug("Sending %d bytes to host", len(frame))
        await self._client.write_gatt_char(
            self._prompt_char_obj, frame, response=self._write_with_response
        )

    async def close(self) -> None:
        client = self._client
        self._client = None
        if client is not None and client.is_connected:
            try:
                await client.stop_notify(self._response_char_obj)
            except Exception:  # pragma: no cover - best effort on teardown
                pass
            await client.disconnect()
