"""BLE central transport for the consumer side (built on ``bleak``).

The consumer connects to the host, subscribes to the RESPONSE characteristic for
notifications, and writes prompt frames to the PROMPT characteristic.  ``bleak``
is cross-platform (CoreBluetooth on macOS, BlueZ on Linux, WinRT on Windows).

The import of ``bleak`` is deferred to construction/use so the rest of keytalk
works without it installed.
"""

from __future__ import annotations

import asyncio
import json
import logging
from typing import List, Optional

from ..transport import Transport, TransportClosed
from .constants import CAPS_CHAR_UUID, L2CAP_PSM_CHAR_UUID, PROMPT_CHAR_UUID, RESPONSE_CHAR_UUID, SERVICE_UUID

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
        caps_char: str = CAPS_CHAR_UUID,
        l2cap_psm_char: str = L2CAP_PSM_CHAR_UUID,
        write_with_response: bool = True,
        reconnect_attempts: int = 5,
        reconnect_delay: float = 1.0,
    ) -> None:
        super().__init__()
        self._address = address
        self._service_uuid = service_uuid
        self._prompt_char = prompt_char
        self._response_char = response_char
        self._caps_char = caps_char
        self._l2cap_psm_char = l2cap_psm_char
        self._write_with_response = write_with_response
        self._reconnect_attempts = reconnect_attempts
        self._reconnect_delay = reconnect_delay
        self._client = None  # type: ignore[assignment]
        self._prompt_char_obj = None  # type: ignore[assignment]
        self._response_char_obj = None  # type: ignore[assignment]
        self._caps_char_obj = None  # type: ignore[assignment]  # None on old hosts
        self._l2cap_psm_char_obj = None  # type: ignore[assignment]  # None when absent
        self._closed = False
        self._reconnect_lock = asyncio.Lock()

    async def start(self) -> None:
        _import_bleak()
        self._closed = False
        await self._connect()

    async def _connect(self) -> None:
        """Open the GATT connection and resolve characteristics/notifications."""

        from bleak import BleakClient

        logger.info("Connecting to BLE host at %s...", self._address)
        self._client = BleakClient(self._address)
        await self._client.connect()
        await self._attempt_pair()
        await self._resolve_and_subscribe()

    async def _attempt_pair(self) -> None:
        """Attempt to pair/bond with the host before negotiation.

        Bonding is required for L2CAP_COC (encrypted PSM) and Classic RFCOMM.
        On macOS CoreBluetooth, explicit pairing is not available; the OS
        auto-pairs on first access to an encrypted characteristic.
        """
        try:
            await self._client.pair()
            logger.info("✓ Paired/bonded with host")
        except NotImplementedError:
            logger.debug(
                "Explicit pairing not supported on this platform "
                "(CoreBluetooth auto-pairs on encrypted characteristic access)"
            )
        except Exception as exc:
            logger.warning(
                "Pairing attempt failed: %s — continuing without pairing; "
                "modes requiring bonding (l2cap_coc, rfcomm) may not be available",
                exc,
            )

    async def _resolve_and_subscribe(self) -> None:
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

        # CAPS characteristic is optional (absent on pre-Phase-1 hosts).
        caps_chars = [c for c in service.characteristics if c.uuid == self._caps_char]
        self._caps_char_obj = max(caps_chars, key=lambda c: c.handle) if caps_chars else None
        if self._caps_char_obj is None:
            logger.debug("Host has no CAPS characteristic — will use legacy mode")

        # L2CAP PSM characteristic is optional (absent on non-L2CAP hosts).
        psm_chars = [c for c in service.characteristics if c.uuid == self._l2cap_psm_char]
        self._l2cap_psm_char_obj = max(psm_chars, key=lambda c: c.handle) if psm_chars else None

        logger.info(
            "Using prompt char handle=%s, response char handle=%s",
            self._prompt_char_obj.handle, self._response_char_obj.handle,
        )
        logger.info("✓ Connected to host")

        def _notification_handler(_sender: object, data: bytearray) -> None:
            # bleak invokes this from the event loop; schedule dispatch so an
            # async callback can run.
            logger.debug("Received %d bytes from host", len(data))

            asyncio.ensure_future(self._dispatch(bytes(data)))

        logger.debug("Setting up notifications for responses...")
        await self._client.start_notify(self._response_char_obj, _notification_handler)
        logger.info("✓ Ready to send prompts")

    async def _ensure_connected(self) -> None:
        """Reconnect transparently if the BLE link dropped between requests.

        A dropped connection (host restart, range, radio glitch) should not
        kill the bridge.  We try a bounded number of reconnects so the next
        prompt can succeed instead of permanently raising ``TransportClosed``.
        """

        if self._closed:
            raise TransportClosed("BLE central has been closed")
        if self._client is not None and self._client.is_connected:
            return

        async with self._reconnect_lock:
            # Another coroutine may have reconnected while we waited for the lock.
            if self._closed:
                raise TransportClosed("BLE central has been closed")
            if self._client is not None and self._client.is_connected:
                return

            logger.warning(
                "BLE link to %s is down; attempting to reconnect...",
                self._address,
            )
            last_exc: Optional[BaseException] = None
            for attempt in range(1, self._reconnect_attempts + 1):
                try:
                    await self._connect()
                    logger.info(
                        "✓ Reconnected to host on attempt %d/%d",
                        attempt, self._reconnect_attempts,
                    )
                    return
                except asyncio.CancelledError:
                    raise
                except Exception as exc:  # noqa: BLE001 - keep retrying
                    last_exc = exc
                    logger.warning(
                        "reconnect attempt %d/%d failed: %s",
                        attempt, self._reconnect_attempts, exc,
                    )
                    if attempt < self._reconnect_attempts:
                        await asyncio.sleep(self._reconnect_delay)
            raise TransportClosed(
                f"BLE central could not reconnect to {self._address}: {last_exc}"
            )

    # -- capability hooks -----------------------------------------------------

    async def read_caps(self) -> Optional[List[str]]:
        """Read and decode the host's CAPS characteristic.

        Returns a list of mode strings, or None if the characteristic is absent
        (pre-Phase-1 host).
        """
        if self._caps_char_obj is None:
            return None
        try:
            raw = await self._client.read_gatt_char(self._caps_char_obj)
            modes: List[str] = json.loads(bytes(raw).decode())
            logger.info("Host CAPS: %s", modes)
            return modes
        except Exception as exc:  # pragma: no cover - network / parse errors
            logger.warning("Failed to read CAPS characteristic: %s; using legacy", exc)
            return None

    @property
    def mtu_size(self) -> int:
        if self._client is not None and hasattr(self._client, "mtu_size"):
            return self._client.mtu_size
        from ..protocol import DEFAULT_ATT_MTU
        return DEFAULT_ATT_MTU

    def configure_write_mode(self, write_with_response: bool) -> None:
        self._write_with_response = write_with_response
        logger.debug("write-with-response set to %s", write_with_response)

    async def read_l2cap_psm(self) -> Optional[int]:
        """Read the host's L2CAP LE PSM from the GATT characteristic.

        Returns the PSM as an integer, or None if the characteristic is absent
        (host does not support L2CAP_COC mode).
        """
        if self._l2cap_psm_char_obj is None:
            return None
        try:
            import struct as _struct
            raw = await self._client.read_gatt_char(self._l2cap_psm_char_obj)
            (psm,) = _struct.unpack_from("<H", bytes(raw))
            logger.info("Host L2CAP PSM: %d", psm)
            return psm
        except Exception as exc:  # pragma: no cover
            logger.warning("Failed to read L2CAP PSM characteristic: %s", exc)
            return None

    async def send(self, frame: bytes) -> None:
        logger.debug("Sending %d bytes to host", len(frame))
        # The link can drop *during* a write (CoreBluetooth raises
        # "disconnected"), not just between requests, so retry the write after
        # a transparent reconnect instead of bubbling the failure up.
        last_exc: Optional[BaseException] = None
        for attempt in range(1, self._reconnect_attempts + 1):
            await self._ensure_connected()
            try:
                await self._client.write_gatt_char(
                    self._prompt_char_obj,
                    frame,
                    response=self._write_with_response,
                )
                return
            except asyncio.CancelledError:
                raise
            except Exception as exc:  # noqa: BLE001 - reconnect and retry
                last_exc = exc
                # A GATT protocol error (e.g. code 0 "Unknown code") is a
                # transient ATT-level rejection; the BLE link is still alive.
                # Retry the write without dropping and re-establishing the
                # connection, which would be slow and disruptive.
                still_connected = (
                    self._client is not None
                    and getattr(self._client, "is_connected", False)
                )
                if still_connected:
                    logger.warning(
                        "GATT write error (attempt %d/%d): %s; retrying",
                        attempt, self._reconnect_attempts, exc,
                    )
                else:
                    logger.warning(
                        "write failed (attempt %d/%d): %s; reconnecting",
                        attempt, self._reconnect_attempts, exc,
                    )
                    # Drop the dead client so _ensure_connected reconnects next loop.
                    self._client = None
                if attempt < self._reconnect_attempts:
                    await asyncio.sleep(self._reconnect_delay)
        raise TransportClosed(
            f"BLE write to {self._address} failed after "
            f"{self._reconnect_attempts} attempts: {last_exc}"
        )

    async def close(self) -> None:
        self._closed = True
        client = self._client
        self._client = None
        if client is not None and client.is_connected:
            try:
                await client.stop_notify(self._response_char_obj)
            except Exception:  # pragma: no cover - best effort on teardown
                pass
            await client.disconnect()
