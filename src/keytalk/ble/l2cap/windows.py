"""Windows L2CAP LE CoC stub.

Windows 10 (build 17134+) exposes LE CoC via WinRT BLE APIs, but an
asyncio-compatible Python binding is not yet available.  L2CAP_COC mode will
not appear in the CAPS advertisement on Windows hosts; consumers fall back to
FAST_GATT automatically.
"""

from __future__ import annotations

from .channel import L2CAPStreamTransport

__all__ = ["WindowsL2CAPHostTransport", "WindowsL2CAPConsumerTransport"]


class WindowsL2CAPHostTransport(L2CAPStreamTransport):
    async def start(self) -> None:
        raise NotImplementedError(
            "L2CAP_COC mode is not yet supported on Windows; use FAST_GATT mode"
        )


class WindowsL2CAPConsumerTransport(L2CAPStreamTransport):
    async def start(self) -> None:
        raise NotImplementedError(
            "L2CAP_COC mode is not yet supported on Windows; use FAST_GATT mode"
        )
