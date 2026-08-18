"""Linux L2CAP LE CoC transport skeleton.

Uses ``AF_BLUETOOTH``/``BTPROTO_L2CAP`` sockets with the LE CoC mode
(``BT_MODE_LE_FLOWCTL``, Linux kernel ≥ 4.9).  Full implementation is deferred
to hardware testing; this file provides the class interface so imports succeed.
"""

from __future__ import annotations

import sys
from typing import Optional

from .channel import L2CAPStreamTransport
from ...transport import TransportClosed

__all__ = ["LinuxL2CAPHostTransport", "LinuxL2CAPConsumerTransport"]

# LE PSM values 0x0080–0x00FF are in the dynamically-assigned range per the
# Bluetooth Assigned Numbers specification (section 2.5).
_DEFAULT_LE_PSM = 0x0080

# Linux-specific socket constants not in the stdlib.
_BT_MODE = 15
_BT_MODE_LE_FLOWCTL = 3  # LE credit-based flow control


def _require_linux() -> None:
    if sys.platform != "linux":
        raise RuntimeError(
            "LinuxL2CAPTransport is only supported on Linux; "
            "use MacOSL2CAPTransport on macOS or fall back to FAST_GATT"
        )


class LinuxL2CAPHostTransport(L2CAPStreamTransport):
    """Host-side L2CAP LE CoC listener on Linux."""

    def __init__(self, psm: int = _DEFAULT_LE_PSM) -> None:
        super().__init__()
        _require_linux()
        self._psm = psm
        self._server: Optional[object] = None

    @property
    def psm(self) -> int:
        return self._psm

    async def start(self) -> None:
        # TODO(phase3-linux): bind AF_BLUETOOTH/BTPROTO_L2CAP socket, set
        # BT_MODE=BT_MODE_LE_FLOWCTL, listen, and accept via asyncio loop.
        raise NotImplementedError(
            "Linux L2CAP host transport is not yet fully implemented"
        )

    async def close(self) -> None:
        if self._server is not None:
            self._server = None  # type: ignore[assignment]
        await super().close()


class LinuxL2CAPConsumerTransport(L2CAPStreamTransport):
    """Consumer-side L2CAP LE CoC connection on Linux."""

    def __init__(self, address: str, psm: int = _DEFAULT_LE_PSM) -> None:
        super().__init__()
        _require_linux()
        self._address = address
        self._psm = psm

    async def start(self) -> None:
        # TODO(phase3-linux): connect AF_BLUETOOTH/BTPROTO_L2CAP socket,
        # set BT_MODE=BT_MODE_LE_FLOWCTL, and attach asyncio streams.
        raise NotImplementedError(
            "Linux L2CAP consumer transport is not yet fully implemented"
        )
