"""macOS Bluetooth Classic RFCOMM transport via IOBluetooth/pyobjc.

macOS has no kernel-level BT socket API (unlike Linux/Windows), so we drive
``IOBluetoothRFCOMMChannel`` through pyobjc.  The bridge works as follows:

 * A daemon thread runs an ``NSRunLoop`` so IOBluetooth callbacks are delivered.
 * Received bytes are fed into an ``asyncio.StreamReader`` via
   ``call_soon_threadsafe``, satisfying the ``_recv_loop`` in the base class.
 * ``send()`` is overridden to call ``writeSync:length:`` directly (in an
   executor so it doesn't block the event loop).

Requires: pyobjc-core (installed as a transitive dep of pyobjc-framework-CoreBluetooth).
The IOBluetooth system framework is loaded at runtime via ``objc.loadBundle``.
"""

from __future__ import annotations

import asyncio
import struct
import sys
import threading
from typing import Any, Optional

from .channel import RFCOMMStreamTransport
from ..transport import TransportClosed

__all__ = ["MacOSRFCOMMHostTransport", "MacOSRFCOMMConsumerTransport"]

# 4-byte big-endian length prefix — mirrors L2CAPStreamTransport._LEN_STRUCT.
_FRAME_LEN = struct.Struct(">I")
# kIOBluetoothUserNotificationChannelDirectionIncoming
_INCOMING = 1


def _require_macos() -> None:
    if sys.platform != "darwin":
        raise RuntimeError(
            "MacOSRFCOMMTransport is only supported on macOS; "
            "use LinuxRFCOMMTransport on Linux or WindowsRFCOMMTransport on Windows"
        )


def _load_iobluetooth() -> dict:
    """Load the IOBluetooth system framework via pyobjc-core; return its namespace."""
    try:
        import objc
    except ImportError as exc:
        raise ImportError(
            "pyobjc-core is required for macOS RFCOMM support (pip install pyobjc-core)"
        ) from exc
    ns: dict = {}
    objc.loadBundle(
        "IOBluetooth",
        bundle_path="/System/Library/Frameworks/IOBluetooth.framework",
        module_globals=ns,
    )
    return ns


# Lazily created once per process; stores the Objective-C delegate class.
_DelegateClass: Any = None


def _get_delegate_class() -> Any:
    global _DelegateClass
    if _DelegateClass is not None:
        return _DelegateClass

    import objc
    from Foundation import NSObject

    class _RFCOMMDelegate(NSObject):
        """Bridges IOBluetoothRFCOMMChannel events into an asyncio.StreamReader."""

        @objc.python_method
        def _setup(
            self,
            loop: asyncio.AbstractEventLoop,
            reader: asyncio.StreamReader,
            on_open: "asyncio.Event | None" = None,
        ) -> None:
            self._loop = loop
            self._reader = reader
            self._on_open = on_open  # set by host to signal incoming connection

        # Called by IOBluetooth when bytes arrive on the channel.
        def rfcommChannelData_data_length_(
            self, channel: Any, data: Any, length: int
        ) -> None:
            raw = bytes(data[:length])
            self._loop.call_soon_threadsafe(self._reader.feed_data, raw)

        # Called by IOBluetooth when the remote side closes the channel.
        def rfcommChannelClosed_(self, channel: Any) -> None:
            self._loop.call_soon_threadsafe(self._reader.feed_eof)

        # Called by IOBluetooth when a new incoming channel is opened (host mode).
        # Selector registered via registerForChannelOpenNotifications:selector:...
        def rfcommChannelOpened_channel_(
            self, notification: Any, channel: Any
        ) -> None:
            channel.setDelegate_(self)
            self._channel_ref = channel  # prevent GC
            if self._on_open is not None:
                self._loop.call_soon_threadsafe(self._on_open.set)

    _DelegateClass = _RFCOMMDelegate
    return _DelegateClass


# ─── Host ─────────────────────────────────────────────────────────────────────

class MacOSRFCOMMHostTransport(RFCOMMStreamTransport):
    """Host-side RFCOMM server via IOBluetoothSDPServiceRecord."""

    def __init__(self) -> None:
        super().__init__()
        _require_macos()
        self._channel_id: int = 0
        self._rfcomm_channel: Any = None
        self._open_notification: Any = None  # must stay alive; prevents GC

    @property
    def channel_id(self) -> int:
        return self._channel_id

    async def start(self) -> None:
        ns = _load_iobluetooth()
        IOBluetoothSDPServiceRecord = ns["IOBluetoothSDPServiceRecord"]
        IOBluetoothRFCOMMChannel = ns["IOBluetoothRFCOMMChannel"]

        loop = asyncio.get_running_loop()
        rx_reader: asyncio.StreamReader = asyncio.StreamReader()
        connected: asyncio.Event = asyncio.Event()

        Delegate = _get_delegate_class()
        delegate = Delegate.alloc().init()
        delegate._setup(loop, rx_reader, on_open=connected)

        bt_ready = threading.Event()
        bt_error: list[str] = []

        def _bt_thread() -> None:
            from Foundation import NSRunLoop, NSDate

            # Publish the SPP service record; channel ID is dynamically assigned.
            sdp = {
                "0001 - ServiceClassIDList": ["00001101-0000-1000-8000-00805F9B34FB"],
                "0004 - ProtocolDescriptorList": [["L2CAP"], ["RFCOMM", 0]],
                "0100 - ServiceName*": "keytalk",
            }
            record = IOBluetoothSDPServiceRecord.publishedServiceRecord_(sdp)
            if record is None:
                bt_error.append("IOBluetooth: failed to publish SPP service record")
                bt_ready.set()
                return

            ret, ch_id = record.getRFCOMMChannelID_(None)
            if ret != 0 or not ch_id:
                bt_error.append(
                    f"IOBluetooth: getRFCOMMChannelID failed (ret=0x{ret:08x})"
                )
                bt_ready.set()
                return
            self._channel_id = int(ch_id)

            # Register for incoming connections on this channel.
            self._open_notification = (
                IOBluetoothRFCOMMChannel
                .registerForChannelOpenNotifications_selector_withChannelID_direction_(
                    delegate,
                    b"rfcommChannelOpened:channel:",
                    self._channel_id,
                    _INCOMING,
                )
            )
            bt_ready.set()

            # Pump the run loop so notifications are delivered.
            rl = NSRunLoop.currentRunLoop()
            while not self._closed:
                rl.runUntilDate_(NSDate.dateWithTimeIntervalSinceNow_(0.25))

        t = threading.Thread(target=_bt_thread, daemon=True, name="keytalk-rfcomm-host")
        t.start()
        await loop.run_in_executor(None, bt_ready.wait)

        if bt_error:
            raise OSError(bt_error[0])

        await connected.wait()
        # The delegate's rfcommChannelOpened_channel_ stored the channel.
        self._rfcomm_channel = delegate._channel_ref

        # Wire the StreamReader directly; send() bypasses the writer.
        self._reader = rx_reader
        self._start_recv()

    async def send(self, frame: bytes) -> None:
        if self._closed or self._rfcomm_channel is None:
            raise TransportClosed("RFCOMM channel is not connected")
        data = _FRAME_LEN.pack(len(frame)) + frame
        channel = self._rfcomm_channel
        loop = asyncio.get_running_loop()
        ret = await loop.run_in_executor(
            None, lambda: channel.writeSync_length_(data, len(data))
        )
        if ret != 0:
            raise OSError(f"IOBluetooth writeSync failed: 0x{ret:08x}")

    async def close(self) -> None:
        channel = self._rfcomm_channel
        self._rfcomm_channel = None
        if channel is not None:
            try:
                channel.closeChannel()
            except Exception:
                pass
        await super().close()


# ─── Consumer ─────────────────────────────────────────────────────────────────

class MacOSRFCOMMConsumerTransport(RFCOMMStreamTransport):
    """Consumer-side RFCOMM connection via IOBluetoothDevice."""

    def __init__(self, address: str, channel_id: int) -> None:
        super().__init__()
        _require_macos()
        self._address = address
        self._channel_id = channel_id
        self._rfcomm_channel: Any = None

    async def start(self) -> None:
        ns = _load_iobluetooth()
        IOBluetoothDevice = ns["IOBluetoothDevice"]

        loop = asyncio.get_running_loop()
        rx_reader: asyncio.StreamReader = asyncio.StreamReader()

        Delegate = _get_delegate_class()
        delegate = Delegate.alloc().init()
        delegate._setup(loop, rx_reader)

        connect_error: list[str] = []
        done = threading.Event()

        def _bt_thread() -> None:
            from Foundation import NSRunLoop, NSDate

            device = IOBluetoothDevice.deviceWithAddressString_(self._address)
            if device is None:
                connect_error.append(
                    f"No paired device found for address {self._address!r}"
                )
                done.set()
                return

            channel_out: list[Any] = [None]
            # openRFCOMMChannelSync:withChannelID:delegate: returns IOReturn (0 = success).
            ret = device.openRFCOMMChannelSync_withChannelID_delegate_(
                channel_out, self._channel_id, delegate
            )
            if ret != 0:
                connect_error.append(
                    f"openRFCOMMChannelSync failed: IOReturn=0x{ret:08x}"
                )
                done.set()
                return

            self._rfcomm_channel = channel_out[0]
            done.set()

            rl = NSRunLoop.currentRunLoop()
            while not self._closed:
                rl.runUntilDate_(NSDate.dateWithTimeIntervalSinceNow_(0.25))

        t = threading.Thread(target=_bt_thread, daemon=True, name="keytalk-rfcomm-consumer")
        t.start()
        await loop.run_in_executor(None, done.wait)

        if connect_error:
            raise OSError(connect_error[0])

        self._reader = rx_reader
        self._start_recv()

    async def send(self, frame: bytes) -> None:
        if self._closed or self._rfcomm_channel is None:
            raise TransportClosed("RFCOMM channel is not connected")
        data = _FRAME_LEN.pack(len(frame)) + frame
        channel = self._rfcomm_channel
        loop = asyncio.get_running_loop()
        ret = await loop.run_in_executor(
            None, lambda: channel.writeSync_length_(data, len(data))
        )
        if ret != 0:
            raise OSError(f"IOBluetooth writeSync failed: 0x{ret:08x}")

    async def close(self) -> None:
        channel = self._rfcomm_channel
        self._rfcomm_channel = None
        if channel is not None:
            try:
                channel.closeChannel()
            except Exception:
                pass
        await super().close()
