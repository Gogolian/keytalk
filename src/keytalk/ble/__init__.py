"""Bluetooth LE GATT integration for keytalk.

The real radio adapters depend on optional third-party libraries (``bless`` for
the peripheral/host role and ``bleak`` for the central/consumer role).  They are
imported lazily so that ``import keytalk`` and the test-suite work without any
Bluetooth stack installed.
"""

from __future__ import annotations

from .constants import (
    PROMPT_CHAR_UUID,
    RESPONSE_CHAR_UUID,
    SERVICE_NAME,
    SERVICE_UUID,
)

__all__ = [
    "SERVICE_UUID",
    "SERVICE_NAME",
    "PROMPT_CHAR_UUID",
    "RESPONSE_CHAR_UUID",
]
