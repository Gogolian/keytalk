"""Selectable transfer-mode profiles for keytalk.

Phase 0: scaffolding only — Mode enum, ProfileConfig dataclass, and the
LEGACY_PROFILE constant that mirrors the original hard-coded behaviour.  No
existing code path is changed; all new code paths fall back to LEGACY_PROFILE.

Phase 1: NegotiationError, mode-ID wire encoding, and negotiate_mode() which
picks the best common mode from the host's CAPS advertisement.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Dict, List, Optional

from .protocol import DEFAULT_ATT_MTU, max_payload_for_mtu

__all__ = [
    "Mode",
    "ProfileConfig",
    "LEGACY_PROFILE",
    "FAST_GATT_PROFILE",
    "L2CAP_COC_PROFILE",
    "CLASSIC_RFCOMM_PROFILE",
    "NegotiationError",
    "make_fast_gatt_profile",
    "make_l2cap_coc_profile",
    "make_classic_rfcomm_profile",
    "profile_for_mode",
    "mode_id_for",
    "mode_for_id",
    "negotiate_mode",
]

_UNIMPLEMENTED = "requested mode is not yet implemented"


class Mode(str, Enum):
    LEGACY = "legacy"
    FAST_GATT = "fast_gatt"
    L2CAP_COC = "l2cap_coc"
    CLASSIC_RFCOMM = "rfcomm"


class NegotiationError(Exception):
    """Raised when the consumer and host cannot agree on a transfer mode."""


@dataclass(frozen=True)
class ProfileConfig:
    mode: Mode
    mtu: int
    write_with_response: bool
    compression_codec: str | None  # None | "zlib" | "zstd" | "lz4"
    flow_control: str  # "fixed_pacing" | "credit_window"
    reliability_window: int

    @property
    def max_payload_size(self) -> int:
        return max_payload_for_mtu(self.mtu)


# Mode 0 — permanent, always available, byte-identical to the original protocol.
LEGACY_PROFILE = ProfileConfig(
    mode=Mode.LEGACY,
    mtu=DEFAULT_ATT_MTU,
    write_with_response=True,
    compression_codec="zlib",
    flow_control="fixed_pacing",
    reliability_window=32,
)

# Mode 1 — same GATT topology but with negotiated MTU, bidirectional compression,
# credit/window flow control, write-without-response, and CRC32 integrity checks.
FAST_GATT_PROFILE = ProfileConfig(
    mode=Mode.FAST_GATT,
    mtu=DEFAULT_ATT_MTU,  # overridden at runtime via make_fast_gatt_profile
    write_with_response=False,
    compression_codec="zlib",
    flow_control="credit_window",
    reliability_window=64,
)


def make_fast_gatt_profile(mtu: int) -> ProfileConfig:
    """Return a FAST_GATT profile sized for the given negotiated MTU."""
    return ProfileConfig(
        mode=Mode.FAST_GATT,
        mtu=mtu,
        write_with_response=False,
        compression_codec="zlib",
        flow_control="credit_window",
        reliability_window=64,
    )


# Default SDU payload size for L2CAP LE CoC — well within the 65 533-byte limit.
_L2CAP_COC_DEFAULT_MTU = 1024

# Mode 2 — BLE LE credit-based Connection-Oriented Channel.  The stream is
# reliable and ordered, so Go-Back-N is dropped; CRC32 is kept for integrity.
L2CAP_COC_PROFILE = ProfileConfig(
    mode=Mode.L2CAP_COC,
    mtu=_L2CAP_COC_DEFAULT_MTU,
    write_with_response=False,
    compression_codec="zlib",
    flow_control="l2cap_credits",
    reliability_window=0,  # stream guarantees delivery
)


def make_l2cap_coc_profile(mtu: int = _L2CAP_COC_DEFAULT_MTU) -> ProfileConfig:
    """Return an L2CAP_COC profile sized for the given SDU payload MTU."""
    return ProfileConfig(
        mode=Mode.L2CAP_COC,
        mtu=mtu,
        write_with_response=False,
        compression_codec="zlib",
        flow_control="l2cap_credits",
        reliability_window=0,
    )


# Default RFCOMM payload MTU — Bluetooth Classic supports larger frames but
# 1024 B is a safe baseline across all platforms and adapters.
_RFCOMM_DEFAULT_MTU = 1024

# Mode 3 — Bluetooth Classic RFCOMM / SPP stream.  Like L2CAP COC: reliable
# ordered delivery, no Go-Back-N; CRC32 kept for integrity.
CLASSIC_RFCOMM_PROFILE = ProfileConfig(
    mode=Mode.CLASSIC_RFCOMM,
    mtu=_RFCOMM_DEFAULT_MTU,
    write_with_response=False,
    compression_codec="zlib",
    flow_control="rfcomm_stream",
    reliability_window=0,  # stream guarantees delivery
)


def make_classic_rfcomm_profile(mtu: int = _RFCOMM_DEFAULT_MTU) -> ProfileConfig:
    """Return a CLASSIC_RFCOMM profile sized for the given payload MTU."""
    return ProfileConfig(
        mode=Mode.CLASSIC_RFCOMM,
        mtu=mtu,
        write_with_response=False,
        compression_codec="zlib",
        flow_control="rfcomm_stream",
        reliability_window=0,
    )


# Stable numeric IDs sent on the wire in SELECT frames (must never be renumbered).
_MODE_IDS: Dict[Mode, int] = {
    Mode.LEGACY: 0,
    Mode.FAST_GATT: 1,
    Mode.L2CAP_COC: 2,
    Mode.CLASSIC_RFCOMM: 3,
}
_MODE_BY_ID: Dict[int, Mode] = {v: k for k, v in _MODE_IDS.items()}

# Modes with a complete implementation; updated as phases land.
_IMPLEMENTED_MODES = frozenset({"legacy", "fast_gatt", "l2cap_coc", "rfcomm"})

# Descending preference order for "auto" negotiation.
_MODE_PRIORITY = [
    Mode.CLASSIC_RFCOMM,
    Mode.L2CAP_COC,
    Mode.FAST_GATT,
    Mode.LEGACY,
]


def mode_id_for(mode: Mode) -> int:
    """Return the stable wire ID for a mode."""
    return _MODE_IDS[mode]


def mode_for_id(mid: int) -> Mode:
    """Return the Mode for a wire ID, or raise ValueError if unknown."""
    m = _MODE_BY_ID.get(mid)
    if m is None:
        raise ValueError(f"unknown mode wire id: {mid}")
    return m


def profile_for_mode(mode: str) -> ProfileConfig:
    """Return the ProfileConfig for the requested mode string.

    ``"auto"`` resolves to LEGACY until negotiation is implemented (Phase 1).
    Raises ValueError for modes that are defined but not yet implemented.
    """
    if mode in ("auto", "legacy"):
        return LEGACY_PROFILE
    try:
        m = Mode(mode)
    except ValueError:
        raise ValueError(f"unknown mode {mode!r}") from None
    if m == Mode.FAST_GATT:
        return FAST_GATT_PROFILE
    if m == Mode.L2CAP_COC:
        return L2CAP_COC_PROFILE
    if m == Mode.CLASSIC_RFCOMM:
        return CLASSIC_RFCOMM_PROFILE
    return LEGACY_PROFILE  # unreachable, but keeps mypy happy


def negotiate_mode(
    host_modes: Optional[List[str]],
    requested: str,
) -> ProfileConfig:
    """Pick the best ProfileConfig from the host's advertised capabilities.

    ``host_modes`` is the list returned by reading the CAPS characteristic, or
    ``None`` if the host is pre-Phase-1 and has no CAPS characteristic.

    ``requested`` is the consumer's ``--mode`` value (``"auto"`` or an explicit
    mode name).

    Rules:
    - ``host_modes is None`` + ``"auto"``  → legacy (silent backward compat)
    - ``host_modes is None`` + explicit   → NegotiationError (host can't confirm)
    - ``"auto"``                           → highest-priority mode in the
                                            intersection of host_modes and
                                            _IMPLEMENTED_MODES
    - explicit mode                        → NegotiationError if not in host_modes;
                                            ValueError if not yet implemented
    """
    if host_modes is None:
        if requested == "auto":
            return LEGACY_PROFILE
        raise NegotiationError(
            f"explicit mode {requested!r} requested but the host has no CAPS "
            "characteristic — cannot confirm support (old host, legacy only)"
        )

    host_set = set(host_modes)

    if requested == "auto":
        for m in _MODE_PRIORITY:
            if m.value in host_set and m.value in _IMPLEMENTED_MODES:
                return profile_for_mode(m.value)
        return LEGACY_PROFILE  # fallback if intersection is somehow empty

    if requested not in host_set:
        # Distinguish a typo (unknown mode string) from a mode the host doesn't offer.
        try:
            Mode(requested)
        except ValueError:
            raise ValueError(f"unknown mode {requested!r}") from None
        raise NegotiationError(
            f"requested mode {requested!r} is not offered by the host "
            f"(host supports: {', '.join(sorted(host_set))})"
        )
    return profile_for_mode(requested)  # raises ValueError if not implemented

