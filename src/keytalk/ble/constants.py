"""Custom GATT service/characteristic UUIDs for the keytalk bridge.

These are randomly-chosen 128-bit UUIDs in a private range; they do not collide
with any standardised Bluetooth SIG service.  Both host and consumer must agree
on them.

Roles, from the GATT point of view:

* ``PROMPT``   - written by the consumer, received by the host (write).
* ``RESPONSE`` - notified by the host, subscribed to by the consumer (notify).
"""

from __future__ import annotations

#: Human-readable name the host advertises.
SERVICE_NAME = "keytalk"

#: Primary custom service exposed by the host.
SERVICE_UUID = "9a8c0001-7b1e-4f9a-8c3d-2f6b1e9a8c00"

#: Consumer -> host: prompt chunks are written here.
PROMPT_CHAR_UUID = "9a8c0002-7b1e-4f9a-8c3d-2f6b1e9a8c00"

#: Host -> consumer: response chunks are delivered here via notifications.
RESPONSE_CHAR_UUID = "9a8c0003-7b1e-4f9a-8c3d-2f6b1e9a8c00"
