#!/usr/bin/env python3
"""Quick test to verify compression functionality."""

import zlib
from src.keytalk.protocol import (
    Frame,
    Flags,
    MessageType,
    Reassembler,
    chunk_message,
    max_payload_for_mtu,
    DEFAULT_ATT_MTU,
)


def test_compression_decompression():
    """Test that compressed prompts can be sent and reassembled correctly."""
    
    # Simulate a large prompt
    original_prompt = "This is a test prompt. " * 100  # ~2.3KB
    payload = original_prompt.encode("utf-8")
    
    print(f"Original payload size: {len(payload)} bytes")
    
    # Compress it
    compressed_payload = zlib.compress(payload, level=6)
    print(f"Compressed payload size: {len(compressed_payload)} bytes")
    print(f"Compression ratio: {100 * len(compressed_payload) / len(payload):.1f}%")
    
    # Create frames with compressed data
    max_payload = max_payload_for_mtu(DEFAULT_ATT_MTU)
    print(f"Max payload per frame: {max_payload} bytes")
    
    frames = chunk_message(
        MessageType.PROMPT,
        message_id=1,
        payload=compressed_payload,
        max_payload_size=max_payload,
    )
    
    print(f"Number of frames: {len(frames)}")
    
    # Mark first frame as compressed
    frames[0] = Frame(
        msg_type=frames[0].msg_type,
        message_id=frames[0].message_id,
        seq=frames[0].seq,
        payload=frames[0].payload,
        flags=frames[0].flags | Flags.COMPRESSED,
        version=frames[0].version,
    )
    
    print(f"First frame flags: {frames[0].flags} (COMPRESSED={Flags.COMPRESSED in frames[0].flags})")
    
    # Reassemble using Reassembler (which should decompress)
    reassembler = Reassembler()
    
    result = None
    for i, frame in enumerate(frames):
        print(f"Feeding frame {i+1}/{len(frames)}...")
        result = reassembler.feed(frame)
        if result:
            print(f"Message complete at frame {i+1}")
            break
    
    if result is None:
        print("ERROR: Message not reassembled!")
        return False
    
    print(f"Reassembled payload size: {len(result.payload)} bytes")
    
    # Verify the decompressed data matches original
    reassembled_text = result.text()
    
    if reassembled_text == original_prompt:
        print("✓ SUCCESS: Compression/decompression works correctly!")
        print(f"  - Original: {len(payload)} bytes")
        print(f"  - Compressed: {len(compressed_payload)} bytes ({100 * len(compressed_payload) / len(payload):.1f}%)")
        print(f"  - Frames needed: {len(frames)} (vs {len(chunk_message(MessageType.PROMPT, 1, payload, max_payload))} without compression)")
        return True
    else:
        print("ERROR: Decompressed data doesn't match!")
        print(f"Expected: {original_prompt[:50]}...")
        print(f"Got: {reassembled_text[:50]}...")
        return False


def test_uncompressed_still_works():
    """Ensure uncompressed messages still work."""
    
    original_prompt = "Hello, world!"
    payload = original_prompt.encode("utf-8")
    
    max_payload = max_payload_for_mtu(DEFAULT_ATT_MTU)
    frames = chunk_message(
        MessageType.PROMPT,
        message_id=2,
        payload=payload,
        max_payload_size=max_payload,
    )
    
    reassembler = Reassembler()
    result = None
    for frame in frames:
        result = reassembler.feed(frame)
    
    if result and result.text() == original_prompt:
        print("✓ SUCCESS: Uncompressed messages still work!")
        return True
    else:
        print("ERROR: Uncompressed messages broken!")
        return False


if __name__ == "__main__":
    print("=" * 60)
    print("Testing compression functionality")
    print("=" * 60)
    print()
    
    success = True
    
    print("Test 1: Compressed messages")
    print("-" * 60)
    success = test_compression_decompression() and success
    print()
    
    print("Test 2: Uncompressed messages (backward compatibility)")
    print("-" * 60)
    success = test_uncompressed_still_works() and success
    print()
    
    print("=" * 60)
    if success:
        print("All tests passed! ✓")
    else:
        print("Some tests failed! ✗")
    print("=" * 60)
