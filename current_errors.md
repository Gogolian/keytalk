16:55:32 [ERROR] keytalk.server: error streaming completion
Traceback (most recent call last):
  File "...\Python3.14.6\Lib\asyncio\tasks.py", line 488, in wait_for
    return await fut
           ^^^^^^^^^
  File "...\keytalk\src\keytalk\consumer.py", line 169, in __aiter__
    item = await self._queue.get()
           ^^^^^^^^^^^^^^^^^^^^^^^
  File "...\Python3.14.6\Lib\asyncio\queues.py", line 186, in get
    await getter
asyncio.exceptions.CancelledError

The above exception was the direct cause of the following exception:

Traceback (most recent call last):
  File "...\keytalk\src\keytalk\server.py", line 804, in _stream_ndjson
    async for piece in self._client.stream(prompt):
    ...<2 lines>...
        await self._write_chunk_json(writer, envelope(piece, False))
  File "...\keytalk\src\keytalk\consumer.py", line 455, in _stream
    piece = await asyncio.wait_for(
            ^^^^^^^^^^^^^^^^^^^^^^^
        iterator.__anext__(), timeout=self._timeout
        ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
    )
    ^
  File "...\Python3.14.6\Lib\asyncio\tasks.py", line 487, in wait_for
    async with timeouts.timeout(timeout):
               ~~~~~~~~~~~~~~~~^^^^^^^^^
  File "...\Python3.14.6\Lib\asyncio\timeouts.py", line 115, in __aexit__
    raise TimeoutError from exc_val
TimeoutError