> keytalk consume --address X --serve
16:21:48 [INFO] keytalk.ble.central: Connecting to BLE host at X...
16:21:52 [WARNING] keytalk.ble.central: Pairing attempt failed: Could not pair with device: FAILED — continuing without pairing; modes requiring bonding (l2cap_coc, rfcomm) may not be available
16:21:52 [INFO] keytalk.ble.central: Reconnecting after failed pairing attempt...
16:21:52 [INFO] keytalk.ble.central: Using prompt char handle=35, response char handle=38
16:21:52 [INFO] keytalk.ble.central: ✓ Connected to host
Traceback (most recent call last):
  File "<frozen runpy>", line 203, in _run_module_as_main
  File "<frozen runpy>", line 88, in _run_code
  File "...\Python3.14.6\Scripts\keytalk.exe\__main__.py", line 5, in <module>
    sys.exit(main())
             ~~~~^^
  File "...\keytalk\src\keytalk\cli.py", line 323, in main
    return asyncio.run(runner(args))
           ~~~~~~~~~~~^^^^^^^^^^^^^^
  File "...\Python3.14.6\Lib\asyncio\runners.py", line 205, in run
    return runner.run(main)
           ~~~~~~~~~~^^^^^^
  File "...\Python3.14.6\Lib\asyncio\runners.py", line 128, in run
    return self._loop.run_until_complete(task)
           ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~^^^^^^
  File "...\Python3.14.6\Lib\asyncio\base_events.py", line 719, in run_until_complete
    return future.result()
           ~~~~~~~~~~~~~^^
  File "...\keytalk\src\keytalk\cli.py", line 203, in _run_consume
    await client.start()
  File "...\keytalk\src\keytalk\consumer.py", line 231, in start
    await self._transport.start()
  File "...\keytalk\src\keytalk\ble\central.py", line 91, in start
    await self._connect()
  File "...\keytalk\src\keytalk\ble\central.py", line 102, in _connect
    await self._resolve_and_subscribe()
  File "...\keytalk\src\keytalk\ble\central.py", line 203, in _resolve_and_subscribe
    await self._client.start_notify(self._response_char_obj, _notification_handler)
  File "...\Python3.14.6\Lib\site-packages\bleak\__init__.py", line 904, in start_notify
    await self._backend.start_notify(
        characteristic, wrapped_callback, bluez=bluez, cb=cb, **kwargs
    )
  File "...\Python3.14.6\Lib\site-packages\bleak\backends\winrt\client.py", line 1035, in start_notify
    await winrt_char.write_client_characteristic_configuration_descriptor_with_result_async(
    ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
        cccd
        ^^^^
    ),
    ^
  File "...\Python3.14.6\Lib\site-packages\winrt\runtime\_internals.py", line 245, in wait
    return op.get_results()
           ~~~~~~~~~~~~~~^^
OSError: [WinError -2147483629] The object has been closed.