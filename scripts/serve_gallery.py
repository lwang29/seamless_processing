#!/usr/bin/env python3
"""Serve a review gallery over localhost, with HTTP range requests.

The reviewer could not drag the progress bar: the dot snapped back every time.
Two things cause that, and both had to be fixed.

The first is in the clip. libx264's default GOP is 250 frames, so a thirty-second
render held three keyframes and a seek could only ever land on 0, 10 or 20
seconds. The renderer now asks for one per second.

The second is here. A player seeks by asking for a byte range, and
``http.server`` answers every request with ``200`` and the whole file --
it has never implemented ``Range``. Chrome and Safari treat that as
"this resource is not seekable", give up, and restore the previous position,
which is exactly the snap-back. This handler answers ``206 Partial Content``
instead.

Binds to 127.0.0.1 only. Participant media stays on the cluster; reach it with
``ssh -L 8000:localhost:8000`` or VS Code's port forwarding, the same as before.
"""

from __future__ import annotations

import argparse
import functools
import os
import re
import socketserver
from http import HTTPStatus
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

RANGE_PATTERN = re.compile(r"^bytes=(\d*)-(\d*)$")


class RangeRequestHandler(SimpleHTTPRequestHandler):
    """``SimpleHTTPRequestHandler`` that honours a single-range ``Range`` header."""

    # Keep-alive: a player scrubbing a 30 s clip issues a burst of small ranges,
    # and HTTP/1.0 would tear down the connection after each one.
    protocol_version = "HTTP/1.1"

    def send_head(self):  # noqa: N802 - base-class name
        header = self.headers.get("Range")
        if not header:
            # Still advertise the capability, so a player knows it may seek.
            return self._send_whole_file()
        match = RANGE_PATTERN.match(header.strip())
        if not match:
            return self._send_whole_file()

        path = self.translate_path(self.path)
        if os.path.isdir(path):
            return super().send_head()
        try:
            handle = open(path, "rb")
        except OSError:
            self.send_error(HTTPStatus.NOT_FOUND, "File not found")
            return None

        try:
            size = os.fstat(handle.fileno()).st_size
            first_text, last_text = match.group(1), match.group(2)
            if first_text:
                first = int(first_text)
                last = int(last_text) if last_text else size - 1
            else:
                # A suffix range: the final N bytes.
                if not last_text:
                    handle.close()
                    self.send_error(HTTPStatus.BAD_REQUEST, "Malformed range")
                    return None
                first = max(0, size - int(last_text))
                last = size - 1
            last = min(last, size - 1)
            if first >= size or first > last:
                handle.close()
                self.send_response(HTTPStatus.REQUESTED_RANGE_NOT_SATISFIABLE)
                self.send_header("Content-Range", f"bytes */{size}")
                self.end_headers()
                return None

            self.send_response(HTTPStatus.PARTIAL_CONTENT)
            self.send_header("Content-Type", self.guess_type(path))
            self.send_header("Accept-Ranges", "bytes")
            self.send_header("Content-Range", f"bytes {first}-{last}/{size}")
            self.send_header("Content-Length", str(last - first + 1))
            self.end_headers()
            handle.seek(first)
            self._remaining = last - first + 1
            return _BoundedReader(handle, self._remaining)
        except Exception:
            handle.close()
            raise

    def _send_whole_file(self):
        result = super().send_head()
        return result

    def end_headers(self) -> None:
        if "Accept-Ranges" not in self._headers_buffer_keys():
            self.send_header("Accept-Ranges", "bytes")
        super().end_headers()

    def _headers_buffer_keys(self) -> set[str]:
        keys = set()
        for raw in getattr(self, "_headers_buffer", []) or []:
            text = raw.decode("latin-1", "replace")
            if ":" in text:
                keys.add(text.split(":", 1)[0])
        return keys

    def log_message(self, fmt: str, *args) -> None:  # noqa: A002
        if self.path.endswith((".mp4", ".png", ".ico")):
            return
        super().log_message(fmt, *args)


class _BoundedReader:
    """A file object that stops after ``limit`` bytes, for ``copyfile``."""

    def __init__(self, handle, limit: int) -> None:
        self._handle = handle
        self._remaining = limit

    def read(self, size: int = -1) -> bytes:
        if self._remaining <= 0:
            return b""
        want = self._remaining if size < 0 else min(size, self._remaining)
        chunk = self._handle.read(want)
        self._remaining -= len(chunk)
        return chunk

    def close(self) -> None:
        self._handle.close()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path, help="gallery directory to serve")
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args()

    root = args.directory.resolve()
    if not root.is_dir():
        raise SystemExit(f"not a directory: {root}")

    handler = functools.partial(RangeRequestHandler, directory=str(root))
    socketserver.TCPServer.allow_reuse_address = True
    with ThreadingHTTPServer(("127.0.0.1", args.port), handler) as server:
        print(f"serving {root} on http://127.0.0.1:{args.port}/  (range requests enabled)")
        print("forward it with:  ssh -N -L "
              f"{args.port}:localhost:{args.port} <this-host>")
        try:
            server.serve_forever()
        except KeyboardInterrupt:
            print("\nstopped")


if __name__ == "__main__":
    main()
