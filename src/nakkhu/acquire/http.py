"""HTTP helpers: HEAD probes, resumable downloads with checksums, and a range-read file object.

Downloads stream into ``<cache>/<key>.part`` and resume with a ``Range`` request when the server
validator (ETag / Last-Modified) is unchanged. On completion the size is checked against
``Content-Length`` and, where the server publishes one, the MD5 (plain S3 ETag, ``x-goog-hash``,
``Content-MD5``); then the file is moved into place with an atomic rename.
"""

from __future__ import annotations

import base64
import hashlib
import io
import json
import os
import re
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import requests

USER_AGENT = "nakkhu-data-fetch/0.1 (academic research; python-requests)"
TIMEOUT = (30, 120)
CHUNK = 1 << 20
_HEX_MD5 = re.compile(r'^"?([0-9a-f]{32})"?$')


class FetchError(RuntimeError):
    """A remote resource could not be fetched or failed an integrity check."""


def session() -> requests.Session:
    s = requests.Session()
    s.headers["User-Agent"] = USER_AGENT
    return s


@dataclass
class RemoteInfo:
    url: str
    final_url: str
    content_length: int | None
    etag: str | None
    last_modified: str | None
    accept_ranges: bool
    md5: str | None  # hex MD5 published by the server, if any

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


def _server_md5(headers: Any, host: str) -> str | None:
    for value in headers.get("x-goog-hash", "").split(","):
        value = value.strip()
        if value.startswith("md5="):
            return base64.b64decode(value[4:]).hex()
    if headers.get("Content-MD5"):
        return base64.b64decode(headers["Content-MD5"]).hex()
    match = _HEX_MD5.match(headers.get("ETag", "") or "")
    # A plain 32-hex ETag is the MD5 on S3 / GCS single-part uploads; other servers (nginx,
    # Azure) use different ETag schemes, so only trust it for those hosts.
    if match and (host.endswith(".amazonaws.com") or host == "storage.googleapis.com"):
        return match.group(1)
    return None


def _retry(call: Any, attempts: int = 6) -> Any:
    for attempt in range(attempts):
        try:
            return call()
        except requests.RequestException:
            if attempt == attempts - 1:
                raise
            time.sleep(2 ** (attempt + 1))
    raise FetchError("unreachable")


def head(url: str, s: requests.Session | None = None) -> RemoteInfo:
    return _retry(lambda: _head(url, s or session()))


def _head(url: str, s: requests.Session) -> RemoteInfo:
    response = s.head(url, allow_redirects=True, timeout=TIMEOUT)
    if response.status_code in (403, 405) or "Content-Length" not in response.headers:
        response = s.get(url, headers={"Range": "bytes=0-0"}, stream=True, timeout=TIMEOUT)
        response.close()
    if response.status_code >= 400:
        raise FetchError(f"HEAD {url}: HTTP {response.status_code}")
    headers = response.headers
    size: int | None = None
    if response.status_code == 206 and "Content-Range" in headers:
        size = int(headers["Content-Range"].rsplit("/", 1)[1])
    elif "Content-Length" in headers:
        size = int(headers["Content-Length"])
    host = requests.utils.urlparse(response.url).netloc
    return RemoteInfo(
        url=url,
        final_url=response.url,
        content_length=size,
        etag=headers.get("ETag"),
        last_modified=headers.get("Last-Modified"),
        accept_ranges=headers.get("Accept-Ranges", "").lower() == "bytes"
        or response.status_code == 206,
        md5=_server_md5(response.headers, host),
    )


@dataclass
class Downloaded:
    path: Path
    bytes: int
    sha256: str
    md5: str
    md5_verified: bool
    info: RemoteInfo

    def record(self, rel: str) -> dict[str, Any]:
        return {
            "path": rel,
            "url": self.info.url,
            "final_url": self.info.final_url,
            "etag": self.info.etag,
            "last_modified": self.info.last_modified,
            "content_length": self.info.content_length,
            "bytes": self.bytes,
            "sha256": self.sha256,
            "md5": self.md5,
            "md5_verified_against_server": self.md5_verified,
        }


def _cache_key(url: str) -> str:
    return hashlib.sha1(url.encode()).hexdigest()[:20]


def download(
    url: str,
    dest: str | Path,
    cache_dir: str | Path,
    *,
    retries: int = 8,
    s: requests.Session | None = None,
    progress: bool = True,
) -> Downloaded:
    """Download ``url`` to ``dest`` (resumable through ``cache_dir``), verifying size and MD5."""
    s = s or session()
    dest = Path(dest)
    cache = Path(cache_dir)
    cache.mkdir(parents=True, exist_ok=True)
    key = _cache_key(url)
    part = cache / f"{key}.part"
    meta_path = cache / f"{key}.json"
    info = head(url, s)
    validator = {"etag": info.etag, "last_modified": info.last_modified,
                 "size": info.content_length, "final_url": info.final_url}
    if meta_path.is_file() and json.loads(meta_path.read_text()) != validator:
        part.unlink(missing_ok=True)  # remote changed: restart
    meta_path.write_text(json.dumps(validator))

    attempt = 0
    last_report = 0.0
    while True:
        have = part.stat().st_size if part.exists() else 0
        if info.content_length is not None and have == info.content_length:
            break
        if info.content_length is not None and have > info.content_length:
            part.unlink()
            have = 0
        headers = {}
        if have and info.accept_ranges:
            headers["Range"] = f"bytes={have}-"
        elif have:
            part.unlink()
            have = 0
        try:
            with s.get(info.final_url, headers=headers, stream=True, timeout=TIMEOUT) as r:
                if r.status_code == 200 and have:
                    part.unlink()  # server ignored Range
                    have = 0
                elif r.status_code not in (200, 206):
                    raise FetchError(f"GET {url}: HTTP {r.status_code}")
                with part.open("ab") as handle:
                    for chunk in r.iter_content(CHUNK):
                        handle.write(chunk)
                        have += len(chunk)
                        if progress and time.monotonic() - last_report > 30:
                            last_report = time.monotonic()
                            total = info.content_length or 0
                            pct = f" ({100 * have / total:.0f} %)" if total else ""
                            print(f"  ... {Path(dest).name}: {have / 1e6:.0f} MB{pct}", flush=True)
            if info.content_length is None:
                break
        except (requests.RequestException, FetchError) as error:
            attempt += 1
            if attempt > retries:
                raise FetchError(f"download failed after {retries} retries: {url}: {error}") from error
            time.sleep(min(60, 2**attempt))

    sha = hashlib.sha256()
    md5 = hashlib.md5()
    size = 0
    with part.open("rb") as handle:
        while chunk := handle.read(CHUNK):
            sha.update(chunk)
            md5.update(chunk)
            size += len(chunk)
    if info.content_length is not None and size != info.content_length:
        raise FetchError(f"{url}: size {size} != Content-Length {info.content_length}")
    verified = False
    if info.md5:
        if md5.hexdigest() != info.md5:
            part.unlink()
            raise FetchError(f"{url}: MD5 mismatch (server {info.md5}, got {md5.hexdigest()})")
        verified = True
    dest.parent.mkdir(parents=True, exist_ok=True)
    os.replace(part, dest)
    meta_path.unlink(missing_ok=True)
    return Downloaded(dest, size, sha.hexdigest(), md5.hexdigest(), verified, info)


def get_json(url: str, s: requests.Session | None = None, **kwargs: Any) -> Any:
    s = s or session()
    for attempt in range(5):
        try:
            response = s.get(url, timeout=TIMEOUT, **kwargs)
            response.raise_for_status()
            return response.json()
        except requests.RequestException:
            if attempt == 4:
                raise
            time.sleep(2 ** (attempt + 1))
    raise FetchError(url)


def post_json(url: str, body: Any, s: requests.Session | None = None) -> Any:
    s = s or session()

    def call() -> Any:
        response = s.post(url, json=body, timeout=TIMEOUT)
        response.raise_for_status()
        return response.json()

    return _retry(call)


class RangeFile(io.RawIOBase):
    """Read-only, seekable view of a remote file through HTTP ``Range`` requests (block-cached).

    Lets :mod:`zipfile` read one member of a large remote zip without downloading all of it.
    """

    def __init__(self, url: str, size: int, s: requests.Session | None = None,
                 block: int = 4 << 20) -> None:
        self.url = url
        self.size = size
        self.s = s or session()
        self.block = block
        self.pos = 0
        self.cache: dict[int, bytes] = {}
        self.transferred = 0

    def readable(self) -> bool:
        return True

    def seekable(self) -> bool:
        return True

    def tell(self) -> int:
        return self.pos

    def seek(self, offset: int, whence: int = 0) -> int:
        base = {0: 0, 1: self.pos, 2: self.size}[whence]
        self.pos = max(0, base + offset)
        return self.pos

    def _block(self, index: int) -> bytes:
        if index not in self.cache:
            start = index * self.block
            end = min(self.size, start + self.block) - 1
            for attempt in range(6):
                try:
                    r = self.s.get(self.url, headers={"Range": f"bytes={start}-{end}"},
                                   timeout=TIMEOUT)
                    if r.status_code != 206:
                        raise FetchError(f"range read {self.url}: HTTP {r.status_code}")
                    break
                except (requests.RequestException, FetchError):
                    if attempt == 5:
                        raise
                    time.sleep(2 ** (attempt + 1))
            self.transferred += len(r.content)
            if len(self.cache) > 16:
                self.cache.pop(next(iter(self.cache)))
            self.cache[index] = r.content
        return self.cache[index]

    def readinto(self, buffer: Any) -> int:
        want = min(len(buffer), self.size - self.pos)
        if want <= 0:
            return 0
        out = bytearray()
        while len(out) < want:
            index, offset = divmod(self.pos + len(out), self.block)
            data = self._block(index)[offset:]
            out += data[: want - len(out)]
        buffer[: len(out)] = out
        self.pos += len(out)
        return len(out)
