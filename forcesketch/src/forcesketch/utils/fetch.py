"""Verified downloads and remote-ZIP member extraction (plan Tasks 0.3, 0.4).

The Zenodo record holding the reference committees is a 634 MB zip, but we need
four ~4 MB models from it. Zenodo honours HTTP Range requests, so we read the
central directory and inflate only the members we want -- under 5 MB transferred
instead of 634 MB. `read_remote_zip_index` falls back to a full download if the
server ever stops returning 206.
"""

from __future__ import annotations

import hashlib
import struct
import subprocess
import zlib
from dataclasses import dataclass
from pathlib import Path

EOCD_SIG = b"PK\x05\x06"
CEN_SIG = b"PK\x01\x02"


def sha256_file(path: Path, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        while block := fh.read(chunk):
            h.update(block)
    return h.hexdigest()


def _curl(url: str, *, byte_range: str | None = None, timeout: int = 180) -> bytes:
    cmd = ["curl", "-sL", "--fail", "--max-time", str(timeout)]
    if byte_range:
        cmd += ["-r", byte_range]
    cmd.append(url)
    res = subprocess.run(cmd, capture_output=True)
    if res.returncode != 0:
        raise RuntimeError(f"curl failed ({res.returncode}) for {url} range={byte_range}")
    return res.stdout


def content_length(url: str) -> int:
    out = subprocess.run(
        ["curl", "-sIL", "--max-time", "60", "-o", "/dev/null",
         "-w", "%{size_download}\n%{response_code}", url],
        capture_output=True, text=True,
    ).stdout
    res = subprocess.run(
        ["curl", "-sIL", "--max-time", "60", url], capture_output=True, text=True
    ).stdout
    for header_line in reversed(res.splitlines()):
        if header_line.lower().startswith("content-length:"):
            return int(header_line.split(":", 1)[1].strip())
    raise RuntimeError(f"no content-length for {url}: {out}")


def download(url: str, dest: Path, *, sha256: str | None = None, md5: str | None = None,
             expected_bytes: int | None = None) -> Path:
    """Idempotent verified download: skips when the target already verifies."""
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists():
        if sha256 and sha256_file(dest) == sha256:
            return dest
        if sha256 is None and expected_bytes and dest.stat().st_size == expected_bytes:
            return dest
    subprocess.run(["curl", "-sL", "--fail", "-o", str(dest), url], check=True)
    if expected_bytes is not None and dest.stat().st_size != expected_bytes:
        raise RuntimeError(f"{dest.name}: got {dest.stat().st_size} bytes, expected {expected_bytes}")
    if sha256:
        got = sha256_file(dest)
        if got != sha256:
            raise RuntimeError(f"{dest.name}: sha256 {got} != expected {sha256}")
    if md5:
        h = hashlib.md5()
        with open(dest, "rb") as fh:
            while block := fh.read(1 << 20):
                h.update(block)
        if h.hexdigest() != md5:
            raise RuntimeError(f"{dest.name}: md5 {h.hexdigest()} != expected {md5}")
    return dest


@dataclass(frozen=True, slots=True)
class ZipMember:
    name: str
    method: int
    comp_size: int
    uncomp_size: int
    header_offset: int


def read_remote_zip_index(url: str, total_size: int | None = None) -> dict[str, ZipMember]:
    """Enumerate a remote zip's members via Range requests, without downloading it."""
    if total_size is None:
        total_size = content_length(url)

    # The End Of Central Directory record lives in the last 64 KiB.
    tail_len = min(65_557, total_size)
    tail = _curl(url, byte_range=f"{total_size - tail_len}-{total_size - 1}")
    idx = tail.rfind(EOCD_SIG)
    if idx < 0:
        raise RuntimeError("no EOCD found; server may not honour Range requests")
    cd_size, cd_offset = struct.unpack_from("<II", tail, idx + 12)

    cd = _curl(url, byte_range=f"{cd_offset}-{cd_offset + cd_size - 1}")
    members: dict[str, ZipMember] = {}
    pos = 0
    while pos + 46 <= len(cd) and cd[pos:pos + 4] == CEN_SIG:
        method, = struct.unpack_from("<H", cd, pos + 10)
        comp_size, uncomp_size = struct.unpack_from("<II", cd, pos + 20)
        n_len, x_len, c_len = struct.unpack_from("<HHH", cd, pos + 28)
        lho, = struct.unpack_from("<I", cd, pos + 42)
        name = cd[pos + 46:pos + 46 + n_len].decode("utf-8", "replace")
        members[name] = ZipMember(name, method, comp_size, uncomp_size, lho)
        pos += 46 + n_len + x_len + c_len
    return members


def extract_remote_zip_member(url: str, member: ZipMember, dest: Path,
                              *, sha256: str | None = None) -> Path:
    """Range-fetch and inflate one member. Local header length is variable, so we
    read it first to find where the compressed payload actually starts."""
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists() and sha256 and sha256_file(dest) == sha256:
        return dest

    head = _curl(url, byte_range=f"{member.header_offset}-{member.header_offset + 29}")
    if head[:4] != b"PK\x03\x04":
        raise RuntimeError(f"bad local header for {member.name}")
    n_len, x_len = struct.unpack_from("<HH", head, 26)
    start = member.header_offset + 30 + n_len + x_len
    payload = _curl(url, byte_range=f"{start}-{start + member.comp_size - 1}")

    data = zlib.decompress(payload, -15) if member.method == 8 else payload
    if len(data) != member.uncomp_size:
        raise RuntimeError(
            f"{member.name}: inflated {len(data)} bytes, expected {member.uncomp_size}"
        )
    dest.write_bytes(data)
    if sha256:
        got = sha256_file(dest)
        if got != sha256:
            raise RuntimeError(f"{dest.name}: sha256 {got} != expected {sha256}")
    return dest
