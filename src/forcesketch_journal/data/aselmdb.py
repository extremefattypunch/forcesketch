"""Minimal reader for fairchem ``.aselmdb`` files (OMat24), without fairchem.

An ``.aselmdb`` is an LMDB whose integer keys hold zlib-compressed JSON of an
``ase.db`` row dictionary (positions, numbers, cell, pbc, calculator results).
``ase.io.jsonio.decode`` turns the JSON back into Python/NumPy objects and
``ase.db.row.AtomsRow`` rebuilds the ``Atoms`` with its energy and forces on a
SinglePointCalculator. Reserved keys (``nextid``, ``deleted_ids``, ``metadata``)
are skipped. Reading through fairchem would need its whole dependency stack
beside a pinned torch; this is ~40 lines and verified against the stored
``metadata.npz`` atom counts.
"""

from __future__ import annotations

import pathlib
import zlib

import lmdb
import orjson
from ase.db.row import AtomsRow
from ase.io.jsonio import decode

RESERVED = {b"nextid", b"deleted_ids", b"metadata"}


def _decode(raw: bytes) -> dict:
    return decode(zlib.decompress(raw).decode("utf-8"))


class AseLMDB:
    def __init__(self, path: str | pathlib.Path):
        self.path = pathlib.Path(path)
        self.env = lmdb.open(str(self.path), subdir=False, readonly=True, lock=False,
                             readahead=True, max_readers=1)
        with self.env.begin() as txn:
            nextid = txn.get(b"nextid")          # every value is zlib(json), keys too
            self.nextid = int(_decode(nextid)) if nextid else None
            deleted = txn.get(b"deleted_ids")
            self.deleted = set(_decode(deleted)) if deleted else set()

    def __len__(self) -> int:
        if self.nextid is None:
            with self.env.begin() as txn:
                return sum(1 for k, _ in txn.cursor() if k not in RESERVED)
        return self.nextid - 1 - len(self.deleted)

    def ids(self) -> list[int]:
        with self.env.begin() as txn:
            return sorted(int(k.decode()) for k, _ in txn.cursor() if k not in RESERVED)

    def get(self, row_id: int):
        with self.env.begin() as txn:
            raw = txn.get(f"{row_id}".encode("ascii"))
        if raw is None:
            raise KeyError(row_id)
        d = _decode(raw)
        row = AtomsRow(d)
        atoms = row.toatoms(add_additional_information=True)
        return atoms

    def __iter__(self):
        with self.env.begin() as txn:
            for k, v in txn.cursor():
                if k in RESERVED:
                    continue
                yield int(k.decode()), AtomsRow(_decode(v)).toatoms(add_additional_information=True)

    def close(self) -> None:
        self.env.close()
