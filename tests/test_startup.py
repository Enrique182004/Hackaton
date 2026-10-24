from __future__ import annotations

import socket
from pathlib import Path
from typing import Any

from biobench_sar.app import _port_free, seed_knowledge
from biobench_sar.config import PACKAGE_DIR, Settings


class Store:
    def __init__(self, existing: int = 0) -> None:
        self.chunks: list[Any] = []
        self.existing = existing

    def count(self) -> int:
        return self.existing + len(self.chunks)

    def add_chunks(self, chunks: list[Any]) -> int:
        self.chunks.extend(chunks)
        return len(chunks)


def test_seed_fills_an_empty_store_from_bundled_docs() -> None:
    store = Store()
    added = seed_knowledge(store, PACKAGE_DIR / "knowledge")
    assert added > 0 and added == len(store.chunks)
    assert {c.source for c in store.chunks} >= {"acanthamoeba_isolation_protocol.md", "laguna_bustillos_site.md"}


def test_seed_leaves_a_populated_store_alone() -> None:
    store = Store(existing=3)
    assert seed_knowledge(store, PACKAGE_DIR / "knowledge") == 0
    assert store.chunks == []


def test_seed_skips_unsupported_and_missing(tmp_path: Path) -> None:
    (tmp_path / "notes.txt").write_text("Autoclave media before plating.")
    (tmp_path / "binary.exe").write_bytes(b"MZ")
    store = Store()
    assert seed_knowledge(store, tmp_path) == 1
    assert seed_knowledge(Store(), tmp_path / "missing") == 0


def test_port_free_detects_a_bound_port() -> None:
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        sock.listen()
        port = sock.getsockname()[1]
        assert _port_free("127.0.0.1", port) is False


def test_default_port_avoids_airplay(monkeypatch: Any) -> None:
    monkeypatch.delenv("SAR_PORT", raising=False)
    assert Settings().port == 5050
