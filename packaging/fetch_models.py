"""Put the models a desktop build ships with into packaging/models/, so the build works offline.

    python packaging/fetch_models.py               # all-MiniLM-L6-v2 embeddings (~90 MB), what CI runs
    python packaging/fetch_models.py --whisper     # + Whisper "base" (~140 MB) instead of a first-run download

The folder uses the Hugging Face cache layout (models--org--name/...), which is what the launcher points
SENTENCE_TRANSFORMERS_HOME / HF_HOME at; symlinks are replaced by real files so PyInstaller and zip copy
them faithfully. Running it again with the model in place only re-checks it.
"""

from __future__ import annotations

import argparse
import os
import shutil
import sys
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any

DEFAULT_DEST = Path(__file__).resolve().parent / "models"
DEFAULT_EMBED_MODEL = "all-MiniLM-L6-v2"
DEFAULT_WHISPER_MODEL = "base"
WEIGHT_FILES = ("model.safetensors", "pytorch_model.bin")

# (model name, cache folder, local_files_only) -> anything with .encode(list[str])
Loader = Callable[[str, Path, bool], Any]


def repo_id(model: str) -> str:
    """sentence-transformers resolves bare official names under its own organisation."""
    return model if "/" in model else f"sentence-transformers/{model}"


def repo_dir(dest: Path, model: str) -> Path:
    return dest / f"models--{repo_id(model).replace('/', '--')}"


def snapshot_dir(dest: Path, model: str) -> Path | None:
    repo = repo_dir(dest, model)
    try:
        revision = (repo / "refs" / "main").read_text(encoding="utf-8").strip()
    except OSError:
        return None
    snapshot = repo / "snapshots" / revision
    return snapshot if snapshot.is_dir() else None


def is_complete(dest: Path, model: str) -> bool:
    snapshot = snapshot_dir(dest, model)
    if snapshot is None or not (snapshot / "modules.json").is_file():
        return False
    if not any((snapshot / name).is_file() for name in WEIGHT_FILES):
        return False
    return not any(path.is_symlink() for path in snapshot.rglob("*"))


def materialize(dest: Path) -> int:
    """Replace the cache's snapshot symlinks with the files they point at, then drop blobs/ and locks."""
    replaced = 0
    for repo in dest.glob("models--*"):
        for link in sorted((repo / "snapshots").rglob("*")):
            if link.is_symlink():
                target = link.resolve(strict=True)
                link.unlink()
                shutil.copy2(target, link)
                replaced += 1
        shutil.rmtree(repo / "blobs", ignore_errors=True)
    shutil.rmtree(dest / ".locks", ignore_errors=True)
    return replaced


def _sentence_transformer(model: str, cache_folder: Path, local_files_only: bool) -> Any:
    from sentence_transformers import SentenceTransformer

    return SentenceTransformer(model, cache_folder=str(cache_folder), local_files_only=local_files_only)


def _verify(model: str, dest: Path, load: Loader) -> None:
    vectors = load(model, dest, True).encode(["free-living amoeba"])
    if len(vectors) != 1 or len(vectors[0]) == 0:
        raise RuntimeError(f"{model} loaded from {dest} but produced no embedding")


def fetch_embedding_model(dest: Path, model: str = DEFAULT_EMBED_MODEL, load: Loader = _sentence_transformer) -> str:
    """Download (or re-check) the embedding model; returns what happened, for the log."""
    dest.mkdir(parents=True, exist_ok=True)
    if is_complete(dest, model):
        materialize(dest)  # drop leftover blobs/ so they aren't bundled twice
        try:
            _verify(model, dest, load)
            return f"{model}: already in {dest}"
        except Exception as exc:
            print(f"{model}: cached copy unusable ({exc}); downloading again", file=sys.stderr)
    # Start clean: a half-finished copy (interrupted CI job, edited by hand) is not worth repairing.
    shutil.rmtree(repo_dir(dest, model), ignore_errors=True)
    load(model, dest, False)
    replaced = materialize(dest)
    if not is_complete(dest, model):
        raise RuntimeError(f"{model}: download finished but {repo_dir(dest, model)} is incomplete")
    _verify(model, dest, load)
    return f"{model}: downloaded to {dest} ({replaced} links replaced by files)"


def _whisper_download(name: str, root: Path) -> Any:
    import whisper

    return whisper.load_model(name, device="cpu", download_root=str(root))


def fetch_whisper_model(
    dest: Path, name: str = DEFAULT_WHISPER_MODEL, download: Callable[[str, Path], Any] = _whisper_download
) -> str:
    """Whisper keeps one checkpoint file per model; the launcher points SAR_WHISPER_MODEL at it."""
    root = dest / "whisper"
    checkpoint = root / f"{name}.pt"
    if checkpoint.is_file():
        return f"whisper {name}: already in {root}"
    root.mkdir(parents=True, exist_ok=True)
    download(name, root)  # verifies the checkpoint's SHA-256
    if not checkpoint.is_file():
        raise RuntimeError(f"whisper {name}: no {checkpoint.name} in {root} after download")
    return f"whisper {name}: downloaded to {root}"


def folder_size_mb(folder: Path) -> float:
    total = sum(p.stat().st_size for p in folder.rglob("*") if p.is_file() and not p.is_symlink())
    return total / 1_000_000


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--dest", type=Path, default=DEFAULT_DEST, help=f"target folder (default: {DEFAULT_DEST})")
    parser.add_argument(
        "--embed-model",
        default=os.environ.get("SAR_EMBED_MODEL", DEFAULT_EMBED_MODEL),
        help="sentence-transformers model (default: SAR_EMBED_MODEL or all-MiniLM-L6-v2)",
    )
    parser.add_argument("--whisper", action="store_true", help="also bundle a Whisper model (~140 MB for base)")
    parser.add_argument(
        "--whisper-model",
        default=os.environ.get("SAR_WHISPER_MODEL", DEFAULT_WHISPER_MODEL),
        help="Whisper model for --whisper (default: SAR_WHISPER_MODEL or base)",
    )
    args = parser.parse_args(argv)
    dest: Path = args.dest.resolve()
    try:
        print(fetch_embedding_model(dest, args.embed_model))
        if args.whisper:
            print(fetch_whisper_model(dest, args.whisper_model))
    except Exception as exc:
        print(f"fetch_models failed: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1
    print(f"{dest}: {folder_size_mb(dest):.0f} MB")
    return 0


if __name__ == "__main__":
    sys.exit(main())
