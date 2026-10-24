"""One atomic JSON write for the small state files (document selection, quiz bank, accessibility settings)."""

from __future__ import annotations

import contextlib
import json
import logging
import os
import tempfile
from pathlib import Path

log = logging.getLogger(__name__)


def save_json(path: Path, payload: object, what: str) -> bool:
    """Write `payload` to a temp file in `path`'s folder, then rename it into place, so a crash or a full disk never
    leaves half a file. On OSError logs a warning naming `what`, removes the temp file and returns False: the
    caller's in-memory value still applies, it just won't survive a restart."""
    tmp = ""
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        fd, tmp = tempfile.mkstemp(dir=path.parent, prefix=f".{path.name}.", suffix=".tmp")
        with os.fdopen(fd, "w", encoding="utf-8") as fh:
            json.dump(payload, fh)
        os.replace(tmp, path)
    except OSError as exc:
        log.warning("could not save %s to %s: %s", what, path, exc)
        if tmp:
            with contextlib.suppress(OSError):
                os.unlink(tmp)
        return False
    return True
