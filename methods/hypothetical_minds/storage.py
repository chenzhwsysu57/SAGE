"""HM-owned artifacts only; deterministic JSON plus immutable evidence records."""

import hashlib
import json
import os
from pathlib import Path
import tempfile

from base.baseline_calls import ResumeConflict
from .protocol import canonical, reject_constant, unique_object


def digest(value) -> str:
    return hashlib.sha256(canonical(value).encode()).hexdigest()


def read_json(path: Path):
    try:
        if path.is_symlink():
            raise ValueError("Symlink")
        return json.loads(path.read_text(encoding="utf-8"), object_pairs_hook=unique_object,
                          parse_constant=reject_constant)
    except (OSError, ValueError):
        raise ResumeConflict("Unreadable or malformed HM artifact") from None


def atomic_json(path: Path, value):
    path = Path(path)
    if any(p.is_symlink() for p in (path, *path.parents)):
        raise ResumeConflict("HM artifacts cannot use symlink paths")
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, name = tempfile.mkstemp(prefix=".hm-", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as stream:
            stream.write(canonical(value))
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(name, path)
    finally:
        Path(name).unlink(missing_ok=True)


def immutable_json(path: Path, value):
    if path.exists():
        if canonical(read_json(path)) != canonical(value):
            raise ResumeConflict("HM evidence changed on replay")
    else:
        atomic_json(path, value)