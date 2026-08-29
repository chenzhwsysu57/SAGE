"""Helpers for creating reproducible experiment artifact directories."""

from __future__ import annotations

import json
import os
import subprocess
from datetime import datetime
from pathlib import Path
from typing import Any


def slugify(text: str) -> str:
    safe = []
    for ch in text.lower():
        if ch.isalnum():
            safe.append(ch)
        elif ch in {"-", "_", ".", " "}:
            safe.append("_")
    slug = "".join(safe).strip("_")
    while "__" in slug:
        slug = slug.replace("__", "_")
    return slug or "run"


def git_commit() -> str | None:
    try:
        out = subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"],
            stderr=subprocess.DEVNULL,
            text=True,
        )
    except Exception:
        return None
    return out.strip() or None


def make_run_dir(output_root: str, slug: str) -> Path:
    root = Path(output_root)
    base_ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    base_slug = slugify(slug)
    run_dir = root / f"{base_ts}_{base_slug}"
    for attempt in range(1, 100):
        try:
            run_dir.mkdir(parents=True, exist_ok=False)
            break
        except FileExistsError:
            # Another process created the same second-resolution run_id
            # concurrently (parallel launchers). Retry with a numeric suffix.
            run_dir = root / f"{base_ts}_{base_slug}_{attempt}"
    else:
        raise FileExistsError(
            f"Could not allocate a unique run directory under {root} "
            f"for slug {slug!r} after 99 collision retries"
        )
    for name in ("trajectories", "metrics", "figures", "logs", "prompts"):
        (run_dir / name).mkdir(exist_ok=True)
    return run_dir


def write_json(path: str | os.PathLike[str], data: Any) -> None:
    with open(path, "w") as f:
        json.dump(data, f, indent=2, ensure_ascii=False, default=str)


def serializable_args(args: Any, *, redact: tuple[str, ...] = ("api_key",)) -> dict:
    data = dict(vars(args))
    for key in redact:
        if key in data and data[key]:
            data[key] = "<redacted>"
    return data
