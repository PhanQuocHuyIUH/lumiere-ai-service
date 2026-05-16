from __future__ import annotations

import logging
import os
import threading
from typing import Any, Callable

logger = logging.getLogger(__name__)

_cache: dict[str, tuple[float, Any]] = {}
_lock = threading.Lock()


def load_cached(path: str, loader: Callable[[str], Any]) -> Any | None:
    """Return cached payload if file mtime is unchanged, else reload via ``loader``.

    Returns ``None`` if the file is missing or the loader raises.
    The cache is keyed by absolute path + mtime so a fresh retrain (which rewrites
    the file) is picked up on the next request without a service restart.
    """
    if not os.path.exists(path):
        return None
    try:
        mtime = os.path.getmtime(path)
    except OSError:
        return None

    with _lock:
        cached = _cache.get(path)
        if cached is not None and cached[0] == mtime:
            return cached[1]

    try:
        value = loader(path)
    except Exception as exc:
        logger.warning("Failed to load %s: %s", path, exc)
        return None

    with _lock:
        _cache[path] = (mtime, value)
    return value


def invalidate(path: str) -> None:
    with _lock:
        _cache.pop(path, None)
