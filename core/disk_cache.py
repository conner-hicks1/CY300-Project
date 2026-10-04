import hashlib
import os
import pickle
import threading

from pathlib import Path

from core.logger import Logger


# =========================================================
# Disk Cache
# =========================================================
#
# Results of the expensive, deterministic work kept on disk
# between runs (data/cache/, git-ignored): tectonic states
# step by step, climates, and terrain chunks. Loading a
# scene, coming back to a planet or reopening the engine
# reads them instead of simulating and building again.
#
# Keys are hashes of everything that determines a result:
# its inputs (settings, the states it was built from) and
# the source code of the generators (CODE_VERSION), so an
# edit to the engine never serves stale terrain. Entries
# are pickles the engine wrote itself (it never reads one
# from elsewhere). The oldest are deleted past a size limit.
#
# The cache is installed by the application
# (set_disk_cache); without one (tests, tools) everything is
# computed as before.

CACHE_DIRECTORY = Path("data/cache/simulation")

# Default size limit (bytes).
DEFAULT_LIMIT = 2 * 1024 ** 3

# Sources whose code shapes the cached results.
_SOURCE_GLOBS = ("planet/*.py", "graphics/mesh_data.py")


def _code_version() -> str:

    digest = hashlib.sha256()

    root = Path(__file__).resolve().parent.parent

    for pattern in _SOURCE_GLOBS:

        for path in sorted(root.glob(pattern)):

            digest.update(path.name.encode())
            digest.update(path.read_bytes())

    return digest.hexdigest()[:16]


CODE_VERSION = _code_version()


def cache_key(
    *parts
) -> str:
    """A key for a result determined by `parts` (their repr) and the generator code."""

    digest = hashlib.sha256(CODE_VERSION.encode())

    for part in parts:
        digest.update(repr(part).encode())
        digest.update(b"\0")

    return digest.hexdigest()[:40]


class DiskCache:

    def __init__(
        self,
        directory: Path = CACHE_DIRECTORY,
        limit_bytes: int = DEFAULT_LIMIT
    ):

        self.directory = Path(directory)
        self.limit_bytes = int(limit_bytes)

        self._lock = threading.Lock()

        self._written = 0

        # Diagnostics.
        self.hits = 0
        self.misses = 0

    def _path(
        self,
        kind: str,
        key: str
    ) -> Path:

        return self.directory / kind / key[:2] / f"{key}.pkl"

    def get(
        self,
        kind: str,
        key: str
    ):
        """The stored result, or None."""

        path = self._path(kind, key)

        try:

            with open(path, "rb") as file:
                value = pickle.load(file)

        except FileNotFoundError:

            with self._lock:
                self.misses += 1

            return None

        except Exception as error:

            # Truncated or from an incompatible build: drop it.
            Logger.warning("[Cache] Unreadable entry %s (%s); recomputing.", path.name, error)

            path.unlink(missing_ok=True)

            with self._lock:
                self.misses += 1

            return None

        # Recently used entries are pruned last.
        try:
            os.utime(path)
        except OSError:
            pass

        with self._lock:
            self.hits += 1

        return value

    def put(
        self,
        kind: str,
        key: str,
        value
    ):

        path = self._path(kind, key)

        try:

            path.parent.mkdir(parents=True, exist_ok=True)

            temporary = path.with_suffix(f".{threading.get_ident()}.tmp")

            with open(temporary, "wb") as file:
                pickle.dump(value, file, protocol=pickle.HIGHEST_PROTOCOL)

            # Atomic: readers never see half a file.
            os.replace(temporary, path)

            size = path.stat().st_size

        except OSError as error:

            Logger.warning("[Cache] Could not write %s: %s", path.name, error)

            return

        with self._lock:

            self._written += size

            prune = self._written > self.limit_bytes // 10

            if prune:
                self._written = 0

        if prune:
            self.prune()

    def prune(self):
        """Delete the least recently used entries beyond the size limit."""

        entries = []

        for path in self.directory.rglob("*.pkl"):

            try:
                stat = path.stat()
            except OSError:
                continue

            entries.append((stat.st_mtime, stat.st_size, path))

        total = sum(size for _, size, _ in entries)

        if total <= self.limit_bytes:
            return

        entries.sort()

        removed = 0

        for _, size, path in entries:

            if total <= self.limit_bytes * 0.8:
                break

            path.unlink(missing_ok=True)

            total -= size
            removed += 1

        Logger.info("[Cache] Pruned %d old entries (now %.0f MB).", removed, total / 1e6)

    def cached(
        self,
        kind: str,
        key: str | None,
        compute
    ):
        """The stored result for `key`, else compute() (stored). key None: compute only."""

        if key is None:
            return compute()

        value = self.get(kind, key)

        if value is not None:
            return value

        value = compute()

        self.put(kind, key, value)

        return value


_cache: DiskCache | None = None


def set_disk_cache(
    cache: DiskCache | None
):

    global _cache

    _cache = cache


def disk_cache() -> DiskCache | None:

    return _cache


def cached(
    kind: str,
    key: str | None,
    compute
):
    """Through the installed cache, if any."""

    cache = _cache

    if cache is None:
        return compute()

    return cache.cached(kind, key, compute)
