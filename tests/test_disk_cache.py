import os
import time

import numpy as np
import pytest

import core.disk_cache as disk_cache_module

from core.disk_cache import DiskCache, cache_key, set_disk_cache
from planet.bodies import components_for, load_presets
from planet.chunk import build_cached_chunk, build_chunk
from planet.cube_sphere import ChunkKey
from planet.terrain import Terrain, TerrainSettings
from systems.planet_system import terrain_settings_for


@pytest.fixture
def cache(tmp_path):

    cache = DiskCache(tmp_path / "cache", limit_bytes=10 ** 9)

    set_disk_cache(cache)

    yield cache

    set_disk_cache(None)


# =========================================================
# The Cache
# =========================================================

def test_round_trip_and_miss(cache):

    assert cache.get("things", "abc123") is None

    cache.put("things", "abc123", {"a": np.arange(5)})

    np.testing.assert_array_equal(cache.get("things", "abc123")["a"], np.arange(5))

    assert (cache.hits, cache.misses) == (1, 1)


def test_unreadable_entries_are_dropped(cache):

    cache.put("things", "deadbeef", [1, 2, 3])

    path = cache._path("things", "deadbeef")

    path.write_bytes(b"not a pickle")

    assert cache.get("things", "deadbeef") is None
    assert not path.exists()


def test_prune_keeps_the_newest(tmp_path):

    cache = DiskCache(tmp_path, limit_bytes=10 ** 9)

    for i in range(10):

        cache.put("things", f"{i:040d}", np.zeros(10_000))

        os.utime(cache._path("things", f"{i:040d}"), (time.time() - 100 + i, time.time() - 100 + i))

    cache.limit_bytes = 5 * 80_200

    cache.prune()

    kept = sorted(path.stem for path in (tmp_path / "things").rglob("*.pkl"))

    assert 2 <= len(kept) <= 5
    assert kept[-1] == f"{9:040d}"


def test_keys_follow_inputs_and_code(monkeypatch):

    a = cache_key("x", TerrainSettings(seed=1))

    assert a == cache_key("x", TerrainSettings(seed=1))
    assert a != cache_key("x", TerrainSettings(seed=2))

    monkeypatch.setattr(disk_cache_module, "CODE_VERSION", "edited")

    assert a != cache_key("x", TerrainSettings(seed=1))


# =========================================================
# What Is Cached
# =========================================================

def test_terrain_and_chunks(cache):

    profile = load_presets()["mercury"]

    parts = components_for(profile)

    settings = terrain_settings_for(parts.planet)

    terrain = Terrain(settings)

    assert terrain.cache_key is not None
    assert terrain.cache_key != Terrain(terrain_settings_for(components_for(load_presets()["moon"]).planet)).cache_key

    key = ChunkKey(2, 3, 1, 4)

    built = build_cached_chunk(key, terrain, 9)
    loaded = build_cached_chunk(key, Terrain(settings), 9)

    assert cache.hits == 1

    np.testing.assert_array_equal(loaded.mesh.vertices, build_chunk(key, terrain, 9).mesh.vertices)
    np.testing.assert_array_equal(loaded.mesh.vertices, built.mesh.vertices)
