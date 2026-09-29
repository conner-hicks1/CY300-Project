import numpy as np
import pytest

from editor.history import Snapshot, UndoHistory
from editor.picking import PickCandidate, pick, ray_box_distance, screen_ray
from math3d.matrices import look_at, perspective
from math3d.transform import Transform


# =========================================================
# Undo History
# =========================================================

def snap(text, label=""):
    return Snapshot(scene=text, label=label)


def test_undo_redo_walks_states():

    history = UndoHistory()
    history.reset(snap("a"))

    history.push(snap("b", "Move"))
    history.push(snap("c", "Rotate"))

    assert history.undo_label == "Rotate"
    assert history.undo().scene == "b"
    assert history.undo().scene == "a"
    assert history.undo() is None

    assert history.redo().scene == "b"
    assert history.redo_label == "Rotate"
    assert history.redo().scene == "c"
    assert history.redo() is None


def test_push_after_undo_discards_redo():

    history = UndoHistory()
    history.reset(snap("a"))
    history.push(snap("b"))
    history.undo()

    history.push(snap("x"))

    assert not history.can_redo
    assert history.undo().scene == "a"


def test_unchanged_push_is_ignored():

    history = UndoHistory()
    history.reset(snap("a"))

    assert history.push(snap("a")) is False
    assert not history.can_undo


def test_limit_drops_oldest():

    history = UndoHistory(limit=2)
    history.reset(snap("0"))

    for text in "1234":
        history.push(snap(text))

    assert history.undo().scene == "3"
    assert history.undo().scene == "2"
    assert history.undo() is None


# =========================================================
# Picking
# =========================================================

def test_ray_box_hit_and_miss():

    box = (np.array([-1.0, -1.0, -1.0]), np.array([1.0, 1.0, 1.0]))

    assert ray_box_distance(np.array([0.0, 0.0, 5.0]), np.array([0.0, 0.0, -1.0]), *box) == pytest.approx(4.0)
    assert ray_box_distance(np.array([3.0, 0.0, 5.0]), np.array([0.0, 0.0, -1.0]), *box) is None
    assert ray_box_distance(np.array([0.0, 0.0, 5.0]), np.array([0.0, 0.0, 1.0]), *box) is None
    assert ray_box_distance(np.zeros(3), np.array([0.0, 0.0, 1.0]), *box) == 0.0


def test_ray_hits_flat_box():

    # A plane has zero height.
    assert ray_box_distance(
        np.array([0.2, 3.0, 0.1]),
        np.array([0.0, -1.0, 0.0]),
        np.array([-0.5, 0.0, -0.5]),
        np.array([0.5, 0.0, 0.5])
    ) == pytest.approx(3.0)


def test_screen_center_ray_points_forward():

    view = look_at((0.0, 0.0, 5.0), (0.0, 0.0, 0.0), (0.0, 1.0, 0.0))
    projection = perspective(60.0, 16 / 9, 0.1, 100.0)

    ray = screen_ray(640, 360, 1280, 720, view, projection)

    assert np.allclose(ray.direction, (0.0, 0.0, -1.0), atol=1e-6)
    assert ray.origin[2] == pytest.approx(4.9, abs=1e-4)


def test_screen_ray_corner_goes_up_left():

    view = look_at((0.0, 0.0, 5.0), (0.0, 0.0, 0.0), (0.0, 1.0, 0.0))
    projection = perspective(60.0, 1.0, 0.1, 100.0)

    ray = screen_ray(0, 0, 800, 800, view, projection)

    assert ray.direction[0] < 0.0
    assert ray.direction[1] > 0.0


def test_pick_nearest_through_transforms():

    view = look_at((0.0, 0.0, 10.0), (0.0, 0.0, 0.0), (0.0, 1.0, 0.0))
    projection = perspective(60.0, 1.0, 0.1, 100.0)
    ray = screen_ray(400, 400, 800, 800, view, projection)

    unit = (np.full(3, -0.5), np.full(3, 0.5))

    near = Transform(position=(0.0, 0.0, 2.0)).matrix
    far = Transform(position=(0.0, 0.0, -2.0), scale=(5, 5, 5)).matrix

    # Rotated 45 degrees and moved off-axis: the ray misses its
    # world AABB centre but a naive world-space box would still hit.
    off = Transform(position=(0.8, 0.0, 4.0), rotation=(0, 0, 45)).matrix

    candidates = [
        PickCandidate("far", far, *unit),
        PickCandidate("near", near, *unit),
        PickCandidate("off", off, *unit),
    ]

    assert pick(ray, candidates) == "near"
    assert pick(ray, [candidates[0]]) == "far"
    assert pick(ray, [candidates[2]]) is None


def test_pick_skips_zero_scale():

    view = look_at((0.0, 0.0, 10.0), (0.0, 0.0, 0.0), (0.0, 1.0, 0.0))
    projection = perspective(60.0, 1.0, 0.1, 100.0)
    ray = screen_ray(400, 400, 800, 800, view, projection)

    flat = Transform(scale=(0.0, 1.0, 1.0)).matrix

    assert pick(ray, [PickCandidate("flat", flat, np.full(3, -0.5), np.full(3, 0.5))]) is None
