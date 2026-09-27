from dataclasses import dataclass

import pytest

from ecs.registry import Registry


@dataclass
class Position:
    x: float = 0.0


@dataclass
class Velocity:
    dx: float = 0.0


def test_view_returns_only_matching_entities():

    registry = Registry()

    a = registry.create()
    b = registry.create()

    registry.add(a, Position())
    registry.add(a, Velocity())
    registry.add(b, Position())

    assert registry.view(Position, Velocity) == [a]
    assert set(registry.view(Position)) == {a, b}


def test_view_of_unknown_component_is_empty():

    assert Registry().view(Position) == []


def test_destroy_during_view_is_safe():

    registry = Registry()

    entities = [registry.create() for _ in range(5)]

    for entity in entities:
        registry.add(entity, Position())

    # Used to raise "dictionary changed size during iteration".
    for entity in registry.view(Position):
        registry.destroy(entity)

    assert registry.entity_count == 0


def test_add_component_during_view_is_safe():

    registry = Registry()

    for _ in range(3):
        registry.add(registry.create(), Position())

    for entity in registry.view(Position):
        registry.add(entity, Velocity())
        registry.add(registry.create(), Position())

    assert len(registry.view(Position, Velocity)) == 3


def test_view_with_returns_components_in_requested_order():

    registry = Registry()

    a = registry.create()
    b = registry.create()

    position = registry.add(a, Position(1.0))
    velocity = registry.add(a, Velocity(2.0))
    registry.add(b, Position(3.0))

    assert registry.view_with(Velocity, Position) == [(a, velocity, position)]
    assert registry.view_with(Velocity, Position)[0][1] is velocity
    assert Registry().view_with(Position) == []


def test_view_with_skips_destroyed_entities():

    registry = Registry()

    a = registry.create()
    b = registry.create()

    registry.add(a, Position())
    registry.add(b, Position())

    registry.destroy(a)

    assert [item[0] for item in registry.view_with(Position)] == [b]


def test_get_rejects_invalid_input_with_engine_error():

    from core.assertions import EngineAssertionError
    from ecs.entity import Entity

    registry = Registry()

    entity = registry.create()

    for bad in (None, Entity.invalid(), Entity(index=99, generation=0)):

        with pytest.raises(EngineAssertionError):
            registry.get(bad, Position)

    with pytest.raises(EngineAssertionError, match="does not have Position"):
        registry.get(entity, Position)


def test_stale_entity_is_rejected_after_slot_reuse():

    registry = Registry()

    old = registry.create()
    registry.destroy(old)

    new = registry.create()

    assert new.index == old.index
    assert not registry.is_alive(old)

    with pytest.raises(Exception):
        registry.add(old, Position())
