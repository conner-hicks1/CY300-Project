from typing import (
    TypeVar,
    cast
)

from core.assertions import engine_assert
from core.logger import Logger

from ecs.entity import Entity


T = TypeVar("T")


class _EntitySlot:

    __slots__ = (
        "generation",
        "alive"
    )

    def __init__(self):

        self.generation = 0
        self.alive = False


class Registry:

    # =====================================================
    # Construction
    # =====================================================

    def __init__(self):

        self._entities: list[
            _EntitySlot
        ] = []

        self._free_indices: list[
            int
        ] = []

        # component type
        #       ↓
        # { entity index : component }

        self._components: dict[
            type,
            dict[int, object]
        ] = {}

        self._entity_count = 0

        Logger.info(
            "[Registry] Initialized."
        )

    # =====================================================
    # Entity Creation
    # =====================================================

    def create(self) -> Entity:

        if self._free_indices:

            index = (
                self._free_indices.pop()
            )

            slot = self._entities[
                index
            ]

        else:

            index = len(
                self._entities
            )

            slot = _EntitySlot()

            self._entities.append(
                slot
            )

        engine_assert(
            not slot.alive,
            (
                "Registry attempted to reuse "
                "an entity slot that is alive."
            )
        )

        slot.alive = True

        self._entity_count += 1

        entity = Entity(
            index=index,
            generation=slot.generation
        )

        Logger.debug(
            "[Registry] Created %s.",
            entity
        )

        return entity

    # =====================================================
    # Entity Validation
    # =====================================================

    def is_alive(
        self,
        entity: Entity
    ) -> bool:

        if not isinstance(
            entity,
            Entity
        ):

            return False

        if not entity.is_valid:

            return False

        if not (
            0
            <= entity.index
            < len(self._entities)
        ):

            return False

        slot = self._entities[
            entity.index
        ]

        return (
            slot.alive
            and
            slot.generation
            == entity.generation
        )

    # =====================================================
    # Entity Destruction
    # =====================================================

    def destroy(
        self,
        entity: Entity
    ):

        self._assert_alive(
            entity
        )

        # Remove every component belonging to entity.

        for storage in (
            self._components.values()
        ):

            storage.pop(
                entity.index,
                None
            )

        slot = self._entities[
            entity.index
        ]

        slot.alive = False

        # Invalidate all old references to this entity.

        slot.generation += 1

        self._free_indices.append(
            entity.index
        )

        self._entity_count -= 1

        Logger.debug(
            "[Registry] Destroyed %s.",
            entity
        )

    # =====================================================
    # Add Component
    # =====================================================

    def add(
        self,
        entity: Entity,
        component: T
    ) -> T:

        self._assert_alive(
            entity
        )

        engine_assert(
            component is not None,
            "Cannot add a None component."
        )

        component_type = type(
            component
        )

        storage = (
            self._components.setdefault(
                component_type,
                {}
            )
        )

        engine_assert(
            entity.index not in storage,
            (
                f"{entity} already has "
                f"{component_type.__name__}."
            )
        )

        storage[
            entity.index
        ] = component

        Logger.debug(
            "[Registry] Added %s to %s.",
            component_type.__name__,
            entity
        )

        return component

    # =====================================================
    # Set / Replace Component
    # =====================================================

    def set(
        self,
        entity: Entity,
        component: T
    ) -> T:

        self._assert_alive(
            entity
        )

        engine_assert(
            component is not None,
            "Cannot set a None component."
        )

        component_type = type(
            component
        )

        storage = (
            self._components.setdefault(
                component_type,
                {}
            )
        )

        storage[
            entity.index
        ] = component

        return component

    # =====================================================
    # Has Component
    # =====================================================

    def has(
        self,
        entity: Entity,
        component_type: type[T]
    ) -> bool:

        if not self.is_alive(
            entity
        ):

            return False

        storage = self._components.get(
            component_type
        )

        if storage is None:

            return False

        return (
            entity.index
            in storage
        )

    # =====================================================
    # Get Component
    # =====================================================

    def get(
        self,
        entity: Entity,
        component_type: type[T]
    ) -> T:

        self._assert_alive(
            entity
        )

        storage = self._components.get(
            component_type
        )

        engine_assert(
            storage is not None,
            (
                f"No {component_type.__name__} "
                f"storage exists."
            )
        )

        component = storage.get(
            entity.index
        )

        engine_assert(
            component is not None,
            (
                f"{entity} does not have "
                f"{component_type.__name__}."
            )
        )

        return cast(
            T,
            component
        )

    # =====================================================
    # Try Get Component
    # =====================================================

    def try_get(
        self,
        entity: Entity,
        component_type: type[T]
    ) -> T | None:

        if not self.is_alive(
            entity
        ):

            return None

        storage = self._components.get(
            component_type
        )

        if storage is None:

            return None

        component = storage.get(
            entity.index
        )

        if component is None:

            return None

        return cast(
            T,
            component
        )

    # =====================================================
    # Remove Component
    # =====================================================

    def remove(
        self,
        entity: Entity,
        component_type: type[T]
    ) -> T:

        self._assert_alive(
            entity
        )

        storage = self._components.get(
            component_type
        )

        engine_assert(
            storage is not None,
            (
                f"No {component_type.__name__} "
                f"storage exists."
            )
        )

        engine_assert(
            entity.index in storage,
            (
                f"{entity} does not have "
                f"{component_type.__name__}."
            )
        )

        component = storage.pop(
            entity.index
        )

        Logger.debug(
            "[Registry] Removed %s from %s.",
            component_type.__name__,
            entity
        )

        return cast(
            T,
            component
        )

    # =====================================================
    # Query
    # =====================================================

    def view(
        self,
        *component_types: type
    ) -> list[Entity]:
        """
        Entities that have every listed component.

        Returns a snapshot list rather than a live
        generator, so callers may add/remove components or
        destroy entities while iterating without hitting
        "dictionary changed size during iteration".
        Entities destroyed mid-iteration are still in the
        snapshot; check is_alive() if that matters.
        """

        engine_assert(
            len(component_types) > 0,
            (
                "Registry.view() requires "
                "at least one component type."
            )
        )

        storages: list[
            dict[int, object]
        ] = []

        for component_type in component_types:

            storage = self._components.get(
                component_type
            )

            if storage is None:
                return []

            storages.append(
                storage
            )

        # Start with the smallest component pool.
        #
        # This reduces unnecessary membership checks.

        smallest = min(
            storages,
            key=len
        )

        result: list[Entity] = []

        for index in smallest:

            if not all(
                index in storage
                for storage in storages
            ):

                continue

            slot = self._entities[
                index
            ]

            if not slot.alive:
                continue

            result.append(
                Entity(
                    index=index,
                    generation=slot.generation
                )
            )

        return result

    # =====================================================
    # Entity Iteration
    # =====================================================

    def entities(
        self
    ) -> list[Entity]:

        # Snapshot for the same reason as view().

        return [
            Entity(
                index=index,
                generation=slot.generation
            )
            for index, slot in enumerate(
                self._entities
            )
            if slot.alive
        ]

    # =====================================================
    # Clear
    # =====================================================

    def clear(self):

        entities = list(
            self.entities()
        )

        for entity in entities:

            self.destroy(
                entity
            )

        self._components.clear()

        Logger.info(
            "[Registry] Cleared."
        )

    # =====================================================
    # Internal Validation
    # =====================================================

    def _assert_alive(
        self,
        entity: Entity
    ):

        engine_assert(
            self.is_alive(entity),
            (
                "Registry received invalid "
                f"or stale entity: {entity}"
            )
        )

    # =====================================================
    # Information
    # =====================================================

    @property
    def entity_count(
        self
    ) -> int:

        return self._entity_count

    def component_count(
        self,
        component_type: type
    ) -> int:

        storage = self._components.get(
            component_type
        )

        if storage is None:
            return 0

        return len(
            storage
        )

    def __len__(self):

        return self._entity_count