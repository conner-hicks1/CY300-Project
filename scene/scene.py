from core.assertions import engine_assert
from core.logger import Logger

from ecs.entity import Entity
from ecs.registry import Registry


class Scene:

    # =====================================================
    # Construction
    # =====================================================

    def __init__(
        self,
        name: str = "Untitled Scene"
    ):

        engine_assert(
            isinstance(name, str),
            "Scene name must be a string."
        )

        engine_assert(
            bool(name.strip()),
            "Scene name cannot be empty."
        )

        self._name = name.strip()

        self._registry = Registry()

        self._shutdown = False

        Logger.info(
            "[Scene] Created '%s'.",
            self._name
        )

    # =====================================================
    # Entity Lifecycle
    # =====================================================

    def create_entity(
        self
    ) -> Entity:

        self._assert_active()

        return self._registry.create()

    def destroy_entity(
        self,
        entity: Entity
    ):

        self._assert_active()

        self._registry.destroy(
            entity
        )

    def is_alive(
        self,
        entity: Entity
    ) -> bool:

        if self._shutdown:
            return False

        return self._registry.is_alive(
            entity
        )

    # =====================================================
    # Components
    # =====================================================

    def add_component(
        self,
        entity: Entity,
        component
    ):

        self._assert_active()

        return self._registry.add(
            entity,
            component
        )

    def set_component(
        self,
        entity: Entity,
        component
    ):

        self._assert_active()

        return self._registry.set(
            entity,
            component
        )

    def get_component(
        self,
        entity: Entity,
        component_type
    ):

        self._assert_active()

        return self._registry.get(
            entity,
            component_type
        )

    def try_get_component(
        self,
        entity: Entity,
        component_type
    ):

        if self._shutdown:
            return None

        return self._registry.try_get(
            entity,
            component_type
        )

    def has_component(
        self,
        entity: Entity,
        component_type
    ) -> bool:

        if self._shutdown:
            return False

        return self._registry.has(
            entity,
            component_type
        )

    def remove_component(
        self,
        entity: Entity,
        component_type
    ):

        self._assert_active()

        return self._registry.remove(
            entity,
            component_type
        )

    # =====================================================
    # Queries
    # =====================================================

    def view(
        self,
        *component_types
    ):

        self._assert_active()

        return self._registry.view(
            *component_types
        )

    def entities(
        self
    ):

        self._assert_active()

        return self._registry.entities()

    # =====================================================
    # Scene Lifecycle
    # =====================================================

    def clear(self):

        self._assert_active()

        Logger.debug(
            "[Scene] Clearing '%s'.",
            self._name
        )

        self._registry.clear()

    def shutdown(self):

        if self._shutdown:
            return

        Logger.info(
            "[Scene] Shutting down '%s'.",
            self._name
        )

        self._registry.clear()

        self._shutdown = True

        Logger.info(
            "[Scene] Shutdown complete '%s'.",
            self._name
        )

    # =====================================================
    # Internal
    # =====================================================

    def _assert_active(self):

        engine_assert(
            not self._shutdown,
            f"Scene '{self._name}' has been shut down."
        )

    # =====================================================
    # Properties
    # =====================================================

    @property
    def name(
        self
    ) -> str:

        return self._name

    @name.setter
    def name(
        self,
        value: str
    ):

        engine_assert(
            isinstance(value, str) and bool(value.strip()),
            "Scene name must be a non-empty string."
        )

        self._name = value.strip()

    @property
    def registry(
        self
    ) -> Registry:

        self._assert_active()

        return self._registry

    @property
    def entity_count(
        self
    ) -> int:

        if self._shutdown:
            return 0

        return self._registry.entity_count

    @property
    def is_shutdown(
        self
    ) -> bool:

        return self._shutdown

    # =====================================================
    # Representation
    # =====================================================

    def __repr__(self):

        return (
            f"Scene("
            f"name='{self._name}', "
            f"entities={self.entity_count}"
            f")"
        )