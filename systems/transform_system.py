import numpy as np

from core.assertions import engine_assert

from ecs.components import (
    HierarchyComponent,
    TransformComponent
)
from ecs.entity import Entity
from ecs.registry import Registry

from scene.scene import Scene


class TransformSystem:

    # =====================================================
    # Update
    # =====================================================
    #
    # Computes TransformComponent.world_matrix for every
    # entity:
    #
    #     world = parent.world @ local
    #
    # Parents are resolved on demand, so entity creation
    # order does not matter. Run this after anything that
    # moves transforms and before anything that reads
    # world-space data (rendering, lights, cameras).

    def update(
        self,
        scene: Scene
    ):

        engine_assert(
            scene is not None,
            "TransformSystem requires a Scene."
        )

        registry = scene.registry

        resolved: dict[int, np.ndarray] = {}

        for entity in registry.view(
            TransformComponent
        ):

            self._resolve(
                registry,
                entity,
                resolved,
                visiting=set()
            )

    # =====================================================
    # Resolve
    # =====================================================

    def _resolve(
        self,
        registry: Registry,
        entity: Entity,
        resolved: dict[int, np.ndarray],
        visiting: set[int]
    ) -> np.ndarray:

        cached = resolved.get(
            entity.index
        )

        if cached is not None:
            return cached

        engine_assert(
            entity.index not in visiting,
            (
                "Transform hierarchy contains a cycle "
                f"involving {entity}."
            )
        )

        visiting.add(
            entity.index
        )

        transform_component = registry.get(
            entity,
            TransformComponent
        )

        local = transform_component.transform.matrix

        hierarchy = registry.try_get(
            entity,
            HierarchyComponent
        )

        if hierarchy is None:

            world = np.array(
                local,
                dtype=np.float64
            )

        else:

            parent = hierarchy.parent

            engine_assert(
                registry.is_alive(parent),
                (
                    f"{entity} has a HierarchyComponent "
                    "whose parent is destroyed or invalid."
                )
            )

            engine_assert(
                registry.has(parent, TransformComponent),
                (
                    f"{entity}'s parent {parent} has no "
                    "TransformComponent."
                )
            )

            parent_world = self._resolve(
                registry,
                parent,
                resolved,
                visiting
            )

            world = (
                parent_world
                @ local
            ).astype(
                np.float64
            )

        visiting.discard(
            entity.index
        )

        transform_component.world_matrix = world

        resolved[
            entity.index
        ] = world

        return world
