from core.assertions import engine_assert

from ecs.components import (
    RotatorComponent,
    TransformComponent
)

from scene.scene import Scene


class RotatorSystem:

    # =====================================================
    # Fixed Update
    # =====================================================

    def fixed_update(
        self,
        scene: Scene,
        fixed_delta_time: float
    ):

        engine_assert(
            scene is not None,
            "RotatorSystem requires a Scene."
        )

        engine_assert(
            fixed_delta_time > 0.0,
            "RotatorSystem fixed delta time must be positive."
        )

        for entity in scene.view(
            TransformComponent,
            RotatorComponent
        ):

            transform = scene.get_component(
                entity,
                TransformComponent
            ).transform

            rotator = scene.get_component(
                entity,
                RotatorComponent
            )

            transform.rotate(
                tuple(
                    speed * fixed_delta_time
                    for speed in rotator.degrees_per_second
                )
            )
