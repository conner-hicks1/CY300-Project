import numpy as np

from core.assertions import engine_assert

from ecs.components import (
    CameraComponent,
    DirectionalLightComponent,
    MeshRendererComponent,
    PointLightComponent,
    SpotLightComponent,
    TransformComponent
)
from ecs.registry import Registry

from graphics.lighting import (
    MAX_POINT_LIGHTS,
    MAX_SPOT_LIGHTS,
    DirectionalLight,
    LightEnvironment,
    PointLight,
    SpotLight
)
from graphics.renderer import Renderer

from math3d.camera import Camera

from resources.resources import Resources

from scene.scene import Scene


class RenderSystem:

    # =====================================================
    # Construction
    # =====================================================

    def __init__(
        self,
        renderer: Renderer,
        resources: Resources
    ):

        engine_assert(
            renderer is not None,
            "RenderSystem requires a Renderer."
        )

        engine_assert(
            resources is not None,
            "RenderSystem requires Resources."
        )

        self._renderer = renderer
        self._resources = resources

    # =====================================================
    # Render
    # =====================================================

    def render(
        self,
        scene: Scene,
        aspect_ratio: float
    ):

        engine_assert(
            scene is not None,
            "RenderSystem requires a Scene."
        )

        engine_assert(
            not scene.is_shutdown,
            "RenderSystem cannot render a shutdown Scene."
        )

        engine_assert(
            aspect_ratio > 0.0,
            "RenderSystem aspect ratio must be positive."
        )

        registry = scene.registry

        # -------------------------------------------------
        # Clear
        # -------------------------------------------------

        self._renderer.clear(
            (0.0, 0.0, 0.0, 1.0)
        )

        # -------------------------------------------------
        # Primary Camera
        # -------------------------------------------------

        camera = self._build_primary_camera(
            registry,
            aspect_ratio
        )

        # -------------------------------------------------
        # Lights
        # -------------------------------------------------

        lighting = self._build_light_environment(
            registry
        )

        # -------------------------------------------------
        # Renderable Entities
        # -------------------------------------------------

        for entity in registry.view(
            TransformComponent,
            MeshRendererComponent
        ):

            transform_component = registry.get(
                entity,
                TransformComponent
            )

            mesh_renderer = registry.get(
                entity,
                MeshRendererComponent
            )

            mesh = (
                self._resources.meshes.get(
                    mesh_renderer.mesh
                )
            )

            material = (
                self._resources.materials.get(
                    mesh_renderer.material
                )
            )

            self._renderer.draw(
                mesh,
                material,
                self._resources,
                transform_component.transform,
                camera,
                lighting
            )

    # =====================================================
    # Primary Camera
    # =====================================================

    def _build_primary_camera(
        self,
        registry: Registry,
        aspect_ratio: float
    ) -> Camera:

        primary_transform = None
        primary_camera = None

        for entity in registry.view(
            TransformComponent,
            CameraComponent
        ):

            camera_component = registry.get(
                entity,
                CameraComponent
            )

            if not camera_component.primary:
                continue

            engine_assert(
                primary_camera is None,
                "Scene contains more than one primary camera."
            )

            primary_transform = registry.get(
                entity,
                TransformComponent
            )

            primary_camera = camera_component

        engine_assert(
            primary_transform is not None,
            "Scene has no primary camera TransformComponent."
        )

        engine_assert(
            primary_camera is not None,
            "Scene has no primary CameraComponent."
        )

        transform = (
            primary_transform.transform
        )

        position = transform.position

        target = (
            position
            + transform.forward
        )

        return Camera(
            position=position,
            target=target,
            up=transform.up,
            fov=primary_camera.fov,
            aspect_ratio=aspect_ratio,
            near=primary_camera.near,
            far=primary_camera.far
        )

    # =====================================================
    # Light Environment
    # =====================================================

    def _build_light_environment(
        self,
        registry: Registry
    ) -> LightEnvironment:

        lighting = LightEnvironment()

        # -------------------------------------------------
        # Directional (optional, at most one)
        # -------------------------------------------------
        #
        # Scene ambient comes from the directional light.
        # With no directional light, ambient is zero.

        for entity in registry.view(
            TransformComponent,
            DirectionalLightComponent
        ):

            engine_assert(
                lighting.directional is None,
                "Scene contains more than one directional light."
            )

            transform = registry.get(
                entity,
                TransformComponent
            ).transform

            component = registry.get(
                entity,
                DirectionalLightComponent
            )

            lighting.directional = DirectionalLight(
                direction=transform.forward,
                color=component.color,
                intensity=component.intensity
            )

            lighting.ambient = component.ambient

        # -------------------------------------------------
        # Point Lights
        # -------------------------------------------------

        for entity in registry.view(
            TransformComponent,
            PointLightComponent
        ):

            transform = registry.get(
                entity,
                TransformComponent
            ).transform

            component = registry.get(
                entity,
                PointLightComponent
            )

            engine_assert(
                component.range > 0.0,
                "PointLightComponent range must be positive."
            )

            lighting.point_lights.append(
                PointLight(
                    position=transform.position,
                    color=component.color,
                    intensity=component.intensity,
                    range=component.range
                )
            )

        engine_assert(
            len(lighting.point_lights) <= MAX_POINT_LIGHTS,
            (
                "Scene exceeds the maximum of "
                f"{MAX_POINT_LIGHTS} point lights."
            )
        )

        # -------------------------------------------------
        # Spot Lights
        # -------------------------------------------------

        for entity in registry.view(
            TransformComponent,
            SpotLightComponent
        ):

            transform = registry.get(
                entity,
                TransformComponent
            ).transform

            component = registry.get(
                entity,
                SpotLightComponent
            )

            engine_assert(
                component.range > 0.0,
                "SpotLightComponent range must be positive."
            )

            engine_assert(
                0.0
                <= component.inner_angle
                <= component.outer_angle
                < 90.0,
                (
                    "SpotLightComponent angles must satisfy "
                    "0 <= inner_angle <= outer_angle < 90."
                )
            )

            lighting.spot_lights.append(
                SpotLight(
                    position=transform.position,
                    direction=transform.forward,
                    color=component.color,
                    intensity=component.intensity,
                    range=component.range,
                    inner_cutoff=float(
                        np.cos(
                            np.radians(
                                component.inner_angle
                            )
                        )
                    ),
                    outer_cutoff=float(
                        np.cos(
                            np.radians(
                                component.outer_angle
                            )
                        )
                    )
                )
            )

        engine_assert(
            len(lighting.spot_lights) <= MAX_SPOT_LIGHTS,
            (
                "Scene exceeds the maximum of "
                f"{MAX_SPOT_LIGHTS} spot lights."
            )
        )

        return lighting