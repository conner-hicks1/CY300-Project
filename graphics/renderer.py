from core.assertions import engine_assert

from graphics.lighting import LightEnvironment
from graphics.material import Material
from graphics.mesh import Mesh
from graphics.render_command import RenderCommand
from graphics.render_state import RenderState

from math3d.camera import Camera
from math3d.transform import Transform

from resources.resources import Resources


class Renderer:

    # =====================================================
    # Construction
    # =====================================================

    def __init__(self):

        RenderState.initialize()

    # =====================================================
    # Clear
    # =====================================================

    def clear(
        self,
        color=(0.0, 0.0, 0.0, 1.0)
    ):

        RenderCommand.set_clear_color(
            color
        )

        RenderCommand.clear()

    # =====================================================
    # Draw
    # =====================================================

    def draw(
        self,
        mesh: Mesh,
        material: Material,
        resources: Resources,
        transform: Transform,
        camera: Camera,
        lighting: LightEnvironment
    ):

        engine_assert(
            mesh is not None,
            "Renderer received a None Mesh."
        )

        engine_assert(
            material is not None,
            "Renderer received a None Material."
        )

        engine_assert(
            resources is not None,
            "Renderer received None Resources."
        )

        engine_assert(
            transform is not None,
            "Renderer received a None Transform."
        )

        engine_assert(
            camera is not None,
            "Renderer received a None Camera."
        )

        engine_assert(
            lighting is not None,
            "Renderer received a None LightEnvironment."
        )

        # -------------------------------------------------
        # Shader
        # -------------------------------------------------

        shader = resources.shaders.get(
            material.shader
        )

        shader.bind()

        # -------------------------------------------------
        # Transform / Camera Uniforms
        # -------------------------------------------------

        shader.set_mat4(
            "uModel",
            transform.matrix
        )

        shader.set_mat4(
            "uView",
            camera.view_matrix
        )

        shader.set_mat4(
            "uProjection",
            camera.projection_matrix
        )

        shader.set_mat3(
            "uNormalMatrix",
            transform.normal_matrix
        )

        shader.set_vec3(
            "uViewPosition",
            camera.position
        )

        # -------------------------------------------------
        # Lighting Uniforms
        # -------------------------------------------------

        self._apply_lighting(
            shader,
            lighting
        )

        # -------------------------------------------------
        # Material Values
        # -------------------------------------------------

        self._apply_material_values(
            shader,
            material
        )

        # -------------------------------------------------
        # Material Textures
        # -------------------------------------------------

        self._apply_material_textures(
            shader,
            material,
            resources
        )

        # -------------------------------------------------
        # Geometry
        # -------------------------------------------------

        mesh.vertex_array.bind()

        # -------------------------------------------------
        # Draw
        # -------------------------------------------------

        if mesh.index_buffer is not None:

            RenderCommand.draw_indexed(
                mesh.index_buffer.count
            )

        else:

            RenderCommand.draw_arrays(
                mesh.vertex_count
            )

    # =====================================================
    # Lighting
    # =====================================================

    @staticmethod
    def _apply_lighting(
        shader,
        lighting: LightEnvironment
    ):

        shader.set_float(
            "uAmbientStrength",
            lighting.ambient
        )

        # -------------------------------------------------
        # Directional
        # -------------------------------------------------

        directional = lighting.directional

        shader.set_bool(
            "uHasDirectionalLight",
            directional is not None
        )

        if directional is not None:

            shader.set_vec3(
                "uDirectionalLight.direction",
                directional.direction
            )

            shader.set_vec3(
                "uDirectionalLight.color",
                directional.color
            )

            shader.set_float(
                "uDirectionalLight.intensity",
                directional.intensity
            )

        # -------------------------------------------------
        # Point Lights
        # -------------------------------------------------

        shader.set_int(
            "uPointLightCount",
            len(lighting.point_lights)
        )

        for index, point in enumerate(
            lighting.point_lights
        ):

            prefix = f"uPointLights[{index}]"

            shader.set_vec3(
                f"{prefix}.position",
                point.position
            )

            shader.set_vec3(
                f"{prefix}.color",
                point.color
            )

            shader.set_float(
                f"{prefix}.intensity",
                point.intensity
            )

            shader.set_float(
                f"{prefix}.range",
                point.range
            )

        # -------------------------------------------------
        # Spot Lights
        # -------------------------------------------------

        shader.set_int(
            "uSpotLightCount",
            len(lighting.spot_lights)
        )

        for index, spot in enumerate(
            lighting.spot_lights
        ):

            prefix = f"uSpotLights[{index}]"

            shader.set_vec3(
                f"{prefix}.position",
                spot.position
            )

            shader.set_vec3(
                f"{prefix}.direction",
                spot.direction
            )

            shader.set_vec3(
                f"{prefix}.color",
                spot.color
            )

            shader.set_float(
                f"{prefix}.intensity",
                spot.intensity
            )

            shader.set_float(
                f"{prefix}.range",
                spot.range
            )

            shader.set_float(
                f"{prefix}.innerCutoff",
                spot.inner_cutoff
            )

            shader.set_float(
                f"{prefix}.outerCutoff",
                spot.outer_cutoff
            )

    # =====================================================
    # Material Values
    # =====================================================

    @staticmethod
    def _apply_material_values(
        shader,
        material: Material
    ):

        for name, value in material.values:

            # bool must be checked before int because:
            #
            # isinstance(True, int) == True

            if isinstance(
                value,
                bool
            ):

                shader.set_bool(
                    name,
                    value
                )

            elif isinstance(
                value,
                int
            ):

                shader.set_int(
                    name,
                    value
                )

            elif isinstance(
                value,
                float
            ):

                shader.set_float(
                    name,
                    value
                )

            elif isinstance(
                value,
                tuple
            ):

                if len(value) == 2:

                    shader.set_vec2(
                        name,
                        value
                    )

                elif len(value) == 3:

                    shader.set_vec3(
                        name,
                        value
                    )

                elif len(value) == 4:

                    shader.set_vec4(
                        name,
                        value
                    )

                else:

                    engine_assert(
                        False,
                        (
                            "Unsupported material vector "
                            f"length for '{name}': "
                            f"{len(value)}"
                        )
                    )

            else:

                engine_assert(
                    False,
                    (
                        "Unsupported material uniform "
                        f"type for '{name}': "
                        f"{type(value).__name__}"
                    )
                )

    # =====================================================
    # Material Textures
    # =====================================================

    @staticmethod
    def _apply_material_textures(
        shader,
        material: Material,
        resources: Resources
    ):

        for slot, (
            sampler_name,
            texture_handle
        ) in enumerate(
            material.textures
        ):

            texture = (
                resources.textures.get(
                    texture_handle
                )
            )

            texture.bind(
                slot
            )

            shader.set_int(
                sampler_name,
                slot
            )

    # =====================================================
    # Shutdown
    # =====================================================

    def shutdown(self):

        RenderState.shutdown()