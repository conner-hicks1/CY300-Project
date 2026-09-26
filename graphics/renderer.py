from dataclasses import dataclass

import numpy as np

from core.assertions import engine_assert

from graphics.buffer import UniformBuffer
from graphics.lighting import LightEnvironment
from graphics.material import Material, MaterialValue
from graphics.mesh import Mesh
from graphics.render_command import RenderCommand
from graphics.render_state import RenderState
from graphics.shader import Shader
from graphics.texture import Texture2D
from graphics.uniform_blocks import (
    CAMERA_BLOCK,
    LIGHTS_BLOCK,
    ShadowParameters,
    pack_camera_block,
    pack_lights_block
)
from graphics.vertex_array import VertexArray

from math3d.camera import Camera
from math3d.matrices import normal_matrix

from resources.resources import Resources


# =========================================================
# Frame Statistics
# =========================================================

@dataclass(slots=True)
class RenderStats:

    draw_calls: int = 0
    triangles: int = 0

    def reset(self):

        self.draw_calls = 0
        self.triangles = 0


class Renderer:

    # =====================================================
    # Texture Units
    # =====================================================
    #
    # Material textures use units 0 .. SHADOW_MAP_SLOT - 1.
    # The shadow map stays bound on its own unit for the
    # whole frame.

    SHADOW_MAP_SLOT = 7

    # =====================================================
    # Material Defaults
    # =====================================================
    #
    # Uniform values persist per shader program, so a value
    # set by one material would otherwise leak into the next
    # draw that uses the same shader but does not set it.
    # Every draw applies these first, then the material's
    # own values.

    DEFAULT_MATERIAL_VALUES: dict[str, MaterialValue] = {
        "uBaseColor": (1.0, 1.0, 1.0),
        "uSpecularStrength": 0.5,
        "uShininess": 32.0,
        "uUVScale": (1.0, 1.0),
        "uNormalStrength": 1.0,
    }

    # =====================================================
    # Construction
    # =====================================================

    def __init__(self):

        RenderState.initialize()

        self._camera_buffer = UniformBuffer(
            CAMERA_BLOCK.size,
            CAMERA_BLOCK.binding
        )

        self._lights_buffer = UniformBuffer(
            LIGHTS_BLOCK.size,
            LIGHTS_BLOCK.binding
        )

        # Core profile requires a bound VAO even for a
        # draw with no vertex attributes (fullscreen
        # triangle generated from gl_VertexID).

        self._empty_vertex_array = VertexArray()

        # sampler name -> texture used when a material
        # does not provide one (white albedo, flat normal).

        self._default_textures: dict[str, Texture2D] = {}

        self._shadow_map_texture_id = 0

        self._in_scene = False

        self.stats = RenderStats()

    # =====================================================
    # Defaults
    # =====================================================

    def set_default_texture(
        self,
        sampler_name: str,
        texture: Texture2D
    ):

        engine_assert(
            texture is not None and texture.is_valid,
            "Default texture must be a valid Texture2D."
        )

        self._default_textures[
            sampler_name
        ] = texture

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
    # Scene
    # =====================================================

    def begin_scene(
        self,
        camera: Camera,
        lighting: LightEnvironment,
        shadows: ShadowParameters | None = None
    ):
        """
        Upload per-frame data (camera + lights) once.
        Every shader that declares CameraBlock/LightsBlock
        reads it without per-draw uniform calls.
        """

        engine_assert(
            not self._in_scene,
            "Renderer.begin_scene() called twice without end_scene()."
        )

        engine_assert(
            camera is not None,
            "Renderer.begin_scene() requires a Camera."
        )

        engine_assert(
            lighting is not None,
            "Renderer.begin_scene() requires a LightEnvironment."
        )

        self._camera_buffer.set_data(
            pack_camera_block(
                camera.view_matrix,
                camera.projection_matrix,
                camera.position
            )
        )

        self._lights_buffer.set_data(
            pack_lights_block(
                lighting,
                shadows
            )
        )

        self._shadow_map_texture_id = 0

        self.stats.reset()

        self._in_scene = True

    def set_shadow_map(
        self,
        texture_id: int
    ):
        """
        Bind the shadow map for the rest of the frame.
        Call after the shadow pass has finished writing it.
        """

        self._assert_in_scene()

        self._shadow_map_texture_id = texture_id

        RenderCommand.bind_texture(
            texture_id,
            self.SHADOW_MAP_SLOT
        )

    def end_scene(self):

        self._assert_in_scene()

        # Unbind the shadow map so it is never bound for
        # sampling while next frame's shadow pass renders
        # into it.

        if self._shadow_map_texture_id:

            RenderCommand.bind_texture(
                0,
                self.SHADOW_MAP_SLOT
            )

            self._shadow_map_texture_id = 0

        self._in_scene = False

    # =====================================================
    # Lit Draw
    # =====================================================

    def draw(
        self,
        mesh: Mesh,
        material: Material,
        resources: Resources,
        model_matrix: np.ndarray
    ):

        self._assert_in_scene()

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

        # -------------------------------------------------
        # Shader
        # -------------------------------------------------

        shader = resources.shaders.get(
            material.shader
        )

        shader.bind()

        # -------------------------------------------------
        # Per-Object Uniforms
        # -------------------------------------------------

        shader.set_mat4(
            "uModel",
            model_matrix
        )

        if shader.has_uniform("uNormalMatrix"):

            shader.set_mat3(
                "uNormalMatrix",
                normal_matrix(model_matrix)
            )

        if shader.has_uniform("uShadowMap"):

            shader.set_int(
                "uShadowMap",
                self.SHADOW_MAP_SLOT
            )

        # -------------------------------------------------
        # Material
        # -------------------------------------------------

        self._apply_material_values(
            shader,
            material
        )

        self._apply_material_textures(
            shader,
            material,
            resources
        )

        # -------------------------------------------------
        # Geometry
        # -------------------------------------------------

        self._submit(
            mesh
        )

    # =====================================================
    # Depth-Only Draw (Shadow Pass)
    # =====================================================

    def draw_depth(
        self,
        mesh: Mesh,
        shader: Shader,
        model_matrix: np.ndarray
    ):
        """
        Draw with an already-bound depth shader. The light
        matrix comes from LightsBlock.
        """

        self._assert_in_scene()

        shader.set_mat4(
            "uModel",
            model_matrix
        )

        self._submit(
            mesh
        )

    # =====================================================
    # Unlit Draw (Gizmos)
    # =====================================================

    def draw_unlit(
        self,
        mesh: Mesh,
        shader: Shader,
        model_matrix: np.ndarray,
        color
    ):

        self._assert_in_scene()

        shader.bind()

        shader.set_mat4(
            "uModel",
            model_matrix
        )

        shader.set_vec3(
            "uColor",
            color
        )

        self._submit(
            mesh
        )

    # =====================================================
    # Fullscreen Draw (Post-Processing)
    # =====================================================

    def draw_fullscreen(
        self,
        shader: Shader
    ):
        """
        Draw one triangle covering the viewport. The vertex
        shader derives positions from gl_VertexID.
        """

        shader.bind()

        self._empty_vertex_array.bind()

        RenderCommand.draw_arrays(
            3
        )

        self.stats.draw_calls += 1

    # =====================================================
    # Submission
    # =====================================================

    def _submit(
        self,
        mesh: Mesh
    ):

        mesh.vertex_array.bind()

        if mesh.index_buffer is not None:

            RenderCommand.draw_indexed(
                mesh.index_buffer.count
            )

            self.stats.triangles += (
                mesh.index_buffer.count // 3
            )

        else:

            RenderCommand.draw_arrays(
                mesh.vertex_count
            )

            self.stats.triangles += (
                mesh.vertex_count // 3
            )

        self.stats.draw_calls += 1

    def _assert_in_scene(self):

        engine_assert(
            self._in_scene,
            "Renderer draw calls must happen between "
            "begin_scene() and end_scene()."
        )

    # =====================================================
    # Material Values
    # =====================================================

    def _apply_material_values(
        self,
        shader: Shader,
        material: Material
    ):

        material_values = dict(
            material.values
        )

        # Defaults only for uniforms this shader has and
        # the material did not set.

        for name, value in self.DEFAULT_MATERIAL_VALUES.items():

            if (
                name not in material_values
                and shader.has_uniform(name)
            ):

                self._set_uniform(
                    shader,
                    name,
                    value
                )

        for name, value in material_values.items():

            self._set_uniform(
                shader,
                name,
                value
            )

    @staticmethod
    def _set_uniform(
        shader: Shader,
        name: str,
        value: MaterialValue
    ):

        # bool must be checked before int because:
        #
        # isinstance(True, int) == True

        if isinstance(value, bool):

            shader.set_bool(name, value)

        elif isinstance(value, int):

            shader.set_int(name, value)

        elif isinstance(value, float):

            shader.set_float(name, value)

        elif isinstance(value, tuple):

            if len(value) == 2:
                shader.set_vec2(name, value)

            elif len(value) == 3:
                shader.set_vec3(name, value)

            elif len(value) == 4:
                shader.set_vec4(name, value)

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

    def _apply_material_textures(
        self,
        shader: Shader,
        material: Material,
        resources: Resources
    ):

        textures: dict[str, Texture2D] = {}

        # Defaults first so a material only needs to set
        # the maps it actually has.

        for sampler_name, texture in self._default_textures.items():

            if shader.has_uniform(sampler_name):

                textures[
                    sampler_name
                ] = texture

        for sampler_name, texture_handle in material.textures:

            textures[
                sampler_name
            ] = resources.textures.get(
                texture_handle
            )

        engine_assert(
            len(textures) <= self.SHADOW_MAP_SLOT,
            (
                f"Material uses {len(textures)} textures; at "
                f"most {self.SHADOW_MAP_SLOT} are available."
            )
        )

        for slot, (sampler_name, texture) in enumerate(
            textures.items()
        ):

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

        self._camera_buffer.delete()
        self._lights_buffer.delete()
        self._empty_vertex_array.delete()

        self._default_textures.clear()

        RenderState.shutdown()
