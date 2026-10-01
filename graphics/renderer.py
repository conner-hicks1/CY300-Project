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
    LightingFrame,
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
    # Material textures use units 0 .. FRAME_TEXTURE_SLOT - 1.
    # Per-frame engine textures (shadow maps, IBL maps) sit
    # on the units from FRAME_TEXTURE_SLOT up and stay bound
    # for the whole frame (see set_frame_textures()).
    #
    # Every sampler a shader uses is pointed at a unit that
    # holds a texture of the matching type; samplers left at
    # unit 0 could otherwise alias a different texture type,
    # which OpenGL reports as an invalid operation.

    FRAME_TEXTURE_SLOT = 7

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
        "uMetallic": 0.0,
        "uRoughness": 0.5,
        "uNormalStrength": 1.0,
        "uOcclusionStrength": 1.0,
        "uEmissive": (0.0, 0.0, 0.0),
        "uUVScale": (1.0, 1.0),
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

        # sampler name -> (unit, texture id, GL target) for
        # the current frame's engine textures.

        self._frame_textures: dict[str, tuple[int, int, int]] = {}

        self._in_scene = False

        self._origin = np.zeros(3)

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
        frame: LightingFrame | None = None
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

        # Camera-relative rendering: the GPU sees the world
        # shifted so the camera sits at the origin. The
        # shift happens here in float64; only then are
        # values narrowed to float32.
        self._origin = np.array(
            camera.position,
            dtype=np.float64
        )

        self._camera_buffer.set_data(
            pack_camera_block(
                camera.view_rotation_matrix,
                camera.projection_matrix,
                (0.0, 0.0, 0.0)
            )
        )

        self._lights_buffer.set_data(
            pack_lights_block(
                lighting,
                frame,
                origin=self._origin
            )
        )

        self.stats.reset()

        self._in_scene = True

    @property
    def origin(
        self
    ) -> np.ndarray:
        """World position of this frame's render origin (the camera)."""

        return self._origin

    def to_render_space(
        self,
        model_matrix
    ) -> np.ndarray:
        """
        World model matrix (float64) -> camera-relative
        float32 matrix for the GPU.
        """

        relative = np.array(
            model_matrix,
            dtype=np.float64
        )

        relative[:3, 3] -= self._origin

        return relative.astype(np.float32)

    def set_frame_textures(
        self,
        textures: dict[str, tuple[int, int]]
    ):
        """
        Bind per-frame engine textures (sampler name ->
        (texture id, GL target)) for the rest of the frame.
        Call after the passes that write them (shadows,
        IBL) have finished. Lit draws point these samplers
        at the right units automatically.
        """

        self._assert_in_scene()

        self._frame_textures = {}

        for offset, (name, (texture_id, target)) in enumerate(
            textures.items()
        ):

            unit = self.FRAME_TEXTURE_SLOT + offset

            RenderCommand.bind_texture(
                texture_id,
                unit,
                target
            )

            self._frame_textures[name] = (
                unit,
                texture_id,
                target
            )

    def end_scene(self):

        self._assert_in_scene()

        # Unbind frame textures so a shadow map is never
        # bound for sampling while next frame's shadow pass
        # renders into it.

        for unit, _, target in self._frame_textures.values():

            RenderCommand.bind_texture(
                0,
                unit,
                target
            )

        self._frame_textures = {}

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
            self.to_render_space(model_matrix)
        )

        if shader.has_uniform("uNormalMatrix"):

            shader.set_mat3(
                "uNormalMatrix",
                normal_matrix(model_matrix)
            )

        for name, (unit, _, _) in self._frame_textures.items():

            if shader.has_uniform(name):

                shader.set_int(
                    name,
                    unit
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
            self.to_render_space(model_matrix)
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
            self.to_render_space(model_matrix)
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
            len(textures) <= self.FRAME_TEXTURE_SLOT,
            (
                f"Material uses {len(textures)} textures; at "
                f"most {self.FRAME_TEXTURE_SLOT} are available."
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
