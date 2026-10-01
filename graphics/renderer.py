import ctypes

from dataclasses import dataclass

import numpy as np

from OpenGL.GL import (
    GL_DRAW_INDIRECT_BUFFER,
    GL_DYNAMIC_DRAW,
    GL_SHADER_STORAGE_BUFFER,
    GL_STREAM_DRAW,
    GL_TRIANGLES,
    GL_UNSIGNED_INT,
    glBindBuffer,
    glBindBufferBase,
    glBufferData,
    glBufferSubData,
    glDeleteBuffers,
    glDrawElementsBaseVertex,
    glGenBuffers,
    glMultiDrawElementsIndirect
)

from core.assertions import engine_assert

from graphics.buffer import UniformBuffer
from graphics.draw_list import (
    DrawItem,
    PreparedDraws,
    build_commands,
    group_by_material,
    prepare as prepare_draw_list
)
from graphics.geometry_pool import GeometryPool
from graphics.lighting import LightEnvironment
from graphics.material import Material, MaterialValue
from graphics.mesh import Mesh
from graphics.render_command import RenderCommand
from graphics.render_state import RenderState
from graphics.shader import Shader
from graphics.texture import Texture2D
from graphics.uniform_blocks import (
    CAMERA_BLOCK,
    ATMOSPHERE_BLOCK,
    LIGHTS_BLOCK,
    LightingFrame,
    pack_camera_block,
    pack_lights_block
)
from graphics.vertex_array import VertexArray

from math3d.camera import Camera

from resources.resources import Resources


# =========================================================
# Frame Statistics
# =========================================================

@dataclass(slots=True)
class RenderStats:

    # GPU draw submissions (one multi-draw counts once).
    draw_calls: int = 0
    triangles: int = 0

    # Scene objects drawn (after culling) / prepared.
    objects_drawn: int = 0
    objects_total: int = 0

    def reset(self):

        self.draw_calls = 0
        self.triangles = 0
        self.objects_drawn = 0
        self.objects_total = 0


# Storage-buffer binding of the per-draw records
# (assets/shaders/include/draw_data.glsl).
DRAW_RECORDS_BINDING = 2


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
        "uTerrainShading": 0.0,
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

        # Starts disabled (all zeros) until a scene with an
        # atmosphere sets it.
        self._atmosphere_buffer = UniformBuffer(
            ATMOSPHERE_BLOCK.size,
            ATMOSPHERE_BLOCK.binding
        )

        self._atmosphere_buffer.set_data(
            bytes(ATMOSPHERE_BLOCK.size)
        )

        # Core profile requires a bound VAO even for a
        # draw with no vertex attributes (fullscreen
        # triangle generated from gl_VertexID).

        self._empty_vertex_array = VertexArray()

        # Batched rendering: per-draw records (storage
        # buffer) and indirect commands.

        self._records_buffer = int(glGenBuffers(1))
        self._command_buffer = int(glGenBuffers(1))

        self._prepared: PreparedDraws | None = None

        self._command_capacity = 0

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

        self._prepared = None

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
    # Atmosphere
    # =====================================================

    def set_atmosphere(
        self,
        data: bytes
    ):
        """
        Upload the AtmosphereBlock (graphics/atmosphere.py
        pack_atmosphere_block). Frame-wide: read by the
        atmosphere passes and lit shaders.
        """

        self._atmosphere_buffer.set_data(data)

    # =====================================================
    # Batched Draws
    # =====================================================
    #
    # Per frame:
    #
    #     renderer.begin_scene(camera, lighting, frame)
    #     renderer.prepare_draws(items)        # all objects
    #     renderer.draw_depth_batch(shader, indices)   # per shadow layer
    #     renderer.draw_batch(resources, indices)      # scene
    #
    # prepare_draws() computes every object's camera-relative
    # matrices at once (numpy) and uploads them to a storage
    # buffer. Each draw_* call then issues one
    # glMultiDrawElementsIndirect per material for any
    # subset of objects (e.g. those that survived culling),
    # instead of one Python-driven draw call per object.

    def prepare_draws(
        self,
        items: list[DrawItem]
    ) -> PreparedDraws:

        self._assert_in_scene()

        prepared = prepare_draw_list(
            items,
            self._origin
        )

        self._prepared = prepared

        count = len(prepared.items)

        self.stats.objects_total = count

        if count == 0:
            return prepared

        GeometryPool.instance().ensure_draw_capacity(count)

        self._upload(
            GL_SHADER_STORAGE_BUFFER,
            self._records_buffer,
            prepared.records
        )

        glBindBufferBase(
            GL_SHADER_STORAGE_BUFFER,
            DRAW_RECORDS_BINDING,
            self._records_buffer
        )

        return prepared

    @property
    def prepared(
        self
    ) -> PreparedDraws | None:

        return self._prepared

    def draw_batch(
        self,
        resources: Resources,
        indices: np.ndarray
    ):
        """Lit draw of prepared objects `indices`, one multi-draw per material."""

        self._assert_prepared()

        for material, members in group_by_material(
            self._prepared,
            indices
        ):

            shader = resources.shaders.get(
                material.shader
            )

            shader.bind()

            for name, (unit, _, _) in self._frame_textures.items():

                if shader.has_uniform(name):
                    shader.set_int(name, unit)

            self._apply_material_values(shader, material)
            self._apply_material_textures(shader, material, resources)

            self._multi_draw(members)

        self.stats.objects_drawn += len(indices)

    def draw_depth_batch(
        self,
        shader: Shader,
        indices: np.ndarray
    ):
        """
        Depth-only draw of prepared objects with an already
        bound shader (uLightMatrix set by the caller).
        """

        self._assert_prepared()

        self._multi_draw(indices)

    def _multi_draw(
        self,
        indices: np.ndarray
    ):

        if len(indices) == 0:
            return

        commands = build_commands(
            self._prepared,
            indices
        )

        self._write_commands(
            commands
        )

        GeometryPool.instance().bind()

        glBindBuffer(GL_DRAW_INDIRECT_BUFFER, self._command_buffer)

        # The indirect "pointer" is a byte offset into the
        # bound GL_DRAW_INDIRECT_BUFFER; pass it as a ctypes
        # pointer value (None makes PyOpenGL treat it as
        # client memory).
        glMultiDrawElementsIndirect(
            GL_TRIANGLES,
            GL_UNSIGNED_INT,
            ctypes.c_void_p(0),
            len(commands),
            0
        )

        glBindBuffer(GL_DRAW_INDIRECT_BUFFER, 0)

        self.stats.draw_calls += 1
        self.stats.triangles += int(commands[:, 0].sum()) // 3

    def _write_commands(
        self,
        commands: np.ndarray
    ):

        # Storage is allocated once (grown when needed) and
        # updated in place. Re-specifying the indirect buffer
        # with glBufferData between multi-draws triggers
        # GL_INVALID_OPERATION on the Intel driver.

        data = np.ascontiguousarray(commands)

        glBindBuffer(GL_DRAW_INDIRECT_BUFFER, self._command_buffer)

        if data.nbytes > self._command_capacity:

            capacity = max(self._command_capacity, 4096)

            while capacity < data.nbytes:
                capacity *= 2

            glBufferData(GL_DRAW_INDIRECT_BUFFER, capacity, None, GL_DYNAMIC_DRAW)

            self._command_capacity = capacity

        glBufferSubData(GL_DRAW_INDIRECT_BUFFER, 0, data.nbytes, data)

        glBindBuffer(GL_DRAW_INDIRECT_BUFFER, 0)

    @staticmethod
    def _upload(
        target: int,
        buffer: int,
        array: np.ndarray
    ):

        # Orphan + refill: the driver hands back fresh
        # storage, so this never waits for the GPU to finish
        # reading last frame's data.

        data = np.ascontiguousarray(array)

        glBindBuffer(target, buffer)
        glBufferData(target, data.nbytes, data, GL_STREAM_DRAW)
        glBindBuffer(target, 0)

    def _assert_prepared(self):

        self._assert_in_scene()

        engine_assert(
            self._prepared is not None,
            "Call Renderer.prepare_draws() before batched draws."
        )

    # =====================================================
    # Single Draws
    # =====================================================

    def draw_unlit(
        self,
        mesh: Mesh,
        shader: Shader,
        model_matrix: np.ndarray,
        color
    ):
        """One flat-colored mesh (light gizmos, selection outline)."""

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

        allocation = mesh.allocation

        GeometryPool.instance().bind()

        glDrawElementsBaseVertex(
            GL_TRIANGLES,
            allocation.index_count,
            GL_UNSIGNED_INT,
            ctypes.c_void_p(allocation.first_index * 4),
            allocation.base_vertex
        )

        self.stats.draw_calls += 1
        self.stats.triangles += allocation.index_count // 3

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
        self._atmosphere_buffer.delete()
        self._empty_vertex_array.delete()

        glDeleteBuffers(
            2,
            [self._records_buffer, self._command_buffer]
        )

        GeometryPool.shutdown()

        self._default_textures.clear()

        RenderState.shutdown()
