from pathlib import Path

import numpy as np

from OpenGL.GL import (
    glCreateShader,
    glShaderSource,
    glCompileShader,
    glGetShaderiv,
    glGetShaderInfoLog,
    glDeleteShader,

    glCreateProgram,
    glAttachShader,
    glLinkProgram,
    glGetProgramiv,
    glGetProgramInfoLog,
    glDeleteProgram,
    glUseProgram,

    glGetUniformLocation,
    glGetUniformBlockIndex,
    glGetActiveUniformBlockiv,
    glUniformBlockBinding,

    glUniform1i,
    glUniform1f,
    glUniform2f,
    glUniform3f,
    glUniform4f,

    glUniformMatrix3fv,
    glUniformMatrix4fv,

    GL_VERTEX_SHADER,
    GL_FRAGMENT_SHADER,
    GL_COMPILE_STATUS,
    GL_LINK_STATUS,
    GL_UNIFORM_BLOCK_DATA_SIZE,
    GL_INVALID_INDEX,
    GL_TRUE
)

from core.assertions import (
    engine_assert,
    engine_assert_not_none
)

from core.exceptions import ShaderError
from core.logger import Logger

from graphics.shader_preprocessor import (
    PreprocessedShader,
    preprocess_shader
)
from graphics.uniform_blocks import (
    ENGINE_SHADER_DEFINES,
    UNIFORM_BLOCKS
)


class Shader:

    # =====================================================
    # Construction
    # =====================================================
    #
    # Sources go through graphics.shader_preprocessor, so
    # they may use #include "..." and see the engine
    # defines (MAX_POINT_LIGHTS, ...). `defines` adds or
    # overrides per-shader defines.

    def __init__(
        self,
        vertex_path,
        fragment_path,
        defines: dict[str, object] | None = None
    ):

        engine_assert_not_none(
            vertex_path,
            "Vertex shader path cannot be None."
        )

        engine_assert_not_none(
            fragment_path,
            "Fragment shader path cannot be None."
        )

        self.id = 0

        self.uniform_locations = {}

        self.vertex_path = Path(
            vertex_path
        )

        self.fragment_path = Path(
            fragment_path
        )

        self.defines = {
            **ENGINE_SHADER_DEFINES,
            **(defines or {})
        }

        # Every file (roots + includes) -> mtime at last
        # successful build. Drives hot reload.

        self._dependencies: dict[Path, float] = {}

        # =================================================
        # File Validation
        # =================================================

        if not self.vertex_path.is_file():

            raise ShaderError(
                f"Vertex shader does not exist: "
                f"{self.vertex_path}"
            )

        if not self.fragment_path.is_file():

            raise ShaderError(
                f"Fragment shader does not exist: "
                f"{self.fragment_path}"
            )

        # =================================================
        # Build
        # =================================================

        self.id, self._dependencies = self._build_program()

        Logger.info(
            "[Shader] Program created: ID=%d (%s, %s).",
            self.id,
            self.vertex_path.name,
            self.fragment_path.name
        )

    # =====================================================
    # Hot Reload
    # =====================================================

    def has_changed_on_disk(
        self
    ) -> bool:

        for path, mtime in self._dependencies.items():

            try:

                if path.stat().st_mtime != mtime:
                    return True

            except OSError:

                # Deleted / mid-save. Report a change; the
                # rebuild will fail and keep the old program.

                return True

        return False

    def reload(
        self
    ) -> bool:
        """
        Rebuild from disk. On failure, logs the error and
        keeps the current program so a typo in a shader
        does not take down the running engine.

        Returns True if the new program was installed.
        """

        engine_assert(
            self.id != 0,
            "Cannot reload a deleted Shader."
        )

        try:

            program, dependencies = self._build_program()

        except ShaderError as error:

            Logger.error(
                "[Shader] Reload failed; keeping previous "
                "program %d.\n%s",
                self.id,
                error
            )

            # Remember the broken files' mtimes so the same
            # failure is not retried every poll.

            self._refresh_dependency_mtimes()

            return False

        glDeleteProgram(
            self.id
        )

        self.id = program
        self._dependencies = dependencies

        self.uniform_locations.clear()

        Logger.info(
            "[Shader] Reloaded %s / %s as program %d.",
            self.vertex_path.name,
            self.fragment_path.name,
            self.id
        )

        return True

    def _refresh_dependency_mtimes(
        self
    ):

        for path in list(self._dependencies):

            try:

                self._dependencies[path] = (
                    path.stat().st_mtime
                )

            except OSError:
                pass

    # =====================================================
    # Program Build
    # =====================================================

    def _build_program(
        self
    ) -> tuple[int, dict[Path, float]]:

        vertex_source = preprocess_shader(
            self.vertex_path,
            self.defines
        )

        fragment_source = preprocess_shader(
            self.fragment_path,
            self.defines
        )

        dependencies: dict[Path, float] = {}

        for path in (
            vertex_source.files
            + fragment_source.files
        ):

            dependencies[path] = path.stat().st_mtime

        # -------------------------------------------------
        # Compile
        # -------------------------------------------------

        vertex_shader = self._compile_shader(
            vertex_source,
            GL_VERTEX_SHADER
        )

        try:

            fragment_shader = self._compile_shader(
                fragment_source,
                GL_FRAGMENT_SHADER
            )

        except Exception:

            glDeleteShader(
                vertex_shader
            )

            raise

        # -------------------------------------------------
        # Link
        # -------------------------------------------------

        program = glCreateProgram()

        glAttachShader(
            program,
            vertex_shader
        )

        glAttachShader(
            program,
            fragment_shader
        )

        glLinkProgram(
            program
        )

        # Shader objects are no longer needed once linked.

        glDeleteShader(
            vertex_shader
        )

        glDeleteShader(
            fragment_shader
        )

        success = glGetProgramiv(
            program,
            GL_LINK_STATUS
        )

        if not success:

            error = self._decode_log(
                glGetProgramInfoLog(
                    program
                )
            )

            glDeleteProgram(
                program
            )

            raise ShaderError(
                "Shader program linking failed "
                f"({self.vertex_path.name}, "
                f"{self.fragment_path.name}):\n"
                f"{error}"
            )

        # -------------------------------------------------
        # Uniform Blocks
        # -------------------------------------------------

        try:

            self._bind_uniform_blocks(
                program
            )

        except Exception:

            glDeleteProgram(
                program
            )

            raise

        return program, dependencies

    def _bind_uniform_blocks(
        self,
        program: int
    ):

        # GLSL 330 has no layout(binding = N) for blocks,
        # so bind each engine block by name. Blocks a
        # shader does not declare are skipped.

        for block in UNIFORM_BLOCKS:

            index = glGetUniformBlockIndex(
                program,
                block.name
            )

            if index == GL_INVALID_INDEX:
                continue

            size = np.zeros(
                1,
                dtype=np.int32
            )

            glGetActiveUniformBlockiv(
                program,
                index,
                GL_UNIFORM_BLOCK_DATA_SIZE,
                size
            )

            if int(size[0]) != block.size:

                raise ShaderError(
                    f"Uniform block '{block.name}' is "
                    f"{int(size[0])} bytes in GLSL but "
                    f"{block.size} bytes in "
                    "graphics/uniform_blocks.py. "
                    "The two layouts are out of sync."
                )

            glUniformBlockBinding(
                program,
                index,
                block.binding
            )

    # =====================================================
    # Utilities
    # =====================================================

    @staticmethod
    def _decode_log(
        value
    ):

        if isinstance(
            value,
            bytes
        ):

            return value.decode(
                "utf-8",
                errors="replace"
            )

        return str(
            value
        )

    # =====================================================
    # Compilation
    # =====================================================

    def _compile_shader(
        self,
        preprocessed: PreprocessedShader,
        shader_type
    ):

        shader = glCreateShader(
            shader_type
        )

        glShaderSource(
            shader,
            preprocessed.source
        )

        glCompileShader(
            shader
        )

        success = glGetShaderiv(
            shader,
            GL_COMPILE_STATUS
        )

        if not success:

            error = self._decode_log(
                glGetShaderInfoLog(
                    shader
                )
            )

            glDeleteShader(
                shader
            )

            # Error lines are "<source number>(<line>)" or
            # "<source number>:<line>" depending on driver;
            # list which file each source number is.

            raise ShaderError(
                f"Shader compilation failed:\n"
                f"{preprocessed.files[0]}\n\n"
                f"{error}\n"
                f"Source numbers:\n"
                f"{preprocessed.describe_files()}"
            )

        Logger.debug(
            "[Shader] Compiled: %s.",
            preprocessed.files[0]
        )

        return shader

    # =====================================================
    # Binding
    # =====================================================

    def bind(self):

        engine_assert(
            self.id != 0,
            "Attempted to bind a deleted Shader."
        )

        glUseProgram(
            self.id
        )

    @staticmethod
    def unbind():

        glUseProgram(
            0
        )

    # =====================================================
    # Uniform Location
    # =====================================================

    def get_uniform_location(
        self,
        name
    ):

        engine_assert(
            self.id != 0,
            "Cannot access uniforms on a deleted Shader."
        )

        engine_assert(
            bool(name),
            "Uniform name cannot be empty."
        )

        if name in self.uniform_locations:

            return self.uniform_locations[
                name
            ]

        location = glGetUniformLocation(
            self.id,
            name
        )

        self.uniform_locations[
            name
        ] = location

        if location == -1:

            Logger.warning(
                "[Shader] Uniform '%s' not found "
                "in program %d (%s).",
                name,
                self.id,
                self.fragment_path.name
            )

        return location

    def has_uniform(
        self,
        name
    ) -> bool:

        # Does not warn; used for optional uniforms.

        if name not in self.uniform_locations:

            self.uniform_locations[name] = glGetUniformLocation(
                self.id,
                name
            )

        return self.uniform_locations[name] != -1

    # =====================================================
    # Scalar Uniforms
    # =====================================================

    def set_bool(
        self,
        name,
        value
    ):

        location = self.get_uniform_location(
            name
        )

        if location != -1:

            glUniform1i(
                location,
                int(value)
            )

    def set_int(
        self,
        name,
        value
    ):

        location = self.get_uniform_location(
            name
        )

        if location != -1:

            glUniform1i(
                location,
                int(value)
            )

    def set_float(
        self,
        name,
        value
    ):

        location = self.get_uniform_location(
            name
        )

        if location != -1:

            glUniform1f(
                location,
                float(value)
            )

    # =====================================================
    # Vector Uniforms
    # =====================================================

    def set_vec2(
        self,
        name,
        value
    ):

        engine_assert(
            len(value) == 2,
            f"Uniform '{name}' requires a vec2."
        )

        location = self.get_uniform_location(
            name
        )

        if location != -1:

            glUniform2f(
                location,
                float(value[0]),
                float(value[1])
            )

    def set_vec3(
        self,
        name,
        value
    ):

        engine_assert(
            len(value) == 3,
            f"Uniform '{name}' requires a vec3."
        )

        location = self.get_uniform_location(
            name
        )

        if location != -1:

            glUniform3f(
                location,
                float(value[0]),
                float(value[1]),
                float(value[2])
            )

    def set_vec4(
        self,
        name,
        value
    ):

        engine_assert(
            len(value) == 4,
            f"Uniform '{name}' requires a vec4."
        )

        location = self.get_uniform_location(
            name
        )

        if location != -1:

            glUniform4f(
                location,
                float(value[0]),
                float(value[1]),
                float(value[2]),
                float(value[3])
            )

    # =====================================================
    # Matrix Uniforms
    # =====================================================

    def set_mat3(
        self,
        name,
        matrix
    ):

        matrix = np.asarray(
            matrix,
            dtype=np.float32
        )

        engine_assert(
            matrix.shape == (3, 3),
            (
                f"Uniform '{name}' requires a 3x3 "
                f"matrix, got {matrix.shape}."
            )
        )

        location = self.get_uniform_location(
            name
        )

        if location != -1:

            glUniformMatrix3fv(
                location,
                1,
                GL_TRUE,
                matrix
            )

    def set_mat4(
        self,
        name,
        matrix
    ):

        matrix = np.asarray(
            matrix,
            dtype=np.float32
        )

        engine_assert(
            matrix.shape == (4, 4),
            (
                f"Uniform '{name}' requires a 4x4 "
                f"matrix, got {matrix.shape}."
            )
        )

        location = self.get_uniform_location(
            name
        )

        if location != -1:

            glUniformMatrix4fv(
                location,
                1,
                GL_TRUE,
                matrix
            )

    # =====================================================
    # Cleanup
    # =====================================================

    def delete(self):

        if self.id == 0:
            return

        Logger.debug(
            "[Shader] Deleting program ID=%d.",
            self.id
        )

        glDeleteProgram(
            self.id
        )

        self.id = 0

        self.uniform_locations.clear()
