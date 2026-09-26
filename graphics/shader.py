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
    GL_TRUE
)

from core.assertions import (
    engine_assert,
    engine_assert_not_none
)

from core.exceptions import ShaderError
from core.logger import Logger


class Shader:

    def __init__(
        self,
        vertex_path,
        fragment_path
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
        # Compile
        # =================================================

        vertex_shader = self._compile_shader(
            self.vertex_path,
            GL_VERTEX_SHADER
        )

        try:

            fragment_shader = self._compile_shader(
                self.fragment_path,
                GL_FRAGMENT_SHADER
            )

        except Exception:

            glDeleteShader(
                vertex_shader
            )

            raise

        # =================================================
        # Link
        # =================================================

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

            glDeleteShader(
                vertex_shader
            )

            glDeleteShader(
                fragment_shader
            )

            glDeleteProgram(
                program
            )

            raise ShaderError(
                "Shader program linking failed:\n"
                f"{error}"
            )

        # Shader objects are no longer needed once linked.

        glDeleteShader(
            vertex_shader
        )

        glDeleteShader(
            fragment_shader
        )

        self.id = program

        Logger.info(
            "[Shader] Program created: ID=%d.",
            self.id
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
        filepath,
        shader_type
    ):

        try:

            source = filepath.read_text(
                encoding="utf-8"
            )

        except OSError as error:

            raise ShaderError(
                f"Failed to read shader "
                f"'{filepath}': {error}"
            ) from error

        shader = glCreateShader(
            shader_type
        )

        glShaderSource(
            shader,
            source
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

            raise ShaderError(
                f"Shader compilation failed:\n"
                f"{filepath}\n\n"
                f"{error}"
            )

        Logger.debug(
            "[Shader] Compiled: %s.",
            filepath
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
                "in program %d.",
                name,
                self.id
            )

        return location

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