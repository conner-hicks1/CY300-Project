from typing import TypeAlias

from core.assertions import engine_assert
from core.handle import Handle


MaterialValue: TypeAlias = (
    bool
    | int
    | float
    | tuple[float, float]
    | tuple[float, float, float]
    | tuple[float, float, float, float]
)


class Material:

    # =====================================================
    # Construction
    # =====================================================

    def __init__(
        self,
        shader: Handle
    ):

        engine_assert(
            isinstance(shader, Handle),
            "Material shader must be a Handle."
        )

        engine_assert(
            shader.is_valid,
            "Material shader handle must be valid."
        )

        self.shader = shader

        # -------------------------------------------------
        # Textures
        # -------------------------------------------------
        #
        # sampler uniform name -> texture handle

        self._textures: dict[
            str,
            Handle
        ] = {}

        # -------------------------------------------------
        # Uniform Values
        # -------------------------------------------------

        self._values: dict[
            str,
            MaterialValue
        ] = {}

    # =====================================================
    # Shader
    # =====================================================

    def set_shader(
        self,
        shader: Handle
    ):

        engine_assert(
            isinstance(shader, Handle),
            "Material shader must be a Handle."
        )

        engine_assert(
            shader.is_valid,
            "Material shader handle must be valid."
        )

        self.shader = shader

    # =====================================================
    # Textures
    # =====================================================

    def set_texture(
        self,
        name: str,
        texture: Handle
    ):

        engine_assert(
            isinstance(name, str),
            "Texture uniform name must be a string."
        )

        engine_assert(
            bool(name.strip()),
            "Texture uniform name cannot be empty."
        )

        engine_assert(
            isinstance(texture, Handle),
            "Material texture must be a Handle."
        )

        engine_assert(
            texture.is_valid,
            "Material texture handle must be valid."
        )

        self._textures[
            name
        ] = texture

    def remove_texture(
        self,
        name: str
    ):

        self._textures.pop(
            name,
            None
        )

    def has_texture(
        self,
        name: str
    ) -> bool:

        return (
            name
            in self._textures
        )

    def get_texture(
        self,
        name: str
    ) -> Handle:

        texture = self._textures.get(
            name
        )

        engine_assert(
            texture is not None,
            (
                "Material has no texture "
                f"named '{name}'."
            )
        )

        return texture

    @property
    def textures(
        self
    ):

        return self._textures.items()

    # =====================================================
    # Scalar Uniforms
    # =====================================================

    def set_bool(
        self,
        name: str,
        value: bool
    ):

        self._set_value(
            name,
            bool(value)
        )

    def set_int(
        self,
        name: str,
        value: int
    ):

        self._set_value(
            name,
            int(value)
        )

    def set_float(
        self,
        name: str,
        value: float
    ):

        self._set_value(
            name,
            float(value)
        )

    # =====================================================
    # Vector Uniforms
    # =====================================================

    def set_vec2(
        self,
        name: str,
        value
    ):

        engine_assert(
            len(value) == 2,
            "Material vec2 must contain two values."
        )

        self._set_value(
            name,
            (
                float(value[0]),
                float(value[1])
            )
        )

    def set_vec3(
        self,
        name: str,
        value
    ):

        engine_assert(
            len(value) == 3,
            "Material vec3 must contain three values."
        )

        self._set_value(
            name,
            (
                float(value[0]),
                float(value[1]),
                float(value[2])
            )
        )

    def set_vec4(
        self,
        name: str,
        value
    ):

        engine_assert(
            len(value) == 4,
            "Material vec4 must contain four values."
        )

        self._set_value(
            name,
            (
                float(value[0]),
                float(value[1]),
                float(value[2]),
                float(value[3])
            )
        )

    # =====================================================
    # Uniform Access
    # =====================================================

    def _set_value(
        self,
        name: str,
        value: MaterialValue
    ):

        engine_assert(
            isinstance(name, str),
            "Material uniform name must be a string."
        )

        engine_assert(
            bool(name.strip()),
            "Material uniform name cannot be empty."
        )

        self._values[
            name
        ] = value

    def get_value(
        self,
        name: str,
        default: MaterialValue | None = None
    ) -> MaterialValue | None:

        return self._values.get(
            name,
            default
        )

    def remove_value(
        self,
        name: str
    ):

        self._values.pop(
            name,
            None
        )

    @property
    def values(
        self
    ):

        return self._values.items()

    # =====================================================
    # Representation
    # =====================================================

    def __repr__(self):

        return (
            f"Material("
            f"shader={self.shader}, "
            f"textures={len(self._textures)}, "
            f"values={len(self._values)}"
            f")"
        )