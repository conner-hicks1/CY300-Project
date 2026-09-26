from dataclasses import dataclass

import numpy as np

from graphics.lighting import (
    MAX_POINT_LIGHTS,
    MAX_SPOT_LIGHTS,
    LightEnvironment
)


# =========================================================
# Uniform Blocks
# =========================================================
#
# Per-frame data shared by every shader lives in uniform
# buffer objects (UBOs) instead of per-draw uniforms:
#
#     CameraBlock  view / projection / camera position
#     LightsBlock  every light + shadow parameters
#
# Each is uploaded once per frame, then any shader that
# declares the block reads it for free. The GLSL
# declarations live in assets/shaders/include/blocks.glsl.
#
# Layout is std140 and uses only vec4/ivec4/mat4 members,
# which keeps offsets trivial (16-byte aligned, arrays
# with a 16-byte stride). Matrices are column-major in
# std140, so row-major numpy matrices are transposed when
# packed.
#
# Shader checks each block's GL-reported size against
# `size` at link time, so a layout mismatch between this
# file and the GLSL fails loudly instead of rendering
# garbage.


@dataclass(frozen=True, slots=True)
class UniformBlockSpec:

    name: str
    binding: int
    size: int


# =========================================================
# Engine Shader Defines
# =========================================================
#
# Injected into every shader by the preprocessor.

ENGINE_SHADER_DEFINES: dict[str, object] = {
    "MAX_POINT_LIGHTS": MAX_POINT_LIGHTS,
    "MAX_SPOT_LIGHTS": MAX_SPOT_LIGHTS,
}


# =========================================================
# Camera Block
# =========================================================
#
#     mat4 uView;          offset   0
#     mat4 uProjection;    offset  64
#     vec4 uViewPosition;  offset 128

CAMERA_BLOCK = UniformBlockSpec(
    name="CameraBlock",
    binding=0,
    size=144
)


def pack_camera_block(
    view: np.ndarray,
    projection: np.ndarray,
    view_position
) -> bytes:

    data = np.zeros(
        CAMERA_BLOCK.size // 4,
        dtype=np.float32
    )

    data[0:16] = _column_major(view)
    data[16:32] = _column_major(projection)
    data[32:35] = np.asarray(view_position, dtype=np.float32)
    data[35] = 1.0

    return data.tobytes()


# =========================================================
# Lights Block
# =========================================================
#
#     ivec4 uLightCounts;       x point, y spot,
#                               z has directional,
#                               w shadows enabled
#     vec4  uLightParams;       x ambient,
#                               y shadow bias min,
#                               z shadow bias max,
#                               w shadow map texel size
#     vec4  uDirectionalDirection;   xyz direction
#     vec4  uDirectionalColor;       rgb color, a intensity
#     mat4  uLightSpaceMatrix;
#     vec4  uPointPositionRange[P];  xyz position, w range
#     vec4  uPointColorIntensity[P]; rgb color, a intensity
#     vec4  uSpotPositionRange[S];   xyz position, w range
#     vec4  uSpotDirectionInner[S];  xyz direction, w cos(inner)
#     vec4  uSpotColorIntensity[S];  rgb color, a intensity
#     vec4  uSpotOuter[S];           x cos(outer)

_VEC4 = 4       # floats
_MAT4 = 16      # floats

_OFFSET_COUNTS = 0
_OFFSET_PARAMS = _OFFSET_COUNTS + _VEC4
_OFFSET_DIR_DIRECTION = _OFFSET_PARAMS + _VEC4
_OFFSET_DIR_COLOR = _OFFSET_DIR_DIRECTION + _VEC4
_OFFSET_LIGHT_SPACE = _OFFSET_DIR_COLOR + _VEC4
_OFFSET_POINT_POSITION = _OFFSET_LIGHT_SPACE + _MAT4
_OFFSET_POINT_COLOR = _OFFSET_POINT_POSITION + _VEC4 * MAX_POINT_LIGHTS
_OFFSET_SPOT_POSITION = _OFFSET_POINT_COLOR + _VEC4 * MAX_POINT_LIGHTS
_OFFSET_SPOT_DIRECTION = _OFFSET_SPOT_POSITION + _VEC4 * MAX_SPOT_LIGHTS
_OFFSET_SPOT_COLOR = _OFFSET_SPOT_DIRECTION + _VEC4 * MAX_SPOT_LIGHTS
_OFFSET_SPOT_OUTER = _OFFSET_SPOT_COLOR + _VEC4 * MAX_SPOT_LIGHTS
_LIGHTS_FLOATS = _OFFSET_SPOT_OUTER + _VEC4 * MAX_SPOT_LIGHTS

LIGHTS_BLOCK = UniformBlockSpec(
    name="LightsBlock",
    binding=1,
    size=_LIGHTS_FLOATS * 4
)


@dataclass(slots=True)
class ShadowParameters:

    enabled: bool = False

    light_space_matrix: np.ndarray | None = None

    bias_min: float = 0.0005
    bias_max: float = 0.005

    texel_size: float = 1.0 / 2048.0


def pack_lights_block(
    lighting: LightEnvironment,
    shadows: ShadowParameters | None = None
) -> bytes:

    shadows = shadows or ShadowParameters()

    data = np.zeros(
        _LIGHTS_FLOATS,
        dtype=np.float32
    )

    # Same memory viewed as int32 for the ivec4 member.

    ints = data.view(
        np.int32
    )

    point_lights = lighting.point_lights[
        :MAX_POINT_LIGHTS
    ]

    spot_lights = lighting.spot_lights[
        :MAX_SPOT_LIGHTS
    ]

    directional = lighting.directional

    shadows_active = (
        shadows.enabled
        and directional is not None
        and shadows.light_space_matrix is not None
    )

    # -----------------------------------------------------
    # Counts / Params
    # -----------------------------------------------------

    ints[_OFFSET_COUNTS + 0] = len(point_lights)
    ints[_OFFSET_COUNTS + 1] = len(spot_lights)
    ints[_OFFSET_COUNTS + 2] = int(directional is not None)
    ints[_OFFSET_COUNTS + 3] = int(shadows_active)

    data[_OFFSET_PARAMS + 0] = lighting.ambient
    data[_OFFSET_PARAMS + 1] = shadows.bias_min
    data[_OFFSET_PARAMS + 2] = shadows.bias_max
    data[_OFFSET_PARAMS + 3] = shadows.texel_size

    # -----------------------------------------------------
    # Directional
    # -----------------------------------------------------

    if directional is not None:

        _write_vec4(
            data,
            _OFFSET_DIR_DIRECTION,
            directional.direction,
            0.0
        )

        _write_vec4(
            data,
            _OFFSET_DIR_COLOR,
            directional.color,
            directional.intensity
        )

    light_space = (
        shadows.light_space_matrix
        if shadows_active
        else np.identity(4, dtype=np.float32)
    )

    data[
        _OFFSET_LIGHT_SPACE:_OFFSET_LIGHT_SPACE + _MAT4
    ] = _column_major(
        light_space
    )

    # -----------------------------------------------------
    # Point Lights
    # -----------------------------------------------------

    for index, point in enumerate(
        point_lights
    ):

        _write_vec4(
            data,
            _OFFSET_POINT_POSITION + index * _VEC4,
            point.position,
            point.range
        )

        _write_vec4(
            data,
            _OFFSET_POINT_COLOR + index * _VEC4,
            point.color,
            point.intensity
        )

    # -----------------------------------------------------
    # Spot Lights
    # -----------------------------------------------------

    for index, spot in enumerate(
        spot_lights
    ):

        _write_vec4(
            data,
            _OFFSET_SPOT_POSITION + index * _VEC4,
            spot.position,
            spot.range
        )

        _write_vec4(
            data,
            _OFFSET_SPOT_DIRECTION + index * _VEC4,
            spot.direction,
            spot.inner_cutoff
        )

        _write_vec4(
            data,
            _OFFSET_SPOT_COLOR + index * _VEC4,
            spot.color,
            spot.intensity
        )

        data[
            _OFFSET_SPOT_OUTER + index * _VEC4
        ] = spot.outer_cutoff

    return data.tobytes()


# =========================================================
# Helpers
# =========================================================

UNIFORM_BLOCKS: tuple[UniformBlockSpec, ...] = (
    CAMERA_BLOCK,
    LIGHTS_BLOCK,
)


def _column_major(
    matrix
) -> np.ndarray:

    matrix = np.asarray(
        matrix,
        dtype=np.float32
    )

    return matrix.T.ravel()


def _write_vec4(
    data: np.ndarray,
    offset: int,
    xyz,
    w: float
):

    data[offset:offset + 3] = np.asarray(
        xyz,
        dtype=np.float32
    )[:3]

    data[offset + 3] = w
