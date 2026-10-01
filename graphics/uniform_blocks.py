from dataclasses import dataclass, field

import numpy as np

from graphics.lighting import (
    MAX_CASCADES,
    MAX_POINT_LIGHTS,
    MAX_SPOT_LIGHTS,
    PREFILTER_MIP_LEVELS,
    LightEnvironment
)
from graphics.shadows import (
    Cascade,
    SpotShadow
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
    "MAX_CASCADES": MAX_CASCADES,
    "PREFILTER_MIP_LEVELS": PREFILTER_MIP_LEVELS,
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
#     ivec4 uLightCounts;            x point, y spot,
#                                    z has directional,
#                                    w cascade count (0 = no
#                                      directional shadows)
#     vec4  uLightParams;            x IBL intensity,
#                                    y shadow depth bias,
#                                    z normal offset (texels),
#                                    w visualize cascades
#     vec4  uDirectionalDirection;   xyz direction
#     vec4  uDirectionalColor;       rgb color, a intensity
#     vec4  uCascadeSplits;          view-space far distance per cascade
#     vec4  uCascadeTexelSizes;      world size of a texel per cascade
#     mat4  uCascadeMatrices[C];     world -> cascade clip
#     vec4  uPointPositionRange[P];  xyz position, w range
#     vec4  uPointColorIntensity[P]; rgb color, a intensity
#     vec4  uSpotPositionRange[S];   xyz position, w range
#     vec4  uSpotDirectionInner[S];  xyz direction, w cos(inner)
#     vec4  uSpotColorIntensity[S];  rgb color, a intensity
#     vec4  uSpotParams[S];          x cos(outer),
#                                    y shadow layer (-1 = none),
#                                    z shadow texel scale
#     mat4  uSpotMatrices[S];        world -> spot shadow clip

_VEC4 = 4       # floats
_MAT4 = 16      # floats

_OFFSET_COUNTS = 0
_OFFSET_PARAMS = _OFFSET_COUNTS + _VEC4
_OFFSET_DIR_DIRECTION = _OFFSET_PARAMS + _VEC4
_OFFSET_DIR_COLOR = _OFFSET_DIR_DIRECTION + _VEC4
_OFFSET_CASCADE_SPLITS = _OFFSET_DIR_COLOR + _VEC4
_OFFSET_CASCADE_TEXELS = _OFFSET_CASCADE_SPLITS + _VEC4
_OFFSET_CASCADE_MATRICES = _OFFSET_CASCADE_TEXELS + _VEC4
_OFFSET_POINT_POSITION = _OFFSET_CASCADE_MATRICES + _MAT4 * MAX_CASCADES
_OFFSET_POINT_COLOR = _OFFSET_POINT_POSITION + _VEC4 * MAX_POINT_LIGHTS
_OFFSET_SPOT_POSITION = _OFFSET_POINT_COLOR + _VEC4 * MAX_POINT_LIGHTS
_OFFSET_SPOT_DIRECTION = _OFFSET_SPOT_POSITION + _VEC4 * MAX_SPOT_LIGHTS
_OFFSET_SPOT_COLOR = _OFFSET_SPOT_DIRECTION + _VEC4 * MAX_SPOT_LIGHTS
_OFFSET_SPOT_PARAMS = _OFFSET_SPOT_COLOR + _VEC4 * MAX_SPOT_LIGHTS
_OFFSET_SPOT_MATRICES = _OFFSET_SPOT_PARAMS + _VEC4 * MAX_SPOT_LIGHTS
_LIGHTS_FLOATS = _OFFSET_SPOT_MATRICES + _MAT4 * MAX_SPOT_LIGHTS

LIGHTS_BLOCK = UniformBlockSpec(
    name="LightsBlock",
    binding=1,
    size=_LIGHTS_FLOATS * 4
)


@dataclass(slots=True)
class LightingFrame:
    """
    Per-frame lighting inputs beyond the lights themselves.

    spot_shadows[i] is the shadow of lighting.spot_lights[i]
    (None if that light casts no shadow). A spot shadow's
    layer in the shadow texture array is its index.
    """

    ibl_intensity: float = 1.0

    cascades: list[Cascade] = field(
        default_factory=list
    )

    spot_shadows: list[SpotShadow | None] = field(
        default_factory=list
    )

    shadow_depth_bias: float = 0.0005
    shadow_normal_offset: float = 1.5

    visualize_cascades: bool = False


def pack_lights_block(
    lighting: LightEnvironment,
    frame: LightingFrame | None = None,
    origin=(0.0, 0.0, 0.0)
) -> bytes:
    """
    origin: render origin (camera position). Light positions
    are stored relative to it, matching camera-relative
    rendering. Shadow matrices in `frame` must already be in
    render space (graphics.shadows.to_render_space).
    """

    frame = frame or LightingFrame()

    origin = np.asarray(
        origin,
        dtype=np.float64
    )

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

    cascades = (
        frame.cascades[:MAX_CASCADES]
        if directional is not None
        else []
    )

    # -----------------------------------------------------
    # Counts / Params
    # -----------------------------------------------------

    ints[_OFFSET_COUNTS + 0] = len(point_lights)
    ints[_OFFSET_COUNTS + 1] = len(spot_lights)
    ints[_OFFSET_COUNTS + 2] = int(directional is not None)
    ints[_OFFSET_COUNTS + 3] = len(cascades)

    data[_OFFSET_PARAMS + 0] = frame.ibl_intensity
    data[_OFFSET_PARAMS + 1] = frame.shadow_depth_bias
    data[_OFFSET_PARAMS + 2] = frame.shadow_normal_offset
    data[_OFFSET_PARAMS + 3] = float(frame.visualize_cascades)

    # -----------------------------------------------------
    # Directional + Cascades
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

    for index, cascade in enumerate(cascades):

        data[_OFFSET_CASCADE_SPLITS + index] = cascade.split_far
        data[_OFFSET_CASCADE_TEXELS + index] = cascade.texel_world_size

        offset = _OFFSET_CASCADE_MATRICES + index * _MAT4

        data[offset:offset + _MAT4] = _column_major(
            cascade.matrix
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
            np.asarray(point.position, dtype=np.float64) - origin,
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
            np.asarray(spot.position, dtype=np.float64) - origin,
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

        shadow = (
            frame.spot_shadows[index]
            if index < len(frame.spot_shadows)
            else None
        )

        params = _OFFSET_SPOT_PARAMS + index * _VEC4

        data[params + 0] = spot.outer_cutoff
        data[params + 1] = float(index) if shadow is not None else -1.0
        data[params + 2] = shadow.texel_scale if shadow is not None else 0.0

        if shadow is not None:

            offset = _OFFSET_SPOT_MATRICES + index * _MAT4

            data[offset:offset + _MAT4] = _column_major(
                shadow.matrix
            )

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
