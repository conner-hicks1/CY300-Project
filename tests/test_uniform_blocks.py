import numpy as np

from graphics.lighting import (
    MAX_CASCADES,
    MAX_POINT_LIGHTS,
    MAX_SPOT_LIGHTS,
    DirectionalLight,
    LightEnvironment,
    PointLight,
    SpotLight
)
from graphics.shadows import Cascade, SpotShadow
from graphics.uniform_blocks import (
    CAMERA_BLOCK,
    ENGINE_SHADER_DEFINES,
    LIGHTS_BLOCK,
    LightingFrame,
    pack_camera_block,
    pack_lights_block
)


# Float offsets (4 bytes each), mirroring the documented layout.
COUNTS, PARAMS, DIR, DIR_COLOR, SPLITS, TEXELS = 0, 4, 8, 12, 16, 20
CASCADES = 24
POINT_POS = CASCADES + 16 * MAX_CASCADES
POINT_COLOR = POINT_POS + 4 * MAX_POINT_LIGHTS
SPOT_POS = POINT_COLOR + 4 * MAX_POINT_LIGHTS
SPOT_DIR = SPOT_POS + 4 * MAX_SPOT_LIGHTS
SPOT_COLOR = SPOT_DIR + 4 * MAX_SPOT_LIGHTS
SPOT_PARAMS = SPOT_COLOR + 4 * MAX_SPOT_LIGHTS
SPOT_MATRICES = SPOT_PARAMS + 4 * MAX_SPOT_LIGHTS


def test_block_sizes():

    assert CAMERA_BLOCK.size == 144

    vec4s = 6 + 2 * MAX_POINT_LIGHTS + 4 * MAX_SPOT_LIGHTS
    mat4s = MAX_CASCADES + MAX_SPOT_LIGHTS

    assert LIGHTS_BLOCK.size == 16 * vec4s + 64 * mat4s


def test_engine_defines_include_limits():

    assert ENGINE_SHADER_DEFINES["MAX_CASCADES"] == MAX_CASCADES


def test_camera_block_is_column_major():

    view = np.arange(16, dtype=np.float32).reshape(4, 4)

    data = np.frombuffer(
        pack_camera_block(view, np.identity(4), (1.0, 2.0, 3.0)),
        dtype=np.float32
    )

    assert len(data) * 4 == CAMERA_BLOCK.size

    # First column of `view` comes first.
    assert np.allclose(data[0:4], view[:, 0])
    assert np.allclose(data[32:36], (1.0, 2.0, 3.0, 1.0))


def test_lights_block_layout():

    lighting = LightEnvironment(
        directional=DirectionalLight(direction=np.array([0.0, -1.0, 0.0]), color=(1.0, 0.5, 0.25), intensity=2.0),
        point_lights=[
            PointLight(position=np.array([1.0, 2.0, 3.0]), color=(1.0, 0.0, 0.0), intensity=4.0, range=6.0)
        ],
        spot_lights=[
            SpotLight(
                position=np.array([4.0, 5.0, 6.0]),
                direction=np.array([0.0, 0.0, -1.0]),
                color=(0.0, 0.0, 1.0),
                intensity=8.0,
                range=9.0,
                inner_cutoff=0.9,
                outer_cutoff=0.8
            ),
            SpotLight(
                position=np.zeros(3),
                direction=np.array([0.0, -1.0, 0.0])
            )
        ]
    )

    cascade_matrix = np.arange(16, dtype=np.float32).reshape(4, 4)
    spot_matrix = cascade_matrix + 100.0

    frame = LightingFrame(
        ibl_intensity=0.75,
        cascades=[
            Cascade(matrix=cascade_matrix, split_far=5.0, texel_world_size=0.01),
            Cascade(matrix=cascade_matrix, split_far=20.0, texel_world_size=0.04),
        ],
        spot_shadows=[SpotShadow(matrix=spot_matrix, texel_scale=0.002), None],
        shadow_depth_bias=0.001,
        shadow_normal_offset=2.0,
        visualize_cascades=True
    )

    raw = pack_lights_block(lighting, frame)

    assert len(raw) == LIGHTS_BLOCK.size

    floats = np.frombuffer(raw, dtype=np.float32)
    ints = np.frombuffer(raw, dtype=np.int32)

    assert list(ints[COUNTS:COUNTS + 4]) == [1, 2, 1, 2]
    assert np.allclose(floats[PARAMS:PARAMS + 4], (0.75, 0.001, 2.0, 1.0))
    assert np.allclose(floats[DIR:DIR + 4], (0.0, -1.0, 0.0, 0.0))
    assert np.allclose(floats[DIR_COLOR:DIR_COLOR + 4], (1.0, 0.5, 0.25, 2.0))
    assert np.allclose(floats[SPLITS:SPLITS + 2], (5.0, 20.0))
    assert np.allclose(floats[TEXELS:TEXELS + 2], (0.01, 0.04))
    assert np.allclose(floats[CASCADES:CASCADES + 16], cascade_matrix.T.ravel())

    assert np.allclose(floats[POINT_POS:POINT_POS + 4], (1.0, 2.0, 3.0, 6.0))
    assert np.allclose(floats[POINT_COLOR:POINT_COLOR + 4], (1.0, 0.0, 0.0, 4.0))

    assert np.allclose(floats[SPOT_POS:SPOT_POS + 4], (4.0, 5.0, 6.0, 9.0))
    assert np.allclose(floats[SPOT_DIR:SPOT_DIR + 4], (0.0, 0.0, -1.0, 0.9))

    # Spot 0 has a shadow on layer 0; spot 1 has none.
    assert np.allclose(floats[SPOT_PARAMS:SPOT_PARAMS + 3], (0.8, 0.0, 0.002))
    assert floats[SPOT_PARAMS + 4 + 1] == -1.0
    assert np.allclose(floats[SPOT_MATRICES:SPOT_MATRICES + 16], spot_matrix.T.ravel())


def test_no_cascades_without_directional_light():

    frame = LightingFrame(
        cascades=[Cascade(matrix=np.identity(4), split_far=5.0, texel_world_size=0.01)]
    )

    ints = np.frombuffer(pack_lights_block(LightEnvironment(), frame), dtype=np.int32)

    assert ints[2] == 0
    assert ints[3] == 0
