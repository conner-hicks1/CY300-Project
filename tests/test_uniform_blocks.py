import numpy as np

from graphics.lighting import (
    MAX_POINT_LIGHTS,
    MAX_SPOT_LIGHTS,
    DirectionalLight,
    LightEnvironment,
    PointLight,
    SpotLight
)
from graphics.uniform_blocks import (
    CAMERA_BLOCK,
    LIGHTS_BLOCK,
    ShadowParameters,
    pack_camera_block,
    pack_lights_block
)


def test_block_sizes():

    assert CAMERA_BLOCK.size == 144

    # 4 vec4 + mat4 + 2 vec4 per point + 4 vec4 per spot.
    assert LIGHTS_BLOCK.size == 16 * (4 + 4 + 2 * MAX_POINT_LIGHTS + 4 * MAX_SPOT_LIGHTS)


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
        ambient=0.25,
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
            )
        ]
    )

    light_space = np.arange(16, dtype=np.float32).reshape(4, 4)

    raw = pack_lights_block(lighting, ShadowParameters(enabled=True, light_space_matrix=light_space))

    assert len(raw) == LIGHTS_BLOCK.size

    floats = np.frombuffer(raw, dtype=np.float32)
    ints = np.frombuffer(raw, dtype=np.int32)

    assert list(ints[0:4]) == [1, 1, 1, 1]
    assert floats[4] == np.float32(0.25)
    assert np.allclose(floats[8:12], (0.0, -1.0, 0.0, 0.0))
    assert np.allclose(floats[12:16], (1.0, 0.5, 0.25, 2.0))
    assert np.allclose(floats[16:32], light_space.T.ravel())

    point = 32
    assert np.allclose(floats[point:point + 4], (1.0, 2.0, 3.0, 6.0))

    point_color = point + 4 * MAX_POINT_LIGHTS
    assert np.allclose(floats[point_color:point_color + 4], (1.0, 0.0, 0.0, 4.0))

    spot = point_color + 4 * MAX_POINT_LIGHTS
    assert np.allclose(floats[spot:spot + 4], (4.0, 5.0, 6.0, 9.0))

    spot_direction = spot + 4 * MAX_SPOT_LIGHTS
    assert np.allclose(floats[spot_direction:spot_direction + 4], (0.0, 0.0, -1.0, 0.9))

    spot_outer = spot_direction + 8 * MAX_SPOT_LIGHTS
    assert floats[spot_outer] == np.float32(0.8)


def test_shadows_disabled_without_directional_light():

    raw = pack_lights_block(
        LightEnvironment(),
        ShadowParameters(enabled=True, light_space_matrix=np.identity(4))
    )

    ints = np.frombuffer(raw, dtype=np.int32)

    assert ints[2] == 0
    assert ints[3] == 0
