from dataclasses import fields

from graphics.render_settings import RenderSettings


# =========================================================
# Graphics Quality
# =========================================================
#
# The settings that cost the most frames, as presets: the
# scene's resolution (its cost is nearly all per pixel), the
# atmosphere's samples per pixel, rocks, trees and grass
# near the ground (~7 ms of GPU time in a dense forest),
# shadow reach and the mountain-shadow map (~6 ms), bloom
# and FXAA. Everything else in RenderSettings is left
# alone.

QUALITY_PRESETS: dict[str, dict[str, object]] = {
    "Low": dict(
        atmosphere_samples=12,
        scatter=False,
        shadows_enabled=True,
        shadow_map_size=1024,
        cascade_count=2,
        shadow_distance=20.0,
        terrain_shadows=False,
        bloom_enabled=False,
        fxaa_enabled=True,
        render_scale=0.7,
    ),
    "Medium": dict(
        atmosphere_samples=18,
        scatter=True,
        shadows_enabled=True,
        shadow_map_size=2048,
        cascade_count=3,
        shadow_distance=30.0,
        terrain_shadows=False,
        bloom_enabled=True,
        fxaa_enabled=True,
        render_scale=0.85,
    ),
    "High": dict(
        atmosphere_samples=24,
        scatter=True,
        shadows_enabled=True,
        shadow_map_size=2048,
        cascade_count=4,
        shadow_distance=30.0,
        terrain_shadows=True,
        bloom_enabled=True,
        fxaa_enabled=True,
        render_scale=1.0,
    ),
}

QUALITY_NAMES: tuple[str, ...] = tuple(QUALITY_PRESETS)

# Medium: the scene at 85% resolution and no mountain
# shadows keeps a laptop GPU (Iris Xe) near 45 fps.
DEFAULT_QUALITY = "Medium"


def apply_quality(
    settings: RenderSettings,
    name: str
):
    """Set the preset's fields on `settings`."""

    for field, value in QUALITY_PRESETS[name].items():
        setattr(settings, field, value)


def current_quality(
    settings: RenderSettings
) -> str | None:
    """The preset `settings` match, or None (customized)."""

    for name, preset in QUALITY_PRESETS.items():

        if all(getattr(settings, field) == value for field, value in preset.items()):
            return name

    return None


# Every preset sets the same, real fields.
_FIELDS = {field.name for field in fields(RenderSettings)}

assert all(set(preset) <= _FIELDS for preset in QUALITY_PRESETS.values())
assert len({frozenset(preset) for preset in QUALITY_PRESETS.values()}) == 1
