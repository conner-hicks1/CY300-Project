from ecs.components import BodyComponent


# =========================================================
# Readable Quantities
# =========================================================

def distance(
    meters: float
) -> str:
    """Human-readable distance."""

    magnitude = abs(meters)

    if magnitude < 1_000.0:
        return f"{meters:,.0f} m"

    if magnitude < 100_000.0:
        return f"{meters / 1000.0:,.1f} km"

    return f"{meters / 1000.0:,.0f} km"


def pressure(
    bar: float
) -> str:
    """Human-readable pressure."""

    if bar >= 10.0:
        return f"{bar:,.0f} bar"

    if bar >= 0.01:
        return f"{bar:.2f} bar"

    return f"{bar * 1e5:.3g} Pa"


def duration(
    hours: float
) -> str:

    hours = abs(hours)

    return f"{hours:.1f} h" if hours < 48.0 else f"{hours / 24.0:.1f} days"


def body_facts(
    component: BodyComponent
) -> list[tuple[str, str]]:
    """Readable facts about a body (the pause menu's travel list)."""

    year = component.year_days

    return [
        ("Gravity", f"{component.surface_gravity:.2f} m/s2  ({component.surface_gravity / 9.80665:.2f} g)"),
        ("Day (solar)", duration(component.solar_day_hours)),
        ("Rotation", duration(component.rotation_hours) + (" (retrograde)" if component.rotation_hours < 0 else "")),
        ("Year", f"{year:,.1f} days" if year < 1000 else f"{year / 365.256:,.1f} Earth years"),
        ("Distance", f"{component.orbit_distance_au:.3f} AU from the star"),
        ("Sunlight", f"{component.star_luminosity / max(component.orbit_distance_au, 1e-9) ** 2 * 100.0:.2f}% of Earth's"),
        ("Mean temperature", f"{component.mean_temperature:+.0f} C"),
        ("Surface pressure", (
            f"{component.surface_pressure_bar:,.3g} bar" if component.surface_pressure_bar > 0 else "none (vacuum)"
        )),
        ("Mass", f"{component.mass:.3e} kg"),
    ]
