import math

from dataclasses import dataclass

import numpy as np

from planet.noise import Perlin, fbm


# =========================================================
# Body Shapes
# =========================================================
#
# The surface a body's relief stands on. Big bodies are
# spheres flattened by their spin (Jupiter 6.5%, Earth
# 0.3%); fast spinners stretch into triaxial ellipsoids
# (Haumea); small ones are whatever their collisions left:
# lumpy potatoes (Phobos, Vesta), contact binaries of two
# lobes (comet 67P), giant basins as big as the body.
#
# The shape is a radial function: height (m) above the
# reference radius along each direction from the center, so
# the cube-sphere mesh, the terrain features and every
# other system keep working in directions on the unit
# sphere. Relief (continents, craters, rivers, sea level) is
# measured from this surface, not from the sphere; the seas
# of a flattened planet follow its shape.
#
# Directions are in the body's own frame: y is the rotation
# axis, longitude is atan2(x, z) (planet/solar.py), so +z is
# longitude 0 (for a moon locked to its planet, the side
# facing it).

# Giant impact basins per body (each lat, lon, diameter,
# depth, central peak height).
MAX_BASINS = 4
BASIN_FIELDS = 5

# Large-scale lumps: base frequency (cycles over the unit
# sphere) and octaves.
_LUMP_FREQUENCY = 1.3
_LUMP_OCTAVES = 4

# Contact binaries: how wide the neck between the lobes is
# (smooth union, in units of the radius), and the ray march
# that finds the outer surface along each direction.
_LOBE_BLEND = 0.35
_MARCH_STEPS = 40
_BISECTIONS = 18

# Basin rims and ejecta, relative to the depth.
_RIM = 0.12
_EJECTA = 2.2


@dataclass(frozen=True, slots=True)
class ShapeSettings:

    # Semi-axes in units of the radius (x, y = rotation
    # axis, z = longitude 0).
    axes: tuple[float, float, float] = (1.0, 1.0, 1.0)

    # A second lobe (contact binary): its center and
    # semi-axes in units of the radius; axes 0 = none. The
    # main ellipsoid sits at main_center.
    main_center: tuple[float, float, float] = (0.0, 0.0, 0.0)
    lobe_center: tuple[float, float, float] = (0.0, 0.0, 0.0)
    lobe_axes: tuple[float, float, float] = (0.0, 0.0, 0.0)

    # Large-scale lumps, as a fraction of the radius.
    lumpiness: float = 0.0

    # Giant basins: flattened (lat deg, lon deg, diameter m,
    # depth m, central peak m) groups; diameter 0 = unused.
    basins: tuple[float, ...] = (0.0,) * (MAX_BASINS * BASIN_FIELDS)

    seed: int = 1

    @property
    def has_lobe(
        self
    ) -> bool:

        return min(self.lobe_axes) > 0.0

    @property
    def spherical(
        self
    ) -> bool:

        return (
            tuple(self.axes) == (1.0, 1.0, 1.0)
            and tuple(self.main_center) == (0.0, 0.0, 0.0)
            and not self.has_lobe
            and self.lumpiness <= 0.0
            and not any(basin[2] > 0.0 for basin in unflat_basins(self.basins))
        )


def shape_settings(
    oblateness: float = 0.0,
    axes=(1.0, 1.0, 1.0),
    main_center=(0.0, 0.0, 0.0),
    lobe_center=(0.0, 0.0, 0.0),
    lobe_axes=(0.0, 0.0, 0.0),
    lumpiness: float = 0.0,
    basins=(),
    seed: int = 1
) -> ShapeSettings | None:
    """
    The settings for a body, or None for a plain sphere.
    oblateness flattens the y axis: polar radius = (1 - f)
    x equatorial.
    """

    f = min(max(float(oblateness), 0.0), 0.5)

    axes = tuple(max(float(a), 1e-3) for a in axes)

    settings = ShapeSettings(
        axes=(axes[0], axes[1] * (1.0 - f), axes[2]),
        main_center=tuple(float(v) for v in main_center),
        lobe_center=tuple(float(v) for v in lobe_center),
        lobe_axes=tuple(max(float(a), 0.0) for a in lobe_axes),
        lumpiness=max(float(lumpiness), 0.0),
        basins=flat_basins(unflat_basins(basins)),
        seed=int(seed)
    )

    return None if settings.spherical else settings


def flat_basins(
    basins
) -> tuple[float, ...]:
    """(lat, lon, diameter, depth, peak) groups -> MAX_BASINS slots."""

    values: list[float] = []

    for basin in list(basins)[:MAX_BASINS]:
        values.extend(float(v) for v in basin)

    values.extend([0.0] * (MAX_BASINS * BASIN_FIELDS - len(values)))

    return tuple(values)


def unflat_basins(
    values
) -> list[tuple[float, ...]]:

    values = list(values)

    return [
        tuple(values[i:i + BASIN_FIELDS])
        for i in range(0, len(values) - BASIN_FIELDS + 1, BASIN_FIELDS)
        if values[i + 2] > 0.0
    ]


def direction_of(
    latitude_deg: float,
    longitude_deg: float
) -> np.ndarray:
    """Unit direction in the body frame (longitude = atan2(x, z))."""

    latitude = math.radians(latitude_deg)
    longitude = math.radians(longitude_deg)

    return np.array([
        math.cos(latitude) * math.sin(longitude),
        math.sin(latitude),
        math.cos(latitude) * math.cos(longitude),
    ])


class BodyShape:

    def __init__(
        self,
        settings: ShapeSettings,
        radius: float
    ):

        self.settings = settings
        self.radius = float(radius)

        self._axes = np.asarray(settings.axes, dtype=np.float64)
        self._center = np.asarray(settings.main_center, dtype=np.float64)

        self._lobe_axes = np.asarray(settings.lobe_axes, dtype=np.float64)
        self._lobe_center = np.asarray(settings.lobe_center, dtype=np.float64)

        self._lumps = Perlin(settings.seed * 7919 + 31) if settings.lumpiness > 0.0 else None

        self._basins = [
            (
                direction_of(latitude, longitude),
                0.5 * diameter / self.radius,
                depth,
                peak
            )
            for latitude, longitude, diameter, depth, peak in unflat_basins(settings.basins)
        ]

        # Farthest any lobe reaches (units of the radius),
        # where the ray march starts.
        self._reach = 1.05 * max(
            float(np.linalg.norm(self._center) + self._axes.max()),
            float(np.linalg.norm(self._lobe_center) + self._lobe_axes.max())
            if settings.has_lobe else 0.0
        )

        self.min_height, self.max_height = self._bounds()

    # -----------------------------------------------------
    # Height
    # -----------------------------------------------------

    def height(
        self,
        directions: np.ndarray
    ) -> np.ndarray:
        """Height (m) of the shape above the reference radius."""

        directions = np.asarray(directions, dtype=np.float64)

        r = self._lobes(directions)

        if self._lumps is not None:

            r = r * (
                1.0
                + self.settings.lumpiness
                * fbm(self._lumps, directions * _LUMP_FREQUENCY, _LUMP_OCTAVES)
            )

        height = (r - 1.0) * self.radius

        for center, angular_radius, depth, peak in self._basins:
            height = height + _basin(directions @ center, angular_radius, depth, peak)

        return height

    def _lobes(
        self,
        directions: np.ndarray
    ) -> np.ndarray:
        """Distance to the lobes' outer surface (units of the radius)."""

        if not self.settings.has_lobe:
            return _ellipsoid_far(directions, self._center, self._axes)

        # Smooth union of the two ellipsoids: march in from
        # outside to the first point inside, then bisect.
        count = len(directions)

        step = self._reach / _MARCH_STEPS

        t = np.full(count, self._reach)

        found = np.zeros(count, dtype=bool)

        outside = np.full(count, self._reach)

        for _ in range(_MARCH_STEPS):

            t = t - step

            inside = (self._field(directions * t[:, None]) < 0.0) & ~found

            found |= inside

            outside = np.where(found, outside, t)

        hi = outside
        lo = np.maximum(outside - step, 0.0)

        for _ in range(_BISECTIONS):

            mid = 0.5 * (lo + hi)

            inside = self._field(directions * mid[:, None]) < 0.0

            lo = np.where(inside, mid, lo)
            hi = np.where(inside, hi, mid)

        # Nothing found (a direction through a gap): the
        # main ellipsoid alone.
        return np.where(found, 0.5 * (lo + hi), _ellipsoid_far(directions, self._center, self._axes))

    def _field(
        self,
        points: np.ndarray
    ) -> np.ndarray:
        """< 0 inside the body (smooth union of the lobes)."""

        a = _ellipsoid_field(points, self._center, self._axes)
        b = _ellipsoid_field(points, self._lobe_center, self._lobe_axes)

        h = np.maximum(_LOBE_BLEND - np.abs(a - b), 0.0) / _LOBE_BLEND

        return np.minimum(a, b) - h * h * _LOBE_BLEND * 0.25

    # -----------------------------------------------------
    # Bounds
    # -----------------------------------------------------

    def _bounds(
        self
    ) -> tuple[float, float]:
        """Lowest and highest heights, with a margin."""

        heights = self.height(fibonacci_directions(4096))

        low = float(heights.min())
        high = float(heights.max())

        margin = 0.03 * (high - low) + 0.01 * self.radius * self.settings.lumpiness

        return low - margin, high + margin

    @property
    def extent(
        self
    ) -> float:
        """Farthest point of the surface from the center (m)."""

        return self.radius + self.max_height

    @property
    def spheroid(
        self
    ) -> bool:
        """
        A sphere flattened along its axis (and nothing else):
        stretching y by 1 / (1 - flattening) makes it a
        sphere again, as the atmosphere does.
        """

        s = self.settings

        return (
            abs(s.axes[0] - s.axes[2]) < 1e-9
            and tuple(s.main_center) == (0.0, 0.0, 0.0)
            and not s.has_lobe
            and s.lumpiness <= 0.0
            and not self._basins
        )


def _ellipsoid_field(
    points: np.ndarray,
    center: np.ndarray,
    axes: np.ndarray
) -> np.ndarray:
    """Approximate signed distance (units of the radius)."""

    return (np.linalg.norm((points - center) / axes, axis=1) - 1.0) * float(axes.min())


def _ellipsoid_far(
    directions: np.ndarray,
    center: np.ndarray,
    axes: np.ndarray
) -> np.ndarray:
    """
    Distance from the origin to where each ray leaves the
    ellipsoid (the far root of |(t d - c) / a| = 1).
    """

    d = directions / axes
    c = center / axes

    dd = np.einsum("ij,ij->i", d, d)
    dc = d @ c
    cc = float(c @ c)

    discriminant = np.maximum(dc * dc - dd * (cc - 1.0), 0.0)

    return np.maximum((dc + np.sqrt(discriminant)) / dd, 1e-3)


def _basin(
    cosine: np.ndarray,
    angular_radius: float,
    depth: float,
    peak: float
) -> np.ndarray:
    """
    A giant impact basin (m): a bowl, a low rim, ejecta, and
    a central peak (Vesta's Rheasilvia: 505 km across, its
    peak 20 km high).
    """

    x = np.arccos(np.clip(cosine, -1.0, 1.0)) / max(angular_radius, 1e-6)

    rim = _RIM * depth

    bowl = -depth + (depth + rim) * x * x

    central = peak * np.exp(-(x / 0.18) ** 2)

    tail = _EJECTA ** -3.0

    ejecta = rim * (np.maximum(x, 1.0) ** -3.0 - tail) / (1.0 - tail)

    return np.where(x < 1.0, bowl + central, np.where(x < _EJECTA, ejecta, 0.0))


def fibonacci_directions(
    count: int
) -> np.ndarray:
    """Nearly even unit directions over the sphere."""

    i = np.arange(count) + 0.5

    y = 1.0 - 2.0 * i / count

    ring = np.sqrt(np.maximum(1.0 - y * y, 0.0))

    angle = i * math.pi * (3.0 - math.sqrt(5.0))

    return np.stack((ring * np.sin(angle), y, ring * np.cos(angle)), axis=1)
