// =========================================================
// Planet Terrain Surface
// =========================================================
//
// Biome colors for planet chunks, evaluated per pixel from
// smoothly interpolated per-vertex inputs (planet/chunk.py):
//
//     aColor.r    elevation (m above sea level, negative
//                 under water)
//     aColor.g    slope: dot(normal, radial up), 1 = flat
//     aColor.b    precipitation (mm / year)
//     aTexCoord.x annual mean temperature (C) at this
//                 point's own height
//
// Temperature and rainfall come from the climate model
// (planet/climate.py), or from latitude and noise without
// one. Biomes follow a Whittaker diagram: how wet a place
// is depends on rain relative to how warm it is (warm air
// dries the ground faster).
//
// Thresholding per pixel keeps coastlines, tree lines and
// snow lines smooth even on coarse chunks whose vertices
// are hundreds of kilometers apart.

struct TerrainSurface
{
    vec3 albedo;
    float roughness;
    vec3 emissive;      // glowing lava

    // 0..1: how much micro-relief shading applies (none on
    // water, little on snow).
    float relief;

    // What the ground is made of, for the close-up detail
    // (terrainCloseUp): x bare rock, y sand / dust / soil,
    // z plants, w snow and ice.
    vec4 material;
};


// ---------------------------------------------------------
// Close-up Detail
// ---------------------------------------------------------
//
// The chunks' vertices carry the terrain's shape and
// climate; between them, rock, soil and plants still vary.
// Fractal value noise in planet-local coordinates (km)
// adds that per pixel: brightness and hue variation, and a
// micro-relief height for bump shading. Octaves run from
// ~8 km down to ~20 m and fade out once smaller than a
// pixel, so distant ground does not shimmer.
//
// Cells are hashed as integers: exact at planet scale
// (a float hash of coordinates ~1e5 would lose its bits).

struct TerrainDetail
{
    float albedo;       // ~-1 .. 1
    float height;       // meters
};

float terrainHash(
    ivec3 cell
)
{
    uvec3 v = uvec3(cell) * uvec3(1597334673u, 3812015801u, 2798796415u);

    uint n = (v.x ^ v.y ^ v.z) * 1597334673u;

    n ^= n >> 16u;
    n *= 2246822519u;
    n ^= n >> 13u;

    return float(n) * (1.0 / 4294967295.0);
}

float terrainValueNoise(
    vec3 p
)
{
    vec3 cellFloor = floor(p);
    ivec3 i = ivec3(cellFloor);
    vec3 f = p - cellFloor;

    f = f * f * (3.0 - 2.0 * f);

    float a = mix(terrainHash(i), terrainHash(i + ivec3(1, 0, 0)), f.x);
    float b = mix(terrainHash(i + ivec3(0, 1, 0)), terrainHash(i + ivec3(1, 1, 0)), f.x);
    float c = mix(terrainHash(i + ivec3(0, 0, 1)), terrainHash(i + ivec3(1, 0, 1)), f.x);
    float d = mix(terrainHash(i + ivec3(0, 1, 1)), terrainHash(i + ivec3(1, 1, 1)), f.x);

    return mix(mix(a, b, f.y), mix(c, d, f.y), f.z) * 2.0 - 1.0;
}

// The detail origin (systems/planet_system.py): a planet-
// fixed grid point near the camera, whose grid cell is
// counted in exactly. Float32 positions from a planet's
// center blur below ~0.1-0.5 m (the noise then bands into
// moire rings up close); measured from this origin they are
// exact, and the cells match the far path's.
uniform vec4 uDetailOrigin;     // camera-relative m; w 1 = in use (the camera's body)
uniform vec3 uDetailCell;       // the origin's grid cell (body frame, 1024 m)

const float DETAIL_CELL = 1024.0;

// Each octave its own pattern.
ivec3 detailSalt(
    int salt
)
{
    return ivec3(salt * 7919, salt * 104729, salt * 15485863);
}

// And its own lattice position: value noise is flat across
// its lattice planes, and octaves a power of two apart would
// share theirs (a grid of creases in the lighting). A
// fraction of a cell, the same near and far.
vec3 detailShift(
    int salt
)
{
    return fract(vec3(0.3719, 0.6113, 0.1307) * float(salt + 1) + vec3(0.17, 0.53, 0.89));
}

// A strong hash for the detail noise's cells (PCG3D, Jarzynski
// & Olano 2020): exact cell indices run to ~1e7 up close,
// where simpler hashes show their structure (moire rings,
// lines).
float detailHash(
    ivec3 cell
)
{
    uvec3 v = uvec3(cell) * 1664525u + 1013904223u;

    v.x += v.y * v.z;
    v.y += v.z * v.x;
    v.z += v.x * v.y;

    v ^= v >> 16u;

    v.x += v.y * v.z;
    v.y += v.z * v.x;
    v.z += v.x * v.y;

    return float(v.x) * (1.0 / 4294967295.0);
}

// Gradient noise in a cell (i: its corner, f: the position
// in it). Value noise would be flat across every lattice
// plane (its interpolation has zero slope there): bump
// lighting shows those planes as lines wherever one octave
// dominates. Gradient noise has random slopes everywhere.
vec3 detailGradient(
    ivec3 cell
)
{
    float h = detailHash(cell) * 6.2831853;
    float z = detailHash(cell + ivec3(7, 13, 29)) * 2.0 - 1.0;

    float r = sqrt(max(1.0 - z * z, 0.0));

    return vec3(r * cos(h), r * sin(h), z);
}

float valueNoiseCell(
    ivec3 i,
    vec3 f
)
{
    vec3 u = f * f * f * (f * (f * 6.0 - 15.0) + 10.0);

    float a = dot(detailGradient(i), f);
    float b = dot(detailGradient(i + ivec3(1, 0, 0)), f - vec3(1.0, 0.0, 0.0));
    float c = dot(detailGradient(i + ivec3(0, 1, 0)), f - vec3(0.0, 1.0, 0.0));
    float d = dot(detailGradient(i + ivec3(1, 1, 0)), f - vec3(1.0, 1.0, 0.0));
    float e = dot(detailGradient(i + ivec3(0, 0, 1)), f - vec3(0.0, 0.0, 1.0));
    float g = dot(detailGradient(i + ivec3(1, 0, 1)), f - vec3(1.0, 0.0, 1.0));
    float h = dot(detailGradient(i + ivec3(0, 1, 1)), f - vec3(0.0, 1.0, 1.0));
    float k = dot(detailGradient(i + ivec3(1, 1, 1)), f - vec3(1.0, 1.0, 1.0));

    // (Scaled to about -1 .. 1, as the value noise was.)
    return 1.6 * mix(mix(mix(a, b, u.x), mix(c, d, u.x), u.y), mix(mix(e, g, u.x), mix(h, k, u.x), u.y), u.z);
}

// Gradient noise with cells `wavelength` m wide (a power of
// two), at q (m from the detail origin): exact at any
// distance from the planet's center.
float closeUpNoise(
    vec3 q,
    float wavelength,
    int salt
)
{
    // The origin's offset in cells, split into whole cells
    // and a fraction (both exact: the grid is a power of two
    // meters), so the position inside stays small and exact.
    vec3 offset = uDetailCell * (DETAIL_CELL / wavelength);

    vec3 whole = floor(offset);

    vec3 p = q / wavelength + (offset - whole) + detailShift(salt);

    vec3 cellFloor = floor(p);

    ivec3 i = ivec3(cellFloor) + ivec3(whole) + detailSalt(salt);

    return valueNoiseCell(i, p - cellFloor);
}

// The same noise far away (no origin): position in km.
float farDetailNoise(
    vec3 positionKm,
    float wavelength,   // m
    int salt
)
{
    vec3 p = positionKm * (1000.0 / wavelength) + detailShift(salt);

    vec3 cellFloor = floor(p);

    return valueNoiseCell(ivec3(cellFloor) + detailSalt(salt), p - cellFloor);
}

TerrainDetail terrainDetail(
    vec3 position,      // planet-local, km (far path)
    vec3 q,             // planet frame, m from the detail origin (near path)
    bool near,
    float footprint     // km per pixel
)
{
    TerrainDetail detail;
    detail.albedo = 0.0;
    detail.height = 0.0;

    float wavelength = 8192.0;  // m
    float weight = 0.5;
    float total = 0.0;

    for (int octave = 0; octave < 9; ++octave)
    {
        // Fade octaves that shrink below ~3 pixels; their
        // relief sooner, below ~8 (bump mapping works on 2 x 2
        // pixel blocks: finer relief shows as a checker).
        float pixels = wavelength / max(footprint * 1000.0, 1e-6);

        float fade = smoothstep(2.0, 4.0, pixels);
        float reliefFade = smoothstep(6.0, 12.0, pixels);

        if (fade <= 0.0)
        {
            break;
        }

        float n = near
            ? closeUpNoise(q, wavelength, octave)
            : farDetailNoise(position, wavelength, octave);

        detail.albedo += weight * fade * n;

        // Fractal relief: amplitude in step with wavelength
        // (slopes of ~10-20 degrees at every scale).
        detail.height += 0.06 * wavelength * reliefFade * n;

        total += weight;
        weight *= 0.62;
        wavelength *= 0.5;
    }

    detail.albedo /= max(total, 1e-3);

    return detail;
}

// ---------------------------------------------------------
// Close-up Ground (below ~20 m)
// ---------------------------------------------------------
//
// Under the chunks' finest vertices the ground still has
// texture all the way down: each material its own, from
// ~16 m to ~5 cm, fading as it shrinks below a pixel:
//
//   rock     cracked and blocky (ridged noise), with strata
//            on cliffs: layers a few meters thick
//   sand     smooth, and with air to blow it, ripples ~20 cm
//            apart across the wind (dust and regolith
//            without air: just grain)
//   plants   patchy: tufts, bare spots, shades of green
//   snow     smooth, faint wind-carved ridges (sastrugi)
//
// Positions: planet-frame meters from uDetailOrigin, a
// planet-fixed grid point near the camera (systems/
// planet_system.py), whose cell uDetailCell is counted in
// exactly; float32 positions from the planet's center
// would blur below ~0.5 m.

uniform float uSurfaceWind;     // 1 = air to make sand ripples

TerrainDetail terrainCloseUp(
    vec3 q,             // planet frame, m from uDetailOrigin
    vec3 up,            // planet frame, unit
    float footprint,    // m per pixel
    vec4 material,      // rock, sand, plants, snow
    float steep         // 0 flat .. 1 cliff
)
{
    TerrainDetail detail;
    detail.albedo = 0.0;
    detail.height = 0.0;

    float wavelength = 16.0;

    for (int octave = 0; octave < 9; ++octave)
    {
        float pixels = wavelength / max(footprint, 1e-6);

        float fade = smoothstep(2.0, 4.0, pixels);
        float reliefFade = smoothstep(6.0, 12.0, pixels);

        if (fade <= 0.0)
        {
            break;
        }

        float n = closeUpNoise(q, wavelength, 20 + octave);

        // Rock: cracks and blocks.
        float ridge = 1.0 - abs(n);

        detail.height += reliefFade * wavelength * (
            material.x * 0.10 * (ridge - 0.5)
            + material.y * 0.012 * n
            + material.z * 0.05 * n
            + material.w * 0.015 * n
        );

        detail.albedo += fade * (
            material.x * 0.35 * (ridge - 0.5)
            + material.y * 0.15 * n
            + material.z * 0.45 * n
            + material.w * 0.04 * n
        ) * (octave < 3 ? 1.0 : 0.6);

        wavelength *= 0.5;
    }

    // Strata on cliffs: layers of a few meters, wavy.
    float stratumFade = 1.0 - smoothstep(0.3, 0.6, footprint);

    if (material.x * steep > 0.01 && stratumFade > 0.0)
    {
        float level = dot(q, up) + 1.5 * closeUpNoise(q, 32.0, 40);

        float layer = fract(level / 3.0);

        float band = smoothstep(0.0, 0.15, layer) * (1.0 - smoothstep(0.8, 1.0, layer));

        detail.albedo += material.x * steep * stratumFade * (0.25 * band - 0.15 + 0.15 * closeUpNoise(vec3(0.0, level, 0.0), 1.0, 41));
        detail.height += material.x * steep * stratumFade * 0.25 * band;
    }

    // Ripples across the wind, on sand with air.
    // (Faded well before they alias: 20 cm waves need ~4
    // pixels each.)
    float rippleFade = 1.0 - smoothstep(0.012, 0.025, footprint);

    if (uSurfaceWind > 0.5 && material.y > 0.01 && rippleFade > 0.0)
    {
        // The wind's direction wanders over ~100 m.
        float angle = 3.0 * closeUpNoise(q, 128.0, 42);

        vec3 east = normalize(cross(up, vec3(0.31, 0.83, 0.47)));
        vec3 north = cross(up, east);

        vec3 wind = cos(angle) * east + sin(angle) * north;

        float wave = sin(6.2831853 * dot(q, wind) / 0.2 + 2.0 * closeUpNoise(q, 1.0, 43));

        // Asymmetric: gentle upwind, steep lee.
        wave = wave > 0.0 ? wave : wave * 0.5;

        detail.height += material.y * rippleFade * 0.01 * wave;
        detail.albedo += material.y * rippleFade * 0.035 * wave;
    }

    return detail;
}

// Bump-mapped normal from a height field, by screen-space
// derivatives (Mikkelsen, "Bump Mapping Unparametrized
// Surfaces on the GPU", 2010): no tangents needed.
vec3 terrainBumpNormal(
    vec3 normal,
    vec3 position,      // camera-relative, m
    float height        // m
)
{
    vec3 dpdx = dFdx(position);
    vec3 dpdy = dFdy(position);

    float dhdx = dFdx(height);
    float dhdy = dFdy(height);

    vec3 r1 = cross(dpdy, normal);
    vec3 r2 = cross(normal, dpdx);

    float det = dot(dpdx, r1);

    vec3 gradient = sign(det) * (dhdx * r1 + dhdy * r2);

    // No tilt past ~60 degrees: where a pixel's derivatives
    // straddle a triangle edge at a grazing view, the
    // estimate can blow up (single dark pixels along edges).
    float limit = 1.7 * abs(det);

    float size = length(gradient);

    if (size > limit)
    {
        gradient *= limit / size;
    }

    return normalize(abs(det) * normal - gradient);
}

// Per planet (its material; planet/bodies.py profiles):
uniform float uSurfacePalette;  // 0 biomes, 1 mineral, 2 cloud bands
uniform float uLiquid;          // 0 none, 1 water, 2 methane, 3 lava
uniform vec3 uColorLow;         // mineral: basins / bands: belts
uniform vec3 uColorHigh;        // mineral: highlands / bands: zones
uniform vec3 uColorSteep;       // mineral: cliffs
uniform vec3 uColorIce;         // mineral: frost and ice caps
uniform float uFrostPoint;      // C: below this the ground frosts over
uniform float uLiquidFreezing;  // C: below this the seas ice over
uniform float uLife;            // 1 = vegetation (biomes palette)

// Giants (cloud bands palette):
uniform float uBandCount;       // belts + zones
uniform vec4 uStorm;            // x latitude, y longitude (rad), z east-west
                                // half size (rad), w strength (0 = none)
uniform vec3 uStormColor;
uniform float uOvals;           // how many small white ovals
uniform float uPolarHexagon;    // 1 = Saturn's north polar hexagon

// Linear-space albedos.
const vec3 TERRAIN_SAND = vec3(0.42, 0.36, 0.22);
const vec3 TERRAIN_HOT_DESERT = vec3(0.55, 0.42, 0.25);
const vec3 TERRAIN_COLD_DESERT = vec3(0.30, 0.28, 0.22);
const vec3 TERRAIN_DRY_GRASS = vec3(0.28, 0.24, 0.10);
const vec3 TERRAIN_GRASS = vec3(0.07, 0.16, 0.04);
const vec3 TERRAIN_FOREST = vec3(0.03, 0.08, 0.025);
const vec3 TERRAIN_RAINFOREST = vec3(0.012, 0.05, 0.012);
const vec3 TERRAIN_TAIGA = vec3(0.02, 0.045, 0.03);
const vec3 TERRAIN_TUNDRA = vec3(0.17, 0.16, 0.10);
const vec3 TERRAIN_ROCK = vec3(0.16, 0.14, 0.12);
const vec3 TERRAIN_SNOW = vec3(0.80, 0.82, 0.86);
const vec3 TERRAIN_SHALLOW_WATER = vec3(0.02, 0.10, 0.14);
const vec3 TERRAIN_DEEP_WATER = vec3(0.004, 0.015, 0.05);
const vec3 TERRAIN_ICE = vec3(0.65, 0.72, 0.78);
const vec3 TERRAIN_DAMP_SOIL = vec3(0.13, 0.11, 0.08);

// Rain relative to what the warmth evaporates: < 0.3
// desert, ~0.5-1 grassland / savanna, > 1.2 forest.
float terrainWetness(
    float precipitation,
    float temperature
)
{
    return precipitation / (300.0 + 30.0 * max(temperature, 0.0));
}

// ---------------------------------------------------------
// Giant Planet Weather
// ---------------------------------------------------------
//
// Cloud bands per pixel, in the planet's own frame:
//
//   belts and zones   bands of irregular width from latitude
//                     (the jet streams between them run east
//                     and west)
//   turbulence        noise stretched east-west and sheared
//                     where the jets meet, warping the bands
//                     into festoons and streaks
//   the great storm   an oval vortex that swirls the bands
//                     around it (Jupiter's Great Red Spot,
//                     Neptune's Great Dark Spot)
//   white ovals       small storms strung along the bands
//   polar hexagon     Saturn's north polar jet, six-sided

// Rotate `v` about the unit axis `axis` by `angle`.
vec3 giantRotate(
    vec3 v,
    vec3 axis,
    float angle
)
{
    float c = cos(angle);
    float s = sin(angle);

    return v * c + cross(axis, v) * s + axis * dot(axis, v) * (1.0 - c);
}

vec3 giantFromLatLon(
    float latitude,
    float longitude
)
{
    return vec3(cos(latitude) * cos(longitude), sin(latitude), cos(latitude) * sin(longitude));
}

// Fractal noise stretched east-west (y is the axis), with
// octaves fading below the pixel footprint (radians).
float giantStreaks(
    vec3 direction,
    float scale,
    float stretch,
    float footprint,
    int octaves
)
{
    vec3 p = vec3(direction.x, direction.y * stretch, direction.z) * scale;

    float sum = 0.0;
    float weight = 0.5;
    float total = 0.0;
    float frequency = 1.0;

    for (int octave = 0; octave < octaves; ++octave)
    {
        // Fade octaves smaller than a few pixels (across the
        // bands they are `stretch` times finer).
        float fade = 1.0 - smoothstep(0.15, 0.35, footprint * scale * stretch * frequency);

        if (fade <= 0.0)
        {
            break;
        }

        sum += weight * fade * terrainValueNoise(p * frequency + float(octave) * 7.31);

        total += weight;
        weight *= 0.55;
        frequency *= 2.1;
    }

    return sum / max(total, 1e-3);
}

vec3 giantBands(
    vec3 direction,     // unit, the planet's frame (y = axis)
    float footprint     // radians per pixel
)
{
    vec3 d = direction;

    // ----- The great storm: swirl the bands around it -----
    float stormMask = 0.0;
    float stormSwirl = 0.0;

    if (uStorm.w > 0.0)
    {
        vec3 center = giantFromLatLon(uStorm.x, uStorm.y);

        // East-west it is wider than north-south (an oval).
        vec3 east = normalize(cross(vec3(0.0, 1.0, 0.0), center));
        vec3 north = cross(center, east);

        vec3 offset = d - center;

        float x = dot(offset, east) / uStorm.z;
        float y = dot(offset, north) / (uStorm.z * 0.6);

        float r = length(vec2(x, y));

        // Anticyclone: spins fastest just inside its rim.
        float spin = uStorm.w * 4.0 * exp(-r * r * 1.2) * (dot(center, d) > 0.0 ? 1.0 : 0.0);

        d = giantRotate(d, center, spin);

        stormMask = (1.0 - smoothstep(0.75, 1.05, r)) * uStorm.w;
        stormSwirl = r;
    }

    float latitude = asin(clamp(d.y, -1.0, 1.0));

    // ----- Turbulence along the jets -----
    float large = giantStreaks(d, 5.0, 5.0, footprint, 6);
    float fine = giantStreaks(d, 30.0, 10.0, footprint, 6);

    float band = latitude;

    // Saturn's hexagon: near the north pole the band
    // coordinate follows a six-sided polar distance.
    if (uPolarHexagon > 0.5 && d.y > 0.85)
    {
        float angle = atan(d.z, d.x);

        float sector = 3.14159265 / 3.0;

        float local = mod(angle + sector * 0.5, sector) - sector * 0.5;

        float polar = acos(clamp(d.y, -1.0, 1.0)) / cos(local);

        float hexagonal = 1.5707963 - polar;

        band = mix(band, hexagonal, smoothstep(0.85, 0.93, d.y));
    }

    // Warp the bands; more where the jets shear (belt edges).
    float n = uBandCount;

    float edge = abs(cos(band * n));

    // Festoons and swirls where the jets meet.
    band += (1.6 * large + 0.4 * fine) * (0.3 + 0.7 * edge) / max(n, 1.0);

    // Irregular widths: slower and faster harmonics.
    float phase = band * n + 0.9 * sin(band * n * 0.5 + 1.3) + 0.45 * sin(band * n * 1.7 + 0.4);

    float zone = smoothstep(-0.45, 0.45, sin(phase));

    vec3 color = mix(uColorLow, uColorHigh, zone);

    // Each band its own shade.
    color *= 0.86 + 0.28 * (0.5 + 0.5 * sin(band * n * 0.73 + 2.1) * sin(band * n * 0.31 + 0.7));

    // Streaks of brighter and darker cloud along the bands.
    color *= 0.88 + 0.24 * (0.5 + 0.5 * fine);

    // Polar regions: duskier, less banded.
    float polar = smoothstep(0.75, 0.97, abs(d.y));

    color = mix(color, mix(uColorLow, uColorHigh, 0.35) * 0.8, polar * 0.6);

    // ----- White ovals -----
    if (uOvals > 0.0)
    {
        float cellLatitude = 0.09;          // ~5 degrees
        float cellLongitude = 0.16;

        float longitude = atan(d.z, d.x);

        vec2 cell = vec2(floor(longitude / cellLongitude), floor(latitude / cellLatitude));

        ivec3 key = ivec3(int(cell.x) + 1000, int(cell.y) + 1000, 17);

        // Only along a few latitudes, a few per row.
        bool row = terrainHash(ivec3(0, int(cell.y) + 1000, 5)) < 0.3;

        if (row && terrainHash(key) < uOvals * 0.12 && abs(latitude) < 1.1)
        {
            vec2 jitter = vec2(terrainHash(key + ivec3(0, 0, 1)), terrainHash(key + ivec3(0, 0, 2)));

            vec2 centerCell = (cell + 0.25 + 0.5 * jitter) * vec2(cellLongitude, cellLatitude);

            vec2 offset = vec2(
                (longitude - centerCell.x) * cos(latitude) / (cellLatitude * 0.32),
                (latitude - centerCell.y) / (cellLatitude * 0.2)
            );

            float oval = 1.0 - smoothstep(0.3, 1.0, length(offset));

            color = mix(color, vec3(0.92, 0.9, 0.86), oval * oval * 0.8);
        }
    }

    // ----- The storm's own color -----
    if (stormMask > 0.0)
    {
        float swirl = 0.5 + 0.5 * fine;

        vec3 storm = uStormColor * (0.85 + 0.3 * swirl) * (0.8 + 0.2 * smoothstep(0.0, 0.7, stormSwirl));

        color = mix(color, storm, stormMask);
    }

    return color;
}

vec3 mineralLand(
    float elevation,
    float slope,
    float temperature,
    float crust
)
{
    // Basins dark, highlands light (the Moon's maria and
    // highlands), cliffs their own color, frost where cold.
    float bright = smoothstep(-3000.0, 4000.0, elevation);

    // With a tectonic regime (planet/regimes.py), mostly
    // its crust type: dark maria and lava, Europa's
    // reddish lineae, bright ice and sulfur plains.
    if (crust >= 0.0)
    {
        bright = mix(bright, crust, 0.75);
    }

    // Past 1 (crater rays): brighter than the highlands.
    vec3 land = mix(uColorLow, uColorHigh, min(bright, 1.0))
        * (1.0 + 0.6 * max(bright - 1.0, 0.0));

    land = mix(land, uColorSteep, 1.0 - smoothstep(0.75, 0.9, slope));

    float frost =
        smoothstep(uFrostPoint + 2.0, uFrostPoint - 4.0, temperature)
        * smoothstep(0.7, 0.82, slope);

    return mix(land, uColorIce, frost);
}

// water: 0..1 rivers, lakes and delta channels (planet/
// hydrology.py), drawn as the planet's liquid, or as ice
// (glaciers) where it is colder than the liquid freezes.
// crust: 0 dark .. 1 bright crust from a tectonic regime
// (the tectonic data's "continental" channel; up to 1.6
// on fresh crater rays), -1 without.
TerrainSurface terrainSurface(
    float elevation,
    float slope,
    float precipitation,
    float temperature,
    float crust,
    float water,
    float detail,
    vec3 planetDirection,   // unit, the planet's own frame
    float footprint         // radians per pixel there
)
{
    TerrainSurface surface;

    surface.emissive = vec3(0.0);
    surface.relief = 0.0;
    surface.material = vec4(0.0);

    int palette = int(uSurfacePalette + 0.5);
    int liquid = int(uLiquid + 0.5);

    if (palette == 2)
    {
        // Giant planet weather, per pixel.
        surface.albedo = giantBands(planetDirection, footprint);
        surface.roughness = 1.0;
        surface.material = vec4(0.0);

        return surface;
    }

    float wetness = terrainWetness(precipitation, temperature);

    // 1 in the tropics; 1 in the boreal zone.
    float tropical = smoothstep(18.0, 24.0, temperature);
    float boreal = smoothstep(8.0, 2.0, temperature);

    // -----------------------------------------------------
    // Land: desert -> grassland -> forest by wetness,
    // flavored by warmth
    // -----------------------------------------------------

    vec3 land;

    // Cliffs: bare rock everywhere.
    float cliff = 1.0 - smoothstep(0.75, 0.88, slope);

    if (palette == 1)
    {
        land = mineralLand(elevation, slope, temperature, crust);

        float frost =
            smoothstep(uFrostPoint + 2.0, uFrostPoint - 4.0, temperature)
            * smoothstep(0.7, 0.82, slope);

        float rock = 1.0 - smoothstep(0.75, 0.9, slope);

        surface.material = vec4(rock, (1.0 - rock) * (1.0 - frost), 0.0, frost);
    }
    else
    {
        vec3 desert = mix(TERRAIN_HOT_DESERT, TERRAIN_COLD_DESERT, boreal);

        vec3 grass = mix(TERRAIN_DRY_GRASS, TERRAIN_GRASS, smoothstep(0.5, 1.0, wetness));

        vec3 forest = mix(TERRAIN_FOREST, TERRAIN_RAINFOREST, tropical);

        forest = mix(forest, TERRAIN_TAIGA, boreal);

        vec3 tundra = TERRAIN_TUNDRA;

        // Lifeless: the same climate zones as bare ground,
        // darker where it rains.
        if (uLife < 0.5)
        {
            grass = mix(desert * 0.85, TERRAIN_DAMP_SOIL, smoothstep(0.5, 1.0, wetness));
            forest = TERRAIN_DAMP_SOIL * 0.8;
            tundra = TERRAIN_COLD_DESERT;
        }

        land = mix(desert, grass, smoothstep(0.15, 0.45, wetness));

        land = mix(land, forest, smoothstep(0.9, 1.6, wetness) * smoothstep(0.9, 0.96, slope));

        // Too cold for trees: tundra (and the alpine zone above
        // the tree line, since temperature follows height).
        land = mix(land, tundra, smoothstep(1.0, -3.0, temperature));

        // Beaches.
        land = mix(TERRAIN_SAND, land, smoothstep(5.0, 60.0, elevation));

        // Cliffs are bare rock, as are cold high slopes.
        land = mix(land, TERRAIN_ROCK, 1.0 - smoothstep(0.75, 0.88, slope));

        land = mix(land, TERRAIN_ROCK, smoothstep(-3.0, -7.0, temperature) * 0.6);

        // Lasting snow and ice where the year averages below
        // the frost point; snow does not stick to cliffs.
        float snow =
            smoothstep(uFrostPoint - 0.1, uFrostPoint - 5.1, temperature)
            * smoothstep(0.7, 0.82, slope);

        land = mix(land, TERRAIN_SNOW, snow);

        // What it is made of.
        float plants = uLife > 0.5 ? smoothstep(0.15, 0.45, wetness) * (1.0 - smoothstep(1.0, -3.0, temperature)) : 0.0;

        float rock = max(cliff, smoothstep(-3.0, -7.0, temperature) * 0.6);

        float sand = (1.0 - smoothstep(0.15, 0.45, wetness)) + (1.0 - smoothstep(5.0, 60.0, elevation));

        surface.material = vec4(
            rock,
            clamp(sand, 0.0, 1.0) * (1.0 - rock),
            plants * (1.0 - rock),
            0.0
        );

        surface.material *= 1.0 - snow;
        surface.material.w = snow;

        // Bare ground where nothing grows.
        if (uLife < 0.5)
        {
            surface.material.y = max(surface.material.y, (1.0 - rock) * (1.0 - snow));
        }
    }

    // Close-up variation: patchy soil, rock and plants
    // (warmer and lighter, cooler and darker), subtler on
    // snow.
    float snowy = smoothstep(0.55, 0.75, dot(land, vec3(0.33)));

    land *= 1.0 + detail * mix(vec3(0.24, 0.22, 0.17), vec3(0.05), snowy);

    float landRelief = mix(1.0, 0.4, snowy);

    // -----------------------------------------------------
    // Rivers, lakes and glaciers on land
    // -----------------------------------------------------

    float landRoughness = 0.9;

    if (water > 0.01 && liquid != 0 && liquid != 3)
    {
        // Crisp banks from the interpolated mask.
        water = smoothstep(0.35, 0.65, water);

        float frozen = smoothstep(uLiquidFreezing + 2.4, uLiquidFreezing - 0.6, temperature);

        vec3 river = liquid == 2
            ? vec3(0.02, 0.016, 0.01)       // methane: dark and glassy
            : TERRAIN_SHALLOW_WATER;

        river = mix(river, TERRAIN_SNOW * 0.92, frozen);

        land = mix(land, river, water);

        landRoughness = mix(0.9, mix(0.08, 0.5, frozen), water);

        landRelief *= 1.0 - water;

        surface.material *= 1.0 - water;
    }

    // -----------------------------------------------------
    // Liquid below sea level
    // -----------------------------------------------------

    if (liquid == 0)
    {
        // Dry world: basins are just low ground.
        surface.albedo = land;
        surface.roughness = landRoughness;
        surface.relief = landRelief;

        return surface;
    }

    float depth = clamp(-elevation / 3000.0, 0.0, 1.0);

    // Seas ice over where the sea surface averages below
    // the liquid's freezing point (sea water about -2 C).
    float frozen = smoothstep(uLiquidFreezing + 0.4, uLiquidFreezing - 3.1, temperature);

    vec3 sea;
    float seaRoughness;
    float ice = 0.0;

    if (liquid == 2)
    {
        // Liquid methane / ethane (Titan): dark and glassy.
        sea = mix(vec3(0.03, 0.025, 0.015), vec3(0.008, 0.006, 0.004), sqrt(depth));

        ice = frozen;

        sea = mix(sea, uColorIce, ice);

        seaRoughness = mix(0.06, 0.6, ice);
    }
    else if (liquid == 3)
    {
        // Lava lakes (Io): a dark crust, glowing where hot.
        sea = vec3(0.03, 0.02, 0.015);
        seaRoughness = 0.7;
    }
    else
    {
        sea = mix(TERRAIN_SHALLOW_WATER, TERRAIN_DEEP_WATER, sqrt(depth));

        ice = frozen;

        sea = mix(sea, TERRAIN_ICE, ice);

        seaRoughness = mix(0.12, 0.6, ice);
    }

    // -----------------------------------------------------
    // Coast (anti-aliased over about a pixel)
    // -----------------------------------------------------

    float width = max(fwidth(elevation), 1.0);

    float wet = 1.0 - smoothstep(-width, width, elevation);

    surface.albedo = mix(land, sea, wet);

    // Open liquid is glossy (sun glint); ice and land rough.
    surface.roughness = mix(landRoughness, seaRoughness, wet);

    surface.relief = landRelief * (1.0 - wet);

    surface.material *= 1.0 - wet;

    if (liquid == 3)
    {
        // Mostly crusted over: a dull red glow in visible
        // light (bright only on the night side).
        surface.emissive = wet * vec3(0.7, 0.14, 0.02);
    }

    return surface;
}

// Whittaker-style biome class (for the Biomes view):
//   0 ice  1 tundra  2 taiga  3 temperate forest
//   4 grassland  5 desert  6 savanna  7 tropical rainforest
// Keep in sync with BIOMES in planet/climate.py.
int terrainBiome(
    float precipitation,
    float temperature
)
{
    float wetness = terrainWetness(precipitation, temperature);

    if (temperature < -5.0) return 0;
    if (temperature < 0.0) return 1;
    if (wetness < 0.3) return 5;

    if (temperature < 6.0) return wetness > 0.9 ? 2 : 4;

    if (temperature < 20.0) return wetness > 1.2 ? 3 : 4;

    return wetness > 1.5 ? 7 : 6;
}

vec3 biomeColor(
    int biome
)
{
    const vec3 colors[8] = vec3[](
        vec3(0.92, 0.95, 1.00),     // ice
        vec3(0.62, 0.60, 0.48),     // tundra
        vec3(0.18, 0.40, 0.35),     // taiga
        vec3(0.15, 0.55, 0.18),     // temperate forest
        vec3(0.72, 0.80, 0.35),     // grassland
        vec3(0.95, 0.80, 0.45),     // desert
        vec3(0.85, 0.65, 0.25),     // savanna
        vec3(0.02, 0.38, 0.10)      // tropical rainforest
    );

    return colors[clamp(biome, 0, 7)];
}


// =========================================================
// Data Views
// =========================================================
//
// Alternative colorings for inspecting the terrain data
// (Planet panel > View). Mode numbers match TERRAIN_VIEWS
// in systems/planet_system.py:
//
//   1 elevation (hypsometric tint, 500 m contours, coast)
//   2 slope        3 temperature    4 rainfall
//   5 detail level (quadtree depth, from aTexCoord.y)
//   6 plates       7 crust age      8 crust type
//   9 plate boundaries (red converging, blue pulling apart)
//  10 biomes
//
// Modes 6-9 read the tectonic data each chunk vertex
// carries in its tangent slot (planet/terrain.py
// tectonic_data): x plate index (-1 = no simulation),
// y age / 400 Myr, z continental fraction, w closing
// speed at a boundary (cm/yr).
//
// New data layers are added here as further modes.

vec3 elevationTint(
    float elevation
)
{
    if (elevation < 0.0)
    {
        return mix(
            vec3(0.55, 0.75, 0.95),
            vec3(0.02, 0.08, 0.35),
            clamp(-elevation / 5000.0, 0.0, 1.0)
        );
    }

    const vec3 stops[6] = vec3[](
        vec3(0.15, 0.45, 0.20),     //    0 m
        vec3(0.55, 0.70, 0.30),     //  500 m
        vec3(0.85, 0.75, 0.40),     // 1500 m
        vec3(0.55, 0.38, 0.22),     // 3000 m
        vec3(0.55, 0.52, 0.50),     // 4500 m
        vec3(0.95, 0.95, 0.97)      // 6000 m
    );

    const float heights[6] = float[](0.0, 500.0, 1500.0, 3000.0, 4500.0, 6000.0);

    vec3 color = stops[5];

    for (int i = 0; i < 5; ++i)
    {
        if (elevation < heights[i + 1])
        {
            color = mix(
                stops[i],
                stops[i + 1],
                (elevation - heights[i]) / (heights[i + 1] - heights[i])
            );

            break;
        }
    }

    return color;
}

// 1 on a line every `spacing` units of `value`, about a
// pixel wide.
float contourLine(
    float value,
    float spacing
)
{
    float scaled = value / spacing;

    float width = max(fwidth(scaled), 1e-4);

    float distance = abs(fract(scaled + 0.5) - 0.5);

    return 1.0 - smoothstep(0.5 * width, 1.5 * width, distance);
}

vec3 plateColor(
    float plate
)
{
    float hue = fract(plate * 0.618034);

    return 0.55 + 0.45 * cos(6.2831853 * (hue + vec3(0.0, 0.33, 0.67)));
}

vec3 tectonicOverlayColor(
    int mode,
    vec4 tectonic
)
{
    if (tectonic.x < -0.5)
    {
        // No simulation on this planet.
        return vec3(0.35);
    }

    if (mode == 6)
    {
        // Nearest-cell plate ids interpolate across a
        // boundary triangle; round, and outline where the
        // id changes.
        float plate = floor(tectonic.x + 0.5);

        float edge = clamp(fwidth(tectonic.x) * 2.0, 0.0, 1.0);

        return plateColor(plate) * (1.0 - 0.8 * edge);
    }

    if (mode == 7)
    {
        float age = tectonic.y * 400.0;

        if (tectonic.z > 0.5)
        {
            return vec3(0.55, 0.5, 0.45);       // continental (old)
        }

        // Young (red, at ridges) to old (blue) sea floor,
        // banded every 20 Myr like magnetic stripes.
        vec3 color = mix(vec3(0.95, 0.25, 0.15), vec3(0.15, 0.25, 0.85), clamp(age / 180.0, 0.0, 1.0));

        return color * (0.85 + 0.15 * step(0.5, fract(age / 20.0)));
    }

    if (mode == 8)
    {
        return mix(vec3(0.1, 0.25, 0.6), vec3(0.75, 0.6, 0.35), smoothstep(0.35, 0.65, tectonic.z));
    }

    // 9: boundaries.
    float closing = tectonic.w;

    vec3 base = vec3(0.6);

    if (closing > 0.0)
    {
        return mix(base, vec3(0.9, 0.1, 0.05), clamp(closing / 6.0, 0.0, 1.0));
    }

    return mix(base, vec3(0.1, 0.3, 0.95), clamp(-closing / 6.0, 0.0, 1.0));
}

vec3 terrainOverlayColor(
    int mode,
    float elevation,
    float slope,
    float precipitation,
    float temperature,
    float detail,
    vec4 tectonic
)
{
    if (mode == 10)
    {
        if (elevation < 0.0)
        {
            return temperature < -2.0 ? biomeColor(0) : vec3(0.15, 0.3, 0.6);
        }

        return biomeColor(terrainBiome(precipitation, temperature));
    }

    if (mode >= 6)
    {
        return tectonicOverlayColor(mode, tectonic);
    }

    if (mode == 1)
    {
        vec3 color = elevationTint(elevation);

        color *= 1.0 - 0.35 * contourLine(elevation, 500.0);

        // Coastline: a one-pixel line where elevation = 0.
        float width = max(fwidth(elevation), 1e-3);

        color *= smoothstep(0.5 * width, 1.5 * width, abs(elevation));

        return color;
    }

    if (mode == 2)
    {
        float steepness = clamp((1.0 - slope) / 0.4, 0.0, 1.0);

        return mix(vec3(0.9), vec3(0.08, 0.05, 0.05), steepness);
    }

    if (mode == 3)
    {
        // Temperature: blue (-30 C) - white (5 C) - red
        // (35 C), with a dark line at freezing.
        vec3 color = temperature < 5.0
            ? mix(vec3(0.1, 0.25, 0.9), vec3(0.95), clamp((temperature + 30.0) / 35.0, 0.0, 1.0))
            : mix(vec3(0.95), vec3(0.9, 0.15, 0.05), clamp((temperature - 5.0) / 30.0, 0.0, 1.0));

        float width = max(fwidth(temperature), 1e-3);

        return color * mix(0.4, 1.0, smoothstep(0.5 * width, 1.5 * width, abs(temperature)));
    }

    if (mode == 4)
    {
        // Rainfall on a log scale: 50 mm (tan) to 4,000 mm
        // (deep blue) a year.
        float t = clamp((log(max(precipitation, 1.0)) / log(10.0) - 1.7) / 1.9, 0.0, 1.0);

        vec3 dry = vec3(0.85, 0.7, 0.4);
        vec3 moist = vec3(0.2, 0.65, 0.25);
        vec3 wet = vec3(0.05, 0.2, 0.75);

        return t < 0.5 ? mix(dry, moist, t * 2.0) : mix(moist, wet, t * 2.0 - 1.0);
    }

    if (mode == 5)
    {
        // A distinct hue per level (golden-ratio steps
        // around the color wheel, so neighbors differ).
        float level = floor(detail * 20.0 + 0.5);

        vec3 hue = 0.5 + 0.5 * cos(6.2831853 * (level * 0.618034 + vec3(0.0, 0.33, 0.67)));

        return mix(hue, vec3(0.9), 0.15);
    }

    return vec3(1.0, 0.0, 1.0);
}

// Data view albedo, scaled to natural ground brightness so
// sunlight does not wash the colors out.
vec3 terrainOverlay(
    int mode,
    float elevation,
    float slope,
    float precipitation,
    float temperature,
    float detail,
    vec4 tectonic
)
{
    return 0.45 * terrainOverlayColor(mode, elevation, slope, precipitation, temperature, detail, tectonic);
}
