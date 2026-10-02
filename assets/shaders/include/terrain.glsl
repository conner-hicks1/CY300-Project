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
//     aColor.b    moisture in [0, 1]
//     aTexCoord.x latitude: |sin| of the planet-space
//                 latitude, 0 = equator, 1 = pole
//
// Thresholding per pixel keeps coastlines and snow lines
// smooth even on coarse chunks whose vertices are hundreds
// of kilometers apart (per-vertex colors there showed as
// triangle-shaped blocks).

struct TerrainSurface
{
    vec3 albedo;
    float roughness;
};

const vec3 TERRAIN_SAND = vec3(0.42, 0.36, 0.22);
const vec3 TERRAIN_DRY_GRASS = vec3(0.28, 0.24, 0.10);
const vec3 TERRAIN_GRASS = vec3(0.07, 0.16, 0.04);
const vec3 TERRAIN_FOREST = vec3(0.025, 0.07, 0.025);
const vec3 TERRAIN_ROCK = vec3(0.16, 0.14, 0.12);
const vec3 TERRAIN_SNOW = vec3(0.80, 0.82, 0.86);
const vec3 TERRAIN_SHALLOW_WATER = vec3(0.02, 0.10, 0.14);
const vec3 TERRAIN_DEEP_WATER = vec3(0.004, 0.015, 0.05);
const vec3 TERRAIN_ICE = vec3(0.65, 0.72, 0.78);

TerrainSurface terrainSurface(
    float elevation,
    float slope,
    float moisture,
    float latitude
)
{
    // -----------------------------------------------------
    // Land
    // -----------------------------------------------------

    vec3 grass = mix(TERRAIN_DRY_GRASS, TERRAIN_GRASS, smoothstep(0.35, 0.65, moisture));

    vec3 land = mix(TERRAIN_SAND, grass, smoothstep(5.0, 60.0, elevation));

    // Forest on moist, low, gentle ground.
    float forest =
        smoothstep(0.5, 0.75, moisture)
        * (1.0 - smoothstep(1200.0, 2500.0, elevation))
        * smoothstep(0.92, 0.97, slope)
        * smoothstep(60.0, 200.0, elevation);

    land = mix(land, TERRAIN_FOREST, forest);

    land = mix(land, TERRAIN_ROCK, smoothstep(1500.0, 3000.0, elevation));

    // Cliffs: steep ground is bare rock.
    land = mix(land, TERRAIN_ROCK, 1.0 - smoothstep(0.75, 0.88, slope));

    // Snow line falls toward the poles; snow does not
    // stick to cliffs.
    float snowLine = 4200.0 * (1.0 - latitude * latitude * latitude) - 300.0;

    float snow =
        smoothstep(snowLine - 200.0, snowLine + 200.0, elevation)
        * smoothstep(0.7, 0.82, slope);

    land = mix(land, TERRAIN_SNOW, snow);

    // -----------------------------------------------------
    // Water
    // -----------------------------------------------------

    float depth = clamp(-elevation / 3000.0, 0.0, 1.0);

    vec3 water = mix(TERRAIN_SHALLOW_WATER, TERRAIN_DEEP_WATER, sqrt(depth));

    // Polar sea ice.
    float ice = smoothstep(0.95, 0.97, latitude);

    water = mix(water, TERRAIN_ICE, ice);

    // -----------------------------------------------------
    // Coast (anti-aliased over about a pixel)
    // -----------------------------------------------------

    float width = max(fwidth(elevation), 1.0);

    float wet = 1.0 - smoothstep(-width, width, elevation);

    TerrainSurface surface;

    surface.albedo = mix(land, water, wet);

    // Open water is glossy (sun glint); ice and land rough.
    surface.roughness = mix(0.9, mix(0.12, 0.6, ice), wet);

    return surface;
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
//   2 slope      3 moisture      4 latitude bands
//   5 detail level (quadtree depth, from aTexCoord.y)
//   6 plates       7 crust age      8 crust type
//   9 plate boundaries (red converging, blue pulling apart)
//
// Modes 6-9 read the tectonic data each chunk vertex
// carries in its tangent slot (planet/terrain.py
// tectonic_data): x plate index (-1 = no simulation),
// y age / 400 Myr, z continental fraction, w closing
// speed at a boundary (cm/yr).
//
// New data layers (plates, crust age, temperature...) are
// added here as further modes.

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
    float moisture,
    float latitude,
    float detail,
    vec4 tectonic
)
{
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
        return mix(vec3(0.55, 0.38, 0.18), vec3(0.10, 0.35, 0.85), moisture);
    }

    if (mode == 4)
    {
        float degrees_ = degrees(asin(clamp(latitude, 0.0, 1.0)));

        float band = mod(floor(degrees_ / 10.0), 2.0);

        vec3 color = mix(vec3(0.25, 0.55, 0.85), vec3(0.85, 0.85, 0.85), degrees_ / 90.0);

        return color * (0.8 + 0.2 * band);
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
    float moisture,
    float latitude,
    float detail,
    vec4 tectonic
)
{
    return 0.45 * terrainOverlayColor(mode, elevation, slope, moisture, latitude, detail, tectonic);
}
