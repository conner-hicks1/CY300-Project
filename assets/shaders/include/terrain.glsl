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
};

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

// Rain relative to what the warmth evaporates: < 0.3
// desert, ~0.5-1 grassland / savanna, > 1.2 forest.
float terrainWetness(
    float precipitation,
    float temperature
)
{
    return precipitation / (300.0 + 30.0 * max(temperature, 0.0));
}

TerrainSurface terrainSurface(
    float elevation,
    float slope,
    float precipitation,
    float temperature
)
{
    float wetness = terrainWetness(precipitation, temperature);

    // 1 in the tropics; 1 in the boreal zone.
    float tropical = smoothstep(18.0, 24.0, temperature);
    float boreal = smoothstep(8.0, 2.0, temperature);

    // -----------------------------------------------------
    // Land: desert -> grassland -> forest by wetness,
    // flavored by warmth
    // -----------------------------------------------------

    vec3 desert = mix(TERRAIN_HOT_DESERT, TERRAIN_COLD_DESERT, boreal);

    vec3 grass = mix(TERRAIN_DRY_GRASS, TERRAIN_GRASS, smoothstep(0.5, 1.0, wetness));

    vec3 forest = mix(TERRAIN_FOREST, TERRAIN_RAINFOREST, tropical);

    forest = mix(forest, TERRAIN_TAIGA, boreal);

    vec3 land = mix(desert, grass, smoothstep(0.15, 0.45, wetness));

    land = mix(land, forest, smoothstep(0.9, 1.6, wetness) * smoothstep(0.9, 0.96, slope));

    // Too cold for trees: tundra (and the alpine zone above
    // the tree line, since temperature follows height).
    land = mix(land, TERRAIN_TUNDRA, smoothstep(1.0, -3.0, temperature));

    // Beaches.
    land = mix(TERRAIN_SAND, land, smoothstep(5.0, 60.0, elevation));

    // Cliffs are bare rock, as are cold high slopes.
    land = mix(land, TERRAIN_ROCK, 1.0 - smoothstep(0.75, 0.88, slope));

    land = mix(land, TERRAIN_ROCK, smoothstep(-3.0, -7.0, temperature) * 0.6);

    // Lasting snow and ice where the year averages below
    // freezing; snow does not stick to cliffs.
    float snow =
        smoothstep(-2.0, -7.0, temperature)
        * smoothstep(0.7, 0.82, slope);

    land = mix(land, TERRAIN_SNOW, snow);

    // -----------------------------------------------------
    // Water
    // -----------------------------------------------------

    float depth = clamp(-elevation / 3000.0, 0.0, 1.0);

    vec3 water = mix(TERRAIN_SHALLOW_WATER, TERRAIN_DEEP_WATER, sqrt(depth));

    // Sea ice where the sea surface averages below freezing
    // (sea water freezes at about -2 C).
    float ice = smoothstep(-1.5, -5.0, temperature);

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
