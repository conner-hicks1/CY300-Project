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
