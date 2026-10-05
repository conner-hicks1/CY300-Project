#version 450 core

#include "include/blocks.glsl"
#include "include/lighting.glsl"
#include "include/shadows.glsl"
#include "include/atmosphere.glsl"
#include "include/clouds.glsl"
#include "include/bodies.glsl"
#include "include/rings.glsl"
#include "include/terrain.glsl"


// =========================================================
// Fragment Inputs
// =========================================================

in vec3 vWorldPosition;
in vec3 vNormal;
in vec4 vTangent;
in vec3 vColor;
in vec2 vTexCoord;


// =========================================================
// Fragment Outputs
// =========================================================
//
// Linear HDR radiance. Bloom, tone mapping and gamma
// correction happen in later passes.

out vec4 FragColor;


// =========================================================
// Material (metallic-roughness, as in glTF 2.0)
// =========================================================
//
// Factors multiply the maps. Samplers without a material
// texture get engine defaults (Renderer.set_default_texture):
// white for color/data maps, a flat normal map.

uniform sampler2D uBaseColorMap;           // sRGB
uniform sampler2D uMetallicRoughnessMap;   // linear: G roughness, B metallic
uniform sampler2D uNormalMap;              // linear, tangent space
uniform sampler2D uOcclusionMap;           // linear: R occlusion
uniform sampler2D uEmissiveMap;            // sRGB

uniform vec3 uBaseColor;
uniform float uMetallic;
uniform float uRoughness;
uniform float uNormalStrength;
uniform float uOcclusionStrength;
uniform vec3 uEmissive;                    // HDR: color * strength

// 1 = planet terrain: the vertex color and UV carry terrain
// inputs, and albedo / roughness come from
// include/terrain.glsl.
uniform float uTerrainShading;

// 1 = scattered rocks, trees and grass (graphics/scatter.py):
// the UV is meters on the thing itself; leaves and grain
// break up its albedo up close.
uniform float uScatterDetail;

// Planet terrain data view (0 = natural colors); see
// terrainOverlay() in include/terrain.glsl.
uniform float uTerrainView;

// The body this surface belongs to (PlanetSystem): its
// center relative to the camera (km), w 0 = none (a prop,
// on the atmosphere's planet), 1 = a body, 2 = an
// irregular one (no sphere horizon); and the quaternion
// taking world directions into its frame.
uniform vec4 uBodyCenter;
uniform vec4 uBodyFrame;

// A real color map of the body (equirectangular, sRGB, left
// edge at 180 W; planet/maps.py): 0 = none; its brightness
// scale to the body's albedo.
uniform sampler2D uColorMap;
uniform float uColorMapStrength;
uniform float uColorMapScale;

vec3 sampleColorMap(
    vec3 direction
)
{
    float longitude = atan(direction.x, direction.z);
    float latitude = asin(clamp(direction.y, -1.0, 1.0));

    vec2 uv = vec2(longitude / (2.0 * ATMOSPHERE_PI) + 0.5, 0.5 + latitude / ATMOSPHERE_PI);

    // Mip level from whichever side of the 180 deg seam has
    // no jump.
    vec2 dx = dFdx(uv);
    vec2 dy = dFdy(uv);

    dx.x = abs(dx.x) > 0.5 ? dx.x - sign(dx.x) : dx.x;
    dy.x = abs(dy.x) > 0.5 ? dy.x - sign(dy.x) : dy.x;

    return textureGrad(uColorMap, uv, dx, dy).rgb;
}


// =========================================================
// Image-Based Lighting (baked from the sky)
// =========================================================

uniform samplerCube uIrradianceMap;        // diffuse
uniform samplerCube uPrefilterMap;         // specular, mip = roughness
uniform sampler2D uBrdfLut;                // (scale, bias) for F0


// =========================================================
// Surface Normal
// =========================================================

vec3 surfaceNormal(
    vec3 geometricNormal
)
{
    // Terrain has no normal map; its "tangent" is data.
    if (uTerrainShading > 0.5)
    {
        return geometricNormal;
    }

    // Re-orthogonalize after interpolation.
    vec3 T = normalize(
        vTangent.xyz
        - geometricNormal * dot(geometricNormal, vTangent.xyz)
    );

    vec3 B = cross(geometricNormal, T) * vTangent.w;

    vec3 mapped =
        texture(uNormalMap, vTexCoord).xyz
        * 2.0 - 1.0;

    mapped.xy *= uNormalStrength;

    return normalize(
        mat3(T, B, geometricNormal) * mapped
    );
}


// =========================================================
// Which Body
// =========================================================
//
// The atmosphere block describes one body's air (the one
// the camera is at). Its sunset light, cloud shadows and
// sky light belong to surfaces of that body; other bodies
// (a moon in the sky) are lit by the sun alone, as in
// space.

bool hasBody()
{
    return uBodyCenter.w > 0.5;
}

bool onAtmosphereBody()
{
    if (!atmospherePresent())
    {
        return false;
    }

    if (!hasBody())
    {
        return true;
    }

    // An irregular body is no sphere: the atmosphere's
    // horizon test would cut its lit side.
    return uBodyCenter.w < 1.5 && distance(uBodyCenter.xyz, uPlanetCenter.xyz) < 1.0;
}


// =========================================================
// Main
// =========================================================

void main()
{
    // -----------------------------------------------------
    // Material
    // -----------------------------------------------------

    vec3 baseColor =
        texture(uBaseColorMap, vTexCoord).rgb
        * vColor
        * uBaseColor;

    vec4 metallicRoughness = texture(uMetallicRoughnessMap, vTexCoord);

    // Very low roughness makes GGX highlights sub-pixel
    // and aliased; clamp to a small minimum.
    float roughness = clamp(uRoughness * metallicRoughness.g, 0.04, 1.0);
    float metallic = clamp(uMetallic * metallicRoughness.b, 0.0, 1.0);

    vec3 terrainEmissive = vec3(0.0);

    // Micro-relief height (m) for bump shading.
    float terrainRelief = 0.0;

    bool atmosphereBody = onAtmosphereBody();

    if (uTerrainShading > 0.5)
    {
        // Planet-local position (km), in the planet's own frame
        // (its detail turns with it).
        vec3 local = vWorldPosition * 0.001 - (hasBody() ? uBodyCenter.xyz : uPlanetCenter.xyz);

        vec3 bodyLocal = hasBody() ? rotateByQuaternion(uBodyFrame, local) : toPlanetFrame(local);

        // Near the camera: meters from the detail origin, exact.
        bool near = uDetailOrigin.w > 0.5 && hasBody();

        vec3 q = near ? rotateByQuaternion(uBodyFrame, vWorldPosition - uDetailOrigin.xyz) : vec3(0.0);

        TerrainDetail detail = terrainDetail(bodyLocal, q, near, length(fwidth(local)));

        // The planet's own frame (giants' bands and storms).
        vec3 planetDirection = normalize(bodyLocal);

        // Skirts (planet/chunk.py) carry a slope of 2: shade
        // them as the ground they hang from, without relief.
        float skirt = smoothstep(1.0, 1.02, vColor.g);

        float slope = min(vColor.g, 1.0);

        TerrainSurface terrain = terrainSurface(
            vColor.r,
            slope,
            vColor.b,
            vTexCoord.x,
            vTangent.x >= 0.0 ? vTangent.z : -1.0,
            fract(vTexCoord.y) / 0.99,
            detail.albedo,
            planetDirection,
            length(fwidth(planetDirection))
        );

        terrainRelief = detail.height * terrain.relief * (1.0 - skirt);

        // Close up, the ground's own texture (only on the
        // camera's body, within reach of the detail origin).
        float closeAlbedo = 0.0;

        if (near && uTerrainView < 0.5)
        {
            float footprint = length(fwidth(vWorldPosition));

            if (footprint < 16.0)
            {
                TerrainDetail close = terrainCloseUp(
                    q,
                    planetDirection,
                    footprint,
                    terrain.material,
                    1.0 - smoothstep(0.6, 0.85, slope)
                );

                terrainRelief += close.height * (1.0 - skirt);
                closeAlbedo = close.albedo;
            }
        }

        baseColor = terrain.albedo * uBaseColor;

        // Measured colors, with the generated fine variation.
        if (uColorMapStrength > 0.0)
        {
            vec3 mapped = sampleColorMap(planetDirection) * uColorMapScale * (1.0 + 0.2 * detail.albedo);

            baseColor = mix(baseColor, mapped, uColorMapStrength);
        }

        // The close-up texture, over measured colors too.
        baseColor *= max(1.0 + closeAlbedo, 0.2);
        roughness = terrain.roughness;
        terrainEmissive = terrain.emissive;

        int view = int(uTerrainView + 0.5);

        if (view > 0)
        {
            baseColor = terrainOverlay(
                view,
                vColor.r,
                slope,
                vColor.b,
                vTexCoord.x,
                floor(vTexCoord.y) / 20.0,
                vTangent
            );

            roughness = 0.9;
        }
    }

    if (uScatterDetail > 0.5)
    {
        // Leaf clusters and gaps (~20 cm), grain finer: faded
        // out once smaller than a pixel.
        float footprint = length(fwidth(vTexCoord));

        float coarse = terrainValueNoise(vec3(vTexCoord * 4.0, 0.5));
        float fine = terrainValueNoise(vec3(vTexCoord * 13.0, 3.5));

        float detail =
            0.55 * coarse * (1.0 - smoothstep(0.1, 0.4, footprint))
            + 0.3 * fine * (1.0 - smoothstep(0.03, 0.12, footprint));

        baseColor *= max(1.0 + detail, 0.1);
    }

    float occlusion = mix(
        1.0,
        texture(uOcclusionMap, vTexCoord).r,
        uOcclusionStrength
    );

    vec3 emissive =
        texture(uEmissiveMap, vTexCoord).rgb
        * uEmissive
        + terrainEmissive;

    vec3 geometricNormal = normalize(vNormal);

    Surface surface;
    surface.N = surfaceNormal(geometricNormal);

    if (uTerrainShading > 0.5)
    {
        surface.N = terrainBumpNormal(surface.N, vWorldPosition, terrainRelief);
    }
    surface.V = normalize(uViewPosition.xyz - vWorldPosition);
    surface.baseColor = baseColor;
    surface.metallic = metallic;
    surface.roughness = roughness;

    // Dielectrics reflect ~4% at normal incidence; metals
    // reflect their base color.
    surface.F0 = mix(vec3(0.04), baseColor, metallic);

    float viewDepth = -(uView * vec4(vWorldPosition, 1.0)).z;

    vec3 color = vec3(0.0);

    // -----------------------------------------------------
    // Directional
    // -----------------------------------------------------

    if (uLightCounts.z != 0)
    {
        vec3 L = normalize(-uDirectionalDirection.xyz);

        vec3 radiance =
            uDirectionalColor.rgb
            * uDirectionalColor.a;

        // On a planet with an atmosphere, the light is the
        // sun above the air: what reaches this point is
        // reddened near the horizon and gone at night.
        if (atmosphereBody)
        {
            radiance *= sunTransmittanceAtWorld(vWorldPosition);

            radiance *= cloudShadow(vWorldPosition * 0.001 - uPlanetCenter.xyz);
        }

        // The rings' shadow, and other bodies' (eclipses).
        radiance *= ringShadow(vWorldPosition * 0.001 - uRingCenter.xyz, L);

        radiance *= eclipse(
            vWorldPosition * 0.001,
            L,
            hasBody() ? uBodyCenter.xyz : vec3(1e30)
        );

        // Geometric normal for the shadow lookup: the
        // normal map must not move where occluders are.
        float shadow = directionalShadow(
            vWorldPosition,
            geometricNormal,
            viewDepth
        );

        // Mountains' shadows beyond the cascades (and over
        // them: whichever is darker).
        shadow = min(shadow, terrainShadow(vWorldPosition, geometricNormal));

        color += shadow * shadeDirect(surface, L, radiance);

        // Planetshine: sunlight off the neighbors (the
        // Moon's night side in earthshine, Jupiter lighting
        // its moons, moonlight on Earth), from below their
        // horizon nothing.
        vec3 sunlight = uDirectionalColor.rgb * uDirectionalColor.a;

        vec3 point = vWorldPosition * 0.001;

        vec3 skip = hasBody() ? uBodyCenter.xyz : vec3(1e30);

        vec3 up = hasBody() ? normalize(point - uBodyCenter.xyz) : surface.N;

        int bodies = int(uBodyParams.x + 0.5);

        for (int i = 0; i < MAX_BODIES; ++i)
        {
            if (i >= bodies)
            {
                break;
            }

            vec3 direction;

            vec3 shine = planetshine(point, L, skip, i, direction);

            if (dot(shine, shine) <= 0.0)
            {
                continue;
            }

            shine *= smoothstep(-0.02, 0.02, dot(up, direction));

            color += shadeDirect(surface, direction, sunlight * shine);
        }
    }

    // -----------------------------------------------------
    // Point Lights
    // -----------------------------------------------------

    for (int i = 0; i < uLightCounts.x; ++i)
    {
        vec3 toLight =
            uPointPositionRange[i].xyz
            - vWorldPosition;

        float distance = length(toLight);

        vec3 radiance =
            uPointColorIntensity[i].rgb
            * uPointColorIntensity[i].a
            * attenuate(distance, uPointPositionRange[i].w);

        color += shadeDirect(surface, toLight / distance, radiance);
    }

    // -----------------------------------------------------
    // Spot Lights
    // -----------------------------------------------------

    for (int i = 0; i < uLightCounts.y; ++i)
    {
        vec3 toLight =
            uSpotPositionRange[i].xyz
            - vWorldPosition;

        float distance = length(toLight);

        vec3 L = toLight / distance;

        float innerCutoff = uSpotDirectionInner[i].w;
        float outerCutoff = uSpotParams[i].x;

        // Angle between the cone axis and this fragment.
        float theta = dot(
            -L,
            normalize(uSpotDirectionInner[i].xyz)
        );

        float cone = clamp(
            (theta - outerCutoff)
            / max(innerCutoff - outerCutoff, 1e-4),
            0.0,
            1.0
        );

        if (cone <= 0.0)
        {
            continue;
        }

        vec3 radiance =
            uSpotColorIntensity[i].rgb
            * uSpotColorIntensity[i].a
            * attenuate(distance, uSpotPositionRange[i].w)
            * cone;

        float shadow = spotShadow(
            i,
            vWorldPosition,
            geometricNormal,
            distance
        );

        color += shadow * shadeDirect(surface, L, radiance);
    }

    // -----------------------------------------------------
    // Ambient: Image-Based Lighting
    // -----------------------------------------------------
    //
    // Split-sum approximation (Karis 2013): diffuse from
    // the irradiance map; specular from the prefiltered
    // map at the reflection direction, scaled by the BRDF
    // lookup table.

    vec3 N = surface.N;
    vec3 V = surface.V;

    float NdotV = max(dot(N, V), 1e-4);

    vec3 F = fresnelSchlickRoughness(NdotV, surface.F0, roughness);

    vec3 kD = (vec3(1.0) - F) * (1.0 - metallic);

    vec3 irradiance = texture(uIrradianceMap, N).rgb;

    vec3 diffuse = irradiance * baseColor;

    vec3 R = reflect(-V, N);

    vec3 prefiltered = textureLod(
        uPrefilterMap,
        R,
        roughness * float(PREFILTER_MIP_LEVELS - 1)
    ).rgb;

    vec2 brdf = texture(uBrdfLut, vec2(NdotV, roughness)).rg;

    vec3 specular = prefiltered * (F * brdf.x + brdf.y);

    // Sky light is the camera's sky: other bodies get none
    // (the night side of a moon in the sky is dark).
    float skyLight = (atmosphereBody || !hasBody()) ? 1.0 : 0.0;

    color += (kD * diffuse + specular) * occlusion * uLightParams.x * skyLight;

    color += emissive;

    if (uLightParams.w > 0.5)
    {
        color *= cascadeDebugColor(viewDepth);
    }

    FragColor = vec4(color, 1.0);
}
