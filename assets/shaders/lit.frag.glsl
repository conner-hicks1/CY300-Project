#version 450 core

#include "include/blocks.glsl"
#include "include/lighting.glsl"
#include "include/shadows.glsl"


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

    float occlusion = mix(
        1.0,
        texture(uOcclusionMap, vTexCoord).r,
        uOcclusionStrength
    );

    vec3 emissive =
        texture(uEmissiveMap, vTexCoord).rgb
        * uEmissive;

    vec3 geometricNormal = normalize(vNormal);

    Surface surface;
    surface.N = surfaceNormal(geometricNormal);
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

        // Geometric normal for the shadow lookup: the
        // normal map must not move where occluders are.
        float shadow = directionalShadow(
            vWorldPosition,
            geometricNormal,
            viewDepth
        );

        color += shadow * shadeDirect(surface, L, radiance);
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

    color += (kD * diffuse + specular) * occlusion * uLightParams.x;

    color += emissive;

    if (uLightParams.w > 0.5)
    {
        color *= cascadeDebugColor(viewDepth);
    }

    FragColor = vec4(color, 1.0);
}
