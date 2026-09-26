#version 330 core

#include "include/blocks.glsl"
#include "include/lighting.glsl"


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
// Linear HDR radiance. Tone mapping and gamma correction
// happen in post.frag.glsl.

out vec4 FragColor;


// =========================================================
// Material
// =========================================================
//
// Samplers without a material texture get engine
// defaults (Renderer.set_default_texture): white albedo,
// flat normal map, full-strength specular map.

uniform sampler2D uTexture;       // albedo, sRGB
uniform sampler2D uNormalMap;     // tangent-space, linear
uniform sampler2D uSpecularMap;   // R = specular mask, linear

uniform vec3 uBaseColor;
uniform float uSpecularStrength;
uniform float uShininess;
uniform float uNormalStrength;

uniform sampler2D uShadowMap;


// =========================================================
// Surface Normal
// =========================================================

vec3 surfaceNormal()
{
    vec3 N = normalize(vNormal);

    // Re-orthogonalize after interpolation.
    vec3 T = normalize(
        vTangent.xyz
        - N * dot(N, vTangent.xyz)
    );

    vec3 B = cross(N, T) * vTangent.w;

    vec3 mapped =
        texture(uNormalMap, vTexCoord).xyz
        * 2.0 - 1.0;

    mapped.xy *= uNormalStrength;

    return normalize(
        mat3(T, B, N) * mapped
    );
}


// =========================================================
// Main
// =========================================================

void main()
{
    vec3 albedo =
        texture(uTexture, vTexCoord).rgb
        * vColor
        * uBaseColor;

    float specularStrength =
        uSpecularStrength
        * texture(uSpecularMap, vTexCoord).r;

    vec3 N = surfaceNormal();
    vec3 V = normalize(uViewPosition.xyz - vWorldPosition);

    vec3 color = uLightParams.x * albedo;

    // -----------------------------------------------------
    // Directional
    // -----------------------------------------------------

    if (uLightCounts.z != 0)
    {
        vec3 L = normalize(-uDirectionalDirection.xyz);

        vec3 radiance =
            uDirectionalColor.rgb
            * uDirectionalColor.a;

        // Bias uses the geometric normal: the normal map
        // must not move where the shadow map says occluders
        // are.
        float shadow = directionalShadow(
            uShadowMap,
            vWorldPosition,
            normalize(vNormal),
            L
        );

        color += shadow * blinnPhong(
            N, V, L, radiance, albedo,
            specularStrength, uShininess
        );
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

        vec3 L = toLight / distance;

        vec3 radiance =
            uPointColorIntensity[i].rgb
            * uPointColorIntensity[i].a
            * attenuate(distance, uPointPositionRange[i].w);

        color += blinnPhong(
            N, V, L, radiance, albedo,
            specularStrength, uShininess
        );
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
        float outerCutoff = uSpotOuter[i].x;

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

        vec3 radiance =
            uSpotColorIntensity[i].rgb
            * uSpotColorIntensity[i].a
            * attenuate(distance, uSpotPositionRange[i].w)
            * cone;

        color += blinnPhong(
            N, V, L, radiance, albedo,
            specularStrength, uShininess
        );
    }

    FragColor = vec4(color, 1.0);
}
