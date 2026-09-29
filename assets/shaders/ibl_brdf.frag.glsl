#version 330 core

#include "include/lighting.glsl"
#include "include/ibl_sampling.glsl"


// =========================================================
// BRDF Integration Lookup Table
// =========================================================
//
// x = NdotV, y = roughness. Stores (A, B) such that the
// specular BRDF integrated over the hemisphere is
// F0 * A + B (split-sum approximation, Karis 2013).
// Depends on nothing in the scene; baked once.

in vec2 vTexCoord;

out vec4 FragColor;

const uint SAMPLE_COUNT = 512u;

// k for image-based lighting differs from direct lighting.
float geometrySmithIBL(
    float NdotV,
    float NdotL,
    float roughness
)
{
    float k = (roughness * roughness) / 2.0;

    return
        geometrySchlickGGX(NdotV, k)
        * geometrySchlickGGX(NdotL, k);
}

void main()
{
    float NdotV = max(vTexCoord.x, 1e-4);
    float roughness = vTexCoord.y;

    vec3 V = vec3(sqrt(1.0 - NdotV * NdotV), 0.0, NdotV);
    vec3 N = vec3(0.0, 0.0, 1.0);

    float A = 0.0;
    float B = 0.0;

    for (uint i = 0u; i < SAMPLE_COUNT; ++i)
    {
        vec2 xi = hammersley(i, SAMPLE_COUNT);

        vec3 H = importanceSampleGGX(xi, N, roughness);
        vec3 L = normalize(2.0 * dot(V, H) * H - V);

        float NdotL = max(L.z, 0.0);
        float NdotH = max(H.z, 0.0);
        float VdotH = max(dot(V, H), 0.0);

        if (NdotL > 0.0)
        {
            float G = geometrySmithIBL(NdotV, NdotL, roughness);

            float visibility = G * VdotH / (NdotH * NdotV);

            float fresnel = pow(1.0 - VdotH, 5.0);

            A += (1.0 - fresnel) * visibility;
            B += fresnel * visibility;
        }
    }

    FragColor = vec4(
        A / float(SAMPLE_COUNT),
        B / float(SAMPLE_COUNT),
        0.0,
        1.0
    );
}
