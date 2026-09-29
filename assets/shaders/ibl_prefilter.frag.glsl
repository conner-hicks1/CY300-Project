#version 330 core

#include "include/cubemap.glsl"
#include "include/lighting.glsl"
#include "include/ibl_sampling.glsl"


// =========================================================
// Specular Prefiltering
// =========================================================
//
// For each mip level (roughness = level / (levels - 1)),
// convolves the environment with the GGX lobe using
// importance sampling. Assumes N = V = R (Karis 2013).
//
// Each sample reads from a mip of the environment chosen
// by the sample's solid angle, which removes the bright
// speckles that point-sampling a high-resolution map
// would leave.

uniform samplerCube uEnvironment;
uniform float uRoughness;
uniform float uEnvironmentSize;     // base resolution

// Samples for this mip. Roughness 0 needs only one: every
// GGX sample then reflects straight back along N.
uniform int uSampleCount;

in vec2 vTexCoord;

out vec4 FragColor;

const uint MAX_SAMPLES = 128u;

void main()
{
    vec3 N = cubeFaceDirection(uFace, vTexCoord * 2.0 - 1.0);
    vec3 V = N;

    vec3 color = vec3(0.0);
    float totalWeight = 0.0;

    float texelSolidAngle =
        4.0 * PI / (6.0 * uEnvironmentSize * uEnvironmentSize);

    uint sampleCount = uint(clamp(uSampleCount, 1, int(MAX_SAMPLES)));

    for (uint i = 0u; i < MAX_SAMPLES; ++i)
    {
        if (i >= sampleCount)
        {
            break;
        }

        vec2 xi = hammersley(i, sampleCount);

        vec3 H = importanceSampleGGX(xi, N, uRoughness);
        vec3 L = normalize(2.0 * dot(V, H) * H - V);

        float NdotL = dot(N, L);

        if (NdotL > 0.0)
        {
            float NdotH = max(dot(N, H), 0.0);
            float HdotV = max(dot(H, V), 0.0);

            float D = distributionGGX(NdotH, uRoughness);

            float pdf = D * NdotH / (4.0 * HdotV) + 1e-4;

            float sampleSolidAngle = 1.0 / (float(sampleCount) * pdf + 1e-4);

            float mip =
                uRoughness == 0.0
                ? 0.0
                : 0.5 * log2(sampleSolidAngle / texelSolidAngle);

            color += textureLod(uEnvironment, L, mip).rgb * NdotL;
            totalWeight += NdotL;
        }
    }

    FragColor = vec4(
        color / max(totalWeight, 1e-4),
        1.0
    );
}
