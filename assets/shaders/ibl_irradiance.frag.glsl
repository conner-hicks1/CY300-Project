#version 450 core

#include "include/cubemap.glsl"

#define PI 3.14159265359


// =========================================================
// Diffuse Irradiance
// =========================================================
//
// For each normal N, integrates incoming radiance over the
// hemisphere weighted by cos(theta): what a Lambertian
// surface facing N receives from the whole sky. The sky is
// smooth, so a coarse uniform grid of samples from a
// blurred mip of the environment is enough.

uniform samplerCube uEnvironment;

in vec2 vTexCoord;

out vec4 FragColor;

void main()
{
    vec3 N = cubeFaceDirection(uFace, vTexCoord * 2.0 - 1.0);

    vec3 up = abs(N.y) < 0.999 ? vec3(0.0, 1.0, 0.0) : vec3(0.0, 0.0, 1.0);
    vec3 right = normalize(cross(up, N));
    up = cross(N, right);

    // The sky is smooth: a coarse grid over a blurred mip
    // is visually identical and ~2x cheaper than 0.08.
    const float sampleDelta = 0.12;

    vec3 irradiance = vec3(0.0);
    float samples = 0.0;

    for (float phi = 0.0; phi < 2.0 * PI; phi += sampleDelta)
    {
        for (float theta = 0.0; theta < 0.5 * PI; theta += sampleDelta)
        {
            vec3 tangentSample = vec3(
                sin(theta) * cos(phi),
                sin(theta) * sin(phi),
                cos(theta)
            );

            vec3 sampleDirection =
                tangentSample.x * right
                + tangentSample.y * up
                + tangentSample.z * N;

            irradiance +=
                textureLod(uEnvironment, sampleDirection, 3.0).rgb
                * cos(theta)
                * sin(theta);

            samples += 1.0;
        }
    }

    FragColor = vec4(
        PI * irradiance / samples,
        1.0
    );
}
