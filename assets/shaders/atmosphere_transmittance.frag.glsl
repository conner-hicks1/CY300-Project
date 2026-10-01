#version 450 core

#include "include/atmosphere.glsl"


// =========================================================
// Transmittance LUT
// =========================================================
//
// x: sun zenith cosine, y: altitude. Optical depth from
// that point to the top of the atmosphere; zero when the
// planet is in the way.

in vec2 vTexCoord;

out vec4 FragColor;

const int STEPS = 40;

void main()
{
    float radius;
    float sunCosZenith;

    lutParameters(vTexCoord, radius, sunCosZenith);

    vec3 origin = vec3(0.0, radius, 0.0);

    vec3 direction = vec3(
        sqrt(max(0.0, 1.0 - sunCosZenith * sunCosZenith)),
        sunCosZenith,
        0.0
    );

    if (raySphere(origin, direction, groundRadius()) > 0.0)
    {
        FragColor = vec4(0.0, 0.0, 0.0, 1.0);
        return;
    }

    float length_ = raySphere(origin, direction, topRadius());

    vec3 opticalDepth = vec3(0.0);

    float dt = length_ / float(STEPS);

    for (int i = 0; i < STEPS; ++i)
    {
        vec3 p = origin + (float(i) + 0.5) * dt * direction;

        vec3 rayleighScattering;
        float mieScattering;
        vec3 extinction;

        mediumAt(length(p) - groundRadius(), rayleighScattering, mieScattering, extinction);

        opticalDepth += extinction * dt;
    }

    FragColor = vec4(exp(-opticalDepth), 1.0);
}
