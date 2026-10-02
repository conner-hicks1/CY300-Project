#version 450 core

#include "include/atmosphere.glsl"

// =========================================================
// Diffuse Daylight LUT
// =========================================================
//
// Under an optically thick atmosphere (Venus's clouds,
// Titan's haze) almost no sunlight arrives directly, yet
// the ground is lit: light scatters down through the
// layers many times. Per (sun zenith cosine, altitude)
// this stores the diffuse irradiance there, as a fraction
// of the sunlight arriving at the top, from a two-stream
// (Eddington) estimate for the column above:
//
//   transmitted ~ mu (2/3 + mu) / (4/3 + tau_s')
//                 * exp(-1.66 tau_a)
//                 / (1 - albedo R)
//
//   tau_s'  scattering depth, aerosols weighted by (1 - g)
//           (forward scattering barely redirects light)
//   tau_a   absorption depth (1.66: diffuse light travels
//           slanted paths)
//   R       the layer's reflectance, tau_s' / (4/3 + tau_s'):
//           light bounces between ground and clouds
//
// Only used where the atmosphere is thick (see
// thickAtmosphereWeight()); Earth's sky keeps Hillaire's
// multiple scattering.


in vec2 vTexCoord;

out vec4 FragColor;

const int STEPS = 64;

void main()
{
    float radius;
    float sunCosZenith;

    lutParameters(vTexCoord, radius, sunCosZenith);

    // Optical depths of the column above, straight up.
    float bottom = radius - groundRadius();
    float span = topRadius() - radius;

    vec3 rayleighDepth = vec3(0.0);
    vec3 mieDepth = vec3(0.0);
    vec3 absorptionDepth = vec3(0.0);

    float dt = span / float(STEPS);

    for (int i = 0; i < STEPS; ++i)
    {
        vec3 rayleighScattering;
        vec3 mieScattering;
        vec3 extinction;

        mediumAt(bottom + (float(i) + 0.5) * dt, rayleighScattering, mieScattering, extinction);

        rayleighDepth += rayleighScattering * dt;
        mieDepth += mieScattering * dt;
        absorptionDepth += (extinction - rayleighScattering - mieScattering) * dt;
    }

    vec3 scatteringDepth = rayleighDepth + mieDepth * (1.0 - uAtmosphereRadii.z);

    // Light keeps diffusing a little past the terminator.
    float mu = max(sunCosZenith + 0.1, 0.0) / 1.1;

    vec3 reflectance = scatteringDepth / (4.0 / 3.0 + scatteringDepth);

    vec3 diffuse =
        mu * (2.0 / 3.0 + mu) / (4.0 / 3.0 + scatteringDepth)
        * exp(-1.66 * absorptionDepth)
        / max(vec3(1.0) - uAtmosphereRadii.w * reflectance, vec3(0.05));

    FragColor = vec4(diffuse, 1.0);
}
