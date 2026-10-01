#version 450 core

#include "include/atmosphere.glsl"


// =========================================================
// Multiple-Scattering LUT
// =========================================================
//
// Hillaire 2020, section 5.5: from a point, gather second-
// order scattered light over the sphere of directions
// (isotropic phase), and the fraction f_ms that would be
// scattered again; the infinite series of further bounces
// sums to L2 / (1 - f_ms).
//
// x: sun zenith cosine, y: altitude.

in vec2 vTexCoord;

out vec4 FragColor;

const int DIRECTIONS = 8;       // per axis: 64 directions
const int STEPS = 20;

void main()
{
    float radius;
    float sunCosZenith;

    lutParameters(vTexCoord, radius, sunCosZenith);

    vec3 origin = vec3(0.0, radius, 0.0);

    vec3 sunDirection = vec3(
        sqrt(max(0.0, 1.0 - sunCosZenith * sunCosZenith)),
        sunCosZenith,
        0.0
    );

    const float isotropicPhase = 1.0 / (4.0 * ATMOSPHERE_PI);

    vec3 secondOrder = vec3(0.0);
    vec3 transfer = vec3(0.0);

    float weight = 1.0 / float(DIRECTIONS * DIRECTIONS);

    for (int i = 0; i < DIRECTIONS; ++i)
    {
        for (int j = 0; j < DIRECTIONS; ++j)
        {
            // Uniform over the sphere.
            float theta = 2.0 * ATMOSPHERE_PI * (float(i) + 0.5) / float(DIRECTIONS);
            float cosPhi = 1.0 - 2.0 * (float(j) + 0.5) / float(DIRECTIONS);
            float sinPhi = sqrt(max(0.0, 1.0 - cosPhi * cosPhi));

            vec3 direction = vec3(cos(theta) * sinPhi, cosPhi, sin(theta) * sinPhi);

            float groundHit = raySphere(origin, direction, groundRadius());
            float topHit = raySphere(origin, direction, topRadius());

            float length_ = groundHit > 0.0 ? groundHit : topHit;

            vec3 luminance = vec3(0.0);
            vec3 transferred = vec3(0.0);
            vec3 transmittance = vec3(1.0);

            float t = 0.0;

            for (int s = 0; s < STEPS; ++s)
            {
                float next = (float(s) + 0.3) / float(STEPS) * length_;
                float dt = next - t;
                t = next;

                vec3 p = origin + t * direction;
                float r = length(p);

                vec3 rayleighScattering;
                float mieScattering;
                vec3 extinction;

                mediumAt(r - groundRadius(), rayleighScattering, mieScattering, extinction);

                vec3 sampleTransmittance = exp(-dt * extinction);

                vec3 scattering = rayleighScattering + vec3(mieScattering);

                vec3 safeExtinction = max(extinction, vec3(1e-7));

                // Fraction scattered (any direction) here.
                transferred +=
                    transmittance
                    * (scattering - scattering * sampleTransmittance) / safeExtinction;

                vec3 sunLight = sunTransmittance(r, dot(p, sunDirection) / r);

                vec3 inScattering = scattering * isotropicPhase * sunLight;

                luminance +=
                    transmittance
                    * (inScattering - inScattering * sampleTransmittance) / safeExtinction;

                transmittance *= sampleTransmittance;
            }

            // Sunlight bounced off the ground.
            if (groundHit > 0.0)
            {
                vec3 hit = normalize(origin + groundHit * direction) * groundRadius();

                float cosine = dot(normalize(hit), sunDirection);

                if (cosine > 0.0)
                {
                    luminance +=
                        transmittance
                        * uAtmosphereRadii.w / ATMOSPHERE_PI
                        * cosine
                        * sunTransmittance(groundRadius(), cosine);
                }
            }

            secondOrder += luminance * weight;
            transfer += transferred * weight;
        }
    }

    // The raymarch multiplies this by the scattering
    // coefficient only (the isotropic phase is already in).
    FragColor = vec4(secondOrder / max(vec3(1.0) - transfer, vec3(1e-3)), 1.0);
}
