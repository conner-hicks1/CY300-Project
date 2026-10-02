#version 450 core

#include "include/cubemap.glsl"
#include "include/atmosphere.glsl"


// =========================================================
// Environment Capture: Atmosphere
// =========================================================
//
// The sky as seen from the camera, baked into one face of
// the environment cubemap for image-based lighting. Below
// the horizon: sunlit ground of the atmosphere's average
// albedo, seen through the air. No sun disc (the
// directional light already provides direct sunlight).

in vec2 vTexCoord;

out vec4 FragColor;

const int STEPS = 32;

void main()
{
    vec3 direction = cubeFaceDirection(
        uFace,
        vTexCoord * 2.0 - 1.0
    );

    vec3 origin = -uPlanetCenter.xyz;

    float start;
    float end;

    vec3 color = vec3(0.0);

    if (uAtmosphereSunDirection.w < 0.5)
    {
        FragColor = vec4(color, 1.0);
        return;
    }

    float groundHit = raySphere(origin, direction, groundRadius());

    vec3 transmittance = vec3(1.0);

    if (atmosphereSegment(origin, direction, start, end))
    {
        if (groundHit > 0.0)
        {
            end = min(end, groundHit);
        }

        color = integrateScattering(
            origin,
            direction,
            start,
            end,
            STEPS,
            0.5,
            transmittance
        );
    }

    if (groundHit > 0.0)
    {
        vec3 normal = normalize(origin + groundHit * direction);

        float cosine = dot(normal, uAtmosphereSunDirection.xyz);

        if (cosine > 0.0)
        {
            color +=
                transmittance
                * uAtmosphereRadii.w / ATMOSPHERE_PI
                * cosine
                * sunTransmittance(groundRadius(), cosine);
        }

        color +=
            transmittance
            * uAtmosphereRadii.w / ATMOSPHERE_PI
            * thickAtmosphereWeight()
            * diffuseDaylight(groundRadius(), cosine);
    }

    FragColor = vec4(color * uSunIlluminance.rgb, 1.0);
}
