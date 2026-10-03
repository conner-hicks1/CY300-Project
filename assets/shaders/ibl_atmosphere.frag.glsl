#version 450 core

#include "include/cubemap.glsl"
#include "include/atmosphere.glsl"
#include "include/clouds.glsl"


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

    vec3 origin = toAtmosphereSpace(-uPlanetCenter.xyz);

    direction = normalize(toAtmosphereSpace(direction));

    float start;
    float end;

    vec3 color = vec3(0.0);

    if (uAtmosphereSunDirection.w < 0.5)
    {
        FragColor = vec4(color, 1.0);
        return;
    }

    // Giants: cloud tops as the ground from above; fog all
    // around below them (see atmosphere.frag.glsl).
    bool underTops = noSolidSurface() && length(origin) < groundRadius();

    float groundHit = underTops ? -1.0 : raySphere(origin, direction, groundRadius());

    vec3 transmittance = vec3(1.0);

    if (atmosphereSegment(origin, direction, start, end))
    {
        if (underTops)
        {
            float ceiling = raySphere(origin, direction, groundRadius());
            float deep = raySphere(origin, direction, groundRadius() - uShape.z);

            if (ceiling > 0.0)
            {
                end = min(end, ceiling);
            }

            if (deep > 0.0)
            {
                end = min(end, deep);
            }
        }

        if (groundHit > 0.0)
        {
            end = min(end, groundHit);
        }

        // Clouds too: an overcast sky lights the scene grey.
        color = scatteringWithClouds(
            origin,
            direction,
            start,
            end,
            STEPS,
            0.5,
            0.02,
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
