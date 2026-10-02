#version 450 core

#include "include/blocks.glsl"
#include "include/atmosphere.glsl"


// =========================================================
// Sky and Aerial Perspective
// =========================================================
//
// One fullscreen pass after the opaque scene. Per pixel:
// march the view ray through the atmosphere, up to the
// surface (from the depth buffer) or out to space.
//
// Output uses dual-source blending
// (glBlendFunc(GL_ONE, GL_SRC1_COLOR)):
//
//     result = inScattered + scene * transmittance
//
// Sky pixels have transmittance 0 (the clear color is
// replaced); geometry is dimmed and hazed by the air in
// front of it.

in vec2 vTexCoord;

uniform sampler2D uSceneDepth;

layout(location = 0, index = 0) out vec4 outInScattered;
layout(location = 0, index = 1) out vec4 outTransmittance;


void main()
{
    vec2 ndc = vTexCoord * 2.0 - 1.0;

    // View-space ray through this pixel (z = -1).
    vec3 viewRay = vec3(
        ndc.x / uProjection[0][0],
        ndc.y / uProjection[1][1],
        -1.0
    );

    vec3 direction = normalize(transpose(mat3(uView)) * viewRay);

    // Reversed-Z infinite projection: depth = near / view
    // distance along -Z; 0 = nothing drawn.
    float depth = texture(uSceneDepth, vTexCoord).r;

    bool geometry = depth > 0.0;

    float near = uProjection[3][2];

    float geometryDistance = geometry
        ? length(viewRay * (near / depth)) * 0.001
        : 0.0;

    vec3 origin = -uPlanetCenter.xyz;

    float start;
    float end;

    bool inside = atmosphereSegment(origin, direction, start, end);

    float groundHit = raySphere(origin, direction, groundRadius());

    if (geometry)
    {
        end = min(end, geometryDistance);
    }
    else if (groundHit > 0.0)
    {
        // Below the horizon with no terrain drawn (still
        // streaming): stop at sea level.
        end = min(end, groundHit);
    }

    vec3 inScattered = vec3(0.0);
    vec3 transmittance = vec3(1.0);

    if (inside && end > start)
    {
        // Samples at step midpoints. (Per-pixel jitter is
        // not needed: steps are packed where the air is
        // dense, so there is no banding to hide, and it
        // showed as a fine grid on the ocean from space.)
        inScattered = integrateScattering(
            origin,
            direction,
            start,
            end,
            int(uOzoneParams.y),
            0.5,
            transmittance
        ) * uSunIlluminance.rgb * uAtmosphereSunDirection.w;
    }

    if (geometry && uOzoneParams.z < 0.5)
    {
        // Haze over geometry switched off (terrain data
        // views): show the surface as is.
        outInScattered = vec4(0.0);
        outTransmittance = vec4(1.0);

        return;
    }

    if (geometry)
    {
        outInScattered = vec4(inScattered, 0.0);
        outTransmittance = vec4(transmittance, 1.0);

        return;
    }

    // Sky: the sun's disc where nothing blocks it.
    if (uAtmosphereSunDirection.w > 0.5 && groundHit < 0.0)
    {
        float cosAngle = dot(direction, uAtmosphereSunDirection.xyz);

        float radius = uMieParams.w;

        // cos(x) ~ 1 - x^2/2: GPU cos() is too coarse at
        // a quarter of a degree (it returns ~1.0).
        float inner = 1.0 - 0.5 * radius * radius;
        float outer = 1.0 - 0.5 * (1.25 * radius) * (1.25 * radius);

        float disc = smoothstep(outer, inner, cosAngle);

        inScattered +=
            disc
            * uSunIlluminance.rgb
            * uSunIlluminance.w
            * transmittance;
    }

    outInScattered = vec4(inScattered, 1.0);
    outTransmittance = vec4(0.0);
}
