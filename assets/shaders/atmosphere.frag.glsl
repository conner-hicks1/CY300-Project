#version 450 core

#include "include/blocks.glsl"
#include "include/bodies.glsl"
#include "include/rings.glsl"

#define ATMOSPHERE_ECLIPSES

#include "include/atmosphere.glsl"
#include "include/clouds.glsl"

// Sun disc brightness relative to its illuminance; far
// brighter than the sky, so it blooms (graphics/atmosphere.py).
const float SUN_DISC_BRIGHTNESS = 40.0;


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

// 1 = another body's air, seen from afar: drawn before the
// camera's own (whose sky goes over it), so it leaves the
// background visible around it and draws no sun.
uniform float uDistantAtmosphere;

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

    vec3 worldDirection = normalize(transpose(mat3(uView)) * viewRay);

    // Angle a pixel covers (cloud detail fades below it).
    float pixelAngle = length(fwidth(worldDirection));

    // Into atmosphere space (a flattened giant becomes a
    // sphere); world distances scale by `stretch`.
    vec3 direction = toAtmosphereSpace(worldDirection);

    float stretch = length(direction);

    direction /= stretch;

    // Reversed-Z infinite projection: depth = near / view
    // distance along -Z; 0 = nothing drawn.
    float depth = texture(uSceneDepth, vTexCoord).r;

    bool geometry = depth > 0.0;

    float near = uProjection[3][2];

    float geometryDistance = geometry
        ? length(viewRay * (near / depth)) * 0.001 * stretch
        : 0.0;

    vec3 origin = toAtmosphereSpace(-uPlanetCenter.xyz);

    float start;
    float end;

    bool inside = atmosphereSegment(origin, direction, start, end);

    // Giants have no ground. From above, their cloud tops
    // act as one; below them the deck overhead closes off
    // the sky and the air thickens on down (taken as opaque
    // at uShape.z below the tops): fog all around.
    bool underTops = noSolidSurface() && length(origin) < groundRadius();

    float groundHit = underTops ? -1.0 : raySphere(origin, direction, groundRadius());

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

    if (geometry)
    {
        end = min(end, geometryDistance);

        // A giant's cloud tops seen from above: end at the
        // true surface (the mesh sags below it between
        // vertices).
        if (noSolidSurface() && !underTops && groundHit > 0.0)
        {
            end = min(end, groundHit);
        }
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
        inScattered = scatteringWithClouds(
            origin,
            direction,
            start,
            end,
            int(uOzoneParams.y),
            0.5,
            pixelAngle,
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

    // Another body's air against space: its glow over
    // whatever is behind it.
    if (uDistantAtmosphere > 0.5)
    {
        outInScattered = vec4(inScattered, 0.0);
        outTransmittance = vec4(transmittance, 1.0);

        return;
    }

    // Sky: the sun's disc where nothing blocks it.
    if (uAtmosphereSunDirection.w > 0.5 && groundHit < 0.0)
    {
        float cosAngle = dot(direction, uAtmosphereSunDirection.xyz);

        float radius = uMieAbsorption.w;

        // cos(x) ~ 1 - x^2/2: GPU cos() is too coarse at
        // a quarter of a degree (it returns ~1.0).
        float inner = 1.0 - 0.5 * radius * radius;
        float outer = 1.0 - 0.5 * (1.25 * radius) * (1.25 * radius);

        float disc = smoothstep(outer, inner, cosAngle);

        inScattered +=
            disc
            * uSunIlluminance.rgb
            * SUN_DISC_BRIGHTNESS
            * transmittance;
    }

    // Over the (black) background: whatever lies beyond the
    // air shows through it (another body's glowing limb).
    outInScattered = vec4(inScattered, 1.0);
    outTransmittance = vec4(transmittance, 0.0);
}
