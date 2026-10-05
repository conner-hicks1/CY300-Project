#version 450 core

#include "include/blocks.glsl"


// =========================================================
// Stars and Distant Bodies (points)
// =========================================================
//
// One point per star (graphics/star_field.py), at infinity:
// a direction, drawn with depth 0 in the reversed-Z buffer
// so anything in front hides it. Planets and moons too
// small to see as discs come through here too, as points of
// light; they fade out as their disc grows past a pixel.

layout(location = 0) in vec4 aStar;     // xyz direction (engine axes), w illuminance
layout(location = 1) in vec4 aColor;    // rgb (luminance 1), w angular radius (rad)

uniform vec2 uViewportSize;             // pixels

// Twinkling (scintillation) seen through air: amplitude at
// the zenith (0 in space), the camera's up, the time (s).
uniform float uTwinkle;
uniform vec3 uCameraUp;
uniform float uTime;

// Points on the far side of the camera's planet: its center
// (km, camera-relative) and radius (km); hidden. (Its
// geometry hides them too once it streams in.)
uniform vec4 uPlanetSphere;

out vec3 vRadiance;         // per pixel at the sprite's center
flat out float vSize;

const float SPRITE = 5.0;   // pixels
const float SIGMA = 0.65;   // pixels: the eye's blur of a point

float hash(
    float n
)
{
    return fract(sin(n) * 43758.5453);
}

void main()
{
    vec3 direction = normalize(aStar.xyz);

    vec3 view = mat3(uView) * direction;

    // (w = 0: a direction; the infinite reversed-Z
    // projection puts it at depth 0.)
    gl_Position = uProjection * vec4(view, 0.0);

    // Pixel solid angle near the center of view.
    float pixelX = 2.0 / (uProjection[0][0] * uViewportSize.x);
    float pixelY = 2.0 / (uProjection[1][1] * uViewportSize.y);

    float illuminance = aStar.w;

    // Resolved discs are drawn as geometry.
    float pixel = sqrt(pixelX * pixelY);

    illuminance *= 1.0 - smoothstep(0.5, 1.5, aColor.w / pixel);

    // Behind the camera's planet.
    if (uPlanetSphere.w > 0.0)
    {
        float along = dot(direction, uPlanetSphere.xyz);

        float miss = dot(uPlanetSphere.xyz, uPlanetSphere.xyz) - along * along;

        if (along > 0.0 && miss < uPlanetSphere.w * uPlanetSphere.w)
        {
            illuminance = 0.0;
        }
    }

    if (uTwinkle > 0.0)
    {
        // Through more air near the horizon (airmass), each
        // color its own way; stars twinkle, planets (small
        // discs, not points) much less.
        float elevation = dot(direction, uCameraUp);

        float airmass = 1.0 / max(elevation + 0.05, 0.05);

        float amount = uTwinkle * min(sqrt(airmass), 4.0) * (aColor.w > 0.0 ? 0.2 : 1.0);

        float seed = float(gl_VertexID) * 1.618;

        vec3 phase = vec3(hash(seed), hash(seed + 1.3), hash(seed + 2.7)) * 6.2832;

        float rate = 9.0 + 5.0 * hash(seed + 4.1);

        vec3 flicker =
            sin(uTime * rate + phase)
            + 0.6 * sin(uTime * rate * 2.13 + phase.yzx)
            + 0.4 * sin(uTime * rate * 3.71 + phase.zxy);

        // Color scintillates near the horizon only.
        flicker = mix(vec3(dot(flicker, vec3(1.0 / 3.0))), flicker, smoothstep(0.4, 0.05, elevation));

        vRadiance = aColor.rgb * illuminance * max(vec3(0.0), 1.0 + amount * flicker / 2.0);
    }
    else
    {
        vRadiance = aColor.rgb * illuminance;
    }

    // Gaussian-weighted over the sprite: radiance at its
    // center for a unit sum.
    vRadiance /= 2.0 * 3.14159265 * SIGMA * SIGMA * pixelX * pixelY;

    vSize = SPRITE;

    gl_PointSize = illuminance > 0.0 ? SPRITE : 0.0;
}
