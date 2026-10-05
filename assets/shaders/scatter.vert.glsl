#version 450 core

#include "include/blocks.glsl"


// =========================================================
// Scattered Things (instanced): Rocks, Trees, Grass
// =========================================================
//
// graphics/scatter.py. Each instance stands on the planet
// (planet/scatter.py), stored in the planet's own frame
// relative to an anchor near the camera (float32 stays
// exact): rotated by the planet's frame into the world and
// placed relative to the camera. Shaded by lit.frag like
// any surface.
//
// They grow from nothing over the last fifth of their
// layer's reach, so its edge never shows.

layout(location = 0) in vec3 aPosition;
layout(location = 1) in vec3 aNormal;
layout(location = 2) in vec3 aColor;
layout(location = 3) in vec2 aTexCoord;
layout(location = 4) in vec4 aTangent;

layout(location = 6) in vec4 aInstanceOffset;   // xyz from the anchor (planet frame, m), w scale (m)
layout(location = 7) in vec4 aInstanceRotation; // quaternion: the instance -> the planet frame
layout(location = 8) in vec4 aInstanceTint;     // rgb, w a seed in [0, 1)

uniform vec4 uScatterFrame;     // quaternion: planet frame -> world
uniform vec3 uScatterAnchor;    // the anchor, camera-relative world (m)
uniform vec3 uScatterCamera;    // the camera, planet frame, from the anchor (m)
uniform float uScatterReach;    // m

// Wind: foliage sways a little (radians at the top).
uniform float uScatterSway;
uniform float uTime;

out vec3 vWorldPosition;
out vec3 vNormal;
out vec4 vTangent;
out vec3 vColor;
out vec2 vTexCoord;

vec3 rotateQ(
    vec4 q,
    vec3 v
)
{
    return v + 2.0 * cross(q.xyz, cross(q.xyz, v) + q.w * v);
}

void main()
{
    float distance_ = length(aInstanceOffset.xyz - uScatterCamera);

    float grow = 1.0 - smoothstep(0.8 * uScatterReach, uScatterReach, distance_);

    float scale = aInstanceOffset.w * grow;

    vec3 local = aPosition * scale;

    // Sway: tops move, roots do not.
    float phase = uTime * 1.3 + dot(aInstanceOffset.xyz, vec3(0.031, 0.017, 0.023));

    local.xz += uScatterSway * aPosition.y * aPosition.y * scale * vec2(sin(phase), 0.6 * sin(phase * 1.37 + 1.0));

    vec3 planet = rotateQ(aInstanceRotation, local) + aInstanceOffset.xyz;

    vec3 world = rotateQ(uScatterFrame, planet) + uScatterAnchor;

    gl_Position = uProjection * uView * vec4(world, 1.0);

    vWorldPosition = world;

    vNormal = rotateQ(uScatterFrame, rotateQ(aInstanceRotation, aNormal));

    vec3 tangent = rotateQ(uScatterFrame, rotateQ(aInstanceRotation, vec3(1.0, 0.0, 0.0)));

    vTangent = vec4(tangent, 1.0);

    vColor = aColor * aInstanceTint.rgb;

    // Coordinates for the surface detail (lit.frag
    // uScatterDetail): the instance's own meters, offset per
    // instance, so leaves and grain stay put on it.
    float seed = aInstanceTint.w * 97.0;

    vTexCoord = (aPosition.xz + aPosition.yy * vec2(0.37, 0.61)) * aInstanceOffset.w + seed;
}
