#version 450 core


// =========================================================
// Scattered Things into the Shadow Maps
// =========================================================
//
// scatter.vert.glsl's placement, into a light's clip space.

layout(location = 0) in vec3 aPosition;

layout(location = 6) in vec4 aInstanceOffset;
layout(location = 7) in vec4 aInstanceRotation;

uniform mat4 uLightMatrix;

uniform vec4 uScatterFrame;
uniform vec3 uScatterAnchor;
uniform vec3 uScatterCamera;
uniform float uScatterReach;

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

    float scale = aInstanceOffset.w * (1.0 - smoothstep(0.8 * uScatterReach, uScatterReach, distance_));

    vec3 planet = rotateQ(aInstanceRotation, aPosition * scale) + aInstanceOffset.xyz;

    vec3 world = rotateQ(uScatterFrame, planet) + uScatterAnchor;

    gl_Position = uLightMatrix * vec4(world, 1.0);
}
