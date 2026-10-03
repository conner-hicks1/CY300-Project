#version 450 core

#include "include/blocks.glsl"


// =========================================================
// Planetary Rings (geometry)
// =========================================================
//
// A unit disc in the planet's equatorial (XZ) plane,
// scaled to the rings' outer radius by uModel (camera-
// relative, like every draw).

layout(location = 0) in vec3 aPosition;

uniform mat4 uModel;

out vec2 vDisc;             // position on the unit disc
out vec3 vWorldPosition;    // camera-relative, m
out vec3 vNormal;           // the ring plane's normal

void main()
{
    vec4 world = uModel * vec4(aPosition, 1.0);

    vDisc = aPosition.xz;
    vWorldPosition = world.xyz;
    vNormal = normalize(mat3(uModel) * vec3(0.0, 1.0, 0.0));

    gl_Position = uProjection * uView * world;
}
