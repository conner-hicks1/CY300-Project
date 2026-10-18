#version 450 core

#include "include/blocks.glsl"


// =========================================================
// Glow (geometry)
// =========================================================
//
// Light-emitting gas: rocket exhaust, the plasma round a
// vessel coming in through the air. Camera-relative, like
// every draw; drawn additively over the scene
// (systems/render_system.py).

layout(location = 0) in vec3 aPosition;
layout(location = 1) in vec3 aNormal;
layout(location = 3) in vec2 aTexCoord;

uniform mat4 uModel;

out vec3 vWorldPosition;    // camera-relative, m
out vec3 vNormal;
out vec2 vTexCoord;

void main()
{
    vec4 world = uModel * vec4(aPosition, 1.0);

    vWorldPosition = world.xyz;

    // (Scaled unevenly: the inverse transpose keeps the
    // normals square to the surface.)
    vNormal = normalize(transpose(inverse(mat3(uModel))) * aNormal);

    vTexCoord = aTexCoord;

    gl_Position = uProjection * uView * world;
}
