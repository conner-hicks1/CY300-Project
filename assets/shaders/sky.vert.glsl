#version 450 core

#include "include/blocks.glsl"


// =========================================================
// Sky Background
// =========================================================
//
// Fullscreen triangle placed on the far plane (depth 0
// under reversed-Z), drawn after opaque geometry with
// depth test GEQUAL so it only fills pixels nothing else
// covered. Each vertex
// passes the world-space view direction through it.

out vec3 vDirection;

void main()
{
    vec2 corner = vec2(
        (gl_VertexID << 1) & 2,
        gl_VertexID & 2
    );

    vec2 ndc = corner * 2.0 - 1.0;

    // View-space ray through this corner, read straight
    // from the projection's focal lengths (the reversed-Z
    // infinite projection has no invertible far plane).
    vec3 viewRay = vec3(
        ndc.x / uProjection[0][0],
        ndc.y / uProjection[1][1],
        -1.0
    );

    // uView is rotation-only (camera-relative rendering);
    // its transpose is its inverse.
    vDirection = transpose(mat3(uView)) * viewRay;

    // Depth 0 = the far plane under reversed-Z.
    gl_Position = vec4(ndc, 0.0, 1.0);
}
