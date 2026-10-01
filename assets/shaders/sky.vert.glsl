#version 450 core

#include "include/blocks.glsl"


// =========================================================
// Sky Background
// =========================================================
//
// Fullscreen triangle placed on the far plane (depth 1),
// drawn after opaque geometry with depth test LEQUAL so it
// only fills pixels nothing else covered. Each vertex
// passes the world-space view direction through it.

out vec3 vDirection;

void main()
{
    vec2 corner = vec2(
        (gl_VertexID << 1) & 2,
        gl_VertexID & 2
    );

    vec2 ndc = corner * 2.0 - 1.0;

    // Only three vertices, so inverting here is cheap.
    mat4 inverseViewProjection = inverse(uProjection * uView);

    vec4 farPoint = inverseViewProjection * vec4(ndc, 1.0, 1.0);

    vDirection = farPoint.xyz / farPoint.w - uViewPosition.xyz;

    gl_Position = vec4(ndc, 1.0, 1.0);
}
