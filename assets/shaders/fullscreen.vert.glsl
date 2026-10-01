#version 450 core


// =========================================================
// Fullscreen Triangle
// =========================================================
//
// No vertex buffer: gl_VertexID 0, 1, 2 produce a triangle
// covering (-1,-1) .. (3,3) in clip space, which contains
// the whole screen. UVs span 0..1 over the visible part.

out vec2 vTexCoord;

void main()
{
    vec2 corner = vec2(
        (gl_VertexID << 1) & 2,
        gl_VertexID & 2
    );

    vTexCoord = corner;

    gl_Position = vec4(
        corner * 2.0 - 1.0,
        0.0,
        1.0
    );
}
