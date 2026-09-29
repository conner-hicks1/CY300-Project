#version 330 core

#include "include/cubemap.glsl"
#include "include/sky.glsl"


// =========================================================
// Environment Capture
// =========================================================
//
// Bakes the procedural sky into one face of the
// environment cubemap. The sun disc is left out; the
// directional light already provides the sun's direct
// light (see include/sky.glsl).

in vec2 vTexCoord;

out vec4 FragColor;

void main()
{
    vec3 direction = cubeFaceDirection(
        uFace,
        vTexCoord * 2.0 - 1.0
    );

    FragColor = vec4(
        skyRadiance(direction, false),
        1.0
    );
}
