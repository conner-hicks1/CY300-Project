#version 330 core

#include "include/blocks.glsl"


// =========================================================
// Shadow Depth Pass
// =========================================================
//
// Renders scene depth from the directional light's point
// of view into the shadow map.

layout(location = 0) in vec3 aPosition;

uniform mat4 uModel;

void main()
{
    gl_Position =
        uLightSpaceMatrix
        * uModel
        * vec4(aPosition, 1.0);
}
