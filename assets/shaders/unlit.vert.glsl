#version 330 core

#include "include/blocks.glsl"


// =========================================================
// Unlit
// =========================================================
//
// Flat-colored geometry (light gizmos, debug shapes).

layout(location = 0) in vec3 aPosition;

uniform mat4 uModel;

void main()
{
    gl_Position =
        uProjection
        * uView
        * uModel
        * vec4(aPosition, 1.0);
}
