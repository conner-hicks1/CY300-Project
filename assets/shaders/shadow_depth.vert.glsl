#version 450 core


// =========================================================
// Shadow Depth Pass
// =========================================================
//
// Renders scene depth from a light's point of view into
// one layer of a shadow map array (a cascade of the
// directional light, or a spot light).

layout(location = 0) in vec3 aPosition;

uniform mat4 uModel;

// World -> this cascade's / spot light's clip space.
uniform mat4 uLightMatrix;

void main()
{
    gl_Position =
        uLightMatrix
        * uModel
        * vec4(aPosition, 1.0);
}
