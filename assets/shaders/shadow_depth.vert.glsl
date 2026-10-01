#version 450 core


// =========================================================
// Shadow Depth Pass
// =========================================================
//
// Renders scene depth from a light's point of view into
// one layer of a shadow map array (a cascade of the
// directional light, or a spot light).

#include "include/draw_data.glsl"

layout(location = 0) in vec3 aPosition;

// Camera-relative world -> this cascade's / spot light's
// clip space (graphics/shadows.py to_render_space).
uniform mat4 uLightMatrix;

void main()
{
    gl_Position =
        uLightMatrix
        * uDraws[aDrawIndex].model
        * vec4(aPosition, 1.0);
}
