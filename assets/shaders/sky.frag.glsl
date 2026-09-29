#version 330 core

#include "include/sky.glsl"


in vec3 vDirection;

out vec4 FragColor;

void main()
{
    FragColor = vec4(
        skyRadiance(vDirection, true),
        1.0
    );
}
