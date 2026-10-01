#version 450 core


// Linear HDR color; values above 1 bloom into white after
// tone mapping.

uniform vec3 uColor;

out vec4 FragColor;

void main()
{
    FragColor = vec4(uColor, 1.0);
}
