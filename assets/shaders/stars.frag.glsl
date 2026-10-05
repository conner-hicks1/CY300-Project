#version 450 core


// =========================================================
// Stars (points): a small Gaussian per star, additive
// =========================================================

in vec3 vRadiance;
flat in float vSize;

out vec4 FragColor;

const float SIGMA = 0.65;   // pixels (stars.vert.glsl)

void main()
{
    vec2 offset = (gl_PointCoord - 0.5) * vSize;

    float weight = exp(-dot(offset, offset) / (2.0 * SIGMA * SIGMA));

    FragColor = vec4(vRadiance * weight, 0.0);
}
