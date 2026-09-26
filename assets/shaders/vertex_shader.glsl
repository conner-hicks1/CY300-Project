#version 330 core


// =========================================================
// Vertex Attributes
// =========================================================

layout(location = 0) in vec3 aPosition;
layout(location = 1) in vec3 aNormal;
layout(location = 2) in vec3 aColor;
layout(location = 3) in vec2 aTexCoord;


// =========================================================
// Vertex Outputs
// =========================================================

out vec3 vWorldPosition;
out vec3 vNormal;
out vec3 vColor;
out vec2 vTexCoord;


// =========================================================
// Uniforms
// =========================================================

uniform mat4 uModel;
uniform mat4 uView;
uniform mat4 uProjection;

// Inverse-transpose of uModel's upper 3x3.
uniform mat3 uNormalMatrix;


// =========================================================
// Main
// =========================================================

void main()
{
    vec4 worldPosition =
        uModel
        * vec4(aPosition, 1.0);

    gl_Position =
        uProjection
        * uView
        * worldPosition;

    vWorldPosition = worldPosition.xyz;

    vNormal = uNormalMatrix * aNormal;

    vColor = aColor;

    vTexCoord = aTexCoord;
}
