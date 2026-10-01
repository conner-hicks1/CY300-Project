#version 450 core

#include "include/blocks.glsl"


// =========================================================
// Vertex Attributes
// =========================================================
//
// Matches graphics/mesh.py standard_layout().

layout(location = 0) in vec3 aPosition;
layout(location = 1) in vec3 aNormal;
layout(location = 2) in vec3 aColor;
layout(location = 3) in vec2 aTexCoord;
layout(location = 4) in vec4 aTangent;   // w = bitangent sign


// =========================================================
// Vertex Outputs
// =========================================================

out vec3 vWorldPosition;
out vec3 vNormal;
out vec4 vTangent;
out vec3 vColor;
out vec2 vTexCoord;


// =========================================================
// Uniforms
// =========================================================

uniform mat4 uModel;

// Cofactor of uModel's upper 3x3 (see math3d/matrices.py).
uniform mat3 uNormalMatrix;

// Material UV tiling.
uniform vec2 uUVScale;


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

    // Tangents lie in the surface, so they transform with
    // the model matrix itself (not the normal matrix).
    vTangent = vec4(
        mat3(uModel) * aTangent.xyz,
        aTangent.w
    );

    vColor = aColor;

    vTexCoord = aTexCoord * uUVScale;
}
