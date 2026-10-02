#version 450 core

#include "include/blocks.glsl"
#include "include/draw_data.glsl"


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

// Model and normal matrices come per object from the
// draw-record buffer (include/draw_data.glsl).

// Material UV tiling.
uniform vec2 uUVScale;

// Planet terrain: the tangent attribute carries tectonic
// data (planet/chunk.py), passed through untransformed.
uniform float uTerrainShading;


// =========================================================
// Main
// =========================================================

void main()
{
    DrawRecord record = uDraws[aDrawIndex];

    // "World" here is camera-relative world space (the
    // camera sits at the origin; see Renderer.prepare_draws).
    vec4 worldPosition =
        record.model
        * vec4(aPosition, 1.0);

    gl_Position =
        uProjection
        * uView
        * worldPosition;

    vWorldPosition = worldPosition.xyz;

    vNormal = mat3(record.normalMatrix) * aNormal;

    // Tangents lie in the surface, so they transform with
    // the model matrix itself (not the normal matrix).
    vTangent = uTerrainShading > 0.5
        ? aTangent
        : vec4(mat3(record.model) * aTangent.xyz, aTangent.w);

    vColor = aColor;

    vTexCoord = aTexCoord * uUVScale;
}
