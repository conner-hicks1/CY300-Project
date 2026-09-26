// =========================================================
// Engine Uniform Blocks
// =========================================================
//
// Uploaded once per frame by Renderer.begin_scene().
// Layout must match graphics/uniform_blocks.py; Shader
// verifies the sizes at link time.
//
// MAX_POINT_LIGHTS / MAX_SPOT_LIGHTS are injected by the
// shader preprocessor from graphics/lighting.py.


// ---------------------------------------------------------
// Camera
// ---------------------------------------------------------

layout(std140) uniform CameraBlock
{
    mat4 uView;
    mat4 uProjection;
    vec4 uViewPosition;         // xyz world position
};


// ---------------------------------------------------------
// Lights
// ---------------------------------------------------------

layout(std140) uniform LightsBlock
{
    ivec4 uLightCounts;         // x point, y spot,
                                // z has directional,
                                // w shadows enabled

    vec4 uLightParams;          // x ambient strength,
                                // y shadow bias min,
                                // z shadow bias max,
                                // w shadow map texel size

    vec4 uDirectionalDirection; // xyz direction light travels
    vec4 uDirectionalColor;     // rgb color, a intensity

    mat4 uLightSpaceMatrix;     // world -> directional light clip

    vec4 uPointPositionRange[MAX_POINT_LIGHTS];   // xyz position, w range
    vec4 uPointColorIntensity[MAX_POINT_LIGHTS];  // rgb color, a intensity

    vec4 uSpotPositionRange[MAX_SPOT_LIGHTS];     // xyz position, w range
    vec4 uSpotDirectionInner[MAX_SPOT_LIGHTS];    // xyz direction, w cos(inner)
    vec4 uSpotColorIntensity[MAX_SPOT_LIGHTS];    // rgb color, a intensity
    vec4 uSpotOuter[MAX_SPOT_LIGHTS];             // x cos(outer)
};
