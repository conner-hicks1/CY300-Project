// =========================================================
// Engine Uniform Blocks
// =========================================================
//
// Uploaded once per frame by Renderer.begin_scene().
// Layout must match graphics/uniform_blocks.py; Shader
// verifies the sizes at link time.
//
// MAX_POINT_LIGHTS / MAX_SPOT_LIGHTS / MAX_CASCADES are
// injected by the shader preprocessor from
// graphics/lighting.py.


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
                                // w cascade count (0 = no
                                //   directional shadows)

    vec4 uLightParams;          // x IBL intensity,
                                // y shadow depth bias,
                                // z normal offset (texels),
                                // w visualize cascades

    vec4 uDirectionalDirection; // xyz direction light travels
    vec4 uDirectionalColor;     // rgb color, a intensity

    vec4 uCascadeSplits;        // view-space far distance per cascade
    vec4 uCascadeTexelSizes;    // world size of a shadow texel per cascade
    mat4 uCascadeMatrices[MAX_CASCADES];

    vec4 uPointPositionRange[MAX_POINT_LIGHTS];   // xyz position, w range
    vec4 uPointColorIntensity[MAX_POINT_LIGHTS];  // rgb color, a intensity

    vec4 uSpotPositionRange[MAX_SPOT_LIGHTS];     // xyz position, w range
    vec4 uSpotDirectionInner[MAX_SPOT_LIGHTS];    // xyz direction, w cos(inner)
    vec4 uSpotColorIntensity[MAX_SPOT_LIGHTS];    // rgb color, a intensity
    vec4 uSpotParams[MAX_SPOT_LIGHTS];            // x cos(outer), y shadow layer
                                                  // (-1 = none), z texel scale
    mat4 uSpotMatrices[MAX_SPOT_LIGHTS];

    mat4 uTerrainShadowMatrix;                    // world -> terrain shadow clip
    vec4 uTerrainShadowParams;                    // x texel world size, y 1 = on,
                                                  // z layer in the cascade maps
};
