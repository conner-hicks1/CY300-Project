// =========================================================
// Shadow Sampling
// =========================================================
//
// Requires include/blocks.glsl.
//
// Both shadow maps are depth texture arrays sampled with
// hardware comparison (sampler2DArrayShadow): each tap
// already returns a bilinearly filtered lit fraction, and
// 3x3 taps on top give soft edges.
//
// Acne is avoided with a *normal offset*: the receiver
// position is pushed along its geometric normal by about
// one shadow texel before lookup, plus a tiny depth bias.

uniform sampler2DArrayShadow uCascadeShadowMaps;
uniform sampler2DArrayShadow uSpotShadowMaps;


// Shadow clip position -> (texture uv, depth). Shadow
// matrices already map depth to [0, 1] (clip control is
// GL_ZERO_TO_ONE; see graphics/shadows.py to_render_space).
vec3 shadowCoordinates(
    vec4 clip
)
{
    vec3 ndc = clip.xyz / clip.w;

    return vec3(ndc.xy * 0.5 + 0.5, ndc.z);
}


float shadowPCF(
    sampler2DArrayShadow shadowMap,
    vec3 projected,
    float layer
)
{
    float texel = 1.0 / float(textureSize(shadowMap, 0).x);

    float reference = projected.z - uLightParams.y;

    float lit = 0.0;

    for (int x = -1; x <= 1; ++x)
    {
        for (int y = -1; y <= 1; ++y)
        {
            lit += texture(
                shadowMap,
                vec4(
                    projected.xy + vec2(x, y) * texel,
                    layer,
                    reference
                )
            );
        }
    }

    return lit / 9.0;
}


// ---------------------------------------------------------
// Directional (cascaded)
// ---------------------------------------------------------

int cascadeIndex(
    float viewDepth
)
{
    for (int i = 0; i < uLightCounts.w; ++i)
    {
        if (viewDepth < uCascadeSplits[i])
        {
            return i;
        }
    }

    // Beyond the shadow distance.
    return -1;
}

float directionalShadow(
    vec3 worldPosition,
    vec3 geometricNormal,
    float viewDepth
)
{
    int cascade = cascadeIndex(viewDepth);

    if (cascade < 0)
    {
        return 1.0;
    }

    vec3 offsetPosition =
        worldPosition
        + geometricNormal
        * uCascadeTexelSizes[cascade]
        * uLightParams.z;

    vec4 clip =
        uCascadeMatrices[cascade]
        * vec4(offsetPosition, 1.0);

    vec3 projected = shadowCoordinates(clip);

    if (projected.z > 1.0)
    {
        return 1.0;
    }

    return shadowPCF(
        uCascadeShadowMaps,
        projected,
        float(cascade)
    );
}


// ---------------------------------------------------------
// Spot
// ---------------------------------------------------------

float spotShadow(
    int light,
    vec3 worldPosition,
    vec3 geometricNormal,
    float distanceToLight
)
{
    float layer = uSpotParams[light].y;

    if (layer < 0.0)
    {
        return 1.0;
    }

    // Texels grow with distance under a perspective
    // projection.
    float texelWorld = distanceToLight * uSpotParams[light].z;

    vec3 offsetPosition =
        worldPosition
        + geometricNormal
        * texelWorld
        * uLightParams.z;

    vec4 clip =
        uSpotMatrices[light]
        * vec4(offsetPosition, 1.0);

    if (clip.w <= 0.0)
    {
        return 1.0;
    }

    vec3 projected = shadowCoordinates(clip);

    return shadowPCF(
        uSpotShadowMaps,
        projected,
        layer
    );
}


// ---------------------------------------------------------
// Debug
// ---------------------------------------------------------

vec3 cascadeDebugColor(
    float viewDepth
)
{
    int cascade = cascadeIndex(viewDepth);

    if (cascade == 0) return vec3(1.0, 0.35, 0.35);
    if (cascade == 1) return vec3(0.35, 1.0, 0.35);
    if (cascade == 2) return vec3(0.35, 0.35, 1.0);
    if (cascade == 3) return vec3(1.0, 1.0, 0.35);

    return vec3(1.0);
}
