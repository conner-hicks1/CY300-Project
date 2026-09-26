// =========================================================
// Lighting Functions
// =========================================================
//
// Requires include/blocks.glsl.


// ---------------------------------------------------------
// Blinn-Phong
// ---------------------------------------------------------
//
// Diffuse + specular for one light.
// L points from the surface toward the light.

vec3 blinnPhong(
    vec3 N,
    vec3 V,
    vec3 L,
    vec3 radiance,
    vec3 albedo,
    float specularStrength,
    float shininess
)
{
    vec3 H = normalize(L + V);

    float NdotL = max(dot(N, L), 0.0);

    float specular =
        NdotL > 0.0
        ? pow(max(dot(N, H), 0.0), shininess)
        : 0.0;

    return
        NdotL * albedo * radiance
        + specularStrength * specular * radiance;
}


// ---------------------------------------------------------
// Distance Attenuation
// ---------------------------------------------------------
//
// Inverse-square falloff, windowed so it reaches exactly
// zero at `range` instead of trailing off forever.

float attenuate(
    float distance,
    float range
)
{
    float ratio = distance / range;

    float window = clamp(
        1.0 - ratio * ratio * ratio * ratio,
        0.0,
        1.0
    );

    return
        (window * window)
        / (distance * distance + 1.0);
}


// ---------------------------------------------------------
// Directional Shadow
// ---------------------------------------------------------
//
// Returns 1.0 when fully lit, 0.0 when fully shadowed.
// 3x3 PCF softens the edge; slope-scaled bias (larger at
// grazing angles) fights shadow acne.

float directionalShadow(
    sampler2D shadowMap,
    vec3 worldPosition,
    vec3 N,
    vec3 L
)
{
    if (uLightCounts.w == 0)
    {
        return 1.0;
    }

    vec4 lightClip =
        uLightSpaceMatrix
        * vec4(worldPosition, 1.0);

    vec3 projected =
        lightClip.xyz / lightClip.w
        * 0.5 + 0.5;

    // Beyond the light's far plane: treat as lit.
    if (projected.z > 1.0)
    {
        return 1.0;
    }

    float bias = max(
        uLightParams.z * (1.0 - dot(N, L)),
        uLightParams.y
    );

    float texel = uLightParams.w;

    float lit = 0.0;

    for (int x = -1; x <= 1; ++x)
    {
        for (int y = -1; y <= 1; ++y)
        {
            float closest = texture(
                shadowMap,
                projected.xy + vec2(x, y) * texel
            ).r;

            lit += (projected.z - bias > closest) ? 0.0 : 1.0;
        }
    }

    return lit / 9.0;
}
