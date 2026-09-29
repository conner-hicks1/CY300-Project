// =========================================================
// Physically Based Shading (Cook-Torrance)
// =========================================================
//
// Metallic-roughness model, as in glTF 2.0:
//
//   specular: D (GGX normal distribution)
//           * G (Smith / Schlick-GGX shadowing-masking)
//           * F (Schlick Fresnel)
//           / (4 NdotV NdotL)
//   diffuse:  Lambert, scaled by (1 - F)(1 - metallic)
//
// References: Karis, "Real Shading in Unreal Engine 4"
// (2013); Burley, "Physically Based Shading at Disney"
// (2012).

#define PI 3.14159265359


// ---------------------------------------------------------
// BRDF Terms
// ---------------------------------------------------------

float distributionGGX(
    float NdotH,
    float roughness
)
{
    // Disney/UE4 remap: alpha = roughness^2.
    float a = roughness * roughness;
    float a2 = a * a;

    float denominator = NdotH * NdotH * (a2 - 1.0) + 1.0;

    return a2 / (PI * denominator * denominator);
}

float geometrySchlickGGX(
    float NdotX,
    float k
)
{
    return NdotX / (NdotX * (1.0 - k) + k);
}

float geometrySmith(
    float NdotV,
    float NdotL,
    float roughness
)
{
    // k for direct lighting (UE4).
    float r = roughness + 1.0;
    float k = (r * r) / 8.0;

    return
        geometrySchlickGGX(NdotV, k)
        * geometrySchlickGGX(NdotL, k);
}

vec3 fresnelSchlick(
    float cosTheta,
    vec3 F0
)
{
    return F0 + (1.0 - F0) * pow(clamp(1.0 - cosTheta, 0.0, 1.0), 5.0);
}

// For image-based lighting: rough surfaces have a weaker
// Fresnel peak at grazing angles (Lagarde).
vec3 fresnelSchlickRoughness(
    float cosTheta,
    vec3 F0,
    float roughness
)
{
    return
        F0
        + (max(vec3(1.0 - roughness), F0) - F0)
        * pow(clamp(1.0 - cosTheta, 0.0, 1.0), 5.0);
}


// ---------------------------------------------------------
// Direct Light
// ---------------------------------------------------------
//
// Outgoing radiance toward V from one light arriving
// from direction L with `radiance`.

struct Surface
{
    vec3 N;
    vec3 V;
    vec3 baseColor;
    float metallic;
    float roughness;
    vec3 F0;
};

vec3 shadeDirect(
    Surface s,
    vec3 L,
    vec3 radiance
)
{
    float NdotL = max(dot(s.N, L), 0.0);

    if (NdotL <= 0.0)
    {
        return vec3(0.0);
    }

    vec3 H = normalize(s.V + L);

    float NdotV = max(dot(s.N, s.V), 1e-4);
    float NdotH = max(dot(s.N, H), 0.0);
    float HdotV = max(dot(H, s.V), 0.0);

    float D = distributionGGX(NdotH, s.roughness);
    float G = geometrySmith(NdotV, NdotL, s.roughness);
    vec3 F = fresnelSchlick(HdotV, s.F0);

    vec3 specular = (D * G * F) / (4.0 * NdotV * NdotL + 1e-4);

    vec3 kD = (vec3(1.0) - F) * (1.0 - s.metallic);

    return (kD * s.baseColor / PI + specular) * radiance * NdotL;
}


// ---------------------------------------------------------
// Distance Attenuation
// ---------------------------------------------------------
//
// Inverse-square falloff, windowed so it reaches exactly
// zero at `range` instead of trailing off forever
// (Karis 2013).

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
