#version 330 core


// =========================================================
// Limits
// =========================================================
//
// Must match MAX_POINT_LIGHTS / MAX_SPOT_LIGHTS in
// graphics/lighting.py.

#define MAX_POINT_LIGHTS 8
#define MAX_SPOT_LIGHTS 4


// =========================================================
// Light Types
// =========================================================

struct DirectionalLight
{
    vec3 direction;     // Direction the light travels.
    vec3 color;
    float intensity;
};

struct PointLight
{
    vec3 position;
    vec3 color;
    float intensity;
    float range;
};

struct SpotLight
{
    vec3 position;
    vec3 direction;     // Direction the cone points.
    vec3 color;
    float intensity;
    float range;
    float innerCutoff;  // cos(inner half-angle)
    float outerCutoff;  // cos(outer half-angle)
};


// =========================================================
// Fragment Inputs
// =========================================================

in vec3 vWorldPosition;
in vec3 vNormal;
in vec3 vColor;
in vec2 vTexCoord;


// =========================================================
// Fragment Outputs
// =========================================================

out vec4 FragColor;


// =========================================================
// Uniforms
// =========================================================

uniform sampler2D uTexture;

uniform vec3 uViewPosition;

uniform float uAmbientStrength;

uniform bool uHasDirectionalLight;
uniform DirectionalLight uDirectionalLight;

uniform int uPointLightCount;
uniform PointLight uPointLights[MAX_POINT_LIGHTS];

uniform int uSpotLightCount;
uniform SpotLight uSpotLights[MAX_SPOT_LIGHTS];

// Material
uniform float uSpecularStrength;
uniform float uShininess;


// =========================================================
// Blinn-Phong
// =========================================================
//
// Diffuse + specular for one light.
// L points from the surface toward the light.

vec3 blinnPhong(
    vec3 N,
    vec3 V,
    vec3 L,
    vec3 radiance,
    vec3 albedo
)
{
    vec3 H = normalize(L + V);

    float NdotL = max(dot(N, L), 0.0);

    float specular =
        NdotL > 0.0
        ? pow(max(dot(N, H), 0.0), uShininess)
        : 0.0;

    return
        NdotL * albedo * radiance
        + uSpecularStrength * specular * radiance;
}


// =========================================================
// Distance Attenuation
// =========================================================
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


// =========================================================
// Main
// =========================================================

void main()
{
    vec3 albedo =
        texture(uTexture, vTexCoord).rgb
        * vColor;

    vec3 N = normalize(vNormal);
    vec3 V = normalize(uViewPosition - vWorldPosition);

    vec3 color = uAmbientStrength * albedo;

    // -----------------------------------------------------
    // Directional
    // -----------------------------------------------------

    if (uHasDirectionalLight)
    {
        vec3 L = normalize(-uDirectionalLight.direction);

        vec3 radiance =
            uDirectionalLight.color
            * uDirectionalLight.intensity;

        color += blinnPhong(N, V, L, radiance, albedo);
    }

    // -----------------------------------------------------
    // Point Lights
    // -----------------------------------------------------

    for (int i = 0; i < uPointLightCount; ++i)
    {
        vec3 toLight =
            uPointLights[i].position
            - vWorldPosition;

        float distance = length(toLight);

        vec3 L = toLight / distance;

        vec3 radiance =
            uPointLights[i].color
            * uPointLights[i].intensity
            * attenuate(distance, uPointLights[i].range);

        color += blinnPhong(N, V, L, radiance, albedo);
    }

    // -----------------------------------------------------
    // Spot Lights
    // -----------------------------------------------------

    for (int i = 0; i < uSpotLightCount; ++i)
    {
        vec3 toLight =
            uSpotLights[i].position
            - vWorldPosition;

        float distance = length(toLight);

        vec3 L = toLight / distance;

        // Angle between the cone axis and this fragment.
        float theta = dot(
            -L,
            normalize(uSpotLights[i].direction)
        );

        float cone = clamp(
            (theta - uSpotLights[i].outerCutoff)
            / max(
                uSpotLights[i].innerCutoff - uSpotLights[i].outerCutoff,
                1e-4
            ),
            0.0,
            1.0
        );

        vec3 radiance =
            uSpotLights[i].color
            * uSpotLights[i].intensity
            * attenuate(distance, uSpotLights[i].range)
            * cone;

        color += blinnPhong(N, V, L, radiance, albedo);
    }

    FragColor = vec4(color, 1.0);
}
