#version 450 core

#include "include/blocks.glsl"
#include "include/bodies.glsl"
#include "include/rings.glsl"


// =========================================================
// Planetary Rings (shading)
// =========================================================
//
// A layer of icy particles, lit by the sun, in front of
// whatever is behind it (premultiplied alpha: the scene is
// dimmed by exp(-tau / mu) and the ring's own light added).
//
// Single scattering in a thin layer of normal optical depth
// tau (as for Saturn's rings), with mu = |cos| of the view
// and sun angles to the ring plane:
//
//   lit face     I/F = w/4 P mu_s / (mu_s + mu_v)
//                        (1 - exp(-tau (1/mu_s + 1/mu_v)))
//   unlit face   I/F = w/4 P mu_s / (mu_s - mu_v)
//                        (exp(-tau/mu_s) - exp(-tau/mu_v))
//
// so dense rings (Saturn's B ring) are bright on the lit
// face and dark from below, while sparse ones glow from
// behind (forward scattering by small particles). The
// planet's shadow falls across them.

in vec2 vDisc;
in vec3 vWorldPosition;
in vec3 vNormal;

out vec4 FragColor;

uniform vec3 uRingColor;            // particle albedo (rgb)

float ringHash(
    float x
)
{
    uint n = uint(int(x) + 100000) * 1597334673u;

    n ^= n >> 16u;
    n *= 2246822519u;
    n ^= n >> 13u;

    return float(n) * (1.0 / 4294967295.0);
}

// Fine ringlets (below the profile's resolution), faded
// out where they would be smaller than a pixel.
float ringlets(
    float radius,
    float footprint
)
{
    float sum = 0.0;
    float total = 0.0;
    float weight = 0.5;
    float scale = 1.0 / 40.0;           // 40 km ringlets, then finer

    for (int octave = 0; octave < 5; ++octave)
    {
        float fade = 1.0 - smoothstep(0.25, 0.6, footprint * scale);

        if (fade <= 0.0)
        {
            break;
        }

        float x = radius * scale;
        float i = floor(x);
        float f = x - i;

        f = f * f * (3.0 - 2.0 * f);

        float n = mix(ringHash(i + float(octave) * 977.0), ringHash(i + 1.0 + float(octave) * 977.0), f);

        sum += weight * fade * (n - 0.5);
        total += weight;

        weight *= 0.6;
        scale *= 2.3;
    }

    return total > 0.0 ? sum / total : 0.0;
}

void main()
{
    float radius = length(vDisc) * uRingParams.y;

    if (radius < uRingParams.x || radius > uRingParams.y)
    {
        discard;
    }

    float footprint = length(fwidth(vWorldPosition)) * 0.001;

    float tau = ringOpticalDepth(radius) * (1.0 + 1.2 * ringlets(radius, footprint));

    if (tau < 1e-5)
    {
        discard;
    }

    vec3 n = normalize(vNormal);
    vec3 v = normalize(-vWorldPosition);
    vec3 l = normalize(-uDirectionalDirection.xyz);

    float muV = max(abs(dot(v, n)), 0.02);
    float muS = max(abs(dot(l, n)), 0.02);

    bool litFace = dot(v, n) * dot(l, n) > 0.0;

    // Phase: big icy particles scatter mostly back toward
    // the sun (bright at low phase angles, with a sharp
    // opposition surge); small ones forward (sparse rings
    // glow seen toward the sun).
    float back = dot(v, l);

    float phase =
        0.6
        + 3.4 * pow(0.5 + 0.5 * back, 3.0)
        + 1.5 * pow(max(back, 0.0), 60.0)
        + 2.5 * pow(max(-back, 0.0), 8.0);

    float reflectance;

    if (litFace)
    {
        reflectance = 0.25 * muS / (muS + muV) * (1.0 - exp(-tau * (1.0 / muS + 1.0 / muV)));

        // Light bouncing between particles in dense rings.
        reflectance *= 1.0 + 0.6 * (1.0 - exp(-tau));
    }
    else if (abs(muS - muV) < 1e-3)
    {
        reflectance = 0.25 * tau / muV * exp(-tau / muV);
    }
    else
    {
        reflectance = 0.25 * muS / (muS - muV) * (exp(-tau / muS) - exp(-tau / muV));
    }

    vec3 sun = uLightCounts.z != 0 ? uDirectionalColor.rgb * uDirectionalColor.a : vec3(0.0);

    // The planet's shadow, with a soft penumbra (in its
    // frame with y stretched by 1 / (1 - flattening) it is
    // a sphere of the equatorial radius). Other bodies'
    // shadows: eclipses.
    float squash = 1.0 - uRingShape.x;

    vec3 position = rotateByQuaternion(uRingFrame, vWorldPosition * 0.001 - uRingCenter.xyz);

    position.y /= squash;

    vec3 towardSun = rotateByQuaternion(uRingFrame, l);

    towardSun.y /= squash;
    towardSun = normalize(towardSun);

    float along = dot(position, towardSun);

    float shadow = 1.0;

    if (along < 0.0)
    {
        float miss = length(position - along * towardSun);

        shadow = smoothstep(uRingCenter.w * 0.99, uRingCenter.w * 1.01, miss);
    }

    vec3 others = eclipse(vWorldPosition * 0.001, l, uRingCenter.xyz);

    vec3 radiance = uRingColor * reflectance * phase * sun * shadow * others / BODIES_PI;

    float alpha = 1.0 - exp(-tau / muV);

    FragColor = vec4(radiance, alpha);
}
