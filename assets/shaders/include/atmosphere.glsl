// =========================================================
// Atmospheric Scattering
// =========================================================
//
// Shared by the LUT builders, the sky / aerial perspective
// pass, the environment bake and lit surfaces (sunlight
// transmittance). See graphics/atmosphere.py.
//
// Units: kilometers. Positions are relative to the planet
// center unless noted.


layout(std140) uniform AtmosphereBlock
{
    vec4 uPlanetCenter;             // xyz relative to the camera (km), w 1 = present
    vec4 uAtmosphereRadii;          // x ground, y top radius (km), z Mie g, w ground albedo
    vec4 uRayleighScattering;       // rgb 1/km, w scale height (km)
    vec4 uMieParams;                // x scattering, y absorption (1/km),
                                    // z scale height (km), w sun angular radius
    vec4 uOzoneAbsorption;          // rgb 1/km, w center altitude (km)
    vec4 uOzoneParams;              // x half width (km), y raymarch steps
    vec4 uAtmosphereSunDirection;   // xyz toward the sun, w 1 = present
    vec4 uSunIlluminance;           // rgb sun color * intensity, w disc brightness
};

uniform sampler2D uTransmittanceLut;
uniform sampler2D uMultiScatteringLut;

const float ATMOSPHERE_PI = 3.14159265358979;


bool atmospherePresent()
{
    return uPlanetCenter.w > 0.5;
}

float groundRadius()
{
    return uAtmosphereRadii.x;
}

float topRadius()
{
    return uAtmosphereRadii.y;
}


// ---------------------------------------------------------
// Geometry
// ---------------------------------------------------------

// Distance along the ray to the sphere's first hit ahead
// (the exit when starting inside), or -1.
float raySphere(
    vec3 origin,
    vec3 direction,
    float radius
)
{
    float b = dot(origin, direction);
    float c = dot(origin, origin) - radius * radius;

    if (c > 0.0 && b > 0.0)
    {
        return -1.0;
    }

    float d = b * b - c;

    if (d < 0.0)
    {
        return -1.0;
    }

    float s = sqrt(d);

    return c > 0.0 ? -b - s : -b + s;
}

// The part of the ray inside the atmosphere: [start, end].
bool atmosphereSegment(
    vec3 origin,
    vec3 direction,
    out float start,
    out float end
)
{
    float top = topRadius();

    float b = dot(origin, direction);
    float c = dot(origin, origin) - top * top;
    float d = b * b - c;

    if (d < 0.0)
    {
        return false;
    }

    float s = sqrt(d);

    end = -b + s;

    if (end <= 0.0)
    {
        return false;
    }

    start = max(-b - s, 0.0);

    return true;
}


// ---------------------------------------------------------
// Medium
// ---------------------------------------------------------

void mediumAt(
    float altitude,
    out vec3 rayleighScattering,
    out float mieScattering,
    out vec3 extinction
)
{
    altitude = max(altitude, 0.0);

    float rayleighDensity = exp(-altitude / uRayleighScattering.w);
    float mieDensity = exp(-altitude / uMieParams.z);

    // Ozone: a tent around its center altitude.
    float ozoneDensity = max(
        0.0,
        1.0 - abs(altitude - uOzoneAbsorption.w) / uOzoneParams.x
    );

    rayleighScattering = uRayleighScattering.rgb * rayleighDensity;
    mieScattering = uMieParams.x * mieDensity;

    extinction =
        rayleighScattering
        + vec3((uMieParams.x + uMieParams.y) * mieDensity)
        + uOzoneAbsorption.rgb * ozoneDensity;
}


// ---------------------------------------------------------
// Phase Functions
// ---------------------------------------------------------

float rayleighPhase(
    float cosTheta
)
{
    return 3.0 / (16.0 * ATMOSPHERE_PI) * (1.0 + cosTheta * cosTheta);
}

// Cornette-Shanks.
float miePhase(
    float cosTheta
)
{
    float g = uAtmosphereRadii.z;
    float g2 = g * g;

    float k = 3.0 / (8.0 * ATMOSPHERE_PI) * (1.0 - g2) / (2.0 + g2);

    return k * (1.0 + cosTheta * cosTheta)
        / pow(max(1.0 + g2 - 2.0 * g * cosTheta, 1e-4), 1.5);
}


// ---------------------------------------------------------
// Lookup Tables
// ---------------------------------------------------------
//
// Both are indexed by (sun zenith cosine, altitude).

vec2 lutCoordinates(
    float radius,
    float sunCosZenith
)
{
    return vec2(
        0.5 + 0.5 * sunCosZenith,
        clamp((radius - groundRadius()) / (topRadius() - groundRadius()), 0.0, 1.0)
    );
}

// Inverse of lutCoordinates for the LUT builders.
void lutParameters(
    vec2 uv,
    out float radius,
    out float sunCosZenith
)
{
    sunCosZenith = uv.x * 2.0 - 1.0;

    // Just above the ground so ground rays are unambiguous.
    radius = mix(groundRadius() + 0.01, topRadius(), uv.y);
}

vec3 sunTransmittance(
    float radius,
    float sunCosZenith
)
{
    return texture(uTransmittanceLut, lutCoordinates(radius, sunCosZenith)).rgb;
}

vec3 multipleScattering(
    float radius,
    float sunCosZenith
)
{
    return texture(uMultiScatteringLut, lutCoordinates(radius, sunCosZenith)).rgb;
}

// Sunlight reaching a camera-relative world position (in
// meters), as a fraction: reddened at sunset, 0 where the
// planet blocks the sun.
vec3 sunTransmittanceAtWorld(
    vec3 worldPosition
)
{
    vec3 p = worldPosition * 0.001 - uPlanetCenter.xyz;

    float r = length(p);

    return sunTransmittance(r, dot(p, uAtmosphereSunDirection.xyz) / r);
}


// ---------------------------------------------------------
// Raymarch
// ---------------------------------------------------------
//
// Single scattering of sunlight along the ray, plus the
// multiple-scattering LUT term; integrated per step
// analytically (energy-conserving; Hillaire 2015).
//
// Air density falls off exponentially with altitude, so
// evenly spaced samples waste most of their budget on thin
// air (seen from space, a ray crosses ~100 km of
// atmosphere but nearly all of the scattering happens in
// its lowest few km). Samples are therefore packed toward
// the ray's lowest point: the ray is split there and each
// half is sampled with quadratically growing steps away
// from it.
//
// Returns radiance per unit sun illuminance; also the
// transmittance along the whole segment.

struct ScatteringState
{
    vec3 luminance;
    vec3 transmittance;
};

// March [from, to] (from < to, moving away from the
// camera), with steps smallest at the start or the end.
// jitter in [0, 1) places each sample within its step.
void marchSegment(
    vec3 origin,
    vec3 direction,
    float from,
    float to,
    bool denseAtStart,
    int steps,
    float jitter,
    float phaseR,
    float phaseM,
    inout ScatteringState state
)
{
    vec3 sunDirection = uAtmosphereSunDirection.xyz;

    float span = to - from;

    for (int i = 0; i < steps; ++i)
    {
        float u0 = float(i) / float(steps);
        float u1 = float(i + 1) / float(steps);
        float us = (float(i) + jitter) / float(steps);

        // Quadratic spacing; mirrored when dense at the end.
        float t0 = denseAtStart ? u0 * u0 : 1.0 - (1.0 - u0) * (1.0 - u0);
        float t1 = denseAtStart ? u1 * u1 : 1.0 - (1.0 - u1) * (1.0 - u1);
        float ts = denseAtStart ? us * us : 1.0 - (1.0 - us) * (1.0 - us);

        float dt = (t1 - t0) * span;

        vec3 p = origin + (from + ts * span) * direction;

        float r = length(p);

        vec3 rayleighScattering;
        float mieScattering;
        vec3 extinction;

        mediumAt(r - groundRadius(), rayleighScattering, mieScattering, extinction);

        vec3 sampleTransmittance = exp(-dt * extinction);

        float sunCosZenith = dot(p, sunDirection) / r;

        vec3 sunLight = sunTransmittance(r, sunCosZenith);
        vec3 multiple = multipleScattering(r, sunCosZenith);

        vec3 inScattering =
            rayleighScattering * (phaseR * sunLight + multiple)
            + mieScattering * (phaseM * sunLight + multiple);

        vec3 integral =
            (inScattering - inScattering * sampleTransmittance)
            / max(extinction, vec3(1e-7));

        state.luminance += integral * state.transmittance;

        state.transmittance *= sampleTransmittance;
    }
}

vec3 integrateScattering(
    vec3 origin,
    vec3 direction,
    float start,
    float end,
    int steps,
    float jitter,
    out vec3 transmittance
)
{
    float cosTheta = dot(direction, uAtmosphereSunDirection.xyz);

    float phaseR = rayleighPhase(cosTheta);
    float phaseM = miePhase(cosTheta);

    ScatteringState state;
    state.luminance = vec3(0.0);
    state.transmittance = vec3(1.0);

    // Lowest point of the segment (closest to the center).
    float lowest = clamp(-dot(origin, direction), start, end);

    float length_ = max(end - start, 1e-6);

    int before = int(round(float(steps) * (lowest - start) / length_));

    before = clamp(before, 0, steps);

    if (before > 0)
    {
        marchSegment(origin, direction, start, lowest, false, before, jitter, phaseR, phaseM, state);
    }

    if (steps - before > 0)
    {
        marchSegment(origin, direction, lowest, end, true, steps - before, jitter, phaseR, phaseM, state);
    }

    transmittance = state.transmittance;

    return state.luminance;
}
