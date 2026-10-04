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
    vec4 uMieScattering;            // rgb 1/km, w scale height (km)
    vec4 uMieAbsorption;            // rgb 1/km, w sun angular radius (rad)
    vec4 uOzoneAbsorption;          // rgb 1/km, w center altitude (km)
    vec4 uOzoneParams;              // x half width (km), y raymarch steps,
                                    // z 1 = haze over geometry,
                                    // w thick-atmosphere weight
    vec4 uAtmosphereSunDirection;   // xyz toward the sun, w 1 = present
    vec4 uSunIlluminance;           // rgb sun color * intensity, w aerosol
                                    // layer altitude (km; 0 = at the ground)
    vec4 uCloudParams;              // x cloud layer altitude (km), y optical
                                    // depth, z feature size (km), w 1 = clouds
    vec4 uCloudColor;               // rgb cloud color, w drift angle (rad)
    vec4 uPlanetFrame;              // quaternion (xyzw): world -> the
                                    // planet's own frame (its climate,
                                    // clouds)
    vec4 uShape;                    // x flattening, y 1 = no solid surface
                                    // (giants), z opaque depth (km), w depth
                                    // where the deep air begins (km)
};

// ---------------------------------------------------------
// Atmosphere Space
// ---------------------------------------------------------
//
// Everything below works in the planet's own frame with y
// stretched by 1 / (1 - flattening): a flattened giant's
// cloud tops, its air and its cloud shell all become
// spheres. World positions (relative to the planet center)
// and directions are converted on entry; uAtmosphereSun-
// Direction is already in this space.

vec3 toPlanetFrame(
    vec3 v
)
{
    vec3 q = uPlanetFrame.xyz;

    return v + 2.0 * cross(q, cross(q, v) + uPlanetFrame.w * v);
}

vec3 toAtmosphereSpace(
    vec3 v
)
{
    v = toPlanetFrame(v);

    v.y /= 1.0 - uShape.x;

    return v;
}

// Back to the planet's frame (unstretched).
vec3 atmosphereToPlanetFrame(
    vec3 a
)
{
    return vec3(a.x, a.y * (1.0 - uShape.x), a.z);
}

// Giants: air continues below the cloud tops.
bool noSolidSurface()
{
    return uShape.y > 0.5;
}

uniform sampler2D uTransmittanceLut;
uniform sampler2D uMultiScatteringLut;
uniform sampler2D uDiffuseLut;

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
    out vec3 mieScattering,
    out vec3 extinction
)
{
    // Below the "ground": solid worlds stop there; on giants
    // the air keeps thickening (to the opaque depth).
    //
    // (The deep air starts uShape.w below the tops: the
    // cloud-top mesh's flat triangles sag below the true
    // surface between vertices, and must not pick it up.)
    altitude = noSolidSurface()
        ? max(min(altitude + uShape.w, 0.0) + max(altitude, 0.0), -uShape.z)
        : max(altitude, 0.0);

    float rayleighDensity = exp(-altitude / uRayleighScattering.w);

    // Aerosols: thickest at the ground (dust, haze), or in a
    // deck around a given height (Venus's clouds).
    float layer = uSunIlluminance.w;

    float mieDensity = layer > 0.0
        ? exp(-abs(altitude - layer) / uMieScattering.w)
        : exp(-altitude / uMieScattering.w);

    // Ozone: a tent around its center altitude.
    float ozoneDensity = max(
        0.0,
        1.0 - abs(altitude - uOzoneAbsorption.w) / uOzoneParams.x
    );

    rayleighScattering = uRayleighScattering.rgb * rayleighDensity;
    mieScattering = uMieScattering.rgb * mieDensity;

    extinction =
        rayleighScattering
        + (uMieScattering.rgb + uMieAbsorption.rgb) * mieDensity
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

// Giants below their cloud tops: the extra air between
// here and the "ground" level dims the light (vertical
// optical depth of the exponential layers, along a slant).
vec3 belowGroundTransmittance(
    float radius,
    float sunCosZenith
)
{
    if (!noSolidSurface() || radius >= groundRadius())
    {
        return vec3(1.0);
    }

    float depth = min(groundRadius() - radius - uShape.w, uShape.z);

    if (depth <= 0.0)
    {
        return vec3(1.0);
    }

    float hr = uRayleighScattering.w;
    float hm = uMieScattering.w;

    vec3 tau =
        uRayleighScattering.rgb * hr * (exp(depth / hr) - 1.0)
        + (uMieScattering.rgb + uMieAbsorption.rgb) * hm * (exp(min(depth / hm, 30.0)) - 1.0);

    return exp(-tau / max(sunCosZenith, 0.05));
}

vec3 sunTransmittance(
    float radius,
    float sunCosZenith
)
{
    return texture(uTransmittanceLut, lutCoordinates(radius, sunCosZenith)).rgb
        * belowGroundTransmittance(radius, sunCosZenith);
}

// 0..1: how optically thick the atmosphere is (see
// AtmosphereParameters.thick_weight).
float thickAtmosphereWeight()
{
    return uOzoneParams.w;
}

// Diffuse daylight at this altitude under a thick
// atmosphere: irradiance as a fraction of the sunlight at
// the top (atmosphere_diffuse.frag.glsl).
vec3 diffuseDaylight(
    float radius,
    float sunCosZenith
)
{
    return texture(uDiffuseLut, lutCoordinates(radius, sunCosZenith)).rgb
        * belowGroundTransmittance(radius, max(sunCosZenith, 0.3));
}

// Light scattered more than once, as radiance per unit sun
// illuminance (multiplied by the scattering coefficient).
// Thin air: Hillaire's LUT. Thick air: the isotropic
// diffuse daylight field (radiance = irradiance / pi).
vec3 multipleScattering(
    float radius,
    float sunCosZenith
)
{
    vec3 hillaire = texture(uMultiScatteringLut, lutCoordinates(radius, sunCosZenith)).rgb
        * belowGroundTransmittance(radius, max(sunCosZenith, 0.3));

    float weight = thickAtmosphereWeight();

    if (weight <= 0.0)
    {
        return hillaire;
    }

    return mix(
        hillaire,
        diffuseDaylight(radius, sunCosZenith) / ATMOSPHERE_PI,
        weight
    );
}

// Sunlight reaching a camera-relative world position (in
// meters), as a fraction: reddened at sunset, 0 where the
// planet blocks the sun.
vec3 sunTransmittanceAtWorld(
    vec3 worldPosition
)
{
    vec3 p = toAtmosphereSpace(worldPosition * 0.001 - uPlanetCenter.xyz);

    float r = length(p);

    // Lit geometry on a giant is its cloud tops (the mesh
    // only sags below them between vertices).
    if (noSolidSurface())
    {
        r = max(r, groundRadius());
    }

    return sunTransmittance(r, dot(p, uAtmosphereSunDirection.xyz) / r);
}


// ---------------------------------------------------------
// Eclipses
// ---------------------------------------------------------
//
// Shaders that define ATMOSPHERE_ECLIPSES (and include
// include/bodies.glsl and include/rings.glsl first) dim
// the sunlight in the air under another body's shadow and
// the rings': the sky darkens in a total solar eclipse,
// with sunset colors all around the horizon where the air
// beyond the umbra is still lit; Saturn's haze is dark in
// its rings' shadow.

#ifdef ATMOSPHERE_ECLIPSES

vec3 eclipseInAtmosphere(
    vec3 p
)
{
    int count = int(uBodyParams.z + 0.5);

    if (count == 0 && !ringsPresent())
    {
        return vec3(1.0);
    }

    // Atmosphere space -> camera-relative world (km).
    vec4 toWorld = vec4(-uPlanetFrame.xyz, uPlanetFrame.w);

    vec3 world = uPlanetCenter.xyz + rotateByQuaternion(toWorld, atmosphereToPlanetFrame(p));

    vec3 sun = normalize(rotateByQuaternion(toWorld, atmosphereToPlanetFrame(uAtmosphereSunDirection.xyz)));

    return eclipseFrom(world, sun, uPlanetCenter.xyz, count) * ringShadow(world - uRingCenter.xyz, sun);
}

#endif


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
        vec3 mieScattering;
        vec3 extinction;

        mediumAt(r - groundRadius(), rayleighScattering, mieScattering, extinction);

        vec3 sampleTransmittance = exp(-dt * extinction);

        float sunCosZenith = dot(p, sunDirection) / r;

        vec3 sunLight = sunTransmittance(r, sunCosZenith);
        vec3 multiple = multipleScattering(r, sunCosZenith);

#ifdef ATMOSPHERE_ECLIPSES
        vec3 shade = eclipseInAtmosphere(p);

        sunLight *= shade;

        // Light scattered in from the sunlit air around the
        // umbra keeps a total eclipse's sky deep blue.
        multiple *= mix(shade, vec3(1.0), 0.03);
#endif

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

    // Inside a giant's deep air: it is densest around the
    // camera, so pack the samples there.
    if (noSolidSurface() && length(origin) < groundRadius())
    {
        lowest = start;
    }

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
