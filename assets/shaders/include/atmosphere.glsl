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
                                    // (giants), z modeled depth (km), w depth
                                    // where the deep air begins (km)
    vec4 uThermal;                  // x temperature at altitude 0 (K), y lapse
                                    // (K/km), z density exponent (cp/R - 1),
                                    // w deep absorption (1/km at the tops)
    vec4 uDeckInfo;                 // x decks, y lightning clock (s), z glow
                                    // scale
    vec4 uDecks[12];                // per deck: (base, top, edge km, texture),
                                    // (scattering rgb 1/km, texture size km),
                                    // (absorption rgb 1/km, lightning rate)
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

// ---------------------------------------------------------
// Heat and Cloud Decks
// ---------------------------------------------------------

int deckCount()
{
    return int(uDeckInfo.x + 0.5);
}

// 0..1 inside deck i (soft edges).
float deckDensity(
    int i,
    float altitude
)
{
    vec4 shape = uDecks[i * 3];

    float edge = max(shape.z, 1e-3);

    return smoothstep(shape.x - edge, shape.x + edge, altitude)
        * (1.0 - smoothstep(shape.y - edge, shape.y + edge, altitude));
}

// Temperature (K) at an altitude: falling by the lapse
// rate above, rising along the adiabat below a giant's tops.
float airTemperature(
    float altitude
)
{
    if (noSolidSurface())
    {
        return uThermal.x + uThermal.y * max(-altitude, 0.0);
    }

    return max(uThermal.x - uThermal.y * altitude, 0.6 * uThermal.x);
}

// A giant's deep air: density relative to the tops at a
// depth (km) down the adiabat, (T / T0)^(cp/R - 1).
float compressionAt(
    float depth
)
{
    return pow(1.0 + uThermal.y * depth / max(uThermal.x, 1.0), uThermal.z);
}

// Thermal glow of air at a temperature: blackbody radiance at
// red, green and blue wavelengths relative to green at 2000 K
// (scaled for eyes adapted to the dark, graphics/atmosphere.py
// GLOW_SCALE). Dull red near 800 K, orange-white past 2000 K.
vec3 thermalGlow(
    float temperature
)
{
    if (temperature < 500.0)
    {
        return vec3(0.0);
    }

    // Planck in micrometers: lambda^-5 / (exp(c2 / lambda T) - 1).
    const vec3 wavelength = vec3(0.61, 0.55, 0.465);
    const float c2 = 14388.0;

    vec3 radiance = pow(wavelength, vec3(-5.0)) / (exp(c2 / (wavelength * temperature)) - 1.0);

    float reference = pow(0.55, -5.0) / (exp(c2 / (0.55 * 2000.0)) - 1.0);

    return uDeckInfo.z * radiance / reference;
}

// The medium at a point: air, haze, ozone, the decks (their
// density times `texture`, 1 = smooth), and the hot deep
// air's absorption.
struct Medium
{
    vec3 rayleigh;          // scattering
    vec3 mie;               // haze and cloud scattering (Mie phase)
    vec3 extinction;
    vec3 absorption;        // the part that also emits (heat)
    float temperature;
};

Medium mediumWithDecks(
    float altitude,
    float textures[4]
)
{
    Medium m;

    // Below the "ground": solid worlds stop there; on giants
    // the air keeps thickening down the adiabat (to the
    // modeled depth).
    //
    // (The deep air starts uShape.w below the tops: the
    // cloud-top mesh's flat triangles sag below the true
    // surface between vertices, and must not pick it up.)
    float air = noSolidSurface()
        ? max(min(altitude + uShape.w, 0.0) + max(altitude, 0.0), -uShape.z)
        : max(altitude, 0.0);

    float depth = max(-air, 0.0);

    float compression = depth > 0.0 ? compressionAt(depth) : 1.0;

    float rayleighDensity = exp(-max(air, 0.0) / uRayleighScattering.w) * compression;

    // Aerosols: thickest at the ground (dust, haze), or in a
    // layer around a given height.
    float layer = uSunIlluminance.w;

    float mieDensity = (
        layer > 0.0
            ? exp(-abs(max(air, 0.0) - layer) / uMieScattering.w)
            : exp(-max(air, 0.0) / uMieScattering.w)
    ) * compression;

    // Ozone: a tent around its center altitude.
    float ozoneDensity = max(
        0.0,
        1.0 - abs(air - uOzoneAbsorption.w) / uOzoneParams.x
    );

    m.rayleigh = uRayleighScattering.rgb * rayleighDensity;
    m.mie = uMieScattering.rgb * mieDensity;

    m.absorption = uMieAbsorption.rgb * mieDensity + uOzoneAbsorption.rgb * ozoneDensity;

    // Collisions of H2 molecules absorb ever more as the
    // deep air compresses (density squared).
    m.absorption += vec3(uThermal.w * compression * compression);

    for (int i = 0; i < 4; ++i)
    {
        if (i >= deckCount())
        {
            break;
        }

        float density = deckDensity(i, altitude) * textures[i];

        m.mie += uDecks[i * 3 + 1].rgb * density;
        m.absorption += uDecks[i * 3 + 2].rgb * density;
    }

    m.extinction = m.rayleigh + m.mie + m.absorption;

    m.temperature = airTemperature(altitude);

    return m;
}

Medium mediumAtAltitude(
    float altitude
)
{
    float smooth4[4] = float[4](1.0, 1.0, 1.0, 1.0);

    return mediumWithDecks(altitude, smooth4);
}

void mediumAt(
    float altitude,
    out vec3 rayleighScattering,
    out vec3 mieScattering,
    out vec3 extinction
)
{
    Medium m = mediumAtAltitude(altitude);

    rayleighScattering = m.rayleigh;
    mieScattering = m.mie;
    extinction = m.extinction;
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

// Giants below their cloud tops: the column between here and
// the tops, straight up. The gas thickens down the adiabat
// (its column is the integral of (1 + a z)^n), the decks add
// their part; `scattering` is what of it scatters.
void columnBelowTops(
    float radius,
    out vec3 extinction,
    out vec3 scattering
)
{
    extinction = vec3(0.0);
    scattering = vec3(0.0);

    if (!noSolidSurface() || radius >= groundRadius())
    {
        return;
    }

    float altitude = radius - groundRadius();

    float depth = clamp(-altitude - uShape.w, 0.0, uShape.z);

    if (depth > 0.0)
    {
        float a = uThermal.y / max(uThermal.x, 1.0);
        float n = uThermal.z;

        float gas = (pow(1.0 + a * depth, n + 1.0) - 1.0) / (a * (n + 1.0));
        float squared = (pow(1.0 + a * depth, 2.0 * n + 1.0) - 1.0) / (a * (2.0 * n + 1.0));

        vec3 gasScattering = uRayleighScattering.rgb + uMieScattering.rgb;

        scattering += gasScattering * gas;
        extinction += (gasScattering + uMieAbsorption.rgb) * gas + vec3(uThermal.w * squared);
    }

    for (int i = 0; i < 4; ++i)
    {
        if (i >= deckCount())
        {
            break;
        }

        vec4 shape = uDecks[i * 3];

        float inside = max(min(shape.y, 0.0) - max(shape.x, altitude), 0.0);

        scattering += uDecks[i * 3 + 1].rgb * inside;
        extinction += (uDecks[i * 3 + 1].rgb + uDecks[i * 3 + 2].rgb) * inside;
    }
}

// Direct sunlight that reaches below the tops.
vec3 belowGroundTransmittance(
    float radius,
    float sunCosZenith
)
{
    vec3 extinction;
    vec3 scattering;

    columnBelowTops(radius, extinction, scattering);

    return exp(-extinction / max(sunCosZenith, 0.05));
}

// Diffuse daylight that filters down below the tops: thick
// cloud scatters most light onward rather than stopping it
// (two-stream: T = exp(-k tau) / (1 + 3/4 (1 - g) tau), with
// k from the absorbed share), so under an overcast deck it
// is dim but not dark.
vec3 belowGroundDiffuse(
    float radius
)
{
    vec3 extinction;
    vec3 scattering;

    columnBelowTops(radius, extinction, scattering);

    vec3 albedo = scattering / max(extinction, vec3(1e-9));

    float g = uAtmosphereRadii.z;

    vec3 k = sqrt(max(3.0 * (1.0 - albedo) * (1.0 - albedo * g), vec3(0.0)));

    return exp(-k * extinction) / (1.0 + 0.75 * (1.0 - g) * scattering);
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
        * belowGroundDiffuse(radius);
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
        * belowGroundDiffuse(radius);

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
// Deck Detail and Lightning
// ---------------------------------------------------------
//
// Seen from inside, a cloud deck is not a smooth fog: thick
// and thin patches drift past (3D value noise at the deck's
// texture size, averaging to 1 so the lookup tables, which
// see the smooth deck, stay right). Lightning flashes in the
// decks that have it: the flash lights the cloud around it,
// its light diffusing tens of km through the droplets.

float atmosphereHash(
    vec3 p
)
{
    p = fract(p * vec3(0.1031, 0.1030, 0.0973));
    p += dot(p, p.yxz + 33.33);

    return fract((p.x + p.y) * p.z);
}

float atmosphereValueNoise(
    vec3 p
)
{
    vec3 i = floor(p);
    vec3 f = fract(p);

    f = f * f * (3.0 - 2.0 * f);

    return mix(
        mix(
            mix(atmosphereHash(i), atmosphereHash(i + vec3(1, 0, 0)), f.x),
            mix(atmosphereHash(i + vec3(0, 1, 0)), atmosphereHash(i + vec3(1, 1, 0)), f.x),
            f.y
        ),
        mix(
            mix(atmosphereHash(i + vec3(0, 0, 1)), atmosphereHash(i + vec3(1, 0, 1)), f.x),
            mix(atmosphereHash(i + vec3(0, 1, 1)), atmosphereHash(i + vec3(1, 1, 1)), f.x),
            f.y
        ),
        f.z
    );
}

// Density multiplier of deck i at p (mean ~1); `footprint`
// (km) is the step length: detail finer than it averages out.
float deckTexture(
    int i,
    vec3 p,
    float footprint
)
{
    float amount = uDecks[i * 3].w;
    float size = max(uDecks[i * 3 + 1].w, 0.1);

    float fade = 1.0 - smoothstep(0.5, 2.0, footprint / size);

    if (amount <= 0.0 || fade <= 0.0)
    {
        return 1.0;
    }

    // Each deck its own pattern.
    vec3 q = p / size + float(i) * 17.31;

    float n =
        0.55 * atmosphereValueNoise(q)
        + 0.30 * atmosphereValueNoise(q * 2.13)
        + 0.15 * atmosphereValueNoise(q * 4.37);

    float patchy = 2.0 * smoothstep(0.2, 0.8, n);

    return mix(1.0, patchy, amount * fade);
}

// Lightning: storm cells LIGHTNING_CELL km across on a grid
// over the deck (on the faces of a cube around the planet,
// projected onto it), each with its own clock; a flash lasts
// ~0.3 s (a few return strokes) and lights the cloud within
// ~LIGHTNING_REACH km. The light is cut off before the cell's
// edge, so each cell only needs its own flash.
const float LIGHTNING_CELL = 200.0;
const float LIGHTNING_REACH = 40.0;
const float LIGHTNING_BRIGHTNESS = 400.0;

// Light per unit of scattering at p from flashes in deck i.
vec3 lightningAt(
    int i,
    vec3 p
)
{
    float rate = uDecks[i * 3 + 2].w;

    if (rate <= 0.0)
    {
        return vec3(0.0);
    }

    // The cube face p is over, and where on it (km on a plane
    // at the deck's radius).
    vec4 shape = uDecks[i * 3];

    float radius = groundRadius() + mix(shape.x, shape.y, 0.3);

    // (Swizzles, not p[axis]: dynamic vector indexing hangs
    // some drivers.)
    vec3 a = abs(p);

    int axis = a.x > a.y ? (a.x > a.z ? 0 : 2) : (a.y > a.z ? 1 : 2);

    vec3 q = axis == 0 ? p : (axis == 1 ? p.yzx : p.zxy);

    vec2 plane = q.yz / abs(q.x) * radius;

    vec2 grid = floor(plane / LIGHTNING_CELL);

    vec3 cell = vec3(grid, float(axis) + (q.x > 0.0 ? 0.0 : 3.0));

    // One-second slots; a flash in a slot with probability
    // `rate`, at a random moment within it.
    float time = uDeckInfo.y + 7.0 * atmosphereHash(cell + 0.5);

    float slot = floor(time);

    if (atmosphereHash(cell + vec3(slot, 3.1, 7.7)) > rate * 0.5)
    {
        return vec3(0.0);
    }

    float age = fract(time) - 0.6 * atmosphereHash(cell + vec3(slot, 9.2, 1.3));

    if (age < 0.0 || age > 0.35)
    {
        return vec3(0.0);
    }

    // Strokes: a bright first, flickering after.
    float strokes = exp(-age / 0.06) + 0.5 * step(0.5, fract(age * 23.0)) * exp(-age / 0.15);

    // Where: near the cell's middle, in the deck's lower part
    // (water clouds flash low, where the updrafts are).
    vec2 offset = vec2(atmosphereHash(cell + 1.7), atmosphereHash(cell + 4.1)) - 0.5;

    vec2 flat_ = (grid + 0.5 + 0.3 * offset) * LIGHTNING_CELL;

    // (None in cells that cross a face's edge.)
    if (max(abs(flat_.x), abs(flat_.y)) > radius - 0.5 * LIGHTNING_CELL)
    {
        return vec3(0.0);
    }

    vec3 local = vec3(q.x > 0.0 ? radius : -radius, flat_);

    vec3 direction = axis == 0 ? local : (axis == 1 ? local.zxy : local.yzx);

    vec3 center = normalize(direction) * radius;

    float d = length(p - center);

    // Diffusion through the cloud: ~exp(-d / L) / d, cut
    // before the cell's edge (on the grid's plane).
    float edge = length(plane - flat_) / LIGHTNING_CELL;

    float light = exp(-d / LIGHTNING_REACH) / (d + 3.0) * (1.0 - smoothstep(0.24, 0.34, edge));

    return vec3(0.85, 0.9, 1.0) * LIGHTNING_BRIGHTNESS * strokes * light;
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
// from it. Where the ray crosses cloud decks it is split at
// their boundaries too, and each piece gets samples by its
// optical depth (a deck a few km thick must not fall between
// two samples).
//
// Returns radiance per unit sun illuminance; also the
// transmittance along the whole segment. Light the air gives
// off itself (heat, lightning) does not scale with the sun:
// it goes to gAtmosphereGlow (absolute).

vec3 gAtmosphereGlow = vec3(0.0);

struct ScatteringState
{
    vec3 luminance;
    vec3 transmittance;
    vec3 glow;
};

const int SPACING_UNIFORM = 0;
const int SPACING_DENSE_START = 1;
const int SPACING_DENSE_END = 2;

// March [from, to] (from < to, moving away from the
// camera), with steps even or smallest at the start or the
// end. jitter in [0, 1) places each sample within its step.
void marchSegment(
    vec3 origin,
    vec3 direction,
    float from,
    float to,
    int spacing,
    int steps,
    float jitter,
    float phaseR,
    float phaseM,
    inout ScatteringState state
)
{
    vec3 sunDirection = uAtmosphereSunDirection.xyz;

    float span = to - from;

    int decks = deckCount();

    for (int i = 0; i < steps; ++i)
    {
        float u0 = float(i) / float(steps);
        float u1 = float(i + 1) / float(steps);
        float us = (float(i) + jitter) / float(steps);

        float t0 = u0;
        float t1 = u1;
        float ts = us;

        // Quadratic spacing; mirrored when dense at the end.
        if (spacing == SPACING_DENSE_START)
        {
            t0 = u0 * u0;
            t1 = u1 * u1;
            ts = us * us;
        }
        else if (spacing == SPACING_DENSE_END)
        {
            t0 = 1.0 - (1.0 - u0) * (1.0 - u0);
            t1 = 1.0 - (1.0 - u1) * (1.0 - u1);
            ts = 1.0 - (1.0 - us) * (1.0 - us);
        }

        float dt = (t1 - t0) * span;

        vec3 p = origin + (from + ts * span) * direction;

        float r = length(p);

        float altitude = r - groundRadius();

        float textures[4] = float[4](1.0, 1.0, 1.0, 1.0);

        vec3 flash = vec3(0.0);

        for (int k = 0; k < 4; ++k)
        {
            if (k >= decks)
            {
                break;
            }

            if (deckDensity(k, altitude) > 0.0)
            {
                textures[k] = deckTexture(k, p, dt);
                flash += lightningAt(k, p);
            }
        }

        Medium m = mediumWithDecks(altitude, textures);

        vec3 sampleTransmittance = exp(-dt * m.extinction);

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
            m.rayleigh * (phaseR * sunLight + multiple)
            + m.mie * (phaseM * sunLight + multiple);

        vec3 emission =
            m.absorption * thermalGlow(m.temperature)
            + m.mie * flash;

        vec3 share = (1.0 - sampleTransmittance) / max(m.extinction, vec3(1e-7));

        state.luminance += inScattering * share * state.transmittance;
        state.glow += emission * share * state.transmittance;

        state.transmittance *= sampleTransmittance;
    }
}

// Both distances along the ray to a sphere (-1 = none).
vec2 raySphereBoth(
    vec3 origin,
    vec3 direction,
    float radius
)
{
    float b = dot(origin, direction);
    float c = dot(origin, origin) - radius * radius;

    float discriminant = b * b - c;

    if (discriminant < 0.0)
    {
        return vec2(-1.0);
    }

    float s = sqrt(discriminant);

    return vec2(-b - s, -b + s);
}

const int MAX_BREAKS = 20;

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
    state.glow = vec3(0.0);

    // Lowest point of the segment (closest to the center).
    float lowest = clamp(-dot(origin, direction), start, end);

    // Inside a giant's deep air: it is densest around the
    // camera, so pack the samples there.
    if (noSolidSurface() && length(origin) < groundRadius())
    {
        lowest = start;
    }

    // Break points: start, end, the lowest point, and where
    // the ray crosses the decks' boundaries.
    float breaks[MAX_BREAKS];
    int count = 0;

    breaks[count++] = start;
    breaks[count++] = end;

    if (lowest > start && lowest < end)
    {
        breaks[count++] = lowest;
    }

    int decks = deckCount();

    for (int k = 0; k < 4; ++k)
    {
        if (k >= decks)
        {
            break;
        }

        vec4 shape = uDecks[k * 3];

        for (int side = 0; side < 2; ++side)
        {
            float height = side == 0 ? shape.x - shape.z : shape.y + shape.z;

            vec2 hits = raySphereBoth(origin, direction, groundRadius() + height);

            for (int h = 0; h < 2; ++h)
            {
                float t = hits[h];

                if (t > start && t < end && count < MAX_BREAKS)
                {
                    breaks[count++] = t;
                }
            }
        }
    }

    // Sort (few entries).
    for (int i = 1; i < MAX_BREAKS; ++i)
    {
        if (i >= count)
        {
            break;
        }

        float value = breaks[i];

        int j = i - 1;

        while (j >= 0 && breaks[j] > value)
        {
            breaks[j + 1] = breaks[j];
            --j;
        }

        breaks[j + 1] = value;
    }

    if (count == 2)
    {
        // No decks crossed: the plain split at the lowest
        // point.
        float length_ = max(end - start, 1e-6);

        int before = clamp(int(round(float(steps) * (lowest - start) / length_)), 0, steps);

        if (before > 0)
        {
            marchSegment(origin, direction, start, lowest, SPACING_DENSE_END, before, jitter, phaseR, phaseM, state);
        }

        if (steps - before > 0)
        {
            marchSegment(origin, direction, lowest, end, SPACING_DENSE_START, steps - before, jitter, phaseR, phaseM, state);
        }
    }
    else
    {
        // Samples by estimated optical depth (at each piece's
        // middle), with a share by length.
        float depths[MAX_BREAKS];
        float reach[MAX_BREAKS];
        float totalDepth = 0.0;

        for (int i = 0; i < MAX_BREAKS - 1; ++i)
        {
            if (i >= count - 1)
            {
                break;
            }

            float middle = 0.5 * (breaks[i] + breaks[i + 1]);

            Medium m = mediumAtAltitude(length(origin + middle * direction) - groundRadius());

            float depth = (breaks[i + 1] - breaks[i]) * min(m.extinction.r, min(m.extinction.g, m.extinction.b));

            depth = isnan(depth) ? 0.0 : max(depth, 0.0);

            // Light from beyond optical depth ~25 never
            // arrives: march only to there (a long ray along a
            // deck otherwise gets steps tens of km long).
            reach[i] = depth > 25.0 ? 25.0 / depth : 1.0;

            depths[i] = min(depth, 25.0);

            totalDepth += depths[i];
        }

        // Rays through decks get more samples than the plain
        // sky (and enough that the count changing from one
        // pixel to the next does not show as bands).
        int budget = 3 * steps;

        float length_ = max(end - start, 1e-6);

        for (int i = 0; i < MAX_BREAKS - 1; ++i)
        {
            if (i >= count - 1)
            {
                break;
            }

            if (max(state.transmittance.r, max(state.transmittance.g, state.transmittance.b)) < 1e-5)
            {
                break;
            }

            float from = breaks[i];
            float to = breaks[i + 1];

            if (to - from <= 1e-6)
            {
                continue;
            }

            float weight =
                0.35 * (to - from) / length_
                + 0.65 * depths[i] / max(totalDepth, 1e-6);

            int n = clamp(int(round(weight * float(budget))), 1, budget);

            // Outside the decks packed toward the lowest point
            // (or the camera, in a giant's depths).
            float middle = 0.5 * (from + to);

            float altitude = length(origin + middle * direction) - groundRadius();

            bool inDeck = false;

            for (int k = 0; k < 4; ++k)
            {
                if (k >= decks)
                {
                    break;
                }

                inDeck = inDeck || deckDensity(k, altitude) > 0.0;
            }

            // (Inside a deck evenly, unless it is cut short:
            // then densest nearest the camera.)
            int spacing = inDeck && reach[i] >= 1.0
                ? SPACING_UNIFORM
                : (inDeck || from >= lowest ? SPACING_DENSE_START : SPACING_DENSE_END);

            float stop = from + (to - from) * reach[i];

            marchSegment(origin, direction, from, stop, spacing, n, jitter, phaseR, phaseM, state);

            if (reach[i] < 1.0)
            {
                // The rest: opaque.
                state.transmittance = vec3(0.0);
            }
        }
    }

    transmittance = state.transmittance;

    gAtmosphereGlow = state.glow;

    return state.luminance;
}
