// =========================================================
// Weather Clouds
// =========================================================
//
// A cloud layer: a shell at uCloudParams.x above the
// ground (graphics/clouds.py). Where it is cloudy comes
// from two scales:
//
//   cover map   equirectangular, from the climate (rainy
//               belts cloudy, subtropical deserts clear),
//               scaled to the body's mean coverage
//   noise       domain-warped fractal value noise per
//               pixel (swirls and streets of cloud down to
//               a few km), faded below the pixel footprint
//
// Shared by the sky / aerial perspective pass (clouds in
// front of and behind the air), the environment bake
// (overcast skies dim the ambient light) and lit surfaces
// (cloud shadows). Needs include/atmosphere.glsl.
//
// Units: km, positions relative to the planet center.

uniform sampler2D uCloudMap;

bool cloudsPresent()
{
    return uCloudParams.w > 0.5;
}

float cloudRadius()
{
    return groundRadius() + uCloudParams.x;
}

float cloudHash(
    ivec3 cell
)
{
    uvec3 v = uvec3(cell) * uvec3(1597334673u, 3812015801u, 2798796415u);

    uint n = (v.x ^ v.y ^ v.z) * 1597334673u;

    n ^= n >> 16u;
    n *= 2246822519u;
    n ^= n >> 13u;

    return float(n) * (1.0 / 4294967295.0);
}

// Value noise in [-1, 1].
float cloudNoise(
    vec3 p
)
{
    vec3 cellFloor = floor(p);
    ivec3 i = ivec3(cellFloor);
    vec3 f = p - cellFloor;

    f = f * f * (3.0 - 2.0 * f);

    float a = mix(cloudHash(i), cloudHash(i + ivec3(1, 0, 0)), f.x);
    float b = mix(cloudHash(i + ivec3(0, 1, 0)), cloudHash(i + ivec3(1, 1, 0)), f.x);
    float c = mix(cloudHash(i + ivec3(0, 0, 1)), cloudHash(i + ivec3(1, 0, 1)), f.x);
    float d = mix(cloudHash(i + ivec3(0, 1, 1)), cloudHash(i + ivec3(1, 1, 1)), f.x);

    return mix(mix(a, b, f.y), mix(c, d, f.y), f.z) * 2.0 - 1.0;
}

// The cover map's fraction (0..1) at a unit direction (in
// the planet's frame).
float cloudCover(
    vec3 direction
)
{
    vec2 uv = vec2(
        atan(direction.z, direction.x) * (0.5 / ATMOSPHERE_PI) + 0.5,
        asin(clamp(direction.y, -1.0, 1.0)) / ATMOSPHERE_PI + 0.5
    );

    // Explicit LOD: no derivatives across the longitude seam.
    return textureLod(uCloudMap, uv, 0.0).r;
}

// Optical depth (vertical) of the cloud layer at a unit
// direction. footprint: km covered by a pixel there.
float cloudDepth(
    vec3 direction,
    float footprint
)
{
    // From atmosphere space to the planet's frame (the
    // cover map's).
    direction = normalize(atmosphereToPlanetFrame(direction));

    float base = cloudCover(direction);

    if (base <= 0.002)
    {
        return 0.0;
    }

    // Slow drift with the winds.
    float angle = uCloudColor.w;

    vec3 drifted = vec3(
        cos(angle) * direction.x + sin(angle) * direction.z,
        direction.y,
        -sin(angle) * direction.x + cos(angle) * direction.z
    );

    float scale = uCloudParams.z;    // km: largest features

    vec3 p = drifted * (cloudRadius() / scale);

    // Domain warp: swirls rather than blobs.
    p += 0.9 * vec3(
        cloudNoise(p * 0.5 + 3.1),
        cloudNoise(p * 0.5 + 7.7),
        cloudNoise(p * 0.5 + 11.3)
    );

    float n = 0.0;
    float weight = 0.5;
    float total = 0.0;
    float frequency = 1.0;

    for (int octave = 0; octave < 13; ++octave)
    {
        // Fade octaves smaller than a few pixels.
        float fade = 1.0 - smoothstep(0.35, 0.8, footprint * frequency / scale);

        if (fade <= 0.0)
        {
            break;
        }

        n += weight * fade * cloudNoise(p * frequency + float(octave) * 13.7);

        total += weight;
        weight *= 0.55;
        frequency *= 2.15;
    }

    // The fractal sum is nearly Gaussian (standard deviation
    // ~0.21): map it through its distribution (a logistic
    // approximation) to ~uniform 0..1, so a cover fraction b
    // leaves about b of the sky cloudy.
    n = 0.5 + 0.5 * tanh(4.05 * n / max(total, 1e-3));

    // Cloudier where the climate is: a lower threshold.
    float threshold = 1.0 - base;

    float cover = smoothstep(threshold - 0.06, threshold + 0.06, n);

    // Thicker toward the middle of each cloud.
    float core = smoothstep(threshold, threshold + 0.25, n);

    return uCloudParams.y * cover * (0.05 + 0.95 * core * core);
}

// Where a ray first meets the cloud layer within
// [start, end], or -1.
float cloudHit(
    vec3 origin,
    vec3 direction,
    float start,
    float end
)
{
    float t = raySphere(origin, direction, cloudRadius());

    return (t >= start && t <= end) ? t : -1.0;
}

struct CloudSample
{
    vec3 radiance;          // per unit sun illuminance
    float depth;            // optical depth along the view
};

// The cloud layer seen at a point on it, from `direction`.
CloudSample shadeCloud(
    vec3 position,
    vec3 direction,
    float footprint
)
{
    CloudSample cloud;
    cloud.radiance = vec3(0.0);
    cloud.depth = 0.0;

    vec3 up = normalize(position);

    float tau = cloudDepth(up, footprint);

    if (tau <= 1e-3)
    {
        return cloud;
    }

    float muView = abs(dot(direction, up));

    // Slanted views cross more cloud (limited at grazing).
    cloud.depth = tau / max(muView, 0.25);

    vec3 sunDirection = uAtmosphereSunDirection.xyz;

    float muSun = dot(up, sunDirection);

    vec3 sun = sunTransmittance(cloudRadius(), muSun);

    // Two-stream estimate for a thick, strongly forward-
    // scattering (g ~ 0.85) layer: reflected from the lit
    // top, diffusely transmitted out of the base.
    float scaled = 0.15 * tau;

    float reflected = scaled / (scaled + 4.0 / 3.0);
    float transmitted = max(1.0 - reflected - exp(-tau), 0.0);

    bool fromAbove = dot(direction, up) < 0.0;

    float side = fromAbove ? reflected : transmitted;

    vec3 light = sun * max(muSun, 0.0) * side / ATMOSPHERE_PI;

    // Lumpy tops and bases: thicker parts of a cloud stand
    // higher in the sun (from above) or darker (from below).
    float thickness = clamp(tau / max(uCloudParams.y, 1e-3), 0.0, 1.0);

    light *= fromAbove ? mix(0.6, 1.1, thickness) : mix(1.2, 0.75, thickness);

    // Thin edges glow toward the sun (forward scattering).
    float cosAngle = dot(direction, sunDirection);

    float g = 0.6;

    float phase = (1.0 - g * g) / (4.0 * ATMOSPHERE_PI * pow(max(1.0 + g * g - 2.0 * g * cosAngle, 1e-4), 1.5));

    light += sun * (1.0 - exp(-cloud.depth)) * exp(-tau) * phase * 2.0;

    // Sky and ground light, dimming through twilight.
    float daylight = smoothstep(-0.15, 0.2, muSun);

    light +=
        (0.06 + 0.12 * uAtmosphereRadii.w)
        * daylight
        * sunTransmittance(cloudRadius(), max(muSun, 0.05))
        * (1.0 - exp(-cloud.depth));

    cloud.radiance = light * uCloudColor.rgb;

#ifdef ATMOSPHERE_ECLIPSES
    cloud.radiance *= eclipseInAtmosphere(position);
#endif

    return cloud;
}

// The atmosphere along [start, end] with the cloud layer
// where the ray crosses it: the air in front, the cloud,
// and (through its gaps) the air behind.
vec3 scatteringWithClouds(
    vec3 origin,
    vec3 direction,
    float start,
    float end,
    int steps,
    float jitter,
    float pixelAngle,
    out vec3 transmittance
)
{
    float hit = cloudsPresent() ? cloudHit(origin, direction, start, end) : -1.0;

    if (hit < 0.0)
    {
        return integrateScattering(origin, direction, start, end, steps, jitter, transmittance);
    }

    CloudSample cloud = shadeCloud(origin + hit * direction, direction, hit * pixelAngle);

    if (cloud.depth <= 1e-3)
    {
        return integrateScattering(origin, direction, start, end, steps, jitter, transmittance);
    }

    float share = (hit - start) / max(end - start, 1e-6);

    int before = clamp(int(round(float(steps) * share)), 2, max(steps - 2, 2));
    int after = max(steps - before, 2);

    vec3 frontTransmittance;
    vec3 front = integrateScattering(origin, direction, start, hit, before, jitter, frontTransmittance);

    vec3 backTransmittance;
    vec3 back = integrateScattering(origin, direction, hit, end, after, jitter, backTransmittance);

    float through = exp(-cloud.depth);

    transmittance = frontTransmittance * through * backTransmittance;

    return front + frontTransmittance * (cloud.radiance + through * back);
}

// Direct sunlight reaching a surface point (km, world
// space relative to the planet center) through the cloud
// layer above it: 1 = clear.
float cloudShadow(
    vec3 worldPosition
)
{
    if (!cloudsPresent())
    {
        return 1.0;
    }

    vec3 position = toAtmosphereSpace(worldPosition);

    vec3 sunDirection = uAtmosphereSunDirection.xyz;

    if (length(position) >= cloudRadius())
    {
        return 1.0;
    }

    float t = raySphere(position, sunDirection, cloudRadius());

    if (t <= 0.0)
    {
        return 1.0;
    }

    vec3 up = normalize(position + t * sunDirection);

    // Coarse detail is enough for soft shadows; forward
    // scattering lets some light through even thick cloud.
    return exp(-0.6 * cloudDepth(up, 0.15 * uCloudParams.z));
}
