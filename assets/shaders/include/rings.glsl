// =========================================================
// Planetary Rings
// =========================================================
//
// The ring system in the bodies block: a disk in its
// planet's equatorial plane from uRingParams.x to .y (km),
// its normal optical depth by radius in a profile texture
// (graphics/rings.py). Used to draw the rings
// (rings.frag.glsl) and for their shadow on whatever lies
// in it (lit.frag.glsl). Needs include/bodies.glsl.

uniform sampler2D uRingProfile;

bool ringsPresent()
{
    return uRingParams.z > 0.5;
}

// Normal optical depth at a distance (km) from the center.
float ringOpticalDepth(
    float radius
)
{
    if (radius < uRingParams.x || radius > uRingParams.y)
    {
        return 0.0;
    }

    float u = (radius - uRingParams.x) / max(uRingParams.y - uRingParams.x, 1e-3);

    return textureLod(uRingProfile, vec2(u, 0.5), 0.0).r * uRingParams.w;
}

// Direct sunlight reaching a point (km, world space
// relative to the ringed planet's center) through the
// rings: 1 = not in their shadow.
float ringShadow(
    vec3 worldPosition,
    vec3 towardSun
)
{
    if (!ringsPresent())
    {
        return 1.0;
    }

    vec3 p = rotateByQuaternion(uRingFrame, worldPosition);
    vec3 l = rotateByQuaternion(uRingFrame, towardSun);

    if (abs(l.y) < 1e-4)
    {
        return 1.0;
    }

    // Where the ray toward the sun crosses the ring plane.
    float t = -p.y / l.y;

    if (t <= 0.0)
    {
        return 1.0;
    }

    vec3 crossing = p + t * l;

    float tau = ringOpticalDepth(length(crossing.xz));

    return exp(-tau / abs(l.y));
}
