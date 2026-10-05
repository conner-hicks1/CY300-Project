// =========================================================
// Bodies: Eclipses
// =========================================================
//
// Every planet and moon as a sphere (graphics/
// bodies_block.py). A point sees the part of the sun's disc
// no body hides: the overlap of two discs, the sun's and
// the body's, as seen from the point. Partly covered =
// penumbra, fully = umbra (a total eclipse), a body smaller
// than the sun = an annular eclipse. Air around the
// occluder bends red sunset light into its umbra.
//
// Units: kilometers, relative to the camera.

layout(std140) uniform BodiesBlock
{
    vec4 uBodyParams;                   // x bodies, y sun angular radius (rad),
                                        // z the first bodies, which shade the
                                        // atmosphere's planet
    vec4 uBodySpheres[MAX_BODIES];      // xyz center (km), w radius (km)
    vec4 uBodyGlow[MAX_BODIES];         // rgb sunlight bent into the umbra
    vec4 uRingCenter;                   // xyz ringed planet's center (km),
                                        // w its equatorial radius (km)
    vec4 uRingFrame;                    // quaternion: world -> its frame
    vec4 uRingParams;                   // x inner, y outer radius (km),
                                        // z 1 = rings, w optical depth scale
    vec4 uRingShape;                    // x the planet's flattening
    vec4 uBodyLight[MAX_BODIES];        // rgb geometric albedo x color
};

const float BODIES_PI = 3.14159265358979;

// ---------------------------------------------------------
// Planetshine
// ---------------------------------------------------------
//
// Sunlight a neighboring body sends a point (illuminance
// per unit sun, per channel), and the direction it comes
// from: its geometric albedo times its disc's solid angle
// over pi, by the Lambert sphere's phase law (graphics/
// star_field.py reflected_illuminance): full when the point
// sees it fully lit ("full Earth" over the Moon's night
// side, ~1e-4 of sunlight), nothing when it sees its night
// side. The nearest bodies are in the block; the point's own
// body is skipped (skip = its center).
vec3 planetshine(
    vec3 point,
    vec3 towardSun,
    vec3 skip,
    int index,
    out vec3 direction
)
{
    vec4 body = uBodySpheres[index];

    direction = vec3(0.0, 1.0, 0.0);

    vec3 albedo = uBodyLight[index].rgb;

    if (distance(body.xyz, skip) < 1.0 || dot(albedo, albedo) <= 0.0)
    {
        return vec3(0.0);
    }

    vec3 toBody = body.xyz - point;

    float range = length(toBody);

    if (range <= body.w)
    {
        return vec3(0.0);
    }

    direction = toBody / range;

    // Phase angle at the body: between the sun and the point.
    float a = acos(clamp(dot(towardSun, -direction), -1.0, 1.0));

    float phase = (sin(a) + (BODIES_PI - a) * cos(a)) / BODIES_PI;

    float size = body.w / range;

    return albedo * size * size * phase;
}

// Rotate v by the unit quaternion q (xyzw).
vec3 rotateByQuaternion(
    vec4 q,
    vec3 v
)
{
    return v + 2.0 * cross(q.xyz, cross(q.xyz, v) + q.w * v);
}

// Area where two discs overlap (radii a and b, centers c
// apart; angles in radians).
float discOverlap(
    float a,
    float b,
    float c
)
{
    if (c >= a + b)
    {
        return 0.0;
    }

    if (c <= abs(a - b))
    {
        float r = min(a, b);

        return BODIES_PI * r * r;
    }

    float a2 = a * a;
    float b2 = b * b;
    float c2 = c * c;

    float alpha = acos(clamp((c2 + a2 - b2) / (2.0 * c * a), -1.0, 1.0));
    float beta = acos(clamp((c2 + b2 - a2) / (2.0 * c * b), -1.0, 1.0));

    float kite = sqrt(max((-c + a + b) * (c + a - b) * (c - a + b) * (c + a + b), 0.0));

    return a2 * alpha + b2 * beta - 0.5 * kite;
}

// Sunlight reaching a point (km, camera-relative) past the
// first `count` bodies, per channel: 1 = the whole sun.
// skip: center (km) of the body the point belongs to (its
// own night side is not an eclipse); any far point for
// none.
vec3 eclipseFrom(
    vec3 point,
    vec3 towardSun,
    vec3 skip,
    int count
)
{
    float sunRadius = uBodyParams.y;

    vec3 light = vec3(1.0);

    for (int i = 0; i < MAX_BODIES; ++i)
    {
        if (i >= count)
        {
            break;
        }

        vec4 body = uBodySpheres[i];

        if (distance(body.xyz, skip) < 1.0)
        {
            continue;
        }

        vec3 toBody = body.xyz - point;

        float along = dot(toBody, towardSun);

        float range = length(toBody);

        // Behind the point, or the point inside the body.
        if (along <= 0.0 || range <= body.w)
        {
            continue;
        }

        float bodyRadius = asin(body.w / range);

        // Angle between the sun's center and the body's
        // (atan2 keeps tiny angles exact).
        float separation = atan(length(cross(toBody, towardSun)), along);

        if (separation >= sunRadius + bodyRadius)
        {
            continue;
        }

        float hidden = clamp(
            discOverlap(sunRadius, bodyRadius, separation) / (BODIES_PI * sunRadius * sunRadius),
            0.0,
            1.0
        );

        light *= mix(vec3(1.0), uBodyGlow[i].rgb, hidden);
    }

    return light;
}

vec3 eclipse(
    vec3 point,
    vec3 towardSun,
    vec3 skip
)
{
    return eclipseFrom(point, towardSun, skip, int(uBodyParams.x + 0.5));
}
