#version 450 core

#include "include/blocks.glsl"


// =========================================================
// Comets: Coma and Tails
// =========================================================
//
// Per pixel, the dust and gas along the view ray in front of
// whatever is drawn there, integrated in closed form
// (graphics/comets.py has the physics and the units: km,
// camera-relative; brightness from the comet's A f rho):
//
//   coma        density A f rho / (2 pi r^2): its column at a
//               projected distance b is A f rho / (2 b) over
//               the whole line, less where the ray starts or
//               ends inside it (two arctangents)
//   tails       Gaussian tubes widening with distance from
//               the nucleus; each piece of a tail is a
//               straight segment, and a ray crosses it once:
//               the column there is the tube's density times
//               its width over the crossing angle's sine.
//               The dust tail bends back along the orbit
//               (eight segments of a parabola); the ion tail
//               is straight.
//
// The dust scatters sunlight forward strongly (Henyey-
// Greenstein, g = 0.6; A f rho as measured near 30 deg of
// phase); the ions glow blue on their own.
//
// Added like the stars, before the air (which dims it).

in vec2 vTexCoord;

out vec4 FragColor;

const int MAX_COMETS = 4;

uniform int uCometCount;
uniform vec4 uCometCenter[MAX_COMETS];  // xyz nucleus (km), w coma size (km)
uniform vec4 uCometAxis[MAX_COMETS];    // xyz away from the sun, w A f rho (km)
uniform vec4 uCometBend[MAX_COMETS];    // xyz behind its motion, w dust tail length (km)
uniform vec4 uCometParams[MAX_COMETS];  // x sunlight, y gas, z ion tail length (km)

uniform sampler2D uSceneDepth;

const float PI = 3.14159265;

// Lag at the dust tail's end, as a share of its length
// (graphics/comets.py DUST_TAIL_BEND).
const float DUST_BEND = 0.12;

// Column (km^-1 * km) of a Gaussian tube along the segment
// a -> b (unit `axis`, length `span`), density on its axis
// `density(s)` = scale / width^2 * exp(-s / length), width
// growing as base + spread * s, s from the nucleus: what the
// ray (from the camera, unit `direction`, to `far`) passes.
float tube(
    vec3 a,
    vec3 axis,
    float span,
    float s0,
    float base,
    float spread,
    float length_,
    vec3 direction,
    float far
)
{
    // Closest approach between the ray and the segment's
    // line.
    float b = dot(direction, axis);

    float denominator = 1.0 - b * b;

    vec3 w0 = -a;   // ray origin (the camera, 0) minus a

    float d = dot(direction, w0);
    float e = dot(axis, w0);

    float t;
    float sigma;

    if (denominator < 1e-8)
    {
        // Looking straight along the tail.
        sigma = 0.0;
        t = max(dot(a, direction), 0.0);
    }
    else
    {
        t = (b * e - d) / denominator;
        sigma = (e - b * d) / denominator;
    }

    // Neighboring pieces share their joint: each weighs in
    // with a smooth window a tube's width wide there, summing
    // to one (no seams, no doubled beads).
    float raw = sigma;

    sigma = clamp(sigma, 0.0, span);
    t = clamp(dot(a + sigma * axis, direction), 0.0, far);

    vec3 gap = t * direction - (a + sigma * axis);

    float s = s0 + sigma;

    float width = base + spread * s;

    float window = smoothstep(-width, width, raw) * (1.0 - smoothstep(span - width, span + width, raw));

    float distance2 = dot(gap, gap);

    float crossing = max(sqrt(denominator), width / max(span, width));

    return window * exp(-0.5 * distance2 / (width * width) - s / length_)
        / (sqrt(2.0 * PI) * width * crossing);
}

void main()
{
    vec2 ndc = vTexCoord * 2.0 - 1.0;

    vec3 viewRay = vec3(ndc.x / uProjection[0][0], ndc.y / uProjection[1][1], -1.0);

    vec3 direction = normalize(transpose(mat3(uView)) * viewRay);

    // Up to whatever is drawn (reversed-Z infinite: depth =
    // near / view distance).
    float depth = texture(uSceneDepth, vTexCoord).r;

    float far = depth > 0.0 ? length(viewRay * (uProjection[3][2] / depth)) * 0.001 : 1e30;

    vec3 light = vec3(0.0);

    for (int i = 0; i < MAX_COMETS; ++i)
    {
        if (i >= uCometCount)
        {
            break;
        }

        vec3 center = uCometCenter[i].xyz;
        float coma = uCometCenter[i].w;

        vec3 away = uCometAxis[i].xyz;
        float afrho = uCometAxis[i].w;

        vec3 lag = uCometBend[i].xyz;
        float dustLength = uCometBend[i].w;

        float sunlight = uCometParams[i].x;
        float gas = uCometParams[i].y;
        float ionLength = uCometParams[i].z;

        // -------------------------------------------------
        // Coma
        // -------------------------------------------------

        float along = dot(center, direction);

        float b = max(length(center - along * direction), 0.05);

        float column = afrho / (2.0 * PI * b)
            * (atan((far - along) / b) - atan(-along / b))
            * exp(-b / coma);

        // -------------------------------------------------
        // Dust tail: a parabola from the nucleus, in eight
        // straight pieces
        // -------------------------------------------------

        float dust = 0.0;

        const int PIECES = 8;

        float reach = 4.0 * dustLength;

        for (int k = 0; k < PIECES; ++k)
        {
            float s0 = reach * float(k) / float(PIECES);
            float s1 = reach * float(k + 1) / float(PIECES);

            vec3 p0 = center + s0 * away + DUST_BEND * s0 * s0 / dustLength * lag;
            vec3 p1 = center + s1 * away + DUST_BEND * s1 * s1 / dustLength * lag;

            vec3 axis = p1 - p0;

            float span = length(axis);

            dust += tube(p0, axis / span, span, s0, 0.3 * coma, 0.07, dustLength, direction, far);
        }

        // The tail carries the coma's dust away (about as much
        // per km as a shell of the coma holds); it grows out
        // of the coma.
        dust *= 1.2 * afrho;

        // -------------------------------------------------
        // Ion tail: straight away from the sun
        // -------------------------------------------------

        float ions = tube(center, away, 3.0 * ionLength, 0.0, 0.1 * coma, 0.015, ionLength, direction, far);

        ions *= 0.3 * gas * afrho;

        // -------------------------------------------------
        // Light
        // -------------------------------------------------

        // Scattering angle: sunlight travels along `away`,
        // on to the camera along -direction.
        float g = 0.6;

        float cosAngle = dot(away, -direction);

        float hg = (1.0 - g * g) / pow(1.0 + g * g - 2.0 * g * cosAngle, 1.5);

        // (Relative to 30 deg of phase: 150 deg of scattering.)
        float reference = (1.0 - g * g) / pow(1.0 + g * g + 2.0 * g * 0.866, 1.5);

        float forward = min(hg / reference, 40.0);

        vec3 dustColor = vec3(1.05, 1.0, 0.9);
        vec3 ionColor = vec3(0.45, 0.9, 2.6);

        light += sunlight / (4.0 * PI) * (
            (column + dust) * forward * dustColor
            + ions * ionColor
        );
    }

    FragColor = vec4(light, 0.0);
}
