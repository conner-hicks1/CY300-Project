#version 450 core

#include "include/blocks.glsl"


// =========================================================
// The Milky Way
// =========================================================
//
// The glow of the Galaxy's unresolved stars, by galactic
// longitude l and latitude b (graphics/star_field.py passes
// the galactic axes):
//
//   disc       a band a few degrees thick, brightest toward
//              the center in Sagittarius, with the bright
//              Cygnus and Carina star clouds, faint toward
//              the anticenter in Auriga; clumpy star clouds
//   bulge      the yellower central bulge
//   dust       dark lanes in the plane: the Great Rift from
//              Cygnus to Sagittarius, the Coalsack; dust
//              reddens what it does not hide
//   others     the Magellanic Clouds and the Andromeda
//              galaxy
//
// Brightness in the engine's units (planet/stars.py): the
// Sagittarius star clouds are ~20 magnitudes per square
// arcsecond (4e-8 per steradian here), the band elsewhere
// ~21.5: faint, seen only by the dark-adapted eye.
//
// The model is costly (noise everywhere) and the Galaxy does
// not move: it is baked once into a map in galactic
// coordinates (uBake, longitude across, latitude up), which
// the sky then samples. Drawn at depth 0 where nothing else
// is (depth-tested), added to the background; the air dims
// it and the day sky drowns it (atmosphere.frag.glsl).

in vec2 vTexCoord;

out vec4 FragColor;

// The galactic x (center), y (l = 90 deg) and z (north)
// axes in engine axes.
uniform vec3 uGalacticX;
uniform vec3 uGalacticY;
uniform vec3 uGalacticZ;

uniform float uBrightness;      // radiance at the brightest (1 = model peak)

uniform bool uBake;
uniform sampler2D uMilkyWayMap;

const float PI = 3.14159265;

float hash(
    vec3 p
)
{
    p = fract(p * 0.3183099 + 0.1);
    p *= 17.0;

    return fract(p.x * p.y * p.z * (p.x + p.y + p.z));
}

float noise(
    vec3 x
)
{
    vec3 i = floor(x);
    vec3 f = fract(x);

    f = f * f * (3.0 - 2.0 * f);

    return mix(
        mix(mix(hash(i), hash(i + vec3(1, 0, 0)), f.x), mix(hash(i + vec3(0, 1, 0)), hash(i + vec3(1, 1, 0)), f.x), f.y),
        mix(mix(hash(i + vec3(0, 0, 1)), hash(i + vec3(1, 0, 1)), f.x), mix(hash(i + vec3(0, 1, 1)), hash(i + vec3(1, 1, 1)), f.x), f.y),
        f.z
    );
}

float fbm(
    vec3 x
)
{
    float sum = 0.0;
    float amplitude = 0.5;

    for (int i = 0; i < 5; ++i)
    {
        sum += amplitude * noise(x);

        x *= 2.03;
        amplitude *= 0.5;
    }

    return sum;
}

// A Gaussian blob at (l0, b0), sizes in radians.
float blob(
    float l,
    float b,
    float l0,
    float b0,
    float width,
    float height
)
{
    float dl = mod(l - l0 + PI, 2.0 * PI) - PI;

    return exp(-0.5 * ((dl * dl) / (width * width) + ((b - b0) * (b - b0)) / (height * height)));
}

// The Galaxy's glow at galactic longitude l and latitude b
// (rad); g is the unit direction in galactic axes.
vec3 milkyWay(
    float l,
    float b,
    vec3 g
)
{
    float absL = abs(l);

    // ---------------------------------------------------
    // The disc
    // ---------------------------------------------------

    // Thicker toward the center.
    float thickness = 0.045 + 0.07 * exp(-l * l / 0.6);

    float band = exp(-abs(b) / thickness);

    float along =
        0.28
        + 0.72 * exp(-l * l / 0.9)
        + 0.35 * blob(l, b, 1.25, 0.0, 0.3, 0.2)       // Cygnus
        + 0.35 * blob(l, b, -1.35, -0.01, 0.25, 0.15)  // Carina
        + 0.25 * blob(l, b, 0.45, 0.0, 0.15, 0.1)      // Scutum cloud
        - 0.12 * smoothstep(2.2, 3.1, absL);           // Auriga, toward the anticenter

    // Star clouds.
    float clumps = 0.25 + 1.5 * fbm(g * 9.0);

    vec3 disc = vec3(0.9, 0.93, 1.0) * band * along * clumps;

    // ---------------------------------------------------
    // The bulge
    // ---------------------------------------------------

    // (Seen mostly south of the plane: the Great Sagittarius
    // Star Cloud, below the dust lanes.)
    float bulge = blob(l, b, 0.0, -0.05, 0.2, 0.12) * (0.7 + 0.6 * fbm(g * 14.0));

    vec3 light = disc + vec3(1.0, 0.88, 0.7) * 1.8 * bulge;

    // ---------------------------------------------------
    // Dust
    // ---------------------------------------------------

    // (Wandering, thicker and thinner: the lanes are clouds.)
    float wander = 0.025 * (fbm(g * 7.0 + 1.9) - 0.5);

    float lane = exp(-abs(b - 0.02 * exp(-l * l / 0.4) - wander) / (0.01 + 0.012 * fbm(g * 5.0 + 4.4)));

    float dust = 0.8 * lane * (0.2 + 1.4 * fbm(g * 22.0 + 3.7)) * (0.35 + 0.65 * exp(-l * l / 2.5));

    // The Great Rift: Cygnus to Sagittarius, north of the
    // plane.
    float rift = smoothstep(-0.2, 0.05, l) * smoothstep(1.55, 1.15, l);

    dust += 1.6 * rift * exp(-pow((b - 0.035 - wander) / 0.03, 2.0)) * (0.3 + 1.4 * fbm(g * 16.0 + 9.1));

    // The Coalsack, beside the Southern Cross.
    dust += 2.0 * blob(l, b, -0.9855, -0.012, 0.045, 0.04);

    light *= exp(-dust * vec3(1.25, 1.0, 0.8));

    // ---------------------------------------------------
    // Other galaxies
    // ---------------------------------------------------

    light += vec3(0.85, 0.9, 1.0) * (
        0.55 * blob(l, b, -1.3883, -0.5742, 0.07, 0.045)    // Large Magellanic Cloud
        + 0.3 * blob(l, b, -0.9983, -0.7734, 0.035, 0.025)  // Small Magellanic Cloud
    ) * (0.6 + 0.8 * fbm(g * 40.0));

    light += vec3(1.0, 0.92, 0.8) * 0.22 * blob(l, b, 2.1148, -0.3765, 0.02, 0.008);   // Andromeda

    return light;
}

void main()
{
    if (uBake)
    {
        float l = (vTexCoord.x * 2.0 - 1.0) * PI;
        float b = (vTexCoord.y - 0.5) * PI;

        vec3 g = vec3(cos(b) * cos(l), cos(b) * sin(l), sin(b));

        FragColor = vec4(milkyWay(l, b, g), 1.0);

        return;
    }

    vec2 ndc = vTexCoord * 2.0 - 1.0;

    vec3 viewRay = vec3(ndc.x / uProjection[0][0], ndc.y / uProjection[1][1], -1.0);

    vec3 direction = normalize(transpose(mat3(uView)) * viewRay);

    vec3 g = vec3(dot(uGalacticX, direction), dot(uGalacticY, direction), dot(uGalacticZ, direction));

    float b = asin(clamp(g.z, -1.0, 1.0));
    float l = atan(g.y, g.x);

    vec3 light = texture(uMilkyWayMap, vec2(l / (2.0 * PI) + 0.5, b / PI + 0.5)).rgb;

    FragColor = vec4(light * uBrightness, 0.0);
}
