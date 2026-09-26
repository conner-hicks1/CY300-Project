#version 330 core


// =========================================================
// Post-Processing: Exposure -> Tone Map -> Gamma
// =========================================================
//
// Input is the linear HDR scene. Output is display-ready
// sRGB-ish (gamma-encoded) color.

in vec2 vTexCoord;

out vec4 FragColor;

uniform sampler2D uHdrBuffer;

uniform float uExposure;
uniform float uGamma;

// Matches graphics/render_settings.py Tonemapper.
uniform int uTonemapper;

#define TONEMAP_NONE      0
#define TONEMAP_REINHARD  1
#define TONEMAP_ACES      2


// Narkowicz 2015, "ACES Filmic Tone Mapping Curve".
vec3 acesFilmic(vec3 x)
{
    return clamp(
        (x * (2.51 * x + 0.03))
        / (x * (2.43 * x + 0.59) + 0.14),
        0.0,
        1.0
    );
}

void main()
{
    vec3 hdr =
        texture(uHdrBuffer, vTexCoord).rgb
        * uExposure;

    vec3 mapped;

    if (uTonemapper == TONEMAP_REINHARD)
    {
        mapped = hdr / (hdr + vec3(1.0));
    }
    else if (uTonemapper == TONEMAP_ACES)
    {
        mapped = acesFilmic(hdr);
    }
    else
    {
        mapped = clamp(hdr, 0.0, 1.0);
    }

    FragColor = vec4(
        pow(mapped, vec3(1.0 / uGamma)),
        1.0
    );
}
