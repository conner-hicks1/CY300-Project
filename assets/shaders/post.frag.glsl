#version 450 core


// =========================================================
// Post-Processing: Bloom -> Exposure -> Tone Map -> Gamma
// =========================================================
//
// Input is the linear HDR scene. Output is display-ready
// (gamma-encoded) color, written either straight to the
// window or to an LDR buffer for FXAA.

in vec2 vTexCoord;

out vec4 FragColor;

uniform sampler2D uHdrBuffer;

// Blurred HDR from the bloom chain (half resolution).
uniform sampler2D uBloomTexture;
uniform bool uBloomEnabled;
uniform float uBloomIntensity;

uniform float uExposure;
uniform float uGamma;

// Eye adaptation: a 1x1 scale (exposure.frag.glsl).
uniform sampler2D uAutoExposureTexture;
uniform bool uAutoExposure;

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
    vec3 hdr = texture(uHdrBuffer, vTexCoord).rgb;

    // Energy-conserving: blend toward the blurred image
    // rather than adding to it.
    if (uBloomEnabled)
    {
        hdr = mix(
            hdr,
            texture(uBloomTexture, vTexCoord).rgb,
            uBloomIntensity
        );
    }

    hdr *= uExposure;

    if (uAutoExposure)
    {
        hdr *= texelFetch(uAutoExposureTexture, ivec2(0), 0).r;
    }

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
