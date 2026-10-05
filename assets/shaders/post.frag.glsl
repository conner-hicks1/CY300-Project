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

// Eye adaptation: a 1x1 log scale (exposure.frag.glsl).
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

    // Night vision: in dim light the eye's rods take over
    // from its cones, which see color: moonlit and starlit
    // scenes look grey-blue (the Purkinje shift), faint
    // stars white. Luminance here is the scene's own (1 =
    // ~1e4 cd/m2, sunlit snow): cones fade below ~3 cd/m2,
    // rods alone below ~1e-3 cd/m2.
    float luminance = dot(hdr, vec3(0.2126, 0.7152, 0.0722));

    float rods = 1.0 - smoothstep(log(1e-7), log(3e-4), log(max(luminance, 1e-12)));

    hdr = mix(hdr, luminance * vec3(0.86, 0.96, 1.16), 0.85 * rods);

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
        hdr *= exp(texelFetch(uAutoExposureTexture, ivec2(0), 0).r);
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

    vec3 encoded = pow(mapped, vec3(1.0 / uGamma));

    // Dither by under one 8-bit step: smooth dim gradients
    // (a night sky) would otherwise show contour bands.
    float noise = fract(52.9829189 * fract(dot(gl_FragCoord.xy, vec2(0.06711056, 0.00583715))));

    encoded += (noise - 0.5) / 255.0;

    FragColor = vec4(encoded, 1.0);
}
