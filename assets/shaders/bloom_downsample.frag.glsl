#version 450 core


// =========================================================
// Bloom Downsample
// =========================================================
//
// 13-tap filter from Jimenez, "Next Generation Post
// Processing in Call of Duty: Advanced Warfare"
// (SIGGRAPH 2014). Box-filters overlapping 4x4 regions,
// which avoids the pulsing artifacts of a plain 2x2
// downsample.
//
// On the first pass (full-resolution HDR input) each
// group is weighted by 1 / (1 + luma) (the "Karis
// average"), so single ultra-bright pixels (fireflies)
// cannot flicker the whole bloom.

uniform sampler2D uSource;
uniform vec2 uSourceTexelSize;
uniform bool uFirstPass;

in vec2 vTexCoord;

out vec4 FragColor;

float karisWeight(
    vec3 color
)
{
    float luma = dot(color, vec3(0.2126, 0.7152, 0.0722));

    return 1.0 / (1.0 + luma);
}

void main()
{
    vec2 t = uSourceTexelSize;
    vec2 uv = vTexCoord;

    vec3 a = texture(uSource, uv + t * vec2(-2.0,  2.0)).rgb;
    vec3 b = texture(uSource, uv + t * vec2( 0.0,  2.0)).rgb;
    vec3 c = texture(uSource, uv + t * vec2( 2.0,  2.0)).rgb;

    vec3 d = texture(uSource, uv + t * vec2(-2.0,  0.0)).rgb;
    vec3 e = texture(uSource, uv).rgb;
    vec3 f = texture(uSource, uv + t * vec2( 2.0,  0.0)).rgb;

    vec3 g = texture(uSource, uv + t * vec2(-2.0, -2.0)).rgb;
    vec3 h = texture(uSource, uv + t * vec2( 0.0, -2.0)).rgb;
    vec3 i = texture(uSource, uv + t * vec2( 2.0, -2.0)).rgb;

    vec3 j = texture(uSource, uv + t * vec2(-1.0,  1.0)).rgb;
    vec3 k = texture(uSource, uv + t * vec2( 1.0,  1.0)).rgb;
    vec3 l = texture(uSource, uv + t * vec2(-1.0, -1.0)).rgb;
    vec3 m = texture(uSource, uv + t * vec2( 1.0, -1.0)).rgb;

    vec3 result;

    if (uFirstPass)
    {
        // Five groups, each Karis-weighted.
        vec3 g0 = (j + k + l + m) * 0.25;
        vec3 g1 = (a + b + d + e) * 0.25;
        vec3 g2 = (b + c + e + f) * 0.25;
        vec3 g3 = (d + e + g + h) * 0.25;
        vec3 g4 = (e + f + h + i) * 0.25;

        float w0 = karisWeight(g0) * 0.5;
        float w1 = karisWeight(g1) * 0.125;
        float w2 = karisWeight(g2) * 0.125;
        float w3 = karisWeight(g3) * 0.125;
        float w4 = karisWeight(g4) * 0.125;

        result =
            (g0 * w0 + g1 * w1 + g2 * w2 + g3 * w3 + g4 * w4)
            / (w0 + w1 + w2 + w3 + w4);
    }
    else
    {
        result =
            e * 0.125
            + (a + c + g + i) * 0.03125
            + (b + d + f + h) * 0.0625
            + (j + k + l + m) * 0.125;
    }

    // Guard against NaN/Inf from any source pixel. (No floor:
    // in the dark the eye adapts ~1e5 times, and a floor
    // would turn into a grey veil.)
    result = any(isnan(result)) || any(isinf(result))
        ? vec3(0.0)
        : clamp(result, vec3(0.0), vec3(60000.0));

    FragColor = vec4(result, 1.0);
}
