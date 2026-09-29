#version 330 core


// =========================================================
// Bloom Upsample
// =========================================================
//
// 3x3 tent filter (Jimenez 2014). Drawn with additive
// blending onto the next larger mip, so each level adds
// its blur to the one above it.

uniform sampler2D uSource;

// Filter radius in UV units (RenderSettings.bloom_radius).
uniform float uRadius;

// Keeps the radius circular on non-square targets.
uniform float uAspectRatio;

in vec2 vTexCoord;

out vec4 FragColor;

void main()
{
    vec2 r = vec2(uRadius, uRadius * uAspectRatio);
    vec2 uv = vTexCoord;

    vec3 a = texture(uSource, uv + vec2(-r.x,  r.y)).rgb;
    vec3 b = texture(uSource, uv + vec2( 0.0,  r.y)).rgb;
    vec3 c = texture(uSource, uv + vec2( r.x,  r.y)).rgb;

    vec3 d = texture(uSource, uv + vec2(-r.x,  0.0)).rgb;
    vec3 e = texture(uSource, uv).rgb;
    vec3 f = texture(uSource, uv + vec2( r.x,  0.0)).rgb;

    vec3 g = texture(uSource, uv + vec2(-r.x, -r.y)).rgb;
    vec3 h = texture(uSource, uv + vec2( 0.0, -r.y)).rgb;
    vec3 i = texture(uSource, uv + vec2( r.x, -r.y)).rgb;

    vec3 result =
        e * 4.0
        + (b + d + f + h) * 2.0
        + (a + c + g + i);

    FragColor = vec4(result / 16.0, 1.0);
}
