#version 450 core

// =========================================================
// Auto Exposure (eye adaptation)
// =========================================================
//
// Rendered into a 1x1 target and blended with what is
// already there (GL constant-alpha blending), so the
// exposure follows the scene over ~half a second instead of
// jumping. post.frag.glsl multiplies it in.
//
// The scene's brightness is the log-average luminance of
// the (already blurred, half resolution) bloom image,
// ignoring near-black pixels: empty space around a planet
// should not make the planet glare. Adaptation is partial,
// as the eye's is: Titan's dim noon still looks dimmer than
// Earth's, but is no longer black.

in vec2 vTexCoord;

out vec4 FragColor;

uniform sampler2D uScene;

// Average luminance that needs no adjustment (Earth
// daylight with the default exposure).
uniform float uReference;

// 0 = none, 1 = full adaptation.
uniform float uAdaptation;

uniform vec2 uLimits;       // min, max exposure scale

// Brightest luminance brightening may lift a surface to.
uniform float uHighlight;

const int SAMPLES = 16;     // per axis

void main()
{
    float logSum = 0.0;
    float count = 0.0;

    // Brightest ordinary sample (the sun's disc excluded).
    float brightest = 0.0;

    for (int y = 0; y < SAMPLES; ++y)
    {
        for (int x = 0; x < SAMPLES; ++x)
        {
            vec2 uv = (vec2(x, y) + 0.5) / float(SAMPLES);

            vec3 color = textureLod(uScene, uv, 0.0).rgb;

            float luminance = dot(color, vec3(0.2126, 0.7152, 0.0722));

            if (luminance > 1e-3)
            {
                logSum += log(min(luminance, 50.0));
                count += 1.0;

                if (luminance < 20.0)
                {
                    brightest = max(brightest, luminance);
                }
            }
        }
    }

    float scale = 1.0;

    float coverage = count / float(SAMPLES * SAMPLES);

    // Mostly black (deep space, night): keep the exposure.
    if (coverage > 0.05)
    {
        float average = exp(logSum / count);

        scale = clamp(pow(uReference / average, uAdaptation), uLimits.x, uLimits.y);

        // Brighten no further than keeps the brightest
        // surfaces out of clipping (lunar regolith beside
        // long black shadows).
        if (scale > 1.0 && brightest > 0.0)
        {
            scale = max(1.0, min(scale, uHighlight / brightest));
        }

        // A small lit subject on black (a planet seen from
        // space) adapts only partly, as a camera's metering
        // would.
        scale = mix(1.0, scale, smoothstep(0.45, 0.9, coverage));
    }

    FragColor = vec4(scale, scale, scale, 1.0);
}
