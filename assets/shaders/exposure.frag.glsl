#version 450 core


// =========================================================
// Auto Exposure (eye adaptation)
// =========================================================
//
// Rendered into a 1x1 target and blended with what is
// already there (GL constant-alpha blending), so the
// exposure follows the scene instead of jumping. The target
// holds the exposure's logarithm: blending logs eases it
// geometrically, as fast from x10000 down to x3 as from x3
// to x1 (a planet coming into view out of the dark).
// post.frag.glsl applies exp() of it.
//
// The scene is measured in 16 x 16 cells, each the average
// of 4 x 4 samples, so a small bright object counts in
// proportion to its size and does not flicker in and out
// between samples.
//
//   daylight       the log-average of the cells sets it;
//                  adaptation is partial, as the eye's is
//                  (Titan's dim noon still looks dim), and
//                  within uLimits
//   the dark       as the light around fades the eye goes
//                  on adapting, up to uDarkLimit: starlight,
//                  the Milky Way, earthshine on the Moon's
//                  night side show at their real brightness
//   highlights     it never brightens past where the
//                  brightest surfaces clip (a sunlit planet
//                  in view outshines the stars, as in any
//                  photograph), and a night scene stays dim;
//                  with the sun's disc in view, not at all

in vec2 vTexCoord;

out vec4 FragColor;

uniform sampler2D uScene;

// The blurred scene (the bloom chain): its peaks are the
// brightest surfaces bigger than a few pixels (a small
// sunlit planet in the dark), without the stars, which the
// blur spreads to nothing. uHasPeak 0: no bloom.
uniform sampler2D uPeak;
uniform bool uHasPeak;

// Average luminance that needs no adjustment (Earth
// daylight with the default exposure).
uniform float uReference;

// 0 = none, 1 = full adaptation.
uniform float uAdaptation;

uniform vec2 uLimits;       // min, max exposure scale in daylight

// Most the eye brightens in the dark.
uniform float uDarkLimit;

// Brightest luminance brightening may lift a surface to, in
// lit scenes and in the dark: a night landscape under
// moonlight or earthlight still looks dim to the adapted
// eye, never like day.
uniform float uHighlight;
uniform float uNightHighlight;

const int CELLS = 16;       // per axis
const int SUBSAMPLES = 4;   // per cell axis

// Below this a cell is black (empty space between the
// stars): it says nothing about the light around.
const float BLACK = 1e-11;

// The sun's disc and its glare.
const float SUN = 20.0;

// What counts as the light around: no dimmer than this
// share of the brightest thing in view (a sunlit planet is
// metered on itself, not on the starry black around it).
const float CONTRAST = 1e-4;

void main()
{
    float logSum = 0.0;
    float count = 0.0;

    float cellLuminance[CELLS * CELLS];

    // Brightest cell (the sun excluded), and whether the
    // sun is in view.
    float brightest = 0.0;
    bool sun = false;

    for (int y = 0; y < CELLS; ++y)
    {
        for (int x = 0; x < CELLS; ++x)
        {
            float luminance = 0.0;
            float peak = 0.0;

            for (int j = 0; j < SUBSAMPLES; ++j)
            {
                for (int i = 0; i < SUBSAMPLES; ++i)
                {
                    vec2 uv = (vec2(x, y) + (vec2(i, j) + 0.5) / float(SUBSAMPLES)) / float(CELLS);

                    vec3 color = textureLod(uScene, uv, 0.0).rgb;

                    float sample_ = dot(color, vec3(0.2126, 0.7152, 0.0722));

                    luminance += sample_;
                    peak = max(peak, sample_);
                }
            }

            luminance /= float(SUBSAMPLES * SUBSAMPLES);

            if (peak > SUN)
            {
                sun = true;
                cellLuminance[y * CELLS + x] = 0.0;
                continue;
            }

            brightest = max(brightest, luminance);

            cellLuminance[y * CELLS + x] = luminance;

            if (uHasPeak)
            {
                vec3 blurred = textureLod(uPeak, (vec2(x, y) + 0.5) / float(CELLS), 0.0).rgb;

                float top = dot(blurred, vec3(0.2126, 0.7152, 0.0722));

                if (top < SUN)
                {
                    brightest = max(brightest, top);
                }
            }

        }
    }

    float floor_ = max(BLACK, CONTRAST * brightest);

    for (int i = 0; i < CELLS * CELLS; ++i)
    {
        float luminance = cellLuminance[i];

        if (luminance > floor_)
        {
            logSum += log(luminance);
            count += 1.0;
        }
    }

    // The light around: its log-average (all black: the
    // darkness of space).
    float average = count > 0.05 * float(CELLS * CELLS) ? exp(logSum / count) : BLACK;

    // Daylight scenes adapt within uLimits (Titan's dim noon
    // still looks dim); in the dark the eye goes on, up to
    // uDarkLimit.
    float day = smoothstep(log(1e-4), log(1e-2), log(average));

    float ceiling = exp(mix(log(uDarkLimit), log(uLimits.y), day));

    float scale = clamp(pow(uReference / average, uAdaptation), uLimits.x, ceiling);

    // No brighter than keeps the brightest surfaces out of
    // clipping (lunar regolith beside long black shadows; a
    // sunlit planet on black), and when the brightest thing
    // in view is itself dim (moonlight, earthlight), not
    // even that: night scenes look dim to the adapted eye.
    if (brightest > 0.0)
    {
        float highlight = exp(mix(
            log(uNightHighlight),
            log(uHighlight),
            smoothstep(log(1e-4), log(1e-2), log(brightest))
        ));

        scale = min(scale, max(highlight / brightest, uLimits.x));
    }

    if (sun)
    {
        scale = min(scale, 1.0);
    }

    scale = clamp(scale, uLimits.x, uDarkLimit);

    FragColor = vec4(vec3(log(scale)), 1.0);
}
