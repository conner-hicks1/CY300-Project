#version 330 core


// =========================================================
// FXAA (Fast Approximate Anti-Aliasing)
// =========================================================
//
// After Lottes, "FXAA" (NVIDIA, 2009-2011), in the
// commonly used simplified form: find edges from luma
// contrast, walk along each edge to its ends, and blend
// across it in proportion to where this pixel sits on the
// edge; plus a sub-pixel pass for thin features.
//
// Runs on the final gamma-encoded image, where luma
// contrast matches what the eye sees.

uniform sampler2D uInput;
uniform vec2 uTexelSize;

in vec2 vTexCoord;

out vec4 FragColor;

#define EDGE_THRESHOLD_MIN 0.0312
#define EDGE_THRESHOLD_MAX 0.125
#define ITERATIONS 12
#define SUBPIXEL_QUALITY 0.75

float luma(vec3 rgb)
{
    return dot(rgb, vec3(0.299, 0.587, 0.114));
}

float stepQuality(int i)
{
    if (i < 5) return 1.0;
    if (i == 5) return 1.5;
    if (i < 10) return 2.0;
    if (i == 10) return 4.0;
    return 8.0;
}

void main()
{
    vec2 uv = vTexCoord;

    vec3 colorCenter = texture(uInput, uv).rgb;

    float lumaCenter = luma(colorCenter);

    float lumaDown  = luma(textureOffset(uInput, uv, ivec2( 0, -1)).rgb);
    float lumaUp    = luma(textureOffset(uInput, uv, ivec2( 0,  1)).rgb);
    float lumaLeft  = luma(textureOffset(uInput, uv, ivec2(-1,  0)).rgb);
    float lumaRight = luma(textureOffset(uInput, uv, ivec2( 1,  0)).rgb);

    float lumaMin = min(lumaCenter, min(min(lumaDown, lumaUp), min(lumaLeft, lumaRight)));
    float lumaMax = max(lumaCenter, max(max(lumaDown, lumaUp), max(lumaLeft, lumaRight)));

    float lumaRange = lumaMax - lumaMin;

    // Not an edge (or too dark to matter): leave it.
    if (lumaRange < max(EDGE_THRESHOLD_MIN, lumaMax * EDGE_THRESHOLD_MAX))
    {
        FragColor = vec4(colorCenter, 1.0);
        return;
    }

    float lumaDownLeft  = luma(textureOffset(uInput, uv, ivec2(-1, -1)).rgb);
    float lumaUpRight   = luma(textureOffset(uInput, uv, ivec2( 1,  1)).rgb);
    float lumaUpLeft    = luma(textureOffset(uInput, uv, ivec2(-1,  1)).rgb);
    float lumaDownRight = luma(textureOffset(uInput, uv, ivec2( 1, -1)).rgb);

    float lumaDownUp = lumaDown + lumaUp;
    float lumaLeftRight = lumaLeft + lumaRight;

    float lumaLeftCorners = lumaDownLeft + lumaUpLeft;
    float lumaDownCorners = lumaDownLeft + lumaDownRight;
    float lumaRightCorners = lumaDownRight + lumaUpRight;
    float lumaUpCorners = lumaUpRight + lumaUpLeft;

    // ---------------------------------------------------
    // Edge orientation
    // ---------------------------------------------------

    float edgeHorizontal =
        abs(-2.0 * lumaLeft + lumaLeftCorners)
        + abs(-2.0 * lumaCenter + lumaDownUp) * 2.0
        + abs(-2.0 * lumaRight + lumaRightCorners);

    float edgeVertical =
        abs(-2.0 * lumaUp + lumaUpCorners)
        + abs(-2.0 * lumaCenter + lumaLeftRight) * 2.0
        + abs(-2.0 * lumaDown + lumaDownCorners);

    bool isHorizontal = edgeHorizontal >= edgeVertical;

    float luma1 = isHorizontal ? lumaDown : lumaLeft;
    float luma2 = isHorizontal ? lumaUp : lumaRight;

    float gradient1 = luma1 - lumaCenter;
    float gradient2 = luma2 - lumaCenter;

    bool is1Steepest = abs(gradient1) >= abs(gradient2);

    float gradientScaled = 0.25 * max(abs(gradient1), abs(gradient2));

    float stepLength = isHorizontal ? uTexelSize.y : uTexelSize.x;

    float lumaLocalAverage;

    if (is1Steepest)
    {
        stepLength = -stepLength;
        lumaLocalAverage = 0.5 * (luma1 + lumaCenter);
    }
    else
    {
        lumaLocalAverage = 0.5 * (luma2 + lumaCenter);
    }

    // Move to the edge between this pixel and its steepest
    // neighbour.
    vec2 currentUv = uv;

    if (isHorizontal)
        currentUv.y += stepLength * 0.5;
    else
        currentUv.x += stepLength * 0.5;

    // ---------------------------------------------------
    // Walk along the edge in both directions
    // ---------------------------------------------------

    vec2 offset = isHorizontal ? vec2(uTexelSize.x, 0.0) : vec2(0.0, uTexelSize.y);

    vec2 uv1 = currentUv - offset;
    vec2 uv2 = currentUv + offset;

    float lumaEnd1 = luma(texture(uInput, uv1).rgb) - lumaLocalAverage;
    float lumaEnd2 = luma(texture(uInput, uv2).rgb) - lumaLocalAverage;

    bool reached1 = abs(lumaEnd1) >= gradientScaled;
    bool reached2 = abs(lumaEnd2) >= gradientScaled;

    if (!reached1) uv1 -= offset;
    if (!reached2) uv2 += offset;

    if (!(reached1 && reached2))
    {
        for (int i = 2; i < ITERATIONS; ++i)
        {
            if (!reached1)
            {
                lumaEnd1 = luma(texture(uInput, uv1).rgb) - lumaLocalAverage;
            }

            if (!reached2)
            {
                lumaEnd2 = luma(texture(uInput, uv2).rgb) - lumaLocalAverage;
            }

            reached1 = abs(lumaEnd1) >= gradientScaled;
            reached2 = abs(lumaEnd2) >= gradientScaled;

            if (!reached1) uv1 -= offset * stepQuality(i);
            if (!reached2) uv2 += offset * stepQuality(i);

            if (reached1 && reached2)
            {
                break;
            }
        }
    }

    // ---------------------------------------------------
    // Blend amount from position along the edge
    // ---------------------------------------------------

    float distance1 = isHorizontal ? (uv.x - uv1.x) : (uv.y - uv1.y);
    float distance2 = isHorizontal ? (uv2.x - uv.x) : (uv2.y - uv.y);

    bool isDirection1 = distance1 < distance2;

    float distanceFinal = min(distance1, distance2);
    float edgeThickness = distance1 + distance2;

    float pixelOffset = -distanceFinal / edgeThickness + 0.5;

    bool isLumaCenterSmaller = lumaCenter < lumaLocalAverage;

    bool correctVariation =
        ((isDirection1 ? lumaEnd1 : lumaEnd2) < 0.0) != isLumaCenterSmaller;

    float finalOffset = correctVariation ? pixelOffset : 0.0;

    // Sub-pixel aliasing (thin lines, single pixels).
    float lumaAverage =
        (1.0 / 12.0)
        * (2.0 * (lumaDownUp + lumaLeftRight) + lumaLeftCorners + lumaRightCorners);

    float subPixelOffset1 = clamp(abs(lumaAverage - lumaCenter) / lumaRange, 0.0, 1.0);
    float subPixelOffset2 = (-2.0 * subPixelOffset1 + 3.0) * subPixelOffset1 * subPixelOffset1;
    float subPixelOffsetFinal = subPixelOffset2 * subPixelOffset2 * SUBPIXEL_QUALITY;

    finalOffset = max(finalOffset, subPixelOffsetFinal);

    vec2 finalUv = uv;

    if (isHorizontal)
        finalUv.y += finalOffset * stepLength;
    else
        finalUv.x += finalOffset * stepLength;

    FragColor = vec4(texture(uInput, finalUv).rgb, 1.0);
}
