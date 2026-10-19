#version 450 core


// =========================================================
// Glow
// =========================================================
//
// Added to the scene (HDR: bright enough, it blooms). A
// shell of gas is brightest where the eye looks through the
// most of it (face on), fading to its silhouette; along the
// mesh's v (0 at its source, 1 at its far end) it fades out
// (a plume's tail, a bow shock's edge). A little flicker. A
// cloud (dust raised by a landing) is drawn over the scene
// instead, as thick as it is.

in vec3 vWorldPosition;
in vec3 vNormal;
in vec2 vTexCoord;

uniform vec3 uGlowColor;        // HDR: color x strength
uniform float uGlowFalloff;     // how sharply the edges fade
uniform float uGlowTail;        // how sharply it fades along v
uniform float uGlowTime;        // s, for the flicker
uniform float uGlowOpacity;     // 0: glowing gas; else a cloud this opaque at its thickest

out vec4 fragColor;

void main()
{
    vec3 view = normalize(-vWorldPosition);

    float facing = abs(dot(normalize(vNormal), view));

    float edge = pow(facing, uGlowFalloff);

    float along = clamp(vTexCoord.y, 0.0, 1.0);

    // Bright from just past the source to a soft end.
    float tail = pow(1.0 - along, uGlowTail) * smoothstep(0.0, 0.06, along + 0.02);

    // A cloud (dust, smoke): drawn over what is behind it
    // (premultiplied alpha), billowing a little.
    if (uGlowOpacity > 0.0)
    {
        float billow = 0.85 + 0.15 * sin(uGlowTime * 3.0 + vTexCoord.x * 31.0) * sin(uGlowTime * 2.3 + along * 17.0);

        float alpha = clamp(uGlowOpacity * edge * tail * billow, 0.0, 1.0);

        fragColor = vec4(uGlowColor * alpha, alpha);

        return;
    }

    float flicker = 0.9 + 0.1 * sin(uGlowTime * 47.0 + along * 23.0) * sin(uGlowTime * 31.0 + vTexCoord.x * 40.0);

    fragColor = vec4(uGlowColor * edge * tail * flicker, 1.0);
}
