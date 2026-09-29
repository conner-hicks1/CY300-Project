// =========================================================
// Procedural Sky
// =========================================================
//
// A cheap analytic sky: horizon-to-zenith gradient, a
// ground color below the horizon, a glow around the sun
// and (optionally) the sun disc itself.
//
// Used twice with the same uniforms (see
// graphics/environment.py SkyParameters):
//
//   * sky.frag draws it as the visible background, with
//     the sun disc;
//   * ibl_sky.frag bakes it into the environment cubemap
//     WITHOUT the disc, because the directional light
//     already adds the sun's direct light; baking the
//     disc too would light everything twice.

uniform vec3 uSunDirection;     // unit vector toward the sun
uniform vec3 uSunRadiance;      // sun color * intensity
uniform float uHasSun;          // 0 = no directional light

uniform vec3 uSkyZenith;
uniform vec3 uSkyHorizon;
uniform vec3 uSkyGround;
uniform float uSkyIntensity;

uniform vec2 uSunDisc;          // cos(outer radius), cos(inner radius)


vec3 skyRadiance(
    vec3 direction,
    bool withSunDisc
)
{
    direction = normalize(direction);

    float height = direction.y;

    // Sky: faster falloff near the horizon.
    vec3 sky = mix(
        uSkyHorizon,
        uSkyZenith,
        pow(clamp(height, 0.0, 1.0), 0.45)
    );

    // Ground: darkens away from the horizon.
    vec3 ground = mix(
        mix(uSkyHorizon, uSkyGround, 0.6),
        uSkyGround,
        clamp(-height * 3.0, 0.0, 1.0)
    );

    float aboveHorizon = smoothstep(-0.015, 0.015, height);

    vec3 color = mix(ground, sky, aboveHorizon);

    if (uHasSun > 0.5)
    {
        float cosAngle = dot(direction, uSunDirection);

        // Wide soft haze plus a tighter halo.
        float glow =
            0.08 * pow(max(cosAngle, 0.0), 6.0)
            + 0.35 * pow(max(cosAngle, 0.0), 64.0);

        color += uSunRadiance * glow * aboveHorizon;

        if (withSunDisc)
        {
            float disc = smoothstep(uSunDisc.x, uSunDisc.y, cosAngle);

            // The disc is far brighter than the sky, so it
            // blooms after tone mapping.
            color += uSunRadiance * disc * 40.0 * aboveHorizon;
        }
    }

    return color * uSkyIntensity;
}
