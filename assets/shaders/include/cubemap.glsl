// =========================================================
// Rendering into Cubemap Faces
// =========================================================
//
// With a fullscreen triangle drawn into face `uFace`,
// uv = vTexCoord * 2 - 1 is the fragment's position on
// the face. This table (the inverse of the OpenGL spec's
// face-selection rules) gives the world direction; it is
// mirrored and tested in graphics/cubemap.py.

uniform int uFace;

vec3 cubeFaceDirection(
    int face,
    vec2 uv
)
{
    vec3 direction;

    if (face == 0)      direction = vec3( 1.0, -uv.y, -uv.x);
    else if (face == 1) direction = vec3(-1.0, -uv.y,  uv.x);
    else if (face == 2) direction = vec3( uv.x,  1.0,  uv.y);
    else if (face == 3) direction = vec3( uv.x, -1.0, -uv.y);
    else if (face == 4) direction = vec3( uv.x, -uv.y,  1.0);
    else                direction = vec3(-uv.x, -uv.y, -1.0);

    return normalize(direction);
}
