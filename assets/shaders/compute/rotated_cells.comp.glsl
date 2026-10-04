#version 450 core

layout(local_size_x = 256) in;


// =========================================================
// Tectonics: Where Each Plate's Crust Came From
// =========================================================
//
// For every plate rotation and every grid direction: the
// sphere-grid cell the rotated direction falls in
// (planet/tectonics.py rotated_cells is the reference;
// planet/sphere_grid.py the grid: equal-angle cube faces).

layout(std430, binding = 0) readonly buffer Directions { float directions[]; };   // cells x 3
layout(std430, binding = 1) readonly buffer Rotations { float rotations[]; };     // count x 9, row-major
layout(std430, binding = 2) writeonly buffer Cells { int cells[]; };              // count x cells

uniform uint uCells;
uniform uint uRotations;
uniform int uResolution;

const vec3 FACE_NORMALS[6] = vec3[6](
    vec3(1, 0, 0), vec3(-1, 0, 0), vec3(0, 1, 0), vec3(0, -1, 0), vec3(0, 0, 1), vec3(0, 0, -1)
);

const vec3 FACE_U[6] = vec3[6](
    vec3(0, 0, -1), vec3(0, 0, 1), vec3(1, 0, 0), vec3(1, 0, 0), vec3(1, 0, 0), vec3(-1, 0, 0)
);

const vec3 FACE_V[6] = vec3[6](
    vec3(0, 1, 0), vec3(0, 1, 0), vec3(0, 0, -1), vec3(0, 0, 1), vec3(0, 1, 0), vec3(0, 1, 0)
);

const float FOUR_OVER_PI = 1.27323954474;

void main()
{
    uint index = gl_GlobalInvocationID.x;

    if (index >= uCells * uRotations)
    {
        return;
    }

    uint rotation = index / uCells;
    uint cell = index % uCells;

    vec3 d = vec3(directions[cell * 3u], directions[cell * 3u + 1u], directions[cell * 3u + 2u]);

    uint base = rotation * 9u;

    vec3 p = vec3(
        dot(vec3(rotations[base], rotations[base + 1u], rotations[base + 2u]), d),
        dot(vec3(rotations[base + 3u], rotations[base + 4u], rotations[base + 5u]), d),
        dot(vec3(rotations[base + 6u], rotations[base + 7u], rotations[base + 8u]), d)
    );

    // The face it points into most (first on ties, like
    // numpy's argmax).
    int face = 0;
    float best = dot(p, FACE_NORMALS[0]);

    for (int f = 1; f < 6; ++f)
    {
        float along = dot(p, FACE_NORMALS[f]);

        if (along > best)
        {
            best = along;
            face = f;
        }
    }

    float a = atan(dot(p, FACE_U[face]) / best) * FOUR_OVER_PI;
    float b = atan(dot(p, FACE_V[face]) / best) * FOUR_OVER_PI;

    int n = uResolution;

    int i = clamp(int((a + 1.0) * 0.5 * float(n)), 0, n - 1);
    int j = clamp(int((b + 1.0) * 0.5 * float(n)), 0, n - 1);

    cells[index] = (face * n + i) * n + j;
}
