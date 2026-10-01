// =========================================================
// Batched Draw Data
// =========================================================
//
// Objects are drawn many at a time with
// glMultiDrawElementsIndirect. Each object's camera-relative
// model matrix and normal matrix live in this storage
// buffer (filled by Renderer.prepare_draws; layout in
// graphics/draw_list.py).
//
// aDrawIndex is a per-instance attribute: each indirect
// command sets baseInstance to its object's index, so the
// draw reads its own record (OpenGL 4.5 has no gl_DrawID
// without an extension).

struct DrawRecord
{
    mat4 model;          // object -> camera-relative world
    mat4 normalMatrix;   // mat3 cofactor, padded
};

layout(std430, binding = 2) readonly buffer DrawRecords
{
    DrawRecord uDraws[];
};

layout(location = 5) in uint aDrawIndex;
