// =========================================================
// One-Workgroup Solvers
// =========================================================
//
// The iterative solvers (compute/temperature.comp.glsl,
// compute/drainage.comp.glsl) run as a single workgroup of
// WORKGROUP invocations that loops by itself, so every
// iteration ends in a barrier instead of a new dispatch.
// Each invocation looks after every WORKGROUP-th cell.
//
// Needs `#define WORKGROUP <n>` (a power of two) before
// inclusion.

layout(local_size_x = WORKGROUP) in;

shared float sharedValues[WORKGROUP];

// Writes to storage buffers from every invocation are seen
// by all of them after this.
void syncAll()
{
    memoryBarrierBuffer();
    memoryBarrierShared();
    barrier();
}

float workgroupSum(
    float value
)
{
    uint lane = gl_LocalInvocationID.x;

    sharedValues[lane] = value;

    barrier();

    for (uint stride = WORKGROUP / 2u; stride > 0u; stride >>= 1u)
    {
        if (lane < stride)
        {
            sharedValues[lane] += sharedValues[lane + stride];
        }

        barrier();
    }

    float total = sharedValues[0];

    barrier();

    return total;
}

float workgroupMax(
    float value
)
{
    uint lane = gl_LocalInvocationID.x;

    sharedValues[lane] = value;

    barrier();

    for (uint stride = WORKGROUP / 2u; stride > 0u; stride >>= 1u)
    {
        if (lane < stride)
        {
            sharedValues[lane] = max(sharedValues[lane], sharedValues[lane + stride]);
        }

        barrier();
    }

    float total = sharedValues[0];

    barrier();

    return total;
}
