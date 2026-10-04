#version 450 core

#define WORKGROUP 1024

#include "include/workgroup.glsl"


// =========================================================
// Climate: Temperature Balance (one Newton step)
// =========================================================
//
// Per cell, absorbed sunlight balances emitted heat and the
// heat conducted to its 4 neighbors:
//
//     absorbed = e sigma T^4 + sum_j c_ij (T_i - T_j)
//
// Newton's method: linearize the emission around the
// current T and solve the linear system for the step with
// Jacobi-preconditioned conjugate gradients, iterations
// inside this one workgroup. A dispatch runs at most
// uMaxIterations of them (a few ms: the GPU cannot draw the
// frame meanwhile) and the next one carries on (uPhase 1)
// until the step is solved and applied. The CPU checks the
// step size between Newton steps (planet/climate.py
// solve_temperature is the same algorithm in float64).

layout(std430, binding = 0) readonly buffer Absorbed { float absorbed[]; };
layout(std430, binding = 1) readonly buffer Emissivity { float emissivity[]; };
layout(std430, binding = 2) readonly buffer Conductance { float conductance[]; };   // cells x 4
layout(std430, binding = 3) readonly buffer Neighbors { int neighbors[]; };         // cells x 4
layout(std430, binding = 4) buffer Kelvin { float kelvin[]; };

// Scratch: the step x, residual r, direction d, A d, the
// radiative slope and the preconditioner.
layout(std430, binding = 5) buffer Step { float stepX[]; };
layout(std430, binding = 6) buffer Residual { float residual[]; };
layout(std430, binding = 7) buffer Direction { float direction[]; };
layout(std430, binding = 8) buffer Applied { float applied[]; };
layout(std430, binding = 9) buffer Slope { float slope[]; };
layout(std430, binding = 10) buffer Diagonal { float diagonal[]; };

// [0] largest |step| (K) once applied, [1] CG iterations
// so far, [2] 1 = step applied, [3] r.z, [4] tolerance.
layout(std430, binding = 11) buffer Status { float status[]; };

uniform uint uCells;
uniform uint uMaxIterations;
uniform uint uTotalIterations;
uniform uint uPhase;            // 0 = new Newton step, 1 = carry on

const float STEFAN_BOLTZMANN = 5.670374e-8;

// sum_j c_ij (v_i - v_j): differences first, so large
// conductances do not swallow small temperature steps in
// single precision.
float conducted(
    uint cell,
    bool fromDirection
)
{
    float here = fromDirection ? direction[cell] : kelvin[cell];

    float total = 0.0;

    for (uint k = 0u; k < 4u; ++k)
    {
        uint neighbor = uint(neighbors[cell * 4u + k]);

        float there = fromDirection ? direction[neighbor] : kelvin[neighbor];

        total += conductance[cell * 4u + k] * (here - there);
    }

    return total;
}

void main()
{
    uint lane = gl_LocalInvocationID.x;

    // -----------------------------------------------------
    // Linearize: residual and radiative slope
    // -----------------------------------------------------

    float rz;
    float limit;
    float done = 0.0;

    if (uPhase == 0u)
    {
        float largestRhs = 0.0;

        rz = 0.0;

        for (uint cell = lane; cell < uCells; cell += WORKGROUP)
        {
            float t = kelvin[cell];

            float emitted = emissivity[cell] * STEFAN_BOLTZMANN * t * t * t * t;

            float rhs = absorbed[cell] - emitted - conducted(cell, false);

            float radiative = 4.0 * emitted / t;

            float totalConductance = 0.0;

            for (uint k = 0u; k < 4u; ++k)
            {
                totalConductance += conductance[cell * 4u + k];
            }

            slope[cell] = radiative;
            diagonal[cell] = radiative + totalConductance;

            stepX[cell] = 0.0;
            residual[cell] = rhs;

            float z = rhs / diagonal[cell];

            direction[cell] = z;

            rz += rhs * z;

            largestRhs = max(largestRhs, abs(rhs));
        }

        rz = workgroupSum(rz);

        limit = max(1e-4 * workgroupMax(largestRhs), 0.01);

        syncAll();
    }
    else
    {
        rz = status[3];
        limit = status[4];
    }

    float before = uPhase == 0u ? 0.0 : status[1];

    // -----------------------------------------------------
    // Conjugate gradients for the step
    // -----------------------------------------------------

    uint iteration = 0u;

    for (; iteration < uMaxIterations; ++iteration)
    {
        // A d = slope d + sum_j c_ij (d_i - d_j)
        float dAd = 0.0;

        for (uint cell = lane; cell < uCells; cell += WORKGROUP)
        {
            float ad = slope[cell] * direction[cell] + conducted(cell, true);

            applied[cell] = ad;

            dAd += direction[cell] * ad;
        }

        dAd = workgroupSum(dAd);

        float alpha = rz / max(dAd, 1e-30);

        float rzNext = 0.0;
        float largest = 0.0;

        for (uint cell = lane; cell < uCells; cell += WORKGROUP)
        {
            stepX[cell] += alpha * direction[cell];

            float r = residual[cell] - alpha * applied[cell];

            residual[cell] = r;

            rzNext += r * (r / diagonal[cell]);

            largest = max(largest, abs(r));
        }

        rzNext = workgroupSum(rzNext);
        largest = workgroupMax(largest);

        if (largest < limit)
        {
            done = 1.0;

            break;
        }

        float beta = rzNext / max(rz, 1e-30);

        // Everyone has read d (for A d) before it changes.
        syncAll();

        for (uint cell = lane; cell < uCells; cell += WORKGROUP)
        {
            direction[cell] = residual[cell] / diagonal[cell] + beta * direction[cell];
        }

        rz = rzNext;

        syncAll();
    }

    syncAll();

    float used = before + float(iteration);

    if (done < 0.5 && used < float(uTotalIterations))
    {
        // Not converged yet: the next dispatch carries on.
        if (lane == 0u)
        {
            status[1] = used;
            status[2] = 0.0;
            status[3] = rz;
            status[4] = limit;
        }

        return;
    }

    // -----------------------------------------------------
    // Damped update
    // -----------------------------------------------------

    float largestStep = 0.0;

    for (uint cell = lane; cell < uCells; cell += WORKGROUP)
    {
        float t = kelvin[cell];

        float step = clamp(stepX[cell], -0.5 * t, t);

        kelvin[cell] = max(t + step, 3.0);

        largestStep = max(largestStep, abs(step));
    }

    largestStep = workgroupMax(largestStep);

    if (lane == 0u)
    {
        status[0] = largestStep;
        status[1] = used;
        status[2] = 1.0;
    }
}
