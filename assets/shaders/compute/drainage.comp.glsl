#version 450 core

#define WORKGROUP 1024

#include "include/workgroup.glsl"


// =========================================================
// Hydrology: Drainage
// =========================================================
//
// Where the rain that falls on land goes (planet/hydrology.py
// drainage is the reference, a priority flood):
//
//   fill        every basin rises to its spill point: the
//               lowest surface W with W >= elevation that
//               falls toward the sea everywhere, W_i =
//               max(z_i, min over neighbors W_j + 1 cm)
//               (Planchon & Darboux 2001), by repeated
//               sweeps until nothing changes
//   route       each land cell drains to the neighbor it was
//               filled from (the lowest)
//   accumulate  discharge = own runoff + what flows in, by
//               sweeps until it stops changing
//
// The flood spreads from a cell to its neighbor list, so a
// cell is filled from its `donors` (the cells listing it as
// a neighbor; -1 = none) and drains into one of them; what
// flows into a cell comes from its own neighbors.
//
// uPhase 0 starts (and runs up to uSweeps sweeps of) the
// fill, 1 continues it, 2 routes and starts accumulating,
// 3 continues; status[0] = 1 while the phase is unfinished.

layout(std430, binding = 0) readonly buffer Elevation { float elevation[]; };
layout(std430, binding = 1) readonly buffer Ocean { uint ocean[]; };
layout(std430, binding = 2) readonly buffer Donors { int donors[]; };       // cells x DONORS
layout(std430, binding = 3) readonly buffer Runoff { float runoff[]; };
layout(std430, binding = 4) buffer Filled { float filled[]; };
layout(std430, binding = 5) buffer Receiver { int receiver[]; };
layout(std430, binding = 6) buffer Discharge { float discharge[]; };
layout(std430, binding = 7) buffer Previous { float previous[]; };
layout(std430, binding = 8) buffer Status { float status[]; };
layout(std430, binding = 9) readonly buffer Neighbors { int neighbors[]; };   // cells x 4

uniform uint uCells;
uniform uint uDonors;
uniform uint uPhase;
uniform uint uSweeps;

const float UNFILLED = 1e30;
const float STEP = 0.01;

shared uint anyChange;

void main()
{
    uint lane = gl_LocalInvocationID.x;

    if (uPhase == 0u)
    {
        for (uint cell = lane; cell < uCells; cell += WORKGROUP)
        {
            filled[cell] = ocean[cell] != 0u ? elevation[cell] : UNFILLED;
        }

        syncAll();
    }

    if (uPhase <= 1u)
    {
        // -------------------------------------------------
        // Fill
        // -------------------------------------------------

        bool changed = true;

        for (uint sweep = 0u; sweep < uSweeps && changed; ++sweep)
        {
            if (lane == 0u)
            {
                anyChange = 0u;
            }

            syncAll();

            for (uint cell = lane; cell < uCells; cell += WORKGROUP)
            {
                if (ocean[cell] != 0u)
                {
                    continue;
                }

                float lowest = UNFILLED;

                for (uint k = 0u; k < uDonors; ++k)
                {
                    int donor = donors[cell * uDonors + k];

                    if (donor >= 0)
                    {
                        lowest = min(lowest, filled[donor]);
                    }
                }

                if (lowest >= UNFILLED)
                {
                    continue;
                }

                float level = max(elevation[cell], lowest + STEP);

                if (level < filled[cell])
                {
                    filled[cell] = level;
                    anyChange = 1u;
                }
            }

            syncAll();

            changed = anyChange != 0u;

            syncAll();
        }

        if (lane == 0u)
        {
            status[0] = changed ? 1.0 : 0.0;
        }

        return;
    }

    if (uPhase == 2u)
    {
        // -------------------------------------------------
        // Route
        // -------------------------------------------------

        for (uint cell = lane; cell < uCells; cell += WORKGROUP)
        {
            int best = -1;
            float lowest = UNFILLED;

            if (ocean[cell] == 0u && filled[cell] < UNFILLED)
            {
                for (uint k = 0u; k < uDonors; ++k)
                {
                    int donor = donors[cell * uDonors + k];

                    if (donor >= 0 && filled[donor] < lowest)
                    {
                        lowest = filled[donor];
                        best = donor;
                    }
                }
            }

            receiver[cell] = best;

            discharge[cell] = runoff[cell];
            previous[cell] = runoff[cell];
        }

        syncAll();
    }

    // -----------------------------------------------------
    // Accumulate (Jacobi: previous -> discharge)
    // -----------------------------------------------------

    bool changed = true;

    for (uint sweep = 0u; sweep < uSweeps && changed; ++sweep)
    {
        if (lane == 0u)
        {
            anyChange = 0u;
        }

        syncAll();

        for (uint cell = lane; cell < uCells; cell += WORKGROUP)
        {
            float total = runoff[cell];

            // A cell draining here has this one among its
            // donors: it is one of this cell's neighbors.
            for (uint k = 0u; k < 4u; ++k)
            {
                int other = neighbors[cell * 4u + k];

                // (Near cube corners a neighbor can be listed
                // twice: count it once.)
                bool repeated = false;

                for (uint before = 0u; before < k; ++before)
                {
                    repeated = repeated || neighbors[cell * 4u + before] == other;
                }

                if (!repeated && receiver[other] == int(cell))
                {
                    total += previous[other];
                }
            }

            discharge[cell] = total;

            if (total != previous[cell])
            {
                anyChange = 1u;
            }
        }

        syncAll();

        for (uint cell = lane; cell < uCells; cell += WORKGROUP)
        {
            previous[cell] = discharge[cell];
        }

        syncAll();

        changed = anyChange != 0u;

        syncAll();
    }

    if (lane == 0u)
    {
        status[0] = changed ? 1.0 : 0.0;
    }
}
