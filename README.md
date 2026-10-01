# CY300 OpenGL Engine

A small 3D engine and scene editor written in Python on top of OpenGL 4.5,
built as a CY300 course project. It renders with physically based materials,
cascaded shadows and image-based lighting, streams an Earth-sized procedural
planet under a physically based atmosphere, and comes with an ImGui editor for
building and saving scenes.

![The procedural planet, seen from the surface](docs/images/scene.png)

## Features

**Procedural planet**

- Earth-sized cube-sphere planet with continents, oceans, ridged mountain
  ranges, hills, snow lines and polar ice, from seeded 3D gradient noise
- Quadtree level of detail per cube face with horizon culling; chunks are built
  on worker threads and streamed in coarse-to-fine without holes or cracks
- Planet-aware camera: radial "up", altitude-scaled flying speed, stays above
  the terrain
- Terrain parameters editable live in the inspector (the planet rebuilds)

**Atmosphere**

| Sunset from 12 km | From orbit |
| --- | --- |
| ![Low sun over the mountains](docs/images/sunset.png) | ![The planet from space](docs/images/orbit.png) |

- Rayleigh, Mie and ozone scattering with multiple scattering (Hillaire 2020):
  transmittance and multiple-scattering lookup tables, then a raymarch per pixel
- Blue sky, sunsets, haze over distant terrain, and the planet's limb seen from
  space, from the same model and Earth's physical constants (all editable)
- Sunlight reaching every surface is filtered by the air above it: low sun
  turns orange, the night side goes dark
- Ambient lighting and reflections baked from the atmosphere at the camera

**Rendering**

- Physically based shading (Cook-Torrance GGX, metallic-roughness as in glTF 2.0)
  with base color, metallic-roughness, normal, occlusion and emissive maps
- Directional, point and spot lights
- Cascaded shadow maps that follow the camera (stabilized against shimmering),
  plus spot light shadows, filtered with hardware PCF
- Procedural sky for scenes without a planet; ambient light and reflections baked from it
  (split-sum image-based lighting: irradiance, prefiltered specular, BRDF LUT)
- HDR pipeline: bloom, exposure, ACES / Reinhard tone mapping, gamma correction, FXAA
- Batched drawing: one shared geometry buffer, one multi-draw-indirect call per
  material, frustum culling of every object and shadow caster
- Planet-scale precision: 64-bit world positions, camera-relative rendering and
  a reversed-Z infinite depth buffer
- OBJ and glTF / GLB model loading, including glTF materials and embedded textures
- Shader `#include`s and hot reload (edit a `.glsl` file while the engine runs)

**Editor**

![The scene editor](docs/images/editor.png)

- Hierarchy with drag-and-drop parenting, inspector for every component
- Click-to-select in the viewport, move / rotate / scale gizmos with snapping
- Undo / redo, scene files (JSON) with native open / save dialogs, model import
- Play / Stop: run the simulation, then restore the scene exactly
- Live render settings, material editing and a frame profiler (CPU + GPU timings)

**Engine**

- Entity-component-system with parent / child transforms (quaternion rotations)
- Background job system: worker threads, results handed back to the main thread
  under a per-frame time budget
- Fixed-timestep simulation, uniform buffers for per-frame data, OpenGL debug
  output routed to the log
- 300 unit tests for everything that does not need a GPU

## Getting Started

### Requirements

- A GPU and driver supporting **OpenGL 4.5** (any desktop GPU from the last
  decade; developed on Intel Iris Xe)
- **Python 3.14** (the version the project is developed and tested with)
- Developed and tested on Windows. Linux should work but is untested; macOS is
  not supported (Apple's OpenGL stops at 4.1).

### Install

```bash
python -m pip install -r requirements.txt
```

On Windows with several Python versions installed, use the launcher:

```bash
py -3.14 -m pip install -r requirements.txt
```

### Run

```bash
python main.py
```

This opens `assets/scenes/planet.scene.json` if it exists, otherwise the
built-in demo: a procedural planet with an atmosphere, the camera above a
mountain valley.
The planet streams in over the first few seconds.

| Option | Meaning |
| --- | --- |
| `--scene PATH` | Open a different scene file |
| `--play` | Start with the simulation running |
| `--hide-ui` | Start with the editor and debug UI hidden (**F1** toggles) |
| `--no-vsync` | Uncapped frame rate (for measuring performance) |
| `--cprofile N` | Record a Python profile of the first N frames into `logs/profiles/` |
| `--exit-after S` | Quit after S seconds (smoke tests) |
| `--screenshot PATH` | With `--exit-after`, save the last frame as an image |

## Controls

| Input | Action |
| --- | --- |
| Hold right mouse button | Look around; **W A S D** move, **Q / E** down / up (on a planet: toward / away from the ground; speed grows with altitude) |
| Left click | Select an object |
| **Q / W / E / R** | Select / move / rotate / scale tool |
| **F** | Focus the camera on the selection |
| **Delete**, **Ctrl+D** | Delete / duplicate the selection |
| **Ctrl+Z**, **Ctrl+Y** | Undo / redo |
| **Ctrl+S**, **Ctrl+O** | Save / open a scene |
| **Ctrl+P** | Play / stop |
| **Esc** | Clear the selection |
| **F1** | Show / hide the UI |
| **F5** | Reload all shaders |

## Project Layout

| Path | Contents |
| --- | --- |
| `main.py`, `application.py` | Entry point, main loop, demo content |
| `core/` | Window, input, events, timer, logging, profiler, job system |
| `ecs/` | Entity registry and components |
| `systems/` | Transform, camera controller, rotator, planet streaming and render systems |
| `planet/` | Noise, cube-sphere mapping, terrain, chunk building, level of detail |
| `graphics/` | Renderer, shaders, textures, meshes, shadows, IBL, atmosphere, bloom |
| `editor/` | Scene editor panels, picking, undo history |
| `scene/` | Scene container and scene file (de)serialization |
| `resources/` | Handle-based resource managers |
| `math3d/` | Transforms, camera, matrix helpers |
| `ui/` | ImGui integration and engine / profiler panels |
| `assets/` | Shaders, textures, scenes |
| `tools/` | Asset generator and micro-benchmarks |
| `tests/` | pytest suite (`tests/fixtures/` holds the OBJ / glTF test models) |

## Development

Run the tests (no GPU needed):

```bash
python -m pytest
```

Regenerate the procedural textures in `assets/` and the test models:

```bash
python tools/generate_assets.py
```

Benchmark the per-frame hot paths:

```bash
python tools/benchmark_hot_paths.py
```

Shaders live in `assets/shaders/` and reload automatically when saved. The
Profiler window (bottom of the screen) shows CPU and GPU time per render pass;
untick VSync there to see the real cost of a frame.

## License

Released into the public domain under the [Unlicense](UNLICENSE).
Dependencies (installed through `requirements.txt`) keep their own licenses.
