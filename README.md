# CY300 OpenGL Engine

A small 3D engine and scene editor written in Python on top of OpenGL 3.3,
built as a CY300 course project. It renders with physically based materials,
cascaded shadows and image-based lighting from a procedural sky, and comes with
an ImGui editor for building and saving scenes.

![Demo scene rendered by the engine](docs/images/scene.png)

## Features

**Rendering**

- Physically based shading (Cook-Torrance GGX, metallic-roughness as in glTF 2.0)
  with base color, metallic-roughness, normal, occlusion and emissive maps
- Directional, point and spot lights
- Cascaded shadow maps that follow the camera (stabilized against shimmering),
  plus spot light shadows, filtered with hardware PCF
- Procedural sky; ambient light and reflections baked from it
  (split-sum image-based lighting: irradiance, prefiltered specular, BRDF LUT)
- HDR pipeline: bloom, exposure, ACES / Reinhard tone mapping, gamma correction, FXAA
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

- Entity-component-system with parent / child transforms
- Fixed-timestep simulation, uniform buffers for per-frame data
- 190+ unit tests for everything that does not need a GPU

## Getting Started

### Requirements

- A GPU and driver supporting **OpenGL 3.3**
- **Python 3.14** (the version the project is developed and tested with)
- Developed and tested on Windows. Linux with OpenGL 3.3 should work but is
  untested; macOS would additionally need a forward-compatible context
  (not requested yet).

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

This opens the demo scene (`assets/scenes/demo.scene.json`).

| Option | Meaning |
| --- | --- |
| `--scene PATH` | Open a different scene file |
| `--play` | Start with the simulation running |
| `--no-vsync` | Uncapped frame rate (for measuring performance) |
| `--cprofile N` | Record a Python profile of the first N frames into `logs/profiles/` |
| `--exit-after S` | Quit after S seconds (smoke tests) |
| `--screenshot PATH` | With `--exit-after`, save the last frame as an image |

## Controls

| Input | Action |
| --- | --- |
| Hold right mouse button | Look around; **W A S D** move, **Q / E** down / up |
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
| `core/` | Window, input, events, timer, logging, profiler |
| `ecs/` | Entity registry and components |
| `systems/` | Transform, camera controller, rotator and render systems |
| `graphics/` | Renderer, shaders, textures, meshes, shadows, IBL, bloom |
| `editor/` | Scene editor panels, picking, undo history |
| `scene/` | Scene container and scene file (de)serialization |
| `resources/` | Handle-based resource managers |
| `math3d/` | Transforms, camera, matrix helpers |
| `ui/` | ImGui integration and engine / profiler panels |
| `assets/` | Shaders, textures, models, scenes |
| `tools/` | Asset generator and micro-benchmarks |
| `tests/` | pytest suite |

## Development

Run the tests (no GPU needed):

```bash
python -m pytest
```

Regenerate the procedural textures and models in `assets/`:

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
