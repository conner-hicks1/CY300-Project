# CY300 OpenGL Engine

A small 3D engine and scene editor written in Python on top of OpenGL 4.5,
built as a CY300 course project. It renders with physically based materials,
cascaded shadows and image-based lighting, streams an Earth-sized procedural
planet under a physically based atmosphere, and comes with an ImGui editor for
building and saving scenes.

![The procedural planet, seen from the surface](docs/images/scene.png)

## Features

**Procedural planet**

- Earth-sized cube-sphere planet: continents and ocean basins from the plate
  simulation (or seeded noise without it), ridged mountain ranges, hills,
  and biomes, snow and sea ice from the climate model
- Quadtree level of detail per cube face with horizon culling; chunks are built
  on worker threads and streamed in coarse-to-fine without holes or cracks
- Planet-aware camera: radial "up", altitude-scaled flying speed, stays above
  the terrain
- Terrain parameters editable live in the inspector (the planet rebuilds)

**Bodies of the solar system**

![Mars from orbit, the Moon's surface, Io, Venus's surface under its clouds, Titan's surface and Europa](docs/images/bodies.png)

- Body profiles (`assets/bodies/*.json`, real data from NASA's fact sheets) for
  Mercury, Venus, Earth, the Moon, Mars, Phobos, Deimos, Vesta, Ceres,
  comet 67P, Jupiter, Io, Europa, Ganymede, Callisto, Saturn, Enceladus,
  Titan, Uranus, Neptune, Triton, Pluto and Haumea:
  size, mass, rotation, tilt, orbit, star, temperature, surface liquid and
  colors, terrain relief, atmosphere (pressure, gases, aerosols), geology
- Physics derived from them: surface gravity, sunlight, equilibrium and
  greenhouse temperature, atmospheric scale height (H = RT / Mg), lapse rate
  (g / cp, less in moist or thin air), light scattering from pressure and gas
  mix, colored dust and haze, the star's color from its temperature, the
  sun's apparent size and (compressed) brightness
- Liquids and ices from phase physics (triple and critical points, vapor
  pressure curves): seas of water or methane freeze where colder than their
  freezing point and boil away (leaving dry basins) when hotter than their
  boiling point at the surface pressure; no liquid at all below the triple
  point (water on Mars). Ice caps of water, carbon dioxide, nitrogen or
  methane form below the frost point set by the gas's partial pressure
  (Mars's water-ice caps at ~-76 C, Pluto's and Triton's nitrogen ice).
  Life is a switch: without it, the same climate zones are bare ground
- File > New Planet builds any of them; Planet panel > Body shows its facts
  and turns the planet into another body (undoable)
- Dry worlds keep their basins, methane seas (Titan), glowing lava lakes
  (Io), mineral surface palettes, gas-giant cloud bands, and a black sky with
  hard shadows where there is no air
- Add a body by writing a new profile file; the loader validates it

**Gas and ice giants**

![Jupiter's Great Red Spot, Saturn (flattened), Uranus and Neptune with its Great Dark Spot](docs/images/giants.png)

- Flattened by their spin: cloud tops on an ellipsoid (Jupiter 6.5% wider
  at the equator, Saturn ~10%); the atmosphere, clouds and lighting work in
  a stretched space where the ellipsoid is a sphere, so the haze and limb
  follow the shape exactly
- Weather per pixel: belts and zones of irregular width, festoons and
  streaks where the jet streams shear, the great storms swirling the bands
  around them (Jupiter's Great Red Spot, Neptune's Great Dark Spot), strings
  of white ovals, Saturn's hexagonal north polar jet
- No ground: fly down through the cloud tops into the decks below (see
  the next section)

**Descending through thick air**

![Down into Jupiter: the cloud tops, inside the ammonia haze, the brown ammonium hydrosulfide deck, the dark water clouds, the deep air glowing orange and white-hot at the bottom of the model; Venus: above its clouds, inside them, and the orange daylight beneath](docs/images/descent.png)

- The air column follows its physics (`planet/air.py`): rocky worlds cool by
  their lapse rate up to a tropopause at the radiative skin temperature
  (Earth: 0.265 bar at 10 km, as in the standard atmosphere); below a giant's
  tops the gas convects, heating along the dry adiabat (Jupiter ~2 K/km;
  ~470 K and ~17 bar 150 km down, close to the Galileo probe's numbers). The
  HUD shows the pressure and temperature around the camera
- Cloud decks as layers of the air, placed where their gases condense:
  Venus's sulfuric-acid haze and cloud decks (31-90 km, optical depth ~28);
  Jupiter's and Saturn's ammonia, ammonium hydrosulfide and water clouds;
  Uranus's and Neptune's methane, hydrogen sulfide and water clouds. They
  are part of the scattering lookup tables, so they light consistently from
  orbit and from inside
- From inside, the decks are patchy (3D noise at each deck's own scale,
  averaging out so it does not change the light); the raymarch splits rays
  at the decks' boundaries and spends its samples by optical depth
- Light dims with depth as it really does: thick cloud passes most light on
  diffusely rather than stopping it (two-stream transmission), so under
  Venus's clouds it is a dim, shadowless orange day; deep in a giant the
  sunlight runs out entirely
- Heat you can see: the deep air of a giant glows by its temperature
  (blackbody color and brightness, in the same units as sunlight: dull red
  past ~1100 K, orange by ~1500 K) down to the bottom of the model at 3000 K
- Lightning in the water clouds: storm cells flash on their own clocks, a few
  strokes each, lighting the cloud around them; seen from above, glows
  spreading through the deck
- The model's bottom (where the air reaches 3000 K, ~1400 km down on
  Jupiter) is only where drawing stops: there are no entry effects, drag,
  heating or crush depth, left for a game built on top to define
- Not drawn: Venus's superrefraction (its dense air bends light enough to make
  the horizon curve up like a bowl), since the haze hides everything beyond
  a few km, where the effect would only shift the horizon by ~2 m

**Rings**

![Saturn's rings from their lit and unlit faces, and Uranus's narrow dark rings](docs/images/rings.png)

- Ring systems in the equatorial plane from the profiles: Saturn's C, B and
  A rings, the Cassini Division, the Encke Gap and the F ring; Uranus's
  narrow, coal-dark rings; Neptune's faint ones; each band made of fine
  ringlets
- Lit as a layer of icy particles (single scattering with backscattering
  particles and a forward-scattering glow): dense rings bright on their lit
  face and dark from below, sparse ones glowing from behind
- The planet's shadow falls across the rings, and the rings' shadow bands
  across the planet (and its haze)

**Shapes**

![Phobos, Vesta with its giant south polar basin, Haumea and its ring, and the two-lobed comet 67P](docs/images/shapes.png)

- Relief stands on a base shape instead of a sphere: every body is
  flattened by its spin (Jupiter 6.5%, Ceres 7.5%, Earth 0.3%), and the
  seas follow the flattened surface
- Irregular small bodies: triaxial ellipsoids (Phobos 27 x 22 x 18 km, its
  long axis toward Mars; Haumea stretched by its 3.9-hour spin into an egg
  twice as long as it is thick), large-scale lumps, contact binaries of two
  lobes joined by a smooth neck (comet 67P), and giant impact basins as big
  as the body (Stickney on Phobos, Vesta's Rheasilvia with its 20 km
  central peak)
- Slopes are measured against the shape (a plain on a potato's flank is not
  a cliff); level of detail, horizon culling and the camera's ground all
  follow it; small bodies are first seen whole from space

**Orbits, the clock and eclipses**

![Jupiter from Io with a moon's shadow crossing it, the Moon's shadow on Earth in the 2027 eclipse, the red Moon of the March 2026 lunar eclipse, and Saturn from Enceladus](docs/images/orbits.png)

- A simulation clock (Planet panel > Sun & time: the date, how fast time
  runs, local time, Now) drives everything: planets on Keplerian orbits
  from JPL's mean elements, moons around their planets, and every body's
  spin from its IAU pole and prime meridian, so the sun's direction, day
  and night, the seasons and the equation of time follow from the date
  (12:00 UTC is solar noon at Greenwich, give or take a quarter hour)
- The Moon follows the main terms of its own theory (the Sun's pull,
  precessing nodes): eclipses fall on their real dates. New planets bring
  their planetary system: Earth its Moon, Mars Phobos and Deimos, Jupiter
  its Galilean moons, a moon its planet and siblings (Planet panel > System
  flies to any of them)
- The body the camera is nearest stays put while the sky wheels around it,
  so the ground never slides away; flying to another body hands over
  seamlessly
- Eclipses from the overlap of the sun's disc with every other body's:
  penumbra, umbra and annular phases, the sky darkening in a total solar
  eclipse, Earth's air bending red light onto the eclipsed Moon, moons'
  shadows crossing Jupiter. Moons and planets in the sky are lit by the
  sun alone (their night sides dark), with their own air drawn as seen
  from afar (Earth's blue limb and clouds from the Moon)
- The star's brightness, color and apparent size follow the live distance;
  chunk building is shared across all the bodies at once, every body whole
  first, then detail where it looks largest

**Real maps**

![Mars from MOLA heights and Viking colors, the Moon from LOLA heights and LROC colors, and Earth from ETOPO1 with its climate's biomes](docs/images/real_maps.png)

- Measured terrain for the real bodies: Mars from MOLA's laser altimeter,
  the Moon from LRO's LOLA, Earth's land and sea floor from ETOPO1, with
  LROC and Viking color mosaics. `python tools/fetch_maps.py` downloads them
  (~86 MB, public domain or free to redistribute) into `data/maps/`; they
  are not part of the repository
- Each map is read in its archive's own format (PDS images and labels,
  classic netCDF, images), resampled once onto the cube-sphere and cached
  (loads in milliseconds); hills and craters smaller than the map can show
  are added on top, colors keep the generated fine variation
- File > New Planet > Real maps (or Planet panel > Body > Real maps) shows
  the body as measured; its climate, biomes and rivers then work on the real
  topography (the Sahara comes out a desert), and the clock turns the real
  continents to the sun at the right hours
- Checking the generators against them: `python tools/compare_terrain.py`
  measures the same statistics of the measured and the generated surface
  (how much lies at each height, how relief grows with scale from 10 to
  2,000 km, slopes) and draws both. That found real gaps, now calibrated:
  Earth's sea floor was uniformly old and deep and its land twice too high;
  the Moon was too smooth below 100 km and had no giant basins; lava flooded
  Mars's whole northern lowlands up to a single level. After calibration
  (generated vs measured): Earth's coasts, the 90th and 98th height
  percentiles 0.09 / 0.85 / 3.3 km vs 0.08 / 0.72 / 2.6; the Moon's relief at
  10, 100 and 1,000 km 0.60 / 1.36 / 2.65 km vs 0.68 / 1.58 / 2.73; Mars's
  lowest and highest 2% -5.1 / +5.5 km vs -5.2 / +5.5

![Shaded relief, measured (left) and generated (right): Mars, the Moon, Earth](docs/images/terrain_check.png)

**Plate tectonics**

![Crust-age view after a few hundred million years: young sea floor (red) at the ridges, older floor (blue), continents (tan)](docs/images/tectonics.png)

- A plate simulation on a global cube-sphere grid (~98,000 cells, ~80 km
  apart) shapes the continents: rigid plates rotate about their own poles at a
  few cm per year, 5 million years per step
- Where plates pull apart, new ocean floor forms at mid-ocean ridges and
  deepens as it ages; where they converge, the denser crust sinks: island
  arcs, Andes-style ranges (ocean under continent) and Himalaya-style ranges
  (continent-continent collisions)
- Erosion wears mountains down; colliding continents weld into one plate and
  large plates rift apart again, opening new oceans (a Wilson cycle)
- Runs in the background while you watch (Tectonics panel: play, pause,
  step, reset, speed); the terrain rebuilds from each new state without
  popping
- Data views: plates, crust age (with magnetic-stripe bands), continental /
  oceanic crust, converging / spreading boundaries
- Deterministic: a saved scene stores the seed and simulated time and
  re-simulates to the same planet on load

**Tectonic regimes**

![The Moon, Mars, Venus (without its clouds), Io and Europa](docs/images/regimes.png)

- Plate tectonics is one of five regimes, each its own simulation on the same
  grid (the terrain, data views and Tectonics panel work with all of them):
  - **Stagnant lid** (Mars, Mercury, the Moon): one rigid shell with
    ancient relief: a crustal dichotomy, giant impact basins, volcanic
    provinces over mantle plumes, lowlands flooded by dark lava (the maria);
    volcanism fades as the planet cools
  - **Episodic resurfacing** (Venus): lava floods nearly the whole surface
    every 400-800 Myr; only high tesserae plateaus survive, and rifts,
    volcanic rises and coronae form in between
  - **Heat-pipe volcanism** (Io): constant eruptions keep the surface a few
    Myr old: sulfur plains, glowing lava lakes in calderas that open and
    fill, tall mountain blocks that rise and slump
  - **Ice shell** (Europa, Ganymede, Triton): tidal cracks become wandering
    dark ridges and spreading bands, with chaos terrain, on young, low-relief
    ice
- Relief scales with gravity; the mineral palette colors each regime's crust
  (dark maria and lava, bright highlands and ice)

**Impact craters**

![Craters on the Moon: at the surface, at the terminator from orbit, and a map of the whole Moon (maria, saturated highlands, ray craters)](docs/images/craters.png)

- Craters at every size from ~150 km down to tens of meters, evaluated per
  vertex like noise (a seamless 3D lattice per size octave), so they appear
  at any level of detail
- A realistic size mix (N(>D) ~ D^-2) and a count that follows the surface's
  age from its tectonic regime, by the lunar cratering chronology with its
  heavy bombardment: saturated lunar highlands, sparser maria, nearly blank
  Io and Europa
- Simple bowls below a transition diameter that shrinks with gravity (Moon
  ~15 km, Mars ~7 km), flat-floored complex craters with central peaks
  above; raised rims, ejecta blankets, older craters worn shallower
- Air burns up small impactors (Venus: nothing under ~3 km); rain and
  methane weather erase old craters (Earth, Titan)
- Young craters on airless bodies throw bright, lopsided ejecta rays
- Each body has its own seed (its own basins, maria and craters)

**Volcanoes**

![Mars's tallest volcano from 850 km (basal cliff, frosted summit caldera) and its relief map](docs/images/volcanoes.png)

- Built where the tectonic simulation brings magma up: shield volcanoes over
  mantle plumes and hot spots (Olympus Mons, Mauna Loa, Maat Mons) with
  summit calderas, radial lava-flow ridges and, on the giants, a basal
  cliff; stratovolcano chains above subduction zones; fields of small
  shields on volcanic plains (Venus)
- Gravity sets the ceiling: the tallest scale with 1/g (Earth ~10 km, Venus
  ~11 km, Mars ~26 km)
- Only bodies that sustain plumes build large edifices (Earth, Venus, Mars,
  Io's low shields); the Moon and Mercury get small domes, icy moons none
- Evaluated per sample on the same seamless lattice as craters (each
  volcano exists if the ground at its center is volcanic), so they appear
  at every level of detail; summits are cold enough for snow and frost

**Erosion: rivers, glaciers and dunes**

![Titan's linear dunes and Mars's transverse dunes (hillshade), and rivers, lakes and a delta on Earth](docs/images/erosion.png)

- Rivers from the climate's rain: a priority flood routes every land cell
  to the sea and fills depressions into lakes; discharge adds up downstream,
  and the largest flows become rivers as wide as their water, smoothed and
  meandering, in valleys carved down to their water level
- Deltas fan out into the sea at the biggest mouths, with distributary
  channels; where it is frozen, rivers become glaciers in wider U-shaped
  valleys
- Matched to the body: water on Earth, methane on Titan, none on dry or
  boiled-away worlds
- Wind-blown dunes where air can move sand and the ground is dry:
  transverse crests with steep slip faces across the wind (Earth's deserts,
  Mars's dark basaltic fields), long linear ridges along it in Titan's
  equatorial belt of dark organic sand; active dunes bury the craters under
  them
- Climate panel: river, glacier, lake and delta counts

**Climate and biomes**

![Natural colors, biomes, temperature and rainfall](docs/images/climate.png)

- Annual-mean climate from an energy balance (~0.5 s, recomputed in the
  background whenever the land or the physics change): sunlight by latitude
  for the star's flux, distance, orbit eccentricity and axial tilt; albedo;
  a greenhouse (infrared optical depth); heat carried by the air (more in
  thick air: Venus is nearly the same temperature everywhere, Mars's thin air
  barely evens anything out) and the oceans; cooler with height by the
  body's own lapse rate
- Each body's greenhouse is calibrated so its global mean matches the
  observed one (Venus 464 C, Earth 15 C, Mars -63 C, Titan -180 C); airless
  bodies follow from sunlight alone
- Earth's wind belts (trade winds, westerlies, polar easterlies) carry ocean
  moisture inland; it rains where air rises (the equatorial belt, mountains
  facing the wind) and stays dry where it sinks (the ~30 degree desert belts)
  and in mountains' rain shadows
- Biomes from temperature and rainfall (a Whittaker diagram): ice, tundra,
  taiga, temperate forest, grassland, desert, savanna, tropical rainforest;
  snow and tree lines follow temperature, so they fall with latitude and rise
  with warmth
- Climate panel: global temperature, humidity, axial tilt, and physical
  what-ifs (move the body closer to its star, change its albedo, thicken its
  greenhouse) with sunlight, equilibrium temperature and lapse rate shown;
  planet-wide min / mean / max, biome shares; temperature, rainfall and biome
  views

**Clouds**

![Earth from space with its weather, and a cloud deck seen from below over the ocean](docs/images/clouds.png)

- A weather cloud layer where the climate rains: the equatorial belt and the
  storm tracks cloudy, the subtropical deserts clear, the planet averaging
  its observed cover (Earth ~55-65%); domain-warped fractal noise per pixel
  shapes the clouds from continent-sized systems down to tens of meters,
  drifting slowly with the winds
- Drawn inside the atmosphere pass: air in front of the clouds, the clouds,
  and the sky behind their gaps; bright tops from above, dark bases from
  below (two-stream light through thick cloud), reddened at sunset, glowing
  at thin edges toward the sun
- Clouds shadow the ground, and overcast skies dim the ambient light
- Per body: Earth's water clouds, Mars's rare thin water-ice clouds, Titan's
  methane clouds; Venus's sulfuric acid deck hides its surface completely

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
- Aerosols near the ground (dust, haze) or in a deck at altitude (Venus's
  sulfuric acid clouds at ~57 km, with clearer air below)
- Colored aerosols (per-channel scattering and absorption): Mars's
  butterscotch dust, Titan's orange haze
- Optically thick atmospheres (Venus's clouds, Titan's haze) pass diffuse
  daylight down to the ground (a two-stream estimate baked into a third lookup
  table), so their surfaces are dimly lit instead of black
- Ambient lighting and reflections baked from the atmosphere at the camera

**Rendering**

- Physically based shading (Cook-Torrance GGX, metallic-roughness as in glTF 2.0)
  with base color, metallic-roughness, normal, occlusion and emissive maps
- Directional, point and spot lights
- Cascaded shadow maps that follow the camera (stabilized against shimmering),
  plus spot light shadows, filtered with hardware PCF
- Procedural sky for scenes without a planet; ambient light and reflections baked from it
  (split-sum image-based lighting: irradiance, prefiltered specular, BRDF LUT)
- HDR pipeline: bloom, exposure with eye adaptation (dim scenes such as
  Titan's surface brighten, glaring ones darken, over half a second; views
  of a planet from space keep their exposure), ACES / Reinhard tone mapping,
  gamma correction, FXAA
- Close-up terrain detail: per-pixel fractal noise varies the ground's
  brightness and hue and adds micro-relief bump shading, fading out with
  distance so it never shimmers; bare worlds' regolith is mottled at every
  scale
- Batched drawing: one shared geometry buffer, one multi-draw-indirect call per
  material, frustum culling of every object and shadow caster
- Planet-scale precision: 64-bit world positions, camera-relative rendering and
  a reversed-Z infinite depth buffer
- OBJ and glTF / GLB model loading, including glTF materials and embedded textures
- Shader `#include`s and hot reload (edit a `.glsl` file while the engine runs)

**Editor**

![The scene editor](docs/images/editor.png)

- Docked layout around the 3D viewport (drag panels to rearrange; View menu to
  show / hide them or reset the layout; Help > Controls lists every shortcut)
- **Planet panel**: jump to the surface / low orbit / whole-planet view, fly
  speed, local **time of day and season** (moves the sun; optional day cycle),
  terrain presets and sliders (the planet rebuilds when you release), seed,
  atmosphere density and haze, and **data views**: elevation with contour
  lines, slope, moisture, latitude, level of detail
- Viewport HUD: altitude above ground and sea level (depth below a giant's
  cloud tops), the air's pressure and temperature, speed, latitude /
  longitude, local time, frame rate
- Hierarchy with drag-and-drop parenting, inspector for every component
- Click-to-select in the viewport, move / rotate / scale gizmos with snapping
- Undo / redo (including planet and sun edits), scene files (JSON) with native
  open / save dialogs, model import
- Play / Stop: run the simulation, then restore the scene exactly
- Render settings, material editing, stats and a frame profiler (CPU + GPU
  timings) in the bottom panel

**Engine**

- Entity-component-system with parent / child transforms (quaternion rotations)
- Background job system: worker threads, results handed back to the main thread
  under a per-frame time budget
- Fixed-timestep simulation, uniform buffers for per-frame data, OpenGL debug
  output routed to the log
- The simulations' heaviest steps on the GPU (compute shaders): the
  climate's radiative balance (Newton steps, each a conjugate-gradient solve
  inside one workgroup), the rivers' basin filling and discharge, and the
  plates' crust tracking. Simulation threads hand them to the main thread,
  which runs a few milliseconds of them per frame, so a solve spreads over
  frames instead of stalling one or holding the Python lock: the climate
  solve goes from 1.6 s to 0.2 s (30 s to 0.4 s at 4x the cells). Each keeps
  its NumPy twin, the reference it is tested against and the fallback
  without compute shaders
- A disk cache (`data/cache/`) of tectonic states step by step, climates and
  terrain chunks, keyed by their inputs and the generators' source code:
  reopening a planet loads instead of simulating (startup 1.9 s to 0.7 s,
  chunks streaming in ~70% faster); oldest entries pruned past 2 GB
- 598 unit tests (the GPU kernels' run where a GPU is available)

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
| Hold right mouse button | Look around; **W A S D** move, **Space** up, **Shift** down. On a planet, W A S D move along the ground (around the planet when in orbit) and Space / Shift move straight away from / toward it; speed grows with altitude |
| Left click | Select an object |
| **Q / W / E / R** | Select / move / rotate / scale tool |
| **F** | Focus the camera on the selection (a planet: fit the whole planet in view) |
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
| `core/` | Window, input, events, timer, logging, profiler, job system, disk cache |
| `ecs/` | Entity registry and components |
| `systems/` | Transform, camera controller, rotator, orbits and clock, tectonics, climate, planet streaming and render systems |
| `planet/` | Body profiles, phases of volatiles, noise, cube-sphere mapping and simulation grid, plate tectonics and other tectonic regimes, impact craters, volcanoes, rivers and dunes, climate, terrain, chunk building, level of detail, body shapes, orbits and spin, solar time, real maps |
| `graphics/` | Renderer, shaders, textures, meshes, shadows, IBL, atmosphere, clouds, rings, eclipses, bloom, GPU compute and the simulations' kernels |
| `editor/` | Scene editor, hierarchy / inspector / planet panels, picking, undo history |
| `scene/` | Scene container and scene file (de)serialization |
| `resources/` | Handle-based resource managers |
| `math3d/` | Transforms, camera, matrix helpers |
| `ui/` | ImGui integration, docked layout and panel registry, render / stats / profiler panels |
| `assets/` | Shaders, textures, scenes, body profiles (`bodies/`) |
| `tools/` | Asset generator, micro-benchmarks, real map downloader, terrain checker |
| `data/` | Downloaded maps and caches (not in the repository) |
| `tests/` | pytest suite (`tests/fixtures/` holds the OBJ / glTF test models) |

## Development

Run the tests (the GPU kernel tests skip without a GPU):

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

Download the real maps (~86 MB; `--list` shows what and from where), then
compare the generators with them:

```bash
python tools/fetch_maps.py
```

```bash
python tools/compare_terrain.py --image terrain_check.png
```

Switches (environment variables): `ENGINE_NO_GPU_SIMULATION` keeps every
simulation on the CPU, `ENGINE_NO_DISK_CACHE` disables the disk cache.

Shaders live in `assets/shaders/` and reload automatically when saved. The
Profiler window (bottom of the screen) shows CPU and GPU time per render pass;
untick VSync there to see the real cost of a frame.

## License

Released into the public domain under the [Unlicense](UNLICENSE).
Dependencies (installed through `requirements.txt`) keep their own licenses.
