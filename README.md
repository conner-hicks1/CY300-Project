# CY300 Spaceflight

A rocketry and spaceflight game in the making, written in Python on top of
OpenGL 4.5 as a CY300 course project. Its engine renders with physically
based materials, cascaded shadows and image-based lighting, and streams the
bodies of the solar system as they were measured (real elevation and color
maps) under physically based atmospheres, on real orbits with a real clock.
You fly a two-stage rocket off any body: real-scale parts, gravity, thrust,
drag in the real air, staging, orbits followed exactly under time warp,
patched conics to the Moon and back, a map of your orbit, and a capsule that
parachutes home; or build your own rocket in the part editor, on the pad.

![The procedural planet, seen from the surface](docs/images/scene.png)

## Features

**Planets**

- Cube-sphere planets at full size: measured heights and colors where the
  body has been mapped, seeded noise (continents, ridged mountain ranges,
  hills) for the rest; biomes, snow and sea ice from an analytic climate
- Quadtree level of detail per cube face with horizon culling; chunks are built
  on worker threads and streamed in coarse-to-fine without holes or cracks
- Planet-aware camera: radial "up", altitude-scaled flying speed, stays above
  the terrain

**Bodies of the solar system**

![Mars from orbit, the Moon's surface, Io, Venus's surface under its clouds, Titan's surface and Europa](docs/images/bodies.png)

- Body profiles (`assets/bodies/*.json`, real data from NASA's fact sheets) for
  Mercury, Venus, Earth, the Moon, Mars, Phobos, Deimos, Vesta, Ceres,
  comet 67P, Jupiter, Io, Europa, Ganymede, Callisto, Saturn, Enceladus,
  Titan, Uranus, Neptune, Triton, Pluto and Haumea:
  size, mass, rotation, tilt, orbit, star, temperature, surface liquid and
  colors, terrain relief, atmosphere (pressure, gases, aerosols), maps
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
- Travel (pause menu) starts over at any of them
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

- A simulation clock (time warp from real time to a million times faster)
  drives everything: planets on Keplerian orbits
  from JPL's mean elements, moons around their planets, and every body's
  spin from its IAU pole and prime meridian, so the sun's direction, day
  and night, the seasons and the equation of time follow from the date
  (12:00 UTC is solar noon at Greenwich, give or take a quarter hour; on
  Venus and Uranus, which spin backwards, the sun rises in the west and
  local time still runs forward)
- The Moon follows the main terms of its own theory (the Sun's pull,
  precessing nodes): eclipses fall on their real dates. New planets bring
  their planetary system: Earth its Moon, Mars Phobos and Deimos, Jupiter
  its Galilean moons, a moon its planet and siblings
- The body the camera is nearest stays put while the sky wheels around it,
  so the ground never slides away; flying to another body hands over
  seamlessly
- Eclipses from the overlap of the sun's disc with every other body's:
  penumbra, umbra and annular phases, the sky darkening in a total solar
  eclipse, Earth's air bending red light onto the eclipsed Moon, moons'
  shadows crossing Jupiter. Moons and planets in the sky have their own
  air drawn as seen from afar (Earth's blue limb and clouds from the Moon)
- The star's brightness, color and apparent size follow the live distance;
  chunk building is shared across all the bodies at once, every body whole
  first, then detail where it looks largest

**The night sky and space**

![The night sky over Earth with Cassiopeia in the Milky Way, the Milky Way toward Sagittarius from space, the Moon's night side in earthshine, Io under a full Jupiter, Earth's night side from orbit, and Halley's Comet in March 1986](docs/images/space.png)

- The real stars: all 9,110 of the Yale Bright Star Catalog (to magnitude
  6.5, everything the eye sees), at their positions, brightness and colors
  (from their B - V color index), so the constellations rise and set with
  the clock; they twinkle through air, more near the horizon, and the air
  dims them and the day sky drowns them. `python tools/fetch_stars.py`
  downloads the catalog (574 KB); without it the sky gets random stars
  with the same counts
- The Milky Way: its band brightest toward Sagittarius, the Cygnus and
  Carina star clouds, the yellowish bulge below the dust of the Great
  Rift, the Coalsack, the Magellanic Clouds and the Andromeda galaxy, in
  galactic coordinates (baked once into a map)
- The planets and moons too far to show a disc are points of light that
  move against the stars, as bright as they really are (their size,
  albedo, distance and phase: Jupiter at opposition magnitude -2.7, the
  full Moon -12.4)
- Light between bodies (planetshine): a sunlit neighbor lights a night
  side by its size, albedo and phase. The Moon's night side in earthshine,
  Io under a full Jupiter, moonlight on Earth
- All of it at its physical brightness (a star ~1e-10 of sunlight), seen
  the way the eye sees it: auto exposure dark-adapts as the light fades,
  up to a million times, so in the dark the stars and the Milky Way come
  out, while a sunlit planet in view outshines them as in any photograph;
  night scenes stay dim, and faint light turns grey-blue (rods instead of
  cones). (The HDR image is 32-bit float for it: starlight is below half
  floats' range)
- Comets come alive near the Sun: a coma and two tails, from the dust
  activity measured for each (A f rho, growing ~r^-3.5 inside ~4 AU): the
  broad dust tail curving back along the orbit, brightest seen toward the
  Sun, and the straight blue ion tail. 67P (next perihelion 2028) stays
  faint, as Rosetta saw it; Halley's Comet (new: its 1986 and 2061
  returns) grows tails millions of km long

**Real maps**

![Mars from MOLA heights and Viking colors, the Moon from LOLA heights and LROC colors, and Earth from ETOPO1](docs/images/real_maps.png)

- Every body that has been mapped is shown as measured, by default:

  | Body | Heights | Colors |
  | --- | --- | --- |
  | Mercury | MESSENGER stereo DEM, 665 m | MESSENGER MDIS basemap (its enhanced color toned down to true greys) |
  | Venus | Magellan radar altimetry, 4.6 km | |
  | Earth | ETOPO1, land and sea floor (1.8 km up close) | Blue Marble Next Generation, July |
  | Moon | LRO LOLA (0.5 km up close) | LROC WAC mosaic |
  | Mars | MGS MOLA (0.9 km up close) | Viking color mosaic |
  | Ceres, Vesta | Dawn HAMO stereo DTMs (~140 m, ~90 m) | |
  | Io, Europa, Ganymede, Callisto | | Galileo SSI and Voyager mosaics |
  | Titan, Enceladus | | Cassini ISS mosaics |
  | Triton | | Voyager 2 mosaic |
  | Pluto | New Horizons stereo DEM, 300 m | New Horizons mosaic |

  `python tools/fetch_maps.py` downloads them (~4.4 GB, from NASA, NOAA and
  USGS Astrogeology; public domain or free to redistribute) into
  `data/maps/`; they are not part of the repository. Without them a body
  falls back to its generated surface
- Each map is read in its archive's own format (PDS images and labels,
  classic netCDF, GeoTIFFs with their scale, offset and projection, images),
  resampled once onto the cube-sphere and cached (loads in milliseconds);
  hills and craters smaller than the map can show are added on top. Areas
  never imaged (Pluto's and Triton's far sides, Venus's gaps) are filled
  smoothly from around them. Heights measured from a sphere carry the
  body's whole shape (Ceres's flattening, Vesta's giant basins) and are
  taken as that
- Sharper data up close: the full-resolution GeoTIFFs, Mars at 64
  pixels/deg (four MOLA tiles), the Moon at 64 pixels/deg, Earth every
  arc-minute (`python tools/fetch_maps.py --sharp`, ~1.5 GB for those three).
  They are memory-mapped, never loaded: only the samples the detailed chunks
  near the camera need are read from disk, blended in over a factor of two
  in detail so nothing pops
- Color mosaics too large for a texture (up to 23,040 x 11,520) are averaged
  down once to 4096 x 2048 images in `data/cache/`
- The clock turns the real continents to the sun at the right hours

**Impact craters**

![Craters on the Moon: at the surface, at the terminator from orbit, and a map of the whole Moon (maria, saturated highlands, ray craters)](docs/images/craters.png)

- Craters at every size from ~150 km down to tens of meters, evaluated per
  vertex like noise (a seamless 3D lattice per size octave), so they appear
  at any level of detail
- A realistic size mix (N(>D) ~ D^-2) and a count that follows the surface's
  age (each body's own), by the lunar cratering chronology with its
  heavy bombardment: saturated lunar highlands, sparser maria, nearly blank
  Io and Europa
- Simple bowls below a transition diameter that shrinks with gravity (Moon
  ~15 km, Mars ~7 km), flat-floored complex craters with central peaks
  above; raised rims, ejecta blankets, older craters worn shallower
- Air burns up small impactors (Venus: nothing under ~3 km); rain and
  methane weather erase old craters (Earth, Titan)
- Young craters on airless bodies throw bright, lopsided ejecta rays
- Each body has its own seed (its own basins, maria and craters)

**Erosion: dunes and gullies**

- Wind-blown dunes where air can move sand and the ground is dry:
  transverse crests with steep slip faces across the wind (Earth's deserts,
  Mars's dark basaltic fields), long linear ridges along it in Titan's
  equatorial belt of dark organic sand; active dunes bury the craters under
  them
- Carved slopes: gullies cut straight down every slope, joining into
  valleys, the ridges between left sharp (an "erosion filter": stripes along
  the slope at each scale from 2.4 km to 130 m, each finer octave following
  the slope the coarser ones left, so small gullies run into big ones);
  where rain runs off (Earth, Titan), half as much on Mars (its ancient
  valley networks), a little on Venus, none on airless worlds; over a
  measured map only what it is too coarse to show

**Close to the ground**

![A Black Forest clearing with conifers, beeches and grass, the forest from 60 m, boulders on an Alpine scree slope, Mars's sand at Gale crater, the Moon's regolith, and crater shadows near the lunar terminator](docs/images/ground.png)

- Rocks, boulders, trees and grass, placed per cell of the cube-sphere grid
  around the camera from the same terrain and climate the ground is shaded
  by: conifers where it is cold, beeches and oaks in temperate forest, tall
  dark rainforest trees, flat-topped acacias on the savanna (life-bearing
  worlds only); grass tufts, green where wet and straw where dry; boulders
  everywhere bare, most of all in scree under cliffs, many small and few
  large, settled into the slope. Placement is deterministic per cell
- Drawn instanced: each layer one buffer of instances sorted into blocks,
  one multi-draw per layer for the blocks in view (trees switch to cheap far
  shapes past 160 m), lit and shadowed like everything else; they grow in
  over the last part of their reach, and sway in the wind. Built on the
  workers after the terrain (it streams first). Off at Low graphics
  quality
- Ground texture all the way down: below the terrain's finest vertices each
  material has its own detail to ~5 cm: rock cracked and blocky with strata
  on cliffs, sand smooth with ripples across the wind where there is air,
  regolith and soil grained, plants patchy, snow faintly wind-carved; fading
  as it gets smaller than a pixel (its relief sooner, so bump lighting never
  aliases)
- Exact up close: the noise is measured from a planet-fixed grid point near
  the camera (32-bit positions from a planet's center blur below ~0.5 m) and
  is gradient noise with a strong hash (value noise lights its lattice
  planes as lines); chunk edges are stitched to the vertices a coarser
  neighbor shares, so there are no cracks
- Mountains' shadows: beyond the cascades (which now reach a few hundred
  meters near the ground, for trees and rocks), one more shadow map covers
  the land in view, from ~20 km around on the ground to ~1,500 km from orbit:
  crater floors in shadow near the Moon's terminator, valleys in shade at
  sunrise. Redrawn only when the view or the sun has moved

**Climate and biomes**

- An analytic annual-mean climate, no simulation: each body's measured mean
  temperature, spread from equator to poles by how much its air evens heat
  out (~45 C on Earth, ~60 C with no air, ~10 C under Venus's or Titan's
  thick air), colder with height by the body's own lapse rate
- Rain in Earth's latitude belts: the wet tropics and storm tracks, the dry
  subtropics and poles, patchy with noise
- Biomes from temperature and rainfall (a Whittaker diagram): ice, tundra,
  taiga, temperate forest, grassland, desert, savanna, tropical rainforest;
  snow and tree lines follow temperature, so they fall with latitude and rise
  with warmth

**Clouds**

![Earth from space with its weather, and a cloud deck seen from below over the ocean](docs/images/clouds.png)

- A weather cloud layer in the rain belts: the equatorial belt and the
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
  plus spot light shadows, filtered with hardware PCF; a terrain shadow map
  far and wide on planets
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

**Vessels and flight**

- Parts (`assets/parts/parts.json`) and vessels built from them
  (`assets/vessels/`), at real scale in SI units: a crew capsule with
  reaction wheels, a 30 m main parachute, kerosene / oxygen tanks, a booster
  engine (2.2 MN, Isp 285 s at sea level and 315 s in vacuum) and a vacuum
  engine (250 kN, 348 s), stage separators. Pathfinder, the starting rocket:
  160 t, 37 m, ~10.3 km/s of delta-v; stages: booster, upper stage, capsule
  separation, parachute. It makes low Earth orbit with ~650 m/s to spare
  (the tests fly it there with simple guidance)
- Flight physics (`vessel/physics.py`) in the body's non-rotating frame:
  gravity, thrust whose push falls with the outside pressure (Isp), drag
  through the air (density from the body's air column, the air turning with
  the body), propellant burned per engine from its own section's tanks;
  steering by reaction wheels and the engines' gimbals, against the stack's
  moments of inertia, with stability assist holding the attitude
- The ground: launch from a pad turning with the body, land below the
  bottom part's crash speed, or be destroyed above it
- Coasting above the air, the orbit is followed exactly (universal-variable
  Kepler propagation): time warp to 1,000,000x. Under thrust, in the air or
  near the ground physics steps every 0.02 s and warp stops at 5x; a
  falling arc goes on rails in pieces short enough not to pass through the
  ground
- Staging (**Space**): the first stage lights the booster; the next fires
  the separator, drops everything below it and lights the upper engine; then
  the capsule separates, and the last arms the parachute. Spent stages fall
  away as debris (gravity and drag) until they hit the ground or are left
  25 km behind; the vessel goes on from its new bottom without a jump
- Parachutes open once the air is thick enough and slow enough (below
  250 m/s), over three seconds; the capsule then hangs nose-up under the
  canopy and lands at ~9 m/s at sea level (its limit is 10)
- Patched conics (`vessel/soi.py`): the vessel feels one body at a time,
  the one whose sphere of influence it is in (the Moon's ~66,000 km, Earth's
  ~925,000 km; the Sun beyond). Crossing a boundary hands it over, its state
  re-expressed about the new body; time warp slows ahead of a boundary so
  the handover is never skipped. Earth to the Moon and round it (a
  gravity-assisted swing back out) works on rails at full warp
- Map view (**M**): the camera pulls back round the body; the vessel's path
  ahead is drawn over the view, patch by patch (its orbit, or its arc to the
  ground, or out of the sphere; an encounter's flyby round the moon it
  reaches, with its closest approach), Ap / Pe with heights, the moons'
  orbits; lines dim where the body hides them
- Attitude ball (navball): sky and ground about the local horizon, the
  compass points, prograde and retrograde markers, heading and pitch.
  Stability assist (**T**) holds the attitude, or (**F**) points prograde
  or retrograde
- Chase camera round the vessel (right mouse, scroll); **V** switches to the
  free camera
- Vessel HUD: situation (landed, flying, sub-orbital, orbiting, flyby,
  escaping) and the body, throttle, propellant, stage, thrust to weight,
  delta-v left, surface / orbital / vertical speed, apoapsis and periapsis,
  SAS mode, parachute; notices as things happen (staging, a new sphere of
  influence, the parachute opening, landing); Revert to launch in the pause
  menu
- Giants and small bodies start in orbit; everywhere else on the ground

**Part editor**

- Build on the launch pad (pause menu > Build vessel): the rocket stands
  there and changes as you edit; right mouse turns the view
- Parts on the left by kind (command, tanks, engines, separators,
  parachutes, structure: 19 at 1.25 m and 2.5 m, with adapters, a nose
  cone, a radial separator and a 1.25 m booster engine; tooltips with mass,
  propellant, thrust and Isp); a click adds one below the selected part, or
  into the selected radial group
- The stack on the right: select a part (it lights up) to move it, remove
  it or change its stage; automatic staging (from the bottom up, each
  section's engines with the side boosters hung on it, the boosters'
  separators the stage after, parachutes last)
- Side boosters and other radial parts: on any core part attach a radial
  group of 2, 3, 4 or 6 copies round it, with a radial separator, then add
  parts into it (a nose cone, tanks, an engine); set its height on the part.
  The copies are counted in mass, thrust, drag and inertia, each burns its
  own tanks, they light with the core and drop off (as debris) when their
  separator fires; their thrust stays on the axis. Pathfinder Heavy (built
  in) is Pathfinder with two: 204 t at liftoff, ~10.8 km/s
- The numbers per stage: delta-v in vacuum and at the launch body's surface,
  thrust to weight there, burn time (each stage from its firing to the next
  burnout: boosters and core burning together, then the core alone); what stops it flying (no command part,
  an unstaged engine) and what would fly badly (an engine with no tank in
  its section, a first stage that cannot lift off)
- Save and open designs (yours in `data/vessels/`, the built-in ones in
  `assets/vessels/`); Launch puts it on the pad, Cancel (or Esc) puts back
  the one before

**Game shell**

- Flight HUD over the view: the date and time warp at the top; at the
  bottom the body you are nearest, altitude above ground and sea level
  (depth below a giant's cloud tops), speed, the air's pressure and
  temperature outside, latitude / longitude and local time
- Time warp in steps from real time to 1,000,000x (**,** and **.**; **/**
  back to real time)
- Pause menu (**Esc**, stops the clock): Revert to launch, Build vessel, Travel to any
  body, Camera (follow the vessel or fly free; surface, low orbit, the whole
  body in view), Graphics quality (Low / Medium / High, Medium by default:
  the scene's resolution, 70% / 85% / 100% scaled up at the end, since the
  scene's cost is nearly all per pixel; atmosphere samples; rocks, trees and
  grass; shadow reach and mountain shadows; bloom), Controls, Quit
- Developer tools (**F3**, or `--dev`): stats, every render setting, a frame
  profiler (CPU + GPU timings) and cProfile capture, shader hot reload

**Engine**

- Entity-component-system with parent / child transforms (quaternion rotations)
- Background job system: worker threads, results handed back to the main thread
  under a per-frame time budget
- Fixed-step vessel physics in simulated time (time warp included), uniform
  buffers for per-frame data, OpenGL debug output routed to the log
- A disk cache (`data/cache/`) of terrain chunks, keyed by their inputs and
  the generators' source code: chunks stream in ~70% faster on a second
  visit; oldest entries pruned past 2 GB
- 574 unit tests

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

This starts on Earth, the camera above a valley in the mountains. The
planet streams in over the first few seconds.

| Option | Meaning |
| --- | --- |
| `--body ID` | Start at another body (`moon`, `mars`, `titan`... any file in `assets/bodies/`) |
| `--dev` | Start with the developer tools open (**F3** toggles) |
| `--hide-ui` | Start with the HUD hidden (**F2** toggles) |
| `--no-vsync` | Uncapped frame rate (for measuring performance) |
| `--cprofile N` | Record a Python profile of the first N frames into `logs/profiles/` |
| `--exit-after S` | Quit after S seconds (smoke tests) |
| `--screenshot PATH` | With `--exit-after`, save the last frame as an image |

## Controls

| Input | Action |
| --- | --- |
| **Shift** / **Ctrl** | Throttle up / down |
| **Z** / **X** | Full throttle / cut |
| **Space** | Next stage (the first lights the engines) |
| **W** / **S** | Pitch the nose forward / back (forward is east at launch: the way to orbit) |
| **A** / **D** | Yaw left / right |
| **Q** / **E** | Roll |
| **T** | Stability assist on / off |
| **F** | Stability assist mode: hold attitude, prograde, retrograde |
| **M** | Map view (right mouse turns it, scroll zooms) |
| Right mouse + move, scroll | Swing the camera round the vessel; closer / farther |
| **V** | Free camera: hold right mouse, **W A S D** move, **Space** up, **Shift** down (along the ground, around the planet from orbit; speed grows with altitude) |
| **.** / **,** | Time warp faster / slower |
| **/** | Back to real time |
| **Esc** | Pause menu (travel, camera, graphics, controls, quit) |
| **F2** | Show / hide the HUD |
| **F3** | Developer tools |
| **F5** | Reload all shaders (with the developer tools open) |

## Project Layout

| Path | Contents |
| --- | --- |
| `main.py`, `application.py` | Entry point, main loop, building a body's scene |
| `core/` | Window, input, events, timer, logging, profiler, job system, disk cache |
| `ecs/` | Entity registry and components |
| `vessel/` | Parts and vessel designs (and drafts being edited, saved designs), staging, propellant and parachutes, flight physics, Kepler orbits, patched conics |
| `systems/` | Transform, camera controller, orbits and clock, vessels, planet streaming and render systems |
| `planet/` | Body profiles, phases of volatiles, noise, cube-sphere mapping, impact craters, dunes, surface climate, terrain, chunk building, level of detail, body shapes, orbits and spin, solar time, real maps |
| `graphics/` | Renderer, shaders, textures, meshes, shadows, IBL, atmosphere, clouds, rings, eclipses, bloom |
| `game/` | Flight HUD and readout, attitude ball, map view, part editor, piloting keys, chase camera, pause menu, time warp, graphics quality presets |
| `scene/` | Scene container and (de)serialization (for save games) |
| `resources/` | Handle-based resource managers |
| `math3d/` | Transforms, camera, matrix helpers |
| `ui/` | ImGui integration, developer tool windows (render / stats / profiler) |
| `assets/` | Shaders, textures, body profiles (`bodies/`), parts (`parts/`) and vessels (`vessels/`) |
| `tools/` | Asset generator, micro-benchmarks, real map and star catalog downloaders |
| `data/` | Downloaded maps and caches (not in the repository) |
| `tests/` | pytest suite (`tests/fixtures/` holds the OBJ / glTF test models) |

## Development

Run the tests (those on the real maps skip until they are downloaded):

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

Download the real maps (~4.4 GB; `--list` shows what and from where; name
bodies to fetch only theirs, e.g. `mars moon`):

```bash
python tools/fetch_maps.py
```

The four-times-sharper Mars, Moon and Earth maps for flying close (~1.5 GB,
memory-mapped):

```bash
python tools/fetch_maps.py --sharp
```

Download the star catalog (574 KB) for the real night sky:

```bash
python tools/fetch_stars.py
```

Switch (environment variable): `ENGINE_NO_DISK_CACHE` disables the disk
cache.

Shaders live in `assets/shaders/` and reload automatically when saved while
the developer tools are open (**F3**). Their Profiler window (Developer menu)
shows CPU and GPU time per render pass; untick VSync there to see the real
cost of a frame.

## License

Released into the public domain under the [Unlicense](UNLICENSE).
Dependencies (installed through `requirements.txt`) keep their own licenses.
