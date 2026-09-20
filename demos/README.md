# Visual Bend experiments

All four demos implement their computation in Bend. The official Bend examples
and guide informed the runtime setup; no JavaScript physics or Life rules are used.
The native demos produce pixels or recorded simulation states. The Rubik demo
runs Bend's JavaScript output live in the browser and displays vector geometry.

## First run

Git, Node.js 24+, and Clang are needed. The scripts download the pinned Bend
compiler on first use if necessary. For CUDA, Linux/NVIDIA, Clang 19+, a working
NVIDIA driver and NVRTC are required:

```sh
npm run setup:gpu  # Only needed if CUDA NVRTC is not already available
npm run raytracer -- --gpu
npm run life -- --gpu
npm run cubes -- --gpu --digits 3
npm run playground
```

Visit http://localhost:3000. The viewer can submit new Bend jobs, download the
RGB output as PPM, and play/pause/step/scrub Life's recorded generations. One compute
job runs at a time. Changing the compute device or settings applies on the next
Render/Generate action. `PORT=3001 npm run playground` changes the server port.
The viewer loads each demo only when opened and refreshes only the result of
the current job. Replay downloads are compressed; revisiting a tab reuses its
loaded replay and preserves your settings. Long jobs show elapsed time and
can be stopped with Cancel computation before trying other settings.

The cubes tab shows flat 2D blocks and has play/pause, step, reset, collision scrubbing and playback
speed (0.01× through 2×). The timeline is simulation time in seconds. Link
directly to it with `#cubes` after the viewer URL.

The command-line programs work without the viewer. CUDA runs explicitly require
the GPU and reserve 1 GiB for Bend's runtime. CPU runs use one thread by default.
Artifacts are saved under `output/`; each run replaces that demo's last output.

## Ray tracer

```sh
npm run raytracer -- --gpu --width 1024 --height 640 --bounces 4
npm run raytracer -- --threads 8 --width 512 --height 320
```

The original `mirrors` scene defaults to 512 × 320, three reflection levels,
and four samples per pixel.
`--bounces 0` shows the sky only. The scene contains four colored/reflective
spheres and a checkerboard floor, with analytic sphere/plane intersections,
directional lighting, hard shadows, recursive reflections and gamma encoding.
A ray stops as soon as it misses the scene. Reflection depth is an upper bound;
raising it does not force escaped rays through unnecessary reflection work.

`raytracer/ray.bend` builds a balanced binary tree of pixel tasks. Padded leaves
outside the image return black. The native program returns packed RGB values;
the Node launcher unpacks those colors into `output/raytracer.ppm`, using the
plain-text P3 format from
[Ray Tracing in One Weekend](https://raytracing.github.io/books/RayTracingInOneWeekend.html#outputanimage/theppmimageformat):

```text
P3
width height
255
red green blue
...
```

Each channel is an integer from 0 to 255. Pixels are ordered left to right,
then top to bottom; padded leaves are omitted. Bend already applies gamma
encoding. The browser parses the RGB triplets, adds an opaque alpha channel,
and draws them using `ImageData` and `putImageData`. It does no ray tracing or
additional gamma correction. The server gzip-compresses PPM transfers, but the
saved/downloaded file remains plain text. No PNG encoder is used.

`output/raytracer.json` stores settings and timings only.

### Book-inspired scenes

```sh
npm run raytracer -- --gpu --scene materials --samples 16 --bounces 12
npm run raytracer -- --gpu --scene weekend --width 256 --height 160 --samples 4
npm run raytracer -- --threads 8 --scene weekend --samples 32 --seed 42
```

The scene selector keeps the original renderer and adds two Monte Carlo path
tracing scenes implemented in `raytracer/path.bend`:

- `materials`: three large spheres demonstrating Lambertian matte, polished
  metal, and glass on a neutral ground sphere.
- `weekend`: a seeded field inspired by the book's
  [final render](https://raytracing.github.io/books/RayTracingInOneWeekend.html#wherenext?/afinalrender),
  with hundreds of small matte, rough-metal, and glass spheres plus the three
  large spheres. Seed 42 produces 471 small spheres; clearances keep them out
  of the large spheres. This is a new deterministic arrangement, not an exact
  reproduction of the book's random image.

Bend computes random camera rays, cosine-weighted diffuse scattering, rough
metal reflection, Snell refraction, Schlick reflectance, total internal
reflection, sky illumination, and gamma encoding after averaging samples.
The camera uses a thin lens for depth of field. A spatial binary hierarchy
rejects groups of small spheres using axis-aligned bounds. CPU and CUDA use
the same Bend source; pixel subtrees are independent parallel tasks.

New scenes default to 16 samples per pixel, 12 bounces, and seed 42. More
samples reduce noise; more bounces extend possible light paths. Zero bounces
returns black in the path tracer. Changing settings reuses the compiled binary.
All scenes produce the same P3 RGB format, and the browser only displays it.

A local CUDA preview at 256 × 160, 4 samples, and 8 bounces took 685 ms end to
end. At 512 × 320, 32 samples, and 12 bounces it took 16.4 s. These are single
measurements, not a promise of real-time rendering. Use small previews while
experimenting; the 500-sample book settings involve much more work.

Tests compare hierarchy intersections with exhaustive sphere searches, check
material scattering and glass refraction, verify seeds, and round-trip Bend
colors through PPM into canvas RGBA without altering channel values.

## Conway's Game of Life

```sh
npm run life -- --gpu --size 256 --steps 240 --pattern random --seed 123
npm run life -- --size 64 --steps 120 --pattern glider
npm run life -- --size 64 --steps 20 --pattern blinker
```

Defaults: 128 × 128, 120 generations, random soup, seed 42. `--steps 0` records
only the initial state. Seeds make random soup reproducible; glider and blinker
are deterministic and ignore the seed.

The grid wraps around both axes (a torus). Size must be a power of two and at
least 32 because each row is packed into 32-cell U32 words. A bit-plane neighbor
counter evaluates Conway's B3/S23 rule across all 32 cells at once. A persistent
tree lets tasks read the same old generation while building the next generation
in parallel. The replay engine in `life/sparse.bend` represents empty subtrees
compactly and tracks live-word bounds to skip regions that cannot change.
The same B3/S23 rules apply to all patterns; there is no glider shortcut.
Generations depend on one another and run sequentially, in batches of up to
32 per GPU call. All intermediate generations are retained.

`output/life.json` includes generation zero plus every computed generation.
With `encoding: "sparse-words"`, each frame contains flat word-index/value pairs;
omitted words are zero. The viewer also accepts older dense-word replays.
Replay speed only controls playback; it does not alter the simulation.
Sparse patterns now need much less storage and computation. Dense random
patterns still require updating most of the grid.

## Pi cubes

```sh
npm run cubes -- --gpu --digits 3
npm run cubes -- --digits 10
```

The counter reuses `accelerated.bend`. `cubes/replay.bend` evaluates selected
collision transformations independently on CPU/GPU, then recovers contact
positions from the conserved quantities `|x*w-y*v| = 1` and
`t*t*x*v+y*w = time-3`, where `t=1/sqrt(heavy mass)`.

This uses the requested mass ratio; it does not animate a smaller substitute
experiment. Up to 1,024 block/wall pairs are recorded. Below that, every impact
is present; above it, spaced pairs include the first and last collisions. The
viewer uses physical time, with constant velocity between impacts and two
seconds of outgoing motion after the final impact. For small runs, every
collision event determines the trajectory. For large runs, Bend independently
finds the last collision at each of 1,024 uniformly spaced timestamps using
extended-precision event-time comparisons, then evaluates the free-flight
position at that time. The viewer interpolates those time samples, so very
fast individual bounces may be unresolved, but omitted impacts no longer
become long pauses or jumps. Slow motion changes the playback clock only.
Step advances to the next collision in small runs or next time sample in
large runs. These sampling counts limit display data, not the collision count.

`output/cubes.json` stores the total count, pi estimate, impact snapshots, and
physical-time motion samples for large runs.
Counts remain exact machine integers; positions/velocities are exported as
F32 for display, after two-float calculations. There is no new digit cutoff.
The counter's finite precision and 48-bit Nat overflow limitations still apply,
and accuracy above 12 digits remains unverified.

Snapshot tests compare with the original event-by-event solver, an independent
double-precision simulation, energy invariants and large-count replay endpoints.
Physical-time queries are also checked against an independent event solver
before, during and after the collision burst; browser-clock tests cover contact
timing, velocity between impacts and continued motion after the final impact.

## Rubik's cube and sticker graph

Open `#rubik` in the playground. Inspired by
[the linked visualization](https://x.com/TheMathFlow/status/2101154346583154801),
the cube and graph share the same 54-sticker state. Each graph node is a sticker
position, grouped by face; edges show face-turn permutations. This is not the
graph of all possible cube states.

`rubik/cube.bend` computes integer-coordinate rotations, permutations, legal
scrambles, inverse moves, and solved-state detection. The server uses the
official Bend compiler to build `build/rubik-engine.js` on startup, reusing it
until the source changes. `npm run build:rubik` can also build it explicitly.
Moves run on the browser CPU, without CUDA, server calls, or compilation per turn.
`web/rubik.js` draws and interpolates SVG geometry; it does not implement the
cube's permutation rules.

Use the face buttons or U/R/F/D/L/B keys (Shift for inverse), or click graph
face labels. Drag the cube to orbit. Scramble adds legal random moves; Undo
move and Undo all reverse the recorded history. There is no arbitrary-state
solver. Pause freezes animation, and Reset cancels queued turns and returns to
the solved state. Switching tabs pauses any ongoing sequence.

Tests check face orientations, bijective permutations, fixed centers, cubie
integrity, inverse moves, four-turn cycles, opposite-face commutation, and
scramble reversal.

## Tests and performance

```sh
npm test
npm run bench:demos
npm run bench:demos -- --demo raytracer --width 1024 --height 640
npm run bench:demos -- --demo life --size 256 --steps 240
```

The benchmark checks result agreement between one CPU thread, up to eight CPU
threads, and CUDA. Life must match exactly; original mirror-scene colors may
differ by at most 2/255 between CPU and GPU floating-point libraries. For the
stochastic scenes, the check uses RGB RMS error of at most 2/255 and reports
the largest channel difference, since grazing intersections can change an
individual path. In the original default runs here,
both outputs matched exactly. Benchmarks save `output/benchmarks.json`.

Initial single-run measurements on this machine's RTX 3050:

| Workload | CPU, 1 thread | CPU, 8 threads | CUDA |
| --- | ---: | ---: | ---: |
| Ray tracer, 512 × 320, depth 3 | 483 ms | 103 ms | 115 ms |
| Life, 128 × 128, 120 generations | 151 ms | 135 ms | 170 ms |

These are compute timings measured around Bend's work, with millisecond
resolution. Whole-process time is reported separately and includes startup,
CUDA context initialization and output. Native compilation is excluded. These
are single runs, not statistical performance claims; GPU startup and scheduling
can outweigh savings on small workloads.

Life's benchmark computes just the final state in one GPU call. Replay
generation batches up to 32 generations per GPU call and retains every state;
it should not be compared directly to that final-state-only timing.

After fixing eager reflection evaluation and sparse Life replay work, these
whole-process GPU measurements used the same settings and data as before:

| Workload | Before | After | Result comparison |
| --- | ---: | ---: | --- |
| Ray tracer, 1024 × 640, depth 128 | 18,324 ms | 1,422 ms | Identical PNG bytes (before the PPM change) |
| Life glider, 512 × 512, 1,000 generations | 62,074 ms | 1,752 ms | All 1,001 grids identical |

The glider replay shrank from 16.4 MB to 37 KB without dropping generations.
These are individual local measurements, not guarantees for every pattern.

The tests compare Life against an independent scalar implementation, including
word-boundary neighbors, torus wrapping, a stable block, a blinker and a glider.
Ray tests cover forward/missing/inside-sphere intersections, reflection, and
finite, varied image output. These are numerical tests, not formal proofs of
the rendering or Life implementations.
