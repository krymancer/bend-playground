# Visual Bend experiments

All three demos are original Bend implementations. The official Bend examples and
guide informed the runtime setup; no JavaScript physics or Life rules are used.
The local web viewer displays PNG pixels or recorded, bit-packed generations.

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
rendered PNG, and play/pause/step/scrub Life's recorded generations. One compute
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

Defaults: 512 × 320, three reflection levels, four samples per pixel.
`--bounces 0` shows the sky only. The scene contains four colored/reflective
spheres and a checkerboard floor, with analytic sphere/plane intersections,
directional lighting, hard shadows, recursive reflections and gamma encoding.
A ray stops as soon as it misses the scene. Reflection depth is an upper bound;
raising it does not force escaped rays through unnecessary reflection work.

`raytracer/ray.bend` builds a balanced binary tree of pixel tasks. Padded leaves
outside the image return black. The native program returns packed RGB values;
the Node launcher merely encodes them as `output/raytracer.png` using zlib.
No image-generation service or graphics library is involved.

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

## Tests and performance

```sh
npm test
npm run bench:demos
npm run bench:demos -- --demo raytracer --width 1024 --height 640
npm run bench:demos -- --demo life --size 256 --steps 240
```

The benchmark checks result agreement between one CPU thread, up to eight CPU
threads, and CUDA. Life must match exactly; ray colors may differ by at most
2/255 between CPU and GPU floating-point libraries. In the default runs here,
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
| Ray tracer, 1024 × 640, depth 128 | 18,324 ms | 1,422 ms | Identical PNG bytes |
| Life glider, 512 × 512, 1,000 generations | 62,074 ms | 1,752 ms | All 1,001 grids identical |

The glider replay shrank from 16.4 MB to 37 KB without dropping generations.
These are individual local measurements, not guarantees for every pattern.

The tests compare Life against an independent scalar implementation, including
word-boundary neighbors, torus wrapping, a stable block, a blinker and a glider.
Ray tests cover forward/missing/inside-sphere intersections, reflection, and
finite, varied image output. These are numerical tests, not formal proofs of
the rendering or Life implementations.
