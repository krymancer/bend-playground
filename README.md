# Bend playground

Three experiments: a ray tracer, Conway's Game of Life, and pi from colliding
blocks. Numerical computation runs in Bend 2, on native CPU or NVIDIA CUDA.

## Visual playground

```sh
npm run raytracer -- --gpu
npm run life -- --gpu
npm run cubes -- --gpu --digits 3
npm run playground
```

Open **http://localhost:3000** to view the rendered scene, Life replay, and π
cubes. You can change settings, choose CPU or GPU, and run each experiment from the viewer.
Life has play, pause, single-step, generation scrubbing, and speed controls.
The ray tracer saves an ordinary PNG you can download.

The **π cubes** tab animates Bend-computed collision positions and velocities,
with play/pause, step, reset, a timeline, and a velocity-space plot. Small runs
include every impact; large runs sample block/wall pairs at the requested mass
ratio. Playback pacing is illustrative rather than physical time. The digit
input has no application-level upper cutoff; the existing numerical and Nat
overflow limitations still apply.

On this machine, Tailscale Serve exposes the viewer privately at
**https://panam.tailcd7688.ts.net/**; append `#cubes` to open the cubes tab.
The local server must be running (`npm run playground`).

The run commands build the relevant native executable automatically. Omit
`--gpu` for CPU execution; use `--threads 8` to try multiple CPU cores. CUDA
setup is described below if this is a fresh checkout.

```sh
npm run raytracer -- --gpu --width 1024 --height 640 --bounces 4
npm run life -- --gpu --size 256 --steps 240 --pattern random --seed 123
npm run life -- --pattern glider --size 64 --steps 120
npm run bench:demos
```

Outputs go to `output/`. The browser only displays Bend's output; it does not
reimplement ray tracing or Life. See [demos/README.md](demos/README.md) for
controls, implementation details, validation, and measured performance.

## Pi from colliding blocks

A terminal implementation of the two blocks and wall from
[3Blue1Brown's video](https://www.youtube.com/watch?v=6dTyOl1fmDo), with a
CPU version and a working NVIDIA GPU version. All numerical computation is
written in Bend 2. The scripts only build and launch the program.

## Run

```sh
npm run setup
PI_DIGITS=10 npm start
```

```text
Pi from composed collision transformations (two-float precision)
10 digits: 3141592653 collisions -> pi = 3.141592653
```

`PI_DIGITS` accepts positive integers with **no application-level upper cutoff**,
including the leading `3`; it defaults to 10. Results above 12 digits are
experimental and print a warning. Twelve digits correspond to **314,159,265,358 collisions** and a heavy
mass of `10^22` times the small mass. Counts use Bend's 48-bit `Nat`, avoiding
a 32-bit integer overflow.

Requirements: Git, Node.js 24+, and Bun. There are no npm dependencies. Setup
pins **Bend 2.0.5** in `.tools/bend` without changing your global tools. Bend 1
and HVM do not support this source. With Bend 2 installed, you can also run
`PI_DIGITS=10 bend main.bend`.

`npm start` runs the CPU/JavaScript version. To build a standalone CPU executable
(with Clang 14+; Bun is unnecessary for this path):

```sh
npm run build
PI_DIGITS=12 ./build/pi --threads 1
```

## Run on the GPU

The GPU path requires **Linux, an NVIDIA GPU and driver, Clang 19+, and CUDA
NVRTC**. It has been executed successfully on this machine's RTX 3050.

```sh
# If CUDA NVRTC is not already installed, install pinned NVIDIA libraries
# inside .tools/cuda (~93 MB download). Requires Python 3; no sudo or pip.
npm run setup:gpu
npm run build:gpu
PI_DIGITS=10 npm run gpu
```

If CUDA is already installed elsewhere, set `CUDA_HOME` before building and
running. The launcher uses `.tools/cuda` when available, otherwise
`/usr/local/cuda`. Local downloads are checked against pinned SHA-256 hashes.

The GPU command explicitly requests CUDA and fails if no compatible GPU is
available; it does not silently run the calculation on the CPU. It reserves
1 GiB for Bend's runtime, including its per-worker stacks. Keep
`build/pi-gpu.gpu` alongside `build/pi-gpu`.

The GPU evaluates **256 independent collision-index candidates per search
round** using Bend's parallel call syntax. The program is genuinely parallel,
but this does not imply it beats the CPU: composing transformations makes one
count so cheap that CUDA initialization and launch overhead can dominate.
For this workload, prefer measuring over assuming GPU acceleration.

## How large counts work

The small mass is 1. For `d` digits the heavy mass is `100^(d-1)`. Rather than
round that huge mass into one float, the accelerated solver represents its
inverse square root, `t = 10^(-(d-1))`, using two F32 components.

An elastic block collision followed by a wall reflection is a rotation of
scaled velocity space. Its coefficients come directly from the collision laws:

```text
a = (1 - t*t) / (1 + t*t)
b = 2*t / (1 + t*t)
```

Multiplying the complex pair `(a, b)` composes those two-impact transformations.
Exponentiation by squaring evaluates the velocity after `k` pairs in
`O(log k)` compositions. A parallel search locates the first pair whose wall
reflection would be unphysical, then checks whether the preceding pair had
already finished to distinguish an even from an odd collision count.

The search spans `0 < k <= 2/t`, where the relevant velocity sign crosses zero
once. Each round reduces that bracket by up to 256 times. It never visits all
billions of intermediate collisions. It uses **no stored pi value, trig
functions, or pi/angle count formula**. This is the collision-composition
approach selected for the larger-number version.

`precise.bend` implements TwoSum and Dekker TwoProduct with two F32 components,
retaining roughly 48 significant binary bits. It is finite precision, not
arbitrary precision. The 1–12 digit range is regression-tested. Higher requests
are allowed, but may return inaccurate results or fail on Bend's 48-bit integer
overflow; removing the input cutoff does not extend numeric precision. Do not build with
fast-math or floating-point reassociation. Bend's CUDA compiler explicitly
turns off fused contraction (`--fmad=false`).

Background:
[3Blue1Brown's velocity-space explanation](https://www.3blue1brown.com/lessons/colliding-blocks-v2/)
and [Hida, Li and Bailey's floating-point expansion algorithms](https://www.davidhbailey.com/dhbpapers/arith15.pdf).

## Original event-by-event simulation

The original solver is preserved for learning and cross-checking:

```sh
npm run simulate                  # Show all four supported cases
PI_DIGITS=4 npm run simulate       # 3141 collisions -> 3.141
```

It advances directly to each block or wall impact. At a block collision:

```text
v' = ((1 - M)*v + 2*M*w) / (1 + M)
w' = (2*v + (M - 1)*w) / (1 + M)
```

At a wall impact, `v` changes sign. It stops when `0 <= v <= w`.
This reference solver retains its **1–4 digit** limit and 10,000-impact budget;
plain F32 velocity updates accumulate enough error to give wrong higher counts.

## Validation

```sh
npm test
npm run check
```

The 21 tests cover all 12 accelerated counts, both sides of the final search
boundary, two-float precision and cancellation, and equivalence of composed
and repeated transformations. They also compare the two solvers for 1–4 digits
and check the reference simulation's event ordering, contacts, energy,
momentum, reflection, termination and budget exhaustion.

`LAWS.bend` / `PROOF.bend` prove three structural properties of the original
reference solver: zero initial count, one increment per impact, and no change
after stopping. These proofs do **not** establish the accelerated algorithm's
floating-point accuracy or prove the pi theorem.

Files:

- `main.bend`, `gpu.bend`: terminal CPU and GPU entry points.
- `accelerated.bend`: collision composition and parallel search.
- `precise.bend`: extended precision using pairs of F32 values.
- `physics.bend`, `simulate.bend`: original event-by-event reference solver.
- `tests/`: numerical regression tests.
- `scripts/`: pinned compiler/CUDA setup, builds and GPU launcher.
