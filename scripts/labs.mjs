// Every browser lab compiled from Bend to JavaScript: name -> [entry, ...imports].
// build-labs, build-pages, the local server and the Vite dev server all read this.
export const labs = {
  fourier: ['demos/fourier/math.bend'],
  sorting: ['demos/sorting/sort.bend', 'demos/shared/grid.bend', 'demos/probability/math.bend'],
  shakespeare: ['demos/shakespeare/evolution.bend', 'demos/probability/math.bend'],
  polar: ['demos/polar/curves.bend'],
  neural: ['demos/neural/net.bend', 'demos/probability/math.bend'],
  neuroevolution: ['demos/neuroevolution/flock.bend', 'demos/probability/math.bend'],
  qlearning: ['demos/qlearning/agent.bend', 'demos/probability/math.bend'],
  convolution: ['demos/convolution/conv.bend'],
  deepfake: ['demos/deepfake/meso.bend', 'demos/probability/math.bend'],
  meadow: ['demos/meadow/ecology.bend', 'demos/probability/math.bend'],
};
export const engines = ['rubik', 'times-table', ...Object.keys(labs)];
