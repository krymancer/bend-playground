import type { DemoName } from './types'
// Every route of the playground. `lab` pages are the original two-pane
// experiments; `doc` pages are long-form explainers with live Bend figures.
export type DocId = 'neural-networks' | 'neuroevolution' | 'reinforcement-learning' | 'cnn' | 'deepfakes' | 'meadow'
export type PageId = 'home' | DemoName | DocId
export interface Page { id: PageId; kind: 'home' | 'lab' | 'doc'; label: string; group: string; blurb: string }
export const GROUPS = ['Machine learning', 'Life & ecology', 'Light & geometry', 'Probability & physics', 'Algorithms & evolution'] as const
export const PAGES: Page[] = [
  { id: 'home', kind: 'home', label: 'home', group: '', blurb: '' },
  { id: 'neural-networks', kind: 'doc', label: 'Neural networks', group: 'Machine learning', blurb: 'neurons, layers, and gradient descent, trained live' },
  { id: 'neuroevolution', kind: 'doc', label: 'Neuroevolution', group: 'Machine learning', blurb: 'flappIA and CarCrush: brains bred, not trained' },
  { id: 'reinforcement-learning', kind: 'doc', label: 'Reinforcement learning', group: 'Machine learning', blurb: 'Q-learning: an agent learns from reward alone' },
  { id: 'cnn', kind: 'doc', label: 'Convolutional nets', group: 'Machine learning', blurb: 'kernels, feature maps, and pooling, pixel by pixel' },
  { id: 'deepfakes', kind: 'doc', label: 'Detecting deepfakes', group: 'Machine learning', blurb: 'my thesis: CNNs that spot synthetic faces' },
  { id: 'meadow', kind: 'doc', label: 'Sheep meadow', group: 'Life & ecology', blurb: 'grazing, breeding, and inherited traits' },
  { id: 'life', kind: 'lab', label: 'Game of Life', group: 'Life & ecology', blurb: 'four rules, endless patterns' },
  { id: 'raytracer', kind: 'lab', label: 'Ray tracer', group: 'Light & geometry', blurb: 'path-traced spheres, glass and metal' },
  { id: 'rubik', kind: 'lab', label: 'Rubik graph', group: 'Light & geometry', blurb: 'face turns as permutations' },
  { id: 'times-table', kind: 'lab', label: 'Times table', group: 'Light & geometry', blurb: 'modular multiplication on a circle' },
  { id: 'polar', kind: 'lab', label: 'Polar curves', group: 'Light & geometry', blurb: 'roses, spirals, cardioids' },
  { id: 'fourier', kind: 'lab', label: 'Fourier', group: 'Light & geometry', blurb: 'a square wave from circles' },
  { id: 'epicycles', kind: 'lab', label: 'Epicycles', group: 'Light & geometry', blurb: 'draw anything with rotating circles' },
  { id: 'cubes', kind: 'lab', label: 'π cubes', group: 'Probability & physics', blurb: 'π from colliding blocks' },
  { id: 'montecarlo', kind: 'lab', label: 'Monte Carlo π', group: 'Probability & physics', blurb: 'π from random points' },
  { id: 'buffon', kind: 'lab', label: 'Buffon’s needle', group: 'Probability & physics', blurb: 'π from dropped toothpicks' },
  { id: 'sorting', kind: 'lab', label: 'Sorting', group: 'Algorithms & evolution', blurb: 'quick, merge and bubble, op by op' },
  { id: 'shakespeare', kind: 'lab', label: 'Shakespeare', group: 'Algorithms & evolution', blurb: 'monkeys, typewriters, and selection' },
]
export const page = (id: string) => PAGES.find(p => p.id === id)
