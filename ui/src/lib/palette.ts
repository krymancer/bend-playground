// Canvas colors matching the CSS theme (bend-lang.com palette). Canvases
// cannot read Tailwind classes, so every drawing imports these instead.
export const P = {
  paper: '#f2eee7', paper2: '#e9e4db', paper3: '#ddd7cc', line: '#e4dfd5', bar: '#cdc7bc',
  ink1: '#a9a59d', ink2: '#87847d', ink3: '#69665f', ink4: '#4d4a44',
  vio: '#8b83b5', grn: '#7e9a5e', ora: '#c4845c', red: '#c46a60', blu: '#5f8fa8',
  font: 'ui-monospace, Menlo, "SF Mono", Consolas, "Liberation Mono", monospace',
} as const
export const mono = (px: number, weight = 400) => `${weight} ${px}px ${P.font}`
// Diverging fill for signed values in [-1, 1]: red below zero, green above.
export function signed(v: number, alpha = 1) {
  const t = Math.max(-1, Math.min(1, v)), [r, g, b] = t < 0 ? [196, 106, 96] : [126, 154, 94], k = Math.abs(t)
  const mix = (c: number, p: number) => Math.round(p + (c - p) * k)
  return `rgba(${mix(r, 242)},${mix(g, 238)},${mix(b, 231)},${alpha})`
}
