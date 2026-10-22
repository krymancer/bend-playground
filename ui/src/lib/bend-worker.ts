import { asset } from './runtime'
// Shared plumbing for workers that call Bend-compiled modules. Constructors are
// `{ $: 'Name', ...fields }`, Nat is BigInt, and List is Con/Nil.
export type Bend = Record<string, (...args: any[]) => any>
const engines = new Map<string, Promise<Bend>>()
export const engine = (name: string) => { if (!engines.has(name)) engines.set(name, import(/* @vite-ignore */ asset(`${name}-engine.js`)).then(m => m.default)); return engines.get(name)! }
export const list = (a: any[]) => a.reduceRight((tail, head) => ({ $: 'Con', head, tail }), { $: 'Nil' })
export const array = (xs: any) => { const a = []; for (let x = xs; x.$ === 'Con'; x = x.tail) a.push(x.head); return a }
// Serve one request at a time, time the Bend work, and transfer typed arrays.
export function serve(name: string, handle: (d: any, bend: Bend) => any) {
  let queue = Promise.resolve()
  self.onmessage = (event: MessageEvent) => {
    queue = queue.then(async () => {
      const { id, ...d } = event.data
      try {
        const bend = await engine(name), start = performance.now(), result = await handle(d, bend)
        const transfers = Object.values(result).filter(v => ArrayBuffer.isView(v)).map(v => (v as Float32Array).buffer)
        self.postMessage({ id, result: { ...result, computeMs: performance.now() - start } }, { transfer: transfers })
      } catch (e) { self.postMessage({ id, error: (e as Error).message }) }
    })
  }
}
