import { useCallback, useEffect, useRef, useState } from 'react'
// Request/response over a module worker. Each lab passes its own factory so Vite
// can bundle the worker: `useWorker(() => new Worker(new URL('./x.worker.ts', import.meta.url), { type: 'module' }))`.
export function useWorker(create: () => Worker) {
  const worker = useRef<Worker | null>(null), serial = useRef(0), factory = useRef(create)
  const pending = useRef(new Map<number, { resolve: (data: any) => void; reject: (e: Error) => void }>())
  const [error, setError] = useState<string | null>(null)
  useEffect(() => {
    const w = factory.current(); worker.current = w
    w.onmessage = ({ data }) => {
      const p = pending.current.get(data.id); pending.current.delete(data.id)
      if (data.error) { setError(data.error); p?.reject(new Error(data.error)) } else p?.resolve(data.result)
    }
    w.onerror = e => { setError(e.message || 'Could not load the Bend worker.'); for (const p of pending.current.values()) p.reject(new Error(e.message)); pending.current.clear() }
    return () => { w.terminate(); worker.current = null; for (const p of pending.current.values()) p.reject(new Error('Worker stopped.')); pending.current.clear() }
  }, [])
  const call = useCallback(<T,>(body: Record<string, unknown>): Promise<T> => new Promise((resolve, reject) => {
    if (!worker.current) { reject(new Error('Worker unavailable.')); return }
    const id = ++serial.current; pending.current.set(id, { resolve, reject }); worker.current.postMessage({ ...body, id })
  }), [])
  return { call, error, setError }
}
export const useLab = () => useWorker(() => new Worker(new URL('./lab.worker.ts', import.meta.url), { type: 'module' }))
