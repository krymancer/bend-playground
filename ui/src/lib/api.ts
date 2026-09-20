export async function readOutput<T>(name: string): Promise<T | null> {
  const r = await fetch(`/output/${name}.json`, { cache: 'no-cache' })
  if (r.status === 404) return null
  if (!r.ok) throw new Error(`Could not load ${name}.`)
  return r.json()
}

export async function runDemo(body: Record<string, unknown>): Promise<string> {
  const r = await fetch('/api/run', { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(body) })
  const text = await r.text()
  if (!r.ok) throw new Error(text)
  return text
}

export async function cancelDemo(): Promise<void> {
  const r = await fetch('/api/cancel', { method: 'POST' })
  if (!r.ok) throw new Error(await r.text())
}
