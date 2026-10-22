import { useEffect, useRef, useState, type ReactNode } from 'react'
import { PAGES, type PageId } from '@/lib/pages'

// Building blocks for the essay-style pages, following bend-lang.com: one
// monospace column, numbered sections, figures with captions, code with copy.

export function DocPage({ id, kicker, title, sub, tag, children }: { id: PageId; kicker?: ReactNode; title: string; sub: ReactNode; tag?: ReactNode; children: ReactNode }) {
  const scroller = useRef<HTMLElement>(null)
  useEffect(() => { scroller.current?.scrollTo(0, 0) }, [id])
  return (
    <main ref={scroller} className="doc min-h-0 flex-1 overflow-y-auto">
      <div className="doc-col">
        <div className="hero">
          {kicker && <p className="kicker">{kicker}</p>}
          <h1 className="cursor">{title}</h1>
          <p className="sub">{sub}</p>
          {tag && <p className="tag">{tag}</p>}
        </div>
        {children}
        <Pager id={id} />
        <footer>made with Bend · <a href="https://github.com/krymancer/bend-playground">source</a></footer>
      </div>
    </main>
  )
}

function Pager({ id }: { id: PageId }) {
  const docs = PAGES.filter(p => p.kind === 'doc'), i = docs.findIndex(p => p.id === id)
  if (i < 0) return null
  const prev = docs[i - 1], next = docs[i + 1]
  return <nav className="pager"><span>{prev ? <a href={`#${prev.id}`}>← {prev.label}</a> : <a href="#">← home</a>}</span><span>{next && <a href={`#${next.id}`}>{next.label} →</a>}</span></nav>
}

export function Section({ n, title, children }: { n: number | string; title: ReactNode; children: ReactNode }) {
  return <section><h2><span className="n">{n}.</span>{title}</h2>{children}</section>
}

export function Figure({ caption, children, className }: { caption?: ReactNode; children: ReactNode; className?: string }) {
  return <figure className={className}>{children}{caption && <figcaption>{caption}</figcaption>}</figure>
}

// A paper-colored drawing surface. Give it a fixed aspect ratio so the page
// does not jump while Bend is computing.
export function Stage({ aspect = '16 / 10', children, label }: { aspect?: string; children: ReactNode; label?: string }) {
  return <div className="stage" style={{ aspectRatio: aspect }} aria-label={label}>{children}</div>
}

export function Controls({ children }: { children: ReactNode }) { return <div className="controls">{children}</div> }
export function Stats({ children }: { children: ReactNode }) { return <div className="stats">{children}</div> }
export function Stat({ label, value }: { label: string; value: ReactNode }) { return <span>{label} <b>{value ?? '—'}</b></span> }

export function Btn({ children, onClick, ghost, pressed, disabled, label }: { children: ReactNode; onClick: () => void; ghost?: boolean; pressed?: boolean; disabled?: boolean; label?: string }) {
  return <button type="button" className={ghost ? 'btn ghost' : 'btn'} onClick={onClick} aria-pressed={pressed} disabled={disabled} aria-label={label}>{children}</button>
}

export function Range({ label, value, min, max, step = 1, onChange, format }: { label: string; value: number; min: number; max: number; step?: number; onChange: (v: number) => void; format?: (v: number) => ReactNode }) {
  return <label>{label}<input type="range" min={min} max={max} step={step} value={value} onChange={e => onChange(Number(e.target.value))} /><b>{format ? format(value) : value}</b></label>
}

export function Choice<T extends string>({ label, value, options, onChange }: { label: string; value: T; options: readonly (readonly [T, string])[]; onChange: (v: T) => void }) {
  return <label>{label}<select value={value} onChange={e => onChange(e.target.value as T)}>{options.map(([v, l]) => <option key={v} value={v}>{l}</option>)}</select></label>
}

// Minimal Bend/Python highlighting: keywords green, types violet, literals orange, comments gray.
const KEYWORDS = /^(def|type|is|match|case|import|as|law|for|exs|where|do|return|import)$/
function highlight(line: string, key: number) {
  const parts: ReactNode[] = [], re = /(#.*$)|("(?:[^"\\]|\\.)*")|(\b\d+(?:\.\d+)?n?\b)|(\b[A-Za-z_][\w.]*\b)/g
  let last = 0, m: RegExpExecArray | null
  while ((m = re.exec(line))) {
    if (m.index > last) parts.push(line.slice(last, m.index))
    const [t] = m
    if (m[1]) parts.push(<span key={m.index} className="c">{t}</span>)
    else if (m[2] || m[3]) parts.push(<span key={m.index} className="o">{t}</span>)
    else if (KEYWORDS.test(t)) parts.push(<span key={m.index} className="k">{t}</span>)
    else if (/^[A-Z]/.test(t)) parts.push(<span key={m.index} className="v">{t}</span>)
    else parts.push(t)
    last = m.index + t.length
  }
  parts.push(line.slice(last))
  return <span key={key}>{parts}{'\n'}</span>
}

export function Code({ children, file }: { children: string; file?: string }) {
  const [copied, setCopied] = useState(false), text = children.replace(/^\n|\n\s*$/g, '')
  return <>
    {file && <p className="file">{file}</p>}
    <pre><code>{text.split('\n').map(highlight)}</code><button type="button" className="cp" onClick={() => navigator.clipboard?.writeText(text).then(() => { setCopied(true); setTimeout(() => setCopied(false), 1200) })}>{copied ? 'copied' : 'copy'}</button></pre>
  </>
}

// Animated bar chart in the bend-lang.com style. Highlight the bar that matters.
export interface BarDatum { label: string; value: number; display?: string; hi?: boolean }
export function Bars({ title, data, caption, max }: { title?: ReactNode; data: BarDatum[]; caption?: ReactNode; max?: number }) {
  const ref = useRef<HTMLDivElement>(null), [on, setOn] = useState(false), top = max ?? Math.max(...data.map(d => d.value), 1e-9)
  useEffect(() => {
    const el = ref.current; if (!el) return
    const o = new IntersectionObserver(([e]) => { if (e.isIntersecting) { setOn(true); o.disconnect() } }, { threshold: .3 })
    o.observe(el); return () => o.disconnect()
  }, [])
  return <figure>
    {title && <p className="ct">{title}</p>}
    <div ref={ref} className={on ? 'bars on' : 'bars'}>
      {data.map((d, i) => <div key={d.label} className={d.hi ? 'bar hi' : 'bar'} style={{ ['--h' as string]: d.value / top, ['--i' as string]: i }}>
        <b>{d.display ?? d.value}</b><i /><span>{d.label}</span>
      </div>)}
    </div>
    {caption && <figcaption>{caption}</figcaption>}
  </figure>
}
