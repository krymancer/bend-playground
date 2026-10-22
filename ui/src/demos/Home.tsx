import { useState } from 'react'
import { Code, DocPage } from '@/components/Doc'
import { GROUPS, PAGES } from '@/lib/pages'

const sample = `
# Two calls, one fork: Bend runs them in parallel.
def pow2(+n: Nat) -> U32:
  match n:
    case 0n: 1
    case 1n+p:
      a b = pow2(p) pow2(p)
      (a + b : U32)
`

export function Home() {
  const [filter, setFilter] = useState<string>('all')
  return (
    <DocPage id="home" title="Bend playground" sub={<>things I find <b>beautiful</b>, explained and <b>computed</b> in <b>Bend</b></>}
      tag={<><b>machine learning</b> · <b>simulation</b> · <b>geometry</b> · <b>probability</b></>}>
      <p>Each page is a small essay with a live experiment. The math behind
        every figure is written in <a href="https://bend-lang.com">Bend</a>,
        compiled to JavaScript, and run in your browser.</p>
      <nav className="filter" aria-label="Filter by category">
        {['all', ...GROUPS].map(g => (
          <button key={g} type="button" aria-pressed={filter === g} onClick={() => setFilter(g)}>{g}</button>
        ))}
      </nav>
      <div className="cards">
        {PAGES.filter(p => p.kind !== 'home' && (filter === 'all' || p.group === filter)).map(p => (
          <a key={p.id} className="card" href={`#${p.id}`}>
            <b>{p.label}</b>
            <span>{p.blurb}</span>
          </a>
        ))}
      </div>
      <section>
        <h2>Why Bend?</h2>
        <p>Bend is a pure, affine language: values are used at most once unless
          you mark them with <code>+</code>. In return, any two independent calls can
          run at the same time, on every core or on the GPU, with no threads or locks.</p>
        <Code file="pow2.bend">{sample}</Code>
        <p>The same property suits machine learning: a population of neural
          networks, a batch of training examples, or every pixel of a feature map
          splits into independent halves.</p>
      </section>
    </DocPage>
  )
}
