import { Code, DocPage, Section } from '@/components/Doc'
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
  return (
    <DocPage id="home" title="Bend playground" sub={<>things I find <b>beautiful</b>, explained and <b>computed</b> in <b>Bend</b></>}
      tag={<><b>machine learning</b> · <b>simulation</b> · <b>geometry</b> · <b>probability</b></>}>
      <p>Each page here is a small essay with a live experiment. The math
        behind every figure is written in <a href="https://bend-lang.com">Bend</a>,
        compiled to JavaScript, and run in your browser. The page only draws
        what Bend computes.</p>
      <p>Some pages started as projects of mine: a
        neuroevolution <a href="https://github.com/krymancer/flappIA">Flappy Bird</a>,
        a <a href="https://github.com/krymancer/carcrush-js-ai">traffic-dodging car</a>,
        a <a href="https://github.com/krymancer/sheep-simulation-3d">3D sheep meadow</a>,
        and my undergraduate thesis
        on <a href="http://repositorio.ufc.br/handle/riufc/78832">detecting deepfakes</a>.
        Here they are rebuilt from first principles.</p>
      <p className="lead">Pick a page. Each one teaches one idea.</p>
      {GROUPS.map((group, i) => (
        <Section key={group} n={i + 1} title={group}>
          <div className="cards">
            {PAGES.filter(p => p.group === group).map(p => (
              <a key={p.id} className="card" href={`#${p.id}`}>
                {p.kind === 'doc' && <span className="new">essay</span>}<b>{p.label}</b>
                <span>{p.blurb}</span>
              </a>
            ))}
          </div>
        </Section>
      ))}
      <Section n={GROUPS.length + 1} title={<>Why <b>Bend</b>?</>}>
        <p>Bend is a pure, affine language: values are used at most once unless
          you mark them with <code>+</code>. In return, any two independent calls can
          run at the same time, on every core or on the GPU, with no threads or locks.</p>
        <Code file="pow2.bend">{sample}</Code>
        <p>The same property suits machine learning: a population of neural
          networks, a batch of training examples, or every pixel of a feature map
          splits into independent halves.</p>
      </Section>
    </DocPage>
  )
}
