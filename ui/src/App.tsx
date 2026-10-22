import { lazy, Suspense, useCallback, useEffect, useState } from 'react'
import { TooltipProvider } from '@/components/ui/tooltip'
import { JobProvider } from '@/components/RunPanel'
import { Raytracer } from '@/demos/Raytracer'
import { Life } from '@/demos/Life'
import { Cubes } from '@/demos/Cubes'
import { Rubik } from '@/demos/Rubik'
import { Probability } from '@/demos/Probability'
import { TimesTable } from '@/demos/TimesTable'
import { Fourier } from '@/demos/Fourier'
import { Sorting } from '@/demos/Sorting'
import { Shakespeare } from '@/demos/Shakespeare'
import { Polar } from '@/demos/Polar'
import { Home } from '@/demos/Home'
import { readOutput } from '@/lib/api'
import { GROUPS, PAGES, page, type PageId } from '@/lib/pages'
import type { CubesOutput, LifeOutput, RayOutput, ProbabilityOutput } from '@/lib/types'

const DOCS = {
  'neural-networks': lazy(() => import('@/docs/NeuralNetworks')),
  neuroevolution: lazy(() => import('@/docs/Neuroevolution')),
  'reinforcement-learning': lazy(() => import('@/docs/ReinforcementLearning')),
  cnn: lazy(() => import('@/docs/Cnn')),
  deepfakes: lazy(() => import('@/docs/Deepfakes')),
  meadow: lazy(() => import('@/docs/Meadow')),
} as const

const fromHash = (): PageId => (page(location.hash.slice(1))?.id ?? 'home')

interface Slot<T> { data: T | null; loading: boolean; error: string | null; loaded: boolean }
const empty = { data: null, loading: false, error: null, loaded: false }

export default function App() {
  const [demo, setDemo] = useState<PageId>(fromHash)
  const [ray, setRay] = useState<Slot<RayOutput>>(empty)
  const [life, setLife] = useState<Slot<LifeOutput>>(empty)
  const [cubes, setCubes] = useState<Slot<CubesOutput>>(empty)

  const [montecarlo, setMontecarlo] = useState<Slot<ProbabilityOutput>>(empty)
  const [buffon, setBuffon] = useState<Slot<ProbabilityOutput>>(empty)

  useEffect(() => { const h = () => setDemo(fromHash()); addEventListener('hashchange', h); return () => removeEventListener('hashchange', h) }, [])

  const load = useCallback(async <T,>(name: string, set: (f: (s: Slot<T>) => Slot<T>) => void) => {
    set(s => ({ ...s, loading: true, error: null }))
    try { const data = await readOutput<T>(name); set(() => ({ data, loading: false, error: null, loaded: true })) }
    catch (e) { set(s => ({ ...s, loading: false, error: (e as Error).message, loaded: true })) }
  }, [])
  const reloadRay = useCallback(() => load<RayOutput>('raytracer', setRay), [load])
  const reloadLife = useCallback(() => load<LifeOutput>('life', setLife), [load])
  const reloadCubes = useCallback(() => load<CubesOutput>('cubes', setCubes), [load])

  const reloadMontecarlo = useCallback(() => load<ProbabilityOutput>('montecarlo', setMontecarlo), [load])
  const reloadBuffon = useCallback(() => load<ProbabilityOutput>('buffon', setBuffon), [load])

  useEffect(() => {
    if (demo === 'raytracer' && !ray.loaded && !ray.loading) void reloadRay()
    if (demo === 'life' && !life.loaded && !life.loading) void reloadLife()
    if (demo === 'cubes' && !cubes.loaded && !cubes.loading) void reloadCubes()
    if (demo === 'montecarlo' && !montecarlo.loaded && !montecarlo.loading) void reloadMontecarlo()
    if (demo === 'buffon' && !buffon.loaded && !buffon.loading) void reloadBuffon()
  }, [demo, ray, life, cubes, montecarlo, buffon, reloadRay, reloadLife, reloadCubes, reloadMontecarlo, reloadBuffon])

  useEffect(() => { document.title = demo === 'home' ? 'Bend playground' : `${page(demo)?.label} · Bend playground` }, [demo])
  const current = page(demo)!, Doc = current.kind === 'doc' ? DOCS[demo as keyof typeof DOCS] : null

  return (
    <TooltipProvider>
      <JobProvider>
        <div className="flex min-h-dvh flex-col lg:h-dvh lg:overflow-hidden">
          <header className="flex shrink-0 items-center justify-between gap-4 px-4 pt-3 pb-2 text-[13px] text-ink-1">
            <a href="#" className="min-w-0 truncate whitespace-nowrap hover:text-vio">~/bend-playground{demo !== 'home' && <span className="text-ink-2">/{demo}</span>}</a>
            <nav className="flex shrink-0 items-center gap-4 text-ink-2">
              <label className="flex items-center gap-2">
                <span className="sr-only">Jump to page</span>
                <select value={demo} onChange={e => { location.hash = e.target.value === 'home' ? '' : e.target.value }}
                  className="max-w-[44vw] cursor-pointer rounded-[3px] border border-input bg-paper px-1.5 py-0.5 text-ink-3 hover:border-vio focus-visible:outline-2 focus-visible:outline-vio">
                  <option value="home">index</option>
                  {GROUPS.map(g => <optgroup key={g} label={g}>{PAGES.filter(p => p.group === g).map(p => <option key={p.id} value={p.id}>{p.label}</option>)}</optgroup>)}
                </select>
              </label>
              <a href="https://github.com/krymancer/bend-playground" target="_blank" rel="noopener" title="github" className="hover:text-vio">
                <svg viewBox="0 0 24 24" className="size-[1.15em] fill-current" aria-hidden><path d="M12 .3a12 12 0 0 0-3.8 23.4c.6.1.8-.3.8-.6v-2.2c-3.3.7-4-1.4-4-1.4-.6-1.4-1.4-1.8-1.4-1.8-1-.7.1-.7.1-.7 1.2.1 1.8 1.2 1.8 1.2 1 1.8 2.8 1.3 3.5 1 .1-.8.4-1.3.7-1.6-2.7-.3-5.5-1.3-5.5-5.9 0-1.3.5-2.4 1.2-3.2-.1-.3-.5-1.5.1-3.2 0 0 1-.3 3.3 1.2a11.5 11.5 0 0 1 6 0c2.3-1.5 3.3-1.2 3.3-1.2.6 1.7.2 2.9.1 3.2.8.8 1.2 1.9 1.2 3.2 0 4.6-2.8 5.6-5.5 5.9.4.4.8 1.1.8 2.2v3.3c0 .3.2.7.8.6A12 12 0 0 0 12 .3"/></svg>
                <span className="sr-only">GitHub</span>
              </a>
            </nav>
          </header>
          {demo === 'home' && <Home />}
          {Doc && <Suspense fallback={<main className="doc flex-1" />}><Doc /></Suspense>}
          {demo === 'raytracer' && <Raytracer {...ray} reload={reloadRay} />}
          {demo === 'life' && <Life {...life} reload={reloadLife} active />}
          {demo === 'cubes' && <Cubes {...cubes} reload={reloadCubes} active />}
          {demo === 'rubik' && <Rubik active />}
          {demo === 'montecarlo' && <Probability key="montecarlo" method="montecarlo" {...montecarlo} reload={reloadMontecarlo} />}
          {demo === 'buffon' && <Probability key="buffon" method="buffon" {...buffon} reload={reloadBuffon} />}
          {demo === 'times-table' && <TimesTable />}
          {demo === 'fourier' && <Fourier key="wave" mode="wave" />}
          {demo === 'epicycles' && <Fourier key="drawing" mode="drawing" />}
          {demo === 'sorting' && <Sorting />}
          {demo === 'shakespeare' && <Shakespeare />}
          {demo === 'polar' && <Polar />}
        </div>
      </JobProvider>
    </TooltipProvider>
  )
}
