import { useEffect, useRef, useState } from 'react'
import { Bars, Btn, Code, Controls, DocPage, Figure, Range, Section, Stage, Stat, Stats } from '@/components/Doc'
import { usePaint } from '@/lib/lab-canvas'
import { useWorker } from '@/lib/use-lab'
import { drawKernels, drawMeso, drawPipeline, drawSamples, drawTraining, type Point } from './deepfake-draw'

const THESIS = 'http://repositorio.ufc.br/handle/riufc/78832'
const MAX_STEPS = 300, STEPS_PER_CALL = 3

interface Data { patches: Float32Array; trace: Float32Array; labels: Float32Array }
interface Snap { kernels: Float32Array; out: Float32Array; maps: Float32Array; preds: Float32Array; confusion: Float32Array; valLoss: number; loss?: number; acc?: number; computeMs?: number }
const accuracy = (c: Float32Array) => (c[0] + c[3]) / Math.max(1, c[0] + c[1] + c[2] + c[3])
const pct = (v: number | undefined) => v === undefined || !Number.isFinite(v) ? '—' : `${(v * 100).toFixed(1)}%`

function useNarrow() {
  const query = '(max-width: 560px)', [narrow, setNarrow] = useState(() => matchMedia(query).matches)
  useEffect(() => { const m = matchMedia(query), f = () => setNarrow(m.matches); m.addEventListener('change', f); return () => m.removeEventListener('change', f) }, [])
  return narrow
}

const artifactCode = `
# A checkerboard, +1 and -1 on alternating pixels ...
+sign = Bool.pick(F32,U32.is_eq(U32.and((x + y : U32),1),0),1.0,F32.neg(1.0))
# ... added only to fakes (fake = 1.0), only inside the face mask m.
(base + ... + grain + fake * amp * m * sign - off : F32)
`
const scanCode = `
# Slide one kernel over every window: ReLU, and remember which windows fired.
def scan(ws: List<&2, Win>, +w: Win, +b: F32, s: F32, n: F32, g: Win) -> Acc:
  match ws:
    case []: Acc{s,n,g}
    case +x <> rest:
      +z = (dot9(w,x) + b : F32)
      +on = Bool.pick(F32,(z > 0.0 : F32),1.0,0.0)
      scan(rest,w,b,(s + on * z : F32),(n + on : F32),axpy(g,on,x))
`
const gradCode = `
# The batch gradient is a sum over examples: split, recurse in parallel, add.
def grad(+net: Net, batch: Batch) -> Grad:
  match batch:
    case One{ex}: example(net,ex)
    case Two{l,r}:
      a b = grad(net,l) grad(net,r)
      plus(a,b)
`

type Call = <T>(body: Record<string, unknown>) => Promise<T>

function Lab({ narrow, call, error }: { narrow: boolean; call: Call; error: string | null }) {
  const [seed, setSeed] = useState(42), [amp, setAmp] = useState(0.05), [resets, setResets] = useState(0)
  const [data, setData] = useState<Data | null>(null), [snap, setSnap] = useState<Snap | null>(null)
  const [history, setHistory] = useState<Point[]>([]), [step, setStep] = useState(0)
  const [running, setRunning] = useState(false), [ready, setReady] = useState(false)
  const generation = useRef(0), steps = useRef(0)

  // A new seed or strength means a new dataset and a fresh network.
  useEffect(() => {
    const g = ++generation.current
    setRunning(false); setReady(false)
    const timer = setTimeout(async () => {
      try {
        const d = await call<Data>({ action: 'reset', seed, amp }); if (g !== generation.current) return
        setData(d)
        const s = await call<Snap>({ action: 'validation' }); if (g !== generation.current) return
        steps.current = 0; setStep(0); setSnap(s); setHistory([{ step: 0, train: NaN, val: accuracy(s.confusion) }]); setReady(true)
      } catch { /* shown through `error` */ }
    }, 150)
    return () => clearTimeout(timer)
  }, [seed, amp, resets, call])

  useEffect(() => {
    if (!running || !ready) return
    let alive = true
    const g = generation.current
    ;(async () => {
      while (alive && steps.current < MAX_STEPS) {
        const r = await call<Snap>({ action: 'train', steps: STEPS_PER_CALL })
        if (g !== generation.current) return
        steps.current += STEPS_PER_CALL
        const at = steps.current
        setSnap(r); setStep(at); setHistory(h => [...h, { step: at, train: r.acc ?? NaN, val: accuracy(r.confusion) }])
        await new Promise(requestAnimationFrame)
      }
      if (alive) setRunning(false)
    })().catch(() => setRunning(false))
    return () => { alive = false }
  }, [running, ready, call])

  const samples = usePaint(c => drawSamples(c, data?.patches ?? null, data?.labels ?? null, snap?.preds ?? null, data?.trace ?? null))
  const training = usePaint(c => drawTraining(c, history, snap?.confusion ?? null, MAX_STEPS))
  const kernels = usePaint(c => drawKernels(c, snap?.kernels ?? null, snap?.out ?? null, snap?.maps ?? null, narrow))
  const done = step >= MAX_STEPS, last = history[history.length - 1]

  return <>
    <Figure caption={<>Six held-out faces from seed {seed}, drawn by Bend. Odd ones are fakes. The bar and number under each are the detector’s current p(fake): <em>green</em> when its verdict is right, red when wrong. The last tile is the artifact alone: the second face minus the same face drawn without it.</>}>
      <Stage aspect="4 / 3.3"><canvas ref={samples} role="img" aria-label="Six synthetic 16 by 16 faces, alternating real and fake, with the detector's probability of fake under each, and the checkerboard artifact on its own." /></Stage>
    </Figure>
    <Code file="demos/deepfake/meso.bend · the generator's trace">{artifactCode}</Code>
    <Figure caption={<>Accuracy on the 128 training faces (gray) and on 128 faces the network never trains on (violet). Below: the held-out verdicts; fake is the positive class. Training is full-batch gradient descent with Adam, run in Bend, {STEPS_PER_CALL} steps per call.</>}>
      <Stage aspect={narrow ? "4 / 5.2" : "4 / 4.4"}><canvas ref={training} role="img" aria-label={`Training curve and confusion matrix. Step ${step}, held-out accuracy ${pct(last?.val)}.`} /></Stage>
      <Controls>
        <Btn onClick={() => setRunning(r => !r)} disabled={!ready || done} pressed={running}>{running ? 'Pause' : done ? 'Done' : step ? 'Resume' : 'Train'}</Btn>
        <Btn ghost onClick={() => setResets(r => r + 1)}>Reset</Btn>
        <Range label="artifact" value={amp} min={0} max={0.1} step={0.005} onChange={setAmp} format={v => v.toFixed(3)} />
        <Range label="seed" value={seed} min={1} max={99} onChange={setSeed} />
      </Controls>
      <Stats>
        <Stat label="step" value={`${step}/${MAX_STEPS}`} />
        <Stat label="train" value={pct(snap?.acc)} />
        <Stat label="held-out" value={pct(last?.val)} />
        <Stat label="held-out loss" value={snap && Number.isFinite(snap.valLoss) ? snap.valLoss.toFixed(3) : undefined} />
        <Stat label="Bend" value={snap?.computeMs !== undefined ? `${snap.computeMs.toFixed(0)} ms/call` : undefined} />
      </Stats>
      {error && <p className="note" role="alert">Bend worker error: {error}</p>}
    </Figure>
    <Figure caption={<>The four learned 3×3 kernels (<em>green</em> positive, red negative weights) and their ReLU feature maps on the first real and the first fake face above. A positive vote pushes the output towards “fake”. With a visible artifact, one kernel usually turns into a checkerboard and lights up mostly inside fake faces.</>}>
      <Stage aspect={narrow ? '6 / 6.3' : '16 / 10'}><canvas ref={kernels} role="img" aria-label="Learned convolution kernels and their feature maps on a real and a fake face." /></Stage>
    </Figure>
  </>
}

function Diagram({ draw, aspect, label }: { draw: (c: HTMLCanvasElement) => void; aspect: string; label: string }) {
  const ref = usePaint(draw)
  return <Stage aspect={aspect}><canvas ref={ref} role="img" aria-label={label} /></Stage>
}

function Pipeline({ narrow, call }: { narrow: boolean; call: Call }) {
  // The pipeline draws two faces from the same Bend generator as the lab.
  const [patches, setPatches] = useState<Float32Array | null>(null)
  useEffect(() => { call<{ patches: Float32Array }>({ action: 'faces', seed: 7, amp: 0.05, count: 2 }).then(d => setPatches(d.patches)).catch(() => {}) }, [call])
  return <Diagram aspect={narrow ? '1 / 1' : '16 / 5.4'} draw={c => drawPipeline(c, patches, narrow)}
    label="Pipeline: a video is split into frames, a face is detected and cropped with MTCNN, a convolutional network classifies the crop as real or fake." />
}

export default function Deepfakes() {
  const narrow = useNarrow()
  const { call, error } = useWorker(() => new Worker(new URL('./deepfake.worker.ts', import.meta.url), { type: 'module' }))
  return (
    <DocPage id="deepfakes" kicker={<>undergraduate thesis · UFC Sobral · 2023</>} title="Detecting deepfakes"
      sub={<>convolutional networks that tell <b>real</b> faces from <b>synthetic</b> ones</>}
      tag={<><b>MesoNet</b> · <b>transfer learning</b> · a <b>toy detector</b> trained live in Bend</>}>
      <p>This page retells my undergraduate thesis, <a href={THESIS}>Detecção de Deepfakes Utilizando
        Redes Neurais Convolucionais</a> (Computer Engineering, Federal University of Ceará, Sobral campus, 2023,
        advised by Prof. Dr. Iális Cavalcante de Paula Júnior). The numbers in sections 2, 3 and 5 are the
        thesis’s own. The experiment in section 4 is a toy built for this page.</p>

      <Section n={1} title={<>Faces can <b>LIE</b>.</>}>
        <p className="lead">A deepfake is a face that a neural network made up.</p>
        <p>Deepfakes are images and videos in which faces are generated, created or altered by deep learning.
          The algorithms create, edit and swap faces automatically, and the results are getting close to impossible
          for an untrained eye to tell from the real thing.</p>
        <p>The thesis opens with a fake video of Mark Zuckerberg claiming that Facebook steals its users’ data.
          Free notebooks with GPUs, such as Google Colab and Kaggle, have made such fakes easy to train. They can be used
          to defame a candidate during an election, to manipulate opinion, to trick people into giving up confidential information,
          and to wreck victims’ reputations and mental health. A 2019 Deeptrace report found that 96% of the deepfake videos online were pornographic.</p>
        <p>They come in three broad families:</p>
        <ul>
          <li><b>face synthesis</b>: entirely new faces, usually from a GAN (StyleGAN2 is behind thispersondoesnotexist.com);</li>
          <li><b>face swap</b>: one person’s face pasted onto another’s video, the usual weapon against public figures;</li>
          <li><b>attributes and expression</b>: changing age, hair, gender or mood, as the FaceApp app does with StarGAN.</li>
        </ul>
        <p>Every tool leaves its own traces, which makes it hard for one network to catch them all. The thesis also notes
          that new ways of making fakes appear faster than new ways of detecting them.</p>
      </Section>

      <Section n={2} title={<>From video to <b>VERDICT</b>.</>}>
        <p>A detector does not look at a whole video. It looks at faces, one frame at a time.</p>
        <Figure caption="The pipeline. Frames are pulled from a video, a face detector finds and crops each face, and a convolutional network scores the crop. The faces here are synthetic, drawn by the same Bend generator as section 4.">
          <Pipeline narrow={narrow} call={call} />
        </Figure>
        <p>The thesis trained on <b>DFGC-21</b> (the DeepFake Game Competition 2021). It is derived from Celeb-DF, a
          collection of more than 5000 deepfake videos of people of different ethnicities and ages. The subset used holds about
          1000 images taken from those videos. Most of its fakes are face swaps with post-processing that makes them harder to
          classify. Other subsets use FaceShifter, FaceController, MegaFS, FaceMorpher with dlib landmarks, facial
          anonymisation, cut-and-paste, and adversarial noise (Table 1).</p>
        <p>Faces were cropped with <b>MTCNN</b>, which beat the Face Extractor and OpenCV detectors in the author’s comparison.
          Data augmentation (rotations, shifts, zoom, flips) enlarged the set, and a 70/20/10 split produced:</p>
        <Stats><Stat label="training images" value="6893" /><Stat label="test + validation" value="2954" /><Stat label="dataset subsets" value="17" /></Stats>
      </Section>

      <Section n={3} title={<>Look in the <b>MIDDLE</b>.</>}>
        <p>The starting point was <b>MesoNet</b> (Afchar et al., 2018), a compact CNN designed for spotting forged faces in video.
          It is light and fast enough for modest hardware.</p>
        <p>The name explains the idea. The argument comes from the MesoNet paper, not from the thesis. At the microscopic level,
          pixel noise is destroyed by video compression. At the semantic level, “is this a face?”, a good fake is convincing.
          In between lies the <em>mesoscopic</em> level: textures and structures a few pixels wide, where generators still
          slip. So the network is kept shallow and small, a few layers of convolutions that see texture rather than identity.</p>
        <Figure caption={<>Meso-4, with the layer sizes given in the thesis (Figure 8, after Afchar et al.). Four convolution blocks with batch normalisation (BN) and max pooling shrink a 256×256 face to 8×8×16 = 1024 features. Two small dense layers with dropout then give one sigmoid output.</>}>
          <Diagram aspect={narrow ? "6 / 7.4" : "6 / 6.4"} draw={c => drawMeso(c)}
            label="Meso-4 architecture: input 256 by 256 by 3; convolution 8 of 3 by 3 then pool to 128 by 128 by 8; convolution 8 of 5 by 5 then pool to 64 by 64 by 8; convolution 16 of 5 by 5 then pool to 32 by 32 by 16; convolution 16 of 5 by 5 then pool 4 by 4 to 8 by 8 by 16; flatten to 1024; dropout and dense 16; dropout and dense 1 with sigmoid." />
        </Figure>
        <p>The thesis built it in Keras: Conv2D layers, BatchNormalization, Flatten, then a fully connected part with
          Dropout against overfitting and a Leaky ReLU with slope 0.01. If convolutions are new to you,
          the <a href="#cnn">convolutional nets</a> page builds them from scratch, and <a href="#neural-networks">neural networks</a> covers
          the dense layers and gradient descent.</p>
      </Section>

      <Section n={4} title={<>A detector you can <b>TRAIN</b>.</>}>
        <p className="lead">A toy, not the thesis model.</p>
        <p>You cannot ship DFGC-21 or a GPU to a web page. What does fit is the core idea: a convolution can learn a trace
          that people cannot see. Bend draws 16×16 synthetic faces from a hash (background, skin, eyes, mouth, sensor grain).
          Every second face is “fake” and carries a faint checkerboard inside the face. Checkerboards are a well-known artifact
          of the upsampling layers in image generators, and seams and mismatched noise are similar traces. This choice is
          mine, not the thesis’s.</p>
        <p>The detector is Meso-4 shrunk to a single block: four 3×3 convolution kernels, ReLU, global average pooling, and one
          logistic neuron. That makes 45 parameters, all trained in Bend.</p>
        <Lab narrow={narrow} call={call} error={error} />
        <p>Each kernel is a function folded over the 196 windows of a face. The pass that computes the ReLU also gathers what
          the gradient needs, so the backward pass costs nothing extra:</p>
        <Code file="demos/deepfake/meso.bend">{scanCode}</Code>
        <p>The batch is a balanced tree, and the gradient of a sum is the sum of the gradients. In Bend’s parallel-call
          notation, <code>a b = f(x) g(y)</code> says the two calls are independent, so the two halves of the batch can run on
          different cores, or on the GPU in a native build. The JavaScript target used in your browser runs them one after the
          other.</p>
        <Code file="demos/deepfake/meso.bend">{gradCode}</Code>
        <p>Try the strength slider. At 0.05, held-out accuracy climbs well above 90%. At 0.02 the network struggles. At 0 there is
          nothing to learn and it stays at chance. That is the thesis’s finding about generation methods in miniature: a detector
          learns the traces of the fakes it was shown.</p>
      </Section>

      <Section n={5} title={<>What the thesis <b>FOUND</b>.</>}>
        <p>Every model was trained on the same faces and judged by accuracy, loss, ROC curve and AUC, precision, recall and
          F-score.</p>
        <p><b>MesoNet as published</b> peaked at 83% accuracy in training and 82% in test (p. 32). Averaged over the
          epochs, however, validation accuracy was only 69% (Table 2). Its ROC curve gave an AUC of 0.87.</p>
        <p><b>Pre-trained networks alone</b> failed. EfficientNetB0, EfficientNetB7 and VGG19 with ImageNet weights scored an
          AUC of 0.50, no better than a coin (pp. 34–35). The first attempts to bolt them onto MesoNet reached validation
          accuracies of only about 40% (p. 30).</p>
        <p><b>Going deeper</b> helped. <b>Eilhart</b> added convolution layers, activations between them and wider dense layers,
          and was trained with early stopping. Its AUC rose from 0.87 to 0.90 (p. 35). <b>Vigo</b> stacked further convolutions
          with dense layers between them (pp. 36–37).</p>
        <p><b>Transfer learning</b> came last. Vigo with EfficientNetB7 in front became <b>Vengerberg</b>. Vigo with VGG19
          became <b>Merigold</b> (p. 37).</p>
        <Bars title="Validation accuracy, mean over epochs" max={1} data={[
          { label: 'MesoNet', value: .691991, display: '69.2%' },
          { label: 'Eilhart', value: .710437, display: '71.0%' },
          { label: 'Merigold', value: .839086, display: '83.9%' },
          { label: 'Vengerberg', value: .863801, display: '86.4%', hi: true },
        ]} caption="Mean of the per-epoch validation accuracy: Tables 2, 3, 4 and 5 (pp. 33, 36, 37, 38)." />
        <Bars title="Area under the ROC curve" max={1} data={[
          { label: 'pre-trained alone', value: .5, display: '0.50' },
          { label: 'MesoNet', value: .87, display: '0.87' },
          { label: 'Eilhart', value: .90, display: '0.90' },
          { label: 'Merigold', value: .95, display: '0.95' },
          { label: 'Vengerberg', value: .97, display: '0.97', hi: true },
        ]} caption="AUC from the ROC figures: 13 (EfficientNetB0, B7 and VGG19 alone), 11, 15 and 18. 0.5 is chance, 1 is perfect." />
        <div style={{ overflowX: 'auto' }}>
          <table>
            <thead><tr><th>model</th><th>built from</th><th>val. acc. mean</th><th>max</th><th>AUC</th></tr></thead>
            <tbody>
              <tr><td>MesoNet</td><td>Meso-4</td><td>69.2%</td><td>82.1%</td><td>0.87</td></tr>
              <tr><td>Eilhart</td><td>MesoNet, deeper</td><td>71.0%</td><td>79.2%</td><td>0.90</td></tr>
              <tr><td>Merigold</td><td>Vigo + VGG19</td><td>83.9%</td><td>87.5%</td><td>0.95</td></tr>
              <tr className="hi"><td>Vengerberg</td><td>Vigo + EfficientNetB7</td><td>86.4%</td><td>92.6%</td><td>0.97</td></tr>
            </tbody>
          </table>
        </div>
        <p>The two transfer models differ in which mistakes they make. Averaged over the epochs, on validation
          (fake is the positive class):</p>
        <table>
          <thead><tr><th>model</th><th>TP</th><th>TN</th><th>FP</th><th>FN</th></tr></thead>
          <tbody>
            <tr><td>Merigold</td><td>681.1</td><td>822.6</td><td>95.4</td><td>192.9</td></tr>
            <tr className="hi"><td>Vengerberg</td><td>755.0</td><td>793.0</td><td>125.1</td><td>119.0</td></tr>
          </tbody>
        </table>
        <table>
          <thead><tr><th>model</th><th>precision</th><th>recall</th><th>F-score</th></tr></thead>
          <tbody>
            <tr><td>Merigold</td><td>0.877</td><td>0.779</td><td>0.824</td></tr>
            <tr className="hi"><td>Vengerberg</td><td>0.857</td><td>0.864</td><td>0.860</td></tr>
          </tbody>
        </table>
        <p>Merigold is more precise: when it says “fake”, it is right more often. Vengerberg misses fewer fakes, and the
          harmonic mean of the two, the F-score, favours it, 0.860 against 0.824 (p. 40). The thesis adds that F-score is one
          metric among several, and that which error matters more depends on the use.</p>
        <p className="note">Counts are the validation means of Tables 4 and 5. Table 6 repeats them but prints Merigold’s true
          negatives as 22.57, apparently a typo for the 822.57 of Table 4. The conclusion (p. 41) reports the two models as
          “MesoNet Custom 512” with EfficientNet and with VGG19, at F-scores of 86.44 and 72.38 and accuracies of 86.3% and 83.9%.</p>
        <p>According to the abstract and the conclusion, what moves these numbers is the <b>generation method</b> behind the fakes,
          the <b>quality and resolution</b> of the source media, and the <b>size of the training set</b>. Hardware matters too:
          getting close to 100% would likely need far heavier networks than the author could train.</p>
      </Section>

      <Section n={6} title={<>Accuracy is not <b>TRUST</b>.</>}>
        <p>The conclusion is careful. High accuracy on held-out data does not make a detector reliable in every situation. The
          data, though large, covers a small part of the deepfakes found in the wild, so generalising to unseen methods
          remains an open question.</p>
        <p>The thesis cites Facebook’s Deepfake Detection Challenge as an example. The best model on the public data reached an
          average precision of 82.56%. Against the private black-box set the ranking changed, and the new winner, by Selim
          Seferbekov, scored 65.18%.</p>
        <p>Then there are <b>adversarial attacks</b>. A model assumes test data looks like training data, and an attacker who
          knows that can break it. Small crafted noise turns a panda into a “gibbon” (Goodfellow et al., 2014). A printed
          sweater hides a person from the YOLOv2 detector (Wu et al., 2019). Several DFGC-21 subsets already apply such noise
          to their fakes.</p>
        <p>The toy above shows the same limit. Its best kernel is a checkerboard detector, so a generator that leaves no
          checkerboard would walk straight past it. For future work, the thesis points to diffusion models (Ho et al., 2020),
          which produce higher-quality images than GANs, and to the first detectors built for them (Corvi et al., 2022;
          Coccomini et al., 2023).</p>
        <p className="lead">Read the full thesis (in Portuguese) at the <a href={THESIS}>UFC repository</a>.</p>
      </Section>
    </DocPage>
  )
}
