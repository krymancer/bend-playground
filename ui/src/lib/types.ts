export type DemoName = 'raytracer' | 'life' | 'cubes' | 'rubik'
export const DEMOS: DemoName[] = ['raytracer', 'life', 'cubes', 'rubik']

export interface OutputMeta {
  backend: string
  kernelMs: number
  wallMs: number
  createdAt: string
}
export interface RayOutput extends OutputMeta {
  width: number
  height: number
  bounces: number
  scene?: 'mirrors' | 'materials' | 'weekend'
  samples?: number
  seed?: number
}
export interface LifeOutput extends OutputMeta {
  size: number
  steps: number
  seed: number
  pattern: string
  encoding?: 'sparse-words'
  frames: number[][]
}
export interface CubeFrame { count: number; x: number; y: number; v: number; w: number; u: number; time: number; wall: boolean }
export interface CubesOutput extends OutputMeta {
  digits: number
  total: string
  pi: string
  sampled: boolean
  frames: CubeFrame[]
  motion?: CubeFrame[]
}
