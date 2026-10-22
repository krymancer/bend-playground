import fs from 'node:fs'
import path from 'node:path'
import { defineConfig, type Plugin } from 'vite'
import react from '@vitejs/plugin-react'
import tailwindcss from '@tailwindcss/vite'
// @ts-expect-error plain ESM shared with the Node scripts
import { engines } from '../scripts/labs.mjs'

const backend = `http://127.0.0.1:${process.env.PORT || 3000}`
const built = path.resolve(import.meta.dirname, '../build')

// In dev, serve Bend-compiled engines straight from build/ (run `npm run build:labs`),
// so browser labs work without the local server.
const bendEngines = (): Plugin => ({
  name: 'bend-engines',
  configureServer(server) {
    server.middlewares.use((req, res, next) => {
      const name = /^\/([\w-]+)-engine\.js$/.exec(req.url?.split('?')[0] ?? '')?.[1]
      const file = name && (engines as string[]).includes(name) && path.join(built, `${name}-engine.js`)
      if (!file || !fs.existsSync(file)) return next()
      res.setHeader('Content-Type', 'text/javascript'); res.setHeader('Cache-Control', 'no-store')
      res.end(fs.readFileSync(file))
    })
  },
})

export default defineConfig({
  base: process.env.VITE_STATIC === 'true' ? process.env.PAGES_BASE : '/',
  plugins: [bendEngines(), react(), tailwindcss()],
  resolve: { alias: { '@': path.resolve(import.meta.dirname, './src') } },
  build: { outDir: process.env.VITE_STATIC === 'true' ? '../build/pages' : '../build/ui', emptyOutDir: true },
  server: { proxy: { '/api': backend, '/output': backend } },
})
