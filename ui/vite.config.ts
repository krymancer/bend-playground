import path from 'node:path'
import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'
import tailwindcss from '@tailwindcss/vite'

const backend = `http://127.0.0.1:${process.env.PORT || 3000}`

export default defineConfig({
  plugins: [react(), tailwindcss()],
  resolve: { alias: { '@': path.resolve(import.meta.dirname, './src') } },
  build: { outDir: '../build/ui', emptyOutDir: true },
  server: {
    proxy: {
      '/api': backend,
      '/output': backend,
      '/rubik-engine.js': backend,
      '/times-table-engine.js': backend,
    },
  },
})
