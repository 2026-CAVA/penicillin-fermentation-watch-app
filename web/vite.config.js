import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'

// /api/* -> FastAPI (python -m main serve; CAVA_API overrides the address) so the browser never needs it
export default defineConfig({
  plugins: [react()],
  server: {
    host: true,
    allowedHosts: ['.trycloudflare.com'], // cloudflared quick tunnel
    proxy: { '/api': { target: process.env.CAVA_API ?? 'http://127.0.0.1:8000', rewrite: (p) => p.replace(/^\/api/, '') } },
  },
})
