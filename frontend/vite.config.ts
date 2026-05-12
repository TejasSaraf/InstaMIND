import path from 'path'
import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'
import tailwindcss from '@tailwindcss/vite'

export default defineConfig({
  plugins: [react(), tailwindcss()],
  resolve: {
    // Avoid duplicate React when libraries pull their own copy (hooks / context break).
    dedupe: ['react', 'react-dom'],
  },
  server: {
    host: 'localhost',
    port: 5173,
    strictPort: true,
    fs: {
      allow: [
        __dirname,
        path.resolve(__dirname, '..', 'data', 'videos'),
      ],
    },
  },
  publicDir: 'public',
})
