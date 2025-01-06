import { defineConfig } from 'vite'
import { svelte } from '@sveltejs/vite-plugin-svelte'

// https://vite.dev/config/
export default defineConfig({
  plugins: [svelte()],
  build: {
    outDir: 'dist', // Ensure the output directory is set to 'dist'
    assetsDir: 'assets', // Ensure the assets directory is set to 'assets'
  },
  base: '/leai/', // Set the base URL to the subdirectory where your project is served
})
