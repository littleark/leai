import { defineConfig } from 'vite'
import { svelte } from '@sveltejs/vite-plugin-svelte'
import copy from 'rollup-plugin-copy';

// https://vite.dev/config/
export default defineConfig({
  plugins: [
    svelte(),
    copy({
      targets: [
        { src: 'src/404.html', dest: 'dist' }, // Copy 404.html to dist
      ],
      hook: 'writeBundle', // Ensure it runs at the end of the build process
    })
  ],
  build: {
    outDir: 'dist', // Ensure the output directory is set to 'dist'
    assetsDir: 'assets', // Ensure the assets directory is set to 'assets'
  },
  base: '/leai/', // Set the base URL to the subdirectory where your project is served
})
