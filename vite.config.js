import { defineConfig } from "vite";

export default defineConfig({
  // config options
  base: "/webgpu-particles/",
  esbuild: {
    supported: {
      'top-level-await': true //browsers can handle top-level-await features
    },
  }
});
