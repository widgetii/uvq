import { defineConfig } from "vite";

export default defineConfig({
  optimizeDeps: { exclude: ["onnxruntime-web"] },
  build: { target: "esnext" },
  server: {
    headers: {
      "Cross-Origin-Embedder-Policy": "require-corp",
      "Cross-Origin-Opener-Policy": "same-origin",
    },
  },
});
