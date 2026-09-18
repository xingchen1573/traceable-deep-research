import { defineConfig, loadEnv } from 'vite';
import react from '@vitejs/plugin-react';
import path from 'path';

export default defineConfig(({ mode }) => {
  const env = loadEnv(mode, process.cwd(), '');
  const isMock = mode === 'mock' || env.VITE_USE_MOCK === 'true';
  const devPort = Number(env.VITE_DEV_PORT || 24681);
  const apiProxyTarget = env.VITE_API_PROXY_TARGET || 'http://127.0.0.1:24680';

  return {
    plugins: [react()],
    resolve: {
      alias: {
        '@': path.resolve(__dirname, './src'),
      },
    },
    server: {
      port: devPort,
      strictPort: true,
      proxy: isMock
        ? {}
        : {
            '/api': {
              target: apiProxyTarget,
              changeOrigin: true,
            },
          },
    },
    build: {
      outDir: 'dist',
      sourcemap: true,
    },
  };
});
