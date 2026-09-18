# Frontend image: Vite/React build served by nginx, reverse-proxying /api to the
# backend container over the compose network (no VITE_API_BASE_URL needed at build time).
#
# The builder stage is pinned to the native build platform (not the target platform) --
# esbuild's prebuilt binary crashes under QEMU cross-arch emulation, and the build output
# is architecture-independent static assets anyway, so there's nothing to gain from
# emulating the target arch here. Only the final nginx stage needs to match the target.

FROM --platform=$BUILDPLATFORM node:20-slim AS builder

WORKDIR /app

COPY frontend/package.json frontend/package-lock.json ./
RUN npm ci

COPY frontend/ ./
RUN npm run build


FROM nginx:1.27-alpine

COPY --from=builder /app/dist /usr/share/nginx/html
COPY frontend/nginx.conf /etc/nginx/conf.d/default.conf

EXPOSE 80
