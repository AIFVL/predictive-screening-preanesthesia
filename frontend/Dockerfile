# ────────────────────────────────────────────────────────────────── builder
FROM node:20-alpine AS builder

WORKDIR /app

# Cache de dependencias.
COPY package.json package-lock.json* ./
RUN npm install --no-audit --no-fund

# Código fuente y build estático.
COPY . .

ARG VITE_API_URL=http://localhost:8000
ENV VITE_API_URL=${VITE_API_URL}

RUN npm run build

# ────────────────────────────────────────────────────────────────── runtime
FROM nginx:1.27-alpine AS runtime

COPY nginx.conf /etc/nginx/conf.d/default.conf

COPY --from=builder /app/dist /usr/share/nginx/html

EXPOSE 80

HEALTHCHECK --interval=30s --timeout=3s --retries=3 \
    CMD wget -qO- http://localhost/ >/dev/null 2>&1 || exit 1

CMD ["nginx", "-g", "daemon off;"]
