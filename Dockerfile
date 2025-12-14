# Multi-stage build для оптимизации размера образа

# Stage 1: Builder
FROM rust:1.75-slim AS builder

WORKDIR /app

# Установка зависимостей для сборки
RUN apt-get update && apt-get install -y \
    pkg-config \
    libssl-dev \
    && rm -rf /var/lib/apt/lists/*

# Копирование файлов зависимостей
COPY Cargo.toml ./

# Копирование исходного кода
COPY src ./src

# Сборка приложения
RUN cargo build --release

# Stage 2: Runtime
FROM debian:bookworm-slim

WORKDIR /app

# Установка только runtime зависимостей
RUN apt-get update && apt-get install -y \
    ca-certificates \
    libssl3 \
    wget \
    && rm -rf /var/lib/apt/lists/*

# Копирование бинарника из builder
COPY --from=builder /app/target/release/kimai-ml /app/kimai-ml

# Переменные окружения
ENV RUST_LOG=info
ENV RUST_BACKTRACE=1

# Порт
EXPOSE 8000

# Healthcheck (curl опционально, можно использовать wget или другой метод)
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
  CMD wget --no-verbose --tries=1 --spider http://localhost:8000/health || exit 1

# Запуск приложения
CMD ["./kimai-ml"]

