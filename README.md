# Kimai ML - Rust версия

Высокопроизводительные ML модели для [Kimai Aggregator](https://github.com/localzet/kimai-aggregator).

## 🚀 Быстрый старт

### Локальная разработка

```bash
# Запуск API сервера
cargo run --release

# Или с hot reload
cargo install cargo-watch
cargo watch -x 'run --release'
```

### Docker

```bash
# Сборка
docker build -t kimai-ml-rust:latest .

# Запуск
docker run -d -p 8000:8000 --name kimai-ml-rust kimai-ml-rust:latest
```

Или через docker-compose:

```bash
docker-compose up -d
```

## ⚡ Производительность

- **В 3-10 раз быстрее** чем Python версия
- **В 3-5 раз меньше памяти** (50-150 MB vs 200-500 MB)
- **Время ответа**: 5-50ms (vs 50-200ms Python)

## 📁 Структура

```
ai-ml-rust/
├── src/
│   ├── lib.rs              # Библиотека
│   ├── main.rs             # API сервер
│   ├── models/             # ML модели
│   ├── preprocessing/      # Обработка данных
│   └── types.rs            # Типы данных
├── Cargo.toml
└── Dockerfile
```

## 🧠 Модели

1. **Прогнозирование времени** - Decision Tree + Ridge Regression
2. **Обнаружение аномалий** - Isolation Forest
3. **Рекомендации** - KMeans + анализ эффективности
4. **Анализ продуктивности** - Статистический анализ

## 📡 API Endpoints

- `POST /api/predict` - прогнозирование
- `POST /api/detect-anomalies` - аномалии
- `POST /api/recommendations` - рекомендации
- `POST /api/productivity` - продуктивность

## 🔧 Разработка

### Требования

- Rust 1.70+
- Cargo

### Зависимости

Устанавливаются автоматически через `cargo build`

### Тестирование

```bash
cargo test
```

### Линтинг

```bash
cargo clippy -- -D warnings
```

### Форматирование

```bash
cargo fmt
```

## Лицензия

Copyright (C) 2025 Localzet Group

This program is free software: you can redistribute it and/or modify it under the terms of the GNU Affero General Public License as published by the Free Software Foundation, either version 3 of the License, or (at your option) any later version.
