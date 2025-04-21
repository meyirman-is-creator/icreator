FROM python:3.10-slim

WORKDIR /app

# Установка системных зависимостей
RUN apt-get update && apt-get install -y \
    build-essential \
    git \
    libpq-dev \
    && rm -rf /var/lib/apt/lists/*

# Создаем директорию для кэша моделей
RUN mkdir -p /app/model_cache && \
    chmod -R 777 /app/model_cache

# Копирование файлов зависимостей
COPY requirements.txt .

# Устанавливаем зависимости - измененный порядок и подход для избежания конфликтов
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir 'numpy<2.0.0' && \
    # Устанавливаем основные зависимости без transformers
    pip install --no-cache-dir fastapi==0.104.1 \
                              uvicorn==0.24.0 \
                              pydantic==2.4.2 \
                              sqlalchemy==2.0.23 \
                              psycopg2-binary==2.9.9 \
                              python-dotenv==1.0.0 \
                              torch==2.1.0 \
                              sentencepiece==0.1.99 \
                              tqdm==4.66.1 \
                              accelerate==0.23.0 \
                              safetensors==0.4.0 \
                              einops==0.7.0 && \
    # Устанавливаем transformers и huggingface-hub в одной команде
    pip install --no-cache-dir transformers==4.36.0 "huggingface_hub>=0.19.3"

# Копирование исходного кода
COPY . .

# Создаем пользователя с ограниченными правами для запуска приложения
RUN groupadd -r appuser && useradd -r -g appuser appuser
RUN chown -R appuser:appuser /app

# Пробуем загрузить модели, но игнорируем ошибки при этом (добавляем || true)
RUN python -m app.utils.download_models || true

# Переключаемся на пользователя с ограниченными правами
USER appuser

# Запуск приложения
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]