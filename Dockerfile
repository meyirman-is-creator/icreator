FROM python:3.10-slim

WORKDIR /app

# Установка системных зависимостей
RUN apt-get update && apt-get install -y \
    build-essential \
    git \
    && rm -rf /var/lib/apt/lists/*

# Копирование файлов зависимостей
COPY requirements.txt .

# Устанавливаем transformers нужной версии
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir numpy<2.0.0 && \
    pip install --no-cache-dir transformers>=4.36.0 && \
    pip install --no-cache-dir -r requirements.txt

# Копирование исходного кода
COPY . .

# Предварительная загрузка моделей (опционально)
# RUN python -c "from transformers import AutoTokenizer; AutoTokenizer.from_pretrained('mistralai/Mistral-7B-Instruct-v0.2')"
# RUN python -c "from transformers import AutoTokenizer; AutoTokenizer.from_pretrained('deepseek-ai/deepseek-coder-6.7b-instruct')"

# Запуск приложения
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]