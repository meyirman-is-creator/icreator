#!/bin/bash

# Цвета для вывода
GREEN='\033[0;32m'
RED='\033[0;31m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Функция для вывода сообщений
echo_message() {
    echo -e "${BLUE}[ICreator]${NC} $1"
}

# Функция для вывода сообщений об успехе
echo_success() {
    echo -e "${GREEN}[УСПЕХ]${NC} $1"
}

# Функция для вывода сообщений об ошибке
echo_error() {
    echo -e "${RED}[ОШИБКА]${NC} $1"
}

# Функция для вывода предупреждений
echo_warning() {
    echo -e "${YELLOW}[ПРЕДУПРЕЖДЕНИЕ]${NC} $1"
}

# Проверка наличия Docker и Docker Compose
echo_message "Проверка наличия Docker и Docker Compose..."

if ! command -v docker &> /dev/null; then
    echo_error "Docker не установлен. Пожалуйста, установите Docker и попробуйте снова."
    exit 1
fi

if ! command -v docker-compose &> /dev/null; then
    echo_warning "Docker Compose не найден, пробуем использовать 'docker compose'..."
    if ! docker compose version &> /dev/null; then
        echo_error "Docker Compose не установлен. Пожалуйста, установите Docker Compose и попробуйте снова."
        exit 1
    else
        docker_compose_cmd="docker compose"
    fi
else
    docker_compose_cmd="docker-compose"
fi

echo_success "Docker и Docker Compose установлены!"

# Проверка наличия .env файла
if [ ! -f .env ]; then
    echo_message "Файл .env не найден. Создаем из шаблона..."
    cat > .env << EOL
# Настройки базы данных
DATABASE_URL=postgresql://meirman_is_creator:password@db/icreator

# Настройки моделей
CONTENT_MODEL=microsoft/phi-2
CODE_MODEL=nicholasKluge/TinyCodeLlama-1B-Python
QUANTIZATION=4bit
CACHE_DIR=/app/model_cache
EOL
    echo_success "Файл .env создан."
else
    echo_message "Файл .env найден."
fi

# Создание директории для кэша моделей, если она не существует
if [ ! -d "model_cache" ]; then
    echo_message "Создание директории для кэша моделей..."
    mkdir -p model_cache
    chmod -R 777 model_cache
    echo_success "Директория для кэша моделей создана."
else
    echo_message "Директория для кэша моделей уже существует."
fi

# Сборка и запуск контейнеров
echo_message "Сборка и запуск контейнеров Docker..."
$docker_compose_cmd up -d --build

# Проверка запущенных контейнеров
if [ $? -eq 0 ]; then
    echo_success "Контейнеры успешно запущены!"
    echo_message "API доступен по адресу: http://localhost:8000"
    echo_message "Документация API: http://localhost:8000/docs"

    # Проверка логов на ошибки
    echo_message "Проверка логов на наличие ошибок..."
    sleep 5
    $docker_compose_cmd logs --tail=100 web > logs.txt

    if grep -i "error\|exception" logs.txt > /dev/null; then
        echo_warning "В логах обнаружены ошибки. Проверьте логи для подробностей."
    else
        echo_success "Ошибок в логах не обнаружено."
    fi

    echo_message "Чтобы увидеть логи, выполните: $docker_compose_cmd logs -f"
    echo_message "Чтобы остановить сервисы, выполните: $docker_compose_cmd down"
else
    echo_error "Не удалось запустить контейнеры. Проверьте логи для подробностей."
    $docker_compose_cmd logs
    exit 1
fi