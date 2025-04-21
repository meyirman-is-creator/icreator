#!/bin/bash

# Устанавливаем переменные для подключения к PostgreSQL
POSTGRES_USER="meirman_is_creator"
POSTGRES_PASSWORD="password123"  # Пароль из docker-compose.yml
POSTGRES_DB="icreator"

# Файл SQL для создания таблиц
cat > create_tables.sql << EOL
-- Создание таблицы презентаций
CREATE TABLE IF NOT EXISTS presentation (
    id SERIAL PRIMARY KEY,
    topic VARCHAR(255) NOT NULL,
    slides_count INTEGER NOT NULL DEFAULT 7,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

-- Создание таблицы слайдов
CREATE TABLE IF NOT EXISTS slide (
    id SERIAL PRIMARY KEY,
    presentation_id INTEGER NOT NULL REFERENCES presentation(id) ON DELETE CASCADE,
    slide_number INTEGER NOT NULL,
    content TEXT NOT NULL,
    code TEXT NOT NULL,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    UNIQUE (presentation_id, slide_number)
);
EOL

# Запускаем команду создания базы данных и таблиц
PGPASSWORD=$POSTGRES_PASSWORD psql -h db -U postgres -c "CREATE DATABASE $POSTGRES_DB;"
PGPASSWORD=$POSTGRES_PASSWORD psql -h db -U postgres -c "CREATE USER $POSTGRES_USER WITH PASSWORD '$POSTGRES_PASSWORD';"
PGPASSWORD=$POSTGRES_PASSWORD psql -h db -U postgres -c "GRANT ALL PRIVILEGES ON DATABASE $POSTGRES_DB TO $POSTGRES_USER;"
PGPASSWORD=$POSTGRES_PASSWORD psql -h db -U postgres -d $POSTGRES_DB -c "GRANT ALL ON SCHEMA public TO $POSTGRES_USER;"

# Запускаем SQL-скрипт для создания таблиц
PGPASSWORD=$POSTGRES_PASSWORD psql -h db -U $POSTGRES_USER -d $POSTGRES_DB -f create_tables.sql

echo "База данных и таблицы успешно созданы!"