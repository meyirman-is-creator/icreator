import os
from dotenv import load_dotenv

# Загрузка переменных окружения
load_dotenv()

# Настройки базы данных
DATABASE_URL = os.getenv("DATABASE_URL", "postgresql://meirman_is_creator:password@localhost/icreator")

# Настройки моделей - используем открытые модели без ограничений доступа
CONTENT_MODEL = os.getenv("CONTENT_MODEL", "microsoft/phi-2")  # Открытая модель для контента
CODE_MODEL = os.getenv("CODE_MODEL", "Xenova/distilgpt2")  # Полностью открытая модель для кода

# Настройки кэширования
CACHE_DIR = os.getenv("CACHE_DIR", "/app/model_cache")

# Отключаем квантизацию для CPU
QUANTIZATION = "none"