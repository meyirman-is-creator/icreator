import os
from dotenv import load_dotenv

# Загрузка переменных окружения
load_dotenv()

# Настройки базы данных
DATABASE_URL = os.getenv("DATABASE_URL", "postgresql://meirman_is_creator:your_password@localhost/icreator")

# Настройки моделей
CONTENT_MODEL = os.getenv("CONTENT_MODEL", "mistralai/Mistral-7B-Instruct-v0.2")
CODE_MODEL = os.getenv("CODE_MODEL", "deepseek-ai/deepseek-coder-6.7b-instruct")