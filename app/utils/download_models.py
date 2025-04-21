import os
import sys
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from huggingface_hub import hf_hub_download, snapshot_download
import time
from tqdm import tqdm

# Добавляем директорию проекта в sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from app.config import CONTENT_MODEL, CODE_MODEL, CACHE_DIR


def download_model(model_name, model_type="model"):
    """
    Загружает модель и токенизатор
    """
    print(f"Загрузка {model_type} модели: {model_name}")
    start_time = time.time()

    # Создаем директорию кэша, если она не существует
    os.makedirs(CACHE_DIR, exist_ok=True)

    try:
        # Загружаем токенизатор
        print(f"Загрузка токенизатора для {model_name}...")
        tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            cache_dir=CACHE_DIR,
            trust_remote_code=True
        )
        print(f"Токенизатор успешно загружен")

        # Используем snapshot_download для загрузки всех файлов модели
        print(f"Загрузка файлов модели {model_name}...")
        snapshot_download(
            repo_id=model_name,
            cache_dir=CACHE_DIR,
            local_files_only=False,
        )

        print(f"Файлы модели успешно загружены")

        # Проверяем, можем ли мы загрузить модель (без квантизации для CPU)
        print(f"Тестовая загрузка модели {model_name}...")

        # Определяем, использовать ли квантизацию - для CPU НЕ используем квантизацию
        if torch.cuda.is_available():
            print("Доступен GPU, загружаем модель с использованием float16")
            model = AutoModelForCausalLM.from_pretrained(
                model_name,
                torch_dtype=torch.float16,
                device_map="auto",
                cache_dir=CACHE_DIR,
                trust_remote_code=True,
            )
        else:
            # Для CPU не используем квантизацию, просто загружаем модель
            print("Загружаем модель для CPU (без квантизации)")
            model = AutoModelForCausalLM.from_pretrained(
                model_name,
                low_cpu_mem_usage=True,
                cache_dir=CACHE_DIR,
                trust_remote_code=True,
            )

        print(f"Тестовая загрузка модели успешно завершена")

        # Освобождаем память
        del model
        torch.cuda.empty_cache() if torch.cuda.is_available() else None

        end_time = time.time()
        print(f"Модель {model_name} успешно загружена за {end_time - start_time:.2f} секунд")
        return True
    except Exception as e:
        print(f"Ошибка при загрузке модели {model_name}: {e}")
        return False


def main():
    """
    Основная функция для загрузки всех моделей
    """
    print("=" * 50)
    print("Начинаем загрузку моделей")
    print("=" * 50)

    # Устанавливаем переменную окружения для кэша
    os.environ["TRANSFORMERS_CACHE"] = CACHE_DIR
    print(f"Директория кэша: {CACHE_DIR}")

    # Загружаем модель для генерации контента
    content_model_success = download_model(CONTENT_MODEL, "контент")

    # Загружаем модель для генерации кода
    code_model_success = download_model(CODE_MODEL, "код")

    # Выводим итоговую информацию
    print("\n" + "=" * 50)
    print("Результаты загрузки моделей:")
    print(f"Модель контента ({CONTENT_MODEL}): {'Успешно' if content_model_success else 'Ошибка'}")
    print(f"Модель кода ({CODE_MODEL}): {'Успешно' if code_model_success else 'Ошибка'}")
    print("=" * 50)

    # Возвращаем статус выполнения - ВАЖНО: меняем логику, чтобы не выходить с ошибкой
    # Если нам не удалось загрузить модели в контейнере, это не фатальная ошибка,
    # так как модели могут быть загружены позже при первом запросе
    print("Продолжаем запуск приложения...")
    return True


if __name__ == "__main__":
    success = main()
    # Всегда выходим с успешным кодом, чтобы Docker не останавливался
    sys.exit(0)