import os
import subprocess
import sys


def check_python_version():
    """Проверяет, что версия Python 3.8 или выше"""
    if sys.version_info < (3, 8):
        print("Требуется Python 3.8 или выше.")
        sys.exit(1)


def create_directory_structure():
    """Создает структуру директорий проекта"""
    directories = [
        "app",
        "app/models",
        "app/routers",
        "app/services",
        "app/utils"
    ]

    for directory in directories:
        os.makedirs(directory, exist_ok=True)
        init_file = os.path.join(directory, "__init__.py")
        if not os.path.exists(init_file):
            with open(init_file, "w") as f:
                f.write("# Initialize package\n")

    print("Структура директорий успешно создана.")


def create_env_file():
    """Создает файл .env, если он не существует"""
    if not os.path.exists(".env"):
        with open(".env", "w") as f:
            f.write("# Настройки базы данных\n")
            f.write("DATABASE_URL=postgresql://meirman_is_creator:your_password@localhost/icreator\n\n")
            f.write("# Настройки моделей\n")
            f.write("CONTENT_MODEL=mistralai/Mistral-7B-Instruct-v0.2\n")
            f.write("CODE_MODEL=deepseek-ai/deepseek-coder-6.7b-instruct\n")
        print("Файл .env создан. Пожалуйста, обновите его своими учетными данными для базы данных.")
    else:
        print("Файл .env уже существует.")


def setup_conda_environment():
    """Настраивает Conda окружение"""
    try:
        # Проверяем, установлена ли conda
        subprocess.run(["conda", "--version"], check=True, stdout=subprocess.PIPE)

        # Создаем conda окружение
        subprocess.run(["conda", "env", "create", "-f", "environment.yml"], check=True)

        print("Conda окружение 'presentation-generator' успешно создано.")
        print("Активируйте его с помощью: conda activate presentation-generator")
    except subprocess.CalledProcessError:
        print("Ошибка: Conda не установлена или создание окружения не удалось.")
    except FileNotFoundError:
        print("Ошибка: Conda не установлена или не находится в PATH.")


def main():
    """Основная функция настройки"""
    print("Настройка проекта генератора презентаций...")

    check_python_version()
    create_directory_structure()
    create_env_file()

    # Спрашиваем, хочет ли пользователь настроить Conda окружение
    setup_env = input("Хотите настроить Conda окружение сейчас? (y/n): ").strip().lower()
    if setup_env == "y":
        setup_conda_environment()

    print("\nНастройка завершена!")
    print("\nСледующие шаги:")
    print("1. Обновите файл .env своими учетными данными для базы данных")
    print("2. Активируйте Conda окружение: conda activate presentation-generator")
    print("3. Запустите приложение: python -m app.main")


if __name__ == "__main__":
    main()