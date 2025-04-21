import subprocess
import os
import sys


def setup_dependencies():
    """Устанавливает нужные зависимости с учетом ограничений"""
    # Устанавливаем NumPy версии 1.x для совместимости
    print("Установка NumPy версии 1.x...")
    subprocess.run([sys.executable, "-m", "pip", "install", "numpy<2.0.0", "--force-reinstall"], check=True)

    # Устанавливаем все остальные зависимости
    print("Установка основных зависимостей...")
    subprocess.run([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"], check=True)

    print("Зависимости успешно установлены!")


if __name__ == "__main__":
    setup_dependencies()