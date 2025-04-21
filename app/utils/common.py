import time
from typing import Dict, Any


def format_response(status: str, data: Dict[str, Any] = None, message: str = None) -> Dict[str, Any]:
    """
    Форматирует стандартный ответ API
    """
    response = {"status": status}

    if data is not None:
        response["data"] = data

    if message is not None:
        response["message"] = message

    return response


def measure_execution_time(func):
    """
    Декоратор для измерения времени выполнения функции
    """

    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        print(f"Функция {func.__name__} выполнилась за {end_time - start_time:.2f} секунд")
        return result

    return wrapper