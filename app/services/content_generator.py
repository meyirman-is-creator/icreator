import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import os
from app.config import CONTENT_MODEL, CACHE_DIR


class ContentGenerator:
    def __init__(self):
        print(f"Загрузка модели для генерации контента: {CONTENT_MODEL}")

        # Установка директории кэша
        os.environ["TRANSFORMERS_CACHE"] = CACHE_DIR

        try:
            # Загрузка токенизатора
            self.tokenizer = AutoTokenizer.from_pretrained(CONTENT_MODEL, trust_remote_code=True)

            # Загрузка модели
            if torch.cuda.is_available():
                print("Используется GPU для генерации контента")
                self.model = AutoModelForCausalLM.from_pretrained(
                    CONTENT_MODEL,
                    torch_dtype=torch.float16,
                    device_map="auto",
                    trust_remote_code=True,
                )
            else:
                print("GPU недоступен, используется CPU (это может быть медленно)")
                self.model = AutoModelForCausalLM.from_pretrained(
                    CONTENT_MODEL,
                    device_map="auto",
                    trust_remote_code=True,
                    low_cpu_mem_usage=True,
                )

            print("Модель для генерации контента успешно загружена")
            self.model_ready = True
        except Exception as e:
            print(f"Ошибка при загрузке модели для генерации контента: {e}")
            print("Будет использоваться заглушка вместо модели")
            self.model_ready = False

    def generate_slide_content(self, topic, slide_number, total_slides):
        """
        Генерирует контент для слайда на основе темы и номера слайда
        """
        # Если модель не загружена, возвращаем заглушку
        if not hasattr(self, 'model_ready') or not self.model_ready:
            return self._get_fallback_content(topic, slide_number, total_slides)

        # Создаём промпт в зависимости от номера слайда
        if slide_number == 1:
            prompt = f"""Ты опытный создатель презентаций. 
            Задача: Создай заголовок и краткое введение для презентации на тему '{topic}'.
            Сделай текст кратким, информативным и привлекающим внимание.
            """
        elif slide_number == total_slides:
            prompt = f"""Ты опытный создатель презентаций. 
            Задача: Создай заключительный слайд для презентации на тему '{topic}'.
            Включи краткое резюме и выводы по теме.
            """
        else:
            # Определяем тип слайда на основе номера
            slide_type = self._get_slide_type(slide_number, total_slides)
            prompt = f"""Ты опытный создатель презентаций. 
            Задача: Создай контент для слайда {slide_number} из {total_slides} по теме '{topic}'.
            Тип слайда: {slide_type}
            Сделай текст кратким, информативным и хорошо структурированным.
            """

        try:
            # Генерация контента
            inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)

            # Параметры генерации
            outputs = self.model.generate(
                inputs["input_ids"],
                max_new_tokens=256,
                temperature=0.7,
                top_p=0.9,
                do_sample=True,
                pad_token_id=self.tokenizer.eos_token_id
            )

            # Декодируем сгенерированный текст
            generated_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)

            # Удаляем промпт из ответа, если он там есть
            if prompt in generated_text:
                response = generated_text.replace(prompt, "").strip()
            else:
                response = generated_text.strip()

            return response
        except Exception as e:
            print(f"Ошибка при генерации контента: {e}")
            return self._get_fallback_content(topic, slide_number, total_slides)

    def _get_slide_type(self, slide_number, total_slides):
        """
        Определяет тип слайда на основе его номера и общего количества слайдов
        """
        # Первая треть презентации - информационные слайды
        if slide_number < total_slides // 3:
            return "Информационный слайд с ключевыми фактами и концепциями"

        # Вторая треть - аналитические слайды
        elif slide_number < 2 * (total_slides // 3):
            return "Аналитический слайд с примерами или сравнениями"

        # Последняя треть - выводы и рекомендации
        else:
            return "Слайд с выводами или рекомендациями"

    def _get_fallback_content(self, topic, slide_number, total_slides):
        """
        Возвращает заглушку контента, когда модель недоступна
        """
        if slide_number == 1:
            return f"# {topic}\n\nВведение в тему: важные аспекты и ключевые понятия"
        elif slide_number == total_slides:
            return f"# Заключение\n\nОсновные выводы по теме '{topic}'"
        else:
            return f"# Слайд {slide_number}\n\nКлючевая информация по теме '{topic}'"

    def generate_all_slides(self, topic, slides_count):
        """
        Генерирует контент для всех слайдов
        """
        slides_content = []

        for i in range(1, slides_count + 1):
            content = self.generate_slide_content(topic, i, slides_count)
            slides_content.append({
                "slide_number": i,
                "content": content
            })

        return slides_content