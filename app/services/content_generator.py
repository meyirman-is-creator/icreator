import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from app.config import CONTENT_MODEL


class ContentGenerator:
    def __init__(self):
        print(f"Загрузка модели для генерации контента: {CONTENT_MODEL}")
        self.tokenizer = AutoTokenizer.from_pretrained(CONTENT_MODEL)

        # Проверяем доступность GPU
        if torch.cuda.is_available():
            print("Используется GPU для генерации контента")
            self.model = AutoModelForCausalLM.from_pretrained(
                CONTENT_MODEL,
                torch_dtype=torch.float16,
                device_map="auto",
            )
        else:
            print("GPU недоступен, используется CPU (это будет медленнее)")
            self.model = AutoModelForCausalLM.from_pretrained(
                CONTENT_MODEL,
                device_map="auto",
                low_cpu_mem_usage=True,
            )
        print("Модель для генерации контента успешно загружена")

    def generate_slide_content(self, topic, slide_number, total_slides):
        """
        Генерирует контент для слайда на основе темы и номера слайда
        """
        # Создаём промпт в зависимости от номера слайда
        if slide_number == 1:
            prompt = f"Создай заголовок и краткое введение для презентации на тему '{topic}'."
        elif slide_number == total_slides:
            prompt = f"Создай заключительный слайд для презентации на тему '{topic}'."
        else:
            prompt = f"Создай контент для слайда {slide_number} из {total_slides} для презентации на тему '{topic}'."

        # Модифицированная генерация для разных моделей
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)
        outputs = self.model.generate(
            inputs["input_ids"],
            max_new_tokens=256,
            temperature=0.7,
            top_p=0.9,
            do_sample=True
        )

        # Декодируем сгенерированный текст
        generated_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)

        # Для моделей Phi и других, ответ часто включает промпт
        if prompt in generated_text:
            response = generated_text.replace(prompt, "").strip()
        else:
            response = generated_text.strip()

        return response

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