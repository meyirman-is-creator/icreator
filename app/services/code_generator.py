import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from app.config import CODE_MODEL


class CodeGenerator:
    def __init__(self):
        print(f"Загрузка модели для генерации кода: {CODE_MODEL}")
        self.tokenizer = AutoTokenizer.from_pretrained(CODE_MODEL)

        # Проверяем доступность GPU
        if torch.cuda.is_available():
            print("Используется GPU для генерации кода")
            self.model = AutoModelForCausalLM.from_pretrained(
                CODE_MODEL,
                torch_dtype=torch.float16,
                device_map="auto",
            )
        else:
            print("GPU недоступен, используется CPU (это будет медленнее)")
            self.model = AutoModelForCausalLM.from_pretrained(
                CODE_MODEL,
                device_map="auto",
                low_cpu_mem_usage=True,
            )
        print("Модель для генерации кода успешно загружена")

    def generate_frontend_code(self, slide_content, layout="single-column", theme="light"):
        """
        Генерирует React-код фронтенда для слайда
        """
        # Создаем промпт для генерации кода
        prompt = f"""
        Сгенерируй компонент React с TypeScript для слайда презентации со следующим содержанием:

        Содержание: {slide_content}
        Макет: {layout}
        Тема: {theme}

        Требования:
        - Используй компоненты Shadcn UI
        - Сделай его адаптивным для мобильных устройств и десктопа
        - Используй современные паттерны React
        - Верни только код, без объяснений
        """

        # Генерируем код
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)
        outputs = self.model.generate(
            inputs["input_ids"],
            max_new_tokens=1024,
            temperature=0.2,  # Более низкая температура для более детерминированных результатов
            top_p=0.95,
            do_sample=True
        )

        # Декодируем сгенерированный код
        generated_code = self.tokenizer.decode(outputs[0], skip_special_tokens=True)

        # Удаляем промпт из вывода, если он включен
        if prompt in generated_code:
            code_part = generated_code.replace(prompt, "").strip()
        else:
            code_part = generated_code.strip()

        # Извлекаем только код React-компонента (если он в блоке кода)
        if "```" in code_part:
            code_blocks = code_part.split("```")
            for block in code_blocks:
                if "tsx" in block or "jsx" in block or "react" in block:
                    # Извлекаем код без идентификатора языка
                    code = block.split("\n", 1)[1] if "\n" in block else block
                    return code
            # Если не найден конкретный блок React, возвращаем первый блок кода
            for block in code_blocks:
                if block.strip() and block.strip() not in ["tsx", "jsx", "react"]:
                    return block.strip()

        return code_part