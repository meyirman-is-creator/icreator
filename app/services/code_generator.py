import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import os
from app.config import CODE_MODEL, CACHE_DIR


class CodeGenerator:
    def __init__(self):
        print(f"Загрузка модели для генерации кода: {CODE_MODEL}")

        # Установка директории кэша
        os.environ["TRANSFORMERS_CACHE"] = CACHE_DIR

        try:
            # Загрузка токенизатора
            self.tokenizer = AutoTokenizer.from_pretrained(CODE_MODEL, trust_remote_code=True)

            # Загрузка модели
            if torch.cuda.is_available():
                print("Используется GPU для генерации кода")
                self.model = AutoModelForCausalLM.from_pretrained(
                    CODE_MODEL,
                    torch_dtype=torch.float16,
                    device_map="auto",
                    trust_remote_code=True,
                )
            else:
                print("GPU недоступен, используется CPU (это может быть медленно)")
                self.model = AutoModelForCausalLM.from_pretrained(
                    CODE_MODEL,
                    device_map="auto",
                    trust_remote_code=True,
                    low_cpu_mem_usage=True,
                )

            print("Модель для генерации кода успешно загружена")
            self.model_ready = True
        except Exception as e:
            print(f"Ошибка при загрузке модели для генерации кода: {e}")
            print("Будет использоваться заглушка вместо модели")
            self.model_ready = False

    def generate_frontend_code(self, slide_content, layout="single-column", theme="light"):
        """
        Генерирует React-код фронтенда для слайда
        """
        # Если модель не загружена, возвращаем заглушку
        if not hasattr(self, 'model_ready') or not self.model_ready:
            return self._get_fallback_code(slide_content, layout, theme)

        # Создаем более структурированный и подробный промпт для маленькой модели
        prompt = f"""
        Задача: Сгенерировать компонент React с TypeScript для слайда презентации.

        Содержание слайда:
        ---
        {slide_content}
        ---

        Спецификации:
        - Макет: {layout}
        - Тема: {theme}
        - Используй компоненты Shadcn UI
        - Код должен быть адаптивным (мобильные устройства и десктоп)
        - Используй современный функциональный стиль React с хуками

        Для простоты можешь использовать следующие шаблоны стилей:
        - light тема: белый фон, темный текст, акценты #0070f3
        - dark тема: темно-серый фон #121212, светлый текст, акценты #0070f3

        Верни только TSX код, без объяснений, начиная с import и заканчивая export default.
        """

        try:
            # Генерируем код
            inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)

            outputs = self.model.generate(
                inputs["input_ids"],
                max_new_tokens=1024,
                temperature=0.2,  # Низкая температура для более точных результатов
                top_p=0.95,
                do_sample=True,
                pad_token_id=self.tokenizer.eos_token_id
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
                    if "tsx" in block or "jsx" in block or "react" in block or "typescript" in block:
                        # Извлекаем код без идентификатора языка
                        code = block.replace("tsx", "").replace("jsx", "").replace("react", "").replace("typescript",
                                                                                                        "")
                        code = code.split("\n", 1)[1] if "\n" in code else code
                        return code.strip()
                # Если не найден конкретный блок React, возвращаем первый блок кода
                for block in code_blocks:
                    if block.strip() and block.strip() not in ["tsx", "jsx", "react", "typescript"]:
                        return block.strip()

            # Применяем дополнительную обработку для кода вне блоков кода
            # Ищем начало импортов и конец экспорта
            if "import " in code_part and "export default" in code_part:
                # Это уже выглядит как код, возвращаем как есть
                return code_part

            # В крайнем случае возвращаем весь сгенерированный код
            return code_part
        except Exception as e:
            print(f"Ошибка при генерации кода: {e}")
            return self._get_fallback_code(slide_content, layout, theme)

    def _get_fallback_code(self, slide_content, layout="single-column", theme="light"):
        """
        Возвращает заглушку кода, когда модель недоступна
        """
        # Простой шаблон React-компонента
        bg_color = "#ffffff" if theme == "light" else "#121212"
        text_color = "#333333" if theme == "light" else "#ffffff"
        accent_color = "#0070f3"

        # Преобразуем markdown-контент в простой HTML
        html_content = slide_content.replace("# ", "<h1>").replace("\n\n", "</h1><p>") + "</p>"

        # Создаем простой шаблон компонента
        return f"""import React from 'react';

interface SlideProps {{
  content?: string;
}}

const Slide: React.FC<SlideProps> = () => {{
  return (
    <div
      style={{
        backgroundColor: '{bg_color}',
        color: '{text_color}',
        padding: '2rem',
        borderRadius: '8px',
        boxShadow: '0 4px 6px rgba(0, 0, 0, 0.1)',
        maxWidth: '100%',
        margin: '0 auto',
      }}
    >
      <div dangerouslySetInnerHTML={{ __html: `{html_content}` }} />
    </div>
  );
}};

export default Slide;
"""