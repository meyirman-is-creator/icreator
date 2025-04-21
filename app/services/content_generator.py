import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import os
from app.config import CONTENT_MODEL, CACHE_DIR
import random


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

        # Определяем структуру и тип слайда
        slide_type, slide_structure = self._get_slide_structure(slide_number, total_slides)

        # Создаём детализированный промпт с конкретными инструкциями
        prompt = self._create_detailed_prompt(topic, slide_number, total_slides, slide_type, slide_structure)

        try:
            # Генерация контента с улучшенными параметрами
            inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)

            # Параметры генерации для более разнообразного текста
            outputs = self.model.generate(
                inputs["input_ids"],
                max_new_tokens=512,  # Увеличиваем максимальную длину для более подробного контента
                temperature=0.75,  # Слегка увеличиваем для разнообразия
                top_p=0.92,
                top_k=50,
                do_sample=True,
                repetition_penalty=1.15,  # Уменьшаем повторения
                pad_token_id=self.tokenizer.eos_token_id
            )

            # Декодируем сгенерированный текст
            generated_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)

            # Удаляем промпт из ответа, если он там есть
            if prompt in generated_text:
                response = generated_text.replace(prompt, "").strip()
            else:
                response = generated_text.strip()

            # Пост-обработка контента для улучшения форматирования и структуры
            formatted_response = self._post_process_content(response, slide_type, slide_number)

            return formatted_response
        except Exception as e:
            print(f"Ошибка при генерации контента: {e}")
            return self._get_fallback_content(topic, slide_number, total_slides)

    def _create_detailed_prompt(self, topic, slide_number, total_slides, slide_type, slide_structure):
        """
        Создает детальный промпт для модели на основе типа слайда и его структуры
        """
        # Базовые инструкции для всех типов слайдов
        base_instructions = f"""You are an expert presentation creator with deep knowledge on various topics. 
        Your task is to write engaging, informative content for a presentation slide in RUSSIAN language.

        Topic of the presentation: '{topic}'
        Slide number: {slide_number} out of {total_slides}
        Slide type: {slide_type}

        Guidelines:
        - Write only in Russian language
        - Be concise but informative
        - Use markdown formatting (use # for titles, ## for subtitles, * for bullet points)
        - Make the content engaging and thought-provoking
        - Focus on quality facts and avoid generic statements
        - Format the output as a properly structured markdown slide
        - Each slide should have a clear purpose and message
        """

        # Специфические инструкции в зависимости от типа слайда
        if slide_number == 1:
            specific_instructions = f"""
            This is the TITLE SLIDE. Create a compelling title and a brief introduction that sets the tone.

            Required structure:
            - A catchy main title (using # heading)
            - A subtitle that explains the presentation's purpose (using ## heading)
            - A brief 1-2 sentence introduction

            Make the title slide capture attention and clearly frame '{topic}' in an intriguing way.
            """

        elif slide_type == "Определение/Концепция":
            specific_instructions = f"""
            This slide should define key concepts related to '{topic}'.

            Required structure:
            - A clear title related to definitions or concepts
            - 2-3 key definitions presented in a structured way
            - Each definition should be concise but comprehensive

            Focus on fundamental concepts the audience needs to understand '{topic}'.
            """

        elif slide_type == "Историческая справка":
            specific_instructions = f"""
            This slide should provide historical context or background about '{topic}'.

            Required structure:
            - A title about the history or origins
            - 3-4 key historical points or timeline elements
            - Brief explanation of how these historical elements impact our understanding today

            Present the most relevant historical context that helps understand '{topic}' better.
            """

        elif slide_type == "Статистика/Данные":
            specific_instructions = f"""
            This slide should present important statistics or data related to '{topic}'.

            Required structure:
            - A title highlighting the data aspect you're focusing on
            - 3-5 statistical points or data insights
            - A brief statement about what these statistics reveal

            Present compelling data that illuminates important aspects of '{topic}'.
            """

        elif slide_type == "Сравнение/Анализ":
            specific_instructions = f"""
            This slide should compare different aspects, approaches, or perspectives on '{topic}'.

            Required structure:
            - A title framing the comparison
            - Clear sections for each element being compared (at least 2)
            - Key differences and similarities highlighted

            Make sure the comparison provides insight and isn't just listing differences.
            """

        elif slide_type == "Пример/Кейс":
            specific_instructions = f"""
            This slide should present a concrete example or case study related to '{topic}'.

            Required structure:
            - A title introducing the example/case
            - Brief background of the example
            - Key points or lessons from this example
            - How this example illustrates important aspects of '{topic}'

            Choose an example that is relevant and illuminating for the audience.
            """

        elif slide_type == "Практическое применение":
            specific_instructions = f"""
            This slide should cover practical applications or implications of '{topic}'.

            Required structure:
            - A title focusing on applications or practical relevance
            - 3-4 practical ways '{topic}' applies in real life
            - Brief explanation of each application

            Focus on how the audience can apply or see '{topic}' in practice.
            """

        elif slide_type == "Вызовы/Проблемы":
            specific_instructions = f"""
            This slide should address challenges, problems, or limitations related to '{topic}'.

            Required structure:
            - A title framing the challenges
            - 3-4 specific challenges listed with brief explanations
            - Optional: hints at potential solutions

            Be honest about difficulties while maintaining a constructive tone.
            """

        elif slide_type == "Будущие тенденции":
            specific_instructions = f"""
            This slide should discuss future trends, developments, or predictions about '{topic}'.

            Required structure:
            - A forward-looking title
            - 3-4 potential future developments
            - Brief rationale for why these trends are likely or important

            Balance realistic predictions with thought-provoking possibilities.
            """

        elif slide_number == total_slides:
            specific_instructions = f"""
            This is the CONCLUDING SLIDE. Summarize key points and provide a memorable ending.

            Required structure:
            - A conclusion title
            - 3-5 key takeaways from the entire presentation
            - A thought-provoking final statement or call to action

            Make the conclusion reinforce the most important aspects of '{topic}' and leave the audience with something to reflect on.
            """

        else:
            specific_instructions = f"""
            This slide should provide key insights on '{topic}' focusing on {slide_type}.

            Required structure:
            - A clear title that relates to an aspect of '{topic}'
            - 3-5 key points organized in a logical structure
            - Each point should be substantive and specific, not generic

            Make sure this slide builds on previous slides and contributes unique content to the presentation.
            """

        # Добавление структуры слайда к инструкциям
        structure_instructions = f"""
        Structure the content in this format:
        {slide_structure}

        IMPORTANT: Output only the final slide content in Russian, no explanations or translations.
        """

        # Объединение всех инструкций
        complete_prompt = f"{base_instructions}\n\n{specific_instructions}\n\n{structure_instructions}"
        return complete_prompt

    def _get_slide_structure(self, slide_number, total_slides):
        """
        Определяет тип и структуру слайда на основе его номера и общего количества слайдов
        """
        # Определяем тип слайда в зависимости от его позиции
        if slide_number == 1:
            slide_type = "Титульный слайд"
            slide_structure = """
            # [Заголовок презентации]
            ## [Подзаголовок или тема]

            [Краткое введение в 1-2 предложения]
            """

        elif slide_number == total_slides:
            slide_type = "Заключение"
            slide_structure = """
            # Заключение

            * [Ключевой вывод 1]
            * [Ключевой вывод 2]
            * [Ключевой вывод 3]

            ## [Итоговое утверждение или призыв к действию]
            """

        else:
            # Для слайдов в середине презентации определяем тип на основе позиции
            middle_slides_types = [
                "Определение/Концепция",
                "Историческая справка",
                "Статистика/Данные",
                "Сравнение/Анализ",
                "Пример/Кейс",
                "Практическое применение",
                "Вызовы/Проблемы",
                "Будущие тенденции"
            ]

            # Определяем индекс типа с учетом позиции слайда, чтобы слайды шли в логичном порядке
            position_index = (slide_number - 2) % len(middle_slides_types)
            slide_type = middle_slides_types[position_index]

            # Разные структуры для разных типов слайдов
            if slide_type == "Определение/Концепция":
                slide_structure = """
                # [Заголовок о ключевых понятиях]

                ## [Концепция 1]
                [Определение и пояснение]

                ## [Концепция 2]
                [Определение и пояснение]

                ## [Концепция 3 (опционально)]
                [Определение и пояснение]
                """

            elif slide_type == "Историческая справка":
                slide_structure = """
                # [Заголовок об историческом контексте]

                * [Исторический факт/период 1] - [краткое описание]
                * [Исторический факт/период 2] - [краткое описание]
                * [Исторический факт/период 3] - [краткое описание]

                ## [Историческая значимость для понимания темы]
                """

            elif slide_type == "Статистика/Данные":
                slide_structure = """
                # [Заголовок о статистических данных]

                ## Ключевые цифры:
                * [Статистический факт 1]
                * [Статистический факт 2]
                * [Статистический факт 3]
                * [Статистический факт 4]

                ### [Вывод на основе данных]
                """

            elif slide_type == "Сравнение/Анализ":
                slide_structure = """
                # [Заголовок о сравнении/анализе]

                ## [Первый элемент сравнения]:
                * [Ключевая характеристика 1]
                * [Ключевая характеристика 2]

                ## [Второй элемент сравнения]:
                * [Ключевая характеристика 1]
                * [Ключевая характеристика 2]

                ### [Ключевой вывод из сравнения]
                """

            elif slide_type == "Пример/Кейс":
                slide_structure = """
                # [Заголовок о конкретном примере]

                ## Описание:
                [Краткое описание примера/кейса]

                ## Ключевые аспекты:
                * [Аспект 1]
                * [Аспект 2]
                * [Аспект 3]

                ## Значимость для темы:
                [Почему этот пример важен]
                """

            elif slide_type == "Практическое применение":
                slide_structure = """
                # [Заголовок о практическом применении]

                ## Как это применяется:
                1. [Практическое применение 1] - [краткое пояснение]
                2. [Практическое применение 2] - [краткое пояснение]
                3. [Практическое применение 3] - [краткое пояснение]

                ### [Практический вывод или рекомендация]
                """

            elif slide_type == "Вызовы/Проблемы":
                slide_structure = """
                # [Заголовок о вызовах/проблемах]

                ## Основные вызовы:
                * [Проблема 1] - [краткое описание]
                * [Проблема 2] - [краткое описание]
                * [Проблема 3] - [краткое описание]

                ## [Возможные подходы к решению]
                """

            elif slide_type == "Будущие тенденции":
                slide_structure = """
                # [Заголовок о будущем развитии]

                ## Ожидаемые тенденции:
                1. [Тенденция 1] - [обоснование и значение]
                2. [Тенденция 2] - [обоснование и значение]
                3. [Тенденция 3] - [обоснование и значение]

                ### [Перспективы и значимость]
                """
            else:
                slide_structure = """
                # [Заголовок слайда]

                ## [Подзаголовок или ключевая мысль]

                * [Пункт 1]
                * [Пункт 2]
                * [Пункт 3]

                ### [Заключительная мысль или связь со следующим слайдом]
                """

        return slide_type, slide_structure

    def _post_process_content(self, content, slide_type, slide_number):
        """
        Пост-обработка сгенерированного контента для улучшения форматирования
        """
        # Удаляем лишние пробелы и переносы строк
        processed_content = content.strip()

        # Добавляем заголовок слайда, если его нет
        if not processed_content.startswith("#"):
            if slide_number == 1:
                processed_content = f"# Введение\n\n{processed_content}"
            elif slide_type == "Заключение":
                processed_content = f"# Заключение\n\n{processed_content}"
            else:
                processed_content = f"# {slide_type}\n\n{processed_content}"

        # Заменяем двойные переносы строк на одинарные для более компактного вида
        processed_content = processed_content.replace('\n\n\n\n', '\n\n')
        processed_content = processed_content.replace('\n\n\n', '\n\n')

        # Удаляем служебные комментарии в квадратных скобках, если они остались
        processed_content = processed_content.replace('[Заголовок презентации]', '')
        processed_content = processed_content.replace('[Подзаголовок или тема]', '')
        processed_content = processed_content.replace('[Заголовок слайда]', '')
        processed_content = processed_content.replace('[Заключение]', 'Заключение')

        return processed_content

    def _get_fallback_content(self, topic, slide_number, total_slides):
        """
        Возвращает заглушку контента с улучшенной структурой, когда модель недоступна
        """
        if slide_number == 1:
            return f"# {topic}\n\n## Введение в увлекательный мир этой темы\n\nРассмотрим ключевые аспекты и фундаментальные концепции, которые помогут нам глубже понять данную тему."

        elif slide_number == total_slides:
            return f"# Заключение\n\n* Мы изучили основные аспекты темы '{topic}'\n* Рассмотрели практическое применение и важные концепции\n* Определили ключевые вызовы и перспективы развития\n\n## Данная тема продолжает развиваться и открывает новые горизонты для исследования"

        elif slide_number == 2:
            return f"# Ключевые концепции\n\n## Определение и сущность\n* Основополагающие идеи темы '{topic}'\n* Фундаментальные принципы и терминология\n* Взаимосвязь с другими областями знаний"

        elif slide_number == 3:
            return f"# Историческое развитие\n\n* Истоки и предпосылки возникновения '{topic}'\n* Ключевые этапы эволюции концепции\n* Влияние исторического контекста на современное понимание"

        elif slide_number == 4:
            return f"# Современное состояние\n\n## Актуальные тенденции\n* Наиболее значимые современные подходы\n* Текущие дискуссии и противоречия\n* Междисциплинарные связи и влияния"

        elif slide_number == 5:
            return f"# Практическое применение\n\n1. Конкретные примеры реализации\n2. Методологические подходы и инструменты\n3. Критерии оценки эффективности\n4. Распространенные трудности и способы их преодоления"

        elif slide_number % 2 == 0:
            return f"# Аналитический взгляд\n\n## Сравнительный анализ\n* Различные подходы к пониманию '{topic}'\n* Преимущества и ограничения каждого подхода\n* Критерии для объективной оценки"

        else:
            return f"# Перспективы развития\n\n## Будущее темы '{topic}'\n* Потенциальные направления эволюции\n* Ожидаемые инновации и трансформации\n* Вызовы и возможности, которые нас ждут"

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