import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import os
import re
import random
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

    def generate_frontend_code(self, slide_content, layout="auto", theme="auto"):
        """
        Генерирует React-код фронтенда для слайда с улучшенным дизайном и анимациями
        """
        # Если модель не загружена, возвращаем заглушку
        if not hasattr(self, 'model_ready') or not self.model_ready:
            return self._get_template_code(slide_content, layout, theme)

        # Определение типа слайда на основе его содержимого
        slide_type = self._determine_slide_type(slide_content)

        # Автоматический выбор макета и темы, если не указаны
        if layout == "auto":
            layout = self._select_layout_for_slide(slide_type)

        if theme == "auto":
            theme = self._select_theme_for_slide(slide_type)

        # Создаем более структурированный и подробный промпт для модели
        prompt = self._create_code_generation_prompt(slide_content, slide_type, layout, theme)

        try:
            # Генерируем код
            inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)

            outputs = self.model.generate(
                inputs["input_ids"],
                max_new_tokens=2048,  # Увеличиваем для более сложного кода
                temperature=0.3,  # Низкая температура для более структурированного кода
                top_p=0.95,
                do_sample=True,
                repetition_penalty=1.2,
                pad_token_id=self.tokenizer.eos_token_id
            )

            # Декодируем сгенерированный код
            generated_code = self.tokenizer.decode(outputs[0], skip_special_tokens=True)

            # Удаляем промпт из вывода, если он включен
            if prompt in generated_code:
                code_part = generated_code.replace(prompt, "").strip()
            else:
                code_part = generated_code.strip()

            # Извлекаем только код React-компонента
            cleaned_code = self._extract_and_clean_code(code_part)

            # Проверяем корректность кода
            if self._is_valid_react_code(cleaned_code):
                return cleaned_code
            else:
                # Если код некорректный, используем шаблонный код
                return self._get_template_code(slide_content, layout, theme)

        except Exception as e:
            print(f"Ошибка при генерации кода: {e}")
            return self._get_template_code(slide_content, layout, theme)

    def _create_code_generation_prompt(self, slide_content, slide_type, layout, theme):
        """
        Создает детальный промпт для генерации кода
        """
        # Основная структура промпта
        prompt = f"""
        Generate a high-quality React TypeScript component for a presentation slide with animations and modern design.

        SLIDE CONTENT:
        ```
        {slide_content}
        ```

        SLIDE TYPE: {slide_type}
        LAYOUT TYPE: {layout}
        DESIGN THEME: {theme}

        REQUIREMENTS:
        1. Create a responsive React TypeScript component
        2. Use modern React (functional components, hooks)
        3. Include beautiful animations and transitions
        4. Make it visually stunning with appropriate styling
        5. Parse and render markdown content from the slide
        6. Include thoughtful micro-interactions and details
        7. Use proper TypeScript typings
        8. Follow best practices for React development
        9. Add appropriate comments to explain complex logic

        STYLING REQUIREMENTS:
        - For light theme: use a clean white base (#ffffff) with dark text (#333333) and accent color (#0070f3)
        - For dark theme: use dark background (#121212) with light text (#ffffff) and accent color (#0070f3)
        - For colorful theme: use gradient backgrounds, vibrant colors, and playful design elements
        - For minimal theme: use subtle colors, elegant typography, and minimalist design
        - For corporate theme: use professional blue tones, clean layout, and business-appropriate styling

        ANIMATION IDEAS:
        - Use subtle fade-in effects for text elements
        - Add sliding animations for lists and key points
        - Include emphasis animations for important content
        - Use CSS transforms and transitions for smooth effects
        - Consider reveal animations for sequential content

        LAYOUT OPTIONS:
        - For "centered" layout: center all content with balanced whitespace
        - For "two-column" layout: use left side for headings, right side for content
        - For "grid" layout: organize content in a responsive grid pattern
        - For "featured" layout: highlight key information with larger elements
        - For "timeline" layout: organize information in chronological sequence

        NOTES:
        - Parse markdown content properly (headings, lists, emphasis)
        - Ensure the component is self-contained
        - Use CSS-in-JS or inline styles for simplicity
        - Make sure animations are tasteful and enhance readability
        - Ensure accessibility for all users

        RETURN ONLY THE COMPLETE REACT TYPESCRIPT CODE without any explanation or markdown code blocks.
        Start with import statements and end with export default.
        """

        return prompt

    def _determine_slide_type(self, slide_content):
        """
        Анализирует содержимое слайда, чтобы определить его тип
        """
        content_lower = slide_content.lower()

        if "введение" in content_lower or content_lower.startswith("# ") and len(content_lower.split("\n")) < 5:
            return "title"

        elif "заключение" in content_lower or "вывод" in content_lower:
            return "conclusion"

        elif "список" in content_lower or content_lower.count("*") > 3 or content_lower.count("-") > 3:
            return "list"

        elif content_lower.count("#") > 3:
            return "section"

        elif "сравнение" in content_lower or "против" in content_lower or "vs" in content_lower:
            return "comparison"

        elif "пример" in content_lower or "кейс" in content_lower:
            return "example"

        elif any(stat in content_lower for stat in ["статистика", "данные", "цифры", "процент", "%"]):
            return "data"

        elif "история" in content_lower or "хронология" in content_lower or "временная шкала" in content_lower:
            return "timeline"

        elif "определение" in content_lower or "концепция" in content_lower or "понятие" in content_lower:
            return "definition"

        elif "проблема" in content_lower or "вызов" in content_lower or "трудность" in content_lower:
            return "problem"

        elif "решение" in content_lower or "стратегия" in content_lower or "метод" in content_lower:
            return "solution"

        elif "будущее" in content_lower or "тенденция" in content_lower or "прогноз" in content_lower:
            return "future"

        else:
            return "general"

    def _select_layout_for_slide(self, slide_type):
        """
        Выбирает подходящий макет для типа слайда
        """
        layout_map = {
            "title": "centered",
            "conclusion": "centered",
            "list": "two-column",
            "section": "featured",
            "comparison": "two-column",
            "example": "featured",
            "data": "grid",
            "timeline": "timeline",
            "definition": "two-column",
            "problem": "featured",
            "solution": "grid",
            "future": "featured",
            "general": "two-column"
        }

        return layout_map.get(slide_type, "two-column")

    def _select_theme_for_slide(self, slide_type):
        """
        Выбирает подходящую тему для типа слайда
        """
        # Набор тем, из которых будем выбирать
        themes = ["light", "dark", "colorful", "minimal", "corporate"]

        # Для некоторых типов слайдов выбираем конкретную тему
        theme_map = {
            "title": random.choice(["light", "dark", "colorful"]),
            "conclusion": random.choice(["light", "dark", "colorful"]),
            "comparison": "minimal",
            "data": "corporate",
            "timeline": "colorful",
            "future": "dark",
            "problem": "dark"
        }

        # Возвращаем тему для данного типа слайда или случайную
        return theme_map.get(slide_type, random.choice(themes))

    def _extract_and_clean_code(self, generated_code):
        """
        Извлекает и очищает код из сгенерированного текста
        """
        # Если код в блоках кода, извлекаем его
        if "```" in generated_code:
            # Ищем блоки кода
            code_blocks = re.findall(r"```(?:tsx|jsx|typescript|javascript|react)?(.*?)```", generated_code, re.DOTALL)

            if code_blocks:
                # Берем самый длинный блок кода (предположительно, полный компонент)
                code = max(code_blocks, key=len).strip()
                return code

        # Ищем начало импортов и конец экспорта
        import_match = re.search(r"import\s+React", generated_code)
        export_match = re.search(r"export\s+default", generated_code)

        if import_match and export_match:
            start_idx = import_match.start()
            # Ищем позицию после export default... до конца строки
            export_line_match = re.search(r"export\s+default\s+\w+;?", generated_code)
            if export_line_match:
                end_idx = export_line_match.end()
                return generated_code[start_idx:end_idx]

        # Если не удалось найти четкие границы, возвращаем весь текст
        return generated_code.strip()

    def _is_valid_react_code(self, code):
        """
        Базовая проверка, является ли код правильным React-компонентом
        """
        # Минимальные требования для React компонента
        has_import = "import" in code and "React" in code
        has_component = "const" in code and ("=>" in code or "function" in code)
        has_export = "export default" in code
        has_jsx = "<" in code and ">" in code

        return has_import and has_component and has_export and has_jsx

    def _get_template_code(self, slide_content, layout="two-column", theme="light"):
        """
        Возвращает шаблонный код на основе типа слайда, макета и темы
        """
        # Определяем тип слайда для выбора шаблона
        slide_type = self._determine_slide_type(slide_content)

        # Если макет и тема не указаны, выбираем автоматически
        if layout == "auto":
            layout = self._select_layout_for_slide(slide_type)

        if theme == "auto":
            theme = self._select_theme_for_slide(slide_type)

        # Выбираем шаблонный код в зависимости от типа слайда
        if slide_type == "title":
            return self._get_title_slide_template(slide_content, theme)
        elif slide_type == "conclusion":
            return self._get_conclusion_slide_template(slide_content, theme)
        elif slide_type == "comparison":
            return self._get_comparison_slide_template(slide_content, theme)
        elif slide_type == "timeline":
            return self._get_timeline_slide_template(slide_content, theme)
        elif slide_type == "data":
            return self._get_data_slide_template(slide_content, theme)
        elif slide_type == "list":
            return self._get_list_slide_template(slide_content, layout, theme)
        else:
            # Для остальных типов используем универсальный шаблон
            return self._get_universal_slide_template(slide_content, layout, theme)

    def _get_title_slide_template(self, slide_content, theme):
        """
        Шаблон для титульного слайда
        """
        # Цвета в зависимости от темы
        colors = self._get_theme_colors(theme)

        # Извлекаем заголовок и подзаголовок из содержимого слайда
        lines = slide_content.split('\n')
        title = ""
        subtitle = ""

        for line in lines:
            if line.startswith('# '):
                title = line.replace('# ', '')
                break

        for line in lines:
            if line.startswith('## '):
                subtitle = line.replace('## ', '')
                break

        # Если не нашли, используем значения по умолчанию
        if not title:
            title = "Презентация"
        if not subtitle:
            subtitle = "Подзаголовок презентации"

        # Рандомно выбираем анимацию для титульного слайда
        animations = [
            "fadeIn 1.5s ease-out",
            "slideInFromTop 1.2s ease-out",
            "zoomIn 1.8s ease-out",
            "fadeInWithScale 2s ease-out"
        ]
        title_animation = random.choice(animations)
        subtitle_animation = random.choice(animations)

        return f"""import React, {{ useEffect, useState }} from 'react';

interface SlideProps {{
  content?: string;
}}

const Slide: React.FC<SlideProps> = () => {{
  const [visible, setVisible] = useState(false);

  useEffect(() => {{
    // Анимация появления при монтировании компонента
    setVisible(true);
  }}, []);

  return (
    <div
      style={{{{
        backgroundColor: '{colors["background"]}',
        color: '{colors["text"]}',
        padding: '2rem',
        borderRadius: '12px',
        boxShadow: '0 8px 30px rgba(0, 0, 0, 0.12)',
        height: '100%',
        display: 'flex',
        flexDirection: 'column',
        justifyContent: 'center',
        alignItems: 'center',
        textAlign: 'center',
        overflow: 'hidden',
        transition: 'all 0.5s ease',
      }}}}
    >
      <h1 
        style={{{{
          fontSize: '3.5rem',
          marginBottom: '1.5rem',
          background: '{colors["gradient"]}',
          WebkitBackgroundClip: 'text',
          WebkitTextFillColor: 'transparent',
          opacity: visible ? 1 : 0,
          transform: visible ? 'translateY(0)' : 'translateY(-20px)',
          transition: '{title_animation}',
        }}}}
      >
        {title}
      </h1>

      <h2
        style={{{{
          fontSize: '1.8rem',
          fontWeight: 400,
          marginBottom: '2rem',
          color: '{colors["secondary"]}',
          opacity: visible ? 1 : 0,
          transform: visible ? 'translateY(0)' : 'translateY(20px)',
          transition: '{subtitle_animation}',
          transitionDelay: '0.3s',
        }}}}
      >
        {subtitle}
      </h2>

      <div
        style={{{{
          width: '60px',
          height: '4px',
          background: '{colors["accent"]}',
          marginTop: '1rem',
          opacity: visible ? 1 : 0,
          transform: visible ? 'scaleX(1)' : 'scaleX(0)',
          transition: 'all 1s ease',
          transitionDelay: '0.6s',
        }}}}
      />
    </div>
  );
}};

export default Slide;
"""

    def _get_conclusion_slide_template(self, slide_content, theme):
        """
        Шаблон для заключительного слайда
        """
        colors = self._get_theme_colors(theme)

        return f"""import React, {{ useState, useEffect }} from 'react';

interface SlideProps {{
  content?: string;
}}

const Slide: React.FC<SlideProps> = () => {{
  const [visibleItems, setVisibleItems] = useState(0);

  // Разбираем markdown-контент
  const title = '{self._extract_title(slide_content)}';
  const bulletPoints = [
    {self._extract_bullet_points(slide_content)}
  ].filter(item => item.trim() !== '');

  useEffect(() => {{
    // Постепенно показываем элементы списка
    const interval = setInterval(() => {{
      setVisibleItems(prev => {{
        if (prev < bulletPoints.length) {{
          return prev + 1;
        }}
        clearInterval(interval);
        return prev;
      }});
    }}, 800);

    return () => clearInterval(interval);
  }}, []);

  return (
    <div
      style={{{{
        backgroundColor: '{colors["background"]}',
        color: '{colors["text"]}',
        padding: '3rem',
        borderRadius: '10px',
        boxShadow: '0 4px 20px rgba(0, 0, 0, 0.15)',
        height: '100%',
        display: 'flex',
        flexDirection: 'column',
        justifyContent: 'center',
        position: 'relative',
        overflow: 'hidden',
      }}}}
    >
      <h1
        style={{{{
          fontSize: '2.8rem',
          marginBottom: '2rem',
          color: '{colors["primary"]}',
          textAlign: 'center',
          animation: 'fadeIn 1s ease-out',
        }}}}
      >
        {{title}}
      </h1>

      <div 
        style={{{{
          width: '100px',
          height: '4px',
          backgroundColor: '{colors["accent"]}',
          margin: '0 auto 2rem',
          animation: 'scaleIn 1.2s ease-out',
        }}}}
      />

      <div style={{{{ marginLeft: '2rem' }}}}>
        {{bulletPoints.map((point, index) => (
          <div
            key={{index}}
            style={{{{
              display: 'flex',
              alignItems: 'center',
              marginBottom: '1.5rem',
              opacity: index < visibleItems ? 1 : 0,
              transform: index < visibleItems ? 'translateX(0)' : 'translateX(-20px)',
              transition: 'all 0.5s ease',
              transitionDelay: `${{index * 0.1}}s`,
            }}}}
          >
            <div
              style={{{{
                minWidth: '30px',
                height: '30px',
                borderRadius: '50%',
                backgroundColor: '{colors["accent"]}',
                display: 'flex',
                justifyContent: 'center',
                alignItems: 'center',
                marginRight: '1rem',
                color: '#ffffff',
                fontWeight: 'bold',
              }}}}
            >
              {{index + 1}}
            </div>
            <p style={{{{ fontSize: '1.4rem', margin: 0 }}}}>{{point}}</p>
          </div>
        ))}}
      </div>

      <div 
        style={{{{
          position: 'absolute',
          bottom: '10px',
          right: '20px',
          fontSize: '1.1rem',
          fontStyle: 'italic',
          color: '{colors["secondary"]}',
          opacity: 0.8,
          animation: 'fadeIn 2s ease-out',
          animationDelay: '2s',
          animationFillMode: 'both',
        }}}}
      >
        Спасибо за внимание!
      </div>

      <style>{{`
        @keyframes fadeIn {{
          from {{ opacity: 0; }}
          to {{ opacity: 1; }}
        }}

        @keyframes scaleIn {{
          from {{ transform: scaleX(0); }}
          to {{ transform: scaleX(1); }}
        }}
      `}}</style>
    </div>
  );
}};

export default Slide;
"""

    def _get_comparison_slide_template(self, slide_content, theme):
        """
        Шаблон для слайда с сравнением
        """
        colors = self._get_theme_colors(theme)

        return f"""import React, {{ useState, useEffect }} from 'react';

interface SlideProps {{
  content?: string;
}}

const Slide: React.FC<SlideProps> = () => {{
  const [showLeft, setShowLeft] = useState(false);
  const [showRight, setShowRight] = useState(false);
  const [showTitle, setShowTitle] = useState(false);

  useEffect(() => {{
    // Последовательная анимация
    setTimeout(() => setShowTitle(true), 300);
    setTimeout(() => setShowLeft(true), 800);
    setTimeout(() => setShowRight(true), 1300);
  }}, []);

  // Заголовок слайда
  const title = '{self._extract_title(slide_content)}';

  // Извлекаем секции для сравнения
  const sections = {self._extract_comparison_sections(slide_content)};

  return (
    <div
      style={{{{
        backgroundColor: '{colors["background"]}',
        color: '{colors["text"]}',
        padding: '2rem',
        borderRadius: '8px',
        boxShadow: '0 4px 6px rgba(0, 0, 0, 0.1)',
        height: '100%',
        display: 'flex',
        flexDirection: 'column',
        position: 'relative',
        overflow: 'hidden',
      }}}}
    >
      <h1
        style={{{{
          textAlign: 'center',
          marginBottom: '2rem',
          color: '{colors["primary"]}',
          opacity: showTitle ? 1 : 0,
          transform: showTitle ? 'translateY(0)' : 'translateY(-20px)',
          transition: 'all 0.7s ease',
        }}}}
      >
        {{title}}
      </h1>

      <div 
        style={{{{
          display: 'flex',
          flexDirection: 'row',
          justifyContent: 'space-between',
          height: 'calc(100% - 100px)',
        }}}}
      >
        <div
          style={{{{
            flex: 1,
            backgroundColor: '{colors["background"]}',
            margin: '0 1rem 0 0',
            padding: '1.5rem',
            borderRadius: '8px',
            boxShadow: '0 2px 10px rgba(0, 0, 0, 0.08)',
            opacity: showLeft ? 1 : 0,
            transform: showLeft ? 'translateX(0)' : 'translateX(-50px)',
            transition: 'all 0.8s ease',
            display: 'flex',
            flexDirection: 'column',
          }}}}
        >
          <h2 style={{{{ color: '{colors["primary"]}', marginBottom: '1rem' }}}}>
            {{sections[0].title || 'Первый аспект'}}
          </h2>
          <ul style={{{{ paddingLeft: '1.5rem' }}}}>
            {{sections[0].points.map((point, idx) => (
              <li 
                key={{idx}} 
                style={{{{
                  marginBottom: '0.8rem',
                  animation: 'fadeIn 0.5s ease-out',
                  animationDelay: `${{0.1 * idx}}s`,
                  animationFillMode: 'both',
                }}}}
              >
                {{point}}
              </li>
            ))}}
          </ul>
        </div>

        <div
          style={{{{
            flex: 1,
            backgroundColor: '{colors["background"]}',
            margin: '0 0 0 1rem',
            padding: '1.5rem',
            borderRadius: '8px',
            boxShadow: '0 2px 10px rgba(0, 0, 0, 0.08)',
            opacity: showRight ? 1 : 0,
            transform: showRight ? 'translateX(0)' : 'translateX(50px)',
            transition: 'all 0.8s ease',
            display: 'flex',
            flexDirection: 'column',
          }}}}
        >
          <h2 style={{{{ color: '{colors["primary"]}', marginBottom: '1rem' }}}}>
            {{sections[1].title || 'Второй аспект'}}
          </h2>
          <ul style={{{{ paddingLeft: '1.5rem' }}}}>
            {{sections[1].points.map((point, idx) => (
              <li 
                key={{idx}} 
                style={{{{
                  marginBottom: '0.8rem',
                  animation: 'fadeIn 0.5s ease-out',
                  animationDelay: `${{0.1 * idx + 0.5}}s`,
                  animationFillMode: 'both',
                }}}}
              >
                {{point}}
              </li>
            ))}}
          </ul>
        </div>
      </div>

      <style>{{`
        @keyframes fadeIn {{
          from {{ opacity: 0; transform: translateY(10px); }}
          to {{ opacity: 1; transform: translateY(0); }}
        }}
      `}}</style>
    </div>
  );
}};

export default Slide;
"""

    def _get_timeline_slide_template(self, slide_content, theme):
        """
        Шаблон для слайда с временной шкалой
        """
        colors = self._get_theme_colors(theme)

        return f"""import React, {{ useState, useEffect }} from 'react';

interface SlideProps {{
  content?: string;
}}

interface TimelineItem {{
  title: string;
  content: string;
}}

const Slide: React.FC<SlideProps> = () => {{
  const [activeItem, setActiveItem] = useState(-1);

  // Заголовок слайда
  const title = '{self._extract_title(slide_content)}';

  // Элементы временной шкалы
  const timelineItems = {self._extract_timeline_items(slide_content)};

  useEffect(() => {{
    // Последовательно активируем элементы временной шкалы
    let index = 0;
    const interval = setInterval(() => {{
      if (index < timelineItems.length) {{
        setActiveItem(index);
        index++;
      }} else {{
        clearInterval(interval);
      }}
    }}, 1000);

    return () => clearInterval(interval);
  }}, []);

  return (
    <div
      style={{{{
        backgroundColor: '{colors["background"]}',
        color: '{colors["text"]}',
        padding: '2rem',
        borderRadius: '8px',
        boxShadow: '0 4px 6px rgba(0, 0, 0, 0.1)',
        height: '100%',
        display: 'flex',
        flexDirection: 'column',
        overflow: 'hidden',
      }}}}
    >
      <h1
        style={{{{
          textAlign: 'center',
          marginBottom: '2.5rem',
          color: '{colors["primary"]}',
          animation: 'fadeInDown 0.8s ease-out',
        }}}}
      >
        {{title}}
      </h1>

      <div style={{{{ flex: 1, display: 'flex', flexDirection: 'column' }}}}>
        {{timelineItems.map((item, index) => (
          <div
            key={{index}}
            style={{{{
              display: 'flex',
              opacity: index <= activeItem ? 1 : 0.3,
              transition: 'all 0.5s ease',
              marginBottom: '1.5rem',
            }}}}
          >
            <div
              style={{{{
                display: 'flex',
                flexDirection: 'column',
                alignItems: 'center',
                marginRight: '1.5rem',
              }}}}
            >
              <div
                style={{{{
                  width: '28px',
                  height: '28px',
                  borderRadius: '50%',
                  backgroundColor: index <= activeItem ? '{colors["accent"]}' : '{colors["secondary"]}',
                  display: 'flex',
                  justifyContent: 'center',
                  alignItems: 'center',
                  color: '#ffffff',
                  fontWeight: 'bold',
                  transition: 'all 0.5s ease',
                  zIndex: 2,
                }}}}
              >
                {{index + 1}}
              </div>
              {{index < timelineItems.length - 1 && (
                <div
                  style={{{{
                    width: '3px',
                    height: '100%',
                    backgroundColor: index < activeItem ? '{colors["accent"]}' : '{colors["secondary"]}',
                    opacity: 0.5,
                    transition: 'all 0.5s ease',
                    flex: 1,
                    marginTop: '5px',
                  }}}}
                />
              )}}
            </div>
            <div
              style={{{{
                flex: 1,
                padding: '1rem 1.5rem',
                backgroundColor: index <= activeItem ? '{colors["cardBackground"]}' : 'transparent',
                borderRadius: '8px',
                boxShadow: index <= activeItem ? '0 2px 10px rgba(0, 0, 0, 0.08)' : 'none',
                transition: 'all 0.5s ease',
                transform: index <= activeItem ? 'translateX(0)' : 'translateX(20px)',
              }}}}
            >
              <h3 
                style={{{{
                  margin: '0 0 0.5rem 0',
                  color: index <= activeItem ? '{colors["primary"]}' : '{colors["secondary"]}',
                }}}}
              >
                {{item.title}}
              </h3>
              <p style={{{{ margin: 0, fontSize: '1rem' }}}}>{{item.content}}</p>
            </div>
          </div>
        ))}}
      </div>

      <style>{{`
        @keyframes fadeInDown {{
          from {{ opacity: 0; transform: translateY(-20px); }}
          to {{ opacity: 1; transform: translateY(0); }}
        }}
      `}}</style>
    </div>
  );
}};

export default Slide;
"""

    def _get_data_slide_template(self, slide_content, theme):
        """
        Шаблон для слайда с данными/статистикой
        """
        colors = self._get_theme_colors(theme)

        return f"""import React, {{ useState, useEffect }} from 'react';

interface SlideProps {{
  content?: string;
}}

interface DataItem {{
  label: string;
  value: string;
}}

const Slide: React.FC<SlideProps> = () => {{
  const [animate, setAnimate] = useState(false);

  useEffect(() => {{
    // Запускаем анимацию после монтирования
    setTimeout(() => setAnimate(true), 300);
  }}, []);

  // Заголовок слайда
  const title = '{self._extract_title(slide_content)}';

  // Данные для отображения
  const dataItems = {self._extract_data_items(slide_content)};

  return (
    <div
      style={{{{
        backgroundColor: '{colors["background"]}',
        color: '{colors["text"]}',
        padding: '2.5rem',
        borderRadius: '8px',
        boxShadow: '0 4px 15px rgba(0, 0, 0, 0.1)',
        height: '100%',
        display: 'flex',
        flexDirection: 'column',
        overflow: 'hidden',
      }}}}
    >
      <h1
        style={{{{
          textAlign: 'center',
          marginBottom: '2.5rem',
          color: '{colors["primary"]}',
          animation: 'fadeIn 1s ease-out',
        }}}}
      >
        {{title}}
      </h1>

      <div 
        style={{{{
          display: 'grid',
          gridTemplateColumns: 'repeat(auto-fit, minmax(250px, 1fr))',
          gap: '2rem',
          flex: 1,
        }}}}
      >
        {{dataItems.map((item, index) => (
          <div
            key={{index}}
            style={{{{
              backgroundColor: '{colors["cardBackground"]}',
              borderRadius: '10px',
              padding: '1.5rem',
              boxShadow: '0 3px 10px rgba(0, 0, 0, 0.08)',
              display: 'flex',
              flexDirection: 'column',
              alignItems: 'center',
              justifyContent: 'center',
              textAlign: 'center',
              transition: 'all 0.5s ease',
              transform: animate ? 'scale(1)' : 'scale(0.9)',
              opacity: animate ? 1 : 0,
              transitionDelay: `${{index * 0.15}}s`,
            }}}}
          >
            <div
              style={{{{
                fontSize: '2.5rem',
                fontWeight: 'bold',
                marginBottom: '1rem',
                color: '{colors["accent"]}',
              }}}}
            >
              {{item.value}}
            </div>
            <div style={{{{ color: '{colors["text"]}' }}}}>
              {{item.label}}
            </div>
          </div>
        ))}}
      </div>

      <div 
        style={{{{
          textAlign: 'center',
          marginTop: '2rem',
          fontSize: '1.1rem',
          fontStyle: 'italic',
          color: '{colors["secondary"]}',
          opacity: animate ? 0.8 : 0,
          transition: 'all 0.5s ease',
          transitionDelay: '1s',
        }}}}
      >
        Источник: аналитические данные
      </div>

      <style>{{`
        @keyframes fadeIn {{
          from {{ opacity: 0; }}
          to {{ opacity: 1; }}
        }}
      `}}</style>
    </div>
  );
}};

export default Slide;
"""

    def _get_list_slide_template(self, slide_content, layout, theme):
        """
        Шаблон для слайда со списком
        """
        colors = self._get_theme_colors(theme)

        # Разные стили для разных макетов
        if layout == "two-column":
            return f"""import React, {{ useState, useEffect }} from 'react';

interface SlideProps {{
  content?: string;
}}

const Slide: React.FC<SlideProps> = () => {{
  const [visibleItems, setVisibleItems] = useState(0);

  // Заголовок слайда
  const title = '{self._extract_title(slide_content)}';

  // Элементы списка
  const listItems = {self._extract_list_items(slide_content)};

  useEffect(() => {{
    // Постепенно показываем элементы списка
    const interval = setInterval(() => {{
      setVisibleItems(prev => {{
        if (prev < listItems.length) {{
          return prev + 1;
        }}
        clearInterval(interval);
        return prev;
      }});
    }}, 500);

    return () => clearInterval(interval);
  }}, []);

  return (
    <div
      style={{{{
        backgroundColor: '{colors["background"]}',
        color: '{colors["text"]}',
        padding: '2rem',
        borderRadius: '8px',
        boxShadow: '0 4px 6px rgba(0, 0, 0, 0.1)',
        height: '100%',
        display: 'grid',
        gridTemplateColumns: '35% 65%',
        overflow: 'hidden',
      }}}}
    >
      <div
        style={{{{
          paddingRight: '2rem',
          display: 'flex',
          flexDirection: 'column',
          justifyContent: 'center',
        }}}}
      >
        <h1
          style={{{{
            fontSize: '2.2rem',
            marginBottom: '1.5rem',
            color: '{colors["primary"]}',
            animation: 'fadeInLeft 1s ease-out',
          }}}}
        >
          {{title}}
        </h1>

        <div
          style={{{{
            width: '80px',
            height: '4px',
            backgroundColor: '{colors["accent"]}',
            marginBottom: '1.5rem',
            animation: 'scaleIn 1.2s ease-out',
          }}}}
        />

        <p
          style={{{{
            color: '{colors["secondary"]}',
            fontSize: '1.1rem',
            animation: 'fadeIn 1.5s ease-out',
          }}}}
        >
          Ключевые пункты, которые помогут лучше понять тему.
        </p>
      </div>

      <div
        style={{{{
          paddingLeft: '2rem',
          borderLeft: `1px solid {colors["accent"]}30`,
        }}}}
      >
        <ul style={{{{ listStyle: 'none', padding: 0, margin: 0 }}}}>
          {{listItems.map((item, index) => (
            <li
              key={{index}}
              style={{{{
                display: 'flex',
                alignItems: 'flex-start',
                marginBottom: '1.2rem',
                opacity: index < visibleItems ? 1 : 0,
                transform: index < visibleItems ? 'translateX(0)' : 'translateX(20px)',
                transition: 'all 0.5s ease',
              }}}}
            >
              <div
                style={{{{
                  backgroundColor: '{colors["accent"]}',
                  borderRadius: '50%',
                  width: '24px',
                  height: '24px',
                  display: 'flex',
                  alignItems: 'center',
                  justifyContent: 'center',
                  marginRight: '1rem',
                  marginTop: '2px',
                  color: 'white',
                  fontWeight: 'bold',
                  fontSize: '0.9rem',
                }}}}
              >
                {{index + 1}}
              </div>
              <div>
                <p style={{{{ margin: 0, fontSize: '1.15rem' }}}}>{{item}}</p>
              </div>
            </li>
          ))}}
        </ul>
      </div>

      <style>{{`
        @keyframes fadeInLeft {{
          from {{ opacity: 0; transform: translateX(-20px); }}
          to {{ opacity: 1; transform: translateX(0); }}
        }}

        @keyframes scaleIn {{
          from {{ transform: scaleX(0); }}
          to {{ transform: scaleX(1); }}
        }}

        @keyframes fadeIn {{
          from {{ opacity: 0; }}
          to {{ opacity: 1; }}
        }}
      `}}</style>
    </div>
  );
}};

export default Slide;
"""
        else:
            return self._get_universal_slide_template(slide_content, layout, theme)

    def _get_universal_slide_template(self, slide_content, layout, theme):
        """
        Универсальный шаблон для остальных типов слайдов
        """
        colors = self._get_theme_colors(theme)

        # Разные макеты
        if layout == "centered":
            return f"""import React, {{ useEffect, useState }} from 'react';

interface SlideProps {{
  content?: string;
}}

const Slide: React.FC<SlideProps> = () => {{
  const [visible, setVisible] = useState(false);

  useEffect(() => {{
    // Запускаем анимацию после короткой задержки
    setTimeout(() => setVisible(true), 100);
  }}, []);

  // Функция для преобразования markdown в HTML
  const markdownToHtml = (markdown) => {{
    // Простая реализация преобразования markdown в HTML
    let html = markdown;

    // Преобразование заголовков
    html = html.replace(/^# (.+)$/gm, '<h1>$1</h1>');
    html = html.replace(/^## (.+)$/gm, '<h2>$1</h2>');
    html = html.replace(/^### (.+)$/gm, '<h3>$1</h3>');

    // Преобразование списков
    html = html.replace(/^\\* (.+)$/gm, '<li>$1</li>');
    html = html.replace(/^- (.+)$/gm, '<li>$1</li>');

    // Оборачиваем списки в <ul>
    const lis = html.match(/<li>(.+?)<\\/li>/g);
    if (lis) {{
      html = html.replace(/<li>(.+?)<\\/li>/g, (match) => {{
        return '<ul>' + match + '</ul>';
      }});
      html = html.replace(/<\\/ul><ul>/g, '');
    }}

    // Преобразование параграфов
    html = html.replace(/^([^<].*?)$/gm, '<p>$1</p>');
    html = html.replace(/<p>\\s*<\\/p>/g, '');
    html = html.replace(/<p><h([1-3])>/g, '<h$1>');
    html = html.replace(/<\\/h([1-3])><\\/p>/g, '</h$1>');
    html = html.replace(/<p><ul>/g, '<ul>');
    html = html.replace(/<\\/ul><\\/p>/g, '</ul>');

    return html;
  }};

  return (
    <div
      style={{{{
        backgroundColor: '{colors["background"]}',
        color: '{colors["text"]}',
        padding: '3rem',
        borderRadius: '12px',
        boxShadow: '0 6px 20px rgba(0, 0, 0, 0.1)',
        height: '100%',
        opacity: visible ? 1 : 0,
        transform: visible ? 'translateY(0)' : 'translateY(20px)',
        transition: 'all 0.8s ease',
        display: 'flex',
        flexDirection: 'column',
        alignItems: 'center',
        justifyContent: 'center',
        textAlign: 'center',
        maxWidth: '800px',
        margin: '0 auto',
      }}}}
    >
      <div
        className="markdown-content"
        style={{{{
          width: '100%',
        }}}}
        dangerouslySetInnerHTML={{ __html: markdownToHtml(`{slide_content.replace('`', "''")}`) }}
      />

      <style>{{`
        .markdown-content h1 {{
          color: {colors["primary"]};
          font-size: 2.5rem;
          margin-bottom: 1.5rem;
          animation: fadeIn 1s ease-out;
        }}

        .markdown-content h2 {{
          color: {colors["secondary"]};
          font-size: 1.8rem;
          margin-top: 1.5rem;
          margin-bottom: 1rem;
          animation: fadeIn 1.2s ease-out;
        }}

        .markdown-content p {{
          font-size: 1.2rem;
          line-height: 1.6;
          margin-bottom: 1rem;
          animation: fadeIn 1.4s ease-out;
        }}

        .markdown-content ul, .markdown-content ol {{
          text-align: left;
          margin: 1rem 0;
          padding-left: 2rem;
        }}

        .markdown-content li {{
          margin-bottom: 0.8rem;
          font-size: 1.2rem;
          animation: fadeIn 1.6s ease-out;
        }}

        @keyframes fadeIn {{
          from {{ opacity: 0; transform: translateY(10px); }}
          to {{ opacity: 1; transform: translateY(0); }}
        }}
      `}}</style>
    </div>
  );
}};

export default Slide;
"""
        elif layout == "two-column":
            return f"""import React, {{ useEffect, useState, useRef }} from 'react';

interface SlideProps {{
  content?: string;
}}

const Slide: React.FC<SlideProps> = () => {{
  const [title, setTitle] = useState('');
  const [content, setContent] = useState('');
  const [visible, setVisible] = useState(false);
  const contentRef = useRef<HTMLDivElement>(null);

  useEffect(() => {{
    // Преобразуем markdown в HTML и разделяем на заголовок и контент
    const rawContent = `{slide_content.replace('`', "''")}`;
    const lines = rawContent.split('\\n');

    // Извлекаем заголовок (первая строка с #)
    let titleText = '';
    let contentLines = [...lines];

    for (let i = 0; i < lines.length; i++) {{
      if (lines[i].startsWith('# ')) {{
        titleText = lines[i].substring(2);
        contentLines.splice(i, 1);
        break;
      }}
    }}

    setTitle(titleText);
    setContent(markdownToHtml(contentLines.join('\\n')));

    // Запускаем анимацию после короткой задержки
    setTimeout(() => setVisible(true), 100);
  }}, []);

  // Функция для преобразования markdown в HTML
  const markdownToHtml = (markdown) => {{
    // Простая реализация преобразования markdown в HTML
    let html = markdown;

    // Преобразование заголовков
    html = html.replace(/^# (.+)$/gm, '<h1>$1</h1>');
    html = html.replace(/^## (.+)$/gm, '<h2>$1</h2>');
    html = html.replace(/^### (.+)$/gm, '<h3>$1</h3>');

    // Преобразование списков
    html = html.replace(/^\\* (.+)$/gm, '<li>$1</li>');
    html = html.replace(/^- (.+)$/gm, '<li>$1</li>');

    // Оборачиваем списки в <ul>
    const lis = html.match(/<li>(.+?)<\\/li>/g);
    if (lis) {{
      html = html.replace(/<li>(.+?)<\\/li>/g, (match) => {{
        return '<ul>' + match + '</ul>';
      }});
      html = html.replace(/<\\/ul><ul>/g, '');
    }}

    // Преобразование параграфов
    html = html.replace(/^([^<].*?)$/gm, '<p>$1</p>');
    html = html.replace(/<p>\\s*<\\/p>/g, '');
    html = html.replace(/<p><h([1-3])>/g, '<h$1>');
    html = html.replace(/<\\/h([1-3])><\\/p>/g, '</h$1>');
    html = html.replace(/<p><ul>/g, '<ul>');
    html = html.replace(/<\\/ul><\\/p>/g, '</ul>');

    return html;
  }};

  return (
    <div
      style={{{{
        backgroundColor: '{colors["background"]}',
        color: '{colors["text"]}',
        padding: '2rem',
        borderRadius: '10px',
        boxShadow: '0 4px 15px rgba(0, 0, 0, 0.1)',
        height: '100%',
        display: 'grid',
        gridTemplateColumns: '35% 65%',
        gap: '2rem',
        overflow: 'hidden',
      }}}}
    >
      <div
        style={{{{
          display: 'flex',
          flexDirection: 'column',
          justifyContent: 'center',
          opacity: visible ? 1 : 0,
          transform: visible ? 'translateX(0)' : 'translateX(-20px)',
          transition: 'all 0.8s ease',
        }}}}
      >
        <h1
          style={{{{
            fontSize: '2.2rem',
            marginBottom: '1.5rem',
            color: '{colors["primary"]}',
          }}}}
        >
          {{title}}
        </h1>

        <div
          style={{{{
            width: '70px',
            height: '4px',
            backgroundColor: '{colors["accent"]}',
            marginBottom: '1.5rem',
            transition: 'all 1s ease',
            transitionDelay: '0.4s',
            transform: visible ? 'scaleX(1)' : 'scaleX(0)',
            transformOrigin: 'left',
          }}}}
        />

        <div
          style={{{{
            position: 'relative',
            height: '70%',
            overflow: 'hidden',
          }}}}
        >
          <div
            style={{{{
              position: 'absolute',
              top: '50%',
              left: '-20px',
              width: '160%',
              height: '160%',
              backgroundColor: '{colors["accent"]}20',
              borderRadius: '50%',
              opacity: visible ? 0.15 : 0,
              transition: 'all 1.5s ease',
              transform: visible ? 'scale(1)' : 'scale(0)',
              zIndex: -1,
            }}}}
          />
        </div>
      </div>

      <div
        ref={{contentRef}}
        style={{{{
          padding: '1.5rem',
          backgroundColor: '{colors["cardBackground"]}',
          borderRadius: '8px',
          boxShadow: 'inset 0 2px 10px rgba(0, 0, 0, 0.05)',
          overflow: 'auto',
          opacity: visible ? 1 : 0,
          transform: visible ? 'translateX(0)' : 'translateX(20px)',
          transition: 'all 0.8s ease',
          transitionDelay: '0.2s',
        }}}}
      >
        <div 
          className="markdown-content"
          dangerouslySetInnerHTML={{ __html: content }} 
        />
      </div>

      <style>{{`
        .markdown-content h2 {{
          color: {colors["secondary"]};
          font-size: 1.6rem;
          margin-top: 0;
          margin-bottom: 1rem;
          animation: fadeIn 1s ease-out;
        }}

        .markdown-content p {{
          font-size: 1.1rem;
          line-height: 1.6;
          margin-bottom: 1rem;
        }}

        .markdown-content ul, .markdown-content ol {{
          margin: 1rem 0;
          padding-left: 1.5rem;
        }}

        .markdown-content li {{
          margin-bottom: 0.7rem;
          position: relative;
          animation: slideIn 0.5s ease-out;
          animation-fill-mode: both;
        }}

        .markdown-content li:nth-child(1) {{ animation-delay: 0.3s; }}
        .markdown-content li:nth-child(2) {{ animation-delay: 0.5s; }}
        .markdown-content li:nth-child(3) {{ animation-delay: 0.7s; }}
        .markdown-content li:nth-child(4) {{ animation-delay: 0.9s; }}
        .markdown-content li:nth-child(5) {{ animation-delay: 1.1s; }}

        @keyframes fadeIn {{
          from {{ opacity: 0; }}
          to {{ opacity: 1; }}
        }}

        @keyframes slideIn {{
          from {{ opacity: 0; transform: translateX(15px); }}
          to {{ opacity: 1; transform: translateX(0); }}
        }}
      `}}</style>
    </div>
  );
}};

export default Slide;
"""
        elif layout == "grid":
            return f"""import React, {{ useEffect, useState }} from 'react';

interface SlideProps {{
  content?: string;
}}

interface ContentBlock {{
  title: string;
  content: string;
}}

const Slide: React.FC<SlideProps> = () => {{
  const [title, setTitle] = useState('');
  const [blocks, setBlocks] = useState<ContentBlock[]>([]);
  const [visibleBlocks, setVisibleBlocks] = useState(0);

  useEffect(() => {{
    // Разбираем markdown-контент на заголовок и блоки
    const rawContent = `{slide_content.replace('`', "''")}`;
    const lines = rawContent.split('\\n');

    // Извлекаем заголовок (первая строка с #)
    let titleText = '';
    let currentBlockTitle = '';
    let currentBlockContent = '';
    const contentBlocks: ContentBlock[] = [];

    lines.forEach((line, index) => {{
      if (line.startsWith('# ')) {{
        titleText = line.substring(2);
      }} else if (line.startsWith('## ')) {{
        // Если уже был заголовок блока, сохраняем предыдущий блок
        if (currentBlockTitle) {{
          contentBlocks.push({{
            title: currentBlockTitle,
            content: markdownToHtml(currentBlockContent)
          }});
        }}

        currentBlockTitle = line.substring(3);
        currentBlockContent = '';
      }} else if (currentBlockTitle) {{
        currentBlockContent += line + '\\n';
      }}
    }});

    // Добавляем последний блок
    if (currentBlockTitle) {{
      contentBlocks.push({{
        title: currentBlockTitle,
        content: markdownToHtml(currentBlockContent)
      }});
    }}

    setTitle(titleText);
    setBlocks(contentBlocks);

    // Анимация появления блоков
    const interval = setInterval(() => {{
      setVisibleBlocks(prev => {{
        if (prev < contentBlocks.length) {{
          return prev + 1;
        }}
        clearInterval(interval);
        return prev;
      }});
    }}, 300);

    return () => clearInterval(interval);
  }}, []);

  // Функция для преобразования markdown в HTML
  const markdownToHtml = (markdown) => {{
    // Простая реализация преобразования markdown в HTML
    let html = markdown;

    // Преобразование списков
    html = html.replace(/^\\* (.+)$/gm, '<li>$1</li>');
    html = html.replace(/^- (.+)$/gm, '<li>$1</li>');

    // Оборачиваем списки в <ul>
    const lis = html.match(/<li>(.+?)<\\/li>/g);
    if (lis) {{
      html = html.replace(/<li>(.+?)<\\/li>/g, (match) => {{
        return '<ul>' + match + '</ul>';
      }});
      html = html.replace(/<\\/ul><ul>/g, '');
    }}

    // Преобразование параграфов
    html = html.replace(/^([^<].*?)$/gm, '<p>$1</p>');
    html = html.replace(/<p>\\s*<\\/p>/g, '');
    html = html.replace(/<p><ul>/g, '<ul>');
    html = html.replace(/<\\/ul><\\/p>/g, '</ul>');

    return html;
  }};

  return (
    <div
      style={{{{
        backgroundColor: '{colors["background"]}',
        color: '{colors["text"]}',
        padding: '2.5rem',
        borderRadius: '10px',
        boxShadow: '0 4px 15px rgba(0, 0, 0, 0.1)',
        height: '100%',
        display: 'flex',
        flexDirection: 'column',
        overflow: 'hidden',
      }}}}
    >
      <h1
        style={{{{
          textAlign: 'center',
          marginBottom: '2rem',
          color: '{colors["primary"]}',
          animation: 'fadeInDown 0.8s ease-out',
        }}}}
      >
        {{title}}
      </h1>

      <div
        style={{{{
          display: 'grid',
          gridTemplateColumns: 'repeat(auto-fit, minmax(250px, 1fr))',
          gap: '1.5rem',
          flex: 1,
          overflow: 'auto',
        }}}}
      >
        {{blocks.map((block, index) => (
          <div
            key={{index}}
            style={{{{
              backgroundColor: '{colors["cardBackground"]}',
              borderRadius: '8px',
              padding: '1.5rem',
              boxShadow: '0 3px 10px rgba(0, 0, 0, 0.07)',
              display: 'flex',
              flexDirection: 'column',
              opacity: index < visibleBlocks ? 1 : 0,
              transform: index < visibleBlocks ? 'scale(1) translateY(0)' : 'scale(0.95) translateY(10px)',
              transition: 'all 0.5s ease',
            }}}}
          >
            <h2
              style={{{{
                fontSize: '1.5rem',
                marginTop: 0,
                marginBottom: '1rem',
                color: '{colors["secondary"]}',
              }}}}
            >
              {{block.title}}
            </h2>

            <div 
              style={{{{
                flex: 1,
                fontSize: '1rem',
              }}}}
              dangerouslySetInnerHTML={{ __html: block.content }} 
            />
          </div>
        ))}}
      </div>

      <style>{{`
        @keyframes fadeInDown {{
          from {{ opacity: 0; transform: translateY(-20px); }}
          to {{ opacity: 1; transform: translateY(0); }}
        }}
      `}}</style>
    </div>
  );
}};

export default Slide;
"""
        else:
            # Для остальных случаев - макет featured
            return f"""import React, {{ useEffect, useState }} from 'react';

interface SlideProps {{
  content?: string;
}}

const Slide: React.FC<SlideProps> = () => {{
  const [visible, setVisible] = useState(false);

  useEffect(() => {{
    // Запускаем анимацию после короткой задержки
    setTimeout(() => setVisible(true), 100);
  }}, []);

  // Функция для преобразования markdown в HTML
  const markdownToHtml = (markdown) => {{
    // Простая реализация преобразования markdown в HTML
    let html = markdown;

    // Преобразование заголовков
    html = html.replace(/^# (.+)$/gm, '<h1>$1</h1>');
    html = html.replace(/^## (.+)$/gm, '<h2>$1</h2>');
    html = html.replace(/^### (.+)$/gm, '<h3>$1</h3>');

    // Преобразование списков
    html = html.replace(/^\\* (.+)$/gm, '<li>$1</li>');
    html = html.replace(/^- (.+)$/gm, '<li>$1</li>');

    // Оборачиваем списки в <ul>
    const lis = html.match(/<li>(.+?)<\\/li>/g);
    if (lis) {{
      html = html.replace(/<li>(.+?)<\\/li>/g, (match) => {{
        return '<ul>' + match + '</ul>';
      }});
      html = html.replace(/<\\/ul><ul>/g, '');
    }}

    // Преобразование параграфов
    html = html.replace(/^([^<].*?)$/gm, '<p>$1</p>');
    html = html.replace(/<p>\\s*<\\/p>/g, '');
    html = html.replace(/<p><h([1-3])>/g, '<h$1>');
    html = html.replace(/<\\/h([1-3])><\\/p>/g, '</h$1>');
    html = html.replace(/<p><ul>/g, '<ul>');
    html = html.replace(/<\\/ul><\\/p>/g, '</ul>');

    return html;
  }};

  return (
    <div
      style={{{{
        backgroundColor: '{colors["background"]}',
        color: '{colors["text"]}',
        padding: '3rem',
        borderRadius: '12px',
        boxShadow: '0 8px 25px rgba(0, 0, 0, 0.12)',
        backgroundImage: 'radial-gradient(circle at 15% 85%, {colors["accent"]}10, transparent 25%)',
        height: '100%',
        display: 'flex',
        flexDirection: 'column',
        position: 'relative',
        overflow: 'hidden',
        opacity: visible ? 1 : 0,
        transition: 'opacity 1s ease',
      }}}}
    >
      <div
        className="markdown-content"
        style={{{{
          position: 'relative',
          zIndex: 2,
        }}}}
        dangerouslySetInnerHTML={{ __html: markdownToHtml(`{slide_content.replace('`', "''")}`) }}
      />

      <div
        style={{{{
          position: 'absolute',
          top: visible ? '10%' : '5%',
          right: visible ? '5%' : '0%',
          width: '200px',
          height: '200px',
          backgroundColor: '{colors["accent"]}15',
          borderRadius: '50%',
          filter: 'blur(40px)',
          transition: 'all 1.5s ease',
          zIndex: 1,
        }}}}
      />

      <style>{{`
        .markdown-content h1 {{
          color: '{colors["primary"]}';
          font-size: 2.8rem;
          margin-bottom: 1.5rem;
          position: relative;
          display: inline-block;
          animation: fadeIn 1s ease-out;
        }}

        .markdown-content h1::after {{
          content: '';
          position: absolute;
          bottom: -10px;
          left: 0;
          width: 100px;
          height: 4px;
          background-color: '{colors["accent"]}';
          animation: scaleIn 1.2s ease-out;
          transform-origin: left;
        }}

        .markdown-content h2 {{
          color: '{colors["secondary"]}';
          font-size: 1.8rem;
          margin-top: 1.5rem;
          margin-bottom: 1rem;
          animation: fadeIn 1.2s ease-out;
        }}

        .markdown-content p {{
          font-size: 1.2rem;
          line-height: 1.7;
          margin-bottom: 1rem;
          animation: fadeIn 1.4s ease-out;
          max-width: 85%;
        }}

        .markdown-content ul, .markdown-content ol {{
          margin: 1.5rem 0;
          padding-left: 1.5rem;
        }}

        .markdown-content li {{
          margin-bottom: 0.8rem;
          font-size: 1.2rem;
          position: relative;
          animation: slideIn 0.5s ease-out;
          animation-fill-mode: both;
        }}

        .markdown-content li:nth-child(1) {{ animation-delay: 0.3s; }}
        .markdown-content li:nth-child(2) {{ animation-delay: 0.5s; }}
        .markdown-content li:nth-child(3) {{ animation-delay: 0.7s; }}
        .markdown-content li:nth-child(4) {{ animation-delay: 0.9s; }}
        .markdown-content li:nth-child(5) {{ animation-delay: 1.1s; }}

        @keyframes fadeIn {{
          from {{ opacity: 0; }}
          to {{ opacity: 1; }}
        }}

        @keyframes scaleIn {{
          from {{ transform: scaleX(0); }}
          to {{ transform: scaleX(1); }}
        }}

        @keyframes slideIn {{
          from {{ opacity: 0; transform: translateX(15px); }}
          to {{ opacity: 1; transform: translateX(0); }}
        }}
      `}}</style>
    </div>
  );
}};

export default Slide;
"""

    def _get_theme_colors(self, theme):
        """
        Возвращает набор цветов для выбранной темы
        """
        colors = {
            "light": {
                "background": "#ffffff",
                "cardBackground": "#f9f9f9",
                "text": "#333333",
                "primary": "#222222",
                "secondary": "#666666",
                "accent": "#0070f3",
                "gradient": "linear-gradient(90deg, #0070f3, #00bfff)"
            },
            "dark": {
                "background": "#121212",
                "cardBackground": "#1e1e1e",
                "text": "#ffffff",
                "primary": "#ffffff",
                "secondary": "#aaaaaa",
                "accent": "#0070f3",
                "gradient": "linear-gradient(90deg, #0070f3, #00bfff)"
            },
            "colorful": {
                "background": "#051937",
                "cardBackground": "#132f58",
                "text": "#ffffff",
                "primary": "#ffffff",
                "secondary": "#e0e0ff",
                "accent": "#ff5e85",
                "gradient": "linear-gradient(45deg, #ff5e85, #ff8e53)"
            },
            "minimal": {
                "background": "#fafafa",
                "cardBackground": "#ffffff",
                "text": "#333333",
                "primary": "#222222",
                "secondary": "#666666",
                "accent": "#888888",
                "gradient": "linear-gradient(90deg, #888888, #aaaaaa)"
            },
            "corporate": {
                "background": "#f5f7fa",
                "cardBackground": "#ffffff",
                "text": "#333333",
                "primary": "#1a365d",
                "secondary": "#2c5282",
                "accent": "#3182ce",
                "gradient": "linear-gradient(90deg, #1a365d, #3182ce)"
            }
        }

        return colors.get(theme, colors["light"])

    def _extract_title(self, content):
        """
        Извлекает заголовок из markdown-контента
        """
        lines = content.split('\n')
        for line in lines:
            if line.startswith('# '):
                return line.replace('# ', '')
        return "Слайд"

    def _extract_bullet_points(self, content):
        """
        Извлекает маркированные пункты из markdown-контента
        """
        bullet_points = []
        lines = content.split('\n')

        for line in lines:
            if line.strip().startswith('* ') or line.strip().startswith('- '):
                bullet_points.append(f'"{line.strip()[2:]}"')

        return ', '.join(bullet_points)

    def _extract_comparison_sections(self, content):
        """
        Извлекает секции для сравнения из markdown-контента
        """
        sections = []
        current_section = None
        section_points = []

        lines = content.split('\n')

        for line in lines:
            if line.startswith('## '):
                # Если у нас уже есть секция, добавляем ее в список
                if current_section:
                    sections.append({
                        "title": current_section,
                        "points": section_points
                    })

                # Начинаем новую секцию
                current_section = line[3:].strip()
                section_points = []

            elif line.strip().startswith('* ') and current_section:
                section_points.append(line.strip()[2:])

        # Добавляем последнюю секцию
        if current_section:
            sections.append({
                "title": current_section,
                "points": section_points
            })

        # Если ничего не нашли или только одну секцию, создаем шаблонные секции
        if len(sections) < 2:
            sections = [
                {
                    "title": "Первый аспект",
                    "points": ["Пункт 1", "Пункт 2", "Пункт 3"]
                },
                {
                    "title": "Второй аспект",
                    "points": ["Пункт 1", "Пункт 2", "Пункт 3"]
                }
            ]

        return sections

    def _extract_timeline_items(self, content):
        """
        Извлекает элементы временной шкалы из markdown-контента
        """
        timeline_items = []
        lines = content.split('\n')

        # Ищем маркированные списки, которые могут быть элементами временной шкалы
        for i, line in enumerate(lines):
            if line.strip().startswith('* ') or line.strip().startswith('- '):
                item_title = line.strip()[2:]
                item_content = ""

                # Проверяем, есть ли дополнительное описание в следующей строке
                if i + 1 < len(lines) and not lines[i + 1].strip().startswith('* ') and not lines[
                    i + 1].strip().startswith('- '):
                    item_content = lines[i + 1].strip()

                timeline_items.append({
                    "title": item_title,
                    "content": item_content
                })

        # Если не нашли элементы, создаем шаблонные
        if not timeline_items:
            timeline_items = [
                {"title": "Начальная стадия", "content": "Описание начальной стадии"},
                {"title": "Основное развитие", "content": "Описание основного развития"},
                {"title": "Современное состояние", "content": "Описание современного состояния"},
                {"title": "Будущие перспективы", "content": "Описание будущих перспектив"}
            ]

        return timeline_items

    def _extract_data_items(self, content):
        """
        Извлекает данные для слайда со статистикой
        """
        data_items = []
        lines = content.split('\n')

        # Ищем списки, которые могут содержать статистические данные
        for line in lines:
            if line.strip().startswith('* ') or line.strip().startswith('- '):
                item_text = line.strip()[2:]

                # Пытаемся извлечь числовое значение и метку
                # Предполагаем, что значение может быть в начале или выделено
                parts = item_text.split(':')
                if len(parts) == 2:
                    data_items.append({
                        "value": parts[0].strip(),
                        "label": parts[1].strip()
                    })
                else:
                    # Если нет четкого разделения, пытаемся найти число
                    import re
                    number_match = re.search(r'(\d+(?:[\.,]\d+)?(?:\s*%)?)', item_text)
                    if number_match:
                        number = number_match.group(0)
                        label = item_text.replace(number, '').strip()
                        data_items.append({
                            "value": number,
                            "label": label
                        })
                    else:
                        # Если не смогли извлечь число, используем весь текст как метку
                        data_items.append({
                            "value": "",
                            "label": item_text
                        })

        # Если не нашли элементы или их меньше 3, создаем шаблонные
        if len(data_items) < 3:
            data_items = [
                {"value": "75%", "label": "Основной показатель"},
                {"value": "2.5x", "label": "Коэффициент роста"},
                {"value": "1200+", "label": "Количество случаев"},
                {"value": "30%", "label": "Доля рынка"}
            ]

        return data_items

    def _extract_list_items(self, content):
        """
        Извлекает элементы списка из markdown-контента
        """
        list_items = []
        lines = content.split('\n')

        for line in lines:
            if line.strip().startswith('* ') or line.strip().startswith('- '):
                list_items.append(line.strip()[2:])
            elif line.strip().startswith('1. ') or line.strip().startswith('2. ') or line.strip().startswith('3. '):
                # Для нумерованных списков
                parts = line.strip().split('. ', 1)
                if len(parts) > 1:
                    list_items.append(parts[1])

        # Если не нашли элементы, создаем шаблонные
        if not list_items:
            list_items = [
                "Первый важный пункт",
                "Второй важный пункт",
                "Третий важный пункт",
                "Четвертый важный пункт"
            ]

        return list_items