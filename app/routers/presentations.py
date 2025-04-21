from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session
from typing import List, Dict, Any

from app.database import get_db
from app.models.presentation import Presentation, Slide
from app.services.content_generator import ContentGenerator
from app.services.code_generator import CodeGenerator

router = APIRouter()

# Инициализация генераторов
content_generator = ContentGenerator()
code_generator = CodeGenerator()


@router.post("/generate_presentation", status_code=status.HTTP_201_CREATED)
async def generate_presentation(
        request: Dict[str, Any],
        db: Session = Depends(get_db)
):
    # Извлекаем параметры из запроса
    topic = request.get("topic")
    slides_count = request.get("slides_count", 14)  # По умолчанию 14 слайдов

    if not topic:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Необходимо указать тему"
        )

    # Создаем запись презентации
    db_presentation = Presentation(
        topic=topic,
        slides_count=slides_count
    )
    db.add(db_presentation)
    db.commit()
    db.refresh(db_presentation)

    # Генерируем контент для всех слайдов
    try:
        slides_content = content_generator.generate_all_slides(topic, slides_count)

        # Генерируем код и сохраняем слайды
        for slide_data in slides_content:
            slide_content = slide_data["content"]
            slide_number = slide_data["slide_number"]

            # Генерируем код фронтенда для слайда
            frontend_code = code_generator.generate_frontend_code(slide_content)

            # Создаем запись слайда
            db_slide = Slide(
                presentation_id=db_presentation.id,
                slide_number=slide_number,
                content=slide_content,
                code=frontend_code
            )
            db.add(db_slide)

        db.commit()

        return {
            "status": "success",
            "presentation_id": db_presentation.id,
            "message": "Презентация успешно сгенерирована"
        }
    except Exception as e:
        db.rollback()
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Ошибка при генерации презентации: {str(e)}"
        )


@router.get("/presentation/{presentation_id}")
async def get_presentation(
        presentation_id: int,
        db: Session = Depends(get_db)
):
    # Получаем презентацию со слайдами
    db_presentation = db.query(Presentation).filter(Presentation.id == presentation_id).first()

    if not db_presentation:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Презентация не найдена"
        )

    # Получаем слайды для презентации
    db_slides = db.query(Slide).filter(Slide.presentation_id == presentation_id).order_by(Slide.slide_number).all()

    # Форматируем ответ
    slides = []
    for slide in db_slides:
        slides.append({
            "slide_id": slide.slide_number,
            "content": slide.content,
            "code": slide.code
        })

    return {
        "presentation_id": db_presentation.id,
        "topic": db_presentation.topic,
        "slides": slides
    }


@router.post("/generate_frontend_code")
async def generate_frontend_code(
        request: Dict[str, Any]
):
    slide_content = request.get("slide_content")
    layout = request.get("layout", "single-column")
    theme = request.get("theme", "light")

    if not slide_content:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Необходимо указать содержимое слайда"
        )

    try:
        # Генерируем код фронтенда
        code = code_generator.generate_frontend_code(slide_content, layout, theme)

        return {
            "status": "success",
            "code": code
        }
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Ошибка при генерации кода: {str(e)}"
        )