import uvicorn
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.database import engine, Base
from app.routers import presentations

# Создаем таблицы базы данных
Base.metadata.create_all(bind=engine)

# Инициализируем приложение FastAPI
app = FastAPI(
    title="Presentation Generator API",
    description="API для генерации презентаций в виде адаптивных веб-сайтов",
    version="1.0.0"
)

# Настраиваем CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # В продакшене укажите домены вашего фронтенда
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Подключаем роутеры
app.include_router(presentations.router, tags=["presentations"])

# Корневой эндпоинт
@app.get("/")
def read_root():
    return {"message": "Добро пожаловать в API генератора презентаций"}

if __name__ == "__main__":
    uvicorn.run("app.main:app", host="0.0.0.0", port=8000, reload=True)