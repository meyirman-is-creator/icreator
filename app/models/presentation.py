from sqlalchemy import Column, Integer, String, Text, DateTime, ForeignKey
from sqlalchemy.sql import func
from sqlalchemy.orm import relationship

from app.database import Base

class Presentation(Base):
    __tablename__ = "presentations"

    id = Column(Integer, primary_key=True, index=True)
    topic = Column(String(255), nullable=False)
    slides_count = Column(Integer, nullable=False)
    created_at = Column(DateTime(timezone=True), server_default=func.now())

    # Связь со слайдами
    slides = relationship("Slide", back_populates="presentation")

class Slide(Base):
    __tablename__ = "slides"

    id = Column(Integer, primary_key=True, index=True)
    presentation_id = Column(Integer, ForeignKey("presentations.id"))
    slide_number = Column(Integer, nullable=False)
    content = Column(Text, nullable=False)
    code = Column(Text, nullable=False)

    # Связь с презентацией
    presentation = relationship("Presentation", back_populates="slides")