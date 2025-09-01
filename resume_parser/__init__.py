"""
Универсальный AI-парсер резюме с машинным обучением

Высокоточная система для парсинга резюме любых форматов с использованием
машинного обучения и достижением точности >90%.

Основные возможности:
- Универсальность: обработка резюме любых форматов (HH.ru, LinkedIn, международные)
- Адаптивность: автоматическое определение структуры и формата
- Высокая точность: >90% точность извлечения данных
- ML-подход: использование машинного обучения для улучшения качества
- Мультиязычность: поддержка русского, английского и других языков
- Обработка PDF: извлечение текста из PDF файлов
"""

__version__ = "1.0.0"
__author__ = "AI Resume Parser Team"

from .main import UniversalResumeParser, parse_resume, parse_resume_batch
from .core.data_models import ResumeData, PersonalInfo, Experience, Education, Skill
from .core.exceptions import ParsingError, FormatNotSupportedError, MLModelError

__all__ = [
    'UniversalResumeParser',
    'parse_resume', 
    'parse_resume_batch',
    'ResumeData',
    'PersonalInfo', 
    'Experience',
    'Education',
    'Skill',
    'ParsingError',
    'FormatNotSupportedError',
    'MLModelError'
]