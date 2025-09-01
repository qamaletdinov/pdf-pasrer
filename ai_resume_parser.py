import json
import re
import logging
import statistics
from typing import Dict, List, Optional, Any, Tuple, Union
from dataclasses import dataclass, asdict
from pathlib import Path
from datetime import datetime
from abc import ABC, abstractmethod
from collections import Counter, defaultdict
import warnings

warnings.filterwarnings("ignore")

# Advanced ML libraries
try:
    import numpy as np
    from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
    from sklearn.metrics.pairwise import cosine_similarity
    from sklearn.cluster import DBSCAN
    from sklearn.preprocessing import StandardScaler
    from scipy import stats
except ImportError:
    np = None
    TfidfVectorizer = None

# NLP libraries
try:
    import spacy
    from spacy.matcher import Matcher, PhraseMatcher
    from spacy.tokens import Doc, Span
except ImportError:
    spacy = None

try:
    import pymorphy2
except ImportError:
    pymorphy2 = None

# Fuzzy matching
try:
    from fuzzywuzzy import fuzz, process
except ImportError:
    fuzz = None


@dataclass
class ConfidenceMetrics:
    """Метрики уверенности для каждого поля"""
    overall_score: float
    field_scores: Dict[str, float]
    validation_results: Dict[str, bool]
    extraction_method: str
    reliability_factors: Dict[str, float]


@dataclass
class PersonalInfo:
    """Личная информация с метриками уверенности"""
    full_name: Optional[str] = None
    gender: Optional[str] = None
    age: Optional[int] = None
    birth_date: Optional[str] = None
    phone: Optional[str] = None
    email: Optional[str] = None
    telegram: Optional[str] = None
    city: Optional[str] = None
    citizenship: Optional[str] = None
    work_permit: Optional[str] = None
    ready_to_relocate: Optional[bool] = None
    ready_for_business_trips: Optional[bool] = None
    social_networks: List[str] = None

    # Метрики уверенности
    confidence: Optional[ConfidenceMetrics] = None

    def __post_init__(self):
        if self.social_networks is None:
            self.social_networks = []


@dataclass
class JobPreferences:
    """Предпочтения по работе"""
    desired_position: Optional[str] = None
    specializations: List[str] = None
    employment_type: Optional[str] = None
    work_schedule: Optional[str] = None
    desired_salary: Optional[str] = None
    travel_time_preference: Optional[str] = None
    confidence: Optional[ConfidenceMetrics] = None

    def __post_init__(self):
        if self.specializations is None:
            self.specializations = []


@dataclass
class TechStack:
    """Технологический стек с детальным анализом"""
    primary_technologies: List[str]
    secondary_technologies: List[str]
    testing_frameworks: List[str]
    databases: List[str]
    devops_tools: List[str]
    confidence_scores: Dict[str, float]

    def __post_init__(self):
        for field in ['primary_technologies', 'secondary_technologies',
                      'testing_frameworks', 'databases', 'devops_tools']:
            if getattr(self, field) is None:
                setattr(self, field, [])
        if self.confidence_scores is None:
            self.confidence_scores = {}


@dataclass
class Project:
    """Проект с детальным анализом"""
    name: Optional[str] = None
    description: Optional[str] = None
    tech_stack: TechStack = None
    responsibilities: List[str] = None
    achievements: List[str] = None
    complexity_score: Optional[float] = None
    confidence: Optional[ConfidenceMetrics] = None

    def __post_init__(self):
        if self.responsibilities is None:
            self.responsibilities = []
        if self.achievements is None:
            self.achievements = []
        if self.tech_stack is None:
            self.tech_stack = TechStack([], [], [], [], [], {})


@dataclass
class WorkExperience:
    """Опыт работы с ML анализом"""
    company: Optional[str] = None
    position: Optional[str] = None
    city: Optional[str] = None
    start_date: Optional[str] = None
    end_date: Optional[str] = None
    duration: Optional[str] = None
    duration_months: Optional[int] = None
    is_current: bool = False
    projects: List[Project] = None
    seniority_level: Optional[str] = None
    overall_description: Optional[str] = None
    confidence: Optional[ConfidenceMetrics] = None

    def __post_init__(self):
        if self.projects is None:
            self.projects = []


@dataclass
class Education:
    """Образование"""
    level: Optional[str] = None
    institution: Optional[str] = None
    city: Optional[str] = None
    faculty: Optional[str] = None
    specialization: Optional[str] = None
    graduation_year: Optional[str] = None
    is_completed: bool = True
    confidence: Optional[ConfidenceMetrics] = None


@dataclass
class Language:
    """Знание языков"""
    language: str
    level: Optional[str] = None
    level_score: Optional[float] = None  # Числовая оценка уровня
    description: Optional[str] = None
    confidence: Optional[ConfidenceMetrics] = None


@dataclass
class Skill:
    """Навык с детальным анализом"""
    name: str
    category: Optional[str] = None
    subcategory: Optional[str] = None
    confidence_score: Optional[float] = None
    experience_level: Optional[str] = None
    mentioned_in_projects: int = 0
    context_analysis: Dict[str, Any] = None
    market_demand_score: Optional[float] = None

    def __post_init__(self):
        if self.context_analysis is None:
            self.context_analysis = {}


@dataclass
class ResumeData:
    """Полная структура резюме с ML анализом"""
    source_text: str
    personal_info: PersonalInfo
    job_preferences: JobPreferences
    work_experience: List[WorkExperience] = None
    education: List[Education] = None
    skills: List[Skill] = None
    languages: List[Language] = None
    additional_info: Optional[str] = None
    last_updated: Optional[str] = None

    # ML метрики
    parsing_confidence: Optional[float] = None
    total_experience_months: Optional[int] = None
    seniority_assessment: Optional[str] = None
    skill_diversity_score: Optional[float] = None
    career_progression_score: Optional[float] = None
    overall_quality_score: Optional[float] = None

    # Детальные метрики
    extraction_metadata: Dict[str, Any] = None

    def __post_init__(self):
        if self.work_experience is None:
            self.work_experience = []
        if self.education is None:
            self.education = []
        if self.skills is None:
            self.skills = []
        if self.languages is None:
            self.languages = []
        if self.extraction_metadata is None:
            self.extraction_metadata = {}


class AdvancedTextPreprocessor:
    """Продвинутый препроцессор текста с ML"""

    def __init__(self):
        self.morph_analyzer = None
        self.spacy_nlp = None
        self._init_nlp_tools()

    def _init_nlp_tools(self):
        """Инициализация NLP инструментов"""
        if pymorphy2:
            try:
                self.morph_analyzer = pymorphy2.MorphAnalyzer()
                logging.info("Pymorphy2 инициализирован")
            except Exception as e:
                logging.warning(f"Ошибка pymorphy2: {e}")

        if spacy:
            try:
                self.spacy_nlp = spacy.load("ru_core_news_sm")
                logging.info("SpaCy модель загружена")
            except OSError:
                try:
                    self.spacy_nlp = spacy.load("en_core_web_sm")
                    logging.info("SpaCy EN модель загружена")
                except OSError:
                    logging.warning("SpaCy модели не найдены")

    def preprocess_text(self, text: str) -> Dict[str, Any]:
        """Предобработка текста с извлечением метаданных"""

        # Базовая очистка
        cleaned_text = self._clean_text(text)

        # Сегментация на строки
        lines = self._extract_lines(cleaned_text)

        # Статистический анализ
        stats = self._calculate_text_stats(cleaned_text, lines)

        # Структурный анализ
        structure = self._analyze_text_structure(lines)

        # Языковой анализ
        linguistic_features = self._extract_linguistic_features(cleaned_text)

        return {
            'original_text': text,
            'cleaned_text': cleaned_text,
            'lines': lines,
            'statistics': stats,
            'structure': structure,
            'linguistic_features': linguistic_features
        }

    def _clean_text(self, text: str) -> str:
        """Умная очистка текста"""
        # Нормализация пробелов
        text = re.sub(r'\s+', ' ', text)

        # Восстановление структуры
        text = re.sub(r'([.!?:])\s+', r'\1\n', text)
        text = re.sub(r'\n\s*\n+', '\n\n', text)

        # Удаление артефактов OCR
        text = re.sub(r'[^\w\s\n@.,:()\-+/№•·|]', ' ', text)

        return text.strip()

    def _extract_lines(self, text: str) -> List[str]:
        """Извлечение значимых строк"""
        lines = [line.strip() for line in text.split('\n') if line.strip()]

        # Фильтрация служебных строк
        filtered_lines = []
        for line in lines:
            if len(line) > 2 and not line.isdigit():
                filtered_lines.append(line)

        return filtered_lines

    def _calculate_text_stats(self, text: str, lines: List[str]) -> Dict[str, Any]:
        """Статистический анализ текста"""
        words = text.split()

        return {
            'total_chars': len(text),
            'total_words': len(words),
            'total_lines': len(lines),
            'avg_line_length': statistics.mean([len(line) for line in lines]) if lines else 0,
            'avg_word_length': statistics.mean([len(word) for word in words]) if words else 0,
            'unique_words': len(set(word.lower() for word in words)),
            'vocabulary_richness': len(set(word.lower() for word in words)) / len(words) if words else 0
        }

    def _analyze_text_structure(self, lines: List[str]) -> Dict[str, Any]:
        """Анализ структуры документа"""
        structure = {
            'potential_headers': [],
            'section_boundaries': [],
            'formatted_lists': [],
            'contact_lines': [],
            'date_lines': []
        }

        for i, line in enumerate(lines):
            # Потенциальные заголовки (короткие строки, заглавные буквы)
            if len(line) < 50 and (line.isupper() or line.istitle()):
                structure['potential_headers'].append((i, line))

            # Границы секций
            if any(keyword in line.lower() for keyword in
                   ['опыт работы', 'образование', 'навыки', 'желаемая должность']):
                structure['section_boundaries'].append((i, line))

            # Форматированные списки
            if line.startswith(('—', '-', '•', '*')) or re.match(r'^\d+\.', line):
                structure['formatted_lists'].append((i, line))

            # Контактная информация
            if any(pattern in line.lower() for pattern in ['@', '+7', 'telegram', 'vk.com']):
                structure['contact_lines'].append((i, line))

            # Строки с датами
            if re.search(r'\d{4}', line) or any(month in line.lower() for month in
                                                ['январь', 'февраль', 'март', 'апрель', 'май', 'июнь']):
                structure['date_lines'].append((i, line))

        return structure

    def _extract_linguistic_features(self, text: str) -> Dict[str, Any]:
        """Извлечение лингвистических особенностей"""
        features = {
            'language_detection': 'ru',  # Предполагаем русский
            'formality_score': 0.0,
            'technical_density': 0.0,
            'entity_mentions': []
        }

        if self.spacy_nlp:
            try:
                doc = self.spacy_nlp(text[:2000])  # Ограничиваем для производительности

                # Извлечение именованных сущностей
                for ent in doc.ents:
                    features['entity_mentions'].append({
                        'text': ent.text,
                        'label': ent.label_,
                        'confidence': getattr(ent, 'confidence', 0.0)
                    })

                # Оценка технической плотности
                tech_terms = ['разработка', 'программирование', 'система', 'проект', 'технология']
                tech_count = sum(1 for token in doc if token.text.lower() in tech_terms)
                features['technical_density'] = tech_count / len(doc) if doc else 0

            except Exception as e:
                logging.warning(f"Ошибка NLP анализа: {e}")

        return features


class MLTextSegmenter:
    """ML-сегментатор текста резюме с высокой точностью"""

    def __init__(self):
        self.vectorizer = None
        self.section_patterns = self._compile_section_patterns()
        self.confidence_thresholds = {
            'high': 0.9,
            'medium': 0.7,
            'low': 0.5
        }

        if TfidfVectorizer:
            self._init_vectorizer()

    def _init_vectorizer(self):
        """Инициализация векторизатора для семантического анализа"""
        self.vectorizer = TfidfVectorizer(
            max_features=500,
            ngram_range=(1, 3),
            lowercase=True,
            stop_words=self._get_russian_stopwords()
        )

    def _get_russian_stopwords(self) -> List[str]:
        """Расширенный список русских стоп-слов"""
        return [
            'и', 'в', 'во', 'не', 'что', 'он', 'на', 'я', 'с', 'со', 'как', 'а', 'то', 'все',
            'она', 'так', 'его', 'но', 'да', 'ты', 'к', 'у', 'же', 'вы', 'за', 'бы', 'по',
            'только', 'ее', 'мне', 'было', 'вот', 'от', 'меня', 'еще', 'нет', 'о', 'из',
            'ему', 'теперь', 'когда', 'даже', 'ну', 'вдруг', 'ли', 'если', 'уже', 'или'
        ]

    def _compile_section_patterns(self) -> Dict[str, List[re.Pattern]]:
        """Компиляция паттернов для поиска секций"""
        return {
            'personal_header': [
                re.compile(r'^([А-Я][а-я]+\s+[А-Я][а-я]+)(?:\s+[А-Я][а-я]+)?$'),
                re.compile(r'^[А-Я][а-я]+\s+[А-Я][а-я]+$')
            ],
            'contact_info': [
                re.compile(r'\+7\s*\([0-9]{3}\)\s*[0-9]{7}'),
                re.compile(r'[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}'),
                re.compile(r'telegram\s*@[\w\d_]+', re.IGNORECASE)
            ],
            'demographics': [
                re.compile(r'(Мужчина|Женщина),\s*(\d+)\s*лет?', re.IGNORECASE),
                re.compile(r'родился\s*(\d{1,2}\s+\w+\s+\d{4})', re.IGNORECASE)
            ],
            'location': [
                re.compile(r'Проживает:\s*(.+)', re.IGNORECASE),
                re.compile(r'Гражданство:\s*(.+)', re.IGNORECASE)
            ],
            'job_preferences': [
                re.compile(r'Желаемая должность', re.IGNORECASE),
                re.compile(r'Frontend-разработчик|Backend-разработчик|Fullstack', re.IGNORECASE)
            ],
            'work_experience': [
                re.compile(r'Опыт работы', re.IGNORECASE),
                re.compile(r'(\w+\s+\d{4})\s*—\s*(настоящее время|\w+\s+\d{4})')
            ],
            'projects': [
                re.compile(r'^Проект:\s*(.+)', re.IGNORECASE),
                re.compile(r'^Стек:\s*(.+)', re.IGNORECASE),
                re.compile(r'^Зона ответственности:\s*(.+)', re.IGNORECASE)
            ],
            'education': [
                re.compile(r'Образование', re.IGNORECASE),
                re.compile(r'(высшее|среднее|неоконченное)', re.IGNORECASE)
            ],
            'skills': [
                re.compile(r'^Навыки$', re.IGNORECASE),
                re.compile(r'\s+([A-Za-z][A-Za-z0-9.]+)\s+')
            ],
            'languages': [
                re.compile(r'Знание языков', re.IGNORECASE),
                re.compile(r'([А-Я][а-я]+)\s*—\s*(.+)')
            ]
        }

    def segment_resume(self, preprocessed_data: Dict[str, Any]) -> Dict[str, Any]:
        """Высокоточная сегментация резюме"""
        lines = preprocessed_data['lines']
        structure = preprocessed_data['structure']

        segments = {
            'header': {'lines': [], 'confidence': 0.0, 'start_idx': None, 'end_idx': None},
            'personal_info': {'lines': [], 'confidence': 0.0, 'start_idx': None, 'end_idx': None},
            'job_preferences': {'lines': [], 'confidence': 0.0, 'start_idx': None, 'end_idx': None},
            'work_experience': {'lines': [], 'confidence': 0.0, 'start_idx': None, 'end_idx': None},
            'education': {'lines': [], 'confidence': 0.0, 'start_idx': None, 'end_idx': None},
            'skills': {'lines': [], 'confidence': 0.0, 'start_idx': None, 'end_idx': None},
            'languages': {'lines': [], 'confidence': 0.0, 'start_idx': None, 'end_idx': None},
            'additional': {'lines': [], 'confidence': 0.0, 'start_idx': None, 'end_idx': None}
        }

        # Определение границ секций с высокой точностью
        section_boundaries = self._detect_section_boundaries(lines, structure)

        # Классификация строк по секциям
        current_section = 'header'

        for i, line in enumerate(lines):
            # Проверяем смену секции
            new_section = self._classify_line_section(line, i, section_boundaries)
            if new_section and new_section != current_section:
                current_section = new_section

            # Добавляем строку в текущую секцию
            if current_section in segments:
                segments[current_section]['lines'].append(line)

                # Устанавливаем границы
                if segments[current_section]['start_idx'] is None:
                    segments[current_section]['start_idx'] = i
                segments[current_section]['end_idx'] = i

        # Расчет уверенности для каждой секции
        for section_name, section_data in segments.items():
            confidence = self._calculate_section_confidence(section_name, section_data['lines'])
            segments[section_name]['confidence'] = confidence

        return segments

    def _detect_section_boundaries(self, lines: List[str], structure: Dict[str, Any]) -> List[Tuple[int, str, float]]:
        """Обнаружение границ секций с ML подходом"""
        boundaries = []

        # Анализ известных границ из структуры
        for idx, line in structure['section_boundaries']:
            section_type = self._determine_section_type(line)
            confidence = self._calculate_boundary_confidence(line, idx, lines)
            boundaries.append((idx, section_type, confidence))

        # Дополнительный анализ с помощью паттернов
        for i, line in enumerate(lines):
            if i < 10:  # Первые строки - заголовок/личная информация
                boundaries.append((i, 'personal_info', 0.8))

            # Поиск секций по ключевым словам
            for section, patterns in self.section_patterns.items():
                for pattern in patterns:
                    if pattern.search(line):
                        confidence = 0.9 if section in line.lower() else 0.7
                        boundaries.append((i, section, confidence))

        # Сортировка по индексу
        boundaries.sort(key=lambda x: x[0])

        # Удаление дубликатов с низкой уверенностью
        filtered_boundaries = []
        for boundary in boundaries:
            if not filtered_boundaries or boundary[0] != filtered_boundaries[-1][0]:
                filtered_boundaries.append(boundary)
            elif boundary[2] > filtered_boundaries[-1][2]:  # Более высокая уверенность
                filtered_boundaries[-1] = boundary

        return filtered_boundaries

    def _determine_section_type(self, line: str) -> str:
        """Определение типа секции по содержимому строки"""
        line_lower = line.lower()

        if 'желаемая должность' in line_lower:
            return 'job_preferences'
        elif 'опыт работы' in line_lower:
            return 'work_experience'
        elif 'образование' in line_lower:
            return 'education'
        elif 'навыки' in line_lower and 'языков' not in line_lower:
            return 'skills'
        elif 'знание языков' in line_lower:
            return 'languages'
        elif 'дополнительная информация' in line_lower:
            return 'additional'
        else:
            return 'additional'

    def _calculate_boundary_confidence(self, line: str, idx: int, lines: List[str]) -> float:
        """Расчет уверенности в границе секции"""
        confidence = 0.5

        # Бонус за точное совпадение ключевых слов
        key_sections = ['опыт работы', 'образование', 'навыки', 'желаемая должность']
        if any(section in line.lower() for section in key_sections):
            confidence += 0.4

        # Бонус за позицию в документе
        if idx < len(lines) * 0.1:  # В начале документа
            confidence += 0.1

        # Анализ контекста (следующие строки)
        context_lines = lines[idx + 1:idx + 4] if idx + 1 < len(lines) else []
        if context_lines and any(len(line) > 20 for line in context_lines):
            confidence += 0.1

        return min(confidence, 1.0)

    def _classify_line_section(self, line: str, idx: int, boundaries: List[Tuple[int, str, float]]) -> Optional[str]:
        """Классификация строки по секции"""
        # Поиск ближайшей границы секции
        current_section = None

        for boundary_idx, section_type, confidence in reversed(boundaries):
            if idx >= boundary_idx and confidence > self.confidence_thresholds['medium']:
                current_section = section_type
                break

        return current_section

    def _calculate_section_confidence(self, section_name: str, lines: List[str]) -> float:
        """Расчет уверенности в правильности сегментации секции"""
        if not lines:
            return 0.0

        confidence = 0.5

        # Анализ содержимого секции
        content_text = ' '.join(lines).lower()

        # Специфичные критерии для каждой секции
        if section_name == 'personal_info':
            has_contact = any(pattern in content_text for pattern in ['@', '+7', 'telegram'])
            has_location = 'проживает' in content_text or 'гражданство' in content_text
            confidence += 0.3 if has_contact else 0
            confidence += 0.2 if has_location else 0

        elif section_name == 'work_experience':
            has_dates = bool(re.search(r'\d{4}', content_text))
            has_projects = 'проект' in content_text
            has_company = len(lines) > 2
            confidence += 0.2 if has_dates else 0
            confidence += 0.2 if has_projects else 0
            confidence += 0.1 if has_company else 0

        elif section_name == 'skills':
            # Проверка на технические термины
            tech_terms = ['vue', 'react', 'javascript', 'python', 'java', 'sql']
            tech_count = sum(1 for term in tech_terms if term in content_text)
            confidence += min(tech_count * 0.1, 0.3)

        elif section_name == 'education':
            edu_terms = ['университет', 'институт', 'высшее', 'образование']
            has_education_terms = any(term in content_text for term in edu_terms)
            confidence += 0.3 if has_education_terms else 0

        # Проверка длины секции
        if len(lines) > 0:
            confidence += 0.1
        if len(lines) > 3:
            confidence += 0.1

        return min(confidence, 1.0)


class PrecisionPersonalInfoExtractor:
    """Высокоточный экстрактор личной информации"""

    def __init__(self):
        self.patterns = self._compile_patterns()
        self.confidence_weights = {
            'exact_match': 1.0,
            'pattern_match': 0.8,
            'context_match': 0.6,
            'fuzzy_match': 0.4
        }

    def _compile_patterns(self) -> Dict[str, Dict[str, re.Pattern]]:
        """Компиляция точных паттернов"""
        return {
            'name': {
                'full_name': re.compile(r'^([А-Я][а-я]+\s+[А-Я][а-я]+)(?:\s+[А-Я][а-я]+)?$'),
                'name_line': re.compile(r'^[А-Я][а-я]+\s+[А-Я][а-я]+$')
            },
            'demographics': {
                'gender_age': re.compile(r'(Мужчина|Женщина),\s*(\d+)\s*лет?', re.IGNORECASE),
                'birth_date': re.compile(r'родился\s*(\d{1,2}\s+\w+\s+\d{4})', re.IGNORECASE),
                'birth_date_numeric': re.compile(r'родился\s*(\d{1,2}[./\-]\d{1,2}[./\-]\d{4})', re.IGNORECASE),
                'birth_date_short': re.compile(r'дата рождения\s*:?\s*(\d{1,2}[./\-]\d{1,2}[./\-]\d{4})', re.IGNORECASE),
                'birth_date_simple': re.compile(r'(\d{1,2}[./\-]\d{1,2}[./\-](19|20)\d{2})'),
                'age_only': re.compile(r'(\d{1,2})\s*лет?', re.IGNORECASE)
            },
            'contacts': {
                'phone': re.compile(r'(\+7|8|7)?[\s\-\.]?[\(\[]?[0-9]{3}[\)\]]?[\s\-\.]?[0-9]{3}[\s\-\.]?[0-9]{2}[\s\-\.]?[0-9]{2}'),
                'phone_strict': re.compile(r'\+7\s*\([0-9]{3}\)\s*[0-9]{7}'),
                'email': re.compile(r'[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}'),
                'telegram': re.compile(r'telegram\s*@([\w\d_]+)', re.IGNORECASE),
                'telegram_alt': re.compile(r'@([\w\d_]+)')
            },
            'location': {
                'residence': re.compile(r'Проживает:\s*([^,\n]+)', re.IGNORECASE),
                'citizenship': re.compile(r'Гражданство:\s*([^,\n]+)', re.IGNORECASE),
                'city_name': re.compile(r'\b(Москва|СПб|Санкт-Петербург|Казань|Новосибирск|Екатеринбург)\b',
                                        re.IGNORECASE)
            },
            'work_status': {
                'relocation': re.compile(r'(не\s+готов|готов)\s+к\s+переезду', re.IGNORECASE),
                'business_trips': re.compile(r'готов\s+к\s+командировкам', re.IGNORECASE)
            },
            'social': {
                'url': re.compile(r'https?://[\w\./\-]+'),
                'vk': re.compile(r'vk\.com/([\w\d_]+)'),
                'github': re.compile(r'github\.com/([\w\d_\-]+)')
            }
        }

    def extract_personal_info(self, segments: Dict[str, Any]) -> PersonalInfo:
        """Высокоточное извлечение личной информации"""
        info = PersonalInfo()
        extraction_log = []

        # Объединяем релевантные секции
        relevant_lines = (
                segments.get('header', {}).get('lines', []) +
                segments.get('personal_info', {}).get('lines', [])
        )

        full_text = ' '.join(relevant_lines)

        # Извлечение ФИО
        name_result = self._extract_name(relevant_lines, full_text)
        info.full_name = name_result['value']
        extraction_log.append(('full_name', name_result))

        # Извлечение демографических данных
        demo_result = self._extract_demographics(relevant_lines, full_text)
        info.gender = demo_result.get('gender', {}).get('value')
        info.age = demo_result.get('age', {}).get('value')
        info.birth_date = demo_result.get('birth_date', {}).get('value')
        extraction_log.extend([
            ('gender', demo_result.get('gender', {})),
            ('age', demo_result.get('age', {})),
            ('birth_date', demo_result.get('birth_date', {}))
        ])

        # Извлечение контактов
        contacts_result = self._extract_contacts(relevant_lines, full_text)
        info.phone = contacts_result.get('phone', {}).get('value')
        info.email = contacts_result.get('email', {}).get('value')
        info.telegram = contacts_result.get('telegram', {}).get('value')
        extraction_log.extend([
            ('phone', contacts_result.get('phone', {})),
            ('email', contacts_result.get('email', {})),
            ('telegram', contacts_result.get('telegram', {}))
        ])

        # Извлечение местоположения
        location_result = self._extract_location(relevant_lines, full_text)
        info.city = location_result.get('city', {}).get('value')
        info.citizenship = location_result.get('citizenship', {}).get('value')
        extraction_log.extend([
            ('city', location_result.get('city', {})),
            ('citizenship', location_result.get('citizenship', {}))
        ])

        # Извлечение статуса работы
        work_status_result = self._extract_work_status(relevant_lines, full_text)
        info.ready_to_relocate = work_status_result.get('ready_to_relocate', {}).get('value')
        info.ready_for_business_trips = work_status_result.get('ready_for_business_trips', {}).get('value')
        extraction_log.extend([
            ('ready_to_relocate', work_status_result.get('ready_to_relocate', {})),
            ('ready_for_business_trips', work_status_result.get('ready_for_business_trips', {}))
        ])

        # Извлечение социальных сетей
        social_result = self._extract_social_networks(relevant_lines, full_text)
        info.social_networks = social_result.get('networks', [])
        extraction_log.append(('social_networks', social_result))

        # Расчет общей уверенности
        confidence_metrics = self._calculate_confidence_metrics(extraction_log)
        info.confidence = confidence_metrics

        return info

    def _extract_name(self, lines: List[str], full_text: str) -> Dict[str, Any]:
        """Извлечение ФИО с высокой точностью"""
        candidates = []

        # Проверяем первые строки (обычно ФИО в заголовке)
        for i, line in enumerate(lines[:3]):
            line = line.strip()

            # Точное совпадение с паттерном ФИО
            full_name_match = self.patterns['name']['full_name'].match(line)
            if full_name_match:
                candidates.append({
                    'value': full_name_match.group(1),
                    'confidence': 0.95,
                    'method': 'exact_pattern',
                    'position': i,
                    'source': line
                })

            # Простое имя и фамилия
            name_match = self.patterns['name']['name_line'].match(line)
            if name_match and len(line.split()) == 2:
                candidates.append({
                    'value': line,
                    'confidence': 0.9,
                    'method': 'name_pattern',
                    'position': i,
                    'source': line
                })

        # Выбираем лучшего кандидата
        if candidates:
            best_candidate = max(candidates, key=lambda x: (x['confidence'], -x['position']))
            return best_candidate

        return {'value': None, 'confidence': 0.0, 'method': 'not_found'}

    def _extract_demographics(self, lines: List[str], full_text: str) -> Dict[str, Dict[str, Any]]:
        """Извлечение демографических данных"""
        results = {}

        # Поиск пола и возраста
        for line in lines:
            gender_age_match = self.patterns['demographics']['gender_age'].search(line)
            if gender_age_match:
                results['gender'] = {
                    'value': gender_age_match.group(1),
                    'confidence': 0.95,
                    'method': 'exact_pattern',
                    'source': line
                }
                results['age'] = {
                    'value': int(gender_age_match.group(2)),
                    'confidence': 0.95,
                    'method': 'exact_pattern',
                    'source': line
                }

            # Поиск даты рождения - пробуем разные паттерны
            birth_value = None
            birth_confidence = 0.0
            birth_source = None
            
            # Попробуем все паттерны даты рождения
            birth_patterns = [
                ('birth_date', 0.9),           # родился дата
                ('birth_date_numeric', 0.85),  # родился ДД.ММ.ГГГГ
                ('birth_date_short', 0.8),     # дата рождения:
                ('birth_date_simple', 0.7)     # просто ДД.ММ.ГГГГ
            ]
            
            for pattern_name, confidence in birth_patterns:
                birth_match = self.patterns['demographics'][pattern_name].search(line)
                if birth_match and birth_confidence < confidence:
                    birth_value = birth_match.group(1)
                    birth_confidence = confidence
                    birth_source = line
                    
            if birth_value:
                results['birth_date'] = {
                    'value': birth_value,
                    'confidence': birth_confidence,
                    'method': 'pattern_match',
                    'source': birth_source
                }

        return results

    def _extract_contacts(self, lines: List[str], full_text: str) -> Dict[str, Dict[str, Any]]:
        """Извлечение контактной информации"""
        results = {}

        # Поиск телефона
        phone_match = self.patterns['contacts']['phone'].search(full_text)
        if phone_match:
            phone_value = phone_match.group(0)
            # Проверяем строгий формат для более высокой уверенности
            strict_match = self.patterns['contacts']['phone_strict'].search(full_text)
            confidence = 0.95 if strict_match else 0.85
            
            results['phone'] = {
                'value': self._normalize_phone(phone_value),
                'confidence': confidence,
                'method': 'pattern_match',
                'source': phone_value
            }

        # Поиск email
        email_match = self.patterns['contacts']['email'].search(full_text)
        if email_match:
            results['email'] = {
                'value': email_match.group(0),
                'confidence': 0.95,
                'method': 'exact_pattern',
                'source': email_match.group(0)
            }

        # Поиск Telegram
        telegram_match = self.patterns['contacts']['telegram'].search(full_text)
        if telegram_match:
            results['telegram'] = {
                'value': f"@{telegram_match.group(1)}",
                'confidence': 0.9,
                'method': 'exact_pattern',
                'source': telegram_match.group(0)
            }
        else:
            # Альтернативный поиск
            telegram_alt_match = self.patterns['contacts']['telegram_alt'].search(full_text)
            if telegram_alt_match and 'telegram' in full_text.lower():
                results['telegram'] = {
                    'value': telegram_alt_match.group(0),
                    'confidence': 0.8,
                    'method': 'context_pattern',
                    'source': telegram_alt_match.group(0)
                }

        return results

    def _extract_location(self, lines: List[str], full_text: str) -> Dict[str, Dict[str, Any]]:
        """Извлечение информации о местоположении"""
        results = {}

        # Поиск места проживания
        residence_match = self.patterns['location']['residence'].search(full_text)
        if residence_match:
            city_value = residence_match.group(1).strip()
            results['city'] = {
                'value': city_value,
                'confidence': 0.9,
                'method': 'exact_pattern',
                'source': residence_match.group(0)
            }

        # Поиск гражданства
        citizenship_match = self.patterns['location']['citizenship'].search(full_text)
        if citizenship_match:
            citizenship_value = citizenship_match.group(1).strip()
            # Извлекаем только основное гражданство
            if ',' in citizenship_value:
                citizenship_value = citizenship_value.split(',')[0].strip()

            results['citizenship'] = {
                'value': citizenship_value,
                'confidence': 0.9,
                'method': 'exact_pattern',
                'source': citizenship_match.group(0)
            }

        return results

    def _extract_work_status(self, lines: List[str], full_text: str) -> Dict[str, Dict[str, Any]]:
        """Извлечение статуса готовности к работе"""
        results = {}

        # Готовность к переезду
        relocation_match = self.patterns['work_status']['relocation'].search(full_text)
        if relocation_match:
            ready = 'не готов' not in relocation_match.group(1).lower()
            results['ready_to_relocate'] = {
                'value': ready,
                'confidence': 0.95,
                'method': 'exact_pattern',
                'source': relocation_match.group(0)
            }

        # Готовность к командировкам
        trips_match = self.patterns['work_status']['business_trips'].search(full_text)
        if trips_match:
            results['ready_for_business_trips'] = {
                'value': True,
                'confidence': 0.95,
                'method': 'exact_pattern',
                'source': trips_match.group(0)
            }

        return results

    def _extract_social_networks(self, lines: List[str], full_text: str) -> Dict[str, Any]:
        """Извлечение социальных сетей"""
        networks = []

        # Поиск URL
        url_matches = self.patterns['social']['url'].findall(full_text)
        for url in url_matches:
            networks.append(url)

        return {
            'networks': networks,
            'confidence': 0.8 if networks else 0.0,
            'method': 'pattern_search',
            'count': len(networks)
        }

    def _normalize_phone(self, phone: str) -> str:
        """Нормализация номера телефона"""
        # Удаляем все кроме цифр и +
        cleaned = re.sub(r'[^\d+]', '', phone)
        
        # Российские номера
        if cleaned.startswith('8') and len(cleaned) == 11:
            cleaned = '+7' + cleaned[1:]
        elif cleaned.startswith('7') and len(cleaned) == 11:
            cleaned = '+' + cleaned
        elif not cleaned.startswith('+') and len(cleaned) == 10:
            cleaned = '+7' + cleaned
            
        return cleaned

    def _calculate_confidence_metrics(self, extraction_log: List[Tuple[str, Dict[str, Any]]]) -> ConfidenceMetrics:
        """Расчет метрик уверенности"""
        field_scores = {}
        validation_results = {}
        reliability_factors = {}

        total_score = 0
        total_fields = 0

        for field_name, extraction_data in extraction_log:
            if isinstance(extraction_data, dict) and 'confidence' in extraction_data:
                confidence = extraction_data.get('confidence', 0.0)
                field_scores[field_name] = confidence

                # Валидация
                is_valid = confidence > 0.5 and extraction_data.get('value') is not None
                validation_results[field_name] = is_valid

                # Фактор надежности
                method = extraction_data.get('method', 'unknown')
                reliability = self.confidence_weights.get(method, 0.5)
                reliability_factors[field_name] = reliability

                total_score += confidence * reliability
                total_fields += 1

        overall_score = total_score / total_fields if total_fields > 0 else 0.0

        return ConfidenceMetrics(
            overall_score=overall_score,
            field_scores=field_scores,
            validation_results=validation_results,
            extraction_method='ml_precision_extractor',
            reliability_factors=reliability_factors
        )


class MLWorkExperienceExtractor:
    """ML экстрактор опыта работы с высокой точностью"""

    def __init__(self):
        self.patterns = self._compile_patterns()
        self.tech_classifier = TechnologyClassifier()
        self.seniority_analyzer = SeniorityAnalyzer()

    def _compile_patterns(self) -> Dict[str, re.Pattern]:
        """Компиляция паттернов для анализа опыта"""
        return {
            'date_range': re.compile(r'(\w+\s+\d{4})\s*—\s*(настоящее время|\w+\s+\d{4})'),
            'duration': re.compile(r'(\d+)\s*года?\s*(\d+)\s*месяцев?'),
            'project_start': re.compile(r'^Проект:\s*(.+)', re.IGNORECASE),
            'tech_stack': re.compile(r'^Стек:\s*(.+)', re.IGNORECASE),
            'responsibilities': re.compile(r'^Зона ответственности:\s*(.+)', re.IGNORECASE),
            'company_city': re.compile(r'^([^,\n]+),?\s*([А-Я][а-я]+)?$'),
            'position_keywords': re.compile(r'(разработчик|developer|программист|engineer|архитектор)', re.IGNORECASE)
        }

    def extract_work_experience(self, segments: Dict[str, Any]) -> Tuple[
        List[WorkExperience], Optional[int], Dict[str, Any]]:
        """Высокоточное извлечение опыта работы"""
        experience_lines = segments.get('work_experience', {}).get('lines', [])

        if not experience_lines:
            return [], None, {'error': 'no_experience_section'}

        # Извлечение общей продолжительности
        total_duration = self._extract_total_duration(experience_lines)

        # Группировка строк по местам работы
        work_blocks = self._group_work_blocks(experience_lines)

        # Парсинг каждого места работы
        experiences = []
        parsing_metadata = {'blocks_found': len(work_blocks), 'parsing_errors': []}

        for i, block in enumerate(work_blocks):
            try:
                experience = self._parse_work_block(block, i)
                if experience:
                    experiences.append(experience)
            except Exception as e:
                parsing_metadata['parsing_errors'].append({
                    'block_index': i,
                    'error': str(e),
                    'block_preview': block[:2] if block else []
                })

        # Анализ карьерного прогресса
        career_analysis = self._analyze_career_progression(experiences)
        parsing_metadata['career_analysis'] = career_analysis

        return experiences, total_duration, parsing_metadata

    def _extract_total_duration(self, lines: List[str]) -> Optional[int]:
        """Извлечение общей продолжительности опыта"""
        for line in lines[:3]:  # Ищем в первых строках
            duration_match = self.patterns['duration'].search(line)
            if duration_match:
                years = int(duration_match.group(1))
                months = int(duration_match.group(2))
                return years * 12 + months
        return None

    def _group_work_blocks(self, lines: List[str]) -> List[List[str]]:
        """Умная группировка строк по местам работы"""
        blocks = []
        current_block = []
        
        i = 0
        while i < len(lines):
            line = lines[i].strip()
            
            # Пропускаем общую информацию об опыте
            if any(keyword in line.lower() for keyword in ['опыт работы', 'общий опыт']) and not self.patterns['date_range'].search(line):
                i += 1
                continue
            
            # Начало нового места работы (дата)
            date_match = self.patterns['date_range'].search(line)
            if date_match:
                if current_block:
                    blocks.append(current_block)
                current_block = [line]
            elif current_block:
                current_block.append(line)
            else:
                # Если это первая строка без даты, но содержит должность/компанию
                if self._is_likely_work_entry(line):
                    current_block = [line]
                
            i += 1
        
        if current_block:
            blocks.append(current_block)
        
        # Фильтрация блоков - удаляем слишком короткие или пустые
        filtered_blocks = []
        for block in blocks:
            if len(block) >= 1 and any(len(line.strip()) > 5 for line in block):
                filtered_blocks.append(block)
        
        return filtered_blocks
    
    def _is_likely_work_entry(self, line: str) -> bool:
        """Проверяет, является ли строка вероятным началом записи о работе"""
        line_lower = line.lower()
        
        # Ключевые слова должностей
        position_keywords = [
            'разработчик', 'developer', 'программист', 'engineer', 'архитектор',
            'тестировщик', 'аналитик', 'менеджер', 'руководитель', 'специалист',
            'консультант', 'инженер', 'администратор', 'дизайнер'
        ]
        
        # Проверяем наличие ключевых слов должностей
        for keyword in position_keywords:
            if keyword in line_lower:
                return True
                
        # Проверяем длину и формат (может быть название компании)
        if len(line) > 10 and len(line) < 100:
            # Исключаем строки с техническими терминами (скорее всего описание)
            tech_keywords = ['html', 'css', 'javascript', 'python', 'java', 'стек', 'технологии']
            if not any(tech in line_lower for tech in tech_keywords):
                return True
                
        return False

    def _parse_work_block(self, block: List[str], block_index: int) -> Optional[WorkExperience]:
        """Высокоточный парсинг блока опыта работы"""
        if not block:
            return None

        experience = WorkExperience()
        projects = []
        current_project = None
        parsing_log = []

        # Анализ первой строки (даты)
        date_result = self._parse_dates(block[0])
        experience.start_date = date_result.get('start_date')
        experience.end_date = date_result.get('end_date')
        experience.is_current = date_result.get('is_current', False)
        experience.duration = date_result.get('duration_text')
        experience.duration_months = date_result.get('duration_months')
        parsing_log.append(('dates', date_result))

        # Анализ остальных строк
        company_position_found = False

        for i, line in enumerate(block[1:], 1):
            line = line.strip()
            if not line:
                continue

            # Проект
            project_match = self.patterns['project_start'].match(line)
            if project_match:
                if current_project:
                    projects.append(current_project)

                current_project = Project(name=project_match.group(1).strip())
                parsing_log.append(('project_found', {'name': current_project.name, 'line_index': i}))
                continue

            # Стек технологий
            stack_match = self.patterns['tech_stack'].match(line)
            if stack_match and current_project:
                tech_result = self.tech_classifier.parse_tech_stack(stack_match.group(1))
                current_project.tech_stack = tech_result['tech_stack']
                parsing_log.append(('tech_stack', tech_result))
                continue

            # Зона ответственности
            resp_match = self.patterns['responsibilities'].match(line)
            if resp_match and current_project:
                responsibilities = self._parse_responsibilities(resp_match.group(1))
                current_project.responsibilities = responsibilities['items']
                parsing_log.append(('responsibilities', responsibilities))
                continue

            # Компания и должность
            if not company_position_found:
                company_position_result = self._parse_company_position(line, i)

                if company_position_result['is_position']:
                    experience.position = company_position_result['value']
                    parsing_log.append(('position', company_position_result))
                elif company_position_result['is_company']:
                    experience.company = company_position_result['value']
                    if company_position_result.get('city'):
                        experience.city = company_position_result['city']
                    parsing_log.append(('company', company_position_result))

                if experience.company and experience.position:
                    company_position_found = True

        # Добавляем последний проект
        if current_project:
            projects.append(current_project)

        experience.projects = projects

        # Анализ уровня сениорности
        seniority_result = self.seniority_analyzer.analyze_seniority(experience, block)
        experience.seniority_level = seniority_result['level']

        # Расчет уверенности
        confidence_metrics = self._calculate_experience_confidence(experience, parsing_log, block)
        experience.confidence = confidence_metrics

        return experience

    def _parse_dates(self, date_line: str) -> Dict[str, Any]:
        """Парсинг дат работы"""
        result = {}

        # Поиск диапазона дат
        date_match = self.patterns['date_range'].search(date_line)
        if date_match:
            result['start_date'] = date_match.group(1)
            result['end_date'] = date_match.group(2)
            result['is_current'] = 'настоящее время' in date_match.group(2)
            result['confidence'] = 0.95

        # Поиск продолжительности
        duration_match = self.patterns['duration'].search(date_line)
        if duration_match:
            years = int(duration_match.group(1))
            months = int(duration_match.group(2))
            result['duration_text'] = f"{years} года {months} месяцев"
            result['duration_months'] = years * 12 + months
            result['duration_confidence'] = 0.9

        return result

    def _parse_company_position(self, line: str, line_index: int) -> Dict[str, Any]:
        """Определение является ли строка компанией или должностью"""
        result = {
            'value': line,
            'is_company': False,
            'is_position': False,
            'confidence': 0.0,
            'line_index': line_index
        }
        
        line_lower = line.lower().strip()
        
        # Расширенные ключевые слова должностей
        position_keywords = [
            'разработчик', 'developer', 'программист', 'engineer', 'архитектор',
            'тестировщик', 'аналитик', 'менеджер', 'руководитель', 'специалист',
            'консультант', 'инженер', 'администратор', 'дизайнер', 'lead',
            'senior', 'junior', 'middle', 'главный', 'ведущий', 'старший'
        ]
        
        # Проверка на должность
        position_score = 0
        for keyword in position_keywords:
            if keyword in line_lower:
                position_score += 1
                if keyword in ['senior', 'junior', 'middle', 'lead']:
                    position_score += 0.5  # Дополнительный бонус за уровни
        
        if position_score > 0:
            result['is_position'] = True
            result['confidence'] = min(0.9, 0.6 + position_score * 0.1)
            return result
        
        # Индикаторы компании
        company_indicators = [
            'ооо', 'зао', 'ао', 'пао', 'оао', 'group', 'ltd', 'company', 'corp',
            'inc', 'ltd.', 'llc', 'гк', 'холдинг', 'банк', 'фонд', 'агентство',
            'центр', 'институт', 'лаборатория', 'студия'
        ]
        
        # Проверка на компанию
        company_score = 0
        for indicator in company_indicators:
            if indicator in line_lower:
                company_score += 1
        
        # Дополнительные эвристики для компании
        if len(line.split()) <= 4 and line_index <= 2:  # Короткое название в начале
            company_score += 0.5
        if line.isupper() or line.istitle():  # Название заглавными буквами
            company_score += 0.3
        if any(char.isdigit() for char in line):  # Содержит цифры (возможно регистрационный номер)
            company_score += 0.2
            
        # Проверка на компанию (с возможным городом)
        company_match = self.patterns['company_city'].match(line)
        if company_match:
            company_name = company_match.group(1).strip()
            city = company_match.group(2)
            
            if company_score > 0:
                result['is_company'] = True
                result['confidence'] = min(0.95, 0.7 + company_score * 0.1)
            elif len(company_name.split()) <= 3 and line_index <= 3:
                result['is_company'] = True
                result['confidence'] = 0.75
            
            if city:
                result['city'] = city
        
        # Если не определили конкретно, используем позицию в блоке
        if not result['is_position'] and not result['is_company']:
            if line_index <= 2 and len(line) > 5:  # Первые строки после даты
                result['is_company'] = True
                result['confidence'] = 0.6
        
        return result

    def _parse_responsibilities(self, resp_text: str) -> Dict[str, Any]:
        """Парсинг зоны ответственности"""
        items = [item.strip() for item in resp_text.split(',') if item.strip()]

        return {
            'items': items,
            'count': len(items),
            'confidence': 0.9 if items else 0.0
        }

    def _analyze_career_progression(self, experiences: List[WorkExperience]) -> Dict[str, Any]:
        """Анализ карьерного прогресса"""
        if not experiences:
            return {'progression_score': 0.0, 'analysis': 'no_experience'}

        # Анализ уровней сениорности
        seniority_levels = [exp.seniority_level for exp in experiences if exp.seniority_level]

        # Анализ количества проектов
        project_counts = [len(exp.projects) for exp in experiences]

        # Анализ технологического разнообразия
        all_technologies = set()
        for exp in experiences:
            for project in exp.projects:
                if hasattr(project.tech_stack, 'primary_technologies'):
                    all_technologies.update(project.tech_stack.primary_technologies)

        return {
            'progression_score': self._calculate_progression_score(experiences),
            'seniority_progression': seniority_levels,
            'project_complexity_trend': project_counts,
            'technology_diversity': len(all_technologies),
            'total_projects': sum(project_counts),
            'analysis': 'completed'
        }

    def _calculate_progression_score(self, experiences: List[WorkExperience]) -> float:
        """Расчет оценки карьерного прогресса"""
        if not experiences:
            return 0.0

        score = 0.5  # Базовый балл

        # Бонус за количество мест работы
        if len(experiences) > 1:
            score += 0.1

        # Бонус за наличие проектов
        total_projects = sum(len(exp.projects) for exp in experiences)
        if total_projects > 0:
            score += min(total_projects * 0.05, 0.3)

        # Бонус за текущую работу
        if any(exp.is_current for exp in experiences):
            score += 0.1

        return min(score, 1.0)

    def _calculate_experience_confidence(self, experience: WorkExperience, parsing_log: List,
                                         block: List[str]) -> ConfidenceMetrics:
        """Расчет уверенности в извлеченном опыте"""
        field_scores = {}
        validation_results = {}

        # Оценка полей
        field_scores['company'] = 0.9 if experience.company else 0.0
        field_scores['position'] = 0.9 if experience.position else 0.0
        field_scores['dates'] = 0.9 if experience.start_date else 0.0
        field_scores['projects'] = min(len(experience.projects) * 0.3, 1.0)

        # Валидация
        validation_results['has_basic_info'] = bool(experience.company and experience.position)
        validation_results['has_dates'] = bool(experience.start_date)
        validation_results['has_projects'] = len(experience.projects) > 0
        validation_results['projects_detailed'] = any(
            len(p.tech_stack.primary_technologies) > 0 for p in experience.projects
        )

        # Общая оценка
        overall_score = statistics.mean(field_scores.values())

        return ConfidenceMetrics(
            overall_score=overall_score,
            field_scores=field_scores,
            validation_results=validation_results,
            extraction_method='ml_work_experience_extractor',
            reliability_factors={'block_completeness': len(block) / 10}
        )


# Продолжение TechnologyClassifier

class TechnologyClassifier:
    """Классификатор технологий с ML подходом"""

    def __init__(self):
        self.tech_categories = {
            'frontend': {
                'frameworks': ['vue', 'react', 'angular', 'svelte', 'ember'],
                'libraries': ['jquery', 'lodash', 'axios', 'moment'],
                'build_tools': ['webpack', 'vite', 'parcel', 'rollup'],
                'ui_frameworks': ['ionic', 'material-ui', 'bootstrap', 'tailwind'],
                'state_management': ['vuex', 'pinia', 'redux', 'rtk', 'mobx', 'ngrx'],
                'languages': ['javascript', 'typescript', 'html', 'css', 'sass', 'scss']
            },
            'backend': {
                'languages': ['node.js', 'python', 'java', 'c#', 'php', 'go', 'rust'],
                'frameworks': ['express', 'django', 'flask', 'spring', 'laravel', 'fastapi'],
                'runtime': ['node.js', 'deno', 'bun']
            },
            'testing': {
                'unit_testing': ['jest', 'vitest', 'mocha', 'jasmine', 'pytest'],
                'e2e_testing': ['cypress', 'selenium', 'playwright', 'puppeteer'],
                'testing_utils': ['testing-library', 'enzyme']
            },
            'databases': {
                'relational': ['mysql', 'postgresql', 'sqlite', 'oracle'],
                'nosql': ['mongodb', 'redis', 'cassandra', 'neo4j'],
                'query_languages': ['sql', 'graphql']
            },
            'devops': {
                'ci_cd': ['jenkins', 'gitlab ci', 'github actions', 'travis', 'ci/cd'],
                'containerization': ['docker', 'kubernetes', 'podman'],
                'cloud': ['aws', 'azure', 'gcp', 'heroku'],
                'monitoring': ['prometheus', 'grafana', 'elk'],
                'iac': ['terraform', 'ansible', 'chef', 'puppet']
            },
            'mobile': {
                'native': ['swift', 'kotlin', 'java', 'objective-c'],
                'cross_platform': ['react native', 'flutter', 'ionic', 'xamarin'],
                'hybrid': ['cordova', 'phonegap']
            }
        }

        # Создаем обратный индекс для быстрого поиска
        self.tech_to_category = {}
        self.tech_to_subcategory = {}
        self._build_reverse_index()

        # ML компоненты
        self.vectorizer = None
        self.similarity_threshold = 0.7

        if TfidfVectorizer:
            self._init_ml_components()

    def _build_reverse_index(self):
        """Построение обратного индекса технологий"""
        for category, subcategories in self.tech_categories.items():
            for subcategory, technologies in subcategories.items():
                for tech in technologies:
                    self.tech_to_category[tech.lower()] = category
                    self.tech_to_subcategory[tech.lower()] = subcategory

    def _init_ml_components(self):
        """Инициализация ML компонентов"""
        try:
            # Подготовка корпуса для векторизации
            all_technologies = []
            for category in self.tech_categories.values():
                for subcategory in category.values():
                    all_technologies.extend(subcategory)

            self.vectorizer = TfidfVectorizer(
                ngram_range=(1, 2),
                lowercase=True,
                max_features=1000
            )

            # Обучение на известных технологиях
            self.vectorizer.fit(all_technologies)

        except Exception as e:
            logging.warning(f"Ошибка инициализации ML компонентов: {e}")

    def parse_tech_stack(self, stack_text: str) -> Dict[str, Any]:
        """Парсинг стека технологий с высокой точностью"""
        # Очистка и разделение
        technologies = self._split_technologies(stack_text)

        # Классификация каждой технологии
        classified_tech = self._classify_technologies(technologies)

        # Создание структурированного стека
        tech_stack = TechStack(
            primary_technologies=classified_tech.get('frontend', {}).get('frameworks', []) +
                                 classified_tech.get('frontend', {}).get('languages', []),
            secondary_technologies=classified_tech.get('backend', {}).get('frameworks', []) +
                                   classified_tech.get('backend', {}).get('languages', []),
            testing_frameworks=classified_tech.get('testing', {}).get('unit_testing', []) +
                               classified_tech.get('testing', {}).get('e2e_testing', []),
            databases=classified_tech.get('databases', {}).get('relational', []) +
                      classified_tech.get('databases', {}).get('nosql', []),
            devops_tools=classified_tech.get('devops', {}).get('ci_cd', []) +
                         classified_tech.get('devops', {}).get('containerization', []),
            confidence_scores=self._calculate_tech_confidence(technologies, classified_tech)
        )

        return {
            'tech_stack': tech_stack,
            'raw_technologies': technologies,
            'classification_details': classified_tech,
            'parsing_confidence': self._calculate_parsing_confidence(technologies, classified_tech)
        }

    def _split_technologies(self, stack_text: str) -> List[str]:
        """Умное разделение технологий"""
        # Разделители
        separators = [r'\|', r',', r';', r'\s+']

        # Пробуем разные разделители
        for separator in separators:
            parts = re.split(separator, stack_text)
            if len(parts) > 1:
                # Очищаем и фильтруем
                technologies = [part.strip() for part in parts if part.strip()]
                return [tech for tech in technologies if len(tech) > 1]

        # Если разделители не найдены, возвращаем весь текст
        return [stack_text.strip()] if stack_text.strip() else []

    def _classify_technologies(self, technologies: List[str]) -> Dict[str, Dict[str, List[str]]]:
        """Классификация технологий по категориям"""
        classified = defaultdict(lambda: defaultdict(list))

        for tech in technologies:
            tech_lower = tech.lower()

            # Точное совпадение
            if tech_lower in self.tech_to_category:
                category = self.tech_to_category[tech_lower]
                subcategory = self.tech_to_subcategory[tech_lower]
                classified[category][subcategory].append(tech)

            # Fuzzy matching если доступен
            elif fuzz:
                best_match = self._find_best_fuzzy_match(tech_lower)
                if best_match:
                    category = self.tech_to_category[best_match]
                    subcategory = self.tech_to_subcategory[best_match]
                    classified[category][subcategory].append(tech)

            # ML классификация
            elif self.vectorizer:
                ml_category = self._ml_classify_technology(tech)
                if ml_category:
                    classified[ml_category]['unknown'].append(tech)

            # Дефолтная категория
            else:
                classified['unknown']['unclassified'].append(tech)

        return dict(classified)

    def _find_best_fuzzy_match(self, tech: str) -> Optional[str]:
        """Поиск лучшего fuzzy совпадения"""
        all_known_tech = list(self.tech_to_category.keys())

        try:
            best_match = process.extractOne(tech, all_known_tech, scorer=fuzz.ratio)
            if best_match and best_match[1] >= 80:  # 80% схожести
                return best_match[0]
        except Exception as e:
            logging.warning(f"Ошибка fuzzy matching: {e}")

        return None

    def _ml_classify_technology(self, tech: str) -> Optional[str]:
        """ML классификация технологии"""
        if not self.vectorizer:
            return None

        try:
            # Векторизация технологии
            tech_vector = self.vectorizer.transform([tech])

            # Сравнение с известными категориями
            best_category = None
            best_similarity = 0

            for category, subcategories in self.tech_categories.items():
                category_technologies = []
                for subcat_techs in subcategories.values():
                    category_technologies.extend(subcat_techs)

                if category_technologies:
                    category_vectors = self.vectorizer.transform(category_technologies)
                    similarities = cosine_similarity(tech_vector, category_vectors)
                    max_similarity = np.max(similarities)

                    if max_similarity > best_similarity and max_similarity > self.similarity_threshold:
                        best_similarity = max_similarity
                        best_category = category

            return best_category

        except Exception as e:
            logging.warning(f"Ошибка ML классификации: {e}")
            return None

    def _calculate_tech_confidence(self, technologies: List[str], classified: Dict) -> Dict[str, float]:
        """Расчет уверенности в классификации технологий"""
        confidence_scores = {}

        for tech in technologies:
            tech_lower = tech.lower()

            if tech_lower in self.tech_to_category:
                confidence_scores[tech] = 0.95  # Точное совпадение
            elif fuzz:
                best_match = self._find_best_fuzzy_match(tech_lower)
                if best_match:
                    similarity = fuzz.ratio(tech_lower, best_match) / 100
                    confidence_scores[tech] = similarity * 0.9
                else:
                    confidence_scores[tech] = 0.3
            else:
                confidence_scores[tech] = 0.5  # Средняя уверенность

        return confidence_scores

    def _calculate_parsing_confidence(self, technologies: List[str], classified: Dict) -> float:
        """Расчет общей уверенности парсинга"""
        if not technologies:
            return 0.0

        total_confidence = 0
        for tech in technologies:
            tech_lower = tech.lower()
            if tech_lower in self.tech_to_category:
                total_confidence += 0.95
            else:
                total_confidence += 0.5

        return total_confidence / len(technologies)


class SeniorityAnalyzer:
    """Анализатор уровня сениорности с ML"""

    def __init__(self):
        self.seniority_keywords = {
            'junior': ['junior', 'младший', 'начинающий', 'стажер', 'intern'],
            'middle': ['middle', 'средний', 'developer'],
            'senior': ['senior', 'старший', 'ведущий', 'lead', 'principal'],
            'lead': ['lead', 'тимлид', 'руководитель', 'manager', 'head', 'cto', 'архитектор']
        }

        self.responsibility_weights = {
            'architecture': 3.0,
            'leadership': 2.5,
            'mentoring': 2.0,
            'development': 1.0,
            'testing': 0.8,
            'support': 0.5
        }

    def analyze_seniority(self, experience: WorkExperience, context_lines: List[str]) -> Dict[str, Any]:
        """Анализ уровня сениорности"""
        context_text = ' '.join(context_lines).lower()

        # Прямое указание уровня
        direct_level = self._find_direct_seniority(context_text)
        if direct_level:
            return {
                'level': direct_level,
                'confidence': 0.9,
                'method': 'direct_indication',
                'evidence': f"Прямое указание в тексте"
            }

        # Анализ по проектам и ответственности
        project_analysis = self._analyze_project_complexity(experience.projects)
        responsibility_analysis = self._analyze_responsibilities(experience.projects)

        # ML предсказание
        ml_prediction = self._ml_predict_seniority(experience, context_text)

        # Комбинированная оценка
        combined_score = self._calculate_combined_seniority(
            project_analysis, responsibility_analysis, ml_prediction
        )

        return {
            'level': combined_score['predicted_level'],
            'confidence': combined_score['confidence'],
            'method': 'ml_analysis',
            'details': {
                'project_analysis': project_analysis,
                'responsibility_analysis': responsibility_analysis,
                'ml_prediction': ml_prediction
            }
        }

    def _find_direct_seniority(self, text: str) -> Optional[str]:
        """Поиск прямого указания на уровень сениорности"""
        for level, keywords in self.seniority_keywords.items():
            for keyword in keywords:
                if keyword in text:
                    return level
        return None

    def _analyze_project_complexity(self, projects: List[Project]) -> Dict[str, Any]:
        """Анализ сложности проектов"""
        if not projects:
            return {'complexity_score': 0.0, 'project_count': 0}

        total_complexity = 0
        for project in projects:
            complexity = self._calculate_project_complexity(project)
            total_complexity += complexity

        avg_complexity = total_complexity / len(projects)

        return {
            'complexity_score': avg_complexity,
            'project_count': len(projects),
            'total_technologies': sum(len(p.tech_stack.primary_technologies) for p in projects),
            'analysis': 'completed'
        }

    def _calculate_project_complexity(self, project: Project) -> float:
        """Расчет сложности отдельного проекта"""
        complexity = 0.0

        # Количество технологий
        tech_count = (len(project.tech_stack.primary_technologies) +
                      len(project.tech_stack.secondary_technologies) +
                      len(project.tech_stack.testing_frameworks) +
                      len(project.tech_stack.databases) +
                      len(project.tech_stack.devops_tools))

        complexity += min(tech_count * 0.1, 0.5)

        # Сложность ответственности
        for resp in project.responsibilities:
            resp_lower = resp.lower()
            if 'архитектура' in resp_lower:
                complexity += 0.3
            elif 'разработка' in resp_lower:
                complexity += 0.2
            elif 'тестирование' in resp_lower:
                complexity += 0.1

        return min(complexity, 1.0)

    def _analyze_responsibilities(self, projects: List[Project]) -> Dict[str, Any]:
        """Анализ зон ответственности"""
        responsibility_counts = defaultdict(int)
        total_responsibilities = 0

        for project in projects:
            for resp in project.responsibilities:
                resp_lower = resp.lower()
                total_responsibilities += 1

                # Категоризация ответственности
                if any(keyword in resp_lower for keyword in ['архитектура', 'проектирование']):
                    responsibility_counts['architecture'] += 1
                elif any(keyword in resp_lower for keyword in ['руководство', 'менторинг', 'код-ревью']):
                    responsibility_counts['leadership'] += 1
                elif 'разработка' in resp_lower:
                    responsibility_counts['development'] += 1
                elif any(keyword in resp_lower for keyword in ['тестирование', 'автотесты']):
                    responsibility_counts['testing'] += 1
                else:
                    responsibility_counts['other'] += 1

        # Расчет взвешенного балла
        weighted_score = 0
        for resp_type, count in responsibility_counts.items():
            weight = self.responsibility_weights.get(resp_type, 1.0)
            weighted_score += count * weight

        return {
            'responsibility_distribution': dict(responsibility_counts),
            'weighted_score': weighted_score,
            'total_responsibilities': total_responsibilities,
            'leadership_ratio': responsibility_counts.get('leadership', 0) / max(total_responsibilities, 1)
        }

    def _ml_predict_seniority(self, experience: WorkExperience, context: str) -> Dict[str, Any]:
        """ML предсказание уровня сениорности"""
        # Простая эвристическая модель (можно заменить на обученную ML модель)
        features = {
            'project_count': len(experience.projects),
            'total_technologies': sum(len(getattr(p.tech_stack, attr, []))
                                      for p in experience.projects
                                      for attr in ['primary_technologies', 'secondary_technologies']),
            'has_architecture_exp': any('архитектура' in r.lower()
                                        for p in experience.projects
                                        for r in p.responsibilities),
            'has_leadership_exp': any(keyword in context
                                      for keyword in ['руководство', 'менторинг', 'тимлид']),
            'duration_months': experience.duration_months or 0
        }

        # Простая scoring модель
        score = 0.0

        # Количество проектов
        score += min(features['project_count'] * 0.15, 0.4)

        # Технологическое разнообразие
        score += min(features['total_technologies'] * 0.05, 0.3)

        # Архитектурный опыт
        if features['has_architecture_exp']:
            score += 0.3

        # Лидерский опыт
        if features['has_leadership_exp']:
            score += 0.4

        # Продолжительность
        if features['duration_months'] > 36:  # Более 3 лет
            score += 0.2
        elif features['duration_months'] > 12:  # Более года
            score += 0.1

        # Предсказание уровня
        if score >= 0.8:
            predicted_level = 'senior'
        elif score >= 0.6:
            predicted_level = 'middle'
        elif score >= 0.3:
            predicted_level = 'junior'
        else:
            predicted_level = 'unknown'

        return {
            'predicted_level': predicted_level,
            'score': score,
            'features': features,
            'confidence': min(score, 1.0)
        }

    def _calculate_combined_seniority(self, project_analysis: Dict, responsibility_analysis: Dict,
                                      ml_prediction: Dict) -> Dict[str, Any]:
        """Комбинированная оценка сениорности"""
        # Весовые коэффициенты для разных методов
        weights = {
            'project_complexity': 0.3,
            'responsibility_weight': 0.4,
            'ml_prediction': 0.3
        }

        # Нормализация оценок
        project_score = min(project_analysis['complexity_score'], 1.0)
        responsibility_score = min(responsibility_analysis['weighted_score'] / 5.0, 1.0)
        ml_score = ml_prediction['score']

        # Взвешенная комбинация
        combined_score = (
                project_score * weights['project_complexity'] +
                responsibility_score * weights['responsibility_weight'] +
                ml_score * weights['ml_prediction']
        )

        # Предсказание финального уровня
        if combined_score >= 0.75:
            final_level = 'senior'
            confidence = 0.9
        elif combined_score >= 0.55:
            final_level = 'middle'
            confidence = 0.8
        elif combined_score >= 0.3:
            final_level = 'junior'
            confidence = 0.7
        else:
            final_level = 'unknown'
            confidence = 0.5

        return {
            'predicted_level': final_level,
            'confidence': confidence,
            'combined_score': combined_score,
            'component_scores': {
                'project_complexity': project_score,
                'responsibility_weight': responsibility_score,
                'ml_prediction': ml_score
            }
        }


class HighPrecisionSkillsExtractor:
    """Высокоточный экстрактор навыков с ML анализом"""

    def __init__(self):
        self.tech_classifier = TechnologyClassifier()
        self.market_analyzer = SkillMarketAnalyzer()
        self.context_analyzer = SkillContextAnalyzer()

    def extract_skills(self, segments: Dict[str, Any], work_experience: List[WorkExperience]) -> List[Skill]:
        """Высокоточное извлечение навыков"""
        # Извлечение из разных источников
        skills_from_section = self._extract_from_skills_section(segments.get('skills', {}).get('lines', []))
        skills_from_projects = self._extract_from_projects(work_experience)
        skills_from_context = self._extract_from_context(segments)

        # Объединение и дедупликация
        all_skills = self._merge_skill_sources(
            skills_from_section, skills_from_projects, skills_from_context
        )

        # ML анализ и обогащение
        enriched_skills = self._enrich_skills_with_ml(all_skills, work_experience)

        # Ранжирование по важности
        ranked_skills = self._rank_skills_by_importance(enriched_skills)

        return ranked_skills

    def _extract_from_skills_section(self, skills_lines: List[str]) -> Dict[str, Dict[str, Any]]:
        """Извлечение навыков из секции навыков"""
        skills = {}

        for line in skills_lines:
            # Поиск навыков в различных форматах
            skill_patterns = [
                r'\b([A-Za-z][A-Za-z0-9./\-+]+)\b',  # Стандартные навыки
                r'([А-Я][а-я]+(?:\s+[А-Я][а-я]+)*)',  # Русские навыки
            ]

            for pattern in skill_patterns:
                matches = re.findall(pattern, line)
                for match in matches:
                    skill_name = match.strip()

                    # Фильтрация слишком коротких или общих слов
                    if len(skill_name) > 2 and skill_name.lower() not in ['навыки', 'skills']:
                        skills[skill_name] = {
                            'source': 'skills_section',
                            'confidence': 0.85,
                            'context': line,
                            'mentioned_count': 1
                        }

        return skills

    def _extract_from_projects(self, work_experience: List[WorkExperience]) -> Dict[str, Dict[str, Any]]:
        """Извлечение навыков из проектов"""
        skills = {}

        for experience in work_experience:
            for project in experience.projects:
                # Технологии из стека
                all_technologies = (
                        project.tech_stack.primary_technologies +
                        project.tech_stack.secondary_technologies +
                        project.tech_stack.testing_frameworks +
                        project.tech_stack.databases +
                        project.tech_stack.devops_tools
                )

                for tech in all_technologies:
                    if tech not in skills:
                        skills[tech] = {
                            'source': 'project_tech_stack',
                            'confidence': 0.9,
                            'project_mentions': 1,
                            'projects': [project.name]
                        }
                    else:
                        skills[tech]['project_mentions'] = skills[tech].get('project_mentions', 0) + 1
                        skills[tech]['projects'] = skills[tech].get('projects', []) + [project.name]

        return skills

    def _extract_from_context(self, segments: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
        """Извлечение навыков из контекста"""
        skills = {}

        # Анализ всех секций на предмет технологий
        all_text = ""
        for segment_name, segment_data in segments.items():
            if isinstance(segment_data, dict) and 'lines' in segment_data:
                all_text += " ".join(segment_data['lines']) + " "

        # Поиск технологий в тексте
        for tech_name in self.tech_classifier.tech_to_category.keys():
            if re.search(r'\b' + re.escape(tech_name) + r'\b', all_text, re.IGNORECASE):
                skills[tech_name] = {
                    'source': 'context_mention',
                    'confidence': 0.7,
                    'context': 'mentioned_in_text'
                }

        return skills

    def _merge_skill_sources(self, *skill_sources: Dict[str, Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
        """Объединение навыков из разных источников"""
        merged_skills = {}

        for skill_source in skill_sources:
            for skill_name, skill_data in skill_source.items():
                normalized_name = self._normalize_skill_name(skill_name)

                if normalized_name not in merged_skills:
                    merged_skills[normalized_name] = {
                        'original_names': [skill_name],
                        'sources': [skill_data['source']],
                        'confidences': [skill_data['confidence']],
                        'total_mentions': 1,
                        'project_mentions': skill_data.get('project_mentions', 0),
                        'contexts': [skill_data.get('context', '')]
                    }
                else:
                    # Объединение данных
                    existing = merged_skills[normalized_name]
                    existing['original_names'].append(skill_name)
                    existing['sources'].append(skill_data['source'])
                    existing['confidences'].append(skill_data['confidence'])
                    existing['total_mentions'] += 1
                    existing['project_mentions'] += skill_data.get('project_mentions', 0)
                    existing['contexts'].append(skill_data.get('context', ''))

        return merged_skills

    def _normalize_skill_name(self, skill_name: str) -> str:
        """Нормализация названия навыка"""
        # Приведение к нижнему регистру
        normalized = skill_name.lower().strip()

        # Удаление версий (например, "Vue3" -> "Vue")
        normalized = re.sub(r'\d+$', '', normalized)

        # Унификация известных вариаций
        skill_aliases = {
            'js': 'javascript',
            'ts': 'typescript',
            'vue.js': 'vue',
            'react.js': 'react',
            'node': 'node.js',
            'postgres': 'postgresql'
        }

        return skill_aliases.get(normalized, normalized)

    def _enrich_skills_with_ml(self, merged_skills: Dict[str, Dict[str, Any]], work_experience: List[WorkExperience]) -> \
    List[Skill]:
        """Обогащение навыков с помощью ML анализа"""
        enriched_skills = []

        for skill_name, skill_data in merged_skills.items():
            # Базовый навык
            skill = Skill(name=skill_name.title())

            # Категоризация
            category_info = self.tech_classifier._classify_technologies([skill_name])
            if category_info:
                # Извлекаем категорию из структуры
                for category, subcategories in category_info.items():
                    for subcategory, techs in subcategories.items():
                        if techs:
                            skill.category = category.title()
                            skill.subcategory = subcategory.title()
                            break
                    if skill.category:
                        break

            # Расчет уверенности
            skill.confidence_score = self._calculate_skill_confidence(skill_data)

            # Анализ опыта
            skill.mentioned_in_projects = skill_data.get('project_mentions', 0)
            skill.experience_level = self._estimate_experience_level(skill_data, work_experience)

            # Контекстный анализ
            skill.context_analysis = self.context_analyzer.analyze_skill_context(
                skill_name, skill_data.get('contexts', [])
            )

            # Рыночный анализ
            skill.market_demand_score = self.market_analyzer.get_market_demand_score(skill_name)

            enriched_skills.append(skill)

        return enriched_skills

    def _calculate_skill_confidence(self, skill_data: Dict[str, Any]) -> float:
        """Расчет уверенности в навыке"""
        # Базовая уверенность как среднее из источников
        base_confidence = statistics.mean(skill_data['confidences'])

        # Бонусы за множественные упоминания
        mention_bonus = min(skill_data['total_mentions'] * 0.1, 0.3)

        # Бонус за упоминания в проектах
        project_bonus = min(skill_data.get('project_mentions', 0) * 0.15, 0.4)

        # Бонус за множественные источники
        source_diversity_bonus = min(len(set(skill_data['sources'])) * 0.1, 0.2)

        total_confidence = base_confidence + mention_bonus + project_bonus + source_diversity_bonus

        return min(total_confidence, 1.0)

    def _estimate_experience_level(self, skill_data: Dict[str, Any], work_experience: List[WorkExperience]) -> str:
        """Оценка уровня опыта с навыком"""
        project_mentions = skill_data.get('project_mentions', 0)

        if project_mentions >= 3:
            return 'advanced'
        elif project_mentions >= 1:
            return 'intermediate'
        else:
            return 'basic'

    def _rank_skills_by_importance(self, skills: List[Skill]) -> List[Skill]:
        """Ранжирование навыков по важности"""

        def skill_importance_score(skill: Skill) -> float:
            score = 0.0

            # Уверенность в навыке
            score += (skill.confidence_score or 0) * 0.4

            # Упоминания в проектах
            score += min(skill.mentioned_in_projects * 0.1, 0.3)

            # Рыночная востребованность
            score += (skill.market_demand_score or 0) * 0.2

            # Бонус за ключевые технологии
            key_technologies = ['javascript', 'python', 'react', 'vue', 'node.js', 'sql']
            if skill.name.lower() in key_technologies:
                score += 0.1

            return score

        # Сортировка по важности
        return sorted(skills, key=skill_importance_score, reverse=True)


class SkillContextAnalyzer:
    """Анализатор контекста использования навыков"""

    def analyze_skill_context(self, skill_name: str, contexts: List[str]) -> Dict[str, Any]:
        """Анализ контекста использования навыка"""
        if not contexts:
            return {}

        context_text = " ".join(contexts).lower()

        # Анализ уровня использования
        usage_level = self._analyze_usage_level(skill_name, context_text)

        # Анализ области применения
        application_areas = self._analyze_application_areas(context_text)

        # Анализ роли в проектах
        project_roles = self._analyze_project_roles(skill_name, context_text)

        return {
            'usage_level': usage_level,
            'application_areas': application_areas,
            'project_roles': project_roles,
            'confidence': 0.8
        }

    def _analyze_usage_level(self, skill_name: str, context: str) -> str:
        """Анализ уровня использования навыка"""
        expert_indicators = ['архитектура', 'проектирование', 'оптимизация', 'руководство']
        advanced_indicators = ['разработка', 'внедрение', 'интеграция']
        basic_indicators = ['изучение', 'знакомство', 'базовые знания']

        if any(indicator in context for indicator in expert_indicators):
            return 'expert'
        elif any(indicator in context for indicator in advanced_indicators):
            return 'advanced'
        elif any(indicator in context for indicator in basic_indicators):
            return 'basic'
        else:
            return 'intermediate'

    def _analyze_application_areas(self, context: str) -> List[str]:
        """Анализ областей применения"""
        areas = []

        area_keywords = {
            'web_development': ['веб', 'сайт', 'frontend', 'backend'],
            'mobile_development': ['мобильное', 'приложение', 'android', 'ios'],
            'data_analysis': ['анализ данных', 'аналитика', 'отчеты'],
            'testing': ['тестирование', 'автотесты', 'qa'],
            'devops': ['ci/cd', 'развертывание', 'инфраструктура']
        }

        for area, keywords in area_keywords.items():
            if any(keyword in context for keyword in keywords):
                areas.append(area)

        return areas

    def _analyze_project_roles(self, skill_name: str, context: str) -> List[str]:
        """Анализ ролей навыка в проектах"""
        roles = []

        if any(keyword in context for keyword in ['основной', 'главный', 'core']):
            roles.append('primary_technology')

        if any(keyword in context for keyword in ['поддержка', 'вспомогательный']):
            roles.append('supporting_technology')

        if any(keyword in context for keyword in ['новый', 'изучение']):
            roles.append('learning_technology')

        return roles or ['general_usage']


class SkillMarketAnalyzer:
    """Анализатор рыночной востребованности навыков"""

    def __init__(self):
        # Статические данные о рыночной востребованности (в реальном проекте можно подключить API)
        self.market_demand_scores = {
            'javascript': 0.95, 'python': 0.9, 'java': 0.85, 'typescript': 0.88,
            'react': 0.9, 'vue': 0.8, 'angular': 0.75, 'node.js': 0.85,
            'sql': 0.9, 'postgresql': 0.8, 'mongodb': 0.7, 'redis': 0.65,
            'docker': 0.85, 'kubernetes': 0.8, 'aws': 0.85, 'jenkins': 0.7,
            'git': 0.95, 'github': 0.9, 'cypress': 0.75, 'jest': 0.7
        }

    def get_market_demand_score(self, skill_name: str) -> float:
        """Получение оценки рыночной востребованности"""
        normalized_name = skill_name.lower()
        return self.market_demand_scores.get(normalized_name, 0.5)  # Дефолтное значение 0.5


class EducationLanguageExtractor:
    """Экстрактор образования с поддержкой русского языка"""
    
    def __init__(self):
        self.patterns = self._compile_patterns()
        
    def _compile_patterns(self) -> Dict[str, re.Pattern]:
        """Компиляция паттернов для извлечения образования"""
        return {
            'institution': re.compile(
                r'(университет|институт|академия|колледж|школа|гимназия|лицей|техникум|училище)',
                re.IGNORECASE
            ),
            'degree': re.compile(
                r'(бакалавр|магистр|специалист|аспирант|кандидат наук|доктор наук|MBA)',
                re.IGNORECASE
            ),
            'field_of_study': re.compile(
                r'(факультет|специальность|направление|кафедра)\s*[:\-]?\s*(.+?)(?:\n|$)',
                re.IGNORECASE
            ),
            'year': re.compile(r'\b(19|20)\d{2}\b'),
            'graduation': re.compile(
                r'(окончил|окончила|выпуск|год окончания)\s*[:\-]?\s*(\d{4})',
                re.IGNORECASE
            ),
            'study_period': re.compile(
                r'(\d{4})\s*[-—]\s*(\d{4})',
                re.IGNORECASE
            )
        }
    
    def extract_education(self, segments: Dict[str, Any]) -> Tuple[List[Education], Dict[str, Any]]:
        """Извлечение образования"""
        education_lines = segments.get('education', {}).get('lines', [])
        
        if not education_lines:
            return [], {'error': 'no_education_section'}
            
        education_entries = []
        current_education = None
        parsing_metadata = {'blocks_found': 0, 'parsing_errors': []}
        
        for i, line in enumerate(education_lines):
            line = line.strip()
            if not line:
                continue
                
            # Поиск учебного заведения
            institution_match = self.patterns['institution'].search(line)
            if institution_match and not current_education:
                current_education = Education()
                current_education.institution = line
                parsing_metadata['blocks_found'] += 1
                continue
                
            if current_education:
                # Поиск специальности/факультета
                field_match = self.patterns['field_of_study'].search(line)
                if field_match:
                    current_education.field_of_study = field_match.group(2).strip()
                    continue
                    
                # Поиск степени
                degree_match = self.patterns['degree'].search(line)
                if degree_match:
                    current_education.degree = degree_match.group(1)
                    continue
                    
                # Поиск годов обучения
                graduation_match = self.patterns['graduation'].search(line)
                if graduation_match:
                    current_education.graduation_date = graduation_match.group(2)
                    continue
                    
                study_period_match = self.patterns['study_period'].search(line)
                if study_period_match:
                    current_education.start_date = study_period_match.group(1)
                    current_education.graduation_date = study_period_match.group(2)
                    continue
                    
                # Поиск года (если не было периода)
                year_match = self.patterns['year'].search(line)
                if year_match and not current_education.graduation_date:
                    current_education.graduation_date = year_match.group(0)
                    continue
                    
                # Если встретили новое учебное заведение, сохраняем текущее
                new_institution = self.patterns['institution'].search(line)
                if new_institution:
                    if current_education.institution:
                        current_education.confidence_score = self._calculate_education_confidence(current_education)
                        education_entries.append(current_education)
                    current_education = Education()
                    current_education.institution = line
                    parsing_metadata['blocks_found'] += 1
        
        # Добавляем последнее образование
        if current_education and current_education.institution:
            current_education.confidence_score = self._calculate_education_confidence(current_education)
            education_entries.append(current_education)
            
        return education_entries, parsing_metadata
    
    def _calculate_education_confidence(self, education: Education) -> float:
        """Расчет уверенности для записи об образовании"""
        confidence = 0.0
        
        if education.institution:
            confidence += 0.4
        if education.degree:
            confidence += 0.2
        if education.field_of_study:
            confidence += 0.2
        if education.graduation_date:
            confidence += 0.2
            
        return min(confidence, 1.0)


class UltraPreciseResumeParser:
    """Ультра-точный парсер резюме с ML и статистическим анализом"""

    def __init__(self):
        # Инициализация компонентов
        self.preprocessor = AdvancedTextPreprocessor()
        self.segmenter = MLTextSegmenter()
        self.personal_extractor = PrecisionPersonalInfoExtractor()
        self.experience_extractor = MLWorkExperienceExtractor()
        self.skills_extractor = HighPrecisionSkillsExtractor()
        self.education_extractor = EducationLanguageExtractor()

        # Метрики и логирование
        self.parsing_metrics = {}
        self.setup_logging()

    def setup_logging(self):
        """Настройка логирования"""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler('ultra_precise_parser.log'),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)

    def parse_resume_text(self, text: str) -> ResumeData:
        """Ультра-точный парсинг резюме"""
        start_time = datetime.now()
        self.logger.info(f"🚀 Начинаем ультра-точный парсинг резюме ({len(text)} символов)")

        try:
            # 1. Предобработка текста
            self.logger.info("📝 Этап 1: Предобработка текста")
            preprocessed_data = self.preprocessor.preprocess_text(text)

            # 2. ML сегментация
            self.logger.info("🔍 Этап 2: ML сегментация")
            segments = self.segmenter.segment_resume(preprocessed_data)

            # 3. Извлечение личной информации
            self.logger.info("👤 Этап 3: Извлечение личной информации")
            personal_info = self.personal_extractor.extract_personal_info(segments)

            # 4. Анализ предпочтений по работе
            self.logger.info("💼 Этап 4: Анализ предпочтений по работе")
            job_preferences = self._extract_job_preferences(segments)

            # 5. ML анализ опыта работы
            self.logger.info("🏢 Этап 5: ML анализ опыта работы")
            work_experience, total_exp_months, exp_metadata = self.experience_extractor.extract_work_experience(
                segments)

            # 6. Высокоточное извлечение навыков
            self.logger.info("🛠️ Этап 6: Высокоточное извлечение навыков")
            skills = self.skills_extractor.extract_skills(segments, work_experience)

            # 7. Анализ образования и языков
            self.logger.info("🎓 Этап 7: Анализ образования и языков")
            education = self.education_extractor.extract_education(segments)
            languages = self.education_extractor.extract_languages(segments)

            # 8. Дополнительная информация
            self.logger.info("📋 Этап 8: Дополнительная информация")
            additional_info = self._extract_additional_info(segments)
            last_updated = self._extract_last_updated(text)

            # 9. ML метрики и анализ
            self.logger.info("🤖 Этап 9: ML анализ и метрики")
            ml_metrics = self._calculate_ml_metrics(work_experience, skills)

            # 10. Создание объекта резюме
            resume_data = ResumeData(
                source_text=text,
                personal_info=personal_info,
                job_preferences=job_preferences,
                work_experience=work_experience,
                education=education,
                skills=skills,
                languages=languages,
                additional_info=additional_info,
                last_updated=last_updated,
                total_experience_months=total_exp_months,
                seniority_assessment=ml_metrics.get('seniority_assessment'),
                skill_diversity_score=ml_metrics.get('skill_diversity_score'),
                career_progression_score=ml_metrics.get('career_progression_score'),
                overall_quality_score=ml_metrics.get('overall_quality_score')
            )

            # 11. Финальная оценка уверенности
            self.logger.info("📊 Этап 10: Финальная оценка уверенности")
            parsing_confidence = self._calculate_ultra_precise_confidence(resume_data, segments, preprocessed_data)
            resume_data.parsing_confidence = parsing_confidence

            # 12. Метаданные извлечения
            extraction_metadata = self._compile_extraction_metadata(
                preprocessed_data, segments, exp_metadata, ml_metrics, start_time
            )
            resume_data.extraction_metadata = extraction_metadata

            end_time = datetime.now()
            processing_time = (end_time - start_time).total_seconds()

            self.logger.info(f"✅ Парсинг завершен за {processing_time:.2f}с. Уверенность: {parsing_confidence:.1%}")

            return resume_data

        except Exception as e:
            self.logger.error(f"❌ Критическая ошибка парсинга: {e}")
            raise

    def _extract_job_preferences(self, segments: Dict[str, Any]) -> JobPreferences:
        """Извлечение предпочтений по работе"""
        preferences = JobPreferences()
        job_lines = segments.get('job_preferences', {}).get('lines', [])

        if not job_lines:
            return preferences

        # Желаемая должность
        for line in job_lines:
            if any(keyword in line.lower() for keyword in ['разработчик', 'developer', 'программист']):
                preferences.desired_position = line.strip()
                break

        # Специализации
        specializations = []
        in_specializations = False
        for line in job_lines:
            if 'Специализации:' in line:
                in_specializations = True
                continue

            if in_specializations:
                if line.startswith('—') or line.startswith('-'):
                    spec = line.replace('—', '').replace('-', '').strip()
                    if spec:
                        specializations.append(spec)
                elif line.startswith('Занятость:'):
                    in_specializations = False

        preferences.specializations = specializations

        # Тип занятости и график
        for line in job_lines:
            if line.startswith('Занятость:'):
                preferences.employment_type = line.replace('Занятость:', '').strip()
            elif line.startswith('График работы:'):
                preferences.work_schedule = line.replace('График работы:', '').strip()
            elif 'время в пути' in line.lower():
                preferences.travel_time_preference = line.split(':')[-1].strip()

        # Расчет уверенности
        confidence_metrics = self._calculate_job_preferences_confidence(preferences, job_lines)
        preferences.confidence = confidence_metrics

        return preferences

    def _calculate_job_preferences_confidence(self, preferences: JobPreferences,
                                              job_lines: List[str]) -> ConfidenceMetrics:
        """Расчет уверенности в предпочтениях по работе"""
        field_scores = {
            'desired_position': 0.9 if preferences.desired_position else 0.0,
            'specializations': min(len(preferences.specializations) * 0.5, 1.0),
            'employment_type': 0.8 if preferences.employment_type else 0.0,
            'work_schedule': 0.8 if preferences.work_schedule else 0.0
        }

        validation_results = {
            'has_position': bool(preferences.desired_position),
            'has_specializations': len(preferences.specializations) > 0,
            'has_employment_details': bool(preferences.employment_type and preferences.work_schedule)
        }

        overall_score = statistics.mean(field_scores.values())

        return ConfidenceMetrics(
            overall_score=overall_score,
            field_scores=field_scores,
            validation_results=validation_results,
            extraction_method='precise_job_preferences_extractor',
            reliability_factors={'section_completeness': len(job_lines) / 10}
        )

    def _extract_additional_info(self, segments: Dict[str, Any]) -> Optional[str]:
        """Извлечение дополнительной информации"""
        additional_lines = segments.get('additional', {}).get('lines', [])
        if additional_lines:
            return '\n'.join(additional_lines)
        return None

    def _extract_last_updated(self, text: str) -> Optional[str]:
        """Извлечение даты обновления резюме"""
        update_pattern = re.compile(r'обновлено\s*(\d{1,2}\s+\w+\s+\d{4})', re.IGNORECASE)
        match = update_pattern.search(text)
        if match:
            return match.group(1)
        return None

    def _calculate_ml_metrics(self, work_experience: List[WorkExperience], skills: List[Skill]) -> Dict[str, Any]:
        """Расчет ML метрик"""
        # Анализ сениорности
        seniority_scores = []
        for exp in work_experience:
            if hasattr(exp, 'seniority_level') and exp.seniority_level:
                level_score = {'junior': 0.3, 'middle': 0.6, 'senior': 0.9, 'lead': 1.0}.get(exp.seniority_level, 0.5)
                seniority_scores.append(level_score)

        seniority_assessment = 'unknown'
        if seniority_scores:
            avg_seniority = statistics.mean(seniority_scores)
            if avg_seniority >= 0.8:
                seniority_assessment = 'senior'
            elif avg_seniority >= 0.5:
                seniority_assessment = 'middle'
            else:
                seniority_assessment = 'junior'

        # Разнообразие навыков
        skill_categories = set(skill.category for skill in skills if skill.category)
        skill_diversity_score = min(len(skill_categories) / 6, 1.0)  # Нормализация к 6 категориям

        # Карьерный прогресс
        career_progression_score = 0.5
        if len(work_experience) > 1:
            career_progression_score += 0.2

        total_projects = sum(len(exp.projects) for exp in work_experience)
        career_progression_score += min(total_projects * 0.05, 0.3)

        # Общее качество
        overall_quality_score = statistics.mean([
            statistics.mean(seniority_scores) if seniority_scores else 0.5,
            skill_diversity_score,
            career_progression_score
        ])

        return {
            'seniority_assessment': seniority_assessment,
            'seniority_scores': seniority_scores,
            'skill_diversity_score': skill_diversity_score,
            'career_progression_score': career_progression_score,
            'overall_quality_score': overall_quality_score,
            'total_projects': total_projects,
            'skill_categories_count': len(skill_categories)
        }

    def _calculate_ultra_precise_confidence(self, resume_data: ResumeData, segments: Dict[str, Any],
                                            preprocessed_data: Dict[str, Any]) -> float:
        """Ультра-точный расчет уверенности парсинга"""

        # Весовые коэффициенты для разных компонентов
        weights = {
            'personal_info': 0.25,
            'work_experience': 0.35,
            'skills': 0.20,
            'education': 0.10,
            'text_quality': 0.10
        }

        scores = {}

        # 1. Оценка личной информации
        personal_confidence = resume_data.personal_info.confidence
        if personal_confidence:
            scores['personal_info'] = personal_confidence.overall_score
        else:
            scores['personal_info'] = 0.5

        # 2. Оценка опыта работы
        if resume_data.work_experience:
            exp_scores = []
            for exp in resume_data.work_experience:
                if hasattr(exp, 'confidence') and exp.confidence:
                    exp_scores.append(exp.confidence.overall_score)
                else:
                    exp_scores.append(0.5)
            scores['work_experience'] = statistics.mean(exp_scores) if exp_scores else 0.0
        else:
            scores['work_experience'] = 0.0

        # 3. Оценка навыков
        if resume_data.skills:
            skill_confidences = [skill.confidence_score for skill in resume_data.skills if skill.confidence_score]
            scores['skills'] = statistics.mean(skill_confidences) if skill_confidences else 0.5
        else:
            scores['skills'] = 0.0

        # 4. Оценка образования
        scores['education'] = 0.8 if resume_data.education else 0.3

        # 5. Качество текста
        text_stats = preprocessed_data.get('statistics', {})
        text_quality = min(
            text_stats.get('vocabulary_richness', 0) * 2,  # Богатство словаря
            len(resume_data.source_text) / 3000,  # Адекватная длина
            1.0
        )
        scores['text_quality'] = text_quality

        # Взвешенная сумма
        weighted_confidence = sum(scores[component] * weights[component] for component in weights.keys())

        # Бонусы за полноту
        completeness_bonus = 0.0
        if resume_data.personal_info.full_name and resume_data.personal_info.email:
            completeness_bonus += 0.05
        if len(resume_data.work_experience) > 0:
            completeness_bonus += 0.05
        if len(resume_data.skills) > 5:
            completeness_bonus += 0.05

        final_confidence = min(weighted_confidence + completeness_bonus, 1.0)

        return final_confidence

    def _compile_extraction_metadata(self, preprocessed_data: Dict, segments: Dict, exp_metadata: Dict,
                                     ml_metrics: Dict, start_time: datetime) -> Dict[str, Any]:
        """Компиляция метаданных извлечения"""
        end_time = datetime.now()

        return {
            'parsing_timestamp': end_time.isoformat(),
            'processing_time_seconds': (end_time - start_time).total_seconds(),
            'parser_version': '3.0_ultra_precise',
            'text_statistics': preprocessed_data.get('statistics', {}),
            'segmentation_confidence': {
                segment_name: segment_data.get('confidence', 0.0)
                for segment_name, segment_data in segments.items()
                if isinstance(segment_data, dict)
            },
            'experience_metadata': exp_metadata,
            'ml_metrics': ml_metrics,
            'extraction_stages': [
                'text_preprocessing',
                'ml_segmentation',
                'personal_info_extraction',
                'job_preferences_analysis',
                'ml_work_experience_analysis',
                'precision_skills_extraction',
                'education_language_analysis',
                'ml_metrics_calculation',
                'confidence_assessment'
            ]
        }

    def parse_to_json(self, text: str, output_path: Optional[str] = None) -> str:
        """Парсинг в JSON с метаданными"""
        resume_data = self.parse_resume_text(text)

        # Конвертация в словарь
        resume_dict = asdict(resume_data)

        # Форматирование JSON
        json_output = json.dumps(resume_dict, ensure_ascii=False, indent=2)

        # Сохранение в файл
        if output_path:
            output_file = Path(output_path)
            output_file.parent.mkdir(parents=True, exist_ok=True)
            with open(output_file, 'w', encoding='utf-8') as f:
                f.write(json_output)
            self.logger.info(f"💾 JSON сохранен: {output_file}")

        return json_output

    def generate_detailed_report(self, resume_data: ResumeData) -> str:
        """Генерация детального отчета анализа"""
        lines = []

        lines.append("🎯 УЛЬТРА-ТОЧНЫЙ АНАЛИЗ РЕЗЮМЕ")
        lines.append("=" * 80)

        # Основная информация
        lines.append(f"👤 Кандидат: {resume_data.personal_info.full_name}")
        lines.append(f"💼 Желаемая должность: {resume_data.job_preferences.desired_position}")
        lines.append(f"📧 Email: {resume_data.personal_info.email}")
        lines.append(f"📱 Телефон: {resume_data.personal_info.phone}")
        lines.append(f"🏙️ Город: {resume_data.personal_info.city}")
        lines.append(f"🎯 Уверенность парсинга: {resume_data.parsing_confidence:.1%}")

        # ML метрики
        lines.append(f"\n🤖 ML АНАЛИЗ:")
        lines.append(f"⭐ Уровень сениорности: {resume_data.seniority_assessment}")
        lines.append(f"🔧 Разнообразие навыков: {resume_data.skill_diversity_score:.1%}")
        lines.append(f"📈 Карьерный прогресс: {resume_data.career_progression_score:.1%}")
        lines.append(f"🏆 Общее качество: {resume_data.overall_quality_score:.1%}")

        # Опыт работы
        lines.append(f"\n💼 ОПЫТ РАБОТЫ:")
        if resume_data.total_experience_months:
            years = resume_data.total_experience_months // 12
            months = resume_data.total_experience_months % 12
            lines.append(f"⏱️ Общий опыт: {years} лет {months} месяцев")

        for i, exp in enumerate(resume_data.work_experience, 1):
            lines.append(f"\n{i}. 🏢 {exp.company} - {exp.position}")
            lines.append(f"   📅 {exp.start_date} — {exp.end_date}")
            if hasattr(exp, 'seniority_level') and exp.seniority_level:
                lines.append(f"   ⭐ Уровень: {exp.seniority_level}")
            if exp.projects:
                lines.append(f"   📋 Проектов: {len(exp.projects)}")
                for j, project in enumerate(exp.projects[:2], 1):  # Показываем первые 2
                    lines.append(f"     {j}. {project.name}")
                    if hasattr(project.tech_stack, 'primary_technologies'):
                        tech_str = ", ".join(project.tech_stack.primary_technologies[:4])
                        if tech_str:
                            lines.append(f"        Стек: {tech_str}")

        # Навыки по категориям
        lines.append(f"\n🛠️ НАВЫКИ ПО КАТЕГОРИЯМ ({len(resume_data.skills)}):")

        skills_by_category = {}
        for skill in resume_data.skills:
            category = skill.category or 'Прочие'
            if category not in skills_by_category:
                skills_by_category[category] = []
            skills_by_category[category].append(skill)

        for category, skills in skills_by_category.items():
            lines.append(f"\n  📂 {category}:")
            for skill in sorted(skills, key=lambda x: x.confidence_score or 0, reverse=True)[:6]:
                confidence_str = f" ({skill.confidence_score:.1%})" if skill.confidence_score else ""
                mentions_str = f" [📋{skill.mentioned_in_projects}]" if skill.mentioned_in_projects > 0 else ""
                market_str = f" [📊{skill.market_demand_score:.1%}]" if skill.market_demand_score else ""
                lines.append(f"    • {skill.name}{confidence_str}{mentions_str}{market_str}")

        # Образование
        lines.append(f"\n🎓 ОБРАЗОВАНИЕ:")
        for edu in resume_data.education:
            lines.append(f"🏫 {edu.institution}")
            lines.append(f"📚 {edu.level} - {edu.specialization}")
            if edu.graduation_year:
                lines.append(f"📅 {edu.graduation_year}")

        # Языки
        if resume_data.languages:
            lines.append(f"\n🌐 ЯЗЫКИ:")
            for lang in resume_data.languages:
                lines.append(f"🗣️ {lang.language} — {lang.level}")

        # Метаданные извлечения
        if resume_data.extraction_metadata:
            metadata = resume_data.extraction_metadata
            lines.append(f"\n📊 МЕТАДАННЫЕ ИЗВЛЕЧЕНИЯ:")
            lines.append(f"⏱️ Время обработки: {metadata.get('processing_time_seconds', 0):.2f}с")
            lines.append(f"🔢 Версия парсера: {metadata.get('parser_version', 'unknown')}")

            if 'text_statistics' in metadata:
                stats = metadata['text_statistics']
                lines.append(f"📝 Статистика текста:")
                lines.append(f"   • Символов: {stats.get('total_chars', 0)}")
                lines.append(f"   • Слов: {stats.get('total_words', 0)}")
                lines.append(f"   • Уникальных слов: {stats.get('unique_words', 0)}")
                lines.append(f"   • Богатство словаря: {stats.get('vocabulary_richness', 0):.1%}")

        lines.append("\n" + "=" * 80)
        return "\n".join(lines)


# Продолжение функции main()

def main():
    """Демонстрация ультра-точного парсера"""

    # Тестовый текст резюме
    resume_text = """
Ивлев Андрей

Мужчина, 21 год, родился 10 сентября 2003

+7 (967) 8272004 — предпочитаемый способ связи  •  telegram @andreyIvlevKzn
pulemetik.ai@gmail.com

Проживает: Казань
Гражданство: Россия, есть разрешение на работу: Россия
Не готов к переезду, готов к командировкам

Желаемая должность и зарплата

Frontend-разработчик
Специализации:

—  Программист, разработчик

Занятость: полная занятость
График работы: полный день, удаленная работа
Желательное время в пути до работы: не имеет значения

Опыт работы — 3 года 5 месяцев

Январь 2022 —
настоящее время
3 года 5 месяцев

The one group
Казань

Frontend-разработчик

Проект: Образовательная платформа (КазГИК university)
Стек: Vue | Vitest | Cypress | Pinia
Зона ответственности: Архитектура UI, Разработка, Автотесты

Проект: Сервис анализа конкурентов
Стек: Vue | Vitest | Jenkins
Зона ответственности: Архитектура UI, Разработка, Взаимодействие с бэкендом, CI/CD пайплайн

Проект: Система складского учёта The One Market
Стек: Vue | Vitest | Vuex | Ionic
Зона ответственности: Архитектура UI, Разработка админки, Разработка мобильного
приложения, Взаимодействие с бэкендом, Автотесты

Проект: Wb Wordstat
Стек: Vue2
Зона ответственности: Разработка, Взаимодействие с бэкендом, Работа с графиками

Проект: Сервис аналитики рекламы The One Group
Стек: Vue | Vitest | Jenkins
Зона ответственности: Архитектура UI, Разработка, Взаимодействие с бэкендом, CI/CD пайплайн

Проект: Freetracker (Платформа для сопровождения перегона грузовиков)
Стек: React | Jest | Cypress | RTK
Зона ответственности:  Архитектура UI, Разработка, Автотесты

Образование

Неоконченное высшее

Резюме обновлено 1 марта 2025 в 23:44

2026

Университет управления "ТИСБИ", Казань
Управления, Бизнес-информатика

Навыки

Знание языков

Русский — Родной
Английский — B2 — Средне-продвинутый
Немецкий — A1 — Начальный

Навыки

 Vue      React      Express      Vitest      Jest      Cypress      TypeScript      SQL 

 Jenkins      CI/CD      Node.js 

Дополнительная информация

Обо мне

Соцсети: https://vk.com/t323911867

Ивлев Андрей  •  Резюме обновлено 1 марта 2025 в 23:44
"""

    try:
        # Инициализация ультра-точного парсера
        parser = UltraPreciseResumeParser()

        print("🎯 ULTRA PRECISE RESUME PARSER v3.0")
        print("=" * 80)
        print("🤖 Парсер с машинным обучением и статистическим анализом")
        print("⚡ Цель: достижение точности >90%")
        print("=" * 80)

        # Парсинг резюме
        resume_data = parser.parse_resume_text(resume_text)

        # Генерация детального отчета
        detailed_report = parser.generate_detailed_report(resume_data)
        print(detailed_report)

        # Сохранение в JSON
        json_output = parser.parse_to_json(resume_text, "ultra_precise_resume.json")

        print(f"\n💾 JSON файл создан: ultra_precise_resume.json")
        print(f"📊 Размер JSON: {len(json_output)} символов")

        # Детальная проверка качества извлечения
        print(f"\n🔍 ДЕТАЛЬНАЯ ПРОВЕРКА КАЧЕСТВА:")

        quality_checks = {
            "ФИО": resume_data.personal_info.full_name == "Ивлев Андрей",
            "Email": resume_data.personal_info.email == "pulemetik.ai@gmail.com",
            "Телефон": resume_data.personal_info.phone == "+7 (967) 8272004",
            "Город": resume_data.personal_info.city == "Казань",
            "Возраст": resume_data.personal_info.age == 21,
            "Telegram": resume_data.personal_info.telegram == "@andreyIvlevKzn",
            "Готовность к переезду": resume_data.personal_info.ready_to_relocate == False,
            "Готовность к командировкам": resume_data.personal_info.ready_for_business_trips == True,
            "Желаемая должность": "Frontend" in str(resume_data.job_preferences.desired_position),
            "Занятость": "полная" in str(resume_data.job_preferences.employment_type),
            "Опыт работы": len(resume_data.work_experience) >= 1,
            "Компания": any("The one group" in str(exp.company) for exp in resume_data.work_experience),
            "Текущая работа": any(exp.is_current for exp in resume_data.work_experience),
            "Проекты": sum(len(exp.projects) for exp in resume_data.work_experience) >= 5,
            "Технологии Vue": any("Vue" in skill.name for skill in resume_data.skills),
            "Технологии React": any("React" in skill.name for skill in resume_data.skills),
            "Образование": len(resume_data.education) >= 1,
            "Уровень образования": any("Неоконченное высшее" in str(edu.level) for edu in resume_data.education),
            "Языки": len(resume_data.languages) >= 3,
            "Русский язык": any("Русский" in lang.language for lang in resume_data.languages),
            "Английский B2": any(
                "Английский" in lang.language and "B2" in str(lang.level) for lang in resume_data.languages),
        }

        passed_checks = sum(quality_checks.values())
        total_checks = len(quality_checks)
        accuracy_percentage = (passed_checks / total_checks) * 100

        for check_name, passed in quality_checks.items():
            status = "✅" if passed else "❌"
            print(f"  {status} {check_name}")

        print(f"\n📈 ИТОГОВАЯ ТОЧНОСТЬ: {accuracy_percentage:.1f}% ({passed_checks}/{total_checks})")

        # Анализ уверенности по компонентам
        print(f"\n🎯 АНАЛИЗ УВЕРЕННОСТИ ПО КОМПОНЕНТАМ:")

        if resume_data.personal_info.confidence:
            print(f"👤 Личная информация: {resume_data.personal_info.confidence.overall_score:.1%}")
            for field, score in resume_data.personal_info.confidence.field_scores.items():
                print(f"   • {field}: {score:.1%}")

        if resume_data.job_preferences.confidence:
            print(f"💼 Предпочтения по работе: {resume_data.job_preferences.confidence.overall_score:.1%}")

        if resume_data.work_experience:
            exp_confidences = []
            for i, exp in enumerate(resume_data.work_experience):
                if hasattr(exp, 'confidence') and exp.confidence:
                    exp_confidences.append(exp.confidence.overall_score)
                    print(f"🏢 Опыт работы {i + 1}: {exp.confidence.overall_score:.1%}")

            if exp_confidences:
                avg_exp_confidence = statistics.mean(exp_confidences)
                print(f"📊 Средняя уверенность в опыте: {avg_exp_confidence:.1%}")

        if resume_data.skills:
            skill_confidences = [skill.confidence_score for skill in resume_data.skills if skill.confidence_score]
            if skill_confidences:
                avg_skill_confidence = statistics.mean(skill_confidences)
                print(f"🛠️ Средняя уверенность в навыках: {avg_skill_confidence:.1%}")

        # Производительность парсера
        if resume_data.extraction_metadata:
            metadata = resume_data.extraction_metadata
            print(f"\n⚡ ПРОИЗВОДИТЕЛЬНОСТЬ:")
            print(f"⏱️ Время обработки: {metadata.get('processing_time_seconds', 0):.2f} секунд")
            print(f"🔢 Версия парсера: {metadata.get('parser_version', 'unknown')}")
            print(f"📈 Этапов обработки: {len(metadata.get('extraction_stages', []))}")

        # Рекомендации по улучшению
        print(f"\n💡 РЕКОМЕНДАЦИИ ПО УЛУЧШЕНИЮ:")
        if accuracy_percentage < 90:
            print("🔧 Необходимо улучшить:")
            for check_name, passed in quality_checks.items():
                if not passed:
                    print(f"   ❌ {check_name}")
        else:
            print("🎉 Отличная работа! Точность парсинга превышает 90%")

        # ML insights
        print(f"\n🤖 ML ИНСАЙТЫ:")
        print(f"⭐ Определенный уровень: {resume_data.seniority_assessment}")
        print(f"🎯 Оценка качества резюме: {resume_data.overall_quality_score:.1%}")
        print(f"📈 Карьерный прогресс: {resume_data.career_progression_score:.1%}")
        print(f"🔧 Разнообразие навыков: {resume_data.skill_diversity_score:.1%}")

        # Топ навыки с уверенностью
        print(f"\n🏆 ТОП-10 НАВЫКОВ С ВЫСОКОЙ УВЕРЕННОСТЬЮ:")
        top_skills = sorted(resume_data.skills, key=lambda x: x.confidence_score or 0, reverse=True)[:10]
        for i, skill in enumerate(top_skills, 1):
            confidence_str = f"{skill.confidence_score:.1%}" if skill.confidence_score else "N/A"
            category_str = f"[{skill.category}]" if skill.category else ""
            mentions_str = f"({skill.mentioned_in_projects} проектов)" if skill.mentioned_in_projects > 0 else ""
            print(f"  {i:2d}. {skill.name} {category_str} - {confidence_str} {mentions_str}")

        print(f"\n🎯 ФИНАЛЬНАЯ ОЦЕНКА ПАРСЕРА:")
        print(f"📊 Точность извлечения: {accuracy_percentage:.1f}%")
        print(f"🤖 ML уверенность: {resume_data.parsing_confidence:.1%}")

        if accuracy_percentage >= 90 and resume_data.parsing_confidence >= 0.9:
            print("🏆 ОТЛИЧНО! Цель достигнута - точность >90%")
        elif accuracy_percentage >= 85:
            print("🎯 ХОРОШО! Близко к цели, требуется небольшая доработка")
        else:
            print("⚠️ ТРЕБУЕТСЯ УЛУЧШЕНИЕ: точность ниже 85%")

    except Exception as e:
        print(f"❌ Критическая ошибка: {e}")
        import traceback
        traceback.print_exc()


# Дополнительные утилиты для тестирования и валидации

class ResumeParserValidator:
    """Валидатор для проверки качества парсинга резюме"""

    def __init__(self):
        self.validation_rules = {
            'personal_info': {
                'full_name': {'required': True, 'type': str, 'min_words': 2},
                'email': {'required': True, 'type': str, 'pattern': r'^[^@]+@[^@]+\.[^@]+$'},
                'phone': {'required': False, 'type': str, 'pattern': r'\+?[\d\s\-\(\)]+'},
                'age': {'required': False, 'type': int, 'min_value': 16, 'max_value': 70},
                'city': {'required': False, 'type': str, 'min_length': 2}
            },
            'work_experience': {
                'min_entries': 0,
                'required_fields': ['company', 'position'],
                'project_validation': True
            },
            'skills': {
                'min_count': 3,
                'confidence_threshold': 0.5
            },
            'education': {
                'min_entries': 0,
                'required_fields': ['institution', 'level']
            }
        }

    def validate_resume_data(self, resume_data: ResumeData) -> Dict[str, Any]:
        """Комплексная валидация извлеченных данных"""
        validation_results = {
            'overall_valid': True,
            'component_results': {},
            'errors': [],
            'warnings': [],
            'quality_score': 0.0
        }

        # Валидация личной информации
        personal_result = self._validate_personal_info(resume_data.personal_info)
        validation_results['component_results']['personal_info'] = personal_result

        # Валидация опыта работы
        experience_result = self._validate_work_experience(resume_data.work_experience)
        validation_results['component_results']['work_experience'] = experience_result

        # Валидация навыков
        skills_result = self._validate_skills(resume_data.skills)
        validation_results['component_results']['skills'] = skills_result

        # Валидация образования
        education_result = self._validate_education(resume_data.education)
        validation_results['component_results']['education'] = education_result

        # Общая оценка
        component_scores = [
            personal_result.get('score', 0),
            experience_result.get('score', 0),
            skills_result.get('score', 0),
            education_result.get('score', 0)
        ]

        validation_results['quality_score'] = statistics.mean(component_scores)
        validation_results['overall_valid'] = validation_results['quality_score'] >= 0.7

        return validation_results

    def _validate_personal_info(self, personal_info: PersonalInfo) -> Dict[str, Any]:
        """Валидация личной информации"""
        result = {'valid': True, 'score': 0.0, 'errors': [], 'field_scores': {}}

        rules = self.validation_rules['personal_info']

        # Проверка имени
        if personal_info.full_name:
            if len(personal_info.full_name.split()) >= rules['full_name']['min_words']:
                result['field_scores']['full_name'] = 1.0
            else:
                result['field_scores']['full_name'] = 0.5
                result['errors'].append("ФИО содержит менее 2 слов")
        else:
            result['field_scores']['full_name'] = 0.0
            result['errors'].append("ФИО не найдено")

        # Проверка email
        if personal_info.email:
            if re.match(rules['email']['pattern'], personal_info.email):
                result['field_scores']['email'] = 1.0
            else:
                result['field_scores']['email'] = 0.5
                result['errors'].append("Некорректный формат email")
        else:
            result['field_scores']['email'] = 0.0
            result['errors'].append("Email не найден")

        # Проверка телефона
        if personal_info.phone:
            if re.search(rules['phone']['pattern'], personal_info.phone):
                result['field_scores']['phone'] = 1.0
            else:
                result['field_scores']['phone'] = 0.5
        else:
            result['field_scores']['phone'] = 0.3

        # Проверка возраста
        if personal_info.age:
            age_rules = rules['age']
            if age_rules['min_value'] <= personal_info.age <= age_rules['max_value']:
                result['field_scores']['age'] = 1.0
            else:
                result['field_scores']['age'] = 0.0
                result['errors'].append(f"Некорректный возраст: {personal_info.age}")
        else:
            result['field_scores']['age'] = 0.5

        result['score'] = statistics.mean(result['field_scores'].values()) if result['field_scores'] else 0.0
        result['valid'] = result['score'] >= 0.7

        return result

    def _validate_work_experience(self, work_experience: List[WorkExperience]) -> Dict[str, Any]:
        """Валидация опыта работы"""
        result = {'valid': True, 'score': 0.0, 'errors': [], 'experience_scores': []}

        if not work_experience:
            result['score'] = 0.3  # Минимальный балл если опыт не найден
            result['errors'].append("Опыт работы не найден")
            return result

        for i, exp in enumerate(work_experience):
            exp_score = 0.0

            # Проверка обязательных полей
            if exp.company:
                exp_score += 0.3
            if exp.position:
                exp_score += 0.3
            if exp.start_date:
                exp_score += 0.2

            # Бонус за проекты
            if exp.projects:
                exp_score += min(len(exp.projects) * 0.05, 0.2)

            result['experience_scores'].append(exp_score)

        result['score'] = statistics.mean(result['experience_scores'])
        result['valid'] = result['score'] >= 0.6

        return result

    def _validate_skills(self, skills: List[Skill]) -> Dict[str, Any]:
        """Валидация навыков"""
        result = {'valid': True, 'score': 0.0, 'errors': []}

        if not skills:
            result['score'] = 0.0
            result['errors'].append("Навыки не найдены")
            return result

        # Количество навыков
        quantity_score = min(len(skills) / 10, 1.0)

        # Качество навыков (уверенность)
        confident_skills = [s for s in skills if s.confidence_score and s.confidence_score >= 0.5]
        quality_score = len(confident_skills) / len(skills) if skills else 0

        # Разнообразие категорий
        categories = set(skill.category for skill in skills if skill.category)
        diversity_score = min(len(categories) / 5, 1.0)

        result['score'] = (quantity_score * 0.4 + quality_score * 0.4 + diversity_score * 0.2)
        result['valid'] = result['score'] >= 0.5

        return result

    def _validate_education(self, education: List[Education]) -> Dict[str, Any]:
        """Валидация образования"""
        result = {'valid': True, 'score': 0.5, 'errors': []}  # Базовый балл

        if education:
            for edu in education:
                if edu.institution and edu.level:
                    result['score'] = 1.0
                    break

        result['valid'] = result['score'] >= 0.5
        return result


class PerformanceBenchmark:
    """Бенчмарк для измерения производительности парсера"""

    def __init__(self):
        self.metrics = {
            'parsing_times': [],
            'accuracy_scores': [],
            'confidence_scores': [],
            'memory_usage': []
        }

    def benchmark_parser(self, parser: UltraPreciseResumeParser, test_resumes: List[str], iterations: int = 1) -> Dict[
        str, Any]:
        """Бенчмарк парсера на тестовых резюме"""
        import time
        import psutil
        import os

        process = psutil.Process(os.getpid())

        for iteration in range(iterations):
            for i, resume_text in enumerate(test_resumes):
                # Замер времени
                start_time = time.time()
                start_memory = process.memory_info().rss / 1024 / 1024  # MB

                try:
                    # Парсинг
                    resume_data = parser.parse_resume_text(resume_text)

                    # Метрики
                    end_time = time.time()
                    end_memory = process.memory_info().rss / 1024 / 1024  # MB

                    parsing_time = end_time - start_time
                    memory_usage = end_memory - start_memory

                    self.metrics['parsing_times'].append(parsing_time)
                    self.metrics['confidence_scores'].append(resume_data.parsing_confidence or 0)
                    self.metrics['memory_usage'].append(memory_usage)

                    print(f"Резюме {i + 1}/{len(test_resumes)} (итерация {iteration + 1}): "
                          f"{parsing_time:.2f}с, уверенность: {resume_data.parsing_confidence:.1%}")

                except Exception as e:
                    print(f"Ошибка парсинга резюме {i + 1}: {e}")

        return self._calculate_benchmark_results()

    def _calculate_benchmark_results(self) -> Dict[str, Any]:
        """Расчет результатов бенчмарка"""
        if not self.metrics['parsing_times']:
            return {'error': 'Нет данных для анализа'}

        return {
            'average_parsing_time': statistics.mean(self.metrics['parsing_times']),
            'median_parsing_time': statistics.median(self.metrics['parsing_times']),
            'max_parsing_time': max(self.metrics['parsing_times']),
            'min_parsing_time': min(self.metrics['parsing_times']),
            'average_confidence': statistics.mean(self.metrics['confidence_scores']),
            'confidence_std': statistics.stdev(self.metrics['confidence_scores']) if len(
                self.metrics['confidence_scores']) > 1 else 0,
            'average_memory_usage': statistics.mean(self.metrics['memory_usage']),
            'total_tests': len(self.metrics['parsing_times'])
        }


if __name__ == "__main__":
    # Запуск основной демонстрации
    main()

    # Дополнительное тестирование валидатора
    print("\n" + "=" * 80)
    print("🧪 ДОПОЛНИТЕЛЬНОЕ ТЕСТИРОВАНИЕ ВАЛИДАТОРА")
    print("=" * 80)

    try:
        # Создаем тестовые данные
        validator = ResumeParserValidator()
        parser = UltraPreciseResumeParser()

        # Тестовое резюме для валидации
        test_resume = """
        Тестов Тест Тестович

        Мужчина, 25 лет
        test@example.com
        +7 (999) 999-99-99

        Желаемая должность: Python Developer

        Опыт работы:
        2020-2023 ООО "Тест" - Python Developer

        Навыки: Python, Django, PostgreSQL

        Образование: МГУ, 2020
        """

        # Парсинг и валидация
        resume_data = parser.parse_resume_text(test_resume)
        validation_results = validator.validate_resume_data(resume_data)

        print(f"✅ Валидация завершена!")
        print(f"📊 Общая оценка качества: {validation_results['quality_score']:.1%}")
        print(f"🎯 Результат валидации: {'ПРОЙДЕНА' if validation_results['overall_valid'] else 'НЕ ПРОЙДЕНА'}")

        for component, result in validation_results['component_results'].items():
            print(f"  📋 {component}: {result.get('score', 0):.1%}")
            if result.get('errors'):
                for error in result['errors']:
                    print(f"    ⚠️ {error}")

    except Exception as e:
        print(f"❌ Ошибка тестирования валидатора: {e}")