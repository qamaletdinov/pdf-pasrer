"""
Детектор формата резюме с ML классификацией
"""

import re
import logging
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass
from collections import Counter

from ..core.base_classes import BaseDetector, BaseMLComponent
from ..core.exceptions import MLModelError
from ..utils.logger import get_logger

# ML библиотеки (опциональные)
try:
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.metrics.pairwise import cosine_similarity
    import numpy as np
except ImportError:
    TfidfVectorizer = None
    cosine_similarity = None
    np = None


@dataclass
class FormatDetectionResult:
    """Результат детекции формата"""
    format_type: str
    confidence_score: float
    detected_features: List[str]
    format_specific_data: Dict[str, Any]


class ResumeFormatDetector(BaseDetector):
    """Детектор формата резюме"""
    
    def __init__(self):
        self.logger = get_logger(__name__)
        self.format_patterns = self._initialize_format_patterns()
        self.ml_classifier = MLFormatClassifier() if TfidfVectorizer else None
    
    def _initialize_format_patterns(self) -> Dict[str, Dict[str, Any]]:
        """Инициализация паттернов форматов"""
        return {
            'hh_ru': {
                'keywords': [
                    'hh.ru', 'headhunter', 'хедхантер',
                    'желаемая должность', 'специализации',
                    'опыт работы', 'ключевые навыки',
                    'о себе', 'дополнительная информация'
                ],
                'patterns': [
                    r'Желаемая должность\s*:',
                    r'Специализации\s*:',
                    r'Тип занятости\s*:',
                    r'График работы\s*:',
                    r'Желаемая зарплата\s*:',
                    r'Опыт работы\s*—',
                    r'Ключевые навыки\s*:'
                ],
                'structure_indicators': [
                    'employment_type_section',
                    'work_schedule_section',
                    'salary_section'
                ]
            },
            'linkedin': {
                'keywords': [
                    'linkedin', 'experience', 'education', 'skills',
                    'summary', 'recommendations', 'certifications',
                    'volunteer experience', 'publications'
                ],
                'patterns': [
                    r'Experience\s*$',
                    r'Education\s*$',
                    r'Skills\s*$',
                    r'Summary\s*$',
                    r'Recommendations\s*$',
                    r'at\s+[A-Z][a-zA-Z\s]+\s*$',  # "at Company Name"
                    r'\d+\s+years?\s+\d+\s+months?'
                ],
                'structure_indicators': [
                    'experience_at_format',
                    'duration_format',
                    'recommendations_section'
                ]
            },
            'academic': {
                'keywords': [
                    'curriculum vitae', 'cv', 'publications',
                    'research', 'academic', 'conference',
                    'journal', 'thesis', 'dissertation',
                    'research interests', 'teaching experience'
                ],
                'patterns': [
                    r'Curriculum\s+Vitae',
                    r'Research\s+Interests',
                    r'Publications\s*:',
                    r'Teaching\s+Experience',
                    r'Conference\s+Presentations',
                    r'Grants\s+and\s+Awards',
                    r'PhD\s+|Ph\.D\.',
                    r'Master\s+of\s+Science|M\.S\.'
                ],
                'structure_indicators': [
                    'publications_section',
                    'research_section',
                    'academic_positions'
                ]
            },
            'european': {
                'keywords': [
                    'europass', 'personal information',
                    'work experience', 'education and training',
                    'personal skills', 'additional information',
                    'driving licence', 'nationality'
                ],
                'patterns': [
                    r'Personal\s+information',
                    r'Work\s+experience',
                    r'Education\s+and\s+training',
                    r'Personal\s+skills',
                    r'Driving\s+licence',
                    r'Nationality\s*:'
                ],
                'structure_indicators': [
                    'europass_structure',
                    'nationality_field',
                    'driving_licence_section'
                ]
            },
            'generic': {
                'keywords': [
                    'resume', 'резюме', 'experience', 'опыт',
                    'education', 'образование', 'skills', 'навыки',
                    'contact', 'контакты'
                ],
                'patterns': [
                    r'(?i)resume',
                    r'(?i)резюме',
                    r'(?i)experience',
                    r'(?i)education',
                    r'(?i)skills'
                ],
                'structure_indicators': []
            }
        }
    
    def detect(self, text: str) -> str:
        """Основная функция детекции формата"""
        detection_result = self.detect_with_details(text)
        return detection_result.format_type
    
    def detect_with_details(self, text: str) -> FormatDetectionResult:
        """Детекция формата с подробной информацией"""
        # Нормализация текста
        normalized_text = self._normalize_text(text)
        
        # Детекция на основе паттернов
        pattern_scores = self._calculate_pattern_scores(normalized_text)
        
        # ML классификация (если доступна)
        ml_scores = {}
        if self.ml_classifier:
            try:
                ml_scores = self.ml_classifier.classify(normalized_text)
            except Exception as e:
                self.logger.warning(f"ML классификация недоступна: {e}")
        
        # Комбинирование результатов
        combined_scores = self._combine_scores(pattern_scores, ml_scores)
        
        # Выбор лучшего формата
        best_format = max(combined_scores.items(), key=lambda x: x[1])
        format_type, confidence = best_format
        
        # Извлечение специфичных для формата данных
        format_data = self._extract_format_specific_data(normalized_text, format_type)
        
        # Выявленные признаки
        detected_features = self._get_detected_features(normalized_text, format_type)
        
        return FormatDetectionResult(
            format_type=format_type,
            confidence_score=confidence,
            detected_features=detected_features,
            format_specific_data=format_data
        )
    
    def _normalize_text(self, text: str) -> str:
        """Нормализация текста для анализа"""
        # Удаление лишних пробелов и переносов
        text = re.sub(r'\s+', ' ', text)
        # Удаление специальных символов (кроме важных)
        text = re.sub(r'[^\w\s\-.:@+()]', ' ', text)
        return text.strip()
    
    def _calculate_pattern_scores(self, text: str) -> Dict[str, float]:
        """Расчет оценок на основе паттернов"""
        scores = {}
        
        for format_name, format_data in self.format_patterns.items():
            score = 0.0
            total_weight = 0.0
            
            # Проверка ключевых слов
            keyword_matches = 0
            for keyword in format_data['keywords']:
                if keyword.lower() in text.lower():
                    keyword_matches += 1
            
            if format_data['keywords']:
                keyword_score = keyword_matches / len(format_data['keywords'])
                score += keyword_score * 0.4
                total_weight += 0.4
            
            # Проверка паттернов
            pattern_matches = 0
            for pattern in format_data['patterns']:
                if re.search(pattern, text, re.IGNORECASE | re.MULTILINE):
                    pattern_matches += 1
            
            if format_data['patterns']:
                pattern_score = pattern_matches / len(format_data['patterns'])
                score += pattern_score * 0.6
                total_weight += 0.6
            
            # Нормализация оценки
            scores[format_name] = score / total_weight if total_weight > 0 else 0.0
        
        return scores
    
    def _combine_scores(self, pattern_scores: Dict[str, float], 
                       ml_scores: Dict[str, float]) -> Dict[str, float]:
        """Комбинирование оценок паттернов и ML"""
        combined = {}
        
        all_formats = set(pattern_scores.keys()) | set(ml_scores.keys())
        
        for format_name in all_formats:
            pattern_score = pattern_scores.get(format_name, 0.0)
            ml_score = ml_scores.get(format_name, 0.0)
            
            # Весовое комбинирование
            if ml_scores:  # Если ML доступно
                combined[format_name] = 0.6 * pattern_score + 0.4 * ml_score
            else:
                combined[format_name] = pattern_score
        
        return combined
    
    def _extract_format_specific_data(self, text: str, format_type: str) -> Dict[str, Any]:
        """Извлечение данных, специфичных для формата"""
        data = {}
        
        if format_type == 'hh_ru':
            # Извлечение HH.ru специфичных полей
            data.update(self._extract_hh_ru_data(text))
        elif format_type == 'linkedin':
            # Извлечение LinkedIn специфичных полей
            data.update(self._extract_linkedin_data(text))
        elif format_type == 'academic':
            # Извлечение академических полей
            data.update(self._extract_academic_data(text))
        elif format_type == 'european':
            # Извлечение европейских полей
            data.update(self._extract_european_data(text))
        
        return data
    
    def _extract_hh_ru_data(self, text: str) -> Dict[str, Any]:
        """Извлечение HH.ru специфичных данных"""
        data = {}
        
        # Желаемая должность
        position_match = re.search(r'Желаемая должность\s*:?\s*(.+?)(?:\n|$)', text, re.IGNORECASE)
        if position_match:
            data['desired_position'] = position_match.group(1).strip()
        
        # Специализации
        spec_match = re.search(r'Специализации\s*:?\s*(.+?)(?:\n|$)', text, re.IGNORECASE)
        if spec_match:
            data['specializations'] = [s.strip() for s in spec_match.group(1).split(',')]
        
        # Тип занятости
        employment_match = re.search(r'Тип занятости\s*:?\s*(.+?)(?:\n|$)', text, re.IGNORECASE)
        if employment_match:
            data['employment_type'] = employment_match.group(1).strip()
        
        return data
    
    def _extract_linkedin_data(self, text: str) -> Dict[str, Any]:
        """Извлечение LinkedIn специфичных данных"""
        data = {}
        
        # Поиск секций LinkedIn
        sections = re.findall(r'^([A-Z][a-z\s]+)\s*$', text, re.MULTILINE)
        data['sections'] = sections
        
        # Поиск "at Company" паттернов
        at_companies = re.findall(r'at\s+([A-Z][a-zA-Z\s&.,]+)', text)
        data['companies'] = at_companies
        
        return data
    
    def _extract_academic_data(self, text: str) -> Dict[str, Any]:
        """Извлечение академических данных"""
        data = {}
        
        # Поиск публикаций
        publications = re.findall(r'(?:Publications?|Публикации)\s*:?\s*(.+?)(?:\n\n|\Z)', 
                                text, re.IGNORECASE | re.DOTALL)
        if publications:
            data['publications'] = publications[0].strip()
        
        # Поиск исследовательских интересов
        research = re.findall(r'(?:Research Interests?|Исследовательские интересы)\s*:?\s*(.+?)(?:\n\n|\Z)', 
                            text, re.IGNORECASE | re.DOTALL)
        if research:
            data['research_interests'] = research[0].strip()
        
        return data
    
    def _extract_european_data(self, text: str) -> Dict[str, Any]:
        """Извлечение европейских (Europass) данных"""
        data = {}
        
        # Национальность
        nationality_match = re.search(r'Nationality\s*:?\s*(.+?)(?:\n|$)', text, re.IGNORECASE)
        if nationality_match:
            data['nationality'] = nationality_match.group(1).strip()
        
        # Водительские права
        driving_match = re.search(r'Driving licence\s*:?\s*(.+?)(?:\n|$)', text, re.IGNORECASE)
        if driving_match:
            data['driving_licence'] = driving_match.group(1).strip()
        
        return data
    
    def _get_detected_features(self, text: str, format_type: str) -> List[str]:
        """Получение списка выявленных признаков"""
        features = []
        
        format_data = self.format_patterns.get(format_type, {})
        
        # Проверка ключевых слов
        for keyword in format_data.get('keywords', []):
            if keyword.lower() in text.lower():
                features.append(f"keyword: {keyword}")
        
        # Проверка паттернов
        for pattern in format_data.get('patterns', []):
            if re.search(pattern, text, re.IGNORECASE | re.MULTILINE):
                features.append(f"pattern: {pattern}")
        
        return features
    
    def get_detection_confidence(self) -> float:
        """Получение общей уверенности детекции"""
        return 0.8  # Базовая уверенность


class MLFormatClassifier(BaseMLComponent):
    """ML классификатор формата резюме"""
    
    def __init__(self):
        self.logger = get_logger(__name__)
        self.vectorizer = None
        self.format_vectors = {}
        self.is_trained = False
        
        if TfidfVectorizer:
            self._initialize_classifier()
    
    def _initialize_classifier(self):
        """Инициализация классификатора"""
        try:
            self.vectorizer = TfidfVectorizer(
                ngram_range=(1, 3),
                max_features=1000,
                lowercase=True,
                stop_words=None  # Не используем стоп-слова для многоязычности
            )
            
            # Обучение на примерах форматов
            self._train_on_examples()
            
        except Exception as e:
            self.logger.warning(f"Ошибка инициализации ML классификатора: {e}")
    
    def _train_on_examples(self):
        """Обучение на примерах"""
        # Примеры текстов для разных форматов
        training_examples = {
            'hh_ru': [
                "Желаемая должность: Python разработчик Специализации: Программирование, разработка ПО",
                "Тип занятости: Полная занятость График работы: Полный день Опыт работы — 3 года",
                "Ключевые навыки: Python, Django, PostgreSQL О себе: Опытный разработчик"
            ],
            'linkedin': [
                "Experience Software Engineer at Tech Company Education Bachelor of Computer Science",
                "Summary Experienced developer Skills Python, Java, React Recommendations Great colleague",
                "Work Experience Senior Developer at StartupCorp 3 years 2 months"
            ],
            'academic': [
                "Curriculum Vitae Research Interests: Machine Learning Publications in top journals",
                "PhD Computer Science Teaching Experience: Database Systems Conference Presentations",
                "Research Publications Journal articles in Nature, Science Grants and Awards NSF Grant"
            ],
            'european': [
                "Personal information Nationality: German Driving licence: Category B",
                "Work experience Education and training Personal skills Additional information",
                "Name Address Nationality Date of birth Driving licence Category A, B"
            ]
        }
        
        # Подготовка данных для обучения
        texts = []
        labels = []
        
        for format_type, examples in training_examples.items():
            texts.extend(examples)
            labels.extend([format_type] * len(examples))
        
        try:
            # Векторизация текстов
            vectors = self.vectorizer.fit_transform(texts)
            
            # Создание представлений для каждого формата
            for format_type in training_examples.keys():
                format_indices = [i for i, label in enumerate(labels) if label == format_type]
                format_vectors = vectors[format_indices]
                # Усредненный вектор для формата
                self.format_vectors[format_type] = np.mean(format_vectors.toarray(), axis=0)
            
            self.is_trained = True
            
        except Exception as e:
            self.logger.error(f"Ошибка обучения классификатора: {e}")
    
    def classify(self, text: str) -> Dict[str, float]:
        """Классификация текста"""
        if not self.is_trained or not self.vectorizer:
            return {}
        
        try:
            # Векторизация входного текста
            text_vector = self.vectorizer.transform([text]).toarray()[0]
            
            # Расчет сходства с каждым форматом
            similarities = {}
            for format_type, format_vector in self.format_vectors.items():
                similarity = cosine_similarity([text_vector], [format_vector])[0][0]
                similarities[format_type] = max(0.0, similarity)  # Неотрицательные значения
            
            # Нормализация оценок
            total_score = sum(similarities.values())
            if total_score > 0:
                similarities = {k: v / total_score for k, v in similarities.items()}
            
            return similarities
            
        except Exception as e:
            self.logger.error(f"Ошибка классификации: {e}")
            return {}
    
    def train(self, data: List[Dict[str, Any]]) -> None:
        """Обучение на новых данных"""
        # Реализация для будущего расширения
        pass
    
    def predict(self, data: Any) -> Dict[str, Any]:
        """Предсказание"""
        return self.classify(str(data))
    
    def get_model_info(self) -> Dict[str, str]:
        """Информация о модели"""
        return {
            'type': 'TF-IDF + Cosine Similarity',
            'status': 'trained' if self.is_trained else 'not_trained',
            'formats': ', '.join(self.format_vectors.keys())
        }