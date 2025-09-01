"""
Вспомогательные функции и классы
"""

import re
import string
from typing import List, Dict, Any, Optional, Tuple, Union
from dataclasses import dataclass
from collections import Counter
import unicodedata

from ..utils.logger import get_logger


class TextProcessor:
    """Процессор для обработки и анализа текста"""
    
    def __init__(self):
        self.logger = get_logger(__name__)
        
        # Паттерны для различных типов данных
        self.email_pattern = re.compile(
            r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'
        )
        
        self.phone_pattern = re.compile(
            r'(?:\+7|8|7)?[\s\-\(\)]?(?:\d{3}[\s\-\(\)]?){2}\d{4}|'
            r'(?:\+7|8|7)?[\s\-\(\)]?\d{3}[\s\-\(\)]?\d{3}[\s\-\(\)]?\d{2}[\s\-\(\)]?\d{2}'
        )
        
        self.url_pattern = re.compile(
            r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+'
        )
        
        # Паттерны для дат
        self.date_patterns = [
            re.compile(r'\b\d{1,2}[./\-]\d{1,2}[./\-]\d{4}\b'),  # DD/MM/YYYY
            re.compile(r'\b\d{4}[./\-]\d{1,2}[./\-]\d{1,2}\b'),  # YYYY/MM/DD
            re.compile(r'\b(?:январ[ья]|феврал[ья]|март[а]?|апрел[ья]|ма[йя]|июн[ья]|июл[ья]|август[а]?|сентябр[ья]|октябр[ья]|ноябр[ья]|декабр[ья])\s+\d{4}\b', re.IGNORECASE),
            re.compile(r'\b\d{1,2}\s+(?:январ[ья]|феврал[ья]|март[а]?|апрел[ья]|ма[йя]|июн[ья]|июл[ья]|август[а]?|сентябр[ья]|октябр[ья]|ноябр[ья]|декабр[ья])\s+\d{4}\b', re.IGNORECASE)
        ]
    
    def normalize_text(self, text: str) -> str:
        """Нормализация текста"""
        if not text:
            return ""
        
        # Нормализация Unicode
        text = unicodedata.normalize('NFKC', text)
        
        # Удаление лишних пробелов
        text = re.sub(r'\s+', ' ', text)
        
        # Удаление непечатаемых символов
        text = ''.join(char for char in text if char.isprintable() or char.isspace())
        
        return text.strip()
    
    def clean_text(self, text: str, remove_punctuation: bool = False) -> str:
        """Очистка текста от ненужных символов"""
        # Нормализация
        text = self.normalize_text(text)
        
        # Удаление специальных символов (кроме важных)
        text = re.sub(r'[^\w\s\-.:@+()/#]', ' ', text)
        
        # Удаление пунктуации (опционально)
        if remove_punctuation:
            text = text.translate(str.maketrans('', '', string.punctuation))
        
        # Финальная очистка пробелов
        text = re.sub(r'\s+', ' ', text).strip()
        
        return text
    
    def extract_emails(self, text: str) -> List[str]:
        """Извлечение email адресов"""
        return self.email_pattern.findall(text)
    
    def extract_phones(self, text: str) -> List[str]:
        """Извлечение телефонных номеров"""
        phones = self.phone_pattern.findall(text)
        # Нормализация номеров
        normalized_phones = []
        for phone in phones:
            # Удаляем все кроме цифр и +
            clean_phone = re.sub(r'[^\d+]', '', phone)
            if len(clean_phone) >= 10:  # Минимальная длина номера
                normalized_phones.append(clean_phone)
        
        return normalized_phones
    
    def extract_urls(self, text: str) -> List[str]:
        """Извлечение URL адресов"""
        return self.url_pattern.findall(text)
    
    def extract_dates(self, text: str) -> List[str]:
        """Извлечение дат"""
        dates = []
        for pattern in self.date_patterns:
            dates.extend(pattern.findall(text))
        return dates
    
    def split_into_sentences(self, text: str) -> List[str]:
        """Разбивка текста на предложения"""
        # Простое разбиение по точкам, учитывая сокращения
        sentences = re.split(r'(?<!\w\.\w.)(?<![A-Z][a-z]\.)(?<=\.|\!|\?)\s', text)
        return [s.strip() for s in sentences if s.strip()]
    
    def split_into_lines(self, text: str, remove_empty: bool = True) -> List[str]:
        """Разбивка текста на строки"""
        lines = text.split('\n')
        if remove_empty:
            lines = [line.strip() for line in lines if line.strip()]
        return lines
    
    def calculate_text_stats(self, text: str) -> Dict[str, Any]:
        """Расчет статистики текста"""
        if not text:
            return {
                'char_count': 0,
                'word_count': 0,
                'line_count': 0,
                'sentence_count': 0,
                'avg_word_length': 0.0,
                'lexical_diversity': 0.0
            }
        
        lines = self.split_into_lines(text)
        sentences = self.split_into_sentences(text)
        words = text.split()
        
        # Подсчет уникальных слов для лексического разнообразия
        unique_words = set(word.lower().strip(string.punctuation) for word in words)
        lexical_diversity = len(unique_words) / len(words) if words else 0.0
        
        # Средняя длина слова
        avg_word_length = sum(len(word) for word in words) / len(words) if words else 0.0
        
        return {
            'char_count': len(text),
            'word_count': len(words),
            'line_count': len(lines),
            'sentence_count': len(sentences),
            'avg_word_length': avg_word_length,
            'lexical_diversity': lexical_diversity,
            'emails_found': len(self.extract_emails(text)),
            'phones_found': len(self.extract_phones(text)),
            'urls_found': len(self.extract_urls(text)),
            'dates_found': len(self.extract_dates(text))
        }
    
    def detect_language(self, text: str) -> str:
        """Простое определение языка текста"""
        # Подсчет кириллических и латинских символов
        cyrillic_count = sum(1 for char in text if '\u0400' <= char <= '\u04FF')
        latin_count = sum(1 for char in text if 'a' <= char.lower() <= 'z')
        
        total_alpha = cyrillic_count + latin_count
        
        if total_alpha == 0:
            return 'unknown'
        
        cyrillic_ratio = cyrillic_count / total_alpha
        
        if cyrillic_ratio > 0.6:
            return 'ru'
        elif cyrillic_ratio < 0.3:
            return 'en'
        else:
            return 'mixed'
    
    def find_section_headers(self, text: str) -> List[Tuple[str, int]]:
        """Поиск заголовков секций"""
        lines = self.split_into_lines(text)
        headers = []
        
        # Паттерны для заголовков
        header_patterns = [
            r'^[А-ЯA-Z][а-яa-z\s]+:?\s*$',  # Заголовок с заглавной буквы
            r'^[А-ЯA-Z\s]+$',               # Полностью заглавными буквами
            r'^\d+\.\s*[А-ЯA-Z][а-яa-z\s]+',  # Нумерованные заголовки
        ]
        
        for i, line in enumerate(lines):
            line = line.strip()
            if not line:
                continue
            
            # Проверка паттернов заголовков
            for pattern in header_patterns:
                if re.match(pattern, line) and len(line) < 100:  # Ограничение длины
                    headers.append((line, i))
                    break
        
        return headers
    
    def extract_key_phrases(self, text: str, min_length: int = 2, max_length: int = 5) -> List[str]:
        """Извлечение ключевых фраз"""
        # Очистка и нормализация
        clean_text = self.clean_text(text, remove_punctuation=True)
        words = clean_text.lower().split()
        
        # Удаление стоп-слов (простой список)
        stop_words = {
            'и', 'в', 'на', 'с', 'по', 'для', 'от', 'до', 'при', 'о', 'об',
            'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for',
            'of', 'with', 'by', 'from', 'about', 'into', 'through', 'during'
        }
        
        filtered_words = [word for word in words if word not in stop_words and len(word) > 2]
        
        # Генерация n-грамм
        phrases = []
        for n in range(min_length, max_length + 1):
            for i in range(len(filtered_words) - n + 1):
                phrase = ' '.join(filtered_words[i:i+n])
                phrases.append(phrase)
        
        # Подсчет частоты и возврат самых частых
        phrase_counts = Counter(phrases)
        return [phrase for phrase, count in phrase_counts.most_common(20)]


class DataValidator:
    """Валидатор для проверки качества извлеченных данных"""
    
    def __init__(self):
        self.logger = get_logger(__name__)
        self.text_processor = TextProcessor()
    
    def validate_email(self, email: str) -> Tuple[bool, float]:
        """Валидация email адреса"""
        if not email:
            return False, 0.0
        
        # Проверка формата
        if not self.text_processor.email_pattern.match(email):
            return False, 0.0
        
        # Дополнительные проверки
        score = 1.0
        
        # Проверка на подозрительные домены
        suspicious_domains = ['example.com', 'test.com', 'localhost']
        domain = email.split('@')[1].lower()
        if domain in suspicious_domains:
            score *= 0.5
        
        # Проверка длины
        if len(email) > 100:
            score *= 0.7
        
        return True, score
    
    def validate_phone(self, phone: str) -> Tuple[bool, float]:
        """Валидация телефонного номера"""
        if not phone:
            return False, 0.0
        
        # Очистка номера
        clean_phone = re.sub(r'[^\d+]', '', phone)
        
        # Проверка длины
        if len(clean_phone) < 10 or len(clean_phone) > 15:
            return False, 0.0
        
        # Проверка формата
        score = 1.0
        
        # Российские номера
        if clean_phone.startswith('+7') or clean_phone.startswith('8'):
            if len(clean_phone) == 11 or len(clean_phone) == 12:
                score = 1.0
            else:
                score = 0.7
        else:
            score = 0.8  # Международные номера
        
        return True, score
    
    def validate_name(self, name: str) -> Tuple[bool, float]:
        """Валидация имени"""
        if not name:
            return False, 0.0
        
        name = name.strip()
        
        # Минимальная длина
        if len(name) < 2:
            return False, 0.0
        
        # Проверка на наличие букв
        if not any(c.isalpha() for c in name):
            return False, 0.0
        
        score = 1.0
        
        # Проверка на слишком много цифр
        digit_ratio = sum(1 for c in name if c.isdigit()) / len(name)
        if digit_ratio > 0.3:
            score *= 0.5
        
        # Проверка на разумную длину
        if len(name) > 100:
            score *= 0.7
        
        # Проверка на наличие пробелов (для полного имени)
        if ' ' in name:
            parts = name.split()
            if len(parts) >= 2 and all(len(part) > 1 for part in parts):
                score = min(score * 1.2, 1.0)  # Бонус за полное имя
        
        return True, score
    
    def validate_date(self, date_str: str) -> Tuple[bool, float]:
        """Валидация даты"""
        if not date_str:
            return False, 0.0
        
        # Попытка найти дату в строке
        dates = self.text_processor.extract_dates(date_str)
        if not dates:
            return False, 0.0
        
        # Простая проверка разумности даты
        score = 1.0
        
        # Проверка на год (должен быть в разумных пределах)
        year_match = re.search(r'\b(19|20)\d{2}\b', date_str)
        if year_match:
            year = int(year_match.group())
            if 1950 <= year <= 2030:  # Разумный диапазон для резюме
                score = 1.0
            else:
                score = 0.5
        else:
            score = 0.7
        
        return True, score
    
    def validate_url(self, url: str) -> Tuple[bool, float]:
        """Валидация URL"""
        if not url:
            return False, 0.0
        
        # Проверка формата
        urls = self.text_processor.extract_urls(url)
        if not urls:
            return False, 0.0
        
        score = 1.0
        
        # Проверка на популярные домены
        professional_domains = ['linkedin.com', 'github.com', 'stackoverflow.com']
        for domain in professional_domains:
            if domain in url.lower():
                score = min(score * 1.2, 1.0)
                break
        
        return True, score
    
    def validate_text_quality(self, text: str) -> Tuple[bool, float]:
        """Валидация качества текста"""
        if not text:
            return False, 0.0
        
        stats = self.text_processor.calculate_text_stats(text)
        
        score = 1.0
        
        # Проверка минимальной длины
        if stats['word_count'] < 3:
            return False, 0.0
        
        # Лексическое разнообразие
        if stats['lexical_diversity'] < 0.3:
            score *= 0.7
        elif stats['lexical_diversity'] > 0.8:
            score = min(score * 1.1, 1.0)
        
        # Средняя длина слова
        if stats['avg_word_length'] < 2 or stats['avg_word_length'] > 15:
            score *= 0.8
        
        # Соотношение букв к общему количеству символов
        alpha_ratio = sum(1 for c in text if c.isalpha()) / len(text)
        if alpha_ratio < 0.5:
            score *= 0.6
        
        return True, score
    
    def validate_resume_section(self, section_name: str, content: str) -> Dict[str, Any]:
        """Валидация секции резюме"""
        validation_result = {
            'section_name': section_name,
            'is_valid': False,
            'confidence_score': 0.0,
            'issues': [],
            'suggestions': []
        }
        
        # Общая валидация контента
        content_valid, content_score = self.validate_text_quality(content)
        
        if not content_valid:
            validation_result['issues'].append("Недостаточно качественный текст")
            return validation_result
        
        # Специфичная валидация по типу секции
        section_score = content_score
        
        if 'опыт' in section_name.lower() or 'experience' in section_name.lower():
            # Валидация секции опыта
            if len(content) < 50:
                validation_result['issues'].append("Слишком короткое описание опыта")
                section_score *= 0.7
            
            # Поиск дат
            dates = self.text_processor.extract_dates(content)
            if not dates:
                validation_result['issues'].append("Не найдены даты в описании опыта")
                section_score *= 0.8
        
        elif 'образование' in section_name.lower() or 'education' in section_name.lower():
            # Валидация секции образования
            educational_keywords = ['университет', 'институт', 'колледж', 'university', 'college', 'degree']
            if not any(keyword in content.lower() for keyword in educational_keywords):
                validation_result['issues'].append("Не найдены образовательные учреждения")
                section_score *= 0.8
        
        elif 'навыки' in section_name.lower() or 'skills' in section_name.lower():
            # Валидация секции навыков
            skills_count = len(content.split(',')) if ',' in content else len(content.split())
            if skills_count < 3:
                validation_result['issues'].append("Слишком мало навыков")
                section_score *= 0.8
            elif skills_count > 50:
                validation_result['suggestions'].append("Рекомендуется сократить список навыков")
        
        validation_result['is_valid'] = section_score > 0.5
        validation_result['confidence_score'] = section_score
        
        return validation_result