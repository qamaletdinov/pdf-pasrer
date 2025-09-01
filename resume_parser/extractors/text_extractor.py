"""
Извлечение текста из PDF файлов с использованием множественных методов
"""

import logging
from typing import List, Optional, Dict, Any
from pathlib import Path
from dataclasses import dataclass

from ..core.base_classes import BaseTextExtractor, ProcessingResult
from ..core.exceptions import TextExtractionError
from ..utils.logger import get_logger

# PDF библиотеки (опциональные)
try:
    import PyPDF2
except ImportError:
    PyPDF2 = None

try:
    import pdfplumber
except ImportError:
    pdfplumber = None

try:
    from pdfminer.high_level import extract_text as pdfminer_extract
except ImportError:
    pdfminer_extract = None


@dataclass
class ExtractionResult:
    """Результат извлечения текста"""
    text: str
    method_used: str
    quality_score: float
    extraction_time: float
    error_message: Optional[str] = None
    metadata: Dict[str, Any] = None


class PyPDF2TextExtractor(BaseTextExtractor):
    """Извлечение текста с помощью PyPDF2"""
    
    def __init__(self):
        self.logger = get_logger(__name__)
    
    def extract_text(self, source: Path) -> str:
        """Извлечение текста из PDF"""
        if PyPDF2 is None:
            raise TextExtractionError("PyPDF2 не установлен")
        
        try:
            with open(source, 'rb') as file:
                reader = PyPDF2.PdfReader(file)
                text = ""
                for page in reader.pages:
                    text += page.extract_text() + "\n"
                return text.strip()
        except Exception as e:
            self.logger.error(f"Ошибка PyPDF2 извлечения: {e}")
            raise TextExtractionError(f"PyPDF2 extraction failed: {e}")
    
    def extract(self, data: Path) -> Dict[str, Any]:
        """Извлечение с метаданными"""
        import time
        start_time = time.time()
        
        try:
            text = self.extract_text(data)
            extraction_time = time.time() - start_time
            
            return {
                'text': text,
                'method': self.get_method_name(),
                'quality_score': self._calculate_quality_score(text),
                'extraction_time': extraction_time,
                'success': True
            }
        except Exception as e:
            return {
                'text': '',
                'method': self.get_method_name(),
                'quality_score': 0.0,
                'extraction_time': time.time() - start_time,
                'success': False,
                'error': str(e)
            }
    
    def get_method_name(self) -> str:
        return "PyPDF2"
    
    def get_confidence_score(self) -> float:
        return 0.7  # Средняя надежность
    
    def _calculate_quality_score(self, text: str) -> float:
        """Расчет качества извлеченного текста"""
        if not text:
            return 0.0
        
        # Простые эвристики качества
        score = 0.0
        
        # Длина текста
        if len(text) > 100:
            score += 0.3
        
        # Наличие кириллицы или латиницы
        if any(c.isalpha() for c in text):
            score += 0.3
        
        # Отсутствие слишком большого количества специальных символов
        special_chars_ratio = sum(1 for c in text if not c.isalnum() and not c.isspace()) / len(text)
        if special_chars_ratio < 0.3:
            score += 0.4
        
        return min(score, 1.0)


class PdfPlumberTextExtractor(BaseTextExtractor):
    """Извлечение текста с помощью pdfplumber"""
    
    def __init__(self):
        self.logger = get_logger(__name__)
    
    def extract_text(self, source: Path) -> str:
        """Извлечение текста из PDF"""
        if pdfplumber is None:
            raise TextExtractionError("pdfplumber не установлен")
        
        try:
            with pdfplumber.open(source) as pdf:
                text = ""
                for page in pdf.pages:
                    page_text = page.extract_text()
                    if page_text:
                        text += page_text + "\n"
                return text.strip()
        except Exception as e:
            self.logger.error(f"Ошибка pdfplumber извлечения: {e}")
            raise TextExtractionError(f"pdfplumber extraction failed: {e}")
    
    def extract(self, data: Path) -> Dict[str, Any]:
        """Извлечение с метаданными"""
        import time
        start_time = time.time()
        
        try:
            text = self.extract_text(data)
            extraction_time = time.time() - start_time
            
            return {
                'text': text,
                'method': self.get_method_name(),
                'quality_score': self._calculate_quality_score(text),
                'extraction_time': extraction_time,
                'success': True
            }
        except Exception as e:
            return {
                'text': '',
                'method': self.get_method_name(),
                'quality_score': 0.0,
                'extraction_time': time.time() - start_time,
                'success': False,
                'error': str(e)
            }
    
    def get_method_name(self) -> str:
        return "pdfplumber"
    
    def get_confidence_score(self) -> float:
        return 0.9  # Высокая надежность
    
    def _calculate_quality_score(self, text: str) -> float:
        """Расчет качества извлеченного текста"""
        if not text:
            return 0.0
        
        score = 0.0
        
        # Длина и структура
        if len(text) > 200:
            score += 0.3
        
        # Наличие структурных элементов
        lines = text.split('\n')
        if len(lines) > 10:
            score += 0.2
        
        # Качество текста
        alpha_ratio = sum(1 for c in text if c.isalpha()) / len(text)
        if alpha_ratio > 0.6:
            score += 0.5
        
        return min(score, 1.0)


class PdfMinerTextExtractor(BaseTextExtractor):
    """Извлечение текста с помощью pdfminer"""
    
    def __init__(self):
        self.logger = get_logger(__name__)
    
    def extract_text(self, source: Path) -> str:
        """Извлечение текста из PDF"""
        if pdfminer_extract is None:
            raise TextExtractionError("pdfminer не установлен")
        
        try:
            return pdfminer_extract(str(source))
        except Exception as e:
            self.logger.error(f"Ошибка pdfminer извлечения: {e}")
            raise TextExtractionError(f"pdfminer extraction failed: {e}")
    
    def extract(self, data: Path) -> Dict[str, Any]:
        """Извлечение с метаданными"""
        import time
        start_time = time.time()
        
        try:
            text = self.extract_text(data)
            extraction_time = time.time() - start_time
            
            return {
                'text': text,
                'method': self.get_method_name(),
                'quality_score': self._calculate_quality_score(text),
                'extraction_time': extraction_time,
                'success': True
            }
        except Exception as e:
            return {
                'text': '',
                'method': self.get_method_name(),
                'quality_score': 0.0,
                'extraction_time': time.time() - start_time,
                'success': False,
                'error': str(e)
            }
    
    def get_method_name(self) -> str:
        return "pdfminer"
    
    def get_confidence_score(self) -> float:
        return 0.8  # Хорошая надежность
    
    def _calculate_quality_score(self, text: str) -> float:
        """Расчет качества извлеченного текста"""
        if not text:
            return 0.0
        
        score = 0.0
        
        # Качество извлечения
        if len(text) > 150:
            score += 0.4
        
        # Читаемость
        words = text.split()
        if len(words) > 50:
            score += 0.3
        
        # Структура
        if '\n' in text and len(text.split('\n')) > 5:
            score += 0.3
        
        return min(score, 1.0)


class AdvancedTextExtractor:
    """Продвинутый извлекатель текста с множественными методами"""
    
    def __init__(self):
        self.logger = get_logger(__name__)
        self.extractors = self._initialize_extractors()
    
    def _initialize_extractors(self) -> List[BaseTextExtractor]:
        """Инициализация доступных экстракторов"""
        extractors = []
        
        # Добавляем экстракторы в порядке предпочтения
        if pdfplumber:
            extractors.append(PdfPlumberTextExtractor())
        if pdfminer_extract:
            extractors.append(PdfMinerTextExtractor())
        if PyPDF2:
            extractors.append(PyPDF2TextExtractor())
        
        if not extractors:
            self.logger.warning("Ни одна PDF библиотека не доступна")
        
        return extractors
    
    def extract_best(self, pdf_path: Path) -> ExtractionResult:
        """Извлечение текста лучшим доступным методом"""
        if not self.extractors:
            raise TextExtractionError("Нет доступных экстракторов PDF")
        
        best_result = None
        best_score = 0.0
        
        for extractor in self.extractors:
            try:
                result = extractor.extract(pdf_path)
                if result['success'] and result['quality_score'] > best_score:
                    best_score = result['quality_score']
                    best_result = result
            except Exception as e:
                self.logger.warning(f"Ошибка {extractor.get_method_name()}: {e}")
                continue
        
        if not best_result:
            raise TextExtractionError("Все методы извлечения завершились с ошибкой")
        
        return ExtractionResult(
            text=best_result['text'],
            method_used=best_result['method'],
            quality_score=best_result['quality_score'],
            extraction_time=best_result['extraction_time'],
            metadata=best_result
        )
    
    def extract_all_methods(self, pdf_path: Path) -> List[ExtractionResult]:
        """Извлечение всеми доступными методами"""
        results = []
        
        for extractor in self.extractors:
            try:
                result = extractor.extract(pdf_path)
                extraction_result = ExtractionResult(
                    text=result['text'],
                    method_used=result['method'],
                    quality_score=result['quality_score'],
                    extraction_time=result['extraction_time'],
                    error_message=result.get('error'),
                    metadata=result
                )
                results.append(extraction_result)
            except Exception as e:
                results.append(ExtractionResult(
                    text="",
                    method_used=extractor.get_method_name(),
                    quality_score=0.0,
                    extraction_time=0.0,
                    error_message=str(e)
                ))
        
        return results
    
    def is_available(self) -> bool:
        """Проверка доступности экстракторов"""
        return len(self.extractors) > 0
    
    def get_available_methods(self) -> List[str]:
        """Получение списка доступных методов"""
        return [extractor.get_method_name() for extractor in self.extractors]