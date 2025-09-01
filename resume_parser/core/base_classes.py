"""
Базовые абстрактные классы для парсера резюме
"""

from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Any, Union
from pathlib import Path
from dataclasses import dataclass

from .data_models import ResumeData
from .exceptions import ParsingError


class BaseParser(ABC):
    """Базовый абстрактный класс для всех парсеров"""
    
    @abstractmethod
    def parse(self, data: Union[str, Path]) -> ResumeData:
        """Парсинг данных и возврат структурированного результата"""
        pass
    
    @abstractmethod
    def get_supported_formats(self) -> List[str]:
        """Получение списка поддерживаемых форматов"""
        pass
    
    @abstractmethod
    def validate_input(self, data: Union[str, Path]) -> bool:
        """Валидация входных данных"""
        pass


class BaseExtractor(ABC):
    """Базовый класс для экстракторов"""
    
    @abstractmethod
    def extract(self, data: Any) -> Dict[str, Any]:
        """Извлечение данных"""
        pass
    
    @abstractmethod
    def get_confidence_score(self) -> float:
        """Получение оценки уверенности"""
        pass


class BaseTextExtractor(BaseExtractor):
    """Базовый класс для извлечения текста"""
    
    @abstractmethod
    def extract_text(self, source: Union[str, Path]) -> str:
        """Извлечение текста из источника"""
        pass
    
    @abstractmethod
    def get_method_name(self) -> str:
        """Получение названия метода извлечения"""
        pass


class BaseMLComponent(ABC):
    """Базовый класс для ML компонентов"""
    
    @abstractmethod
    def train(self, data: List[Dict[str, Any]]) -> None:
        """Обучение модели"""
        pass
    
    @abstractmethod
    def predict(self, data: Any) -> Dict[str, Any]:
        """Предсказание на основе данных"""
        pass
    
    @abstractmethod
    def get_model_info(self) -> Dict[str, str]:
        """Получение информации о модели"""
        pass


class BaseSegmenter(ABC):
    """Базовый класс для сегментации текста"""
    
    @abstractmethod
    def segment(self, text: str) -> Dict[str, List[str]]:
        """Сегментация текста на разделы"""
        pass
    
    @abstractmethod
    def detect_structure(self, text: str) -> Dict[str, Any]:
        """Определение структуры документа"""
        pass


class BaseDetector(ABC):
    """Базовый класс для детекторов"""
    
    @abstractmethod
    def detect(self, data: Any) -> str:
        """Детекция и возврат результата"""
        pass
    
    @abstractmethod
    def get_detection_confidence(self) -> float:
        """Получение уверенности детекции"""
        pass


class BaseValidator(ABC):
    """Базовый класс для валидаторов"""
    
    @abstractmethod
    def validate(self, data: Any) -> Dict[str, bool]:
        """Валидация данных"""
        pass
    
    @abstractmethod
    def get_validation_score(self) -> float:
        """Получение оценки валидности"""
        pass


class BaseProcessor(ABC):
    """Базовый класс для процессоров"""
    
    @abstractmethod
    def process(self, data: Any) -> Any:
        """Обработка данных"""
        pass
    
    @abstractmethod
    def preprocess(self, data: Any) -> Any:
        """Предварительная обработка"""
        pass
    
    @abstractmethod
    def postprocess(self, data: Any) -> Any:
        """Постобработка"""
        pass


@dataclass
class ProcessingResult:
    """Результат обработки"""
    data: Any
    confidence_score: float
    processing_time: float
    method_used: str
    metadata: Dict[str, Any]