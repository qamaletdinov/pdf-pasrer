"""
Система логирования для парсера резюме
"""

import logging
import sys
from pathlib import Path
from typing import Optional
from datetime import datetime


def setup_logging(log_level: str = "INFO", 
                 log_file: Optional[str] = None,
                 log_dir: str = "logs") -> None:
    """
    Настройка системы логирования
    
    Args:
        log_level: Уровень логирования (DEBUG, INFO, WARNING, ERROR)
        log_file: Имя файла лога (если None, используется дата)
        log_dir: Директория для логов
    """
    # Создание директории для логов
    log_path = Path(log_dir)
    log_path.mkdir(exist_ok=True)
    
    # Имя файла лога
    if log_file is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_file = f"resume_parser_{timestamp}.log"
    
    log_file_path = log_path / log_file
    
    # Настройка форматирования
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    # Получение корневого логгера
    root_logger = logging.getLogger()
    root_logger.setLevel(getattr(logging, log_level.upper()))
    
    # Очистка существующих обработчиков
    for handler in root_logger.handlers[:]:
        root_logger.removeHandler(handler)
    
    # Консольный обработчик
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(getattr(logging, log_level.upper()))
    console_handler.setFormatter(formatter)
    root_logger.addHandler(console_handler)
    
    # Файловый обработчик
    file_handler = logging.FileHandler(log_file_path, encoding='utf-8')
    file_handler.setLevel(logging.DEBUG)  # В файл записываем все
    file_handler.setFormatter(formatter)
    root_logger.addHandler(file_handler)
    
    # Логирование инициализации
    logger = logging.getLogger(__name__)
    logger.info(f"Логирование инициализировано. Файл: {log_file_path}")


def get_logger(name: str) -> logging.Logger:
    """
    Получение именованного логгера
    
    Args:
        name: Имя логгера (обычно __name__)
        
    Returns:
        Настроенный логгер
    """
    logger = logging.getLogger(name)
    
    # Если логгер еще не настроен, настраиваем базовое логирование
    if not logger.handlers and not logger.parent.handlers:
        setup_logging()
    
    return logger


class ResumeParserLogger:
    """Специализированный логгер для парсера резюме"""
    
    def __init__(self, name: str):
        self.logger = get_logger(name)
        self.processing_context = {}
    
    def set_context(self, **kwargs):
        """Установка контекста для логирования"""
        self.processing_context.update(kwargs)
    
    def clear_context(self):
        """Очистка контекста"""
        self.processing_context.clear()
    
    def _format_message(self, message: str) -> str:
        """Форматирование сообщения с контекстом"""
        if self.processing_context:
            context_str = " | ".join([f"{k}={v}" for k, v in self.processing_context.items()])
            return f"[{context_str}] {message}"
        return message
    
    def debug(self, message: str, **kwargs):
        """Debug логирование"""
        self.logger.debug(self._format_message(message), **kwargs)
    
    def info(self, message: str, **kwargs):
        """Info логирование"""
        self.logger.info(self._format_message(message), **kwargs)
    
    def warning(self, message: str, **kwargs):
        """Warning логирование"""
        self.logger.warning(self._format_message(message), **kwargs)
    
    def error(self, message: str, **kwargs):
        """Error логирование"""
        self.logger.error(self._format_message(message), **kwargs)
    
    def critical(self, message: str, **kwargs):
        """Critical логирование"""
        self.logger.critical(self._format_message(message), **kwargs)
    
    def log_parsing_start(self, source: str, format_type: str = None):
        """Логирование начала парсинга"""
        self.set_context(source=source, format=format_type)
        self.info(f"Начало парсинга: {source}")
    
    def log_parsing_end(self, success: bool, processing_time: float, confidence: float = None):
        """Логирование завершения парсинга"""
        status = "успешно" if success else "с ошибками"
        message = f"Парсинг завершен {status}. Время: {processing_time:.2f}с"
        if confidence is not None:
            message += f", Уверенность: {confidence:.1%}"
        
        if success:
            self.info(message)
        else:
            self.warning(message)
        
        self.clear_context()
    
    def log_extraction_step(self, step_name: str, data_extracted: bool, details: str = None):
        """Логирование шага извлечения"""
        status = "✓" if data_extracted else "✗"
        message = f"{status} {step_name}"
        if details:
            message += f": {details}"
        
        if data_extracted:
            self.debug(message)
        else:
            self.warning(message)
    
    def log_ml_prediction(self, model_name: str, confidence: float, result: str):
        """Логирование ML предсказания"""
        self.debug(f"ML {model_name}: {result} (уверенность: {confidence:.1%})")
    
    def log_validation_result(self, field_name: str, is_valid: bool, score: float = None):
        """Логирование результата валидации"""
        status = "валидно" if is_valid else "невалидно"
        message = f"Валидация {field_name}: {status}"
        if score is not None:
            message += f" (оценка: {score:.2f})"
        
        level = self.debug if is_valid else self.warning
        level(message)
    
    def log_performance_metric(self, metric_name: str, value: float, unit: str = ""):
        """Логирование метрики производительности"""
        self.debug(f"Метрика {metric_name}: {value:.2f}{unit}")
    
    def log_error_with_context(self, error: Exception, context: str = None):
        """Логирование ошибки с контекстом"""
        error_message = f"Ошибка: {str(error)}"
        if context:
            error_message += f" | Контекст: {context}"
        
        self.error(error_message, exc_info=True)


# Глобальный логгер для модуля
module_logger = ResumeParserLogger(__name__)


def log_function_call(func):
    """Декоратор для логирования вызовов функций"""
    def wrapper(*args, **kwargs):
        func_name = func.__name__
        module_logger.debug(f"Вызов функции: {func_name}")
        
        try:
            result = func(*args, **kwargs)
            module_logger.debug(f"Функция {func_name} выполнена успешно")
            return result
        except Exception as e:
            module_logger.log_error_with_context(e, f"Функция: {func_name}")
            raise
    
    return wrapper


def log_processing_time(func):
    """Декоратор для логирования времени выполнения"""
    import time
    
    def wrapper(*args, **kwargs):
        func_name = func.__name__
        start_time = time.time()
        
        try:
            result = func(*args, **kwargs)
            processing_time = time.time() - start_time
            module_logger.log_performance_metric(f"{func_name}_time", processing_time, "с")
            return result
        except Exception as e:
            processing_time = time.time() - start_time
            module_logger.log_performance_metric(f"{func_name}_time_failed", processing_time, "с")
            raise
    
    return wrapper