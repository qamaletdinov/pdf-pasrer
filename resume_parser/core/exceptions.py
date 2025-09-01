"""
Кастомные исключения для парсера резюме
"""


class ParsingError(Exception):
    """Базовое исключение для ошибок парсинга"""
    pass


class FormatNotSupportedError(ParsingError):
    """Исключение для неподдерживаемых форматов"""
    pass


class MLModelError(ParsingError):
    """Исключение для ошибок ML моделей"""
    pass


class TextExtractionError(ParsingError):
    """Исключение для ошибок извлечения текста"""
    pass


class ValidationError(ParsingError):
    """Исключение для ошибок валидации данных"""
    pass


class ConfigurationError(ParsingError):
    """Исключение для ошибок конфигурации"""
    pass