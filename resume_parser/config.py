"""
Конфигурация парсера резюме
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any
from pathlib import Path


@dataclass
class MLConfig:
    """Конфигурация ML компонентов"""
    enable_ml: bool = True
    confidence_threshold: float = 0.85
    use_format_classifier: bool = True
    use_ner_extractor: bool = True
    use_skill_analyzer: bool = True
    use_experience_evaluator: bool = True
    model_cache_dir: str = "models"


@dataclass
class ExtractionConfig:
    """Конфигурация извлечения данных"""
    pdf_extraction_methods: List[str] = field(default_factory=lambda: ['pdfplumber', 'pdfminer', 'pypdf2'])
    enable_fuzzy_matching: bool = True
    fuzzy_threshold: int = 80
    max_text_length: int = 1024 * 1024  # 1MB
    min_text_length: int = 50


@dataclass
class ProcessingConfig:
    """Конфигурация обработки"""
    normalize_text: bool = True
    clean_text: bool = True
    detect_language: bool = True
    extract_sections: bool = True
    validate_data: bool = True
    enhance_data: bool = True


@dataclass
class QualityConfig:
    """Конфигурация качества"""
    target_accuracy: float = 0.90
    min_confidence_threshold: float = 0.70
    require_personal_info: bool = True
    require_contact_info: bool = True
    enable_quality_checks: bool = True


@dataclass
class PerformanceConfig:
    """Конфигурация производительности"""
    enable_caching: bool = True
    cache_dir: str = "cache"
    max_processing_time: float = 60.0  # секунды
    enable_parallel_processing: bool = True
    max_workers: int = 4


@dataclass
class LoggingConfig:
    """Конфигурация логирования"""
    log_level: str = "INFO"
    log_to_file: bool = True
    log_dir: str = "logs"
    log_format: str = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    enable_performance_logging: bool = True
    enable_debug_output: bool = False


@dataclass
class FormatConfig:
    """Конфигурация форматов"""
    supported_formats: List[str] = field(default_factory=lambda: [
        'hh_ru', 'linkedin', 'academic', 'european', 'generic'
    ])
    auto_detect_format: bool = True
    fallback_to_generic: bool = True
    format_confidence_threshold: float = 0.60


@dataclass
class ValidationConfig:
    """Конфигурация валидации"""
    validate_emails: bool = True
    validate_phones: bool = True
    validate_urls: bool = True
    validate_dates: bool = True
    strict_validation: bool = False
    fix_common_errors: bool = True


@dataclass
class OutputConfig:
    """Конфигурация вывода"""
    include_metadata: bool = True
    include_confidence_scores: bool = True
    include_processing_stats: bool = True
    output_format: str = "dataclass"  # dataclass, dict, json
    pretty_print: bool = True


@dataclass
class ParserConfig:
    """Основная конфигурация парсера резюме"""
    
    # Подконфигурации
    ml: MLConfig = field(default_factory=MLConfig)
    extraction: ExtractionConfig = field(default_factory=ExtractionConfig)
    processing: ProcessingConfig = field(default_factory=ProcessingConfig)
    quality: QualityConfig = field(default_factory=QualityConfig)
    performance: PerformanceConfig = field(default_factory=PerformanceConfig)
    logging: LoggingConfig = field(default_factory=LoggingConfig)
    formats: FormatConfig = field(default_factory=FormatConfig)
    validation: ValidationConfig = field(default_factory=ValidationConfig)
    output: OutputConfig = field(default_factory=OutputConfig)
    
    # Общие настройки
    version: str = "1.0.0"
    language_support: List[str] = field(default_factory=lambda: ['ru', 'en'])
    timezone: str = "UTC"
    encoding: str = "utf-8"
    
    # Пользовательские настройки
    custom_settings: Dict[str, Any] = field(default_factory=dict)
    
    def __post_init__(self):
        """Постинициализация для валидации конфигурации"""
        self._validate_config()
        self._setup_directories()
    
    def _validate_config(self):
        """Валидация параметров конфигурации"""
        # Проверка пороговых значений
        if not 0.0 <= self.ml.confidence_threshold <= 1.0:
            raise ValueError("confidence_threshold must be between 0.0 and 1.0")
        
        if not 0.0 <= self.quality.target_accuracy <= 1.0:
            raise ValueError("target_accuracy must be between 0.0 and 1.0")
        
        if not 0.0 <= self.quality.min_confidence_threshold <= 1.0:
            raise ValueError("min_confidence_threshold must be between 0.0 and 1.0")
        
        # Проверка размеров
        if self.extraction.max_text_length <= self.extraction.min_text_length:
            raise ValueError("max_text_length must be greater than min_text_length")
        
        # Проверка времени обработки
        if self.performance.max_processing_time <= 0:
            raise ValueError("max_processing_time must be positive")
        
        # Проверка воркеров
        if self.performance.max_workers <= 0:
            raise ValueError("max_workers must be positive")
    
    def _setup_directories(self):
        """Создание необходимых директорий"""
        directories = [
            self.ml.model_cache_dir,
            self.performance.cache_dir,
            self.logging.log_dir
        ]
        
        for directory in directories:
            Path(directory).mkdir(parents=True, exist_ok=True)
    
    def to_dict(self) -> Dict[str, Any]:
        """Преобразование конфигурации в словарь"""
        from dataclasses import asdict
        return asdict(self)
    
    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> 'ParserConfig':
        """Создание конфигурации из словаря"""
        # Создание подконфигураций
        ml_config = MLConfig(**config_dict.get('ml', {}))
        extraction_config = ExtractionConfig(**config_dict.get('extraction', {}))
        processing_config = ProcessingConfig(**config_dict.get('processing', {}))
        quality_config = QualityConfig(**config_dict.get('quality', {}))
        performance_config = PerformanceConfig(**config_dict.get('performance', {}))
        logging_config = LoggingConfig(**config_dict.get('logging', {}))
        formats_config = FormatConfig(**config_dict.get('formats', {}))
        validation_config = ValidationConfig(**config_dict.get('validation', {}))
        output_config = OutputConfig(**config_dict.get('output', {}))
        
        # Основные параметры
        main_params = {k: v for k, v in config_dict.items() 
                      if k not in ['ml', 'extraction', 'processing', 'quality', 
                                  'performance', 'logging', 'formats', 'validation', 'output']}
        
        return cls(
            ml=ml_config,
            extraction=extraction_config,
            processing=processing_config,
            quality=quality_config,
            performance=performance_config,
            logging=logging_config,
            formats=formats_config,
            validation=validation_config,
            output=output_config,
            **main_params
        )
    
    @classmethod
    def from_file(cls, config_path: str) -> 'ParserConfig':
        """Загрузка конфигурации из файла"""
        import json
        import yaml
        
        config_path = Path(config_path)
        
        if not config_path.exists():
            raise FileNotFoundError(f"Config file not found: {config_path}")
        
        # Определение формата файла
        if config_path.suffix.lower() == '.json':
            with open(config_path, 'r', encoding='utf-8') as f:
                config_dict = json.load(f)
        elif config_path.suffix.lower() in ['.yaml', '.yml']:
            with open(config_path, 'r', encoding='utf-8') as f:
                config_dict = yaml.safe_load(f)
        else:
            raise ValueError(f"Unsupported config file format: {config_path.suffix}")
        
        return cls.from_dict(config_dict)
    
    def save_to_file(self, config_path: str):
        """Сохранение конфигурации в файл"""
        import json
        import yaml
        
        config_path = Path(config_path)
        config_dict = self.to_dict()
        
        # Создание директории если не существует
        config_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Сохранение в зависимости от расширения
        if config_path.suffix.lower() == '.json':
            with open(config_path, 'w', encoding='utf-8') as f:
                json.dump(config_dict, f, ensure_ascii=False, indent=2)
        elif config_path.suffix.lower() in ['.yaml', '.yml']:
            with open(config_path, 'w', encoding='utf-8') as f:
                yaml.dump(config_dict, f, default_flow_style=False, allow_unicode=True)
        else:
            raise ValueError(f"Unsupported config file format: {config_path.suffix}")
    
    def merge_with(self, other_config: 'ParserConfig') -> 'ParserConfig':
        """Слияние с другой конфигурацией"""
        # Преобразуем обе конфигурации в словари
        self_dict = self.to_dict()
        other_dict = other_config.to_dict()
        
        # Рекурсивное слияние
        def merge_dicts(dict1, dict2):
            result = dict1.copy()
            for key, value in dict2.items():
                if key in result and isinstance(result[key], dict) and isinstance(value, dict):
                    result[key] = merge_dicts(result[key], value)
                else:
                    result[key] = value
            return result
        
        merged_dict = merge_dicts(self_dict, other_dict)
        return self.from_dict(merged_dict)
    
    def create_profile(self, profile_name: str) -> 'ParserConfig':
        """Создание профиля конфигурации для специфичных случаев"""
        profile_config = self.to_dict()
        
        if profile_name == "high_accuracy":
            # Конфигурация для максимальной точности
            profile_config['ml']['confidence_threshold'] = 0.95
            profile_config['quality']['target_accuracy'] = 0.95
            profile_config['quality']['min_confidence_threshold'] = 0.80
            profile_config['validation']['strict_validation'] = True
            profile_config['processing']['enhance_data'] = True
            
        elif profile_name == "fast_processing":
            # Конфигурация для быстрой обработки
            profile_config['ml']['enable_ml'] = False
            profile_config['processing']['enhance_data'] = False
            profile_config['validation']['strict_validation'] = False
            profile_config['performance']['max_workers'] = 8
            
        elif profile_name == "minimal":
            # Минимальная конфигурация
            profile_config['ml']['enable_ml'] = False
            profile_config['processing']['enhance_data'] = False
            profile_config['validation']['strict_validation'] = False
            profile_config['output']['include_metadata'] = False
            profile_config['output']['include_confidence_scores'] = False
            
        elif profile_name == "academic":
            # Конфигурация для академических резюме
            profile_config['formats']['supported_formats'] = ['academic', 'generic']
            profile_config['quality']['require_contact_info'] = False
            profile_config['validation']['validate_urls'] = True
            
        elif profile_name == "international":
            # Конфигурация для международных резюме
            profile_config['language_support'] = ['ru', 'en', 'de', 'fr', 'es']
            profile_config['formats']['supported_formats'] = ['linkedin', 'european', 'generic']
            
        return self.from_dict(profile_config)


# Предустановленные конфигурации
class PresetConfigs:
    """Предустановленные конфигурации для разных сценариев"""
    
    @staticmethod
    def default() -> ParserConfig:
        """Конфигурация по умолчанию"""
        return ParserConfig()
    
    @staticmethod
    def high_accuracy() -> ParserConfig:
        """Конфигурация для максимальной точности"""
        return ParserConfig().create_profile("high_accuracy")
    
    @staticmethod
    def fast_processing() -> ParserConfig:
        """Конфигурация для быстрой обработки"""
        return ParserConfig().create_profile("fast_processing")
    
    @staticmethod
    def minimal() -> ParserConfig:
        """Минимальная конфигурация"""
        return ParserConfig().create_profile("minimal")
    
    @staticmethod
    def academic() -> ParserConfig:
        """Конфигурация для академических резюме"""
        return ParserConfig().create_profile("academic")
    
    @staticmethod
    def international() -> ParserConfig:
        """Конфигурация для международных резюме"""
        return ParserConfig().create_profile("international")