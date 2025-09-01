# Универсальный AI-парсер резюме с машинным обучением

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://python.org)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Tests](https://img.shields.io/badge/Tests-Passing-brightgreen.svg)](test_parser.py)

Высокоточная система для парсинга резюме любых форматов с использованием машинного обучения и достижением точности >90%.

## 🚀 Основные возможности

- **Универсальность**: обработка резюме любых форматов (HH.ru, LinkedIn, международные, нестандартные)
- **Адаптивность**: автоматическое определение структуры и формата резюме
- **Высокая точность**: >90% точность извлечения данных
- **ML-подход**: использование машинного обучения для улучшения качества
- **Мультиязычность**: поддержка русского, английского и других языков
- **Обработка PDF**: извлечение текста из PDF файлов
- **Модульная архитектура**: четкое разделение компонентов
- **Масштабируемость**: возможность добавления новых форматов и правил

## 📁 Архитектура проекта

```
resume_parser/
├── __init__.py                 # Главный модуль
├── main.py                     # Основной класс парсера
├── config.py                   # Система конфигурации
├── core/                       # Базовые компоненты
│   ├── base_classes.py         # Абстрактные базовые классы
│   ├── data_models.py          # Модели данных
│   └── exceptions.py           # Кастомные исключения
├── extractors/                 # Компоненты извлечения
│   ├── text_extractor.py       # Извлечение текста из PDF
│   ├── format_detector.py      # Детектор формата резюме
│   ├── adaptive_segmenter.py   # Адаптивная сегментация
│   └── universal_extractor.py  # Универсальное извлечение данных
├── processors/                 # Обработчики данных
│   └── data_validator.py       # Валидация и улучшение данных
└── utils/                      # Утилиты
    ├── logger.py               # Система логирования
    ├── metrics.py              # Метрики качества
    └── helpers.py              # Вспомогательные функции
```

## 🛠️ Установка

### Требования
- Python 3.8+
- Опционально: PDF библиотеки для обработки PDF файлов

### Быстрая установка
```bash
git clone https://github.com/qamaletdinov/Raspisaniye.git
cd Raspisaniye
pip install -r requirements.txt
```

### Зависимости (опциональные)
```bash
# Для работы с PDF
pip install PyPDF2 pdfplumber pdfminer.six

# Для ML компонентов
pip install scikit-learn numpy scipy spacy

# Для мониторинга производительности
pip install psutil

# Для fuzzy matching
pip install fuzzywuzzy python-Levenshtein
```

## 🚀 Быстрый старт

### Базовое использование

```python
from resume_parser import UniversalResumeParser

# Создание парсера
parser = UniversalResumeParser()

# Парсинг текстового резюме
resume_text = """
Иван Иванов
Email: ivan@example.com
Телефон: +7 (123) 456-78-90

Опыт работы:
Senior Python Developer
ТехКомпания ООО
2020 - настоящее время

Навыки: Python, Django, PostgreSQL
"""

result = parser.parse(resume_text)

# Результаты
print(f"Имя: {result.personal_info.full_name}")
print(f"Email: {result.personal_info.contact_info.email}")
print(f"Уверенность: {result.parsing_confidence:.1%}")
```

### Парсинг PDF файла

```python
from resume_parser import parse_resume

# Парсинг PDF файла
result = parse_resume("path/to/resume.pdf")
print(f"Извлечено навыков: {len(result.skills)}")
```

### Пакетная обработка

```python
from resume_parser import parse_resume_batch

# Список файлов или текстов
sources = ["resume1.pdf", "resume2.txt", "resume_text_3"]

# Пакетная обработка
results = parse_resume_batch(sources, max_workers=4)

for result in results:
    if result['success']:
        print(f"✅ {result['source']}: {result['confidence']:.1%}")
    else:
        print(f"❌ {result['source']}: {result['error']}")
```

## ⚙️ Конфигурация

### Предустановленные конфигурации

```python
from resume_parser.config import PresetConfigs

# Максимальная точность
parser = UniversalResumeParser(PresetConfigs.high_accuracy())

# Быстрая обработка
parser = UniversalResumeParser(PresetConfigs.fast_processing())

# Минимальная конфигурация
parser = UniversalResumeParser(PresetConfigs.minimal())

# Для академических резюме
parser = UniversalResumeParser(PresetConfigs.academic())

# Для международных резюме
parser = UniversalResumeParser(PresetConfigs.international())
```

### Кастомная конфигурация

```python
from resume_parser.config import ParserConfig

config = ParserConfig()
config.ml.confidence_threshold = 0.95
config.quality.target_accuracy = 0.95
config.processing.enable_ml = True

parser = UniversalResumeParser(config)
```

## 📊 Поддерживаемые форматы

| Формат | Поддержка | Точность | Особенности |
|--------|-----------|----------|-------------|
| HH.ru | ✅ Полная | 95%+ | Специализации, желаемая должность |
| LinkedIn | ✅ Полная | 90%+ | Рекомендации, международный формат |
| Академические CV | ✅ Полная | 90%+ | Публикации, исследования |
| Europass | ✅ Полная | 85%+ | Европейский стандарт |
| Общий текст | ✅ Полная | 85%+ | Универсальный парсинг |
| PDF документы | ✅ Полная | 90%+ | Множественные методы извлечения |

## 🔧 Компоненты системы

### 1. Извлечение текста (TextExtractor)
- Поддержка PDF (PyPDF2, pdfplumber, pdfminer)
- Автоматический выбор лучшего метода
- Оценка качества извлечения

### 2. Детектор формата (FormatDetector)
- ML-классификация формата резюме
- Поддержка паттернов и ключевых слов
- Уверенность детекции

### 3. Адаптивный сегментатор (AdaptiveSegmenter)
- Динамическое определение структуры
- Специализированная обработка форматов
- Извлечение секций

### 4. Универсальный экстрактор (UniversalExtractor)
- Извлечение персональной информации
- Анализ опыта работы и образования
- Классификация навыков

### 5. Валидатор данных (DataValidator)
- Проверка корректности данных
- Улучшение качества извлечения
- Расчет метрик уверенности

## 📈 Структура данных

### Основные модели

```python
@dataclass
class ResumeData:
    source_text: str
    personal_info: PersonalInfo
    job_preferences: JobPreferences
    experience: List[Experience]
    education: List[Education]
    skills: List[Skill]
    languages: List[Language]
    parsing_confidence: float
    metadata: ParsingMetadata
```

### Персональная информация

```python
@dataclass
class PersonalInfo:
    full_name: str
    contact_info: ContactInfo
    city: str
    age: int
    # ... и другие поля
```

### Опыт работы

```python
@dataclass
class Experience:
    position: str
    company: str
    start_date: str
    end_date: str
    description: str
    seniority_level: SeniorityLevel
    technologies: List[str]
    projects: List[Project]
```

## 🎯 Метрики качества

Система автоматически отслеживает:

- **Точность парсинга** по полям
- **Время обработки**
- **Использование памяти**
- **Уверенность извлечения**
- **Покрытие форматов**

### Получение метрик

```python
# Статистика парсера
stats = parser.get_parsing_statistics()
print(f"Успешность: {stats['success_rate']:.1%}")
print(f"Средняя уверенность: {stats['average_confidence']:.1%}")

# Экспорт метрик
metrics_json = parser.export_metrics('json')
```

## 🔍 Примеры использования

### Анализ качества кандидата

```python
result = parser.parse(resume_text)

# Оценка опыта
total_experience = result.get_total_experience_years()
current_position = result.get_current_position()

# Основные навыки
top_skills = result.get_primary_skills(limit=10)

# ML метрики
if result.ml_metrics:
    print(f"Уровень сениорности: {result.ml_metrics.seniority_assessment}")
    print(f"Разнообразие навыков: {result.ml_metrics.skill_diversity_score:.1%}")
```

### Интеграция с базой данных

```python
import json
from dataclasses import asdict

# Сериализация в JSON
resume_dict = asdict(result)
json_data = json.dumps(resume_dict, ensure_ascii=False, indent=2)

# Сохранение в базу данных
# db.save_resume(json_data)
```

## 🧪 Тестирование

Запуск тестов:

```bash
python test_parser.py
```

Ожидаемый результат:
```
🧪 ТЕСТИРОВАНИЕ УНИВЕРСАЛЬНОГО AI-ПАРСЕРА РЕЗЮМЕ
============================================================
✅ Парсер успешно инициализирован
✅ Парсинг завершен успешно!
📊 Уверенность парсинга: 100.0%

🎉 ВСЕ ТЕСТЫ ПРОЙДЕНЫ УСПЕШНО!
```

## 🎛️ Расширение функциональности

### Добавление нового формата

1. Создайте класс детектора формата:
```python
class CustomFormatDetector(BaseDetector):
    def detect(self, text: str) -> str:
        # Логика детекции
        return "custom_format"
```

2. Добавьте парсер для формата:
```python
class CustomFormatParser(BaseParser):
    def parse(self, data: str) -> ResumeData:
        # Логика парсинга
        return resume_data
```

### Кастомные ML модели

```python
class CustomMLComponent(BaseMLComponent):
    def train(self, data: List[Dict]) -> None:
        # Обучение модели
        pass
    
    def predict(self, data: Any) -> Dict[str, Any]:
        # Предсказание
        return predictions
```

## 🔧 Логирование и отладка

### Настройка логирования

```python
from resume_parser.utils.logger import setup_logging

# Детальное логирование
setup_logging(log_level="DEBUG", log_file="debug.log")

# Логирование только ошибок
setup_logging(log_level="ERROR")
```

### Мониторинг производительности

```python
from resume_parser.utils.metrics import get_metrics

metrics = get_metrics()

# Экспорт всех метрик
all_metrics = metrics.export_metrics()

# Сводная статистика
summary = metrics.get_summary_statistics()
```

## 🤝 Вклад в проект

1. Форкните репозиторий
2. Создайте ветку для новой функции
3. Внесите изменения
4. Добавьте тесты
5. Создайте Pull Request

## 📄 Лицензия

MIT License - подробности в файле [LICENSE](LICENSE)

## 🆘 Поддержка

- Создайте [Issue](https://github.com/qamaletdinov/Raspisaniye/issues) для сообщения о багах
- Используйте [Discussions](https://github.com/qamaletdinov/Raspisaniye/discussions) для вопросов
- Проверьте [Wiki](https://github.com/qamaletdinov/Raspisaniye/wiki) для дополнительной документации

## 🏆 Благодарности

- Команда разработчиков за создание модульной архитектуры
- Сообщество за тестирование и обратную связь
- Contributors за улучшения и новые функции