"""
Примеры использования универсального AI-парсера резюме
"""

import json
from dataclasses import asdict
from resume_parser import UniversalResumeParser, parse_resume, parse_resume_batch
from resume_parser.config import PresetConfigs


def example_basic_usage():
    """Базовый пример использования"""
    print("=" * 60)
    print("🚀 БАЗОВЫЙ ПРИМЕР ИСПОЛЬЗОВАНИЯ")
    print("=" * 60)
    
    # Тестовое резюме
    resume_text = """
    Анна Смирнова
    Email: anna.smirnova@example.com
    Телефон: +7 (987) 654-32-10
    Москва
    
    Желаемая должность: Frontend разработчик
    
    Опыт работы:
    Senior Frontend Developer
    WebTech Ltd
    2021 - настоящее время
    - Разработка SPA на React
    - Оптимизация производительности
    - Наставничество junior разработчиков
    
    Middle Frontend Developer  
    StartupCorp
    2019 - 2021
    - Создание интерфейсов на Vue.js
    - Интеграция с REST API
    
    Образование:
    МГУ имени М.В. Ломоносова
    Факультет вычислительной математики и кибернетики
    Бакалавр, 2019
    
    Навыки:
    JavaScript, TypeScript, React, Vue.js, HTML5, CSS3, SCSS, 
    Webpack, Git, Jest, Cypress, Figma
    """
    
    # Создание парсера
    parser = UniversalResumeParser()
    
    # Парсинг
    result = parser.parse(resume_text)
    
    print(f"👤 Имя: {result.personal_info.full_name}")
    if result.personal_info.contact_info:
        print(f"📧 Email: {result.personal_info.contact_info.email}")
        print(f"📞 Телефон: {result.personal_info.contact_info.phone}")
    print(f"🏙️  Город: {result.personal_info.city}")
    print(f"💼 Желаемая должность: {result.job_preferences.desired_position}")
    print(f"📊 Уверенность парсинга: {result.parsing_confidence:.1%}")
    
    print(f"\n💡 Опыт работы ({len(result.experience)} позиций):")
    for i, exp in enumerate(result.experience, 1):
        print(f"  {i}. {exp.position} в {exp.company}")
        if exp.seniority_level:
            print(f"     Уровень: {exp.seniority_level.value}")
    
    print(f"\n🛠️  Навыки ({len(result.skills)}):")
    skills_by_category = {}
    for skill in result.skills:
        category = skill.category.value if skill.category else 'other'
        if category not in skills_by_category:
            skills_by_category[category] = []
        skills_by_category[category].append(skill.name)
    
    for category, skills in skills_by_category.items():
        print(f"  {category.title()}: {', '.join(skills[:5])}")
    
    return result


def example_different_configs():
    """Примеры с разными конфигурациями"""
    print("\n" + "=" * 60)
    print("⚙️  ПРИМЕРЫ С РАЗНЫМИ КОНФИГУРАЦИЯМИ")
    print("=" * 60)
    
    resume_text = "Петр Иванов\nEmail: petr@test.com\nPython Developer\nНавыки: Python, Django"
    
    configs = [
        ("По умолчанию", PresetConfigs.default()),
        ("Высокая точность", PresetConfigs.high_accuracy()),
        ("Быстрая обработка", PresetConfigs.fast_processing()),
        ("Минимальная", PresetConfigs.minimal()),
    ]
    
    for config_name, config in configs:
        parser = UniversalResumeParser(config)
        result = parser.parse(resume_text)
        
        print(f"📝 {config_name}: {result.parsing_confidence:.1%} уверенность")


def example_batch_processing():
    """Пример пакетной обработки"""
    print("\n" + "=" * 60)
    print("📦 ПРИМЕР ПАКЕТНОЙ ОБРАБОТКИ")
    print("=" * 60)
    
    # Несколько тестовых резюме
    resumes = [
        """Сергей Петров
        Email: sergey@example.com
        Senior Java Developer
        Навыки: Java, Spring, PostgreSQL""",
        
        """Елена Иванова  
        Телефон: +7-999-888-77-66
        UX Designer
        Навыки: Figma, Sketch, Prototyping""",
        
        """Михаил Сидоров
        Email: mikhail@test.org
        DevOps Engineer  
        Навыки: Docker, Kubernetes, AWS, Jenkins"""
    ]
    
    # Пакетная обработка
    results = parse_resume_batch(resumes, max_workers=2)
    
    print(f"📊 Обработано: {len(results)} резюме")
    successful = sum(1 for r in results if r['success'])
    print(f"✅ Успешно: {successful}/{len(results)}")
    
    for i, result in enumerate(results, 1):
        if result['success']:
            data = result['data']
            name = data['personal_info']['full_name']
            confidence = result['confidence']
            print(f"  {i}. {name}: {confidence:.1%}")
        else:
            print(f"  {i}. Ошибка: {result['error']}")


def example_json_export():
    """Пример экспорта в JSON"""
    print("\n" + "=" * 60)
    print("📄 ПРИМЕР ЭКСПОРТА В JSON")
    print("=" * 60)
    
    resume_text = """
    Ольга Козлова
    Email: olga@example.com
    Data Scientist
    Навыки: Python, Machine Learning, TensorFlow
    """
    
    result = parse_resume(resume_text)
    
    # Простое представление основных данных
    simple_data = {
        'name': result.personal_info.full_name,
        'email': result.personal_info.contact_info.email if result.personal_info.contact_info else None,
        'phone': result.personal_info.contact_info.phone if result.personal_info.contact_info else None,
        'city': result.personal_info.city,
        'desired_position': result.job_preferences.desired_position if result.job_preferences else None,
        'experience_count': len(result.experience),
        'skills_count': len(result.skills),
        'education_count': len(result.education),
        'confidence': result.parsing_confidence,
        'skills': [skill.name for skill in result.skills],
        'experience': [{'position': exp.position, 'company': exp.company} for exp in result.experience]
    }
    
    # Экспорт в JSON
    json_output = json.dumps(simple_data, ensure_ascii=False, indent=2)
    
    print("📋 JSON структура:")
    print(json_output)
    
    # Сохранение в файл
    with open('/tmp/resume_example.json', 'w', encoding='utf-8') as f:
        f.write(json_output)
    
    print(f"\n💾 Сохранено в файл: /tmp/resume_example.json")


def example_metrics_analysis():
    """Пример анализа метрик"""
    print("\n" + "=" * 60)
    print("📈 ПРИМЕР АНАЛИЗА МЕТРИК")
    print("=" * 60)
    
    parser = UniversalResumeParser()
    
    # Парсим несколько резюме для сбора статистики
    test_resumes = [
        "Иван Тестов\nEmail: ivan@test.com\nДизайнер\nНавыки: Photoshop",
        "Мария Примерова\nТелефон: +7-111-222-33-44\nМаркетолог\nНавыки: SMM, Analytics",
        "Андрей Кодеров\nEmail: andrey@dev.com\nBackend Developer\nНавыки: Python, Django, Redis"
    ]
    
    for resume in test_resumes:
        parser.parse(resume)
    
    # Получение статистики
    stats = parser.get_parsing_statistics()
    
    print(f"📊 Статистика парсера:")
    print(f"  Всего обработано: {stats['total_parsed']}")
    print(f"  Успешно: {stats['successful_parsed']}")
    print(f"  Процент успеха: {stats['success_rate']:.1%}")
    print(f"  Средняя уверенность: {stats['average_confidence']:.1%}")
    print(f"  Среднее время: {stats['average_processing_time']:.3f}с")
    
    # Экспорт детальных метрик
    detailed_metrics = parser.export_metrics()
    print(f"\n📋 Собрано метрик: {len(detailed_metrics['metrics'])}")


def example_error_handling():
    """Пример обработки ошибок"""
    print("\n" + "=" * 60)
    print("⚠️  ПРИМЕР ОБРАБОТКИ ОШИБОК")
    print("=" * 60)
    
    from resume_parser.core.exceptions import ParsingError, TextExtractionError
    
    parser = UniversalResumeParser()
    
    # Тест с некорректными данными
    test_cases = [
        ("Слишком короткий текст", "abc"),
        ("Пустая строка", ""),
        ("Только пробелы", "   \n\n   "),
        ("Корректный текст", "Иван Иванов\nEmail: test@example.com\nРазработчик")
    ]
    
    for test_name, test_text in test_cases:
        try:
            result = parser.parse(test_text)
            print(f"✅ {test_name}: успешно (уверенность: {result.parsing_confidence:.1%})")
        except ParsingError as e:
            print(f"❌ {test_name}: ошибка парсинга - {e}")
        except Exception as e:
            print(f"❌ {test_name}: неожиданная ошибка - {e}")


def example_custom_validation():
    """Пример кастомной валидации"""
    print("\n" + "=" * 60)
    print("🔍 ПРИМЕР КАСТОМНОЙ ВАЛИДАЦИИ")  
    print("=" * 60)
    
    from resume_parser.utils.helpers import DataValidator
    
    validator = DataValidator()
    
    # Тестирование валидации разных полей
    test_data = [
        ("Email", "test@example.com", validator.validate_email),
        ("Невалидный email", "invalid-email", validator.validate_email),
        ("Телефон РФ", "+7-999-123-45-67", validator.validate_phone),
        ("Имя", "Иван Петров", validator.validate_name),
        ("URL", "https://github.com/user", validator.validate_url),
    ]
    
    for field_name, value, validate_func in test_data:
        try:
            is_valid, score = validate_func(value)
            status = "✅" if is_valid else "❌"
            print(f"{status} {field_name}: {value} (оценка: {score:.2f})")
        except Exception as e:
            print(f"❌ {field_name}: ошибка валидации - {e}")


def main():
    """Запуск всех примеров"""
    print("🧪 ПРИМЕРЫ ИСПОЛЬЗОВАНИЯ УНИВЕРСАЛЬНОГО AI-ПАРСЕРА РЕЗЮМЕ")
    print("=" * 60)
    
    try:
        # Запуск всех примеров
        example_basic_usage()
        example_different_configs()
        example_batch_processing() 
        example_json_export()
        example_metrics_analysis()
        example_error_handling()
        example_custom_validation()
        
        print("\n" + "=" * 60)
        print("🎉 ВСЕ ПРИМЕРЫ ВЫПОЛНЕНЫ УСПЕШНО!")
        print("=" * 60)
        print("Парсер готов к использованию в production!")
        
    except Exception as e:
        print(f"\n❌ Ошибка выполнения примеров: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()