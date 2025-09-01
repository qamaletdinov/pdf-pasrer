"""
Тест универсального парсера резюме
"""

import sys
sys.path.append('/home/runner/work/Raspisaniye/Raspisaniye')

from resume_parser import UniversalResumeParser
from resume_parser.config import PresetConfigs

def test_text_parsing():
    """Тест парсинга текстового резюме"""
    
    # Тестовый текст резюме
    test_resume = """
    Иван Петров
    Email: ivan.petrov@example.com
    Телефон: +7 (123) 456-78-90
    Москва
    
    Желаемая должность: Python разработчик
    
    Опыт работы:
    Senior Python Developer
    ТехКомпания ООО
    2020 - настоящее время
    Разработка веб-приложений на Django и Flask
    
    Python Developer
    StartupCorp
    2018 - 2020
    Разработка REST API и микросервисов
    
    Образование:
    Московский государственный университет
    Факультет вычислительной математики и кибернетики
    Бакалавр, 2018
    
    Навыки:
    Python, Django, Flask, PostgreSQL, Redis, Docker, Git, Linux
    """
    
    print("🚀 Тестирование универсального парсера резюме...")
    print("=" * 60)
    
    try:
        # Создание парсера с конфигурацией по умолчанию
        config = PresetConfigs.default()
        parser = UniversalResumeParser(config)
        
        print("✅ Парсер успешно инициализирован")
        
        # Парсинг резюме
        print("\n📄 Парсинг текстового резюме...")
        result = parser.parse(test_resume)
        
        print("✅ Парсинг завершен успешно!")
        print(f"📊 Уверенность парсинга: {result.parsing_confidence:.1%}")
        
        # Вывод результатов
        print("\n" + "=" * 60)
        print("📋 РЕЗУЛЬТАТЫ ПАРСИНГА:")
        print("=" * 60)
        
        # Личная информация
        if result.personal_info:
            print(f"👤 Имя: {result.personal_info.full_name}")
            if result.personal_info.contact_info:
                print(f"📧 Email: {result.personal_info.contact_info.email}")
                print(f"📞 Телефон: {result.personal_info.contact_info.phone}")
            print(f"🏙️  Город: {result.personal_info.city}")
        
        # Предпочтения по работе
        if result.job_preferences:
            print(f"💼 Желаемая должность: {result.job_preferences.desired_position}")
        
        # Опыт работы
        if result.experience:
            print(f"\n💡 Опыт работы ({len(result.experience)} позиций):")
            for i, exp in enumerate(result.experience, 1):
                print(f"  {i}. {exp.position}")
                if exp.company:
                    print(f"     Компания: {exp.company}")
                if exp.seniority_level:
                    print(f"     Уровень: {exp.seniority_level.value}")
        
        # Образование
        if result.education:
            print(f"\n🎓 Образование ({len(result.education)} записей):")
            for i, edu in enumerate(result.education, 1):
                print(f"  {i}. {edu.institution}")
        
        # Навыки
        if result.skills:
            print(f"\n🛠️  Навыки ({len(result.skills)} навыков):")
            skills_by_category = {}
            for skill in result.skills:
                category = skill.category.value if skill.category else 'other'
                if category not in skills_by_category:
                    skills_by_category[category] = []
                skills_by_category[category].append(skill.name)
            
            for category, skills in skills_by_category.items():
                print(f"  {category.title()}: {', '.join(skills)}")
        
        # Метаданные
        if result.metadata:
            print(f"\n⚙️  Метаданные:")
            print(f"  Время обработки: {result.metadata.parsing_duration_ms:.0f}мс")
            print(f"  Длина текста: {result.metadata.text_length} символов")
            print(f"  Формат: {result.metadata.detected_format}")
        
        # Статистика парсера
        print("\n" + "=" * 60)
        print("📈 СТАТИСТИКА ПАРСЕРА:")
        print("=" * 60)
        
        stats = parser.get_parsing_statistics()
        print(f"Всего обработано: {stats['total_parsed']}")
        print(f"Успешно: {stats['successful_parsed']}")
        print(f"Процент успеха: {stats['success_rate']:.1%}")
        print(f"Средняя уверенность: {stats['average_confidence']:.1%}")
        print(f"Среднее время: {stats['average_processing_time']:.3f}с")
        
        return True
        
    except Exception as e:
        print(f"❌ Ошибка при тестировании: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_batch_processing():
    """Тест пакетной обработки"""
    
    test_resumes = [
        "Анна Сидорова\nEmail: anna@test.com\nНавыки: JavaScript, React, CSS",
        "Петр Иванов\nТелефон: +7-999-123-45-67\nРазработчик Java\nНавыки: Java, Spring, MySQL",
        "Мария Козлова\nEmail: maria@example.org\nДизайнер\nНавыки: Photoshop, Figma, UI/UX"
    ]
    
    print("\n" + "=" * 60)
    print("🔄 ТЕСТ ПАКЕТНОЙ ОБРАБОТКИ:")
    print("=" * 60)
    
    try:
        config = PresetConfigs.fast_processing()
        parser = UniversalResumeParser(config)
        
        results = parser.parse_batch(test_resumes)
        
        print(f"✅ Пакетная обработка завершена!")
        print(f"📊 Обработано: {len(results)} резюме")
        
        successful = sum(1 for r in results if r['success'])
        print(f"✅ Успешно: {successful}/{len(results)}")
        
        for i, result in enumerate(results, 1):
            status = "✅" if result['success'] else "❌"
            confidence = f"{result['confidence']:.1%}" if result['success'] else "N/A"
            print(f"  {status} Резюме {i}: уверенность {confidence}")
        
        return True
        
    except Exception as e:
        print(f"❌ Ошибка пакетной обработки: {e}")
        return False


if __name__ == "__main__":
    print("🧪 ТЕСТИРОВАНИЕ УНИВЕРСАЛЬНОГО AI-ПАРСЕРА РЕЗЮМЕ")
    print("=" * 60)
    
    # Тест текстового парсинга
    test1_success = test_text_parsing()
    
    # Тест пакетной обработки
    test2_success = test_batch_processing()
    
    # Итоговый результат
    print("\n" + "=" * 60)
    print("📋 ИТОГОВЫЕ РЕЗУЛЬТАТЫ ТЕСТИРОВАНИЯ:")
    print("=" * 60)
    
    print(f"Тест текстового парсинга: {'✅ ПРОЙДЕН' if test1_success else '❌ ПРОВАЛЕН'}")
    print(f"Тест пакетной обработки: {'✅ ПРОЙДЕН' if test2_success else '❌ ПРОВАЛЕН'}")
    
    if test1_success and test2_success:
        print("\n🎉 ВСЕ ТЕСТЫ ПРОЙДЕНЫ УСПЕШНО!")
        print("Универсальный AI-парсер резюме готов к использованию!")
    else:
        print("\n⚠️  НЕКОТОРЫЕ ТЕСТЫ ПРОВАЛЕНЫ")
        print("Необходима дополнительная отладка")