"""Примеры использования ультра-точного парсера резюме"""

from ai_resume_parser import UltraPreciseResumeParser, ResumeParserValidator, PerformanceBenchmark
import json

def example_single_file_parsing():
    """Пример парсинга одного файла с детальной диагностикой"""
    parser = UltraPreciseResumeParser()
    validator = ResumeParserValidator()

    # Ваш PDF файл
    pdf_path = "resume.pdf"

    try:
        # Извлечение текста (используйте ваш метод)
        with open("extracted_text.txt", "r", encoding="utf-8") as f:
            resume_text = f.read()

        print("🚀 Начинаем ультра-точный парсинг...")

        # Парсинг
        resume_data = parser.parse_resume_text(resume_text)

        # Валидация
        validation_results = validator.validate_resume_data(resume_data)

        # Детальный отчет
        report = parser.generate_detailed_report(resume_data)

        # Сохранение результатов
        json_output = parser.parse_to_json(resume_text, "output/parsed_resume.json")

        with open("output/detailed_report.txt", "w", encoding="utf-8") as f:
            f.write(report)

        with open("output/validation_results.json", "w", encoding="utf-8") as f:
            json.dump(validation_results, f, ensure_ascii=False, indent=2)

        print(f"✅ Парсинг завершен!")
        print(f"🎯 Точность: {resume_data.parsing_confidence:.1%}")
        print(f"✔️ Валидация: {'ПРОЙДЕНА' if validation_results['overall_valid'] else 'НЕ ПРОЙДЕНА'}")
        print(f"💾 Результаты сохранены в папку output/")

    except Exception as e:
        print(f"❌ Ошибка: {e}")

def example_batch_processing():
    """Пример пакетной обработки резюме"""
    parser = UltraPreciseResumeParser()
    validator = ResumeParserValidator()

    # Список файлов резюме
    resume_files = ["resume1.txt", "resume2.txt", "resume3.txt"]

    results = []

    for i, file_path in enumerate(resume_files, 1):
        try:
            print(f"📄 Обрабатываем файл {i}/{len(resume_files)}: {file_path}")

            with open(file_path, "r", encoding="utf-8") as f:
                resume_text = f.read()

            # Парсинг
            resume_data = parser.parse_resume_text(resume_text)

            # Валидация
            validation = validator.validate_resume_data(resume_data)

            # Сохранение результата
            output_file = f"output/batch_result_{i}.json"
            parser.parse_to_json(resume_text, output_file)

            results.append({
                'file': file_path,
                'confidence': resume_data.parsing_confidence,
                'valid': validation['overall_valid'],
                'quality_score': validation['quality_score'],
                'candidate_name': resume_data.personal_info.full_name,
                'position': resume_data.job_preferences.desired_position
            })

            print(f"  ✅ Успешно: {resume_data.parsing_confidence:.1%} точность")

        except Exception as e:
            print(f"  ❌ Ошибка: {e}")
            results.append({
                'file': file_path,
                'error': str(e)
            })

    # Сохранение сводного отчета
    with open("output/batch_summary.json", "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    print(f"\n📊 Пакетная обработка завершена. Обработано: {len(results)} файлов")

def example_performance_benchmark():
    """Пример бенчмарка производительности"""
    parser = UltraPreciseResumeParser()
    benchmark = PerformanceBenchmark()

    # Тестовые резюме (в реальности загрузите из файлов)
    test_resumes = [
        "Тестовое резюме 1...",
        "Тестовое резюме 2...",
        "Тестовое резюме 3..."
    ]

    print("🏃‍♂️ Запуск бенчмарка производительности...")

    # Запуск бенчмарка
    benchmark_results = benchmark.benchmark_parser(parser, test_resumes, iterations=3)

    print(f"\n📊 РЕЗУЛЬТАТЫ БЕНЧМАРКА:")
    print(f"⏱️ Среднее время парсинга: {benchmark_results['average_parsing_time']:.2f}с")
    print(f"📈 Медианное время: {benchmark_results['median_parsing_time']:.2f}с")
    print(f"🎯 Средняя уверенность: {benchmark_results['average_confidence']:.1%}")
    print(f"💾 Среднее использование памяти: {benchmark_results['average_memory_usage']:.1f}MB")
    print(f"🧪 Всего тестов: {benchmark_results['total_tests']}")

if __name__ == "__main__":
    # Выберите нужный пример
    print("Выберите пример для запуска:")
    print("1. Парсинг одного файла")
    print("2. Пакетная обработка")
    print("3. Бенчмарк производительности")

    choice = input("Введите номер (1-3): ")

    if choice == "1":
        example_single_file_parsing()
    elif choice == "2":
        example_batch_processing()
    elif choice == "3":
        example_performance_benchmark()
    else:
        print("Некорректный выбор!")