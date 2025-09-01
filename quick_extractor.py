"""Быстрый извлекатель для тестирования одного файла"""

from extractor import AdvancedPDFExtractor
import sys


def quick_extract(pdf_path: str):
    """Быстрое извлечение для анализа структуры"""
    extractor = AdvancedPDFExtractor()

    print(f"🔄 Извлечение текста из: {pdf_path}")

    try:
        # Получаем лучший результат
        result = extractor.extract_best_text(pdf_path)

        print(f"✅ Успешно извлечено {result.text_length} символов")
        print(f"📊 Метод: {result.method_used}")
        print(f"🎯 Качество: {result.extraction_quality_score:.3f}")

        # Сохраняем в простой текстовый файл
        output_file = f"extracted_text.txt"
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write("=== ИЗВЛЕЧЕННЫЙ ТЕКСТ ===\n")
            f.write(f"Источник: {pdf_path}\n")
            f.write(f"Метод: {result.method_used}\n")
            f.write(f"Качество: {result.extraction_quality_score:.3f}\n")
            f.write("=" * 50 + "\n\n")
            f.write(result.text)

        print(f"💾 Текст сохранен в: {output_file}")

        # Показываем первые 1000 символов
        print("\n📖 ПРЕДВАРИТЕЛЬНЫЙ ПРОСМОТР:")
        print("=" * 60)
        print(result.text[:1000])
        if len(result.text) > 1000:
            print("\n... (текст обрезан, полный текст в файле)")
        print("=" * 60)

        return result.text

    except Exception as e:
        print(f"❌ Ошибка: {e}")
        return None


if __name__ == "__main__":
    if len(sys.argv) > 1:
        pdf_file = sys.argv[1]
    else:
        pdf_file = input("Введите путь к PDF файлу: ").strip()

    if pdf_file:
        quick_extract(pdf_file)
    else:
        print("Путь к файлу не указан!")