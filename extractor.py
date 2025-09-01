import json
import logging
from typing import Optional, Dict, Any
from pathlib import Path
from dataclasses import dataclass, asdict
from abc import ABC, abstractmethod

# PDF библиотеки
try:
    import PyPDF2
except ImportError:
    PyPDF2 = None

try:
    import pdfplumber
except ImportError:
    pdfplumber = None

try:
    from pdfminer.high_level import extract_text
except ImportError:
    extract_text = None


@dataclass
class ExtractionResult:
    """Результат извлечения текста"""
    file_path: str
    method_used: str
    text: str
    text_length: int
    lines_count: int
    words_count: int
    extraction_quality_score: float
    error_message: Optional[str] = None


class PDFTextExtractor(ABC):
    """Абстрактный базовый класс для извлечения текста"""

    @abstractmethod
    def extract(self, pdf_path: Path) -> str:
        """Извлекает текст из PDF файла"""
        pass

    @abstractmethod
    def get_method_name(self) -> str:
        """Возвращает название метода извлечения"""
        pass


class PyPDF2Extractor(PDFTextExtractor):
    """Извлечение текста с помощью PyPDF2"""

    def extract(self, pdf_path: Path) -> str:
        if PyPDF2 is None:
            raise ImportError("PyPDF2 не установлен")

        text = ""
        with open(pdf_path, 'rb') as file:
            reader = PyPDF2.PdfReader(file)
            for page_num, page in enumerate(reader.pages):
                try:
                    page_text = page.extract_text()
                    if page_text:
                        text += f"\n=== СТРАНИЦА {page_num + 1} ===\n"
                        text += page_text + "\n"
                except Exception as e:
                    logging.warning(f"Ошибка извлечения страницы {page_num + 1}: {e}")

        return text

    def get_method_name(self) -> str:
        return "PyPDF2"


class PdfPlumberExtractor(PDFTextExtractor):
    """Извлечение текста с помощью pdfplumber"""

    def extract(self, pdf_path: Path) -> str:
        if pdfplumber is None:
            raise ImportError("pdfplumber не установлен")

        text = ""
        with pdfplumber.open(pdf_path) as pdf:
            for page_num, page in enumerate(pdf.pages):
                try:
                    page_text = page.extract_text()
                    if page_text:
                        text += f"\n=== СТРАНИЦА {page_num + 1} ===\n"
                        text += page_text + "\n"

                    # Дополнительно извлекаем таблицы если есть
                    tables = page.extract_tables()
                    if tables:
                        text += f"\n--- ТАБЛИЦЫ НА СТРАНИЦЕ {page_num + 1} ---\n"
                        for table_num, table in enumerate(tables):
                            text += f"Таблица {table_num + 1}:\n"
                            for row in table:
                                if row:
                                    text += " | ".join(str(cell) if cell else "" for cell in row) + "\n"
                            text += "\n"

                except Exception as e:
                    logging.warning(f"Ошибка извлечения страницы {page_num + 1}: {e}")

        return text

    def get_method_name(self) -> str:
        return "pdfplumber"


class PdfMinerExtractor(PDFTextExtractor):
    """Извлечение текста с помощью pdfminer"""

    def extract(self, pdf_path: Path) -> str:
        if extract_text is None:
            raise ImportError("pdfminer не установлен")

        return extract_text(str(pdf_path))

    def get_method_name(self) -> str:
        return "pdfminer"


class TextQualityAnalyzer:
    """Анализатор качества извлеченного текста"""

    @staticmethod
    def analyze_quality(text: str) -> float:
        """Анализирует качество текста и возвращает оценку от 0 до 1"""
        if not text:
            return 0.0

        # Критерии качества
        scores = []

        # 1. Длина текста (больше = лучше, до определенного предела)
        length_score = min(len(text) / 5000, 1.0)
        scores.append(length_score * 0.2)

        # 2. Соотношение читаемых символов
        total_chars = len(text)
        readable_chars = len([c for c in text if c.isalnum() or c.isspace() or c in '.,!?;:()[]{}'])
        readable_ratio = readable_chars / total_chars if total_chars > 0 else 0
        scores.append(readable_ratio * 0.3)

        # 3. Наличие структуры (переносы строк)
        lines = text.count('\n')
        structure_score = min(lines / 100, 1.0)
        scores.append(structure_score * 0.2)

        # 4. Разнообразие слов
        words = text.split()
        unique_words = len(set(word.lower() for word in words))
        diversity_score = min(unique_words / len(words), 1.0) if words else 0
        scores.append(diversity_score * 0.15)

        # 5. Отсутствие повторяющихся символов (признак плохого извлечения)
        repeated_chars = sum(1 for i in range(len(text) - 2)
                             if text[i] == text[i + 1] == text[i + 2])
        no_repetition_score = max(0, 1 - (repeated_chars / len(text)))
        scores.append(no_repetition_score * 0.15)

        return sum(scores)


class AdvancedPDFExtractor:
    """Продвинутый извлекатель текста с множественными методами"""

    def __init__(self):
        self.extractors = []
        self.quality_analyzer = TextQualityAnalyzer()

        # Инициализация доступных extractors
        if pdfplumber:
            self.extractors.append(PdfPlumberExtractor())
        if PyPDF2:
            self.extractors.append(PyPDF2Extractor())
        if extract_text:
            self.extractors.append(PdfMinerExtractor())

        if not self.extractors:
            raise ImportError(
                "Не установлена ни одна библиотека для работы с PDF. "
                "Установите: pip install PyPDF2 pdfplumber pdfminer.six"
            )

        # Настройка логирования
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler('pdf_extraction.log'),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)

    def extract_with_all_methods(self, pdf_path: str) -> Dict[str, ExtractionResult]:
        """Извлекает текст всеми доступными методами для сравнения"""
        path = Path(pdf_path)

        if not path.exists():
            raise FileNotFoundError(f"Файл {pdf_path} не найден")

        results = {}

        for extractor in self.extractors:
            method_name = extractor.get_method_name()
            self.logger.info(f"Извлечение текста методом: {method_name}")

            try:
                # Извлечение текста
                text = extractor.extract(path)

                # Анализ качества
                quality_score = self.quality_analyzer.analyze_quality(text)

                # Статистика
                lines_count = text.count('\n')
                words_count = len(text.split())

                result = ExtractionResult(
                    file_path=str(path),
                    method_used=method_name,
                    text=text,
                    text_length=len(text),
                    lines_count=lines_count,
                    words_count=words_count,
                    extraction_quality_score=quality_score
                )

                results[method_name] = result

                self.logger.info(
                    f"{method_name}: {len(text)} символов, "
                    f"качество: {quality_score:.3f}"
                )

            except Exception as e:
                error_msg = f"Ошибка {method_name}: {str(e)}"
                self.logger.error(error_msg)

                result = ExtractionResult(
                    file_path=str(path),
                    method_used=method_name,
                    text="",
                    text_length=0,
                    lines_count=0,
                    words_count=0,
                    extraction_quality_score=0.0,
                    error_message=error_msg
                )

                results[method_name] = result

        return results

    def extract_best_text(self, pdf_path: str) -> ExtractionResult:
        """Извлекает текст лучшим доступным методом"""
        results = self.extract_with_all_methods(pdf_path)

        # Находим лучший результат по качеству
        best_result = max(
            results.values(),
            key=lambda r: r.extraction_quality_score
        )

        self.logger.info(
            f"Лучший метод: {best_result.method_used} "
            f"(качество: {best_result.extraction_quality_score:.3f})"
        )

        return best_result

    def save_extraction_results(self, pdf_path: str, output_dir: str = "extraction_results"):
        """Сохраняет результаты извлечения всеми методами"""
        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True)

        pdf_name = Path(pdf_path).stem
        results = self.extract_with_all_methods(pdf_path)

        # Сохранение текста каждым методом
        for method_name, result in results.items():
            if result.text:
                text_file = output_path / f"{pdf_name}_{method_name}.txt"
                with open(text_file, 'w', encoding='utf-8') as f:
                    f.write(f"=== ИЗВЛЕЧЕНИЕ МЕТОДОМ: {method_name} ===\n")
                    f.write(f"Файл: {result.file_path}\n")
                    f.write(f"Длина текста: {result.text_length} символов\n")
                    f.write(f"Количество строк: {result.lines_count}\n")
                    f.write(f"Количество слов: {result.words_count}\n")
                    f.write(f"Оценка качества: {result.extraction_quality_score:.3f}\n")
                    f.write("=" * 50 + "\n\n")
                    f.write(result.text)

                self.logger.info(f"Сохранен: {text_file}")

        # Сохранение сравнительного отчета
        report_file = output_path / f"{pdf_name}_comparison_report.json"
        report_data = {
            "file_path": pdf_path,
            "extraction_timestamp": "2025-09-01 18:51:34",
            "methods_comparison": {}
        }

        for method_name, result in results.items():
            report_data["methods_comparison"][method_name] = {
                "success": result.error_message is None,
                "text_length": result.text_length,
                "lines_count": result.lines_count,
                "words_count": result.words_count,
                "quality_score": result.extraction_quality_score,
                "error_message": result.error_message
            }

        # Определяем рекомендуемый метод
        best_method = max(
            results.keys(),
            key=lambda k: results[k].extraction_quality_score
        )
        report_data["recommended_method"] = best_method

        with open(report_file, 'w', encoding='utf-8') as f:
            json.dump(report_data, f, ensure_ascii=False, indent=2)

        self.logger.info(f"Отчет сохранен: {report_file}")

        return report_data

    def analyze_text_structure(self, text: str) -> Dict[str, Any]:
        """Анализирует структуру извлеченного текста"""
        lines = text.split('\n')

        analysis = {
            "общая_статистика": {
                "всего_символов": len(text),
                "всего_строк": len(lines),
                "всего_слов": len(text.split()),
                "пустых_строк": sum(1 for line in lines if not line.strip())
            },
            "структурные_элементы": {
                "заголовки": [],
                "секции": [],
                "контактная_информация": {},
                "даты": [],
                "email_адреса": [],
                "телефоны": []
            }
        }

        # Поиск потенциальных заголовков (короткие строки в верхнем регистре)
        for i, line in enumerate(lines[:20]):
            line = line.strip()
            if line and len(line) < 50 and (line.isupper() or line.istitle()):
                analysis["структурные_элементы"]["заголовки"].append({
                    "строка": i + 1,
                    "текст": line
                })

        # Поиск контактной информации
        import re

        email_pattern = re.compile(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b')
        phone_pattern = re.compile(r'(\+?7[\s\-\(\)]?\d{3}[\s\-\(\)]?\d{3}[\s\-]?\d{2}[\s\-]?\d{2})')
        date_pattern = re.compile(r'\b\d{1,2}[\.\/\-]\d{1,2}[\.\/\-]\d{2,4}\b|\b\d{4}\b')

        analysis["структурные_элементы"]["email_адреса"] = email_pattern.findall(text)
        analysis["структурные_элементы"]["телефоны"] = phone_pattern.findall(text)
        analysis["структурные_элементы"]["даты"] = date_pattern.findall(text)

        return analysis


def main():
    """Основная функция для демонстрации работы"""
    extractor = AdvancedPDFExtractor()

    print("🔍 PDF Text Extractor v1.0")
    print("=" * 50)
    print("Доступные методы извлечения:")

    for i, ext in enumerate(extractor.extractors, 1):
        print(f"  {i}. {ext.get_method_name()}")

    print("\n" + "=" * 50)

    # Пример использования
    pdf_file = input("Введите путь к PDF файлу: ").strip()

    if not pdf_file:
        # Используем тестовый файл для демонстрации
        print("Для демонстрации создайте файл 'test_resume.pdf' или укажите путь к существующему PDF")
        return

    try:
        print(f"\n📄 Обработка файла: {pdf_file}")

        # Извлечение всеми методами
        print("🔄 Извлечение текста всеми доступными методами...")
        report = extractor.save_extraction_results(pdf_file)

        print(f"\n📊 Результаты сравнения:")
        for method, stats in report["methods_comparison"].items():
            if stats["success"]:
                print(f"  ✅ {method}: {stats['text_length']} символов, "
                      f"качество: {stats['quality_score']:.3f}")
            else:
                print(f"  ❌ {method}: {stats['error_message']}")

        print(f"\n🏆 Рекомендуемый метод: {report['recommended_method']}")

        # Получение лучшего результата
        best_result = extractor.extract_best_text(pdf_file)

        if best_result.text:
            # Анализ структуры
            print("\n🔍 Анализ структуры текста...")
            structure = extractor.analyze_text_structure(best_result.text)

            print(f"📈 Статистика:")
            stats = structure["общая_статистика"]
            print(f"  • Символов: {stats['всего_символов']}")
            print(f"  • Строк: {stats['всего_строк']}")
            print(f"  • Слов: {stats['всего_слов']}")

            elements = structure["структурные_элементы"]
            if elements["email_адреса"]:
                print(f"  • Email: {', '.join(elements['email_адреса'])}")
            if elements["телефоны"]:
                print(f"  • Телефоны: {', '.join(elements['телефоны'])}")

            print(f"\n💾 Файлы сохранены в папку 'extraction_results/'")

            # Показываем первые 500 символов для предварительного просмотра
            print(f"\n📖 Предварительный просмотр (первые 500 символов):")
            print("-" * 50)
            print(best_result.text[:500])
            if len(best_result.text) > 500:
                print("...")
            print("-" * 50)

    except Exception as e:
        print(f"❌ Ошибка: {e}")


if __name__ == "__main__":
    main()