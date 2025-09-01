import json
import re
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, asdict
from pathlib import Path
import logging
from abc import ABC, abstractmethod

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

try:
    import spacy
    from spacy.matcher import Matcher
except ImportError:
    spacy = None
    Matcher = None


@dataclass
class PersonalInfo:
    """Класс для хранения персональной информации"""
    name: Optional[str] = None
    email: Optional[str] = None
    phone: Optional[str] = None
    address: Optional[str] = None
    linkedin: Optional[str] = None
    github: Optional[str] = None


@dataclass
class Experience:
    """Класс для хранения информации об опыте работы"""
    position: Optional[str] = None
    company: Optional[str] = None
    start_date: Optional[str] = None
    end_date: Optional[str] = None
    description: Optional[str] = None
    responsibilities: List[str] = None

    def __post_init__(self):
        if self.responsibilities is None:
            self.responsibilities = []


@dataclass
class Education:
    """Класс для хранения информации об образовании"""
    degree: Optional[str] = None
    institution: Optional[str] = None
    graduation_date: Optional[str] = None
    gpa: Optional[str] = None


@dataclass
class ResumeData:
    """Основной класс для хранения всех данных резюме"""
    personal_info: PersonalInfo
    summary: Optional[str] = None
    skills: List[str] = None
    experience: List[Experience] = None
    education: List[Education] = None
    languages: List[str] = None
    certifications: List[str] = None

    def __post_init__(self):
        if self.skills is None:
            self.skills = []
        if self.experience is None:
            self.experience = []
        if self.education is None:
            self.education = []
        if self.languages is None:
            self.languages = []
        if self.certifications is None:
            self.certifications = []


class PDFExtractor(ABC):
    """Абстрактный базовый класс для извлечения текста из PDF"""

    @abstractmethod
    def extract_text(self, pdf_path: Path) -> str:
        """Извлекает текст из PDF файла"""
        pass


class PyPDF2Extractor(PDFExtractor):
    """Извлечение текста с помощью PyPDF2"""

    def extract_text(self, pdf_path: Path) -> str:
        if PyPDF2 is None:
            raise ImportError("PyPDF2 не установлен")

        try:
            with open(pdf_path, 'rb') as file:
                reader = PyPDF2.PdfReader(file)
                text = ""
                for page in reader.pages:
                    text += page.extract_text()
                return text
        except Exception as e:
            logging.error(f"Ошибка PyPDF2 извлечения: {e}")
            return ""


class PdfPlumberExtractor(PDFExtractor):
    """Извлечение текста с помощью pdfplumber"""

    def extract_text(self, pdf_path: Path) -> str:
        if pdfplumber is None:
            raise ImportError("pdfplumber не установлен")

        try:
            with pdfplumber.open(pdf_path) as pdf:
                text = ""
                for page in pdf.pages:
                    page_text = page.extract_text()
                    if page_text:
                        text += page_text
                return text
        except Exception as e:
            logging.error(f"Ошибка pdfplumber извлечения: {e}")
            return ""


class PdfMinerExtractor(PDFExtractor):
    """Извлечение текста с помощью pdfminer"""

    def extract_text(self, pdf_path: Path) -> str:
        if extract_text is None:
            raise ImportError("pdfminer не установлен")

        try:
            return extract_text(str(pdf_path))
        except Exception as e:
            logging.error(f"Ошибка pdfminer извлечения: {e}")
            return ""


class TextProcessor:
    """Класс для обработки и анализа текста"""

    def __init__(self, model_name: str = "ru_core_news_sm"):
        self.nlp = None
        self.matcher = None

        if spacy is not None:
            try:
                self.nlp = spacy.load(model_name)
                self.matcher = Matcher(self.nlp.vocab)
                self._setup_patterns()
            except OSError:
                logging.warning(f"Модель {model_name} не найдена. Используется базовая обработка.")
        else:
            logging.warning("spaCy не установлен. Используется базовая обработка.")

    def _setup_patterns(self):
        """Настройка паттернов для поиска"""
        if not self.matcher:
            return

        try:
            # Паттерны для email
            email_pattern = [{"LIKE_EMAIL": True}]
            self.matcher.add("EMAIL", [email_pattern])

            # Паттерны для телефона
            phone_patterns = [
                [{"TEXT": {"REGEX": r"\+?[1-9]\d{1,14}"}}],
                [{"TEXT": {"REGEX": r"\(\d{3}\)\s?\d{3}-\d{4}"}}]
            ]
            self.matcher.add("PHONE", phone_patterns)
        except Exception as e:
            logging.warning(f"Ошибка настройки паттернов: {e}")

    def clean_text(self, text: str) -> str:
        """Очистка текста от лишних символов"""
        # Удаление лишних пробелов и переносов строк
        text = re.sub(r'\s+', ' ', text)
        # Удаление специальных символов
        text = re.sub(r'[^\w\s@.\-+()]', ' ', text)
        return text.strip()

    def extract_entities(self, text: str) -> Dict[str, List[str]]:
        """Извлечение именованных сущностей"""
        entities = {"PERSON": [], "ORG": [], "GPE": []}

        if self.nlp:
            try:
                doc = self.nlp(text)
                for ent in doc.ents:
                    if ent.label_ in entities:
                        entities[ent.label_].append(ent.text)
            except Exception as e:
                logging.warning(f"Ошибка извлечения сущностей: {e}")

        return entities


class ResumeTextAnalyzer:
    """Класс для анализа текста резюме и извлечения структурированных данных"""

    def __init__(self):
        self.processor = TextProcessor()

        # Регулярные выражения для поиска различных элементов
        self.email_pattern = re.compile(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b')
        self.phone_pattern = re.compile(r'(\+?[1-9]\d{1,14}|[\(\)\d\s\-\+\.]{10,})')
        self.date_pattern = re.compile(r'\b\d{1,2}[\/\-\.]\d{1,2}[\/\-\.]\d{2,4}\b|\b\d{4}\b')

        # Ключевые слова для различных разделов
        self.section_keywords = {
            'experience': ['опыт', 'работа', 'experience', 'employment', 'career', 'трудовая деятельность'],
            'education': ['образование', 'education', 'university', 'институт', 'университет', 'учеба'],
            'skills': ['навыки', 'skills', 'технологии', 'technologies', 'умения', 'компетенции'],
            'summary': ['о себе', 'summary', 'профиль', 'profile', 'цель', 'резюме'],
            'languages': ['языки', 'languages', 'язык', 'иностранные языки'],
            'certifications': ['сертификаты', 'certifications', 'курсы', 'courses', 'дипломы']
        }

    def extract_personal_info(self, text: str) -> PersonalInfo:
        """Извлечение персональной информации"""
        personal_info = PersonalInfo()

        # Email
        email_matches = self.email_pattern.findall(text)
        if email_matches:
            personal_info.email = email_matches[0]

        # Телефон
        phone_matches = self.phone_pattern.findall(text)
        if phone_matches:
            # Очистка и форматирование телефона
            phone = re.sub(r'[^\d\+]', '', phone_matches[0])
            if len(phone) >= 10:
                personal_info.phone = phone

        # LinkedIn
        linkedin_match = re.search(r'linkedin\.com/in/[\w\-]+', text, re.IGNORECASE)
        if linkedin_match:
            personal_info.linkedin = linkedin_match.group()

        # GitHub
        github_match = re.search(r'github\.com/[\w\-]+', text, re.IGNORECASE)
        if github_match:
            personal_info.github = github_match.group()

        # Имя (используем NLP если доступно)
        entities = self.processor.extract_entities(text[:500])  # Ищем в начале документа
        if entities['PERSON']:
            personal_info.name = entities['PERSON'][0]
        else:
            # Альтернативный поиск имени в начале текста
            lines = text.split('\n')[:5]
            for line in lines:
                line = line.strip()
                # Ищем строку, которая может быть именем (2-3 слова, начинающиеся с заглавной)
                if re.match(r'^[А-ЯA-Z][а-яa-z]+\s+[А-ЯA-Z][а-яa-z]+(\s+[А-ЯA-Z][а-яa-z]+)?$', line):
                    personal_info.name = line
                    break

        return personal_info

    def extract_skills(self, text: str) -> List[str]:
        """Извлечение навыков"""
        skills = []

        # Технические навыки (расширенный список)
        tech_skills = [
            # Языки программирования
            'python', 'javascript', 'java', 'c++', 'c#', 'php', 'ruby', 'go', 'rust',
            'typescript', 'kotlin', 'swift', 'scala', 'r', 'matlab', 'perl',

            # Web технологии
            'html', 'css', 'react', 'vue', 'angular', 'nodejs', 'django', 'flask',
            'express', 'fastapi', 'spring', 'laravel', 'rails',

            # Базы данных
            'sql', 'mysql', 'postgresql', 'mongodb', 'redis', 'elasticsearch',
            'oracle', 'sqlite', 'cassandra', 'neo4j',

            # Облачные технологии
            'aws', 'azure', 'gcp', 'docker', 'kubernetes', 'jenkins',
            'terraform', 'ansible', 'vagrant',

            # Инструменты разработки
            'git', 'github', 'gitlab', 'svn', 'jira', 'confluence',

            # Machine Learning и Data Science
            'machine learning', 'deep learning', 'tensorflow', 'pytorch', 'scikit-learn',
            'data science', 'pandas', 'numpy', 'matplotlib', 'seaborn', 'jupyter',
            'keras', 'opencv', 'nlp', 'computer vision',

            # Методологии
            'agile', 'scrum', 'kanban', 'devops', 'ci/cd', 'tdd', 'bdd'
        ]

        text_lower = text.lower()
        for skill in tech_skills:
            if skill.lower() in text_lower:
                # Проверяем, что это отдельное слово, а не часть другого слова
                if re.search(r'\b' + re.escape(skill.lower()) + r'\b', text_lower):
                    skills.append(skill.title())

        return list(set(skills))  # Удаление дубликатов

    def extract_experience(self, text: str) -> List[Experience]:
        """Извлечение опыта работы"""
        experiences = []

        # Поиск секций с опытом работы
        experience_section = self._find_section(text, 'experience')
        if not experience_section:
            return experiences

        # Разделение на отдельные записи опыта
        # Ищем паттерны дат или разделители
        exp_patterns = [
            r'\d{4}\s*[-–—]\s*(\d{4}|настоящее время|present)',
            r'\d{1,2}[\.\/]\d{1,2}[\.\/]\d{4}\s*[-–—]\s*(\d{1,2}[\.\/]\d{1,2}[\.\/]\d{4}|настоящее время)',
            r'\n\s*\n',  # Двойной перенос строки
        ]

        sections = [experience_section]
        for pattern in exp_patterns:
            new_sections = []
            for section in sections:
                new_sections.extend(re.split(pattern, section))
            sections = new_sections

        for entry in sections:
            if len(entry.strip()) > 20:  # Минимальная длина записи
                exp = self._parse_experience_entry(entry.strip())
                if exp.position or exp.company:  # Добавляем только если есть основная информация
                    experiences.append(exp)

        return experiences

    def _parse_experience_entry(self, entry: str) -> Experience:
        """Парсинг отдельной записи опыта работы"""
        exp = Experience()

        # Поиск дат
        date_matches = self.date_pattern.findall(entry)
        if len(date_matches) >= 2:
            exp.start_date = date_matches[0]
            exp.end_date = date_matches[1]
        elif len(date_matches) == 1:
            if 'настоящее время' in entry.lower() or 'present' in entry.lower():
                exp.start_date = date_matches[0]
                exp.end_date = 'настоящее время'

        # Поиск должности и компании
        lines = [line.strip() for line in entry.split('\n') if line.strip()]

        for i, line in enumerate(lines):
            # Пропускаем строки с датами
            if self.date_pattern.search(line):
                continue

            # Первая значимая строка - обычно должность
            if not exp.position and len(line) > 3:
                exp.position = line
            # Вторая значимая строка - обычно компания
            elif not exp.company and len(line) > 3 and line != exp.position:
                exp.company = line

            if exp.position and exp.company:
                break

        exp.description = entry

        # Извлечение обязанностей (строки, начинающиеся с маркеров)
        responsibility_lines = re.findall(r'[-•·*]\s*(.+)', entry)
        if responsibility_lines:
            exp.responsibilities = [resp.strip() for resp in responsibility_lines]

        return exp

    def extract_education(self, text: str) -> List[Education]:
        """Извлечение информации об образовании"""
        educations = []

        education_section = self._find_section(text, 'education')
        if not education_section:
            return educations

        # Разделение на записи об образовании
        edu_entries = re.split(r'\n\s*\n|\d{4}', education_section)

        for entry in edu_entries:
            if len(entry.strip()) > 10:
                edu = self._parse_education_entry(entry.strip())
                if edu.institution or edu.degree:
                    educations.append(edu)

        return educations

    def _parse_education_entry(self, entry: str) -> Education:
        """Парсинг отдельной записи об образовании"""
        edu = Education()

        lines = [line.strip() for line in entry.split('\n') if line.strip()]

        for line in lines:
            # Поиск степени
            degree_keywords = ['бакалавр', 'магистр', 'кандидат', 'доктор', 'bachelor', 'master', 'phd', 'специалист']
            if any(keyword in line.lower() for keyword in degree_keywords):
                edu.degree = line

            # Поиск учебного заведения
            elif any(inst in line.lower() for inst in
                     ['университет', 'институт', 'university', 'college', 'училище', 'техникум']):
                edu.institution = line

            # Поиск даты окончания
            elif self.date_pattern.search(line):
                edu.graduation_date = line

            # Поиск GPA
            elif re.search(r'gpa|средний балл|оценка', line.lower()):
                edu.gpa = line

        return edu

    def _find_section(self, text: str, section_type: str) -> str:
        """Поиск конкретной секции в тексте"""
        keywords = self.section_keywords.get(section_type, [])

        # Создаем паттерн для поиска секции
        all_keywords = []
        for section_keywords_list in self.section_keywords.values():
            all_keywords.extend(section_keywords_list)

        for keyword in keywords:
            # Ищем секцию от ключевого слова до следующего раздела или конца текста
            keyword_pattern = re.escape(keyword)

            # Создаем список других ключевых слов для поиска конца секции
            other_keywords = [kw for kw in all_keywords if kw != keyword]
            other_keywords_pattern = '|'.join(re.escape(kw) for kw in other_keywords)

            if other_keywords_pattern:
                pattern = rf'({keyword_pattern}.*?)(?=\n\s*(?:{other_keywords_pattern})|$)'
            else:
                pattern = rf'({keyword_pattern}.*?)$'

            match = re.search(pattern, text, re.IGNORECASE | re.DOTALL)
            if match:
                return match.group(1)

        return ""


class ResumeParser:
    """Основной класс для парсинга резюме"""

    def __init__(self):
        self.extractors = []

        # Добавляем доступные extractors
        if pdfplumber:
            self.extractors.append(PdfPlumberExtractor())
        if PyPDF2:
            self.extractors.append(PyPDF2Extractor())
        if extract_text:
            self.extractors.append(PdfMinerExtractor())

        if not self.extractors:
            raise ImportError("Необходимо установить хотя бы одну библиотеку: PyPDF2, pdfplumber или pdfminer")

        self.analyzer = ResumeTextAnalyzer()

        # Настройка логирования
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s'
        )
        self.logger = logging.getLogger(__name__)

    def extract_text_from_pdf(self, pdf_path: Path) -> str:
        """Извлечение текста из PDF с использованием нескольких методов"""
        text = ""

        for extractor in self.extractors:
            try:
                text = extractor.extract_text(pdf_path)
                if text and len(text.strip()) > 100:  # Минимальная длина текста
                    self.logger.info(f"Успешно извлечен текст с помощью {extractor.__class__.__name__}")
                    break
            except Exception as e:
                self.logger.warning(f"Ошибка {extractor.__class__.__name__}: {e}")
                continue

        if not text:
            raise ValueError("Не удалось извлечь текст из PDF файла")

        return text

    def parse_resume(self, pdf_path: str) -> ResumeData:
        """Основной метод парсинга резюме"""
        pdf_path = Path(pdf_path)

        if not pdf_path.exists():
            raise FileNotFoundError(f"Файл {pdf_path} не найден")

        self.logger.info(f"Начало парсинга файла: {pdf_path}")

        # Извлечение текста
        text = self.extract_text_from_pdf(pdf_path)
        clean_text = self.analyzer.processor.clean_text(text)

        # Анализ и извлечение данных
        personal_info = self.analyzer.extract_personal_info(clean_text)
        skills = self.analyzer.extract_skills(clean_text)
        experience = self.analyzer.extract_experience(clean_text)
        education = self.analyzer.extract_education(clean_text)

        # Извлечение summary (первый абзац после имени)
        summary = self._extract_summary(clean_text)

        # Создание объекта с результатами
        resume_data = ResumeData(
            personal_info=personal_info,
            summary=summary,
            skills=skills,
            experience=experience,
            education=education
        )

        self.logger.info("Парсинг завершен успешно")
        return resume_data

    def _extract_summary(self, text: str) -> Optional[str]:
        """Извлечение краткого описания/цели"""
        summary_section = self.analyzer._find_section(text, 'summary')
        if summary_section:
            # Берем первые несколько предложений
            sentences = re.split(r'[.!?]+', summary_section)
            if sentences:
                return '. '.join(sentences[:3]).strip()
        return None

    def parse_to_json(self, pdf_path: str, output_path: Optional[str] = None) -> str:
        """Парсинг резюме и сохранение в JSON формат"""
        resume_data = self.parse_resume(pdf_path)

        # Конвертация в словарь
        resume_dict = asdict(resume_data)

        # Форматирование JSON
        json_output = json.dumps(resume_dict, ensure_ascii=False, indent=2)

        # Сохранение в файл если указан путь
        if output_path:
            output_path = Path(output_path)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            with open(output_path, 'w', encoding='utf-8') as f:
                f.write(json_output)
            self.logger.info(f"JSON сохранен в файл: {output_path}")

        return json_output

    def batch_parse(self, input_dir: str, output_dir: str) -> Dict[str, Any]:
        """Пакетная обработка PDF файлов"""
        input_path = Path(input_dir)
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        results = {
            'успешно_обработано': 0,
            'ошибки': 0,
            'файлы': []
        }

        pdf_files = list(input_path.glob("*.pdf"))

        for pdf_file in pdf_files:
            try:
                json_output = self.parse_to_json(
                    str(pdf_file),
                    str(output_path / f"{pdf_file.stem}.json")
                )

                results['файлы'].append({
                    'файл': pdf_file.name,
                    'статус': 'успех',
                    'размер_json': len(json_output)
                })
                results['успешно_обработано'] += 1

            except Exception as e:
                self.logger.error(f"Ошибка обработки {pdf_file}: {e}")
                results['файлы'].append({
                    'файл': pdf_file.name,
                    'статус': 'ошибка',
                    'ошибка': str(e)
                })
                results['ошибки'] += 1

        return results


def main():
    """Пример использования"""
    try:
        parser = ResumeParser()

        # Создаем тестовый пример, если нет PDF файла
        print("Парсер резюме успешно инициализирован!")
        print("Доступные extractors:", [extractor.__class__.__name__ for extractor in parser.extractors])

        # Пример парсинга (раскомментируйте, когда у вас будет PDF файл)
        # json_result = parser.parse_to_json("resume.pdf", "output/resume.json")
        # print("Результат парсинга:")
        # print(json_result)

    except Exception as e:
        print(f"Ошибка инициализации: {e}")
        print("Установите необходимые библиотеки:")
        print("pip install PyPDF2 pdfplumber pdfminer.six")


if __name__ == "__main__":
    main()