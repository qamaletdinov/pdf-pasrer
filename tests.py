"""
Тесты для парсера резюме
"""

import unittest
from pathlib import Path
import tempfile
import json
from resume_parser import ResumeParser, PersonalInfo, ResumeData


class TestResumeParser(unittest.TestCase):

    def setUp(self):
        self.parser = ResumeParser()
        self.test_text = """
        Иван Иванов
        Email: ivan@example.com
        Телефон: +7 (123) 456-78-90

        Опыт работы:
        Senior Python Developer
        ТехКомпания ООО
        2020 - настоящее время
        """

    def test_extract_personal_info(self):
        """Тест извлечения персональной информации"""
        personal_info = self.parser.analyzer.extract_personal_info(self.test_text)

        self.assertEqual(personal_info.email, "ivan@example.com")
        self.assertIsNotNone(personal_info.phone)

    def test_extract_skills(self):
        """Тест извлечения навыков"""
        text_with_skills = "Python, JavaScript, React, Docker, AWS"
        skills = self.parser.analyzer.extract_skills(text_with_skills)

        self.assertIn("Python", skills)
        self.assertIn("Javascript", skills)

    def test_json_output(self):
        """Тест JSON вывода"""
        personal_info = PersonalInfo(name="Test", email="test@test.com")
        resume_data = ResumeData(personal_info=personal_info)

        # Создание временного файла
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            temp_path = f.name

        try:
            # Тестирование сериализации
            resume_dict = resume_data.__dict__.copy()
            resume_dict['personal_info'] = resume_data.personal_info.__dict__

            json_str = json.dumps(resume_dict, ensure_ascii=False, indent=2)
            self.assertIn("test@test.com", json_str)

        finally:
            Path(temp_path).unlink(missing_ok=True)


if __name__ == "__main__":
    unittest.main()