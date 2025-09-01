"""
Универсальный экстрактор данных резюме (stub implementation)
"""

import re
from typing import Dict, List, Any, Optional
from datetime import datetime

from ..core.base_classes import BaseExtractor
from ..core.data_models import (
    ResumeData, PersonalInfo, ContactInfo, Experience, Education, 
    Skill, JobPreferences, SkillCategory, SeniorityLevel
)
from ..utils.logger import get_logger
from ..utils.helpers import TextProcessor


class UniversalDataExtractor(BaseExtractor):
    """Универсальный экстрактор данных из резюме"""
    
    def __init__(self):
        self.logger = get_logger(__name__)
        self.text_processor = TextProcessor()
        
        # Паттерны для извлечения данных
        self.name_patterns = [
            r'^([А-ЯA-Z][а-яa-z]+\s+[А-ЯA-Z][а-яa-z]+(?:\s+[А-ЯA-Z][а-яa-z]+)?)',
            r'(?:Имя|Name|ФИО):\s*([А-ЯA-Z][а-яa-z\s]+)',
        ]
        
        self.skills_keywords = [
            'python', 'java', 'javascript', 'react', 'vue', 'angular', 'node.js',
            'docker', 'kubernetes', 'aws', 'azure', 'linux', 'sql', 'postgresql',
            'mysql', 'mongodb', 'redis', 'git', 'ci/cd', 'jenkins', 'html', 'css'
        ]
        
    def extract(self, data: Any) -> Dict[str, Any]:
        """Базовое извлечение данных"""
        if isinstance(data, dict):
            text = data.get('text', '')
            segments = data.get('segments', {})
        else:
            text = str(data)
            segments = {}
            
        return self.extract_comprehensive(text, segments)
    
    def extract_comprehensive(self, text: str, segments: Dict[str, Any], 
                            format_info: Any = None) -> ResumeData:
        """Комплексное извлечение данных резюме"""
        # Извлечение основных компонентов
        personal_info = self._extract_personal_info(text, segments)
        job_preferences = self._extract_job_preferences(text, segments)
        experience = self._extract_experience(text, segments)
        education = self._extract_education(text, segments)
        skills = self._extract_skills(text, segments)
        
        # Создание объекта резюме
        resume_data = ResumeData(
            source_text=text,
            personal_info=personal_info,
            job_preferences=job_preferences,
            experience=experience,
            education=education,
            skills=skills,
            parsing_confidence=self._calculate_confidence(
                personal_info, experience, education, skills
            )
        )
        
        return resume_data
    
    def _extract_personal_info(self, text: str, segments: Dict[str, Any]) -> PersonalInfo:
        """Извлечение личной информации"""
        # Извлечение контактной информации
        emails = self.text_processor.extract_emails(text)
        phones = self.text_processor.extract_phones(text)
        
        contact_info = ContactInfo(
            email=emails[0] if emails else None,
            phone=phones[0] if phones else None
        )
        
        # Извлечение имени
        name = self._extract_name(text)
        
        # Извлечение города
        city = self._extract_city(text)
        
        return PersonalInfo(
            full_name=name,
            contact_info=contact_info,
            city=city
        )
    
    def _extract_name(self, text: str) -> Optional[str]:
        """Извлечение имени"""
        lines = text.split('\n')
        
        # Проверяем первые несколько строк
        for line in lines[:5]:
            line = line.strip()
            if not line:
                continue
                
            # Проверяем паттерны имени
            for pattern in self.name_patterns:
                match = re.search(pattern, line)
                if match:
                    return match.group(1).strip()
            
            # Простая эвристика: строка из 2-3 слов, начинающихся с заглавной буквы
            words = line.split()
            if (2 <= len(words) <= 3 and 
                all(word[0].isupper() and word[1:].islower() for word in words if word.isalpha())):
                return line
        
        return None
    
    def _extract_city(self, text: str) -> Optional[str]:
        """Извлечение города"""
        # Простые паттерны городов
        city_patterns = [
            r'(?:г\.|город)\s*([А-ЯA-Z][а-яa-z]+)',
            r'(?:Москва|Санкт-Петербург|Екатеринбург|Новосибирск|Казань|Челябинск)',
            r'(?:Moscow|Saint Petersburg|London|New York|Berlin|Paris)'
        ]
        
        for pattern in city_patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                return match.group(1) if match.groups() else match.group(0)
        
        return None
    
    def _extract_job_preferences(self, text: str, segments: Dict[str, Any]) -> Optional[JobPreferences]:
        """Извлечение предпочтений по работе"""
        # Поиск желаемой должности
        position_patterns = [
            r'(?:Желаемая должность|Desired position):\s*(.+?)(?:\n|$)',
            r'(?:Должность|Position):\s*(.+?)(?:\n|$)'
        ]
        
        desired_position = None
        for pattern in position_patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                desired_position = match.group(1).strip()
                break
        
        if not desired_position:
            # Попытка определить по контексту
            if 'разработчик' in text.lower() or 'developer' in text.lower():
                desired_position = 'Разработчик'
        
        return JobPreferences(desired_position=desired_position) if desired_position else None
    
    def _extract_experience(self, text: str, segments: Dict[str, Any]) -> List[Experience]:
        """Извлечение опыта работы"""
        experience_list = []
        
        # Поиск секции опыта
        experience_text = ""
        if 'experience' in segments:
            experience_text = '\n'.join(segments['experience'])
        else:
            # Поиск в общем тексте
            lines = text.split('\n')
            in_experience = False
            
            for line in lines:
                if re.search(r'(?i)(опыт работы|work experience|experience)', line):
                    in_experience = True
                    continue
                
                if in_experience:
                    if re.search(r'^[А-ЯA-Z][а-яa-z\s]+:$', line.strip()):
                        break
                    experience_text += line + '\n'
        
        # Извлечение отдельных позиций
        if experience_text:
            positions = self._parse_experience_positions(experience_text)
            experience_list.extend(positions)
        
        return experience_list
    
    def _parse_experience_positions(self, text: str) -> List[Experience]:
        """Парсинг позиций в опыте работы"""
        positions = []
        lines = text.split('\n')
        
        current_position = None
        current_company = None
        current_description = []
        
        for line in lines:
            line = line.strip()
            if not line:
                continue
            
            # Поиск должности и компании
            if self._is_position_line(line):
                # Сохраняем предыдущую позицию
                if current_position:
                    positions.append(Experience(
                        position=current_position,
                        company=current_company,
                        description='\n'.join(current_description) if current_description else None
                    ))
                
                # Парсим новую позицию
                position_info = self._parse_position_line(line)
                current_position = position_info.get('position')
                current_company = position_info.get('company')
                current_description = []
            else:
                # Добавляем к описанию
                current_description.append(line)
        
        # Добавляем последнюю позицию
        if current_position:
            positions.append(Experience(
                position=current_position,
                company=current_company,
                description='\n'.join(current_description) if current_description else None
            ))
        
        return positions
    
    def _is_position_line(self, line: str) -> bool:
        """Определение строки с должностью"""
        # Простые эвристики
        if len(line) > 100:  # Слишком длинная строка
            return False
        
        # Содержит ключевые слова должностей
        position_keywords = [
            'разработчик', 'developer', 'инженер', 'engineer', 'менеджер', 'manager',
            'аналитик', 'analyst', 'дизайнер', 'designer', 'архитектор', 'architect'
        ]
        
        return any(keyword in line.lower() for keyword in position_keywords)
    
    def _parse_position_line(self, line: str) -> Dict[str, str]:
        """Парсинг строки с должностью и компанией"""
        # Паттерны для должности и компании
        patterns = [
            r'(.+?)\s+(?:в|at|—)\s+(.+)',  # Должность в/at Компания
            r'(.+?),\s*(.+)',              # Должность, Компания
        ]
        
        for pattern in patterns:
            match = re.match(pattern, line.strip())
            if match:
                return {
                    'position': match.group(1).strip(),
                    'company': match.group(2).strip()
                }
        
        # Если паттерн не найден, считаем всю строку должностью
        return {'position': line.strip(), 'company': None}
    
    def _extract_education(self, text: str, segments: Dict[str, Any]) -> List[Education]:
        """Извлечение образования"""
        education_list = []
        
        # Поиск секции образования
        education_text = ""
        if 'education' in segments:
            education_text = '\n'.join(segments['education'])
        else:
            # Поиск в общем тексте
            lines = text.split('\n')
            in_education = False
            
            for line in lines:
                if re.search(r'(?i)(образование|education)', line):
                    in_education = True
                    continue
                
                if in_education:
                    if re.search(r'^[А-ЯA-Z][а-яa-z\s]+:$', line.strip()):
                        break
                    education_text += line + '\n'
        
        # Извлечение учебных заведений
        if education_text:
            institutions = self._parse_education_institutions(education_text)
            education_list.extend(institutions)
        
        return education_list
    
    def _parse_education_institutions(self, text: str) -> List[Education]:
        """Парсинг учебных заведений"""
        institutions = []
        lines = text.split('\n')
        
        for line in lines:
            line = line.strip()
            if not line:
                continue
            
            # Поиск университетов и институтов
            if self._is_education_line(line):
                education_info = self._parse_education_line(line)
                institutions.append(Education(**education_info))
        
        return institutions
    
    def _is_education_line(self, line: str) -> bool:
        """Определение строки с образованием"""
        education_keywords = [
            'университет', 'university', 'институт', 'institute', 'колледж', 'college',
            'академия', 'academy', 'школа', 'school'
        ]
        
        return any(keyword in line.lower() for keyword in education_keywords)
    
    def _parse_education_line(self, line: str) -> Dict[str, Optional[str]]:
        """Парсинг строки с образованием"""
        # Простое извлечение названия учреждения
        return {
            'institution': line.strip(),
            'degree': None,
            'field_of_study': None,
            'graduation_date': None
        }
    
    def _extract_skills(self, text: str, segments: Dict[str, Any]) -> List[Skill]:
        """Извлечение навыков"""
        skills_list = []
        
        # Поиск секции навыков
        skills_text = ""
        if 'skills' in segments:
            skills_text = '\n'.join(segments['skills'])
        else:
            # Поиск в общем тексте
            lines = text.split('\n')
            in_skills = False
            
            for line in lines:
                if re.search(r'(?i)(навыки|skills|технологии|technologies)', line):
                    in_skills = True
                    continue
                
                if in_skills:
                    if re.search(r'^[А-ЯA-Z][а-яa-z\s]+:$', line.strip()):
                        break
                    skills_text += line + '\n'
        
        # Извлечение отдельных навыков
        if skills_text:
            skills = self._parse_skills(skills_text)
            skills_list.extend(skills)
        
        # Дополнительный поиск по ключевым словам в полном тексте
        additional_skills = self._extract_skills_by_keywords(text)
        skills_list.extend(additional_skills)
        
        # Удаление дубликатов
        unique_skills = []
        seen_names = set()
        
        for skill in skills_list:
            if skill.name.lower() not in seen_names:
                unique_skills.append(skill)
                seen_names.add(skill.name.lower())
        
        return unique_skills
    
    def _parse_skills(self, text: str) -> List[Skill]:
        """Парсинг навыков из текста"""
        skills = []
        
        # Разделение по запятым, точкам с запятой и переносам строк
        skill_text = re.sub(r'[,;]\s*', '\n', text)
        skill_lines = skill_text.split('\n')
        
        for line in skill_lines:
            line = line.strip()
            if line and len(line) < 50:  # Разумная длина для навыка
                skills.append(Skill(
                    name=line,
                    category=self._categorize_skill(line),
                    confidence_score=0.8
                ))
        
        return skills
    
    def _extract_skills_by_keywords(self, text: str) -> List[Skill]:
        """Извлечение навыков по ключевым словам"""
        skills = []
        text_lower = text.lower()
        
        for keyword in self.skills_keywords:
            if keyword.lower() in text_lower:
                skills.append(Skill(
                    name=keyword,
                    category=self._categorize_skill(keyword),
                    confidence_score=0.9
                ))
        
        return skills
    
    def _categorize_skill(self, skill_name: str) -> SkillCategory:
        """Категоризация навыка"""
        skill_lower = skill_name.lower()
        
        # Языки программирования
        if any(lang in skill_lower for lang in ['python', 'java', 'javascript', 'c++', 'c#', 'ruby', 'php']):
            return SkillCategory.TECHNICAL
        
        # Фреймворки
        if any(fw in skill_lower for fw in ['react', 'vue', 'angular', 'django', 'flask', 'spring']):
            return SkillCategory.FRAMEWORK
        
        # Инструменты
        if any(tool in skill_lower for tool in ['docker', 'git', 'jenkins', 'kubernetes']):
            return SkillCategory.TOOL
        
        # По умолчанию - технический навык
        return SkillCategory.TECHNICAL
    
    def _calculate_confidence(self, personal_info: PersonalInfo, 
                            experience: List[Experience], 
                            education: List[Education],
                            skills: List[Skill]) -> float:
        """Расчет общей уверенности парсинга"""
        confidence_factors = []
        
        # Личная информация
        if personal_info.full_name:
            confidence_factors.append(0.9)
        if personal_info.contact_info and personal_info.contact_info.email:
            confidence_factors.append(0.8)
        if personal_info.contact_info and personal_info.contact_info.phone:
            confidence_factors.append(0.8)
        
        # Опыт работы
        if experience:
            confidence_factors.append(0.85)
            if any(exp.company for exp in experience):
                confidence_factors.append(0.8)
        
        # Образование
        if education:
            confidence_factors.append(0.7)
        
        # Навыки
        if skills:
            confidence_factors.append(0.75)
            if len(skills) >= 5:
                confidence_factors.append(0.8)
        
        # Общая уверенность
        if confidence_factors:
            return sum(confidence_factors) / len(confidence_factors)
        else:
            return 0.5  # Базовая уверенность
    
    def get_confidence_score(self) -> float:
        """Получение оценки уверенности экстрактора"""
        return 0.85  # Базовая уверенность экстрактора