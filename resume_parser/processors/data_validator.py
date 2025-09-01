"""
Комплексный валидатор данных резюме (stub implementation)
"""

from typing import Dict, List, Any, Optional
from ..core.base_classes import BaseValidator
from ..core.data_models import ResumeData, PersonalInfo, Experience, Education, Skill
from ..utils.logger import get_logger
from ..utils.helpers import DataValidator


class ComprehensiveDataValidator(BaseValidator):
    """Комплексный валидатор данных резюме"""
    
    def __init__(self):
        self.logger = get_logger(__name__)
        self.data_validator = DataValidator()
        
    def validate(self, data: ResumeData) -> Dict[str, bool]:
        """Базовая валидация данных резюме"""
        validation_results = {
            'personal_info_valid': self._validate_personal_info(data.personal_info),
            'experience_valid': self._validate_experience(data.experience),
            'education_valid': self._validate_education(data.education),
            'skills_valid': self._validate_skills(data.skills),
            'overall_valid': False
        }
        
        # Общая валидность
        validation_results['overall_valid'] = (
            validation_results['personal_info_valid'] and
            len([v for v in validation_results.values() if v]) >= 3
        )
        
        return validation_results
    
    def validate_and_enhance(self, resume_data: ResumeData) -> ResumeData:
        """Валидация и улучшение данных резюме"""
        # Выполняем валидацию
        validation_results = self.validate(resume_data)
        
        # Улучшаем данные
        enhanced_data = self._enhance_resume_data(resume_data)
        
        # Обновляем уверенность на основе валидации
        confidence_score = self._calculate_enhanced_confidence(
            enhanced_data, validation_results
        )
        enhanced_data.parsing_confidence = confidence_score
        
        return enhanced_data
    
    def _validate_personal_info(self, personal_info: PersonalInfo) -> bool:
        """Валидация личной информации"""
        if not personal_info:
            return False
        
        # Проверка имени
        if not personal_info.full_name:
            return False
        
        # Проверка контактной информации
        if personal_info.contact_info:
            if personal_info.contact_info.email:
                valid, _ = self.data_validator.validate_email(personal_info.contact_info.email)
                if not valid:
                    return False
            
            if personal_info.contact_info.phone:
                valid, _ = self.data_validator.validate_phone(personal_info.contact_info.phone)
                if not valid:
                    return False
        
        return True
    
    def _validate_experience(self, experience: List[Experience]) -> bool:
        """Валидация опыта работы"""
        if not experience:
            return True  # Опыт может отсутствовать
        
        for exp in experience:
            if not exp.position:
                return False
            
            # Валидация текста должности
            valid, _ = self.data_validator.validate_text_quality(exp.position)
            if not valid:
                return False
        
        return True
    
    def _validate_education(self, education: List[Education]) -> bool:
        """Валидация образования"""
        if not education:
            return True  # Образование может отсутствовать
        
        for edu in education:
            if not edu.institution:
                return False
            
            # Валидация названия учреждения
            valid, _ = self.data_validator.validate_text_quality(edu.institution)
            if not valid:
                return False
        
        return True
    
    def _validate_skills(self, skills: List[Skill]) -> bool:
        """Валидация навыков"""
        if not skills:
            return False  # Навыки должны присутствовать
        
        # Проверка минимального количества навыков
        if len(skills) < 2:
            return False
        
        # Проверка качества навыков
        valid_skills_count = 0
        for skill in skills:
            if skill.name and len(skill.name.strip()) >= 2:
                valid_skills_count += 1
        
        return valid_skills_count >= 2
    
    def _enhance_resume_data(self, resume_data: ResumeData) -> ResumeData:
        """Улучшение данных резюме"""
        # Создаем копию данных
        enhanced_data = resume_data
        
        # Улучшение личной информации
        if enhanced_data.personal_info:
            enhanced_data.personal_info = self._enhance_personal_info(enhanced_data.personal_info)
        
        # Улучшение навыков
        if enhanced_data.skills:
            enhanced_data.skills = self._enhance_skills(enhanced_data.skills)
        
        # Улучшение опыта
        if enhanced_data.experience:
            enhanced_data.experience = self._enhance_experience(enhanced_data.experience)
        
        return enhanced_data
    
    def _enhance_personal_info(self, personal_info: PersonalInfo) -> PersonalInfo:
        """Улучшение личной информации"""
        # Нормализация имени
        if personal_info.full_name:
            # Простая нормализация: удаление лишних пробелов
            personal_info.full_name = ' '.join(personal_info.full_name.split())
            
            # Разделение на имя и фамилию если возможно
            name_parts = personal_info.full_name.split()
            if len(name_parts) >= 2:
                personal_info.first_name = name_parts[0]
                personal_info.last_name = name_parts[1]
        
        # Нормализация контактной информации
        if personal_info.contact_info:
            if personal_info.contact_info.email:
                personal_info.contact_info.email = personal_info.contact_info.email.lower().strip()
            
            if personal_info.contact_info.phone:
                # Простая нормализация телефона
                import re
                phone = re.sub(r'[^\d+]', '', personal_info.contact_info.phone)
                personal_info.contact_info.phone = phone
        
        return personal_info
    
    def _enhance_skills(self, skills: List[Skill]) -> List[Skill]:
        """Улучшение навыков"""
        enhanced_skills = []
        
        for skill in skills:
            # Нормализация названия навыка
            if skill.name:
                skill.name = skill.name.strip().title()
                
                # Улучшение уверенности на основе длины и качества
                if len(skill.name) >= 3 and skill.name.replace(' ', '').isalpha():
                    skill.confidence_score = min(skill.confidence_score + 0.1, 1.0)
                
                enhanced_skills.append(skill)
        
        return enhanced_skills
    
    def _enhance_experience(self, experience: List[Experience]) -> List[Experience]:
        """Улучшение опыта работы"""
        enhanced_experience = []
        
        for exp in experience:
            # Нормализация должности
            if exp.position:
                exp.position = exp.position.strip()
            
            # Нормализация компании
            if exp.company:
                exp.company = exp.company.strip()
            
            # Определение уровня сениорности по должности
            if exp.position:
                position_lower = exp.position.lower()
                if 'senior' in position_lower or 'главный' in position_lower:
                    from ..core.data_models import SeniorityLevel
                    exp.seniority_level = SeniorityLevel.SENIOR
                elif 'junior' in position_lower or 'младший' in position_lower:
                    exp.seniority_level = SeniorityLevel.JUNIOR
                elif 'middle' in position_lower or 'средний' in position_lower:
                    exp.seniority_level = SeniorityLevel.MIDDLE
                elif 'lead' in position_lower or 'ведущий' in position_lower:
                    exp.seniority_level = SeniorityLevel.LEAD
            
            enhanced_experience.append(exp)
        
        return enhanced_experience
    
    def _calculate_enhanced_confidence(self, resume_data: ResumeData, 
                                     validation_results: Dict[str, bool]) -> float:
        """Расчет улучшенной уверенности"""
        base_confidence = resume_data.parsing_confidence or 0.5
        
        # Бонусы за валидность секций
        validation_bonus = 0.0
        if validation_results['personal_info_valid']:
            validation_bonus += 0.1
        if validation_results['experience_valid']:
            validation_bonus += 0.1
        if validation_results['education_valid']:
            validation_bonus += 0.05
        if validation_results['skills_valid']:
            validation_bonus += 0.1
        
        # Бонусы за полноту данных
        completeness_bonus = 0.0
        if resume_data.personal_info and resume_data.personal_info.contact_info:
            if resume_data.personal_info.contact_info.email:
                completeness_bonus += 0.05
            if resume_data.personal_info.contact_info.phone:
                completeness_bonus += 0.05
        
        if resume_data.experience and len(resume_data.experience) >= 2:
            completeness_bonus += 0.1
        
        if resume_data.skills and len(resume_data.skills) >= 5:
            completeness_bonus += 0.1
        
        # Итоговая уверенность
        final_confidence = min(base_confidence + validation_bonus + completeness_bonus, 1.0)
        
        return final_confidence
    
    def get_validation_score(self) -> float:
        """Получение оценки валидности"""
        return 0.8  # Базовая оценка валидатора