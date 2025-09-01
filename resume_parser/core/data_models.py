"""
Модели данных для парсера резюме
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Union
from datetime import datetime
from enum import Enum


class EmploymentType(Enum):
    """Тип занятости"""
    FULL_TIME = "full_time"
    PART_TIME = "part_time"
    CONTRACT = "contract"
    FREELANCE = "freelance"
    INTERNSHIP = "internship"
    TEMPORARY = "temporary"


class WorkSchedule(Enum):
    """График работы"""
    FULL_DAY = "full_day"
    SHIFT = "shift"
    FLEXIBLE = "flexible"
    REMOTE = "remote"
    HYBRID = "hybrid"


class SkillCategory(Enum):
    """Категории навыков"""
    TECHNICAL = "technical"
    SOFT = "soft"
    LANGUAGE = "language"
    TOOL = "tool"
    FRAMEWORK = "framework"
    DOMAIN = "domain"


class SeniorityLevel(Enum):
    """Уровень сениорности"""
    INTERN = "intern"
    JUNIOR = "junior"
    MIDDLE = "middle"
    SENIOR = "senior"
    LEAD = "lead"
    PRINCIPAL = "principal"


@dataclass
class ConfidenceMetrics:
    """Метрики уверенности для каждого поля"""
    overall_score: float
    field_scores: Dict[str, float] = field(default_factory=dict)
    validation_results: Dict[str, bool] = field(default_factory=dict)
    extraction_method: str = ""
    reliability_factors: Dict[str, float] = field(default_factory=dict)


@dataclass
class ContactInfo:
    """Контактная информация"""
    phone: Optional[str] = None
    email: Optional[str] = None
    telegram: Optional[str] = None
    linkedin: Optional[str] = None
    github: Optional[str] = None
    website: Optional[str] = None
    social_networks: List[str] = field(default_factory=list)
    confidence: Optional[ConfidenceMetrics] = None


@dataclass
class PersonalInfo:
    """Личная информация"""
    full_name: Optional[str] = None
    first_name: Optional[str] = None
    last_name: Optional[str] = None
    gender: Optional[str] = None
    age: Optional[int] = None
    birth_date: Optional[str] = None
    city: Optional[str] = None
    citizenship: Optional[str] = None
    work_permit: Optional[str] = None
    ready_to_relocate: Optional[bool] = None
    ready_for_business_trips: Optional[bool] = None
    contact_info: Optional[ContactInfo] = None
    confidence: Optional[ConfidenceMetrics] = None


@dataclass
class JobPreferences:
    """Предпочтения по работе"""
    desired_position: Optional[str] = None
    specializations: List[str] = field(default_factory=list)
    employment_type: Optional[EmploymentType] = None
    work_schedule: Optional[WorkSchedule] = None
    desired_salary: Optional[str] = None
    travel_time_preference: Optional[str] = None
    preferred_locations: List[str] = field(default_factory=list)
    confidence: Optional[ConfidenceMetrics] = None


@dataclass
class Skill:
    """Навык с детальным анализом"""
    name: str
    category: Optional[SkillCategory] = None
    proficiency_level: Optional[str] = None
    years_of_experience: Optional[int] = None
    last_used: Optional[str] = None
    confidence_score: float = 0.0
    market_demand: Optional[float] = None
    is_verified: bool = False
    context: List[str] = field(default_factory=list)


@dataclass
class Project:
    """Проект с детальным анализом"""
    name: str
    description: Optional[str] = None
    role: Optional[str] = None
    technologies: List[str] = field(default_factory=list)
    start_date: Optional[str] = None
    end_date: Optional[str] = None
    duration_months: Optional[int] = None
    team_size: Optional[int] = None
    achievements: List[str] = field(default_factory=list)
    links: List[str] = field(default_factory=list)
    confidence_score: float = 0.0


@dataclass
class Experience:
    """Опыт работы с ML анализом"""
    position: str
    company: Optional[str] = None
    start_date: Optional[str] = None
    end_date: Optional[str] = None
    duration_months: Optional[int] = None
    location: Optional[str] = None
    employment_type: Optional[EmploymentType] = None
    description: Optional[str] = None
    responsibilities: List[str] = field(default_factory=list)
    achievements: List[str] = field(default_factory=list)
    technologies: List[str] = field(default_factory=list)
    projects: List[Project] = field(default_factory=list)
    seniority_level: Optional[SeniorityLevel] = None
    industry: Optional[str] = None
    company_size: Optional[str] = None
    is_current: bool = False
    confidence_score: float = 0.0


@dataclass
class Education:
    """Образование"""
    degree: Optional[str] = None
    field_of_study: Optional[str] = None
    institution: Optional[str] = None
    start_date: Optional[str] = None
    graduation_date: Optional[str] = None
    gpa: Optional[str] = None
    location: Optional[str] = None
    thesis_topic: Optional[str] = None
    honors: List[str] = field(default_factory=list)
    relevant_courses: List[str] = field(default_factory=list)
    confidence_score: float = 0.0


@dataclass
class Language:
    """Знание языков"""
    name: str
    proficiency_level: Optional[str] = None
    certification: Optional[str] = None
    is_native: bool = False
    confidence_score: float = 0.0


@dataclass
class Certification:
    """Сертификация"""
    name: str
    issuer: Optional[str] = None
    issue_date: Optional[str] = None
    expiry_date: Optional[str] = None
    credential_id: Optional[str] = None
    url: Optional[str] = None
    confidence_score: float = 0.0


@dataclass
class ExtractedSection:
    """Извлеченная секция резюме"""
    name: str
    content: str
    start_position: int
    end_position: int
    confidence_score: float = 0.0
    structure_type: Optional[str] = None


@dataclass
class MLMetrics:
    """ML метрики анализа резюме"""
    seniority_assessment: Optional[SeniorityLevel] = None
    skill_diversity_score: float = 0.0
    career_progression_score: float = 0.0
    overall_quality_score: float = 0.0
    market_fit_score: float = 0.0
    total_experience_months: int = 0
    technology_stack_size: int = 0
    project_complexity_score: float = 0.0


@dataclass
class ParsingMetadata:
    """Метаданные парсинга"""
    parsing_timestamp: datetime = field(default_factory=datetime.now)
    parsing_duration_ms: float = 0.0
    source_format: Optional[str] = None
    detected_format: Optional[str] = None
    extraction_method: Optional[str] = None
    text_length: int = 0
    sections_detected: List[str] = field(default_factory=list)
    ml_models_used: List[str] = field(default_factory=list)
    confidence_scores_by_section: Dict[str, float] = field(default_factory=dict)
    processing_errors: List[str] = field(default_factory=list)
    processing_warnings: List[str] = field(default_factory=list)


@dataclass
class ResumeData:
    """Полная структура резюме с ML анализом"""
    # Основная информация
    source_text: str
    personal_info: PersonalInfo
    job_preferences: Optional[JobPreferences] = None
    
    # Детальные секции
    experience: List[Experience] = field(default_factory=list)
    education: List[Education] = field(default_factory=list)
    skills: List[Skill] = field(default_factory=list)
    languages: List[Language] = field(default_factory=list)
    certifications: List[Certification] = field(default_factory=list)
    
    # Дополнительная информация
    summary: Optional[str] = None
    additional_info: Optional[str] = None
    hobbies: List[str] = field(default_factory=list)
    references: List[str] = field(default_factory=list)
    
    # Извлеченные секции
    extracted_sections: List[ExtractedSection] = field(default_factory=list)
    
    # ML анализ
    ml_metrics: Optional[MLMetrics] = None
    parsing_confidence: float = 0.0
    
    # Метаданные
    metadata: Optional[ParsingMetadata] = None
    last_updated: Optional[str] = None
    
    def get_total_experience_years(self) -> float:
        """Получение общего опыта в годах"""
        if self.ml_metrics:
            return self.ml_metrics.total_experience_months / 12.0
        return 0.0
    
    def get_primary_skills(self, limit: int = 10) -> List[Skill]:
        """Получение основных навыков"""
        return sorted(self.skills, 
                     key=lambda x: x.confidence_score, 
                     reverse=True)[:limit]
    
    def get_current_position(self) -> Optional[Experience]:
        """Получение текущей позиции"""
        current_jobs = [exp for exp in self.experience if exp.is_current]
        return current_jobs[0] if current_jobs else None