"""
Метрики качества и производительности парсера
"""

import time
from typing import Dict, List, Optional, Any

# Optional dependency
try:
    import psutil
except ImportError:
    psutil = None
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from collections import defaultdict
import statistics

from ..utils.logger import get_logger


@dataclass
class ParsingMetric:
    """Отдельная метрика парсинга"""
    name: str
    value: float
    unit: str
    timestamp: datetime = field(default_factory=datetime.now)
    context: Dict[str, Any] = field(default_factory=dict)


@dataclass
class PerformanceSnapshot:
    """Снимок производительности системы"""
    timestamp: datetime
    cpu_percent: float
    memory_mb: float
    memory_percent: float
    processing_time: float


class ParsingMetrics:
    """Система метрик для парсера резюме"""
    
    def __init__(self):
        self.logger = get_logger(__name__)
        self.metrics: List[ParsingMetric] = []
        self.performance_snapshots: List[PerformanceSnapshot] = []
        self.session_start = datetime.now()
        
        # Счетчики по категориям
        self.counters = defaultdict(int)
        self.timers = defaultdict(list)
        self.scores = defaultdict(list)
        
        # Состояние текущей сессии
        self.current_parsing_start = None
        self.current_process = None
    
    def start_parsing_session(self, source: str, format_type: str = None):
        """Начало сессии парсинга"""
        self.current_parsing_start = time.time()
        
        if psutil:
            self.current_process = psutil.Process()
        else:
            self.current_process = None
        
        # Записываем начальный снимок производительности
        self._take_performance_snapshot()
        
        # Метрика начала сессии
        self.add_metric("parsing_session_started", 1.0, "count", {
            'source': source,
            'format_type': format_type
        })
        
        self.logger.debug(f"Начата сессия парсинга: {source}")
    
    def end_parsing_session(self, success: bool, confidence: float = None):
        """Завершение сессии парсинга"""
        if self.current_parsing_start is None:
            self.logger.warning("Попытка завершить сессию без начала")
            return
        
        processing_time = time.time() - self.current_parsing_start
        
        # Финальный снимок производительности
        self._take_performance_snapshot()
        
        # Метрики завершения
        self.add_metric("parsing_session_completed", 1.0, "count", {
            'success': success,
            'processing_time': processing_time,
            'confidence': confidence
        })
        
        self.add_metric("parsing_time", processing_time, "seconds")
        
        if success:
            self.counters['successful_parsings'] += 1
            if confidence is not None:
                self.scores['parsing_confidence'].append(confidence)
        else:
            self.counters['failed_parsings'] += 1
        
        self.timers['parsing_times'].append(processing_time)
        
        self.current_parsing_start = None
        self.current_process = None
        
        self.logger.debug(f"Сессия парсинга завершена: {processing_time:.2f}с, успех: {success}")
    
    def add_metric(self, name: str, value: float, unit: str, context: Dict[str, Any] = None):
        """Добавление метрики"""
        metric = ParsingMetric(
            name=name,
            value=value,
            unit=unit,
            context=context or {}
        )
        self.metrics.append(metric)
        
        # Обновление счетчиков и таймеров
        if unit == "count":
            self.counters[name] += value
        elif unit in ["seconds", "ms"]:
            self.timers[name].append(value)
        elif unit in ["score", "percent", "ratio"]:
            self.scores[name].append(value)
    
    def _take_performance_snapshot(self):
        """Создание снимка производительности"""
        if not psutil or self.current_process is None:
            return
        
        try:
            cpu_percent = self.current_process.cpu_percent()
            memory_info = self.current_process.memory_info()
            memory_mb = memory_info.rss / 1024 / 1024  # В мегабайтах
            memory_percent = self.current_process.memory_percent()
            
            processing_time = (time.time() - self.current_parsing_start 
                             if self.current_parsing_start else 0.0)
            
            snapshot = PerformanceSnapshot(
                timestamp=datetime.now(),
                cpu_percent=cpu_percent,
                memory_mb=memory_mb,
                memory_percent=memory_percent,
                processing_time=processing_time
            )
            
            self.performance_snapshots.append(snapshot)
            
        except Exception as e:
            self.logger.warning(f"Ошибка создания снимка производительности: {e}")
    
    def get_summary_statistics(self) -> Dict[str, Any]:
        """Получение сводной статистики"""
        total_parsings = self.counters['successful_parsings'] + self.counters['failed_parsings']
        
        summary = {
            'session_info': {
                'start_time': self.session_start.isoformat(),
                'duration_hours': (datetime.now() - self.session_start).total_seconds() / 3600,
                'total_metrics': len(self.metrics)
            },
            'parsing_stats': {
                'total_parsings': total_parsings,
                'successful_parsings': self.counters['successful_parsings'],
                'failed_parsings': self.counters['failed_parsings'],
                'success_rate': (self.counters['successful_parsings'] / total_parsings 
                               if total_parsings > 0 else 0.0)
            },
            'performance_stats': {},
            'quality_stats': {}
        }
        
        # Статистика времени
        if self.timers['parsing_times']:
            times = self.timers['parsing_times']
            summary['performance_stats']['parsing_times'] = {
                'count': len(times),
                'mean': statistics.mean(times),
                'median': statistics.median(times),
                'min': min(times),
                'max': max(times),
                'std_dev': statistics.stdev(times) if len(times) > 1 else 0.0
            }
        
        # Статистика уверенности
        if self.scores['parsing_confidence']:
            confidences = self.scores['parsing_confidence']
            summary['quality_stats']['confidence'] = {
                'count': len(confidences),
                'mean': statistics.mean(confidences),
                'median': statistics.median(confidences),
                'min': min(confidences),
                'max': max(confidences)
            }
        
        # Статистика производительности системы
        if self.performance_snapshots:
            cpu_values = [s.cpu_percent for s in self.performance_snapshots]
            memory_values = [s.memory_mb for s in self.performance_snapshots]
            
            summary['performance_stats']['system'] = {
                'cpu_usage': {
                    'mean': statistics.mean(cpu_values),
                    'max': max(cpu_values)
                },
                'memory_usage': {
                    'mean': statistics.mean(memory_values),
                    'max': max(memory_values)
                }
            }
        
        return summary
    
    def get_recent_metrics(self, hours: int = 1) -> List[ParsingMetric]:
        """Получение метрик за последние часы"""
        cutoff_time = datetime.now() - timedelta(hours=hours)
        return [m for m in self.metrics if m.timestamp >= cutoff_time]
    
    def get_metrics_by_name(self, name: str) -> List[ParsingMetric]:
        """Получение метрик по имени"""
        return [m for m in self.metrics if m.name == name]
    
    def clear_metrics(self):
        """Очистка всех метрик"""
        self.metrics.clear()
        self.performance_snapshots.clear()
        self.counters.clear()
        self.timers.clear()
        self.scores.clear()
        self.session_start = datetime.now()
        
        self.logger.info("Метрики очищены")
    
    def export_metrics(self) -> Dict[str, Any]:
        """Экспорт всех метрик в словарь"""
        return {
            'session_start': self.session_start.isoformat(),
            'metrics': [
                {
                    'name': m.name,
                    'value': m.value,
                    'unit': m.unit,
                    'timestamp': m.timestamp.isoformat(),
                    'context': m.context
                }
                for m in self.metrics
            ],
            'performance_snapshots': [
                {
                    'timestamp': s.timestamp.isoformat(),
                    'cpu_percent': s.cpu_percent,
                    'memory_mb': s.memory_mb,
                    'memory_percent': s.memory_percent,
                    'processing_time': s.processing_time
                }
                for s in self.performance_snapshots
            ],
            'summary': self.get_summary_statistics()
        }


class PerformanceTracker:
    """Трекер производительности для отдельных операций"""
    
    def __init__(self, operation_name: str, metrics: ParsingMetrics):
        self.operation_name = operation_name
        self.metrics = metrics
        self.logger = get_logger(__name__)
        self.start_time = None
        self.start_memory = None
    
    def __enter__(self):
        """Начало отслеживания"""
        self.start_time = time.time()
        
        if psutil:
            try:
                process = psutil.Process()
                self.start_memory = process.memory_info().rss / 1024 / 1024  # MB
            except Exception:
                self.start_memory = None
        else:
            self.start_memory = None
        
        self.logger.debug(f"Начало отслеживания: {self.operation_name}")
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Завершение отслеживания"""
        if self.start_time is None:
            return
        
        processing_time = time.time() - self.start_time
        success = exc_type is None
        
        # Добавление метрик времени
        self.metrics.add_metric(
            f"{self.operation_name}_time",
            processing_time,
            "seconds",
            {'success': success}
        )
        
        # Метрика памяти (если доступна)
        if psutil and self.start_memory is not None:
            try:
                process = psutil.Process()
                end_memory = process.memory_info().rss / 1024 / 1024  # MB
                memory_delta = end_memory - self.start_memory
                
                self.metrics.add_metric(
                    f"{self.operation_name}_memory_delta",
                    memory_delta,
                    "MB",
                    {'success': success}
                )
            except Exception:
                pass
        
        # Счетчик успешности
        if success:
            self.metrics.add_metric(f"{self.operation_name}_success", 1.0, "count")
        else:
            self.metrics.add_metric(f"{self.operation_name}_failure", 1.0, "count")
        
        status = "успешно" if success else "с ошибкой"
        self.logger.debug(f"Операция {self.operation_name} завершена {status}: {processing_time:.3f}с")
    
    def add_checkpoint(self, checkpoint_name: str, value: float = None):
        """Добавление промежуточной точки"""
        if self.start_time is None:
            return
        
        elapsed_time = time.time() - self.start_time
        
        self.metrics.add_metric(
            f"{self.operation_name}_{checkpoint_name}_time",
            elapsed_time,
            "seconds"
        )
        
        if value is not None:
            self.metrics.add_metric(
                f"{self.operation_name}_{checkpoint_name}_value",
                value,
                "value"
            )
        
        self.logger.debug(f"Чекпоинт {checkpoint_name}: {elapsed_time:.3f}с")


# Глобальный экземпляр метрик
global_metrics = ParsingMetrics()


def get_metrics() -> ParsingMetrics:
    """Получение глобального экземпляра метрик"""
    return global_metrics


def track_performance(operation_name: str, metrics: ParsingMetrics = None):
    """Декоратор для отслеживания производительности функций"""
    if metrics is None:
        metrics = global_metrics
    
    def decorator(func):
        def wrapper(*args, **kwargs):
            with PerformanceTracker(f"func_{operation_name}", metrics):
                return func(*args, **kwargs)
        return wrapper
    return decorator