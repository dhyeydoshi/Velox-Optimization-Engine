import time
import cProfile
import pstats
import psutil
import io
import functools
import asyncio
from typing import Dict, List, Optional, Any, Callable
from dataclasses import dataclass, asdict
from datetime import datetime
import json
import threading
from contextlib import contextmanager
import tracemalloc

_tracemalloc_lock = threading.Lock()

from code.code_optimizer_ai.config.settings import settings
from code.code_optimizer_ai.utils.logger import get_logger

logger = get_logger(__name__)


@dataclass
class PerformanceMetrics:
    execution_time: float
    cpu_usage: float
    memory_usage: float
    peak_memory: float
    function_calls: int
    unique_functions: int
    cpu_time: float
    wall_time: float
    timestamp: datetime
    context: str


@dataclass
class BaselineMetrics:
    metric_name: str
    baseline_value: float
    standard_deviation: float
    last_updated: datetime
    sample_count: int
    percentile_95: float
    percentile_99: float


@dataclass
class OptimizationResult:
    optimization_id: str
    original_metrics: PerformanceMetrics
    optimized_metrics: PerformanceMetrics
    improvement_percentage: float
    success: bool
    applied_at: datetime
    context: str


class PerformanceProfiler:
    
    def __init__(self):
        self.baseline_cache: Dict[str, BaselineMetrics] = {}
        self.monitoring_active = False
        self.monitoring_thread = None
        self.metrics_buffer: List[PerformanceMetrics] = []
        self.lock = threading.Lock()
        
    @contextmanager
    def profile_function(
        self,
        context: str = "",
        track_memory: bool = True,
        track_cpu: bool = True
    ):
        
        profiler = cProfile.Profile()
        
        if track_memory:
            _tracemalloc_lock.acquire()
            tracemalloc.start()
        
        start_time = time.perf_counter()
        proc = psutil.Process()
        start_cpu_times = proc.cpu_times()
        start_memory = proc.memory_info().rss
        
        try:
            profiler.enable()
            yield profiler
            
        finally:
            profiler.disable()
            
            end_time = time.perf_counter()
            end_cpu_times = proc.cpu_times()
            end_memory = proc.memory_info().rss
            cpu_delta = (
                (end_cpu_times.user - start_cpu_times.user)
                + (end_cpu_times.system - start_cpu_times.system)
            )
            
            if track_memory:
                current, peak = tracemalloc.get_traced_memory()
                tracemalloc.stop()
                _tracemalloc_lock.release()
            else:
                current = peak = 0
            
            stats_buffer = io.StringIO()
            stats = pstats.Stats(profiler, stream=stats_buffer)
            stats.sort_stats('cumulative')
            
            total_calls = stats.total_calls
            unique_functions = len(stats.stats)
            
            cpu_time = 0
            wall_time = 0
            for (filename, lineno, function), (ccalls, ncalls, tt, ct, callers) in stats.stats.items():
                cpu_time += ct
                wall_time += tt
            
            metrics = PerformanceMetrics(
                execution_time=end_time - start_time,
                cpu_usage=cpu_delta,
                memory_usage=end_memory - start_memory,
                peak_memory=peak,
                function_calls=total_calls,
                unique_functions=unique_functions,
                cpu_time=cpu_time,
                wall_time=wall_time,
                timestamp=datetime.now(),
                context=context or "unknown"
            )
            
            # Store metrics (cap buffer to prevent unbounded growth)
            with self.lock:
                self.metrics_buffer.append(metrics)
                if len(self.metrics_buffer) > 10_000:
                    self.metrics_buffer = self.metrics_buffer[-10_000:]
            
            logger.debug(f"Performance metrics recorded: {context} - {metrics.execution_time:.3f}s")
    
    def profile_async_function(
        self,
        context: str = "",
        track_memory: bool = True
    ):
        
        def decorator(func: Callable):
            @functools.wraps(func)
            async def wrapper(*args, **kwargs):
                with self.profile_function(context or f"{func.__name__}", track_memory):
                    return await func(*args, **kwargs)
            return wrapper
        return decorator
    
    def start_monitoring(
        self,
        interval: int = 60,
        context: str = "system_monitoring"
    ):
        
        if self.monitoring_active:
            logger.warning("Performance monitoring already active")
            return
        
        self.monitoring_active = True
        self.monitoring_thread = threading.Thread(
            target=self._monitoring_loop,
            args=(interval, context),
            daemon=True
        )
        self.monitoring_thread.start()
        logger.info(f"Performance monitoring started with {interval}s interval")
    
    def stop_monitoring(self):
        
        self.monitoring_active = False
        if self.monitoring_thread:
            self.monitoring_thread.join(timeout=5)
            self.monitoring_thread = None
        
        logger.info("Performance monitoring stopped")
    
    def _monitoring_loop(self, interval: int, context: str):
        
        while self.monitoring_active:
            try:
                with self.profile_function(context, track_memory=True):
                    # Collect system metrics
                    cpu_percent = psutil.cpu_percent(interval=1)
                    memory = psutil.virtual_memory()
                    disk = psutil.disk_usage('/')
                    
                    # Log system metrics
                    logger.info(
                        f"System metrics - CPU: {cpu_percent}%, "
                        f"Memory: {memory.percent}%, "
                        f"Disk: {disk.percent}%"
                    )
                    
                time.sleep(interval)
                
            except Exception as e:
                logger.error(f"Monitoring loop error: {e}")
                time.sleep(interval)
    
    def create_baseline(
        self,
        metric_name: str,
        metrics: List[PerformanceMetrics],
        context_filter: str = ""
    ) -> BaselineMetrics:
        
        if context_filter:
            metrics = [m for m in metrics if context_filter in m.context]
        
        if not metrics:
            raise ValueError("No metrics provided for baseline creation")
        
        values = []
        for metric in metrics:
            if metric_name == "execution_time":
                values.append(metric.execution_time)
            elif metric_name == "memory_usage":
                values.append(metric.memory_usage)
            elif metric_name == "cpu_usage":
                values.append(metric.cpu_usage)
            elif metric_name == "peak_memory":
                values.append(metric.peak_memory)
            else:
                raise ValueError(f"Unknown metric name: {metric_name}")
        
        import statistics
        
        mean_value = statistics.mean(values)
        stdev = statistics.stdev(values) if len(values) > 1 else 0
        
        sorted_values = sorted(values)
        p95_index = int(0.95 * len(sorted_values))
        p99_index = int(0.99 * len(sorted_values))
        
        p95 = sorted_values[p95_index] if p95_index < len(sorted_values) else mean_value
        p99 = sorted_values[p99_index] if p99_index < len(sorted_values) else mean_value
        
        baseline = BaselineMetrics(
            metric_name=metric_name,
            baseline_value=mean_value,
            standard_deviation=stdev,
            last_updated=datetime.now(),
            sample_count=len(values),
            percentile_95=p95,
            percentile_99=p99
        )
        
        self.baseline_cache[f"{metric_name}_{context_filter}"] = baseline
        
        logger.info(f"Created baseline for {metric_name}: {mean_value:.3f} ± {stdev:.3f}")
        return baseline
    
    def compare_to_baseline(
        self,
        metric: PerformanceMetrics,
        metric_name: str,
        context_filter: str = ""
    ) -> Dict[str, Any]:
        
        baseline_key = f"{metric_name}_{context_filter}"
        baseline = self.baseline_cache.get(baseline_key)
        
        if not baseline:
            return {"error": f"No baseline found for {baseline_key}"}
        
        # Get current value
        if metric_name == "execution_time":
            current_value = metric.execution_time
        elif metric_name == "memory_usage":
            current_value = metric.memory_usage
        elif metric_name == "cpu_usage":
            current_value = metric.cpu_usage
        elif metric_name == "peak_memory":
            current_value = metric.peak_memory
        else:
            return {"error": f"Unknown metric name: {metric_name}"}
        
        # Calculate deviation
        deviation = current_value - baseline.baseline_value
        if baseline.baseline_value != 0:
            deviation_percent = (deviation / baseline.baseline_value) * 100
        else:
            deviation_percent = 0.0 if deviation == 0 else float("inf")
        
        # Determine performance status
        if deviation > baseline.standard_deviation * 2:
            status = "degraded"
        elif deviation < -baseline.standard_deviation * 2:
            status = "improved"
        else:
            status = "normal"
        
        return {
            "current_value": current_value,
            "baseline_value": baseline.baseline_value,
            "deviation": deviation,
            "deviation_percent": deviation_percent,
            "status": status,
            "baseline_sample_count": baseline.sample_count
        }
    
    def analyze_optimization_impact(
        self,
        original: PerformanceMetrics,
        optimized: PerformanceMetrics,
        context: str = ""
    ) -> OptimizationResult:
        
        # Calculate improvement in key metrics
        time_improvement = (
            (original.execution_time - optimized.execution_time) /
            original.execution_time * 100
        ) if original.execution_time > 0 else 0.0
        
        memory_improvement = (
            (original.peak_memory - optimized.peak_memory) / 
            original.peak_memory * 100
        ) if original.peak_memory > 0 else 0.0
        
        # Overall improvement score (weighted average)
        overall_improvement = (
            time_improvement * 0.7 + memory_improvement * 0.3
        )
        
        # Determine success based on improvement threshold
        success = overall_improvement >= settings.MIN_PERFORMANCE_IMPROVEMENT * 100
        
        return OptimizationResult(
            optimization_id=f"opt_{int(time.time())}",
            original_metrics=original,
            optimized_metrics=optimized,
            improvement_percentage=overall_improvement,
            success=success,
            applied_at=datetime.now(),
            context=context
        )
    
    def get_performance_summary(self) -> Dict[str, Any]:
        
        with self.lock:
            metrics = self.metrics_buffer.copy()
        
        if not metrics:
            return {"error": "No performance metrics available"}
        
        # Group by context
        context_groups = {}
        for metric in metrics:
            context = metric.context
            if context not in context_groups:
                context_groups[context] = []
            context_groups[context].append(metric)
        
        summary = {
            "total_measurements": len(metrics),
            "contexts": list(context_groups.keys()),
            "summary_by_context": {}
        }
        
        # Calculate summary statistics for each context
        for context, context_metrics in context_groups.items():
            if not context_metrics:
                continue
            
            execution_times = [m.execution_time for m in context_metrics]
            memory_usages = [m.memory_usage for m in context_metrics]
            cpu_usages = [m.cpu_usage for m in context_metrics]
            
            summary["summary_by_context"][context] = {
                "measurement_count": len(context_metrics),
                "avg_execution_time": sum(execution_times) / len(execution_times),
                "max_execution_time": max(execution_times),
                "avg_memory_usage": sum(memory_usages) / len(memory_usages),
                "max_memory_usage": max(memory_usages),
                "avg_cpu_usage": sum(cpu_usages) / len(cpu_usages),
                "max_cpu_usage": max(cpu_usages)
            }
        
        return summary
    
    def export_metrics(self, file_path: str, format: str = "json"):
        
        with self.lock:
            metrics_data = [asdict(m) for m in self.metrics_buffer]
        
        # Convert datetime objects to ISO format for JSON serialization
        for metric_data in metrics_data:
            metric_data["timestamp"] = metric_data["timestamp"].isoformat()
        
        if format.lower() == "json":
            with open(file_path, 'w') as f:
                json.dump(metrics_data, f, indent=2)
        else:
            raise ValueError(f"Unsupported export format: {format}")
        
        logger.info(f"Exported {len(metrics_data)} metrics to {file_path}")
    
    def clear_metrics(self):
        with self.lock:
            self.metrics_buffer.clear()
        logger.info("Performance metrics cleared")


performance_profiler = PerformanceProfiler()
