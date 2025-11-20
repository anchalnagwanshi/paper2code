import time
import json
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Any
from collections import defaultdict, deque
from dataclasses import dataclass, asdict
import logging
import config

logger = logging.getLogger(__name__)


@dataclass
class PipelineMetrics:
    """Metrics for a single pipeline run."""
    run_id: str
    paper_path: str
    start_time: float
    end_time: Optional[float] = None
    status: str = "running"  # running, success, failed
    
    # Stage timings
    researcher_duration: float = 0.0
    coder_duration: float = 0.0
    qa_duration: float = 0.0
    total_duration: float = 0.0
    
    # Retry stats
    retries: int = 0
    failed_stages: List[str] = None
    
    # Code quality
    code_length: int = 0
    code_valid: bool = False
    
    # LLM stats
    llm_calls: int = 0
    llm_tokens: int = 0
    llm_cost: float = 0.0
    
    # Error info
    error_type: Optional[str] = None
    error_message: Optional[str] = None
    
    def __post_init__(self):
        if self.failed_stages is None:
            self.failed_stages = []
    
    def to_dict(self) -> Dict:
        return asdict(self)


class MetricsCollector:
    """Collects and aggregates metrics across runs."""
    
    def __init__(self):
        self.current_run: Optional[PipelineMetrics] = None
        self.historical_runs: List[PipelineMetrics] = []
        self.stage_timings = defaultdict(list)
        self.error_counts = defaultdict(int)
        self.success_rate_window = deque(maxlen=100)  # Last 100 runs
        
        # Load historical data
        self._load_historical_data()
    
    def start_run(self, run_id: str, paper_path: str) -> PipelineMetrics:
        """Start tracking a new pipeline run."""
        self.current_run = PipelineMetrics(
            run_id=run_id,
            paper_path=paper_path,
            start_time=time.time()
        )
        logger.info(f"Started tracking run: {run_id}")
        return self.current_run
    
    def end_run(self, status: str, error: Optional[Exception] = None):
        """Mark the current run as complete."""
        if not self.current_run:
            logger.warning("end_run called but no current run")
            return
        
        self.current_run.end_time = time.time()
        self.current_run.total_duration = (
            self.current_run.end_time - self.current_run.start_time
        )
        self.current_run.status = status
        
        if error:
            self.current_run.error_type = type(error).__name__
            self.current_run.error_message = str(error)
            self.error_counts[type(error).__name__] += 1
        
        # Update success rate
        self.success_rate_window.append(status == "success")
        
        # Store historical data
        self.historical_runs.append(self.current_run)
        self._save_metrics(self.current_run)
        
        logger.info(
            f"Run {self.current_run.run_id} completed: {status} "
            f"(duration: {self.current_run.total_duration:.2f}s)"
        )
        
        self.current_run = None
    
    def record_stage_duration(self, stage: str, duration: float):
        """Record duration for a pipeline stage."""
        self.stage_timings[stage].append(duration)
        
        if self.current_run:
            setattr(self.current_run, f"{stage}_duration", duration)
    
    def record_llm_call(self, tokens: int = 0, cost: float = 0.0):
        """Record an LLM API call."""
        if self.current_run:
            self.current_run.llm_calls += 1
            self.current_run.llm_tokens += tokens
            self.current_run.llm_cost += cost
    
    def record_retry(self, stage: str):
        """Record a retry attempt."""
        if self.current_run:
            self.current_run.retries += 1
            self.current_run.failed_stages.append(stage)
    
    def record_code_quality(self, code: str, valid: bool):
        """Record code quality metrics."""
        if self.current_run:
            self.current_run.code_length = len(code)
            self.current_run.code_valid = valid
    
    def get_summary_stats(self) -> Dict[str, Any]:
        """Get summary statistics across all runs."""
        if not self.historical_runs:
            return {"message": "No runs recorded yet"}
        
        total_runs = len(self.historical_runs)
        successful_runs = sum(1 for r in self.historical_runs if r.status == "success")
        failed_runs = total_runs - successful_runs
        
        # Calculate averages
        avg_duration = sum(r.total_duration for r in self.historical_runs) / total_runs
        avg_retries = sum(r.retries for r in self.historical_runs) / total_runs
        avg_llm_calls = sum(r.llm_calls for r in self.historical_runs) / total_runs
        total_cost = sum(r.llm_cost for r in self.historical_runs)
        
        # Recent success rate (last 100 runs)
        recent_success_rate = (
            sum(self.success_rate_window) / len(self.success_rate_window)
            if self.success_rate_window else 0.0
        )
        
        # Stage timing averages
        stage_stats = {}
        for stage, timings in self.stage_timings.items():
            if timings:
                stage_stats[stage] = {
                    "avg": sum(timings) / len(timings),
                    "min": min(timings),
                    "max": max(timings),
                    "count": len(timings)
                }
        
        return {
            "total_runs": total_runs,
            "successful_runs": successful_runs,
            "failed_runs": failed_runs,
            "success_rate": successful_runs / total_runs,
            "recent_success_rate": recent_success_rate,
            "avg_duration": avg_duration,
            "avg_retries": avg_retries,
            "avg_llm_calls": avg_llm_calls,
            "total_cost": total_cost,
            "stage_timings": stage_stats,
            "error_distribution": dict(self.error_counts),
        }
    
    def _save_metrics(self, metrics: PipelineMetrics):
        """Save metrics to file."""
        metrics_file = config.METRICS_DIR / f"run_{metrics.run_id}.json"
        try:
            with open(metrics_file, 'w') as f:
                json.dump(metrics.to_dict(), f, indent=2)
        except Exception as e:
            logger.error(f"Failed to save metrics: {e}")
    
    def _load_historical_data(self):
        """Load historical metrics from disk."""
        try:
            for metrics_file in config.METRICS_DIR.glob("run_*.json"):
                with open(metrics_file, 'r') as f:
                    data = json.load(f)
                    metrics = PipelineMetrics(**data)
                    self.historical_runs.append(metrics)
            
            logger.info(f"Loaded {len(self.historical_runs)} historical runs")
        except Exception as e:
            logger.warning(f"Could not load historical data: {e}")
    
    def export_report(self, output_path: Optional[Path] = None):
        """Export a detailed metrics report."""
        if output_path is None:
            output_path = config.METRICS_DIR / f"report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        
        report = {
            "generated_at": datetime.now().isoformat(),
            "summary": self.get_summary_stats(),
            "recent_runs": [r.to_dict() for r in self.historical_runs[-20:]],
        }
        
        with open(output_path, 'w') as f:
            json.dump(report, f, indent=2)
        
        logger.info(f"Exported metrics report to {output_path}")
        return output_path


class CostTracker:
    """Track and manage costs for LLM API calls."""
    
    def __init__(self, budget: float = config.COST_BUDGET_PER_RUN):
        self.budget = budget
        self.current_cost = 0.0
        self.alert_threshold = budget * config.COST_ALERT_THRESHOLD
        self.cost_history = []
    
    def add_cost(self, cost: float, operation: str):
        """Add a cost and check against budget."""
        self.current_cost += cost
        self.cost_history.append({
            "timestamp": datetime.now().isoformat(),
            "operation": operation,
            "cost": cost,
            "total": self.current_cost
        })
        
        if self.current_cost >= self.alert_threshold and self.current_cost < self.budget:
            logger.warning(
                f"Cost alert: ${self.current_cost:.4f} / ${self.budget:.2f} "
                f"({self.current_cost/self.budget*100:.1f}%)"
            )
        elif self.current_cost >= self.budget:
            raise BudgetExceededError(
                f"Budget exceeded: ${self.current_cost:.4f} / ${self.budget:.2f}"
            )
    
    def get_remaining_budget(self) -> float:
        """Get remaining budget."""
        return max(0, self.budget - self.current_cost)
    
    def reset(self):
        """Reset cost tracking for a new run."""
        self.current_cost = 0.0
        self.cost_history = []


class BudgetExceededError(Exception):
    """Raised when cost budget is exceeded."""
    pass


class PerformanceMonitor:
    """Monitor system performance and resource usage."""
    
    def __init__(self):
        self.start_time = time.time()
        self.checkpoints = {}
    
    def checkpoint(self, name: str):
        """Create a timing checkpoint."""
        self.checkpoints[name] = time.time()
    
    def get_duration(self, start_checkpoint: str, end_checkpoint: Optional[str] = None) -> float:
        """Get duration between checkpoints."""
        if start_checkpoint not in self.checkpoints:
            return 0.0
        
        start = self.checkpoints[start_checkpoint]
        end = self.checkpoints.get(end_checkpoint, time.time())
        return end - start
    
    def get_total_duration(self) -> float:
        """Get total duration since monitor start."""
        return time.time() - self.start_time


# Global instances
metrics_collector = MetricsCollector()
cost_tracker = CostTracker()
performance_monitor = PerformanceMonitor()