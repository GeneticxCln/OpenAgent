"""
Observability infrastructure for OpenAgent.

This module provides:
- Structured JSON logging
- Request ID tracking and tracing
- Metrics collection (Prometheus format)
- Performance monitoring
- Distributed tracing support
"""

import json
import logging
import time
import uuid
from contextlib import contextmanager
from dataclasses import dataclass, asdict
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Dict, Any, Optional, List, Callable
from collections import defaultdict
import threading

# Try to import prometheus client
try:
    from prometheus_client import (
        Counter, Histogram, Gauge, Summary,
        generate_latest, CONTENT_TYPE_LATEST,
        CollectorRegistry
    )
    PROMETHEUS_AVAILABLE = True
except ImportError:
    PROMETHEUS_AVAILABLE = False
    # Mock classes for when prometheus_client is not installed
    class Counter:
        def __init__(self, *args, **kwargs): pass
        def inc(self, *args, **kwargs): pass
        def labels(self, **kwargs): return self
    
    class Histogram:
        def __init__(self, *args, **kwargs): pass
        def observe(self, *args, **kwargs): pass
        def time(self): return self
        def __enter__(self): return self
        def __exit__(self, *args): pass
        def labels(self, **kwargs): return self
    
    class Gauge:
        def __init__(self, *args, **kwargs): pass
        def set(self, *args, **kwargs): pass
        def inc(self, *args, **kwargs): pass
        def dec(self, *args, **kwargs): pass
        def labels(self, **kwargs): return self
    
    class Summary:
        def __init__(self, *args, **kwargs): pass
        def observe(self, *args, **kwargs): pass
        def time(self): return self
        def __enter__(self): return self
        def __exit__(self, *args): pass
        def labels(self, **kwargs): return self


class LogLevel(Enum):
    """Log levels for structured logging."""
    DEBUG = "debug"
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"


@dataclass
class LogEntry:
    """Structured log entry."""
    timestamp: str
    level: str
    message: str
    request_id: Optional[str] = None
    trace_id: Optional[str] = None
    span_id: Optional[str] = None
    user_id: Optional[str] = None
    service: str = "openagent"
    component: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None
    error: Optional[Dict[str, Any]] = None
    duration_ms: Optional[float] = None


class StructuredLogger:
    """Structured JSON logger with request tracking."""
    
    def __init__(self, name: str, output_file: Optional[Path] = None):
        """Initialize structured logger.
        
        Args:
            name: Logger name (usually module name)
            output_file: Optional file path for JSON log output
        """
        self.name = name
        self.logger = logging.getLogger(name)
        self.output_file = output_file
        
        # Thread-local storage for request context
        self._context = threading.local()
        
        # Setup JSON formatter if output file specified
        if output_file:
            self._setup_json_handler(output_file)
    
    def _setup_json_handler(self, output_file: Path):
        """Setup JSON file handler."""
        output_file.parent.mkdir(parents=True, exist_ok=True)
        
        handler = logging.FileHandler(output_file)
        handler.setFormatter(JSONFormatter())
        self.logger.addHandler(handler)
    
    def set_context(self, request_id: Optional[str] = None,
                    trace_id: Optional[str] = None,
                    span_id: Optional[str] = None,
                    user_id: Optional[str] = None):
        """Set request context for all subsequent logs.
        
        Args:
            request_id: Request identifier
            trace_id: Distributed trace ID
            span_id: Span ID within trace
            user_id: User identifier
        """
        self._context.request_id = request_id
        self._context.trace_id = trace_id
        self._context.span_id = span_id
        self._context.user_id = user_id
    
    def clear_context(self):
        """Clear request context."""
        self._context = threading.local()
    
    def _create_entry(self, level: str, message: str, **kwargs) -> LogEntry:
        """Create a structured log entry."""
        return LogEntry(
            timestamp=datetime.now(timezone.utc).isoformat(),
            level=level,
            message=message,
            request_id=getattr(self._context, 'request_id', None),
            trace_id=getattr(self._context, 'trace_id', None),
            span_id=getattr(self._context, 'span_id', None),
            user_id=getattr(self._context, 'user_id', None),
            component=self.name,
            **kwargs
        )
    
    def _log(self, level: LogLevel, message: str, **kwargs):
        """Internal log method."""
        entry = self._create_entry(level.value, message, **kwargs)
        
        # Log as JSON
        log_dict = asdict(entry)
        log_dict = {k: v for k, v in log_dict.items() if v is not None}
        
        getattr(self.logger, level.value)(json.dumps(log_dict))
    
    def debug(self, message: str, **kwargs):
        """Log debug message."""
        self._log(LogLevel.DEBUG, message, **kwargs)
    
    def info(self, message: str, **kwargs):
        """Log info message."""
        self._log(LogLevel.INFO, message, **kwargs)
    
    def warning(self, message: str, **kwargs):
        """Log warning message."""
        self._log(LogLevel.WARNING, message, **kwargs)
    
    def error(self, message: str, error: Optional[Exception] = None, **kwargs):
        """Log error message."""
        error_dict = None
        if error:
            error_dict = {
                "type": type(error).__name__,
                "message": str(error),
                "traceback": None  # Could add traceback if needed
            }
        self._log(LogLevel.ERROR, message, error=error_dict, **kwargs)
    
    def critical(self, message: str, **kwargs):
        """Log critical message."""
        self._log(LogLevel.CRITICAL, message, **kwargs)
    
    @contextmanager
    def timed_operation(self, operation_name: str, **kwargs):
        """Context manager for timing operations.
        
        Args:
            operation_name: Name of the operation being timed
            **kwargs: Additional metadata to log
        """
        start_time = time.time()
        
        try:
            yield
        finally:
            duration_ms = (time.time() - start_time) * 1000
            self.info(
                f"Operation completed: {operation_name}",
                duration_ms=duration_ms,
                metadata={"operation": operation_name, **kwargs}
            )


class JSONFormatter(logging.Formatter):
    """JSON formatter for Python logging."""
    
    def format(self, record):
        """Format log record as JSON with redaction."""
        from openagent.core.redact import redact_text
        # Try to parse the message as JSON if it looks like JSON
        try:
            if isinstance(record.msg, str) and record.msg.startswith('{'):
                # Redact the raw JSON string fields best-effort
                return redact_text(record.msg)
        except Exception:
            pass
        
        # Otherwise create a basic JSON structure
        msg = record.getMessage()
        try:
            msg = redact_text(msg)
        except Exception:
            pass
        log_dict = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "level": record.levelname.lower(),
            "message": msg,
            "component": record.name,
        }
        
        return json.dumps(log_dict)


class MetricsCollector:
    """Metrics collector with Prometheus format support."""
    
    def __init__(self, namespace: str = "openagent"):
        """Initialize metrics collector.
        
        Args:
            namespace: Prometheus metrics namespace
        """
        self.namespace = namespace
        self.registry = CollectorRegistry() if PROMETHEUS_AVAILABLE else None
        
        # Define metrics
        self._setup_metrics()
        
        # In-memory metrics storage (for when Prometheus is not available)
        self._metrics_data = defaultdict(lambda: defaultdict(float))
    
    def _setup_metrics(self):
        """Setup Prometheus metrics."""
        if not PROMETHEUS_AVAILABLE:
            return
        
        # Request metrics
        self.request_total = Counter(
            f'{self.namespace}_requests_total',
            'Total number of requests',
            ['method', 'endpoint', 'status'],
            registry=self.registry
        )
        
        self.request_duration = Histogram(
            f'{self.namespace}_request_duration_seconds',
            'Request duration in seconds',
            ['method', 'endpoint'],
            registry=self.registry
        )
        
        # Tool metrics
        self.tool_executions = Counter(
            f'{self.namespace}_tool_executions_total',
            'Total number of tool executions',
            ['tool_name', 'status'],
            registry=self.registry
        )
        
        self.tool_duration = Histogram(
            f'{self.namespace}_tool_duration_seconds',
            'Tool execution duration in seconds',
            ['tool_name'],
            registry=self.registry
        )
        
        # Agent metrics
        self.agent_messages = Counter(
            f'{self.namespace}_agent_messages_total',
            'Total number of agent messages processed',
            ['agent_name', 'status'],
            registry=self.registry
        )
        
        self.agent_response_time = Histogram(
            f'{self.namespace}_agent_response_time_seconds',
            'Agent response time in seconds',
            ['agent_name'],
            registry=self.registry
        )
        
        # Model metrics
        self.model_inference = Counter(
            f'{self.namespace}_model_inference_total',
            'Total number of model inferences',
            ['model_name', 'status'],
            registry=self.registry
        )
        
        self.model_latency = Histogram(
            f'{self.namespace}_model_latency_seconds',
            'Model inference latency in seconds',
            ['model_name'],
            registry=self.registry
        )
        
        # Policy metrics
        self.policy_evaluations = Counter(
            f'{self.namespace}_policy_evaluations_total',
            'Total number of policy evaluations',
            ['decision', 'risk_level'],
            registry=self.registry
        )
        
        self.commands_blocked = Counter(
            f'{self.namespace}_commands_blocked_total',
            'Total number of commands blocked by policy',
            ['risk_level'],
            registry=self.registry
        )
        
        # System metrics
        self.active_sessions = Gauge(
            f'{self.namespace}_active_sessions',
            'Number of active sessions',
            registry=self.registry
        )
        
        self.memory_usage = Gauge(
            f'{self.namespace}_memory_usage_bytes',
            'Memory usage in bytes',
            registry=self.registry
        )
    
    def record_request(self, method: str, endpoint: str, status: int, duration: float):
        """Record HTTP request metrics."""
        if PROMETHEUS_AVAILABLE:
            self.request_total.labels(method=method, endpoint=endpoint, status=str(status)).inc()
            self.request_duration.labels(method=method, endpoint=endpoint).observe(duration)
        else:
            self._metrics_data['request_total'][(method, endpoint, status)] += 1
            self._metrics_data['request_duration'][(method, endpoint)] = duration
    
    def record_tool_execution(self, tool_name: str, success: bool, duration: float):
        """Record tool execution metrics."""
        status = "success" if success else "failure"
        if PROMETHEUS_AVAILABLE:
            self.tool_executions.labels(tool_name=tool_name, status=status).inc()
            self.tool_duration.labels(tool_name=tool_name).observe(duration)
        else:
            self._metrics_data['tool_executions'][(tool_name, status)] += 1
            self._metrics_data['tool_duration'][tool_name] = duration
    
    def record_agent_message(self, agent_name: str, success: bool, response_time: float):
        """Record agent message processing metrics."""
        status = "success" if success else "failure"
        if PROMETHEUS_AVAILABLE:
            self.agent_messages.labels(agent_name=agent_name, status=status).inc()
            self.agent_response_time.labels(agent_name=agent_name).observe(response_time)
        else:
            self._metrics_data['agent_messages'][(agent_name, status)] += 1
            self._metrics_data['agent_response_time'][agent_name] = response_time
    
    def record_model_inference(self, model_name: str, success: bool, latency: float):
        """Record model inference metrics."""
        status = "success" if success else "failure"
        if PROMETHEUS_AVAILABLE:
            self.model_inference.labels(model_name=model_name, status=status).inc()
            self.model_latency.labels(model_name=model_name).observe(latency)
        else:
            self._metrics_data['model_inference'][(model_name, status)] += 1
            self._metrics_data['model_latency'][model_name] = latency
    
    def record_policy_evaluation(self, decision: str, risk_level: str):
        """Record policy evaluation metrics."""
        if PROMETHEUS_AVAILABLE:
            self.policy_evaluations.labels(decision=decision, risk_level=risk_level).inc()
            if decision == "deny":
                self.commands_blocked.labels(risk_level=risk_level).inc()
        else:
            self._metrics_data['policy_evaluations'][(decision, risk_level)] += 1
            if decision == "deny":
                self._metrics_data['commands_blocked'][risk_level] += 1
    
    def update_active_sessions(self, count: int):
        """Update active sessions gauge."""
        if PROMETHEUS_AVAILABLE:
            self.active_sessions.set(count)
        else:
            self._metrics_data['active_sessions']['value'] = count
    
    def update_memory_usage(self, bytes_used: int):
        """Update memory usage gauge."""
        if PROMETHEUS_AVAILABLE:
            self.memory_usage.set(bytes_used)
        else:
            self._metrics_data['memory_usage']['value'] = bytes_used
    
    def get_metrics(self) -> bytes:
        """Get metrics in Prometheus format."""
        if PROMETHEUS_AVAILABLE:
            return generate_latest(self.registry)
        else:
            # Return a simple text representation
            lines = []
            for metric_name, values in self._metrics_data.items():
                for labels, value in values.items():
                    if isinstance(labels, tuple):
                        label_str = ','.join(f'{i}="{v}"' for i, v in enumerate(labels))
                        lines.append(f'{self.namespace}_{metric_name}{{{label_str}}} {value}')
                    else:
                        lines.append(f'{self.namespace}_{metric_name} {value}')
            return '\n'.join(lines).encode('utf-8')
    
    def get_metrics_dict(self) -> Dict[str, Any]:
        """Get metrics as a dictionary."""
        return dict(self._metrics_data)


class RequestTracker:
    """Request tracker for distributed tracing."""
    
    def __init__(self):
        """Initialize request tracker."""
        self._context = threading.local()
    
    def start_request(self, request_id: Optional[str] = None,
                     trace_id: Optional[str] = None) -> Dict[str, str]:
        """Start tracking a new request.
        
        Args:
            request_id: Optional request ID (generated if not provided)
            trace_id: Optional trace ID (generated if not provided)
            
        Returns:
            Dictionary with request_id, trace_id, and span_id
        """
        if not request_id:
            request_id = str(uuid.uuid4())
        if not trace_id:
            trace_id = str(uuid.uuid4())
        
        span_id = str(uuid.uuid4())
        
        self._context.request_id = request_id
        self._context.trace_id = trace_id
        self._context.span_id = span_id
        self._context.start_time = time.time()
        
        return {
            "request_id": request_id,
            "trace_id": trace_id,
            "span_id": span_id
        }
    
    def get_context(self) -> Dict[str, str]:
        """Get current request context."""
        return {
            "request_id": getattr(self._context, 'request_id', None),
            "trace_id": getattr(self._context, 'trace_id', None),
            "span_id": getattr(self._context, 'span_id', None)
        }
    
    def end_request(self) -> float:
        """End request tracking and return duration.
        
        Returns:
            Request duration in seconds
        """
        start_time = getattr(self._context, 'start_time', None)
        if start_time:
            duration = time.time() - start_time
            self.clear()
            return duration
        return 0.0
    
    def clear(self):
        """Clear request context."""
        self._context = threading.local()
    
    @contextmanager
    def track_request(self, request_id: Optional[str] = None,
                     trace_id: Optional[str] = None):
        """Context manager for request tracking.
        
        Args:
            request_id: Optional request ID
            trace_id: Optional trace ID
        """
        context = self.start_request(request_id, trace_id)
        try:
            yield context
        finally:
            self.end_request()


# Global instances
_logger_cache: Dict[str, StructuredLogger] = {}
_metrics_collector: Optional[MetricsCollector] = None
_request_tracker: Optional[RequestTracker] = None


def get_logger(name: str, output_file: Optional[Path] = None) -> StructuredLogger:
    """Get or create a structured logger.
    
    Args:
        name: Logger name
        output_file: Optional JSON output file
        
    Returns:
        StructuredLogger instance
    """
    if name not in _logger_cache:
        _logger_cache[name] = StructuredLogger(name, output_file)
    return _logger_cache[name]


def get_metrics_collector() -> MetricsCollector:
    """Get the global metrics collector."""
    global _metrics_collector
    if _metrics_collector is None:
        _metrics_collector = MetricsCollector()
    return _metrics_collector


def get_request_tracker() -> RequestTracker:
    """Get the global request tracker."""
    global _request_tracker
    if _request_tracker is None:
        _request_tracker = RequestTracker()
    return _request_tracker


def configure_observability(
    log_level: str = "INFO",
    log_file: Optional[Path] = None,
    enable_metrics: bool = True,
    enable_tracing: bool = True
):
    """Configure global observability settings.
    
    Args:
        log_level: Logging level
        log_file: Optional JSON log file path
        enable_metrics: Enable metrics collection
        enable_tracing: Enable distributed tracing
    """
    import os
    # Allow env override for log file path
    if log_file is None:
        env_path = os.getenv("OPENAGENT_LOG_FILE")
        if env_path:
            try:
                log_file = Path(env_path)
            except Exception:
                log_file = None

    # Configure root logger
    logging.basicConfig(
        level=getattr(logging, log_level.upper()),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Setup JSON logging if specified
    if log_file:
        root_logger = logging.getLogger()
        try:
            handler = logging.FileHandler(log_file)
            handler.setFormatter(JSONFormatter())
            root_logger.addHandler(handler)
        except Exception:
            # Fallback silently if file cannot be opened
            pass
    
    # Initialize global collectors
    if enable_metrics:
        get_metrics_collector()
    
    if enable_tracing:
        get_request_tracker()
