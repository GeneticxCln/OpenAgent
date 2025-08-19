"""
Resource monitoring system for OpenAgent.

This module provides:
- Real-time CPU, memory, and GPU monitoring
- Performance trend analysis and alerting
- Automatic scaling recommendations
- Resource utilization optimization
"""

import asyncio
import logging
import statistics
import time
from collections import deque
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Tuple

import psutil

try:
    import nvidia_ml_py as nvml

    nvml.nvmlInit()
    HAS_NVIDIA = True
except (ImportError, Exception):
    HAS_NVIDIA = False

try:
    import GPUtil

    HAS_GPUTIL = True
except ImportError:
    HAS_GPUTIL = False

from openagent.core.exceptions import AgentError

logger = logging.getLogger(__name__)


class AlertLevel(Enum):
    """Resource alert levels."""

    INFO = "info"
    WARNING = "warning"
    CRITICAL = "critical"


class ScalingAction(Enum):
    """Auto-scaling actions."""

    SCALE_UP = "scale_up"
    SCALE_DOWN = "scale_down"
    OPTIMIZE = "optimize"
    NO_ACTION = "no_action"


@dataclass
class CPUStats:
    """CPU usage statistics."""

    percent: float = 0.0
    count: int = 0
    frequency: float = 0.0  # MHz
    per_core: List[float] = field(default_factory=list)
    load_avg: Tuple[float, float, float] = (0.0, 0.0, 0.0)  # 1, 5, 15 min averages

    @property
    def is_high_usage(self) -> bool:
        """Check if CPU usage is high (>80%)."""
        return self.percent > 80.0


@dataclass
class MemoryStats:
    """Memory usage statistics."""

    total: int = 0  # bytes
    available: int = 0  # bytes
    used: int = 0  # bytes
    percent: float = 0.0
    swap_total: int = 0  # bytes
    swap_used: int = 0  # bytes
    swap_percent: float = 0.0

    @property
    def used_gb(self) -> float:
        """Memory used in GB."""
        return self.used / (1024**3)

    @property
    def total_gb(self) -> float:
        """Total memory in GB."""
        return self.total / (1024**3)

    @property
    def is_high_usage(self) -> bool:
        """Check if memory usage is high (>80%)."""
        return self.percent > 80.0


@dataclass
class GPUStats:
    """GPU usage statistics."""

    gpu_id: int = 0
    name: str = ""
    memory_total: int = 0  # MB
    memory_used: int = 0  # MB
    memory_percent: float = 0.0
    gpu_percent: float = 0.0
    temperature: float = 0.0  # Celsius
    power_draw: float = 0.0  # Watts
    power_limit: float = 0.0  # Watts

    @property
    def is_high_memory_usage(self) -> bool:
        """Check if GPU memory usage is high (>80%)."""
        return self.memory_percent > 80.0

    @property
    def is_high_utilization(self) -> bool:
        """Check if GPU utilization is high (>80%)."""
        return self.gpu_percent > 80.0

    @property
    def is_overheating(self) -> bool:
        """Check if GPU is overheating (>85°C)."""
        return self.temperature > 85.0


@dataclass
class NetworkStats:
    """Network usage statistics."""

    bytes_sent: int = 0
    bytes_recv: int = 0
    packets_sent: int = 0
    packets_recv: int = 0
    bytes_sent_rate: float = 0.0  # bytes/sec
    bytes_recv_rate: float = 0.0  # bytes/sec


@dataclass
class DiskStats:
    """Disk usage statistics."""

    total: int = 0  # bytes
    used: int = 0  # bytes
    free: int = 0  # bytes
    percent: float = 0.0
    read_count: int = 0
    write_count: int = 0
    read_bytes: int = 0
    write_bytes: int = 0
    read_rate: float = 0.0  # bytes/sec
    write_rate: float = 0.0  # bytes/sec

    @property
    def is_high_usage(self) -> bool:
        """Check if disk usage is high (>85%)."""
        return self.percent > 85.0


@dataclass
class ResourceMetrics:
    """Comprehensive system resource metrics."""

    timestamp: float = 0.0
    cpu: CPUStats = field(default_factory=CPUStats)
    memory: MemoryStats = field(default_factory=MemoryStats)
    gpus: List[GPUStats] = field(default_factory=list)
    network: NetworkStats = field(default_factory=NetworkStats)
    disk: DiskStats = field(default_factory=DiskStats)

    @property
    def has_resource_pressure(self) -> bool:
        """Check if system is under resource pressure."""
        return (
            self.cpu.is_high_usage
            or self.memory.is_high_usage
            or any(
                gpu.is_high_memory_usage or gpu.is_high_utilization for gpu in self.gpus
            )
            or self.disk.is_high_usage
        )


@dataclass
class ResourceAlert:
    """Resource monitoring alert."""

    timestamp: float
    level: AlertLevel
    component: str  # cpu, memory, gpu, disk, network
    message: str
    value: float
    threshold: float
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ScalingRecommendation:
    """Auto-scaling recommendation."""

    timestamp: float
    action: ScalingAction
    component: str
    reason: str
    confidence: float  # 0.0 to 1.0
    parameters: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ResourceThresholds:
    """Resource monitoring thresholds."""

    cpu_warning: float = 70.0  # %
    cpu_critical: float = 85.0  # %
    memory_warning: float = 75.0  # %
    memory_critical: float = 90.0  # %
    gpu_memory_warning: float = 75.0  # %
    gpu_memory_critical: float = 90.0  # %
    gpu_util_warning: float = 80.0  # %
    gpu_util_critical: float = 95.0  # %
    disk_warning: float = 80.0  # %
    disk_critical: float = 90.0  # %
    temperature_warning: float = 80.0  # °C
    temperature_critical: float = 85.0  # °C


class ResourceMonitor:
    """
    Advanced resource monitoring system.

    Features:
    - Real-time monitoring of CPU, memory, GPU, disk, network
    - Historical trend analysis
    - Alert generation based on thresholds
    - Auto-scaling recommendations
    - Performance bottleneck detection
    """

    def __init__(
        self,
        monitoring_interval: float = 5.0,  # seconds
        history_size: int = 1440,  # 24 hours at 1-minute intervals
        enable_gpu_monitoring: bool = True,
        enable_alerting: bool = True,
        thresholds: Optional[ResourceThresholds] = None,
    ):
        """
        Initialize resource monitor.

        Args:
            monitoring_interval: Monitoring frequency in seconds
            history_size: Number of historical metrics to keep
            enable_gpu_monitoring: Enable GPU monitoring
            enable_alerting: Enable alert generation
            thresholds: Custom resource thresholds
        """
        self.monitoring_interval = monitoring_interval
        self.history_size = history_size
        self.enable_gpu_monitoring = enable_gpu_monitoring
        self.enable_alerting = enable_alerting
        self.thresholds = thresholds or ResourceThresholds()

        # Historical data
        self._metrics_history: deque = deque(maxlen=history_size)
        self._alerts: deque = deque(maxlen=1000)  # Keep last 1000 alerts
        self._scaling_recommendations: deque = deque(maxlen=100)

        # Previous values for rate calculations
        self._prev_network_stats: Optional[Dict] = None
        self._prev_disk_stats: Optional[Dict] = None
        self._prev_timestamp: float = 0.0

        # Alert callbacks
        self._alert_callbacks: List[Callable] = []
        self._scaling_callbacks: List[Callable] = []

        # Background tasks
        self._monitoring_task: Optional[asyncio.Task] = None
        self._analysis_task: Optional[asyncio.Task] = None
        self._shutdown_event = asyncio.Event()

        # Performance tracking
        self._monitoring_start_time = time.time()
        self._total_samples = 0

        logger.info(
            f"ResourceMonitor initialized: interval={monitoring_interval}s, gpu_monitoring={enable_gpu_monitoring}"
        )
        self._start_monitoring()

    def get_current_metrics(self) -> ResourceMetrics:
        """Get current system resource metrics."""
        timestamp = time.time()
        metrics = ResourceMetrics(timestamp=timestamp)

        # CPU metrics
        metrics.cpu = self._get_cpu_stats()

        # Memory metrics
        metrics.memory = self._get_memory_stats()

        # GPU metrics
        if self.enable_gpu_monitoring:
            metrics.gpus = self._get_gpu_stats()

        # Network metrics
        metrics.network = self._get_network_stats()

        # Disk metrics
        metrics.disk = self._get_disk_stats()

        return metrics

    def get_metrics_history(
        self, duration: Optional[float] = None
    ) -> List[ResourceMetrics]:
        """
        Get historical metrics.

        Args:
            duration: Duration in seconds (None for all history)

        Returns:
            List of historical metrics
        """
        if duration is None:
            return list(self._metrics_history)

        cutoff_time = time.time() - duration
        return [
            metrics
            for metrics in self._metrics_history
            if metrics.timestamp >= cutoff_time
        ]

    def get_recent_alerts(self, duration: float = 3600) -> List[ResourceAlert]:
        """
        Get recent alerts.

        Args:
            duration: Duration in seconds (default: 1 hour)

        Returns:
            List of recent alerts
        """
        cutoff_time = time.time() - duration
        return [alert for alert in self._alerts if alert.timestamp >= cutoff_time]

    def get_scaling_recommendations(
        self, duration: float = 3600
    ) -> List[ScalingRecommendation]:
        """
        Get recent scaling recommendations.

        Args:
            duration: Duration in seconds (default: 1 hour)

        Returns:
            List of scaling recommendations
        """
        cutoff_time = time.time() - duration
        return [
            rec for rec in self._scaling_recommendations if rec.timestamp >= cutoff_time
        ]

    def add_alert_callback(self, callback: Callable[[ResourceAlert], None]) -> None:
        """Add callback for resource alerts."""
        self._alert_callbacks.append(callback)

    def add_scaling_callback(
        self, callback: Callable[[ScalingRecommendation], None]
    ) -> None:
        """Add callback for scaling recommendations."""
        self._scaling_callbacks.append(callback)

    async def analyze_performance_trends(
        self, duration: float = 3600
    ) -> Dict[str, Any]:
        """
        Analyze performance trends over a duration.

        Args:
            duration: Analysis duration in seconds

        Returns:
            Performance trend analysis
        """
        history = self.get_metrics_history(duration)
        if len(history) < 10:
            return {"error": "Insufficient data for trend analysis"}

        # CPU analysis
        cpu_usage = [m.cpu.percent for m in history]
        cpu_trend = self._analyze_trend(cpu_usage)

        # Memory analysis
        memory_usage = [m.memory.percent for m in history]
        memory_trend = self._analyze_trend(memory_usage)

        # GPU analysis
        gpu_trends = {}
        if history[0].gpus:
            for i in range(len(history[0].gpus)):
                gpu_usage = [m.gpus[i].gpu_percent for m in history if i < len(m.gpus)]
                gpu_memory = [
                    m.gpus[i].memory_percent for m in history if i < len(m.gpus)
                ]

                gpu_trends[f"gpu_{i}"] = {
                    "utilization": self._analyze_trend(gpu_usage),
                    "memory": self._analyze_trend(gpu_memory),
                }

        # Performance bottlenecks
        bottlenecks = self._identify_bottlenecks(history)

        return {
            "duration": duration,
            "sample_count": len(history),
            "trends": {
                "cpu": cpu_trend,
                "memory": memory_trend,
                "gpus": gpu_trends,
            },
            "bottlenecks": bottlenecks,
            "recommendations": self._generate_performance_recommendations(history),
        }

    async def get_system_health_report(self) -> Dict[str, Any]:
        """Get comprehensive system health report."""
        current_metrics = self.get_current_metrics()
        recent_alerts = self.get_recent_alerts(3600)  # Last hour
        recent_scaling_recs = self.get_scaling_recommendations(3600)

        # Health score calculation
        health_score = self._calculate_health_score(current_metrics)

        # Resource utilization summary
        utilization = {
            "cpu": current_metrics.cpu.percent,
            "memory": current_metrics.memory.percent,
            "disk": current_metrics.disk.percent,
        }

        if current_metrics.gpus:
            utilization["gpus"] = [
                {
                    "id": gpu.gpu_id,
                    "utilization": gpu.gpu_percent,
                    "memory": gpu.memory_percent,
                }
                for gpu in current_metrics.gpus
            ]

        # System status
        status = "healthy"
        if health_score < 60:
            status = "critical"
        elif health_score < 80:
            status = "warning"

        return {
            "timestamp": current_metrics.timestamp,
            "status": status,
            "health_score": health_score,
            "utilization": utilization,
            "alerts": {
                "total": len(recent_alerts),
                "by_level": {
                    level.value: len([a for a in recent_alerts if a.level == level])
                    for level in AlertLevel
                },
            },
            "scaling_recommendations": len(recent_scaling_recs),
            "uptime": time.time() - self._monitoring_start_time,
            "total_samples": self._total_samples,
        }

    async def shutdown(self) -> None:
        """Shutdown the resource monitor."""
        logger.info("Shutting down resource monitor...")

        self._shutdown_event.set()

        # Cancel background tasks
        if self._monitoring_task:
            self._monitoring_task.cancel()
        if self._analysis_task:
            self._analysis_task.cancel()

        logger.info("Resource monitor shutdown complete")

    def _get_cpu_stats(self) -> CPUStats:
        """Get CPU statistics."""
        cpu_percent = psutil.cpu_percent(interval=None)
        cpu_count = psutil.cpu_count()
        per_core = psutil.cpu_percent(interval=None, percpu=True)

        try:
            cpu_freq = psutil.cpu_freq()
            frequency = cpu_freq.current if cpu_freq else 0.0
        except:
            frequency = 0.0

        try:
            load_avg = psutil.getloadavg()
        except:
            load_avg = (0.0, 0.0, 0.0)

        return CPUStats(
            percent=cpu_percent,
            count=cpu_count,
            frequency=frequency,
            per_core=per_core,
            load_avg=load_avg,
        )

    def _get_memory_stats(self) -> MemoryStats:
        """Get memory statistics."""
        memory = psutil.virtual_memory()
        swap = psutil.swap_memory()

        return MemoryStats(
            total=memory.total,
            available=memory.available,
            used=memory.used,
            percent=memory.percent,
            swap_total=swap.total,
            swap_used=swap.used,
            swap_percent=swap.percent,
        )

    def _get_gpu_stats(self) -> List[GPUStats]:
        """Get GPU statistics."""
        gpus = []

        # Try nvidia-ml-py first
        if HAS_NVIDIA:
            try:
                device_count = nvml.nvmlDeviceGetCount()

                for i in range(device_count):
                    handle = nvml.nvmlDeviceGetHandleByIndex(i)

                    # Basic info
                    name = nvml.nvmlDeviceGetName(handle).decode("utf-8")

                    # Memory info
                    mem_info = nvml.nvmlDeviceGetMemoryInfo(handle)
                    memory_total = mem_info.total // (1024 * 1024)  # MB
                    memory_used = mem_info.used // (1024 * 1024)  # MB
                    memory_percent = (mem_info.used / mem_info.total) * 100

                    # Utilization
                    try:
                        util = nvml.nvmlDeviceGetUtilizationRates(handle)
                        gpu_percent = util.gpu
                    except:
                        gpu_percent = 0.0

                    # Temperature
                    try:
                        temperature = nvml.nvmlDeviceGetTemperature(
                            handle, nvml.NVML_TEMPERATURE_GPU
                        )
                    except:
                        temperature = 0.0

                    # Power
                    try:
                        power_draw = (
                            nvml.nvmlDeviceGetPowerUsage(handle) / 1000.0
                        )  # Watts
                        power_limit = (
                            nvml.nvmlDeviceGetPowerManagementLimitConstraints(handle)[1]
                            / 1000.0
                        )
                    except:
                        power_draw = 0.0
                        power_limit = 0.0

                    gpus.append(
                        GPUStats(
                            gpu_id=i,
                            name=name,
                            memory_total=memory_total,
                            memory_used=memory_used,
                            memory_percent=memory_percent,
                            gpu_percent=gpu_percent,
                            temperature=temperature,
                            power_draw=power_draw,
                            power_limit=power_limit,
                        )
                    )

            except Exception as e:
                logger.warning(f"Failed to get NVIDIA GPU stats: {e}")

        # Fallback to GPUtil
        elif HAS_GPUTIL:
            try:
                gpu_list = GPUtil.getGPUs()
                for gpu in gpu_list:
                    gpus.append(
                        GPUStats(
                            gpu_id=gpu.id,
                            name=gpu.name,
                            memory_total=gpu.memoryTotal,
                            memory_used=gpu.memoryUsed,
                            memory_percent=(gpu.memoryUsed / gpu.memoryTotal) * 100,
                            gpu_percent=gpu.load * 100,
                            temperature=gpu.temperature,
                        )
                    )

            except Exception as e:
                logger.warning(f"Failed to get GPU stats via GPUtil: {e}")

        return gpus

    def _get_network_stats(self) -> NetworkStats:
        """Get network statistics."""
        net_io = psutil.net_io_counters()
        current_time = time.time()

        stats = NetworkStats(
            bytes_sent=net_io.bytes_sent,
            bytes_recv=net_io.bytes_recv,
            packets_sent=net_io.packets_sent,
            packets_recv=net_io.packets_recv,
        )

        # Calculate rates
        if self._prev_network_stats and self._prev_timestamp:
            time_delta = current_time - self._prev_timestamp
            if time_delta > 0:
                stats.bytes_sent_rate = (
                    net_io.bytes_sent - self._prev_network_stats["bytes_sent"]
                ) / time_delta
                stats.bytes_recv_rate = (
                    net_io.bytes_recv - self._prev_network_stats["bytes_recv"]
                ) / time_delta

        # Update previous values
        self._prev_network_stats = {
            "bytes_sent": net_io.bytes_sent,
            "bytes_recv": net_io.bytes_recv,
        }

        return stats

    def _get_disk_stats(self) -> DiskStats:
        """Get disk statistics."""
        disk_usage = psutil.disk_usage("/")
        disk_io = psutil.disk_io_counters()
        current_time = time.time()

        stats = DiskStats(
            total=disk_usage.total,
            used=disk_usage.used,
            free=disk_usage.free,
            percent=disk_usage.used / disk_usage.total * 100,
        )

        if disk_io:
            stats.read_count = disk_io.read_count
            stats.write_count = disk_io.write_count
            stats.read_bytes = disk_io.read_bytes
            stats.write_bytes = disk_io.write_bytes

            # Calculate rates
            if self._prev_disk_stats and self._prev_timestamp:
                time_delta = current_time - self._prev_timestamp
                if time_delta > 0:
                    stats.read_rate = (
                        disk_io.read_bytes - self._prev_disk_stats["read_bytes"]
                    ) / time_delta
                    stats.write_rate = (
                        disk_io.write_bytes - self._prev_disk_stats["write_bytes"]
                    ) / time_delta

            # Update previous values
            self._prev_disk_stats = {
                "read_bytes": disk_io.read_bytes,
                "write_bytes": disk_io.write_bytes,
            }

        return stats

    def _start_monitoring(self) -> None:
        """Start background monitoring tasks."""
        self._monitoring_task = asyncio.create_task(self._monitoring_loop())
        self._analysis_task = asyncio.create_task(self._analysis_loop())

    async def _monitoring_loop(self) -> None:
        """Main monitoring loop."""
        while not self._shutdown_event.is_set():
            try:
                # Collect metrics
                metrics = self.get_current_metrics()
                self._metrics_history.append(metrics)
                self._total_samples += 1

                # Update previous timestamp
                self._prev_timestamp = metrics.timestamp

                # Check for alerts
                if self.enable_alerting:
                    await self._check_alerts(metrics)

                # Wait for next interval
                await asyncio.sleep(self.monitoring_interval)

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in monitoring loop: {e}")
                await asyncio.sleep(self.monitoring_interval)

    async def _analysis_loop(self) -> None:
        """Background analysis and scaling recommendation loop."""
        while not self._shutdown_event.is_set():
            try:
                await asyncio.sleep(60)  # Analyze every minute

                # Generate scaling recommendations
                if len(self._metrics_history) >= 10:
                    await self._generate_scaling_recommendations()

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in analysis loop: {e}")

    async def _check_alerts(self, metrics: ResourceMetrics) -> None:
        """Check for resource alerts."""
        alerts = []

        # CPU alerts
        if metrics.cpu.percent > self.thresholds.cpu_critical:
            alerts.append(
                ResourceAlert(
                    timestamp=metrics.timestamp,
                    level=AlertLevel.CRITICAL,
                    component="cpu",
                    message=f"Critical CPU usage: {metrics.cpu.percent:.1f}%",
                    value=metrics.cpu.percent,
                    threshold=self.thresholds.cpu_critical,
                )
            )
        elif metrics.cpu.percent > self.thresholds.cpu_warning:
            alerts.append(
                ResourceAlert(
                    timestamp=metrics.timestamp,
                    level=AlertLevel.WARNING,
                    component="cpu",
                    message=f"High CPU usage: {metrics.cpu.percent:.1f}%",
                    value=metrics.cpu.percent,
                    threshold=self.thresholds.cpu_warning,
                )
            )

        # Memory alerts
        if metrics.memory.percent > self.thresholds.memory_critical:
            alerts.append(
                ResourceAlert(
                    timestamp=metrics.timestamp,
                    level=AlertLevel.CRITICAL,
                    component="memory",
                    message=f"Critical memory usage: {metrics.memory.percent:.1f}%",
                    value=metrics.memory.percent,
                    threshold=self.thresholds.memory_critical,
                )
            )
        elif metrics.memory.percent > self.thresholds.memory_warning:
            alerts.append(
                ResourceAlert(
                    timestamp=metrics.timestamp,
                    level=AlertLevel.WARNING,
                    component="memory",
                    message=f"High memory usage: {metrics.memory.percent:.1f}%",
                    value=metrics.memory.percent,
                    threshold=self.thresholds.memory_warning,
                )
            )

        # GPU alerts
        for gpu in metrics.gpus:
            # GPU memory
            if gpu.memory_percent > self.thresholds.gpu_memory_critical:
                alerts.append(
                    ResourceAlert(
                        timestamp=metrics.timestamp,
                        level=AlertLevel.CRITICAL,
                        component="gpu",
                        message=f"Critical GPU {gpu.gpu_id} memory: {gpu.memory_percent:.1f}%",
                        value=gpu.memory_percent,
                        threshold=self.thresholds.gpu_memory_critical,
                        metadata={"gpu_id": gpu.gpu_id, "gpu_name": gpu.name},
                    )
                )
            elif gpu.memory_percent > self.thresholds.gpu_memory_warning:
                alerts.append(
                    ResourceAlert(
                        timestamp=metrics.timestamp,
                        level=AlertLevel.WARNING,
                        component="gpu",
                        message=f"High GPU {gpu.gpu_id} memory: {gpu.memory_percent:.1f}%",
                        value=gpu.memory_percent,
                        threshold=self.thresholds.gpu_memory_warning,
                        metadata={"gpu_id": gpu.gpu_id, "gpu_name": gpu.name},
                    )
                )

            # GPU utilization
            if gpu.gpu_percent > self.thresholds.gpu_util_critical:
                alerts.append(
                    ResourceAlert(
                        timestamp=metrics.timestamp,
                        level=AlertLevel.CRITICAL,
                        component="gpu",
                        message=f"Critical GPU {gpu.gpu_id} utilization: {gpu.gpu_percent:.1f}%",
                        value=gpu.gpu_percent,
                        threshold=self.thresholds.gpu_util_critical,
                        metadata={"gpu_id": gpu.gpu_id, "gpu_name": gpu.name},
                    )
                )

            # Temperature
            if gpu.temperature > self.thresholds.temperature_critical:
                alerts.append(
                    ResourceAlert(
                        timestamp=metrics.timestamp,
                        level=AlertLevel.CRITICAL,
                        component="gpu",
                        message=f"GPU {gpu.gpu_id} overheating: {gpu.temperature:.1f}°C",
                        value=gpu.temperature,
                        threshold=self.thresholds.temperature_critical,
                        metadata={"gpu_id": gpu.gpu_id, "gpu_name": gpu.name},
                    )
                )

        # Disk alerts
        if metrics.disk.percent > self.thresholds.disk_critical:
            alerts.append(
                ResourceAlert(
                    timestamp=metrics.timestamp,
                    level=AlertLevel.CRITICAL,
                    component="disk",
                    message=f"Critical disk usage: {metrics.disk.percent:.1f}%",
                    value=metrics.disk.percent,
                    threshold=self.thresholds.disk_critical,
                )
            )
        elif metrics.disk.percent > self.thresholds.disk_warning:
            alerts.append(
                ResourceAlert(
                    timestamp=metrics.timestamp,
                    level=AlertLevel.WARNING,
                    component="disk",
                    message=f"High disk usage: {metrics.disk.percent:.1f}%",
                    value=metrics.disk.percent,
                    threshold=self.thresholds.disk_warning,
                )
            )

        # Store and notify
        for alert in alerts:
            self._alerts.append(alert)
            logger.log(
                (
                    logging.CRITICAL
                    if alert.level == AlertLevel.CRITICAL
                    else logging.WARNING
                ),
                alert.message,
            )

            # Call alert callbacks
            for callback in self._alert_callbacks:
                try:
                    callback(alert)
                except Exception as e:
                    logger.error(f"Alert callback failed: {e}")

    async def _generate_scaling_recommendations(self) -> None:
        """Generate scaling recommendations based on trends."""
        if len(self._metrics_history) < 10:
            return

        recent_metrics = list(self._metrics_history)[-10:]  # Last 10 samples
        recommendations = []

        # CPU scaling analysis
        cpu_usage = [m.cpu.percent for m in recent_metrics]
        cpu_avg = statistics.mean(cpu_usage)
        cpu_trend = self._analyze_trend(cpu_usage)

        if cpu_avg > 80 and cpu_trend.get("direction") == "increasing":
            recommendations.append(
                ScalingRecommendation(
                    timestamp=time.time(),
                    action=ScalingAction.SCALE_UP,
                    component="cpu",
                    reason=f"High CPU usage ({cpu_avg:.1f}%) with increasing trend",
                    confidence=0.8,
                    parameters={"current_usage": cpu_avg, "trend": cpu_trend},
                )
            )
        elif cpu_avg < 30 and cpu_trend.get("direction") == "decreasing":
            recommendations.append(
                ScalingRecommendation(
                    timestamp=time.time(),
                    action=ScalingAction.SCALE_DOWN,
                    component="cpu",
                    reason=f"Low CPU usage ({cpu_avg:.1f}%) with decreasing trend",
                    confidence=0.6,
                    parameters={"current_usage": cpu_avg, "trend": cpu_trend},
                )
            )

        # Memory scaling analysis
        memory_usage = [m.memory.percent for m in recent_metrics]
        memory_avg = statistics.mean(memory_usage)
        memory_trend = self._analyze_trend(memory_usage)

        if memory_avg > 85 and memory_trend.get("direction") == "increasing":
            recommendations.append(
                ScalingRecommendation(
                    timestamp=time.time(),
                    action=ScalingAction.SCALE_UP,
                    component="memory",
                    reason=f"High memory usage ({memory_avg:.1f}%) with increasing trend",
                    confidence=0.9,
                    parameters={"current_usage": memory_avg, "trend": memory_trend},
                )
            )

        # GPU scaling analysis
        if recent_metrics[0].gpus:
            for i in range(len(recent_metrics[0].gpus)):
                gpu_memory_usage = [
                    m.gpus[i].memory_percent for m in recent_metrics if i < len(m.gpus)
                ]
                gpu_util_usage = [
                    m.gpus[i].gpu_percent for m in recent_metrics if i < len(m.gpus)
                ]

                if gpu_memory_usage and gpu_util_usage:
                    gpu_mem_avg = statistics.mean(gpu_memory_usage)
                    gpu_util_avg = statistics.mean(gpu_util_usage)

                    if gpu_mem_avg > 80 or gpu_util_avg > 80:
                        recommendations.append(
                            ScalingRecommendation(
                                timestamp=time.time(),
                                action=ScalingAction.OPTIMIZE,
                                component="gpu",
                                reason=f"GPU {i} high usage (mem: {gpu_mem_avg:.1f}%, util: {gpu_util_avg:.1f}%)",
                                confidence=0.7,
                                parameters={
                                    "gpu_id": i,
                                    "memory_usage": gpu_mem_avg,
                                    "utilization": gpu_util_avg,
                                },
                            )
                        )

        # Store and notify
        for rec in recommendations:
            self._scaling_recommendations.append(rec)
            logger.info(
                f"Scaling recommendation: {rec.action.value} {rec.component} - {rec.reason}"
            )

            # Call scaling callbacks
            for callback in self._scaling_callbacks:
                try:
                    callback(rec)
                except Exception as e:
                    logger.error(f"Scaling callback failed: {e}")

    def _analyze_trend(self, values: List[float]) -> Dict[str, Any]:
        """Analyze trend in values."""
        if len(values) < 3:
            return {"direction": "insufficient_data"}

        # Simple linear regression
        n = len(values)
        x = list(range(n))

        # Calculate slope
        x_mean = statistics.mean(x)
        y_mean = statistics.mean(values)

        numerator = sum((x[i] - x_mean) * (values[i] - y_mean) for i in range(n))
        denominator = sum((x[i] - x_mean) ** 2 for i in range(n))

        if denominator == 0:
            slope = 0
        else:
            slope = numerator / denominator

        # Determine direction
        if abs(slope) < 0.1:
            direction = "stable"
        elif slope > 0:
            direction = "increasing"
        else:
            direction = "decreasing"

        return {
            "direction": direction,
            "slope": slope,
            "current": values[-1],
            "average": y_mean,
            "min": min(values),
            "max": max(values),
            "std_dev": statistics.stdev(values) if len(values) > 1 else 0,
        }

    def _identify_bottlenecks(
        self, history: List[ResourceMetrics]
    ) -> List[Dict[str, Any]]:
        """Identify performance bottlenecks."""
        bottlenecks = []

        if not history:
            return bottlenecks

        # Analyze recent metrics
        cpu_usage = [m.cpu.percent for m in history]
        memory_usage = [m.memory.percent for m in history]

        cpu_avg = statistics.mean(cpu_usage)
        memory_avg = statistics.mean(memory_usage)

        # CPU bottleneck
        if cpu_avg > 80:
            bottlenecks.append(
                {
                    "component": "cpu",
                    "severity": "high" if cpu_avg > 90 else "medium",
                    "description": f"CPU usage averaging {cpu_avg:.1f}%",
                    "recommendation": "Consider scaling up CPU resources or optimizing workload",
                }
            )

        # Memory bottleneck
        if memory_avg > 80:
            bottlenecks.append(
                {
                    "component": "memory",
                    "severity": "high" if memory_avg > 90 else "medium",
                    "description": f"Memory usage averaging {memory_avg:.1f}%",
                    "recommendation": "Consider adding more RAM or optimizing memory usage",
                }
            )

        # GPU bottleneck
        if history[0].gpus:
            for i in range(len(history[0].gpus)):
                gpu_memory = [
                    m.gpus[i].memory_percent for m in history if i < len(m.gpus)
                ]
                gpu_util = [m.gpus[i].gpu_percent for m in history if i < len(m.gpus)]

                if gpu_memory and gpu_util:
                    gpu_mem_avg = statistics.mean(gpu_memory)
                    gpu_util_avg = statistics.mean(gpu_util)

                    if gpu_mem_avg > 80 or gpu_util_avg > 80:
                        bottlenecks.append(
                            {
                                "component": f"gpu_{i}",
                                "severity": (
                                    "high"
                                    if max(gpu_mem_avg, gpu_util_avg) > 90
                                    else "medium"
                                ),
                                "description": f"GPU {i} high usage (mem: {gpu_mem_avg:.1f}%, util: {gpu_util_avg:.1f}%)",
                                "recommendation": "Consider GPU optimization or adding more GPU resources",
                            }
                        )

        return bottlenecks

    def _generate_performance_recommendations(
        self, history: List[ResourceMetrics]
    ) -> List[str]:
        """Generate performance optimization recommendations."""
        recommendations = []

        if not history:
            return recommendations

        cpu_usage = [m.cpu.percent for m in history]
        memory_usage = [m.memory.percent for m in history]

        cpu_avg = statistics.mean(cpu_usage)
        memory_avg = statistics.mean(memory_usage)

        # CPU recommendations
        if cpu_avg > 80:
            recommendations.append(
                "High CPU usage detected. Consider optimizing algorithms or scaling up."
            )
        elif cpu_avg < 20:
            recommendations.append(
                "Low CPU usage. Consider consolidating workloads or scaling down."
            )

        # Memory recommendations
        if memory_avg > 80:
            recommendations.append(
                "High memory usage. Consider memory optimization or adding more RAM."
            )

        # GPU recommendations
        if history[0].gpus:
            gpu_high_usage = any(
                any(
                    m.gpus[i].memory_percent > 80 or m.gpus[i].gpu_percent > 80
                    for m in history
                    if i < len(m.gpus)
                )
                for i in range(len(history[0].gpus))
            )
            if gpu_high_usage:
                recommendations.append(
                    "High GPU usage detected. Consider GPU optimization or model quantization."
                )

        return recommendations

    def _calculate_health_score(self, metrics: ResourceMetrics) -> int:
        """Calculate system health score (0-100)."""
        score = 100

        # CPU penalty
        if metrics.cpu.percent > 90:
            score -= 30
        elif metrics.cpu.percent > 80:
            score -= 15
        elif metrics.cpu.percent > 70:
            score -= 5

        # Memory penalty
        if metrics.memory.percent > 90:
            score -= 30
        elif metrics.memory.percent > 80:
            score -= 15
        elif metrics.memory.percent > 70:
            score -= 5

        # Disk penalty
        if metrics.disk.percent > 90:
            score -= 20
        elif metrics.disk.percent > 80:
            score -= 10

        # GPU penalty
        for gpu in metrics.gpus:
            if gpu.memory_percent > 90 or gpu.gpu_percent > 90:
                score -= 15
            elif gpu.memory_percent > 80 or gpu.gpu_percent > 80:
                score -= 8

            if gpu.is_overheating:
                score -= 20

        return max(0, score)


class ResourceMonitorSingleton:
    """Global resource monitor."""

    def __init__(self):
        self._monitor: Optional[ResourceMonitor] = None

    def initialize(self, **kwargs) -> ResourceMonitor:
        """Initialize the global resource monitor."""
        if self._monitor is None:
            self._monitor = ResourceMonitor(**kwargs)
        return self._monitor

    def get_monitor(self) -> Optional[ResourceMonitor]:
        """Get the global resource monitor."""
        return self._monitor

    async def shutdown(self) -> None:
        """Shutdown the resource monitor."""
        if self._monitor:
            await self._monitor.shutdown()
            self._monitor = None


# Global resource monitor
_resource_monitor = ResourceMonitorSingleton()


def get_resource_monitor(**kwargs) -> ResourceMonitor:
    """Get or create the global resource monitor."""
    monitor = _resource_monitor.get_monitor()
    if monitor is None:
        monitor = _resource_monitor.initialize(**kwargs)
    return monitor
