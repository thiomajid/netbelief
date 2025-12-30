"""
Network simulation modules for realistic traffic generation and telemetry collection.

This package provides components for:
- Realistic traffic generation with human-like behavior patterns
- Traffic profiles for various application types
- Network variability simulation
- Data collection and processing
"""

from src.simulation.collectors import DataCollectorManager
from src.simulation.network import (
    cleanup_network,
    ensure_dir,
    load_topology_class,
    setup_background_pairs,
    setup_voip_pairs,
)
from src.simulation.processor import process_and_export
from src.simulation.qos_metrics import (
    HostQoSMetrics,
    LinkQoSMetrics,
    NetworkQoSSnapshot,
    QoSCalculator,
    QoSCategory,
)
from src.simulation.qos_processor import QoSDataProcessor, process_qos_metrics
from src.simulation.realistic_processor import process_and_export_realistic
from src.simulation.realistic_traffic import RealisticTrafficGenerator
from src.simulation.traffic import TrafficManager, TrafficType
from src.simulation.traffic_profiles import (
    TRAFFIC_PROFILES,
    USER_PROFILES,
    ApplicationType,
    TrafficProfile,
    UserBehaviorProfile,
)
from src.simulation.variability import (
    NetworkVariabilityManager,
    VariabilityConfig,
)

__all__ = [
    "ApplicationType",
    "DataCollectorManager",
    "HostQoSMetrics",
    "LinkQoSMetrics",
    "NetworkQoSSnapshot",
    "NetworkVariabilityManager",
    "QoSCalculator",
    "QoSCategory",
    "QoSDataProcessor",
    "RealisticTrafficGenerator",
    "TrafficManager",
    "TrafficProfile",
    "TrafficType",
    "TRAFFIC_PROFILES",
    "USER_PROFILES",
    "UserBehaviorProfile",
    "VariabilityConfig",
    "cleanup_network",
    "ensure_dir",
    "load_topology_class",
    "process_and_export",
    "process_qos_metrics",
    "process_and_export_realistic",
    "setup_background_pairs",
    "setup_voip_pairs",
]
