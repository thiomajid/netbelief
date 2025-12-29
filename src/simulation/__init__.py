"""
VoIP Network Simulation Package.

This package provides components for simulating VoIP traffic
over Mininet networks and collecting comprehensive metrics.

Modules:
    metrics: Data classes for all collected and derived metrics
    parsers: Log file parsers for iperf, ping, switch stats, and tcpdump
    collectors: Background data collection utilities
    calculations: Metric calculation and aggregation functions
    network: Network setup and topology utilities
    processor: Log processing and CSV export
"""

from src.simulation.calculations import (
    calculate_derived_metrics,
    calculate_mos,
    calculate_tcpdump_derived_metrics,
    find_closest_ping,
    merge_metrics_to_combined,
)
from src.simulation.collectors import (
    DataCollectorManager,
    collect_switch_stats,
    start_all_tcpdump_captures,
    start_tcpdump_on_host,
    stop_tcpdump_captures,
)
from src.simulation.metrics import (
    AggregatedSwitchStats,
    CombinedMetrics,
    DerivedMetrics,
    IperfMetrics,
    PingMetrics,
    QoSMetrics,
    SwitchPortStats,
    TcpdumpDerivedMetrics,
    TcpdumpFlowStats,
    TcpdumpPacketInfo,
)
from src.simulation.network import (
    cleanup_network,
    ensure_dir,
    load_topology_class,
    setup_background_pairs,
    setup_voip_pairs,
)
from src.simulation.parsers import (
    aggregate_switch_stats,
    aggregate_tcpdump_flow_stats,
    convert_to_bps,
    parse_all_switch_stats,
    parse_iperf_udp_line,
    parse_ping_line,
    parse_switch_port_stats,
    parse_switch_stats_file,
    parse_tcpdump_file,
    parse_tcpdump_packet,
)
from src.simulation.processor import (
    export_to_csv,
    process_and_export,
    process_logs,
    process_ping_logs,
    process_tcpdump_logs,
)
from src.simulation.traffic import TrafficManager, TrafficType
from src.simulation.variability import (
    BandwidthVariationConfig,
    CongestionConfig,
    DelayVariationConfig,
    EventType,
    LinkFailureConfig,
    LinkScope,
    LinkState,
    NetworkEvent,
    NetworkVariabilityManager,
    PacketLossConfig,
    VariabilityConfig,
)

__all__ = [
    # Metrics data classes
    "IperfMetrics",
    "PingMetrics",
    "SwitchPortStats",
    "AggregatedSwitchStats",
    "TcpdumpPacketInfo",
    "TcpdumpFlowStats",
    "DerivedMetrics",
    "TcpdumpDerivedMetrics",
    "QoSMetrics",
    "CombinedMetrics",
    # Parsers
    "convert_to_bps",
    "parse_iperf_udp_line",
    "parse_ping_line",
    "parse_switch_port_stats",
    "parse_switch_stats_file",
    "parse_all_switch_stats",
    "aggregate_switch_stats",
    "parse_tcpdump_packet",
    "parse_tcpdump_file",
    "aggregate_tcpdump_flow_stats",
    # Collectors
    "collect_switch_stats",
    "start_tcpdump_on_host",
    "start_all_tcpdump_captures",
    "stop_tcpdump_captures",
    "DataCollectorManager",
    # Calculations
    "calculate_mos",
    "calculate_derived_metrics",
    "calculate_tcpdump_derived_metrics",
    "merge_metrics_to_combined",
    "find_closest_ping",
    # Network utilities
    "load_topology_class",
    "ensure_dir",
    "setup_voip_pairs",
    "setup_background_pairs",
    "cleanup_network",
    # Processor
    "process_ping_logs",
    "process_tcpdump_logs",
    "process_logs",
    "export_to_csv",
    "process_and_export",
    # Variability
    "NetworkVariabilityManager",
    "VariabilityConfig",
    "LinkFailureConfig",
    "DelayVariationConfig",
    "BandwidthVariationConfig",
    "PacketLossConfig",
    "CongestionConfig",
    "NetworkEvent",
    "EventType",
    "LinkState",
    "LinkScope",
]
