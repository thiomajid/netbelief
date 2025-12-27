"""
Dataclass configuration for VoIP network simulation.
"""

from dataclasses import dataclass, field
from typing import List, Optional


@dataclass
class TopologyConfig:
    """Configuration for network topology."""

    # Module name in the pytopo package (e.g., "airtel" for pytopo/airtel.py)
    module_name: str = "airtel"
    # Class name to import from the module
    class_name: str = "GeneratedTopo"


@dataclass
class TrafficConfig:
    """Configuration for traffic generation."""

    # Bandwidth for VoIP traffic (approx G.711 + overhead)
    voip_bandwidth: str = "100k"
    # Number of VoIP flow pairs
    voip_flow_count: int = 5
    # Number of background TCP flows
    background_traffic_count: int = 5


@dataclass
class SimulationConfig:
    """Main simulation configuration."""

    # Simulation duration in seconds
    duration: int = 30
    # Interval for data collection/polling in seconds (affects switch stats, iperf, ping)
    polling_interval: float = 0.2
    # Output directory for logs and data
    output_dir: str = "dumps"
    # STP convergence wait time in seconds
    stp_convergence_time: int = 15
    # Random seed for reproducibility (None for random)
    seed: Optional[int] = None
    # Link capacity in bits per second (used for utilization calculation)
    link_capacity_bps: int = 10_000_000  # 10 Mbps default


# Network variability configuration dataclasses


@dataclass
class LinkFailureConfig:
    """Configuration for link failure events."""

    enabled: bool = False
    # Probability of failure when checked
    probability: float = 0.05
    # Interval range between failure checks (seconds)
    min_interval: float = 60.0
    max_interval: float = 300.0
    # Duration range for failures (seconds)
    min_duration: float = 10.0
    max_duration: float = 60.0
    # Maximum fraction of links that can fail concurrently
    max_concurrent_failures: float = 0.1


@dataclass
class DelayVariationConfig:
    """Configuration for delay variation events."""

    enabled: bool = False
    # Probability of delay change when checked
    probability: float = 0.1
    # Interval range between checks (seconds)
    min_interval: float = 30.0
    max_interval: float = 120.0
    # Delay multiplier range (1.0 = original delay)
    min_multiplier: float = 0.8
    max_multiplier: float = 2.0
    # Time before delay reverts to original (None = no auto-revert)
    revert_after: Optional[float] = None


@dataclass
class BandwidthVariationConfig:
    """Configuration for bandwidth variation events."""

    enabled: bool = False
    # Probability of bandwidth change when checked
    probability: float = 0.1
    # Interval range between checks (seconds)
    min_interval: float = 30.0
    max_interval: float = 120.0
    # Bandwidth fraction range (1.0 = original bandwidth)
    min_fraction: float = 0.3
    max_fraction: float = 1.0
    # Time before bandwidth reverts to original (None = no auto-revert)
    revert_after: Optional[float] = None


@dataclass
class PacketLossConfig:
    """Configuration for packet loss variation events."""

    enabled: bool = False
    # Probability of loss change when checked
    probability: float = 0.08
    # Interval range between checks (seconds)
    min_interval: float = 45.0
    max_interval: float = 180.0
    # Packet loss percentage range
    min_loss: float = 0.0
    max_loss: float = 5.0
    # Time before loss reverts to zero (None = no auto-revert)
    revert_after: Optional[float] = None


@dataclass
class CongestionConfig:
    """Configuration for network congestion events."""

    enabled: bool = False
    # Probability of congestion when checked
    probability: float = 0.05
    # Interval range between checks (seconds)
    min_interval: float = 120.0
    max_interval: float = 600.0
    # Duration range for congestion events (seconds)
    min_duration: float = 30.0
    max_duration: float = 180.0
    # Congestion effects
    bandwidth_reduction: float = 0.4  # Fraction of original bandwidth
    additional_delay_ms: float = 50.0  # Added delay in milliseconds
    loss_percent: float = 2.0  # Packet loss percentage


@dataclass
class VariabilityConfig:
    """Master configuration for network variability features."""

    # Global enable/disable
    enabled: bool = False
    # Random seed for reproducibility (None = random)
    seed: Optional[int] = None
    # Link scope: "all" (backbone + access), "backbone" (switch-switch), "access" (switch-host)
    link_scope: str = "all"
    # Links to exclude from variability events (list of [node1, node2] pairs)
    excluded_links: List[List[str]] = field(default_factory=list)
    # Specific links to target (empty = use link_scope)
    target_links: List[List[str]] = field(default_factory=list)

    # Default values for links without QoS parameters (e.g., access links)
    # Set to null/None to skip variability on links without original values
    default_bandwidth: Optional[float] = 10.0  # Mbps
    default_delay: Optional[str] = "1ms"  # Latency

    # Feature-specific configurations
    link_failure: LinkFailureConfig = field(default_factory=LinkFailureConfig)
    delay_variation: DelayVariationConfig = field(default_factory=DelayVariationConfig)
    bandwidth_variation: BandwidthVariationConfig = field(
        default_factory=BandwidthVariationConfig
    )
    packet_loss: PacketLossConfig = field(default_factory=PacketLossConfig)
    congestion: CongestionConfig = field(default_factory=CongestionConfig)


@dataclass
class VoIPSimulationConfig:
    """Root configuration for VoIP simulation."""

    topology: TopologyConfig = field(default_factory=TopologyConfig)
    traffic: TrafficConfig = field(default_factory=TrafficConfig)
    simulation: SimulationConfig = field(default_factory=SimulationConfig)
    variability: VariabilityConfig = field(default_factory=VariabilityConfig)
