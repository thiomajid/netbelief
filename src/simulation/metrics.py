"""
Data classes for VoIP network simulation metrics.

This module contains strongly-typed data classes representing various
network metrics collected during simulation. Each class is fully documented
with field-level docstrings for clarity and maintainability.
"""

from dataclasses import dataclass, field
from typing import List, Optional


@dataclass
class IperfMetrics:
    """
    Metrics collected from iperf UDP traffic measurements.

    These metrics represent the core VoIP traffic characteristics
    measured during each reporting interval.
    """

    timestamp: float
    """Relative timestamp in seconds from simulation start."""

    flow_id: str
    """Unique identifier for the traffic flow (e.g., 'h1->h2')."""

    interval_start: float
    """Start time of the measurement interval in seconds."""

    interval_end: float
    """End time of the measurement interval in seconds."""

    bandwidth_bps: float
    """Measured bandwidth in bits per second."""

    jitter_ms: float
    """Jitter (variation in packet delay) in milliseconds."""

    packet_loss_cnt: int
    """Number of packets lost during the interval."""

    total_packets: int
    """Total number of packets transmitted during the interval."""

    packet_loss_percent: float
    """Packet loss as a percentage of total packets."""


@dataclass
class PingMetrics:
    """
    Metrics collected from ping/ICMP measurements.

    Used for measuring round-trip time (RTT) and detecting
    network connectivity issues.
    """

    timestamp: float
    """Relative timestamp in seconds from simulation start."""

    abs_timestamp: float
    """Absolute Unix timestamp of the measurement."""

    flow_id: str
    """Unique identifier for the traffic flow."""

    rtt_ms: float
    """Round-trip time in milliseconds."""

    icmp_seq: Optional[int] = None
    """ICMP sequence number for packet ordering analysis."""

    ttl: Optional[int] = None
    """Time-to-live value from the response."""


@dataclass
class SwitchPortStats:
    """
    Statistics for a single switch port.

    These are raw counters from OVS switch ports that need to be
    differenced between intervals to get per-interval metrics.
    """

    timestamp: float
    """Timestamp when the stats were collected."""

    switch: str
    """Name of the switch (e.g., 's1')."""

    port: str
    """Port identifier on the switch."""

    rx_pkts: int
    """Total received packets (cumulative counter)."""

    rx_bytes: int
    """Total received bytes (cumulative counter)."""

    rx_drop: int
    """Total received packets dropped (cumulative counter)."""

    rx_errs: int
    """Total receive errors (cumulative counter)."""

    tx_pkts: int
    """Total transmitted packets (cumulative counter)."""

    tx_bytes: int
    """Total transmitted bytes (cumulative counter)."""

    tx_drop: int
    """Total transmitted packets dropped (cumulative counter)."""

    tx_errs: int
    """Total transmit errors (cumulative counter)."""


@dataclass
class AggregatedSwitchStats:
    """
    Aggregated switch statistics across all ports.

    These are delta values (differences between intervals) summed
    across all switch ports, representing network-wide statistics.
    """

    net_rx_pkts: int = 0
    """Total received packets in the interval."""

    net_rx_bytes: int = 0
    """Total received bytes in the interval."""

    net_rx_drop: int = 0
    """Total received packets dropped in the interval."""

    net_rx_errs: int = 0
    """Total receive errors in the interval."""

    net_tx_pkts: int = 0
    """Total transmitted packets in the interval."""

    net_tx_bytes: int = 0
    """Total transmitted bytes in the interval."""

    net_tx_drop: int = 0
    """Total transmitted packets dropped in the interval."""

    net_tx_errs: int = 0
    """Total transmit errors in the interval."""


@dataclass
class TcpdumpPacketInfo:
    """
    Information extracted from a single captured packet via tcpdump.

    Provides deep packet inspection data for advanced analysis.
    """

    timestamp: float
    """Packet capture timestamp (Unix epoch)."""

    src_ip: str
    """Source IP address."""

    dst_ip: str
    """Destination IP address."""

    src_port: Optional[int] = None
    """Source port number (for TCP/UDP)."""

    dst_port: Optional[int] = None
    """Destination port number (for TCP/UDP)."""

    protocol: str = "UDP"
    """Protocol type (TCP, UDP, ICMP, etc.)."""

    length: int = 0
    """Packet length in bytes."""

    ttl: Optional[int] = None
    """Time-to-live value from IP header."""

    tcp_flags: Optional[str] = None
    """TCP flags if applicable (SYN, ACK, FIN, etc.)."""

    tcp_seq: Optional[int] = None
    """TCP sequence number."""

    tcp_ack: Optional[int] = None
    """TCP acknowledgment number."""

    tcp_window: Optional[int] = None
    """TCP window size."""

    udp_checksum: Optional[str] = None
    """UDP checksum value."""

    icmp_type: Optional[int] = None
    """ICMP message type."""

    icmp_code: Optional[int] = None
    """ICMP message code."""

    rtp_payload_type: Optional[int] = None
    """RTP payload type for VoIP analysis."""

    rtp_sequence: Optional[int] = None
    """RTP sequence number for VoIP packet ordering."""

    rtp_timestamp: Optional[int] = None
    """RTP timestamp for jitter calculation."""

    rtp_ssrc: Optional[str] = None
    """RTP synchronization source identifier."""


@dataclass
class TcpdumpFlowStats:
    """
    Aggregated statistics for a traffic flow from tcpdump capture.

    These metrics provide deeper insight than iperf alone, including
    inter-packet timing for burstiness analysis.
    """

    flow_id: str
    """Unique identifier for the traffic flow."""

    total_packets: int = 0
    """Total packets captured for this flow."""

    total_bytes: int = 0
    """Total bytes captured for this flow."""

    start_time: float = 0.0
    """First packet timestamp."""

    end_time: float = 0.0
    """Last packet timestamp."""

    min_packet_size: int = 0
    """Minimum packet size in bytes."""

    max_packet_size: int = 0
    """Maximum packet size in bytes."""

    avg_packet_size: float = 0.0
    """Average packet size in bytes."""

    inter_packet_times: List[float] = field(default_factory=list)
    """List of inter-packet arrival times in seconds."""

    retransmissions: int = 0
    """Number of detected TCP retransmissions."""

    out_of_order_packets: int = 0
    """Number of out-of-order packets detected."""

    duplicate_packets: int = 0
    """Number of duplicate packets detected."""

    fragmented_packets: int = 0
    """Number of IP fragmented packets."""

    rtp_packets: int = 0
    """Number of RTP packets (VoIP-specific)."""

    rtp_jitter_samples: List[float] = field(default_factory=list)
    """RTP inter-arrival jitter samples for analysis."""


@dataclass
class DerivedMetrics:
    """
    Derived/computed metrics for SDN decision making.

    These metrics are calculated from raw measurements and provide
    actionable insights for network optimization.
    """

    link_utilization_percent: float = 0.0
    """
    Link utilization as percentage of capacity.
    Formula: (bandwidth_bps / link_capacity_bps) × 100
    Critical for SDN routing decisions.
    """

    packets_per_second: float = 0.0
    """
    Packet rate in packets per second.
    Formula: total_packets / interval_duration
    Useful for detecting traffic spikes.
    """

    drop_rate_percent: float = 0.0
    """
    Packet drop rate as percentage.
    Formula: (rx_drop + tx_drop) / (rx_pkts + tx_pkts) × 100
    Indicates buffer overflow or congestion.
    """

    error_rate_percent: float = 0.0
    """
    Error rate as percentage.
    Formula: (rx_errs + tx_errs) / (rx_pkts + tx_pkts) × 100
    Indicates physical layer problems.
    """

    one_way_delay_ms: float = 0.0
    """
    Estimated one-way delay in milliseconds.
    Formula: rtt_ms / 2 (approximation)
    More accurate for MOS calculation.
    """

    congestion_score: float = 0.0
    """
    Weighted congestion indicator (0-1 scale).
    Formula: 0.4×util_norm + 0.4×loss_norm + 0.2×jitter_norm
    Single metric capturing overall network congestion.
    Higher values indicate more congestion.
    """

    burstiness_score: float = 0.0
    """
    Traffic burstiness indicator.
    Calculated as standard deviation of inter-packet arrival times.
    Higher values indicate more bursty traffic patterns.
    """

    throughput_efficiency: float = 0.0
    """
    Ratio of goodput to total throughput.
    Formula: (total_packets - lost_packets) / total_packets
    Indicates how efficiently bandwidth is being used.
    """


@dataclass
class TcpdumpDerivedMetrics:
    """
    Advanced metrics derived from tcpdump packet capture analysis.

    These provide deeper network insights not available from iperf alone.
    """

    avg_inter_packet_time_ms: float = 0.0
    """Average time between consecutive packets in milliseconds."""

    std_inter_packet_time_ms: float = 0.0
    """Standard deviation of inter-packet times (burstiness indicator)."""

    packet_size_variance: float = 0.0
    """Variance in packet sizes (indicates traffic heterogeneity)."""

    retransmission_rate_percent: float = 0.0
    """TCP retransmission rate as percentage."""

    out_of_order_rate_percent: float = 0.0
    """Rate of out-of-order packet arrivals."""

    duplicate_rate_percent: float = 0.0
    """Rate of duplicate packets (indicates routing issues)."""

    fragmentation_rate_percent: float = 0.0
    """IP fragmentation rate (indicates MTU issues)."""

    rtp_jitter_ms: float = 0.0
    """RTP-calculated jitter in milliseconds (more accurate for VoIP)."""

    rtp_loss_rate_percent: float = 0.0
    """RTP packet loss rate based on sequence number gaps."""

    flow_duration_seconds: float = 0.0
    """Total duration of the captured flow."""

    bytes_per_second: float = 0.0
    """Average throughput in bytes per second."""


@dataclass
class QoSMetrics:
    """
    Quality of Service metrics for VoIP analysis.

    These metrics are specifically designed for voice quality assessment
    following ITU-T recommendations.
    """

    mos: float = 1.0
    """
    Mean Opinion Score (1-5 scale).
    Calculated using E-model approximation (ITU-T G.107).
    5 = Excellent, 4 = Good, 3 = Fair, 2 = Poor, 1 = Bad
    """

    r_factor: float = 0.0
    """
    R-factor from E-model (0-100 scale).
    Transmission rating factor used to calculate MOS.
    """

    delay_impairment: float = 0.0
    """
    Delay impairment factor (Id) from E-model.
    Represents quality degradation due to delay.
    """

    equipment_impairment: float = 0.0
    """
    Equipment impairment factor (Ie) from E-model.
    Represents quality degradation due to codec and packet loss.
    """

    effective_latency_ms: float = 0.0
    """
    Effective end-to-end latency in milliseconds.
    Includes propagation, processing, and buffering delays.
    """

    jitter_buffer_delay_ms: float = 0.0
    """
    Estimated jitter buffer delay in milliseconds.
    Typically 2-3x the measured jitter.
    """


@dataclass
class CombinedMetrics:
    """
    Combined metrics record for a single measurement interval.

    This is the primary data structure for CSV export, containing
    all collected and derived metrics in a single record.
    """

    # Timing and identification
    timestamp: float = 0.0
    """Relative timestamp from simulation start."""

    flow_id: str = ""
    """Unique flow identifier."""

    interval_start: float = 0.0
    """Interval start time."""

    interval_end: float = 0.0
    """Interval end time."""

    # Core iperf metrics
    bandwidth_bps: float = 0.0
    """Measured bandwidth in bits per second."""

    jitter_ms: float = 0.0
    """Jitter in milliseconds."""

    packet_loss_cnt: int = 0
    """Number of packets lost."""

    total_packets: int = 0
    """Total packets in interval."""

    packet_loss_percent: float = 0.0
    """Packet loss percentage."""

    # RTT from ping
    rtt_ms: float = 0.0
    """Round-trip time in milliseconds."""

    # QoS metrics
    mos: float = 1.0
    """Mean Opinion Score."""

    r_factor: float = 0.0
    """E-model R-factor."""

    # Switch statistics
    net_rx_pkts: int = 0
    """Network received packets."""

    net_rx_bytes: int = 0
    """Network received bytes."""

    net_rx_drop: int = 0
    """Network received drops."""

    net_rx_errs: int = 0
    """Network receive errors."""

    net_tx_pkts: int = 0
    """Network transmitted packets."""

    net_tx_bytes: int = 0
    """Network transmitted bytes."""

    net_tx_drop: int = 0
    """Network transmit drops."""

    net_tx_errs: int = 0
    """Network transmit errors."""

    # Derived metrics
    link_utilization_percent: float = 0.0
    """Link utilization percentage."""

    packets_per_second: float = 0.0
    """Packets per second."""

    drop_rate_percent: float = 0.0
    """Drop rate percentage."""

    error_rate_percent: float = 0.0
    """Error rate percentage."""

    one_way_delay_ms: float = 0.0
    """One-way delay estimate."""

    congestion_score: float = 0.0
    """Congestion score (0-1)."""

    burstiness_score: float = 0.0
    """Burstiness score."""

    throughput_efficiency: float = 0.0
    """Throughput efficiency ratio."""

    # Tcpdump-derived metrics
    avg_inter_packet_time_ms: float = 0.0
    """Average inter-packet time."""

    std_inter_packet_time_ms: float = 0.0
    """Std dev of inter-packet time."""

    packet_size_variance: float = 0.0
    """Packet size variance."""

    retransmission_rate_percent: float = 0.0
    """Retransmission rate."""

    out_of_order_rate_percent: float = 0.0
    """Out-of-order packet rate."""

    bytes_per_second: float = 0.0
    """Bytes per second throughput."""
