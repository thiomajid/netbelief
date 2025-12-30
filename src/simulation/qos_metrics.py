"""
QoS Metrics collection and calculation for network simulation.

This module provides comprehensive Quality of Service metrics including:
- RTT (Round Trip Time)
- Jitter (packet delay variation)
- MOS (Mean Opinion Score)
- R-Factor
- Link Utilization
- Throughput Efficiency
- Congestion metrics
"""

import math
import typing as tp
from dataclasses import dataclass
from enum import Enum


class QoSCategory(Enum):
    """QoS category based on overall score."""

    EXCELLENT = "excellent"
    GOOD = "good"
    FAIR = "fair"
    POOR = "poor"
    BAD = "bad"


@dataclass
class HostQoSMetrics:
    """
    Comprehensive QoS metrics for a single host at a point in time.

    Metrics are organized into categories:
    - Traffic counters: Raw byte/packet counts
    - Bandwidth: Actual used bandwidth in various units
    - Latency: RTT and related metrics
    - Jitter: Packet delay variation
    - Utilization: Link usage as percentage of capacity
    - Loss: Packet loss and error rates
    - Voice Quality: ITU-T G.107 E-model (MOS, R-Factor)
    - Congestion: Congestion indicators and scores
    - Quality: Overall QoS assessment
    """

    timestamp: float
    host: str

    # ===== TRAFFIC COUNTERS (cumulative) =====
    rx_bytes: int = 0
    tx_bytes: int = 0
    rx_packets: int = 0
    tx_packets: int = 0

    # Error counters
    rx_errors: int = 0
    tx_errors: int = 0
    rx_dropped: int = 0
    tx_dropped: int = 0

    # ===== BANDWIDTH (actual used bandwidth per interval) =====
    # Bytes transferred in this interval
    rx_bytes_delta: int = 0
    tx_bytes_delta: int = 0

    # Bandwidth in bits per second
    rx_bandwidth_bps: float = 0.0
    tx_bandwidth_bps: float = 0.0
    total_bandwidth_bps: float = 0.0

    # Bandwidth in Kbps (more readable for lower rates)
    rx_bandwidth_kbps: float = 0.0
    tx_bandwidth_kbps: float = 0.0
    total_bandwidth_kbps: float = 0.0

    # Bandwidth in Mbps (standard unit)
    rx_bandwidth_mbps: float = 0.0
    tx_bandwidth_mbps: float = 0.0
    total_bandwidth_mbps: float = 0.0

    # ===== LATENCY METRICS (ms) =====
    rtt_ms: float = 0.0
    rtt_min_ms: float = 0.0
    rtt_max_ms: float = 0.0
    rtt_avg_ms: float = 0.0

    # ===== JITTER METRICS (ms) =====
    jitter_ms: float = 0.0
    jitter_avg_ms: float = 0.0

    # Legacy throughput fields (kept for compatibility)
    rx_throughput_mbps: float = 0.0
    tx_throughput_mbps: float = 0.0
    rx_throughput_kbps: float = 0.0
    tx_throughput_kbps: float = 0.0

    # ===== UTILIZATION (percentage of link capacity) =====
    link_capacity_mbps: float = 10.0
    rx_utilization_pct: float = 0.0
    tx_utilization_pct: float = 0.0
    avg_utilization_pct: float = 0.0

    # ===== PACKET LOSS & ERROR METRICS =====
    packet_loss_pct: float = 0.0
    packet_loss_rate: float = 0.0  # Packets lost per second
    error_rate: float = 0.0  # Error percentage

    # ===== VOICE QUALITY (ITU-T G.107 E-model) =====
    # R-Factor: 0-100, where 93.2 is perfect, <50 is unacceptable
    r_factor: float = 93.2

    # MOS (Mean Opinion Score): 1-5, where 4.41 is excellent, <3 is poor
    mos: float = 4.41

    # ===== CONGESTION METRICS =====
    # Congestion index: 0-1 normalized score (0=no congestion, 1=severe)
    congestion_index: float = 0.0

    # Congestion score: 0-100 (higher = more congested, inverse of quality)
    # Formula: 100 * congestion_index
    congestion_score: float = 0.0

    # Queue delay: estimated queuing delay in ms
    queue_delay_ms: float = 0.0

    # Buffer occupancy: estimated buffer usage percentage
    buffer_occupancy_pct: float = 0.0

    # ===== EFFICIENCY METRICS =====
    # Throughput efficiency: actual/expected ratio (0-1)
    throughput_efficiency: float = 1.0

    # Goodput ratio: useful data / total data (0-1)
    goodput_ratio: float = 1.0

    # ===== QOS CLASSIFICATION =====
    # QoS Score: 0-100 composite score
    # Formula: weighted average of MOS (35%), packet loss (25%),
    #          jitter (20%), utilization (20%)
    # Interpretation:
    #   >= 90: Excellent - Perfect for all applications
    #   75-89: Good - Suitable for VoIP and video
    #   60-74: Fair - Acceptable for web browsing
    #   40-59: Poor - Degraded experience
    #   < 40: Bad - Unusable for real-time applications
    qos_score: float = 100.0
    qos_category: str = "excellent"


@dataclass
class LinkQoSMetrics:
    """QoS metrics for a network link."""

    timestamp: float
    switch: str
    port: str

    # Traffic metrics
    rx_bytes: int = 0
    tx_bytes: int = 0
    rx_packets: int = 0
    tx_packets: int = 0
    rx_dropped: int = 0
    tx_dropped: int = 0

    # Derived metrics
    utilization_pct: float = 0.0
    throughput_mbps: float = 0.0
    packet_loss_pct: float = 0.0
    congestion_level: float = 0.0


@dataclass
class NetworkQoSSnapshot:
    """Network-wide QoS metrics at a point in time."""

    timestamp: float

    # Aggregate metrics
    total_hosts: int = 0
    active_hosts: int = 0
    total_switches: int = 0

    # Network-wide throughput
    total_rx_mbps: float = 0.0
    total_tx_mbps: float = 0.0
    avg_throughput_mbps: float = 0.0
    peak_throughput_mbps: float = 0.0

    # Network-wide latency
    avg_rtt_ms: float = 0.0
    max_rtt_ms: float = 0.0
    p95_rtt_ms: float = 0.0
    p99_rtt_ms: float = 0.0

    # Network-wide jitter
    avg_jitter_ms: float = 0.0
    max_jitter_ms: float = 0.0

    # Network-wide packet loss
    avg_packet_loss_pct: float = 0.0
    max_packet_loss_pct: float = 0.0
    total_dropped_packets: int = 0

    # Network-wide utilization
    avg_utilization_pct: float = 0.0
    max_utilization_pct: float = 0.0
    congested_links_pct: float = 0.0

    # QoS scores
    avg_mos: float = 4.41
    min_mos: float = 4.41
    avg_r_factor: float = 93.2
    min_r_factor: float = 93.2

    # Network health
    network_health_score: float = 100.0
    qos_violations: int = 0


class QoSCalculator:
    """
    Calculator for QoS metrics based on network measurements.

    Implements ITU-T G.107 E-model for MOS and R-Factor calculation.
    """

    # E-model constants
    R0 = 93.2  # Basic signal-to-noise ratio
    Is = 1.41  # Simultaneous impairment factor
    A = 0  # Advantage factor (0 for wired)

    # Codec-specific values (G.711)
    Ie = 0  # Equipment impairment factor
    Bpl = 4.3  # Packet-loss robustness factor

    @classmethod
    def calculate_r_factor(
        cls,
        delay_ms: float,
        jitter_ms: float,
        packet_loss_pct: float,
        codec_ie: float = 0,
        codec_bpl: float = 4.3,
    ) -> float:
        """
        Calculate R-Factor using ITU-T G.107 E-model.

        Args:
            delay_ms: One-way delay in milliseconds
            jitter_ms: Jitter in milliseconds
            packet_loss_pct: Packet loss percentage
            codec_ie: Codec impairment factor
            codec_bpl: Packet-loss robustness factor

        Returns:
            R-Factor value (0-100)
        """
        # Calculate delay impairment (Id)
        # Total delay including jitter buffer
        total_delay = delay_ms + jitter_ms * 2  # Approximate jitter buffer

        if total_delay < 177.3:
            Id = 0.024 * total_delay + 0.11 * (total_delay - 177.3) * (
                1 if total_delay > 177.3 else 0
            )
        else:
            Id = 0.024 * total_delay + 0.11 * (total_delay - 177.3)

        Id = max(0, Id)

        # Calculate equipment impairment (Ie-eff)
        # Accounts for packet loss
        if packet_loss_pct > 0:
            Ie_eff = codec_ie + (95 - codec_ie) * (
                packet_loss_pct / (packet_loss_pct + codec_bpl)
            )
        else:
            Ie_eff = codec_ie

        # Calculate R-Factor
        R = cls.R0 - cls.Is - Id - Ie_eff + cls.A

        # Clamp to valid range
        return max(0, min(100, R))

    @classmethod
    def calculate_mos(cls, r_factor: float) -> float:
        """
        Convert R-Factor to MOS using ITU-T G.107.

        Args:
            r_factor: R-Factor value (0-100)

        Returns:
            MOS value (1-5)
        """
        if r_factor < 0:
            return 1.0
        elif r_factor > 100:
            return 4.5

        # ITU-T G.107 formula
        if r_factor < 6.5:
            mos = 1.0
        elif r_factor > 100:
            mos = 4.5
        else:
            mos = (
                1
                + 0.035 * r_factor
                + r_factor * (r_factor - 60) * (100 - r_factor) * 7e-6
            )

        return max(1.0, min(5.0, mos))

    @classmethod
    def calculate_jitter(
        cls,
        delay_samples: tp.List[float],
    ) -> tp.Tuple[float, float]:
        """
        Calculate jitter from delay samples using RFC 3550 algorithm.

        Args:
            delay_samples: List of delay measurements

        Returns:
            Tuple of (instantaneous_jitter, average_jitter)
        """
        if len(delay_samples) < 2:
            return 0.0, 0.0

        jitter_samples = []
        for i in range(1, len(delay_samples)):
            diff = abs(delay_samples[i] - delay_samples[i - 1])
            jitter_samples.append(diff)

        if not jitter_samples:
            return 0.0, 0.0

        instant_jitter = jitter_samples[-1] if jitter_samples else 0.0
        avg_jitter = sum(jitter_samples) / len(jitter_samples)

        return instant_jitter, avg_jitter

    @classmethod
    def calculate_link_utilization(
        cls,
        throughput_bps: float,
        capacity_bps: float,
    ) -> float:
        """
        Calculate link utilization percentage.

        Args:
            throughput_bps: Current throughput in bits per second
            capacity_bps: Link capacity in bits per second

        Returns:
            Utilization percentage (0-100)
        """
        if capacity_bps <= 0:
            return 0.0

        utilization = (throughput_bps / capacity_bps) * 100
        return min(100.0, max(0.0, utilization))

    @classmethod
    def calculate_congestion_index(
        cls,
        utilization_pct: float,
        packet_loss_pct: float,
        delay_increase_pct: float = 0.0,
    ) -> float:
        """
        Calculate congestion index (0-1).

        Higher values indicate more congestion.

        Args:
            utilization_pct: Link utilization percentage
            packet_loss_pct: Packet loss percentage
            delay_increase_pct: Delay increase from baseline

        Returns:
            Congestion index (0-1)
        """
        # Weighted combination of indicators
        util_weight = 0.4
        loss_weight = 0.35
        delay_weight = 0.25

        # Normalize utilization (high utilization = congestion)
        util_factor = min(1.0, utilization_pct / 100)

        # Normalize packet loss (any loss is bad)
        loss_factor = min(1.0, packet_loss_pct / 5)  # 5% loss = max

        # Normalize delay increase
        delay_factor = min(1.0, delay_increase_pct / 100)  # 100% increase = max

        congestion = (
            util_weight * util_factor
            + loss_weight * loss_factor
            + delay_weight * delay_factor
        )

        return min(1.0, max(0.0, congestion))

    @classmethod
    def calculate_throughput_efficiency(
        cls,
        actual_throughput_bps: float,
        expected_throughput_bps: float,
    ) -> float:
        """
        Calculate throughput efficiency ratio.

        Args:
            actual_throughput_bps: Measured throughput
            expected_throughput_bps: Expected/target throughput

        Returns:
            Efficiency ratio (0-1)
        """
        if expected_throughput_bps <= 0:
            return 1.0 if actual_throughput_bps > 0 else 0.0

        efficiency = actual_throughput_bps / expected_throughput_bps
        return min(1.0, max(0.0, efficiency))

    @classmethod
    def calculate_goodput_ratio(
        cls,
        total_bytes: int,
        retransmitted_bytes: int,
        overhead_bytes: int,
    ) -> float:
        """
        Calculate goodput ratio (useful data / total data).

        Args:
            total_bytes: Total bytes transmitted
            retransmitted_bytes: Retransmitted bytes
            overhead_bytes: Protocol overhead bytes

        Returns:
            Goodput ratio (0-1)
        """
        if total_bytes <= 0:
            return 1.0

        useful_bytes = total_bytes - retransmitted_bytes - overhead_bytes
        return max(0.0, min(1.0, useful_bytes / total_bytes))

    @classmethod
    def calculate_qos_score(
        cls,
        mos: float,
        packet_loss_pct: float,
        utilization_pct: float,
        jitter_ms: float,
    ) -> tp.Tuple[float, str]:
        """
        Calculate overall QoS score and category.

        Args:
            mos: Mean Opinion Score
            packet_loss_pct: Packet loss percentage
            jitter_ms: Jitter in milliseconds
            utilization_pct: Link utilization percentage

        Returns:
            Tuple of (score 0-100, category string)
        """
        # Component scores (0-100)
        mos_score = ((mos - 1) / 4) * 100  # Convert 1-5 to 0-100
        loss_score = max(0, 100 - packet_loss_pct * 20)  # 5% loss = 0
        jitter_score = max(0, 100 - jitter_ms * 2)  # 50ms jitter = 0
        util_score = (
            100 if utilization_pct < 80 else max(0, 100 - (utilization_pct - 80) * 5)
        )

        # Weighted average
        score = (
            0.35 * mos_score
            + 0.25 * loss_score
            + 0.20 * jitter_score
            + 0.20 * util_score
        )

        # Determine category
        if score >= 90:
            category = QoSCategory.EXCELLENT.value
        elif score >= 75:
            category = QoSCategory.GOOD.value
        elif score >= 60:
            category = QoSCategory.FAIR.value
        elif score >= 40:
            category = QoSCategory.POOR.value
        else:
            category = QoSCategory.BAD.value

        return score, category


def estimate_rtt_from_traffic(
    tx_bytes: int,
    rx_bytes: int,
    tx_packets: int,
    rx_packets: int,
    baseline_delay_ms: float = 1.0,
    link_capacity_mbps: float = 10.0,
) -> float:
    """
    Estimate RTT based on traffic patterns and link characteristics.

    This is an approximation when actual RTT measurements aren't available.

    Args:
        tx_bytes: Bytes transmitted
        rx_bytes: Bytes received
        tx_packets: Packets transmitted
        rx_packets: Packets received
        baseline_delay_ms: Baseline link delay
        link_capacity_mbps: Link capacity in Mbps

    Returns:
        Estimated RTT in milliseconds
    """
    if tx_packets == 0 and rx_packets == 0:
        return baseline_delay_ms * 2  # Round trip baseline

    # Calculate average packet size
    total_packets = max(1, tx_packets + rx_packets)
    total_bytes = tx_bytes + rx_bytes
    avg_packet_size = total_bytes / total_packets

    # Estimate queuing delay based on traffic load
    throughput_mbps = (total_bytes * 8) / 1_000_000  # Rough estimate
    utilization = (
        min(1.0, throughput_mbps / link_capacity_mbps) if link_capacity_mbps > 0 else 0
    )

    # Queuing delay increases exponentially with utilization (M/M/1 model approximation)
    if utilization < 0.95:
        queue_delay = baseline_delay_ms * (utilization / (1 - utilization + 0.01))
    else:
        queue_delay = baseline_delay_ms * 20  # High congestion

    # Total RTT estimate
    rtt = (baseline_delay_ms * 2) + queue_delay

    return max(baseline_delay_ms, min(1000, rtt))  # Cap at 1 second


def estimate_jitter_from_variance(
    packet_counts: tp.List[int],
    time_interval: float = 0.5,
) -> float:
    """
    Estimate jitter from packet count variance over time.

    Args:
        packet_counts: List of packet counts per interval
        time_interval: Time interval in seconds

    Returns:
        Estimated jitter in milliseconds
    """
    if len(packet_counts) < 2:
        return 0.0

    # Calculate variance in inter-packet arrival
    diffs = [
        abs(packet_counts[i] - packet_counts[i - 1])
        for i in range(1, len(packet_counts))
    ]

    if not diffs:
        return 0.0

    avg_diff = sum(diffs) / len(diffs)
    variance = sum((d - avg_diff) ** 2 for d in diffs) / len(diffs)

    # Convert to jitter estimate (ms)
    # Higher variance in packet counts suggests higher jitter
    jitter_ms = math.sqrt(variance) * time_interval * 10  # Scale factor

    return min(100, jitter_ms)  # Cap at 100ms
