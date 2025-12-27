"""
Metrics calculation utilities for VoIP network simulation.

This module provides functions for calculating derived metrics,
QoS scores, and aggregated statistics from raw measurements.
"""

import math
import typing as tp

from src.simulation.metrics import (
    AggregatedSwitchStats,
    CombinedMetrics,
    DerivedMetrics,
    IperfMetrics,
    PingMetrics,
    QoSMetrics,
    TcpdumpDerivedMetrics,
    TcpdumpFlowStats,
)


def calculate_mos(latency_ms: float, packet_loss_percent: float) -> QoSMetrics:
    """
    Calculate Mean Opinion Score (MOS) using E-model approximation.

    Based on ITU-T G.107 recommendation for voice quality estimation.

    Args:
        latency_ms: One-way delay in milliseconds
        packet_loss_percent: Packet loss percentage

    Returns:
        QoSMetrics object with MOS and related values
    """
    d = latency_ms

    # Id (Delay impairment factor)
    if d < 177.3:
        Id = 0.024 * d
    else:
        Id = 0.024 * d + 0.11 * (d - 177.3)

    # Ie (Equipment impairment - based on packet loss)
    p = packet_loss_percent / 100.0
    Ie = 30 * math.log(1 + 15 * p) if p > 0 else 0

    # R factor calculation
    R = max(0, min(100, 93.2 - Id - Ie))

    # MOS mapping from R factor
    if R < 0:
        MOS = 1.0
    elif R > 100:
        MOS = 4.5
    else:
        MOS = 1 + 0.035 * R + R * (R - 60) * (100 - R) * 7e-6

    MOS = max(1.0, min(5.0, MOS))

    # Estimate jitter buffer delay (typically 2-3x jitter)
    jitter_buffer = 0.0  # Will be calculated if jitter is available

    return QoSMetrics(
        mos=MOS,
        r_factor=R,
        delay_impairment=Id,
        equipment_impairment=Ie,
        effective_latency_ms=d,
        jitter_buffer_delay_ms=jitter_buffer,
    )


def calculate_derived_metrics(
    bandwidth_bps: float,
    link_capacity_bps: float,
    total_packets: int,
    interval_duration: float,
    switch_stats: AggregatedSwitchStats,
    jitter_ms: float,
    packet_loss_percent: float,
    rtt_ms: float,
    inter_packet_times: tp.Optional[tp.List[float]] = None,
) -> DerivedMetrics:
    """
    Calculate derived metrics for SDN decision making.

    Args:
        bandwidth_bps: Current bandwidth in bits per second
        link_capacity_bps: Link capacity in bits per second
        total_packets: Total packets in the interval
        interval_duration: Duration of the measurement interval in seconds
        switch_stats: Aggregated switch statistics for the interval
        jitter_ms: Jitter in milliseconds
        packet_loss_percent: Packet loss percentage
        rtt_ms: Round-trip time in milliseconds
        inter_packet_times: List of inter-packet arrival times (optional)

    Returns:
        DerivedMetrics object with calculated values
    """
    metrics = DerivedMetrics()

    # Link utilization percentage
    if link_capacity_bps > 0:
        metrics.link_utilization_percent = (bandwidth_bps / link_capacity_bps) * 100

    # Packets per second
    if interval_duration > 0:
        metrics.packets_per_second = total_packets / interval_duration

    # Drop rate percentage (from switch stats)
    total_pkts = switch_stats.net_rx_pkts + switch_stats.net_tx_pkts
    total_drops = switch_stats.net_rx_drop + switch_stats.net_tx_drop
    if total_pkts > 0:
        metrics.drop_rate_percent = (total_drops / total_pkts) * 100

    # Error rate percentage (from switch stats)
    total_errs = switch_stats.net_rx_errs + switch_stats.net_tx_errs
    if total_pkts > 0:
        metrics.error_rate_percent = (total_errs / total_pkts) * 100

    # One-way delay approximation (RTT / 2)
    metrics.one_way_delay_ms = rtt_ms / 2.0

    # Normalized values for congestion score (0-1 range)
    util_norm = min(metrics.link_utilization_percent / 100.0, 1.0)
    loss_norm = min(packet_loss_percent / 10.0, 1.0)  # 10% = max
    jitter_norm = min(jitter_ms / 50.0, 1.0)  # 50ms = max

    # Congestion score: weighted combination (higher = more congested)
    metrics.congestion_score = 0.4 * util_norm + 0.4 * loss_norm + 0.2 * jitter_norm

    # Burstiness score from inter-packet times
    if inter_packet_times and len(inter_packet_times) > 1:
        mean_ipt = sum(inter_packet_times) / len(inter_packet_times)
        variance = sum((t - mean_ipt) ** 2 for t in inter_packet_times) / len(
            inter_packet_times
        )
        metrics.burstiness_score = math.sqrt(variance)

    # Throughput efficiency
    if total_packets > 0:
        lost = int(total_packets * packet_loss_percent / 100)
        metrics.throughput_efficiency = (total_packets - lost) / total_packets

    return metrics


def calculate_tcpdump_derived_metrics(
    flow_stats: TcpdumpFlowStats,
) -> TcpdumpDerivedMetrics:
    """
    Calculate derived metrics from tcpdump flow statistics.

    Args:
        flow_stats: Aggregated flow statistics from tcpdump

    Returns:
        TcpdumpDerivedMetrics object
    """
    metrics = TcpdumpDerivedMetrics()

    if not flow_stats or flow_stats.total_packets == 0:
        return metrics

    # Inter-packet timing metrics
    if flow_stats.inter_packet_times:
        ipt = flow_stats.inter_packet_times
        metrics.avg_inter_packet_time_ms = (sum(ipt) / len(ipt)) * 1000

        if len(ipt) > 1:
            mean_ipt = sum(ipt) / len(ipt)
            variance = sum((t - mean_ipt) ** 2 for t in ipt) / len(ipt)
            metrics.std_inter_packet_time_ms = math.sqrt(variance) * 1000

    # Packet size variance
    if flow_stats.avg_packet_size > 0:
        # Estimate variance from min/max
        range_size = flow_stats.max_packet_size - flow_stats.min_packet_size
        metrics.packet_size_variance = (range_size / 4) ** 2  # Approximate

    # Rate calculations
    if flow_stats.total_packets > 0:
        metrics.retransmission_rate_percent = (
            flow_stats.retransmissions / flow_stats.total_packets
        ) * 100

        metrics.out_of_order_rate_percent = (
            flow_stats.out_of_order_packets / flow_stats.total_packets
        ) * 100

        metrics.duplicate_rate_percent = (
            flow_stats.duplicate_packets / flow_stats.total_packets
        ) * 100

        metrics.fragmentation_rate_percent = (
            flow_stats.fragmented_packets / flow_stats.total_packets
        ) * 100

    # RTP jitter (if available)
    if flow_stats.rtp_jitter_samples:
        metrics.rtp_jitter_ms = sum(flow_stats.rtp_jitter_samples) / len(
            flow_stats.rtp_jitter_samples
        )

    # Flow duration and throughput
    metrics.flow_duration_seconds = flow_stats.end_time - flow_stats.start_time
    if metrics.flow_duration_seconds > 0:
        metrics.bytes_per_second = (
            flow_stats.total_bytes / metrics.flow_duration_seconds
        )

    return metrics


def merge_metrics_to_combined(
    iperf: IperfMetrics,
    ping: tp.Optional[PingMetrics],
    switch_stats: AggregatedSwitchStats,
    derived: DerivedMetrics,
    qos: QoSMetrics,
    tcpdump_derived: tp.Optional[TcpdumpDerivedMetrics] = None,
) -> CombinedMetrics:
    """
    Merge all metric types into a single CombinedMetrics record.

    Args:
        iperf: Iperf UDP metrics
        ping: Ping/RTT metrics (optional)
        switch_stats: Aggregated switch statistics
        derived: Derived/calculated metrics
        qos: QoS metrics (MOS, R-factor, etc.)
        tcpdump_derived: Tcpdump-derived metrics (optional)

    Returns:
        CombinedMetrics object with all fields populated
    """
    combined = CombinedMetrics(
        # Timing and identification
        timestamp=iperf.timestamp,
        flow_id=iperf.flow_id,
        interval_start=iperf.interval_start,
        interval_end=iperf.interval_end,
        # Core iperf metrics
        bandwidth_bps=iperf.bandwidth_bps,
        jitter_ms=iperf.jitter_ms,
        packet_loss_cnt=iperf.packet_loss_cnt,
        total_packets=iperf.total_packets,
        packet_loss_percent=iperf.packet_loss_percent,
        # RTT from ping
        rtt_ms=ping.rtt_ms if ping else 0.0,
        # QoS metrics
        mos=qos.mos,
        r_factor=qos.r_factor,
        # Switch statistics
        net_rx_pkts=switch_stats.net_rx_pkts,
        net_rx_bytes=switch_stats.net_rx_bytes,
        net_rx_drop=switch_stats.net_rx_drop,
        net_rx_errs=switch_stats.net_rx_errs,
        net_tx_pkts=switch_stats.net_tx_pkts,
        net_tx_bytes=switch_stats.net_tx_bytes,
        net_tx_drop=switch_stats.net_tx_drop,
        net_tx_errs=switch_stats.net_tx_errs,
        # Derived metrics
        link_utilization_percent=derived.link_utilization_percent,
        packets_per_second=derived.packets_per_second,
        drop_rate_percent=derived.drop_rate_percent,
        error_rate_percent=derived.error_rate_percent,
        one_way_delay_ms=derived.one_way_delay_ms,
        congestion_score=derived.congestion_score,
        burstiness_score=derived.burstiness_score,
        throughput_efficiency=derived.throughput_efficiency,
    )

    # Add tcpdump-derived metrics if available
    if tcpdump_derived:
        combined.avg_inter_packet_time_ms = tcpdump_derived.avg_inter_packet_time_ms
        combined.std_inter_packet_time_ms = tcpdump_derived.std_inter_packet_time_ms
        combined.packet_size_variance = tcpdump_derived.packet_size_variance
        combined.retransmission_rate_percent = (
            tcpdump_derived.retransmission_rate_percent
        )
        combined.out_of_order_rate_percent = tcpdump_derived.out_of_order_rate_percent
        combined.bytes_per_second = tcpdump_derived.bytes_per_second

    return combined


def find_closest_ping(
    pings: tp.List[PingMetrics],
    target_timestamp: float,
    max_diff: float = 2.0,
) -> tp.Optional[PingMetrics]:
    """
    Find the ping measurement closest to a target timestamp.

    Args:
        pings: List of ping measurements
        target_timestamp: Target timestamp to match
        max_diff: Maximum allowed time difference in seconds

    Returns:
        Closest PingMetrics or None if no match within threshold
    """
    if not pings:
        return None

    closest = min(pings, key=lambda x: abs(x.timestamp - target_timestamp))

    if abs(closest.timestamp - target_timestamp) <= max_diff:
        return closest

    return None
