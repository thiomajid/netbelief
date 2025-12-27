"""
Log processing and CSV export for VoIP simulation.

This module handles the processing of collected logs and
generation of the final CSV output file.
"""

import csv
import os
import typing as tp
from dataclasses import asdict

from loguru import logger
from mininet.node import Node

from src.simulation.calculations import (
    calculate_derived_metrics,
    calculate_mos,
    calculate_tcpdump_derived_metrics,
    find_closest_ping,
    merge_metrics_to_combined,
)
from src.simulation.metrics import (
    AggregatedSwitchStats,
    CombinedMetrics,
    PingMetrics,
)
from src.simulation.parsers import (
    aggregate_switch_stats,
    aggregate_tcpdump_flow_stats,
    parse_all_switch_stats,
    parse_iperf_udp_line,
    parse_ping_line,
    parse_tcpdump_file,
)


def process_ping_logs(
    voip_pairs: tp.List[tp.Tuple[Node, Node]],
    output_dir: str,
) -> tp.Dict[str, tp.List[PingMetrics]]:
    """
    Process all ping log files.

    Args:
        voip_pairs: List of VoIP host pairs
        output_dir: Directory containing log files

    Returns:
        Dictionary mapping flow_id to list of PingMetrics
    """
    ping_data: tp.Dict[str, tp.List[PingMetrics]] = {}

    for h1, h2 in voip_pairs:
        for src, dst in [(h1, h2), (h2, h1)]:
            flow_id = f"{src.name}->{dst.name}"
            log_path = os.path.join(output_dir, f"ping_{src.name}_to_{dst.name}.log")

            if not os.path.exists(log_path):
                continue

            with open(log_path, "r") as f:
                lines = f.readlines()

            # Find start reference time
            import re

            start_ref = None
            for line in lines:
                match = re.search(r"\[(\d+\.\d+)\]", line)
                if match:
                    start_ref = float(match.group(1))
                    break

            if start_ref:
                ping_data[flow_id] = []
                for line in lines:
                    parsed = parse_ping_line(line, flow_id, start_ref)
                    if parsed:
                        ping_data[flow_id].append(parsed)

    return ping_data


def process_tcpdump_logs(
    voip_pairs: tp.List[tp.Tuple[Node, Node]],
    output_dir: str,
) -> tp.Dict[str, tp.Any]:
    """
    Process all tcpdump log files.

    Args:
        voip_pairs: List of VoIP host pairs
        output_dir: Directory containing log files

    Returns:
        Dictionary mapping host name to flow statistics
    """
    tcpdump_data = {}

    # Get unique hosts
    hosts = set()
    for h1, h2 in voip_pairs:
        hosts.add(h1)
        hosts.add(h2)

    for host in hosts:
        log_path = os.path.join(output_dir, f"tcpdump_{host.name}.log")

        if not os.path.exists(log_path):
            continue

        packets = parse_tcpdump_file(log_path)
        if packets:
            # Group packets by destination
            grouped: tp.Dict[str, tp.List] = {}
            for pkt in packets:
                key = f"{pkt.src_ip}->{pkt.dst_ip}"
                if key not in grouped:
                    grouped[key] = []
                grouped[key].append(pkt)

            # Aggregate stats per flow
            for flow_key, flow_packets in grouped.items():
                stats = aggregate_tcpdump_flow_stats(flow_packets, flow_key)
                tcpdump_data[flow_key] = stats

    return tcpdump_data


def process_logs(
    voip_pairs: tp.List[tp.Tuple[Node, Node]],
    output_dir: str,
    polling_interval: float = 0.2,
    link_capacity_bps: int = 10_000_000,
) -> tp.List[CombinedMetrics]:
    """
    Process simulation logs and generate combined metrics.

    Args:
        voip_pairs: List of VoIP host pairs
        output_dir: Output directory containing logs
        polling_interval: Data collection interval in seconds
        link_capacity_bps: Link capacity for utilization calculation

    Returns:
        List of CombinedMetrics objects
    """
    logger.info("Processing logs and merging data...")

    # Process switch stats
    switch_stats_raw = parse_all_switch_stats(output_dir)
    switch_deltas = aggregate_switch_stats(switch_stats_raw)

    # Process ping logs
    ping_data = process_ping_logs(voip_pairs, output_dir)

    # Process tcpdump logs
    tcpdump_data = process_tcpdump_logs(voip_pairs, output_dir)

    # Empty switch stats for missing intervals
    empty_switch_stats = AggregatedSwitchStats()

    # Process iperf logs and merge data
    data_points: tp.List[CombinedMetrics] = []

    def process_flow(server: Node, client: Node, flow_id: str) -> None:
        log_path = os.path.join(output_dir, f"iperf_server_{server.name}.log")
        if not os.path.exists(log_path):
            return

        with open(log_path, "r") as f:
            lines = f.read().split("\n")

        for line in lines:
            iperf = parse_iperf_udp_line(line, flow_id, 0)
            if not iperf:
                continue

            # Find matching RTT measurement
            rtt = 0.0
            ping = None
            if flow_id in ping_data and ping_data[flow_id]:
                ping = find_closest_ping(ping_data[flow_id], iperf.timestamp)
                if ping:
                    rtt = ping.rtt_ms

            # Get switch stats for this interval
            rel_sec = int(round(iperf.timestamp))
            s_stats = switch_deltas.get(rel_sec, empty_switch_stats)

            # Calculate interval duration
            interval_duration = iperf.interval_end - iperf.interval_start
            if interval_duration <= 0:
                interval_duration = polling_interval

            # Calculate QoS metrics
            qos = calculate_mos(rtt / 2.0, iperf.packet_loss_percent)

            # Get inter-packet times from tcpdump if available
            inter_packet_times = None
            # Try to find matching tcpdump flow
            for key, stats in tcpdump_data.items():
                if server.name in key or client.name in key:
                    inter_packet_times = stats.inter_packet_times
                    break

            # Calculate derived metrics
            derived = calculate_derived_metrics(
                bandwidth_bps=iperf.bandwidth_bps,
                link_capacity_bps=link_capacity_bps,
                total_packets=iperf.total_packets,
                interval_duration=interval_duration,
                switch_stats=s_stats,
                jitter_ms=iperf.jitter_ms,
                packet_loss_percent=iperf.packet_loss_percent,
                rtt_ms=rtt,
                inter_packet_times=inter_packet_times,
            )

            # Calculate tcpdump-derived metrics if available
            tcpdump_derived = None
            for key, stats in tcpdump_data.items():
                if server.name in key or client.name in key:
                    tcpdump_derived = calculate_tcpdump_derived_metrics(stats)
                    break

            # Merge all metrics
            combined = merge_metrics_to_combined(
                iperf=iperf,
                ping=ping,
                switch_stats=s_stats,
                derived=derived,
                qos=qos,
                tcpdump_derived=tcpdump_derived,
            )

            data_points.append(combined)

    # Process all flows
    for h1, h2 in voip_pairs:
        process_flow(h1, h2, f"{h2.name}->{h1.name}")
        process_flow(h2, h1, f"{h1.name}->{h2.name}")

    logger.info(f"Processed {len(data_points)} data points")
    return data_points


def export_to_csv(
    data_points: tp.List[CombinedMetrics],
    output_path: str,
) -> None:
    """
    Export combined metrics to CSV file.

    Args:
        data_points: List of CombinedMetrics objects
        output_path: Path for output CSV file
    """
    if not data_points:
        logger.warning("No data points to export")
        return

    # Convert to dictionaries
    rows = [asdict(dp) for dp in data_points]

    # Write CSV
    with open(output_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)

    logger.info(f"Exported {len(rows)} records to {output_path}")


def process_and_export(
    voip_pairs: tp.List[tp.Tuple[Node, Node]],
    output_dir: str,
    polling_interval: float = 0.2,
    link_capacity_bps: int = 10_000_000,
) -> None:
    """
    Process logs and export to CSV in one step.

    Args:
        voip_pairs: List of VoIP host pairs
        output_dir: Output directory for logs and CSV
        polling_interval: Data collection interval
        link_capacity_bps: Link capacity for utilization
    """
    data_points = process_logs(
        voip_pairs,
        output_dir,
        polling_interval,
        link_capacity_bps,
    )

    csv_path = os.path.join(output_dir, "voip_simulation_data.csv")
    export_to_csv(data_points, csv_path)
