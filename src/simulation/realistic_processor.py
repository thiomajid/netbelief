"""
Data processor for realistic network simulation.

Processes switch statistics, interface stats, and tcpdump captures
to generate time series data suitable for forecasting.
"""

import csv
import glob
import os
import re
import typing as tp
from dataclasses import asdict, dataclass

from loguru import logger


@dataclass
class SwitchPortMetrics:
    """Metrics for a single switch port at a given time."""

    timestamp: float
    switch: str
    port: str
    rx_packets: int = 0
    rx_bytes: int = 0
    tx_packets: int = 0
    tx_bytes: int = 0
    rx_dropped: int = 0
    tx_dropped: int = 0
    rx_errors: int = 0
    tx_errors: int = 0


@dataclass
class SwitchMetrics:
    """Aggregated metrics for a switch at a given time."""

    timestamp: float
    switch: str
    total_rx_packets: int = 0
    total_rx_bytes: int = 0
    total_tx_packets: int = 0
    total_tx_bytes: int = 0
    total_rx_dropped: int = 0
    total_tx_dropped: int = 0
    total_rx_errors: int = 0
    total_tx_errors: int = 0
    port_count: int = 0


@dataclass
class NetworkSnapshot:
    """Complete network metrics at a point in time."""

    timestamp: float
    total_rx_bytes: int = 0
    total_tx_bytes: int = 0
    total_rx_packets: int = 0
    total_tx_packets: int = 0
    total_dropped: int = 0
    total_errors: int = 0
    active_switches: int = 0
    rx_throughput_mbps: float = 0.0
    tx_throughput_mbps: float = 0.0


@dataclass
class InterfaceMetrics:
    """Metrics from /proc/net/dev for a host interface."""

    timestamp: float
    host: str
    interface: str
    rx_bytes: int = 0
    rx_packets: int = 0
    rx_errors: int = 0
    rx_dropped: int = 0
    tx_bytes: int = 0
    tx_packets: int = 0
    tx_errors: int = 0
    tx_dropped: int = 0


def parse_switch_stats_file(filepath: str) -> tp.List[SwitchPortMetrics]:
    """Parse a switch stats log file."""
    metrics = []
    switch_name = (
        os.path.basename(filepath).replace("switch_stats_", "").replace(".log", "")
    )

    current_timestamp = 0.0
    current_port = None
    rx_pkts = rx_bytes = tx_pkts = tx_bytes = 0
    rx_drop = tx_drop = rx_err = tx_err = 0

    with open(filepath, "r") as f:
        for line in f:
            line = line.strip()

            if line.startswith("TIMESTAMP:"):
                current_timestamp = float(line.split(":")[1])

            elif line.startswith("port"):
                match = re.match(r"port\s+(\S+):", line)
                if match:
                    if current_port is not None:
                        metrics.append(
                            SwitchPortMetrics(
                                timestamp=current_timestamp,
                                switch=switch_name,
                                port=current_port,
                                rx_packets=rx_pkts,
                                rx_bytes=rx_bytes,
                                tx_packets=tx_pkts,
                                tx_bytes=tx_bytes,
                                rx_dropped=rx_drop,
                                tx_dropped=tx_drop,
                                rx_errors=rx_err,
                                tx_errors=tx_err,
                            )
                        )

                    current_port = match.group(1)
                    rx_pkts = rx_bytes = tx_pkts = tx_bytes = 0
                    rx_drop = tx_drop = rx_err = tx_err = 0

            elif "rx pkts=" in line:
                match = re.search(
                    r"rx pkts=(\d+), bytes=(\d+), drop=(\d+), errs=(\d+)", line
                )
                if match:
                    rx_pkts = int(match.group(1))
                    rx_bytes = int(match.group(2))
                    rx_drop = int(match.group(3))
                    rx_err = int(match.group(4))

            elif "tx pkts=" in line:
                match = re.search(
                    r"tx pkts=(\d+), bytes=(\d+), drop=(\d+), errs=(\d+)", line
                )
                if match:
                    tx_pkts = int(match.group(1))
                    tx_bytes = int(match.group(2))
                    tx_drop = int(match.group(3))
                    tx_err = int(match.group(4))

            elif line == "END_SNAPSHOT":
                if current_port is not None:
                    metrics.append(
                        SwitchPortMetrics(
                            timestamp=current_timestamp,
                            switch=switch_name,
                            port=current_port,
                            rx_packets=rx_pkts,
                            rx_bytes=rx_bytes,
                            tx_packets=tx_pkts,
                            tx_bytes=tx_bytes,
                            rx_dropped=rx_drop,
                            tx_dropped=tx_drop,
                            rx_errors=rx_err,
                            tx_errors=tx_err,
                        )
                    )
                current_port = None

    return metrics


def parse_interface_stats_file(filepath: str) -> tp.List[InterfaceMetrics]:
    """Parse an interface stats log file from /proc/net/dev."""
    metrics = []
    host_name = (
        os.path.basename(filepath).replace("interface_stats_", "").replace(".log", "")
    )

    current_timestamp = 0.0

    with open(filepath, "r") as f:
        for line in f:
            line = line.strip()

            if line.startswith("TIMESTAMP:"):
                current_timestamp = float(line.split(":")[1])

            elif "|" not in line and ":" in line and not line.startswith("Inter"):
                parts = line.split(":")
                if len(parts) == 2:
                    interface = parts[0].strip()
                    values = parts[1].split()

                    if len(values) >= 16 and interface not in ("lo",):
                        try:
                            metrics.append(
                                InterfaceMetrics(
                                    timestamp=current_timestamp,
                                    host=host_name,
                                    interface=interface,
                                    rx_bytes=int(values[0]),
                                    rx_packets=int(values[1]),
                                    rx_errors=int(values[2]),
                                    rx_dropped=int(values[3]),
                                    tx_bytes=int(values[8]),
                                    tx_packets=int(values[9]),
                                    tx_errors=int(values[10]),
                                    tx_dropped=int(values[11]),
                                )
                            )
                        except (ValueError, IndexError):
                            pass

    return metrics


def aggregate_by_timestamp(
    port_metrics: tp.List[SwitchPortMetrics],
    interval: float = 1.0,
) -> tp.List[NetworkSnapshot]:
    """Aggregate port metrics into network snapshots by time interval."""
    if not port_metrics:
        return []

    # Group by timestamp bucket
    buckets: tp.Dict[float, tp.List[SwitchPortMetrics]] = {}
    for m in port_metrics:
        # Round to nearest interval
        bucket = round(m.timestamp / interval) * interval
        if bucket not in buckets:
            buckets[bucket] = []
        buckets[bucket].append(m)

    snapshots = []
    sorted_buckets = sorted(buckets.keys())

    prev_snapshot = None
    for bucket in sorted_buckets:
        metrics_in_bucket = buckets[bucket]

        total_rx_bytes = sum(m.rx_bytes for m in metrics_in_bucket)
        total_tx_bytes = sum(m.tx_bytes for m in metrics_in_bucket)
        total_rx_packets = sum(m.rx_packets for m in metrics_in_bucket)
        total_tx_packets = sum(m.tx_packets for m in metrics_in_bucket)
        total_dropped = sum(m.rx_dropped + m.tx_dropped for m in metrics_in_bucket)
        total_errors = sum(m.rx_errors + m.tx_errors for m in metrics_in_bucket)
        active_switches = len(set(m.switch for m in metrics_in_bucket))

        # Calculate throughput as delta from previous snapshot
        rx_throughput = 0.0
        tx_throughput = 0.0
        if prev_snapshot is not None:
            delta_rx = total_rx_bytes - prev_snapshot.total_rx_bytes
            delta_tx = total_tx_bytes - prev_snapshot.total_tx_bytes
            rx_throughput = (delta_rx * 8) / (interval * 1_000_000)  # Mbps
            tx_throughput = (delta_tx * 8) / (interval * 1_000_000)  # Mbps

        snapshot = NetworkSnapshot(
            timestamp=float(bucket),
            total_rx_bytes=total_rx_bytes,
            total_tx_bytes=total_tx_bytes,
            total_rx_packets=total_rx_packets,
            total_tx_packets=total_tx_packets,
            total_dropped=total_dropped,
            total_errors=total_errors,
            active_switches=active_switches,
            rx_throughput_mbps=max(0, rx_throughput),
            tx_throughput_mbps=max(0, tx_throughput),
        )
        snapshots.append(snapshot)
        prev_snapshot = snapshot

    return snapshots


def process_realistic_simulation(
    output_dir: str,
    polling_interval: float = 0.5,
) -> tp.Tuple[tp.List[NetworkSnapshot], tp.List[InterfaceMetrics]]:
    """
    Process all logs from a realistic simulation.

    Returns network snapshots and interface metrics.
    """
    logger.info("Processing realistic simulation data...")

    # Parse all switch stats files
    all_port_metrics: tp.List[SwitchPortMetrics] = []
    switch_files = glob.glob(os.path.join(output_dir, "switch_stats_*.log"))

    for filepath in switch_files:
        metrics = parse_switch_stats_file(filepath)
        all_port_metrics.extend(metrics)
        logger.debug(
            f"Parsed {len(metrics)} port metrics from {os.path.basename(filepath)}"
        )

    logger.info(f"Total port metrics: {len(all_port_metrics)}")

    # Parse all interface stats files
    all_interface_metrics: tp.List[InterfaceMetrics] = []
    interface_files = glob.glob(os.path.join(output_dir, "interface_stats_*.log"))

    for filepath in interface_files:
        metrics = parse_interface_stats_file(filepath)
        all_interface_metrics.extend(metrics)

    logger.info(f"Total interface metrics: {len(all_interface_metrics)}")

    # Aggregate into snapshots
    snapshots = aggregate_by_timestamp(all_port_metrics, interval=polling_interval)
    logger.info(f"Generated {len(snapshots)} network snapshots")

    return snapshots, all_interface_metrics


def export_realistic_simulation_data(
    output_dir: str,
    snapshots: tp.List[NetworkSnapshot],
    interface_metrics: tp.List[InterfaceMetrics],
) -> None:
    """Export processed data to CSV files."""

    # Export network snapshots
    if snapshots:
        snapshot_file = os.path.join(output_dir, "network_snapshots.csv")
        with open(snapshot_file, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(asdict(snapshots[0]).keys()))
            writer.writeheader()
            for snapshot in snapshots:
                writer.writerow(asdict(snapshot))
        logger.info(f"Exported {len(snapshots)} snapshots to {snapshot_file}")
    else:
        logger.warning("No network snapshots to export")

    # Export interface metrics grouped by host
    if interface_metrics:
        # Group by host
        by_host: tp.Dict[str, tp.List[InterfaceMetrics]] = {}
        for m in interface_metrics:
            if m.host not in by_host:
                by_host[m.host] = []
            by_host[m.host].append(m)

        # Export aggregated interface data
        interface_file = os.path.join(output_dir, "interface_metrics.csv")
        with open(interface_file, "w", newline="") as f:
            writer = csv.DictWriter(
                f, fieldnames=list(asdict(interface_metrics[0]).keys())
            )
            writer.writeheader()
            for m in interface_metrics:
                writer.writerow(asdict(m))
        logger.info(
            f"Exported {len(interface_metrics)} interface metrics to {interface_file}"
        )
    else:
        logger.warning("No interface metrics to export")


def process_and_export_realistic(
    output_dir: str,
    polling_interval: float = 0.5,
) -> None:
    """Main entry point for processing realistic simulation data."""
    snapshots, interface_metrics = process_realistic_simulation(
        output_dir=output_dir,
        polling_interval=polling_interval,
    )

    export_realistic_simulation_data(
        output_dir=output_dir,
        snapshots=snapshots,
        interface_metrics=interface_metrics,
    )
