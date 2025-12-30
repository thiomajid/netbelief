"""
Enhanced QoS data processor for realistic network simulation.

Processes raw network data and calculates comprehensive QoS metrics
per host for time series forecasting.
"""

import csv
import glob
import os
import typing as tp
from dataclasses import asdict, dataclass

from loguru import logger

from src.simulation.qos_metrics import (
    HostQoSMetrics,
    LinkQoSMetrics,
    NetworkQoSSnapshot,
    QoSCalculator,
    estimate_rtt_from_traffic,
)


@dataclass
class RawHostData:
    """Raw data collected for a host at a timestamp."""

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


@dataclass
class RawSwitchPortData:
    """Raw data collected for a switch port."""

    timestamp: float
    switch: str
    port: str
    rx_bytes: int = 0
    rx_packets: int = 0
    rx_dropped: int = 0
    rx_errors: int = 0
    tx_bytes: int = 0
    tx_packets: int = 0
    tx_dropped: int = 0
    tx_errors: int = 0


class QoSDataProcessor:
    """
    Processes raw network data into comprehensive QoS metrics.

    Calculates per-host metrics suitable for time series forecasting.
    """

    def __init__(
        self,
        output_dir: str,
        polling_interval: float = 0.5,
        link_capacity_mbps: float = 10.0,
        baseline_delay_ms: float = 1.0,
    ):
        self.output_dir = output_dir
        self.polling_interval = polling_interval
        self.link_capacity_mbps = link_capacity_mbps
        self.link_capacity_bps = link_capacity_mbps * 1_000_000
        self.baseline_delay_ms = baseline_delay_ms

        # Raw data storage
        self.host_data: tp.Dict[str, tp.List[RawHostData]] = {}
        self.switch_data: tp.Dict[str, tp.List[RawSwitchPortData]] = {}

        # Processed metrics
        self.host_qos_metrics: tp.List[HostQoSMetrics] = []
        self.link_qos_metrics: tp.List[LinkQoSMetrics] = []
        self.network_snapshots: tp.List[NetworkQoSSnapshot] = []

    def load_interface_stats(self) -> None:
        """Load and parse interface stats files."""
        logger.info("Loading interface stats...")

        interface_files = glob.glob(
            os.path.join(self.output_dir, "interface_stats_*.log")
        )

        for filepath in interface_files:
            host_name = (
                os.path.basename(filepath)
                .replace("interface_stats_", "")
                .replace(".log", "")
            )

            if host_name not in self.host_data:
                self.host_data[host_name] = []

            current_timestamp = 0.0

            with open(filepath, "r") as f:
                for line in f:
                    line = line.strip()

                    if line.startswith("TIMESTAMP:"):
                        current_timestamp = float(line.split(":", 1)[1])

                    elif (
                        "|" not in line and ":" in line and not line.startswith("Inter")
                    ):
                        parts = line.split(":")
                        if len(parts) == 2:
                            interface = parts[0].strip()
                            values = parts[1].split()

                            if len(values) >= 16 and interface not in ("lo",):
                                try:
                                    data = RawHostData(
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
                                    self.host_data[host_name].append(data)
                                except (ValueError, IndexError):
                                    pass

            logger.debug(
                f"Loaded {len(self.host_data.get(host_name, []))} records for {host_name}"
            )

        logger.info(f"Loaded data for {len(self.host_data)} hosts")

    def load_switch_stats(self) -> None:
        """Load and parse switch stats files."""
        logger.info("Loading switch stats...")

        switch_files = glob.glob(os.path.join(self.output_dir, "switch_stats_*.log"))

        import re

        for filepath in switch_files:
            switch_name = (
                os.path.basename(filepath)
                .replace("switch_stats_", "")
                .replace(".log", "")
            )

            if switch_name not in self.switch_data:
                self.switch_data[switch_name] = []

            current_timestamp = 0.0
            current_port = None
            rx_pkts = rx_bytes = tx_pkts = tx_bytes = 0
            rx_drop = tx_drop = rx_err = tx_err = 0

            with open(filepath, "r") as f:
                for line in f:
                    line = line.strip()

                    if line.startswith("TIMESTAMP:"):
                        current_timestamp = float(line.split(":", 1)[1])

                    elif line.startswith("port"):
                        match = re.match(r"port\s+(\S+):", line)
                        if match:
                            if current_port is not None:
                                data = RawSwitchPortData(
                                    timestamp=current_timestamp,
                                    switch=switch_name,
                                    port=current_port,
                                    rx_bytes=rx_bytes,
                                    rx_packets=rx_pkts,
                                    rx_dropped=rx_drop,
                                    rx_errors=rx_err,
                                    tx_bytes=tx_bytes,
                                    tx_packets=tx_pkts,
                                    tx_dropped=tx_drop,
                                    tx_errors=tx_err,
                                )
                                self.switch_data[switch_name].append(data)

                            current_port = match.group(1).strip('"')
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
                            data = RawSwitchPortData(
                                timestamp=current_timestamp,
                                switch=switch_name,
                                port=current_port,
                                rx_bytes=rx_bytes,
                                rx_packets=rx_pkts,
                                rx_dropped=rx_drop,
                                rx_errors=rx_err,
                                tx_bytes=tx_bytes,
                                tx_packets=tx_pkts,
                                tx_dropped=tx_drop,
                                tx_errors=tx_err,
                            )
                            self.switch_data[switch_name].append(data)
                        current_port = None

            logger.debug(
                f"Loaded {len(self.switch_data.get(switch_name, []))} records for {switch_name}"
            )

        logger.info(f"Loaded data for {len(self.switch_data)} switches")

    def calculate_host_qos_metrics(self) -> None:
        """Calculate comprehensive QoS metrics for each host."""
        logger.info("Calculating host QoS metrics...")

        for host_name, records in self.host_data.items():
            if len(records) < 2:
                continue

            # Sort by timestamp
            records.sort(key=lambda x: x.timestamp)

            # Track historical data for derived metrics
            packet_counts: tp.List[int] = []
            delay_estimates: tp.List[float] = []

            for i in range(1, len(records)):
                prev = records[i - 1]
                curr = records[i]

                time_delta = curr.timestamp - prev.timestamp
                if time_delta <= 0:
                    continue

                # Calculate deltas
                delta_rx_bytes = max(0, curr.rx_bytes - prev.rx_bytes)
                delta_tx_bytes = max(0, curr.tx_bytes - prev.tx_bytes)
                delta_rx_packets = max(0, curr.rx_packets - prev.rx_packets)
                delta_tx_packets = max(0, curr.tx_packets - prev.tx_packets)
                delta_rx_dropped = max(0, curr.rx_dropped - prev.rx_dropped)
                delta_tx_dropped = max(0, curr.tx_dropped - prev.tx_dropped)
                delta_rx_errors = max(0, curr.rx_errors - prev.rx_errors)
                delta_tx_errors = max(0, curr.tx_errors - prev.tx_errors)

                # Calculate throughput
                rx_throughput_bps = (delta_rx_bytes * 8) / time_delta
                tx_throughput_bps = (delta_tx_bytes * 8) / time_delta
                rx_throughput_mbps = rx_throughput_bps / 1_000_000
                tx_throughput_mbps = tx_throughput_bps / 1_000_000

                # Calculate utilization
                rx_utilization = QoSCalculator.calculate_link_utilization(
                    rx_throughput_bps, self.link_capacity_bps
                )
                tx_utilization = QoSCalculator.calculate_link_utilization(
                    tx_throughput_bps, self.link_capacity_bps
                )
                avg_utilization = (rx_utilization + tx_utilization) / 2

                # Calculate packet loss
                total_sent = delta_tx_packets
                total_dropped = delta_rx_dropped + delta_tx_dropped
                if total_sent > 0:
                    packet_loss_pct = (
                        total_dropped / max(1, total_sent + total_dropped)
                    ) * 100
                else:
                    packet_loss_pct = 0.0

                # Error rate
                total_packets = delta_rx_packets + delta_tx_packets
                total_errors = delta_rx_errors + delta_tx_errors
                error_rate = (
                    (total_errors / max(1, total_packets)) * 100
                    if total_packets > 0
                    else 0.0
                )

                # Estimate RTT
                rtt_ms = estimate_rtt_from_traffic(
                    delta_tx_bytes,
                    delta_rx_bytes,
                    delta_tx_packets,
                    delta_rx_packets,
                    self.baseline_delay_ms,
                    self.link_capacity_mbps,
                )
                delay_estimates.append(rtt_ms)

                # Track packet counts for jitter estimation
                packet_counts.append(delta_tx_packets + delta_rx_packets)

                # Calculate jitter from recent history
                recent_counts = packet_counts[-10:]  # Last 10 samples
                jitter_ms, jitter_avg_ms = 0.0, 0.0
                if len(delay_estimates) >= 2:
                    recent_delays = delay_estimates[-10:]
                    jitter_ms, jitter_avg_ms = QoSCalculator.calculate_jitter(
                        recent_delays
                    )

                # Calculate R-Factor and MOS
                r_factor = QoSCalculator.calculate_r_factor(
                    delay_ms=rtt_ms / 2,  # One-way delay
                    jitter_ms=jitter_ms,
                    packet_loss_pct=packet_loss_pct,
                )
                mos = QoSCalculator.calculate_mos(r_factor)

                # Calculate congestion index
                congestion_index = QoSCalculator.calculate_congestion_index(
                    avg_utilization, packet_loss_pct, 0
                )

                # Calculate throughput efficiency (compared to link capacity)
                max_throughput = max(rx_throughput_mbps, tx_throughput_mbps)
                throughput_efficiency = QoSCalculator.calculate_throughput_efficiency(
                    max_throughput * 1_000_000,
                    self.link_capacity_bps,
                )

                # Goodput ratio (accounting for retransmissions approximation)
                overhead_bytes = total_packets * 40  # Approximate header overhead
                goodput_ratio = QoSCalculator.calculate_goodput_ratio(
                    delta_tx_bytes + delta_rx_bytes,
                    0,  # No retransmission data available
                    overhead_bytes,
                )

                # Calculate QoS score
                qos_score, qos_category = QoSCalculator.calculate_qos_score(
                    mos, packet_loss_pct, avg_utilization, jitter_ms
                )

                # Calculate bandwidth values in multiple units
                total_bandwidth_bps = rx_throughput_bps + tx_throughput_bps
                total_bandwidth_kbps = total_bandwidth_bps / 1000
                total_bandwidth_mbps = total_bandwidth_bps / 1_000_000

                # Calculate congestion score (0-100, inverse of quality)
                congestion_score = congestion_index * 100

                # Create metric record
                metrics = HostQoSMetrics(
                    timestamp=curr.timestamp,
                    host=host_name,
                    # Cumulative counters
                    rx_bytes=curr.rx_bytes,
                    tx_bytes=curr.tx_bytes,
                    rx_packets=curr.rx_packets,
                    tx_packets=curr.tx_packets,
                    rx_errors=curr.rx_errors,
                    tx_errors=curr.tx_errors,
                    rx_dropped=curr.rx_dropped,
                    tx_dropped=curr.tx_dropped,
                    # Bandwidth - bytes delta
                    rx_bytes_delta=delta_rx_bytes,
                    tx_bytes_delta=delta_tx_bytes,
                    # Bandwidth - bps
                    rx_bandwidth_bps=rx_throughput_bps,
                    tx_bandwidth_bps=tx_throughput_bps,
                    total_bandwidth_bps=total_bandwidth_bps,
                    # Bandwidth - Kbps
                    rx_bandwidth_kbps=rx_throughput_bps / 1000,
                    tx_bandwidth_kbps=tx_throughput_bps / 1000,
                    total_bandwidth_kbps=total_bandwidth_kbps,
                    # Bandwidth - Mbps
                    rx_bandwidth_mbps=rx_throughput_mbps,
                    tx_bandwidth_mbps=tx_throughput_mbps,
                    total_bandwidth_mbps=total_bandwidth_mbps,
                    # Latency
                    rtt_ms=rtt_ms,
                    rtt_min_ms=min(delay_estimates) if delay_estimates else rtt_ms,
                    rtt_max_ms=max(delay_estimates) if delay_estimates else rtt_ms,
                    rtt_avg_ms=sum(delay_estimates) / len(delay_estimates)
                    if delay_estimates
                    else rtt_ms,
                    # Jitter
                    jitter_ms=jitter_ms,
                    jitter_avg_ms=jitter_avg_ms,
                    # Legacy throughput fields (for compatibility)
                    rx_throughput_mbps=rx_throughput_mbps,
                    tx_throughput_mbps=tx_throughput_mbps,
                    rx_throughput_kbps=rx_throughput_mbps * 1000,
                    tx_throughput_kbps=tx_throughput_mbps * 1000,
                    # Utilization
                    link_capacity_mbps=self.link_capacity_mbps,
                    rx_utilization_pct=rx_utilization,
                    tx_utilization_pct=tx_utilization,
                    avg_utilization_pct=avg_utilization,
                    # Packet loss
                    packet_loss_pct=packet_loss_pct,
                    packet_loss_rate=total_dropped / max(1, time_delta),
                    error_rate=error_rate,
                    # Voice quality
                    r_factor=r_factor,
                    mos=mos,
                    # Congestion
                    congestion_index=congestion_index,
                    congestion_score=congestion_score,
                    queue_delay_ms=max(0, rtt_ms - self.baseline_delay_ms * 2),
                    buffer_occupancy_pct=min(100, congestion_index * 100),
                    # Efficiency
                    throughput_efficiency=throughput_efficiency,
                    goodput_ratio=goodput_ratio,
                    # QoS classification
                    qos_score=qos_score,
                    qos_category=qos_category,
                )

                self.host_qos_metrics.append(metrics)

        logger.info(f"Calculated {len(self.host_qos_metrics)} host QoS metric records")

    def calculate_network_snapshots(self) -> None:
        """Calculate network-wide QoS snapshots."""
        logger.info("Calculating network-wide snapshots...")

        # Group metrics by timestamp bucket
        from collections import defaultdict

        time_buckets: tp.Dict[float, tp.List[HostQoSMetrics]] = defaultdict(list)

        for m in self.host_qos_metrics:
            bucket = round(m.timestamp / self.polling_interval) * self.polling_interval
            time_buckets[bucket].append(m)

        for timestamp in sorted(time_buckets.keys()):
            metrics = time_buckets[timestamp]

            if not metrics:
                continue

            # Aggregate metrics
            total_rx_mbps = sum(m.rx_throughput_mbps for m in metrics)
            total_tx_mbps = sum(m.tx_throughput_mbps for m in metrics)
            avg_throughput = (total_rx_mbps + total_tx_mbps) / 2

            rtt_values = [m.rtt_ms for m in metrics if m.rtt_ms > 0]
            jitter_values = [m.jitter_ms for m in metrics]
            loss_values = [m.packet_loss_pct for m in metrics]
            util_values = [m.avg_utilization_pct for m in metrics]
            mos_values = [m.mos for m in metrics]
            r_factor_values = [m.r_factor for m in metrics]

            # Calculate percentiles for RTT
            sorted_rtt = sorted(rtt_values) if rtt_values else [0]
            p95_idx = int(len(sorted_rtt) * 0.95)
            p99_idx = int(len(sorted_rtt) * 0.99)

            snapshot = NetworkQoSSnapshot(
                timestamp=timestamp,
                total_hosts=len(self.host_data),
                active_hosts=len(
                    [
                        m
                        for m in metrics
                        if m.tx_throughput_mbps > 0 or m.rx_throughput_mbps > 0
                    ]
                ),
                total_switches=len(self.switch_data),
                total_rx_mbps=total_rx_mbps,
                total_tx_mbps=total_tx_mbps,
                avg_throughput_mbps=avg_throughput,
                peak_throughput_mbps=max(
                    m.tx_throughput_mbps + m.rx_throughput_mbps for m in metrics
                ),
                avg_rtt_ms=sum(rtt_values) / len(rtt_values) if rtt_values else 0,
                max_rtt_ms=max(rtt_values) if rtt_values else 0,
                p95_rtt_ms=sorted_rtt[p95_idx] if sorted_rtt else 0,
                p99_rtt_ms=sorted_rtt[min(p99_idx, len(sorted_rtt) - 1)]
                if sorted_rtt
                else 0,
                avg_jitter_ms=sum(jitter_values) / len(jitter_values)
                if jitter_values
                else 0,
                max_jitter_ms=max(jitter_values) if jitter_values else 0,
                avg_packet_loss_pct=sum(loss_values) / len(loss_values)
                if loss_values
                else 0,
                max_packet_loss_pct=max(loss_values) if loss_values else 0,
                total_dropped_packets=sum(m.rx_dropped + m.tx_dropped for m in metrics),
                avg_utilization_pct=sum(util_values) / len(util_values)
                if util_values
                else 0,
                max_utilization_pct=max(util_values) if util_values else 0,
                congested_links_pct=len([u for u in util_values if u > 80])
                / max(1, len(util_values))
                * 100,
                avg_mos=sum(mos_values) / len(mos_values) if mos_values else 4.41,
                min_mos=min(mos_values) if mos_values else 4.41,
                avg_r_factor=sum(r_factor_values) / len(r_factor_values)
                if r_factor_values
                else 93.2,
                min_r_factor=min(r_factor_values) if r_factor_values else 93.2,
                network_health_score=sum(m.qos_score for m in metrics) / len(metrics)
                if metrics
                else 100,
                qos_violations=len([m for m in metrics if m.qos_score < 60]),
            )

            self.network_snapshots.append(snapshot)

        logger.info(f"Generated {len(self.network_snapshots)} network snapshots")

    def export_metrics(self) -> None:
        """Export all metrics to CSV files."""
        logger.info("Exporting QoS metrics...")

        # Export host QoS metrics
        if self.host_qos_metrics:
            filepath = os.path.join(self.output_dir, "host_qos_metrics.csv")
            with open(filepath, "w", newline="") as f:
                fieldnames = list(asdict(self.host_qos_metrics[0]).keys())
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                writer.writeheader()
                for m in self.host_qos_metrics:
                    writer.writerow(asdict(m))
            logger.info(
                f"Exported {len(self.host_qos_metrics)} host QoS metrics to {filepath}"
            )

        # Export network snapshots
        if self.network_snapshots:
            filepath = os.path.join(self.output_dir, "network_qos_snapshots.csv")
            with open(filepath, "w", newline="") as f:
                fieldnames = list(asdict(self.network_snapshots[0]).keys())
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                writer.writeheader()
                for m in self.network_snapshots:
                    writer.writerow(asdict(m))
            logger.info(
                f"Exported {len(self.network_snapshots)} network snapshots to {filepath}"
            )

        # Export per-host time series (for forecasting)
        self._export_per_host_timeseries()

        # Export consolidated forecasting dataset (single file, ready to use)
        self._export_forecasting_dataset()

    def _export_per_host_timeseries(self) -> None:
        """Export time series data per host for forecasting."""
        from collections import defaultdict

        # Group by host
        by_host: tp.Dict[str, tp.List[HostQoSMetrics]] = defaultdict(list)
        for m in self.host_qos_metrics:
            by_host[m.host].append(m)

        # Create per-host directory
        per_host_dir = os.path.join(self.output_dir, "per_host_timeseries")
        os.makedirs(per_host_dir, exist_ok=True)

        for host, metrics in by_host.items():
            metrics.sort(key=lambda x: x.timestamp)

            filepath = os.path.join(per_host_dir, f"{host}_qos_timeseries.csv")
            with open(filepath, "w", newline="") as f:
                fieldnames = list(asdict(metrics[0]).keys())
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                writer.writeheader()
                for m in metrics:
                    writer.writerow(asdict(m))

        logger.info(
            f"Exported per-host time series for {len(by_host)} hosts to {per_host_dir}"
        )

    def _export_forecasting_dataset(self) -> None:
        """
        Export a single consolidated CSV file ready for forecasting.

        This file contains all metrics for all hosts, sorted by timestamp and host,
        with an additional host index column for easier processing.
        """
        from datetime import datetime

        if not self.host_qos_metrics:
            logger.warning("No QoS metrics to export for forecasting dataset")
            return

        # Sort by timestamp, then by host for consistent ordering
        sorted_metrics = sorted(
            self.host_qos_metrics, key=lambda x: (x.timestamp, x.host)
        )

        # Create host index mapping for ML models
        unique_hosts = sorted(set(m.host for m in sorted_metrics))
        host_to_idx = {host: idx for idx, host in enumerate(unique_hosts)}

        filepath = os.path.join(self.output_dir, "forecasting_dataset.csv")

        with open(filepath, "w", newline="") as f:
            # Get base fieldnames and add extra columns
            base_fields = list(asdict(sorted_metrics[0]).keys())

            # Add additional useful columns for forecasting
            extra_fields = [
                "host_idx",  # Numeric host identifier
                "datetime",  # ISO format datetime
                "time_seconds",  # Seconds since start
                "rx_mbps",  # Convenience: RX in Mbps
                "tx_mbps",  # Convenience: TX in Mbps
                "total_mbps",  # Total throughput in Mbps
                "is_congested",  # Binary: congestion_index > 0.5
                "is_degraded",  # Binary: qos_score < 70
            ]

            fieldnames = base_fields + extra_fields
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()

            # Calculate start time for relative timestamps
            start_time = min(m.timestamp for m in sorted_metrics)

            for m in sorted_metrics:
                row = asdict(m)

                # Add extra computed fields
                row["host_idx"] = host_to_idx[m.host]
                row["datetime"] = datetime.fromtimestamp(m.timestamp).isoformat()
                row["time_seconds"] = m.timestamp - start_time
                row["rx_mbps"] = m.rx_throughput_mbps
                row["tx_mbps"] = m.tx_throughput_mbps
                row["total_mbps"] = m.rx_throughput_mbps + m.tx_throughput_mbps
                row["is_congested"] = 1 if m.congestion_index > 0.5 else 0
                row["is_degraded"] = 1 if m.qos_score < 70 else 0

                writer.writerow(row)

        logger.info(
            f"Exported consolidated forecasting dataset: {filepath} "
            f"({len(sorted_metrics)} rows, {len(unique_hosts)} hosts)"
        )

        # Also export a metadata file with host mappings and stats
        self._export_dataset_metadata(unique_hosts, host_to_idx, sorted_metrics)

    def _export_dataset_metadata(
        self,
        unique_hosts: tp.List[str],
        host_to_idx: tp.Dict[str, int],
        metrics: tp.List[HostQoSMetrics],
    ) -> None:
        """Export metadata about the forecasting dataset."""
        import json
        from collections import defaultdict

        # Calculate per-host statistics
        by_host: tp.Dict[str, tp.List[HostQoSMetrics]] = defaultdict(list)
        for m in metrics:
            by_host[m.host].append(m)

        host_stats = {}
        for host in unique_hosts:
            host_metrics = by_host[host]
            host_stats[host] = {
                "host_idx": host_to_idx[host],
                "num_samples": len(host_metrics),
                "time_range_seconds": (
                    max(m.timestamp for m in host_metrics)
                    - min(m.timestamp for m in host_metrics)
                )
                if len(host_metrics) > 1
                else 0,
                "avg_qos_score": sum(m.qos_score for m in host_metrics)
                / len(host_metrics),
                "avg_mos": sum(m.mos for m in host_metrics) / len(host_metrics),
                "avg_rtt_ms": sum(m.rtt_ms for m in host_metrics) / len(host_metrics),
                "avg_utilization_pct": sum(m.avg_utilization_pct for m in host_metrics)
                / len(host_metrics),
                "congestion_ratio": sum(
                    1 for m in host_metrics if m.congestion_index > 0.5
                )
                / len(host_metrics),
                "degraded_ratio": sum(1 for m in host_metrics if m.qos_score < 70)
                / len(host_metrics),
            }

        metadata = {
            "dataset_info": {
                "total_samples": len(metrics),
                "num_hosts": len(unique_hosts),
                "polling_interval_seconds": self.polling_interval,
                "link_capacity_mbps": self.link_capacity_mbps,
                "start_timestamp": min(m.timestamp for m in metrics),
                "end_timestamp": max(m.timestamp for m in metrics),
                "duration_seconds": max(m.timestamp for m in metrics)
                - min(m.timestamp for m in metrics),
            },
            "host_mapping": host_to_idx,
            "host_statistics": host_stats,
            "feature_columns": {
                "bandwidth": {
                    "columns": [
                        "rx_bytes_delta",
                        "tx_bytes_delta",
                        "rx_bandwidth_bps",
                        "tx_bandwidth_bps",
                        "total_bandwidth_bps",
                        "rx_bandwidth_kbps",
                        "tx_bandwidth_kbps",
                        "total_bandwidth_kbps",
                        "rx_bandwidth_mbps",
                        "tx_bandwidth_mbps",
                        "total_bandwidth_mbps",
                    ],
                    "description": "Actual used bandwidth in various units (bps, Kbps, Mbps)",
                },
                "latency": {
                    "columns": ["rtt_ms", "rtt_min_ms", "rtt_max_ms", "rtt_avg_ms"],
                    "description": "Round-trip time metrics in milliseconds",
                },
                "jitter": {
                    "columns": ["jitter_ms", "jitter_avg_ms"],
                    "description": "Packet delay variation in milliseconds",
                },
                "utilization": {
                    "columns": [
                        "rx_utilization_pct",
                        "tx_utilization_pct",
                        "avg_utilization_pct",
                    ],
                    "description": "Link utilization as percentage of link_capacity_mbps",
                },
                "packet_loss": {
                    "columns": ["packet_loss_pct", "packet_loss_rate", "error_rate"],
                    "description": "Packet loss and error metrics",
                },
                "voice_quality": {
                    "columns": ["mos", "r_factor"],
                    "description": "ITU-T G.107 E-model voice quality metrics. MOS: 1-5 (4.41=excellent). R-Factor: 0-100 (93.2=excellent)",
                },
                "congestion": {
                    "columns": [
                        "congestion_index",
                        "congestion_score",
                        "queue_delay_ms",
                        "buffer_occupancy_pct",
                    ],
                    "description": "Congestion metrics. congestion_index: 0-1 normalized. congestion_score: 0-100 (higher=more congested)",
                },
                "efficiency": {
                    "columns": ["throughput_efficiency", "goodput_ratio"],
                    "description": "Efficiency metrics (0-1 ratio)",
                },
                "qos_classification": {
                    "columns": ["qos_score", "qos_category"],
                    "description": "Overall QoS assessment. qos_score: 0-100 weighted composite of MOS(35%), loss(25%), jitter(20%), utilization(20%)",
                },
                "target_candidates": [
                    "qos_score",
                    "mos",
                    "r_factor",
                    "rtt_ms",
                    "jitter_ms",
                    "total_bandwidth_mbps",
                    "avg_utilization_pct",
                    "congestion_score",
                    "congestion_index",
                    "throughput_efficiency",
                ],
                "categorical": ["host", "qos_category"],
                "binary": ["is_congested", "is_degraded"],
                "temporal": ["timestamp", "datetime", "time_seconds"],
            },
            "metric_definitions": {
                "qos_score": {
                    "range": "0-100",
                    "formula": "0.35*MOS_normalized + 0.25*loss_score + 0.20*jitter_score + 0.20*util_score",
                    "interpretation": "Higher is better. >=90=excellent, 75-89=good, 60-74=fair, 40-59=poor, <40=bad",
                },
                "congestion_score": {
                    "range": "0-100",
                    "formula": "congestion_index * 100",
                    "interpretation": "Higher means more congested. 0=no congestion, 100=severe congestion",
                },
                "congestion_index": {
                    "range": "0-1",
                    "formula": "0.4*util_factor + 0.35*loss_factor + 0.25*delay_factor",
                    "interpretation": "Normalized congestion. >0.5 indicates significant congestion",
                },
                "total_bandwidth_mbps": {
                    "unit": "Mbps",
                    "formula": "rx_bandwidth_mbps + tx_bandwidth_mbps",
                    "interpretation": "Actual used bandwidth in the polling interval",
                },
                "mos": {
                    "range": "1-5",
                    "formula": "ITU-T G.107 E-model conversion from R-Factor",
                    "interpretation": "Voice quality. 4.0+=excellent, 3.6-4.0=good, 2.6-3.6=fair, <2.6=poor",
                },
                "r_factor": {
                    "range": "0-100",
                    "formula": "R0 - Is - Id - Ie_eff + A (ITU-T G.107)",
                    "interpretation": "Voice transmission quality. 80+=high, 70-80=medium, <50=unacceptable",
                },
            },
            "qos_thresholds": {
                "excellent": {
                    "qos_score": ">= 90",
                    "mos": ">= 4.0",
                    "congestion_score": "< 10",
                },
                "good": {
                    "qos_score": "75-89",
                    "mos": "3.6-4.0",
                    "congestion_score": "10-30",
                },
                "fair": {
                    "qos_score": "60-74",
                    "mos": "2.6-3.6",
                    "congestion_score": "30-50",
                },
                "poor": {
                    "qos_score": "40-59",
                    "mos": "1.0-2.6",
                    "congestion_score": "50-70",
                },
                "bad": {
                    "qos_score": "< 40",
                    "mos": "< 1.0",
                    "congestion_score": "> 70",
                },
            },
        }

        filepath = os.path.join(self.output_dir, "forecasting_metadata.json")
        with open(filepath, "w") as f:
            json.dump(metadata, f, indent=2)

        logger.info(f"Exported dataset metadata to {filepath}")

    def process(self) -> None:
        """Run complete processing pipeline."""
        self.load_interface_stats()
        self.load_switch_stats()
        self.calculate_host_qos_metrics()
        self.calculate_network_snapshots()
        self.export_metrics()


def process_qos_metrics(
    output_dir: str,
    polling_interval: float = 0.5,
    link_capacity_mbps: float = 10.0,
) -> None:
    """
    Main entry point for QoS metrics processing.

    Args:
        output_dir: Directory containing simulation output files
        polling_interval: Data collection interval in seconds
        link_capacity_mbps: Link capacity in Mbps
    """
    processor = QoSDataProcessor(
        output_dir=output_dir,
        polling_interval=polling_interval,
        link_capacity_mbps=link_capacity_mbps,
    )
    processor.process()
