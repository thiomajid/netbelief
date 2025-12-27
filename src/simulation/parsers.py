"""
Log parsers for VoIP network simulation.

This module contains parsers for various log formats including:
- iperf UDP output
- ping/ICMP output
- OVS switch statistics
- tcpdump packet captures
"""

import os
import re
import typing as tp

from loguru import logger

from src.simulation.metrics import (
    AggregatedSwitchStats,
    IperfMetrics,
    PingMetrics,
    SwitchPortStats,
    TcpdumpFlowStats,
    TcpdumpPacketInfo,
)


def convert_to_bps(value: float, unit: str) -> float:
    """
    Convert bandwidth value to bits per second.

    Args:
        value: Numeric bandwidth value
        unit: Unit string (e.g., "Kbits/sec", "Mbits/sec")

    Returns:
        Bandwidth in bits per second
    """
    if "Kbits" in unit:
        return value * 1000
    elif "Mbits" in unit:
        return value * 1_000_000
    elif "Gbits" in unit:
        return value * 1_000_000_000
    return value


def parse_iperf_udp_line(
    line: str,
    flow_id: str,
    start_time: float = 0.0,
) -> tp.Optional[IperfMetrics]:
    """
    Parse an iperf UDP output line.

    Example line:
    [  3]  0.0- 1.0 sec  12.9 KBytes   106 Kbits/sec  0.034 ms    0/    9 (0%)

    Args:
        line: Raw iperf output line
        flow_id: Flow identifier for this measurement
        start_time: Reference start time for timestamp calculation

    Returns:
        IperfMetrics object or None if parsing fails
    """
    pattern = (
        r"\[\s*\d+\]\s*(\d+\.\d+)-\s*(\d+\.\d+)\s*sec\s*"
        r"(\d+(\.\d+)?)\s*([KMG]Bytes)\s*"
        r"(\d+(\.\d+)?)\s*([KMG]bits/sec)\s*"
        r"(\d+(\.\d+)?)\s*ms\s*"
        r"(\d+)/\s*(\d+)\s*\((\d+(\.\d+)?)%\)"
    )
    match = re.search(pattern, line)
    if match:
        return IperfMetrics(
            timestamp=start_time + float(match.group(2)),
            flow_id=flow_id,
            interval_start=float(match.group(1)),
            interval_end=float(match.group(2)),
            bandwidth_bps=convert_to_bps(float(match.group(6)), match.group(8)),
            jitter_ms=float(match.group(9)),
            packet_loss_cnt=int(match.group(11)),
            total_packets=int(match.group(12)),
            packet_loss_percent=float(match.group(13)),
        )
    return None


def parse_ping_line(
    line: str,
    flow_id: str,
    start_time_ref: tp.Optional[float] = None,
) -> tp.Optional[PingMetrics]:
    """
    Parse a ping output line with Unix timestamp.

    Example: [1632123456.789012] 64 bytes from 10.0.0.2: icmp_seq=1 ttl=64 time=0.045 ms

    Args:
        line: Raw ping output line
        flow_id: Flow identifier
        start_time_ref: Reference timestamp for relative time calculation

    Returns:
        PingMetrics object or None if parsing fails
    """
    # Pattern with icmp_seq and ttl extraction
    pattern = r"\[(\d+\.\d+)\].*icmp_seq=(\d+)\s+ttl=(\d+)\s+time=(\d+\.?\d*)\s*ms"
    match = re.search(pattern, line)

    if match:
        ts = float(match.group(1))
        rel_time = ts - start_time_ref if start_time_ref else ts
        return PingMetrics(
            timestamp=rel_time,
            abs_timestamp=ts,
            flow_id=flow_id,
            rtt_ms=float(match.group(4)),
            icmp_seq=int(match.group(2)),
            ttl=int(match.group(3)),
        )

    # Fallback pattern without icmp_seq/ttl
    simple_pattern = r"\[(\d+\.\d+)\].*time=(\d+\.?\d*)\s*ms"
    match = re.search(simple_pattern, line)
    if match:
        ts = float(match.group(1))
        rel_time = ts - start_time_ref if start_time_ref else ts
        return PingMetrics(
            timestamp=rel_time,
            abs_timestamp=ts,
            flow_id=flow_id,
            rtt_ms=float(match.group(2)),
        )
    return None


def parse_switch_port_stats(
    output: str,
    switch_name: str,
    timestamp: float,
) -> tp.List[SwitchPortStats]:
    """
    Parse OVS switch port statistics output.

    Args:
        output: Raw output from 'ovs-ofctl dump-ports'
        switch_name: Name of the switch
        timestamp: Collection timestamp

    Returns:
        List of SwitchPortStats objects for each port
    """
    stats = []

    port_pattern_rx = (
        r"port\s+(\S+):.*rx pkts=(\d+), bytes=(\d+), drop=(\d+), errs=(\d+)"
    )
    port_pattern_tx = r"tx pkts=(\d+), bytes=(\d+), drop=(\d+), errs=(\d+)"

    lines = output.split("\n")
    current_port = None
    rx_data = None

    for line in lines:
        rx_match = re.search(port_pattern_rx, line)
        if rx_match:
            current_port = rx_match.group(1)
            rx_data = {
                "rx_pkts": int(rx_match.group(2)),
                "rx_bytes": int(rx_match.group(3)),
                "rx_drop": int(rx_match.group(4)),
                "rx_errs": int(rx_match.group(5)),
            }
            continue

        if current_port and rx_data:
            tx_match = re.search(port_pattern_tx, line)
            if tx_match:
                stats.append(
                    SwitchPortStats(
                        timestamp=timestamp,
                        switch=switch_name,
                        port=current_port,
                        rx_pkts=rx_data["rx_pkts"],
                        rx_bytes=rx_data["rx_bytes"],
                        rx_drop=rx_data["rx_drop"],
                        rx_errs=rx_data["rx_errs"],
                        tx_pkts=int(tx_match.group(1)),
                        tx_bytes=int(tx_match.group(2)),
                        tx_drop=int(tx_match.group(3)),
                        tx_errs=int(tx_match.group(4)),
                    )
                )
                current_port = None
                rx_data = None

    return stats


def parse_switch_stats_file(filepath: str) -> tp.List[SwitchPortStats]:
    """
    Parse a switch statistics log file.

    Args:
        filepath: Path to the switch stats log file

    Returns:
        List of all SwitchPortStats entries from the file
    """
    stats_data = []

    filename = os.path.basename(filepath)
    switch_name = filename.replace("switch_stats_", "").replace(".log", "")

    with open(filepath, "r") as f:
        content = f.read()

    snapshots = content.split("END_SNAPSHOT\n")
    for snap in snapshots:
        if not snap.strip():
            continue

        lines = snap.split("\n")
        timestamp = 0.0
        if lines[0].startswith("TIMESTAMP:"):
            timestamp = float(lines[0].split(":")[1])

        stats = parse_switch_port_stats(snap, switch_name, timestamp)
        stats_data.extend(stats)

    return stats_data


def parse_all_switch_stats(output_dir: str) -> tp.List[SwitchPortStats]:
    """
    Parse all switch statistics log files in a directory.

    Args:
        output_dir: Directory containing switch stats log files

    Returns:
        List of all SwitchPortStats entries
    """
    all_stats = []

    for filename in os.listdir(output_dir):
        if filename.startswith("switch_stats_") and filename.endswith(".log"):
            filepath = os.path.join(output_dir, filename)
            stats = parse_switch_stats_file(filepath)
            all_stats.extend(stats)

    return all_stats


def aggregate_switch_stats(
    raw_stats: tp.List[SwitchPortStats],
) -> tp.Dict[int, AggregatedSwitchStats]:
    """
    Aggregate switch port statistics into per-second deltas.

    Args:
        raw_stats: List of raw switch port statistics

    Returns:
        Dictionary mapping relative seconds to aggregated stats
    """
    if not raw_stats:
        return {}

    timestamps = sorted(list(set(s.timestamp for s in raw_stats)))
    if not timestamps:
        return {}

    start_time_ref = timestamps[0]

    # Group by timestamp
    grouped_by_ts: tp.Dict[float, tp.List[SwitchPortStats]] = {
        t: [] for t in timestamps
    }
    for entry in raw_stats:
        grouped_by_ts[entry.timestamp].append(entry)

    # Calculate aggregates per timestamp
    aggregates: tp.Dict[float, AggregatedSwitchStats] = {}
    for t in timestamps:
        agg = AggregatedSwitchStats()
        for entry in grouped_by_ts[t]:
            agg.net_rx_pkts += entry.rx_pkts
            agg.net_rx_bytes += entry.rx_bytes
            agg.net_rx_drop += entry.rx_drop
            agg.net_rx_errs += entry.rx_errs
            agg.net_tx_pkts += entry.tx_pkts
            agg.net_tx_bytes += entry.tx_bytes
            agg.net_tx_drop += entry.tx_drop
            agg.net_tx_errs += entry.tx_errs
        aggregates[t] = agg

    # Calculate deltas between consecutive timestamps
    switch_deltas: tp.Dict[int, AggregatedSwitchStats] = {}
    for i in range(1, len(timestamps)):
        t_curr = timestamps[i]
        t_prev = timestamps[i - 1]

        curr = aggregates[t_curr]
        prev = aggregates[t_prev]

        delta = AggregatedSwitchStats(
            net_rx_pkts=max(0, curr.net_rx_pkts - prev.net_rx_pkts),
            net_rx_bytes=max(0, curr.net_rx_bytes - prev.net_rx_bytes),
            net_rx_drop=max(0, curr.net_rx_drop - prev.net_rx_drop),
            net_rx_errs=max(0, curr.net_rx_errs - prev.net_rx_errs),
            net_tx_pkts=max(0, curr.net_tx_pkts - prev.net_tx_pkts),
            net_tx_bytes=max(0, curr.net_tx_bytes - prev.net_tx_bytes),
            net_tx_drop=max(0, curr.net_tx_drop - prev.net_tx_drop),
            net_tx_errs=max(0, curr.net_tx_errs - prev.net_tx_errs),
        )

        rel_time = int(round(t_curr - start_time_ref))
        switch_deltas[rel_time] = delta

    return switch_deltas


def parse_tcpdump_packet(line: str) -> tp.Optional[TcpdumpPacketInfo]:
    """
    Parse a tcpdump verbose output line.

    Handles various packet types (TCP, UDP, ICMP) with detailed field extraction.

    Args:
        line: Raw tcpdump output line

    Returns:
        TcpdumpPacketInfo object or None if parsing fails
    """
    # Timestamp pattern: HH:MM:SS.microseconds or epoch timestamp
    ts_pattern = r"(\d+:\d+:\d+\.\d+|\d+\.\d+)"

    # IP packet pattern
    ip_pattern = (
        r"IP\s+(?:\(tos[^)]+\)\s+)?"
        r"(\d+\.\d+\.\d+\.\d+)(?:\.(\d+))?\s*>\s*"
        r"(\d+\.\d+\.\d+\.\d+)(?:\.(\d+))?"
    )

    # Length pattern
    length_pattern = r"length\s+(\d+)"

    # TTL pattern
    ttl_pattern = r"ttl\s+(\d+)"

    # Try to extract timestamp
    ts_match = re.search(ts_pattern, line)
    if not ts_match:
        return None

    # Parse timestamp
    ts_str = ts_match.group(1)
    if ":" in ts_str:
        # HH:MM:SS.microseconds format - convert to relative
        parts = ts_str.split(":")
        hours, mins = int(parts[0]), int(parts[1])
        secs = float(parts[2])
        timestamp = hours * 3600 + mins * 60 + secs
    else:
        timestamp = float(ts_str)

    # Extract IP addresses and ports
    ip_match = re.search(ip_pattern, line)
    if not ip_match:
        return None

    src_ip = ip_match.group(1)
    src_port = int(ip_match.group(2)) if ip_match.group(2) else None
    dst_ip = ip_match.group(3)
    dst_port = int(ip_match.group(4)) if ip_match.group(4) else None

    # Extract length
    length = 0
    length_match = re.search(length_pattern, line)
    if length_match:
        length = int(length_match.group(1))

    # Extract TTL
    ttl = None
    ttl_match = re.search(ttl_pattern, line)
    if ttl_match:
        ttl = int(ttl_match.group(1))

    # Determine protocol and extract protocol-specific fields
    protocol = "IP"
    tcp_flags = None
    tcp_seq = None
    tcp_ack = None
    tcp_window = None
    icmp_type = None
    icmp_code = None

    if "UDP" in line or "udp" in line:
        protocol = "UDP"
    elif "Flags [" in line:
        protocol = "TCP"
        # Extract TCP flags
        flags_match = re.search(r"Flags \[([^\]]+)\]", line)
        if flags_match:
            tcp_flags = flags_match.group(1)
        # Extract seq
        seq_match = re.search(r"seq\s+(\d+)", line)
        if seq_match:
            tcp_seq = int(seq_match.group(1))
        # Extract ack
        ack_match = re.search(r"ack\s+(\d+)", line)
        if ack_match:
            tcp_ack = int(ack_match.group(1))
        # Extract window
        win_match = re.search(r"win\s+(\d+)", line)
        if win_match:
            tcp_window = int(win_match.group(1))
    elif "ICMP" in line or "icmp" in line:
        protocol = "ICMP"
        icmp_match = re.search(r"ICMP.*type\s+(\d+)", line)
        if icmp_match:
            icmp_type = int(icmp_match.group(1))

    return TcpdumpPacketInfo(
        timestamp=timestamp,
        src_ip=src_ip,
        dst_ip=dst_ip,
        src_port=src_port,
        dst_port=dst_port,
        protocol=protocol,
        length=length,
        ttl=ttl,
        tcp_flags=tcp_flags,
        tcp_seq=tcp_seq,
        tcp_ack=tcp_ack,
        tcp_window=tcp_window,
        icmp_type=icmp_type,
        icmp_code=icmp_code,
    )


def parse_tcpdump_file(filepath: str) -> tp.List[TcpdumpPacketInfo]:
    """
    Parse a tcpdump output file.

    Args:
        filepath: Path to tcpdump output file

    Returns:
        List of TcpdumpPacketInfo objects
    """
    packets = []

    if not os.path.exists(filepath):
        logger.warning(f"Tcpdump file not found: {filepath}")
        return packets

    with open(filepath, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            packet = parse_tcpdump_packet(line)
            if packet:
                packets.append(packet)

    return packets


def aggregate_tcpdump_flow_stats(
    packets: tp.List[TcpdumpPacketInfo],
    flow_id: str,
) -> TcpdumpFlowStats:
    """
    Aggregate packet-level data into flow statistics.

    Args:
        packets: List of packets for this flow
        flow_id: Flow identifier

    Returns:
        TcpdumpFlowStats with aggregated statistics
    """
    if not packets:
        return TcpdumpFlowStats(flow_id=flow_id)

    stats = TcpdumpFlowStats(flow_id=flow_id)
    stats.total_packets = len(packets)
    stats.total_bytes = sum(p.length for p in packets)

    timestamps = [p.timestamp for p in packets]
    stats.start_time = min(timestamps)
    stats.end_time = max(timestamps)

    # Packet size statistics
    sizes = [p.length for p in packets if p.length > 0]
    if sizes:
        stats.min_packet_size = min(sizes)
        stats.max_packet_size = max(sizes)
        stats.avg_packet_size = sum(sizes) / len(sizes)

    # Inter-packet times
    sorted_ts = sorted(timestamps)
    if len(sorted_ts) > 1:
        stats.inter_packet_times = [
            sorted_ts[i + 1] - sorted_ts[i] for i in range(len(sorted_ts) - 1)
        ]

    # Count TCP-specific metrics
    tcp_seqs = set()
    prev_seq = None
    for p in packets:
        if p.protocol == "TCP" and p.tcp_seq is not None:
            if p.tcp_seq in tcp_seqs:
                stats.retransmissions += 1
            else:
                tcp_seqs.add(p.tcp_seq)
            if prev_seq is not None and p.tcp_seq < prev_seq:
                stats.out_of_order_packets += 1
            prev_seq = p.tcp_seq

    return stats
