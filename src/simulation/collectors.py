"""
Data collectors for VoIP network simulation.

This module provides background data collection from various sources:
- Switch statistics via OVS
- Packet capture via tcpdump
- Network interface statistics
"""

import os
import subprocess
import threading
import time
import typing as tp

from loguru import logger
from mininet.net import Mininet


def collect_switch_stats(
    net: Mininet,
    stop_event: threading.Event,
    interval: float,
    output_dir: str,
) -> None:
    """
    Collect switch statistics in a background thread.

    Periodically queries all OVS switches for port statistics
    and writes them to log files for later parsing.

    Args:
        net: Mininet network object
        stop_event: Event to signal collection should stop
        interval: Collection interval in seconds
        output_dir: Directory to write log files
    """
    logger.info(f"Starting switch stats collection (interval: {interval}s)...")

    while not stop_event.is_set():
        timestamp = time.time()
        for sw in net.switches:
            try:
                out = sw.cmd(f"ovs-ofctl dump-ports {sw.name}")
                with open(f"{output_dir}/switch_stats_{sw.name}.log", "a") as f:
                    f.write(f"TIMESTAMP:{timestamp}\n")
                    f.write(out)
                    f.write("END_SNAPSHOT\n")
            except Exception as e:
                logger.error(f"Error collecting stats for {sw.name}: {e}")

        if stop_event.wait(interval):
            break


def start_tcpdump_capture(
    interface: str,
    output_file: str,
    filter_expr: str = "",
    packet_count: tp.Optional[int] = None,
    snapshot_len: int = 96,
    verbose: bool = True,
) -> subprocess.Popen:
    """
    Start a tcpdump packet capture process.

    Args:
        interface: Network interface to capture on
        output_file: File to write captured packets
        filter_expr: BPF filter expression (e.g., "udp port 5001")
        packet_count: Maximum packets to capture (None for unlimited)
        snapshot_len: Bytes to capture per packet (default 96 for headers)
        verbose: Enable verbose output with detailed packet info

    Returns:
        Subprocess Popen object for the tcpdump process
    """
    cmd = ["tcpdump", "-i", interface, "-s", str(snapshot_len)]

    if verbose:
        # -tttt: Print timestamp in default format with date
        # -nn: Don't resolve hostnames or ports
        # -v: Verbose output (includes TTL, etc.)
        cmd.extend(["-tttt", "-nn", "-v"])

    if packet_count:
        cmd.extend(["-c", str(packet_count)])

    if filter_expr:
        cmd.append(filter_expr)

    # Write to file
    with open(output_file, "w") as f:
        proc = subprocess.Popen(
            cmd,
            stdout=f,
            stderr=subprocess.DEVNULL,
            text=True,
        )

    logger.debug(f"Started tcpdump on {interface}: {' '.join(cmd)}")
    return proc


def start_tcpdump_on_host(
    host,
    output_dir: str,
    filter_expr: str = "udp or icmp",
    duration: tp.Optional[int] = None,
) -> subprocess.Popen:
    """
    Start tcpdump capture on a Mininet host.

    Uses the host's default interface for capture.

    Args:
        host: Mininet host object
        output_dir: Directory to write capture file
        filter_expr: BPF filter expression
        duration: Optional capture duration (handled by caller)

    Returns:
        Popen object for the tcpdump process
    """
    interface = host.defaultIntf()
    output_file = os.path.join(output_dir, f"tcpdump_{host.name}.log")

    # Build command for Mininet host execution
    cmd = (
        f"tcpdump -i {interface} -tttt -nn -v -s 256 {filter_expr} > {output_file} 2>&1"
    )

    proc = host.popen(cmd, shell=True)
    logger.debug(f"Started tcpdump on host {host.name} interface {interface}")

    return proc


def start_tcpdump_on_switch_ports(
    net: Mininet,
    output_dir: str,
    filter_expr: str = "",
) -> tp.List[subprocess.Popen]:
    """
    Start tcpdump on all switch ports.

    Captures traffic on all switch interfaces for comprehensive analysis.

    Args:
        net: Mininet network object
        output_dir: Directory for capture files
        filter_expr: BPF filter expression

    Returns:
        List of Popen objects for all tcpdump processes
    """
    processes = []

    for sw in net.switches:
        for intf in sw.intfList():
            if intf.name == "lo":
                continue

            output_file = os.path.join(
                output_dir, f"tcpdump_switch_{sw.name}_{intf.name}.log"
            )
            cmd = f"tcpdump -i {intf.name} -tttt -nn -v -s 256 {filter_expr}"

            with open(output_file, "w") as f:
                proc = subprocess.Popen(
                    cmd.split(),
                    stdout=f,
                    stderr=subprocess.DEVNULL,
                    text=True,
                )
                processes.append(proc)

            logger.debug(f"Started tcpdump on switch {sw.name} port {intf.name}")

    return processes


def start_all_tcpdump_captures(
    net: Mininet,
    output_dir: str,
    voip_pairs: tp.List[tp.Tuple],
    capture_filter: str = "udp or icmp or tcp",
) -> tp.List[subprocess.Popen]:
    """
    Start comprehensive tcpdump captures across the network.

    Captures on all hosts involved in VoIP traffic for maximum insight.

    Args:
        net: Mininet network object
        output_dir: Directory for capture files
        voip_pairs: List of (host1, host2) VoIP pairs
        capture_filter: BPF filter for captures

    Returns:
        List of all tcpdump Popen processes
    """
    processes = []

    # Get unique hosts from VoIP pairs
    captured_hosts = set()
    for h1, h2 in voip_pairs:
        captured_hosts.add(h1)
        captured_hosts.add(h2)

    # Start capture on each unique host
    for host in captured_hosts:
        try:
            proc = start_tcpdump_on_host(
                host,
                output_dir,
                filter_expr=capture_filter,
            )
            processes.append(proc)
        except Exception as e:
            logger.error(f"Failed to start tcpdump on {host.name}: {e}")

    logger.info(f"Started {len(processes)} tcpdump captures")
    return processes


def stop_tcpdump_captures(processes: tp.List[subprocess.Popen]) -> None:
    """
    Stop all tcpdump capture processes.

    Args:
        processes: List of Popen objects to terminate
    """
    for proc in processes:
        try:
            proc.terminate()
            proc.wait(timeout=2)
        except subprocess.TimeoutExpired:
            proc.kill()
        except Exception as e:
            logger.warning(f"Error stopping tcpdump process: {e}")

    # Also kill any remaining tcpdump processes
    os.system("pkill -9 tcpdump 2>/dev/null")
    logger.info(f"Stopped {len(processes)} tcpdump captures")


def collect_interface_stats(
    net: Mininet,
    stop_event: threading.Event,
    interval: float,
    output_dir: str,
) -> None:
    """
    Collect network interface statistics in a background thread.

    Captures /proc/net/dev statistics for all hosts for additional insight.

    Args:
        net: Mininet network object
        stop_event: Event to signal collection should stop
        interval: Collection interval in seconds
        output_dir: Directory to write log files
    """
    logger.info("Starting interface stats collection...")

    while not stop_event.is_set():
        timestamp = time.time()

        for host in net.hosts:
            try:
                # Get interface stats from /proc/net/dev
                out = host.cmd("cat /proc/net/dev")
                with open(f"{output_dir}/interface_stats_{host.name}.log", "a") as f:
                    f.write(f"TIMESTAMP:{timestamp}\n")
                    f.write(out)
                    f.write("END_SNAPSHOT\n")
            except Exception as e:
                logger.error(f"Error collecting interface stats for {host.name}: {e}")

        if stop_event.wait(interval):
            break


class DataCollectorManager:
    """
    Manager class for coordinating multiple data collection threads.

    Provides a unified interface for starting and stopping all
    data collection activities.
    """

    def __init__(self, net: Mininet, output_dir: str, polling_interval: float = 0.2):
        """
        Initialize the collector manager.

        Args:
            net: Mininet network object
            output_dir: Directory for output files
            polling_interval: Default collection interval
        """
        self.net = net
        self.output_dir = output_dir
        self.polling_interval = polling_interval

        self._stop_event = threading.Event()
        self._threads: tp.List[threading.Thread] = []
        self._tcpdump_processes: tp.List[subprocess.Popen] = []

    def start_switch_stats_collection(self) -> None:
        """Start switch statistics collection thread."""
        thread = threading.Thread(
            target=collect_switch_stats,
            args=(
                self.net,
                self._stop_event,
                self.polling_interval,
                self.output_dir,
            ),
            daemon=True,
        )
        thread.start()
        self._threads.append(thread)
        logger.info("Switch stats collection started")

    def start_interface_stats_collection(self) -> None:
        """Start interface statistics collection thread."""
        thread = threading.Thread(
            target=collect_interface_stats,
            args=(
                self.net,
                self._stop_event,
                self.polling_interval,
                self.output_dir,
            ),
            daemon=True,
        )
        thread.start()
        self._threads.append(thread)
        logger.info("Interface stats collection started")

    def start_tcpdump_captures(
        self,
        voip_pairs: tp.List[tp.Tuple],
        capture_filter: str = "udp or icmp",
    ) -> None:
        """
        Start tcpdump captures for VoIP pairs.

        Args:
            voip_pairs: List of (host1, host2) VoIP pairs
            capture_filter: BPF filter expression
        """
        self._tcpdump_processes = start_all_tcpdump_captures(
            self.net,
            self.output_dir,
            voip_pairs,
            capture_filter,
        )

    def start_all(self, voip_pairs: tp.List[tp.Tuple]) -> None:
        """
        Start all data collection activities.

        Args:
            voip_pairs: List of VoIP host pairs for tcpdump
        """
        self.start_switch_stats_collection()
        self.start_interface_stats_collection()
        self.start_tcpdump_captures(voip_pairs)
        logger.info("All data collectors started")

    def stop_all(self) -> None:
        """Stop all data collection activities."""
        # Signal threads to stop
        self._stop_event.set()

        # Wait for threads to finish
        for thread in self._threads:
            thread.join(timeout=5)

        # Stop tcpdump processes
        stop_tcpdump_captures(self._tcpdump_processes)

        logger.info("All data collectors stopped")
