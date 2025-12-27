"""
Network utility functions for VoIP simulation.

This module provides helper functions for network setup,
topology loading, and traffic generation.
"""

import importlib
import os
import random
import typing as tp

from loguru import logger
from mininet.node import Node


def load_topology_class(module_name: str, class_name: str) -> type:
    """
    Dynamically load a topology class from the pytopo package.

    Args:
        module_name: Name of the module in pytopo (e.g., "airtel")
        class_name: Name of the class to import (e.g., "GeneratedTopo")

    Returns:
        The topology class

    Raises:
        ImportError: If module or class cannot be found
    """
    module_path = f"pytopo.{module_name}"
    try:
        module = importlib.import_module(module_path)
        topo_class = getattr(module, class_name)
        logger.info(f"Loaded topology class '{class_name}' from '{module_path}'")
        return topo_class
    except ModuleNotFoundError as e:
        raise ImportError(f"Could not find module '{module_path}': {e}")
    except AttributeError as e:
        raise ImportError(
            f"Could not find class '{class_name}' in module '{module_path}': {e}"
        )


def ensure_dir(directory: str) -> None:
    """
    Create directory if it doesn't exist.

    Args:
        directory: Path to directory to create
    """
    if not os.path.exists(directory):
        os.makedirs(directory)


def setup_voip_pairs(
    hosts: tp.List[Node],
    voip_flow_count: int,
    seed: tp.Optional[int] = None,
) -> tp.List[tp.Tuple[Node, Node]]:
    """
    Setup VoIP traffic pairs from available hosts.

    Randomly pairs hosts for bidirectional VoIP traffic.

    Args:
        hosts: List of available Mininet hosts
        voip_flow_count: Minimum number of VoIP flows
        seed: Random seed for reproducibility

    Returns:
        List of (host1, host2) tuples for VoIP pairs
    """
    if seed is not None:
        random.seed(seed)

    available_hosts = list(hosts)
    random.shuffle(available_hosts)

    num_hosts = len(available_hosts)
    foreground_flow_count = max(voip_flow_count, num_hosts // 2)

    pairs = []
    for _ in range(foreground_flow_count):
        if len(available_hosts) >= 2:
            h1 = available_hosts.pop()
            h2 = available_hosts.pop()
            pairs.append((h1, h2))

    logger.info(f"Created {len(pairs)} VoIP pairs")
    return pairs


def setup_background_pairs(
    hosts: tp.List[Node],
    background_count: int,
    seed: tp.Optional[int] = None,
) -> tp.List[tp.Tuple[Node, Node]]:
    """
    Setup background traffic pairs.

    Args:
        hosts: List of available Mininet hosts
        background_count: Number of background flows
        seed: Random seed for reproducibility

    Returns:
        List of (server, client) tuples for background traffic
    """
    if seed is not None:
        random.seed(seed)

    bg_hosts = random.sample(
        hosts,
        min(len(hosts), background_count * 2),
    )

    pairs = []
    for i in range(0, len(bg_hosts) - 1, 2):
        pairs.append((bg_hosts[i], bg_hosts[i + 1]))

    logger.info(f"Created {len(pairs)} background traffic pairs")
    return pairs


def cleanup_network() -> None:
    """
    Clean up network resources after simulation.

    Kills remaining iperf and tcpdump processes and
    runs Mininet cleanup.
    """
    os.system("pkill -9 iperf 2>/dev/null")
    os.system("pkill -9 tcpdump 2>/dev/null")
    os.system("mn -c 2>/dev/null")
    logger.info("Network cleanup completed")
