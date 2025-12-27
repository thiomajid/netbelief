"""
VoIP Network Simulation with Hydra configuration.

This script simulates VoIP traffic over a Mininet network topology,
collecting comprehensive metrics including:
- Bandwidth, jitter, packet loss (iperf)
- Round-trip time (ping)
- Switch port statistics (OVS)
- Deep packet inspection (tcpdump)

The simulation uses a modular architecture with components in src/simulation/.
"""

import time
import typing as tp

import hydra
from hydra.core.config_store import ConfigStore
from loguru import logger
from mininet.link import TCLink
from mininet.net import Mininet
from mininet.node import Node, OVSController, OVSKernelSwitch
from omegaconf import DictConfig, OmegaConf
from tqdm import tqdm

from src.simulation import (
    DataCollectorManager,
    NetworkVariabilityManager,
    cleanup_network,
    ensure_dir,
    load_topology_class,
    process_and_export,
    setup_background_pairs,
    setup_voip_pairs,
)
from src.simulation.variability import VariabilityConfig
from src.simulation_config import VoIPSimulationConfig

# Register the config with Hydra
cs = ConfigStore.instance()
cs.store(name="voip_simulation_config", node=VoIPSimulationConfig)


def start_iperf_servers(
    pairs: tp.List[tp.Tuple[Node, Node]],
    output_dir: str,
    polling_interval: float,
) -> tp.List:
    """
    Start iperf UDP servers for all VoIP pairs.

    Args:
        pairs: List of (host1, host2) VoIP pairs
        output_dir: Directory for log files
        polling_interval: Reporting interval for iperf

    Returns:
        List of server Popen processes
    """
    servers = []
    for h1, h2 in pairs:
        servers.append(
            h1.popen(
                f"iperf -s -u -i {polling_interval} > {output_dir}/iperf_server_{h1.name}.log",
                shell=True,
            )
        )
        servers.append(
            h2.popen(
                f"iperf -s -u -i {polling_interval} > {output_dir}/iperf_server_{h2.name}.log",
                shell=True,
            )
        )
    return servers


def start_background_servers(
    bg_pairs: tp.List[tp.Tuple[Node, Node]],
    output_dir: str,
    polling_interval: float,
) -> tp.List:
    """
    Start iperf TCP servers for background traffic.

    Args:
        bg_pairs: List of (server, client) background pairs
        output_dir: Directory for log files
        polling_interval: Reporting interval

    Returns:
        List of server Popen processes
    """
    servers = []
    for s, _ in bg_pairs:
        servers.append(
            s.popen(
                f"iperf -s -i {polling_interval} > {output_dir}/iperf_bg_server_{s.name}.log",
                shell=True,
            )
        )
    return servers


def start_voip_clients(
    pairs: tp.List[tp.Tuple[Node, Node]],
    output_dir: str,
    duration: int,
    voip_bw: str,
    polling_interval: float,
) -> tp.Tuple[tp.List, tp.List]:
    """
    Start VoIP iperf clients and ping measurements.

    Establishes bidirectional traffic between all VoIP pairs.

    Args:
        pairs: List of (host1, host2) VoIP pairs
        output_dir: Directory for log files
        duration: Simulation duration in seconds
        voip_bw: Bandwidth string (e.g., "100k")
        polling_interval: Measurement interval

    Returns:
        Tuple of (client processes, ping processes)
    """
    clients = []
    pings = []
    ping_count = int(duration / polling_interval)

    for h1, h2 in pairs:
        # h1 -> h2
        clients.append(
            h1.popen(
                f"iperf -c {h2.IP()} -u -b {voip_bw} -t {duration} -i {polling_interval} "
                f"> {output_dir}/iperf_client_{h1.name}_to_{h2.name}.log",
                shell=True,
            )
        )
        pings.append(
            h1.popen(
                f"ping {h2.IP()} -i {polling_interval} -D -c {ping_count} "
                f"> {output_dir}/ping_{h1.name}_to_{h2.name}.log",
                shell=True,
            )
        )

        # h2 -> h1
        clients.append(
            h2.popen(
                f"iperf -c {h1.IP()} -u -b {voip_bw} -t {duration} -i {polling_interval} "
                f"> {output_dir}/iperf_client_{h2.name}_to_{h1.name}.log",
                shell=True,
            )
        )
        pings.append(
            h2.popen(
                f"ping {h1.IP()} -i {polling_interval} -D -c {ping_count} "
                f"> {output_dir}/ping_{h2.name}_to_{h1.name}.log",
                shell=True,
            )
        )

    return clients, pings


def start_background_clients(
    bg_pairs: tp.List[tp.Tuple[Node, Node]],
    output_dir: str,
    duration: int,
    polling_interval: float,
) -> tp.List:
    """
    Start background TCP traffic clients.

    Args:
        bg_pairs: List of (server, client) pairs
        output_dir: Directory for log files
        duration: Traffic duration
        polling_interval: Reporting interval

    Returns:
        List of client Popen processes
    """
    clients = []
    for s, c in bg_pairs:
        clients.append(
            c.popen(
                f"iperf -c {s.IP()} -t {duration} -i {polling_interval} "
                f"> {output_dir}/iperf_bg_client_{c.name}_to_{s.name}.log",
                shell=True,
            )
        )
    return clients


def run_simulation(cfg: VoIPSimulationConfig) -> None:
    """
    Run the VoIP network simulation.

    This is the main simulation loop that:
    1. Initializes the network topology
    2. Sets up traffic pairs
    3. Starts data collection
    4. Optionally starts network variability simulation
    5. Runs traffic for the specified duration
    6. Processes and exports results

    Args:
        cfg: Simulation configuration
    """
    output_dir = cfg.simulation.output_dir
    ensure_dir(output_dir)

    logger.info(f"Starting VoIP Simulation (Duration: {cfg.simulation.duration}s)")
    logger.info(f"Polling interval: {cfg.simulation.polling_interval}s")
    logger.info(f"Link capacity: {cfg.simulation.link_capacity_bps} bps")

    # Load topology dynamically
    TopoClass = load_topology_class(
        cfg.topology.module_name,
        cfg.topology.class_name,
    )

    # Initialize network
    topo = TopoClass()
    net = Mininet(
        topo=topo,
        controller=OVSController,
        link=TCLink,
        switch=OVSKernelSwitch,
    )

    # Track VoIP pairs for later processing
    voip_pairs: tp.List[tp.Tuple[Node, Node]] = []

    # Track variability manager for cleanup
    variability_manager: tp.Optional[NetworkVariabilityManager] = None

    try:
        net.start()

        # Enable STP on all switches
        logger.info("Enabling STP on switches")
        for sw in net.switches:
            sw.cmd(f"ovs-vsctl set bridge {sw.name} stp_enable=true")

        logger.info("Waiting for switch connections")
        net.waitConnected(timeout=cfg.simulation.stp_convergence_time)

        # Wait for STP convergence
        logger.info(
            f"Waiting for STP convergence ({cfg.simulation.stp_convergence_time}s)..."
        )
        time.sleep(cfg.simulation.stp_convergence_time)

        hosts = net.hosts
        if len(hosts) < 2:
            logger.error("Not enough hosts for simulation")
            net.stop()
            return

        # Setup VoIP traffic pairs
        voip_pairs = setup_voip_pairs(
            hosts,
            cfg.traffic.voip_flow_count,
            cfg.simulation.seed,
        )
        logger.info(f"Setting up {len(voip_pairs)} VoIP pairs")

        # Setup background traffic pairs
        bg_pairs = setup_background_pairs(
            hosts,
            cfg.traffic.background_traffic_count,
            cfg.simulation.seed,
        )

        # Initialize data collector manager
        collector = DataCollectorManager(
            net=net,
            output_dir=output_dir,
            polling_interval=cfg.simulation.polling_interval,
        )

        # Start all data collection (switch stats, interface stats, tcpdump)
        collector.start_all(voip_pairs)

        # Initialize and start network variability if enabled
        if cfg.variability.enabled:
            variability_config = _build_variability_config(cfg.variability)
            variability_manager = NetworkVariabilityManager(
                net=net,
                config=variability_config,
                output_dir=output_dir,
            )
            variability_manager.start()
            logger.info("Network variability simulation enabled")

        # Start iperf servers
        servers = start_iperf_servers(
            voip_pairs,
            output_dir,
            cfg.simulation.polling_interval,
        )
        servers.extend(
            start_background_servers(
                bg_pairs,
                output_dir,
                cfg.simulation.polling_interval,
            )
        )

        logger.info("Starting traffic generation")
        time.sleep(2)  # Wait for servers to start

        # Start VoIP clients (bidirectional)
        clients, pings = start_voip_clients(
            voip_pairs,
            output_dir,
            cfg.simulation.duration,
            cfg.traffic.voip_bandwidth,
            cfg.simulation.polling_interval,
        )

        # Start background traffic
        bg_clients = start_background_clients(
            bg_pairs,
            output_dir,
            cfg.simulation.duration,
            cfg.simulation.polling_interval,
        )
        clients.extend(bg_clients)

        logger.info(f"Running simulation for {cfg.simulation.duration} seconds...")

        # Progress bar for simulation duration
        for _ in tqdm(
            range(cfg.simulation.duration), desc="Simulation Progress", unit="s"
        ):
            time.sleep(1)

        logger.info("Simulation finished. Stopping data collection.")

        # Stop variability simulation if running
        if variability_manager is not None:
            variability_manager.stop()

        # Stop data collection
        collector.stop_all()

        # Terminate ping processes
        for p in pings:
            try:
                p.terminate()
            except Exception:
                pass

    finally:
        net.stop()
        cleanup_network()

    # Process logs and export to CSV
    logger.info("Processing logs...")
    process_and_export(
        voip_pairs,
        output_dir,
        cfg.simulation.polling_interval,
        cfg.simulation.link_capacity_bps,
    )


def _build_variability_config(cfg) -> VariabilityConfig:
    """
    Build a VariabilityConfig from the Hydra/OmegaConf configuration.

    Args:
        cfg: Variability section from the simulation config

    Returns:
        VariabilityConfig instance with all nested configurations
    """
    from src.simulation.variability import (
        BandwidthVariationConfig,
        CongestionConfig,
        DelayVariationConfig,
        LinkFailureConfig,
        PacketLossConfig,
    )

    # Convert excluded_links and target_links to tuple format
    excluded = [tuple(link) for link in (cfg.excluded_links or [])]
    target = [tuple(link) for link in (cfg.target_links or [])]

    return VariabilityConfig(
        enabled=cfg.enabled,
        seed=cfg.seed,
        link_scope=cfg.link_scope,
        excluded_links=excluded,
        target_links=target,
        default_bandwidth=getattr(cfg, "default_bandwidth", 10.0),
        default_delay=getattr(cfg, "default_delay", "1ms"),
        link_failure=LinkFailureConfig(
            enabled=cfg.link_failure.enabled,
            probability=cfg.link_failure.probability,
            min_interval=cfg.link_failure.min_interval,
            max_interval=cfg.link_failure.max_interval,
            min_duration=cfg.link_failure.min_duration,
            max_duration=cfg.link_failure.max_duration,
            max_concurrent_failures=cfg.link_failure.max_concurrent_failures,
        ),
        delay_variation=DelayVariationConfig(
            enabled=cfg.delay_variation.enabled,
            probability=cfg.delay_variation.probability,
            min_interval=cfg.delay_variation.min_interval,
            max_interval=cfg.delay_variation.max_interval,
            min_multiplier=cfg.delay_variation.min_multiplier,
            max_multiplier=cfg.delay_variation.max_multiplier,
            revert_after=cfg.delay_variation.revert_after,
        ),
        bandwidth_variation=BandwidthVariationConfig(
            enabled=cfg.bandwidth_variation.enabled,
            probability=cfg.bandwidth_variation.probability,
            min_interval=cfg.bandwidth_variation.min_interval,
            max_interval=cfg.bandwidth_variation.max_interval,
            min_fraction=cfg.bandwidth_variation.min_fraction,
            max_fraction=cfg.bandwidth_variation.max_fraction,
            revert_after=cfg.bandwidth_variation.revert_after,
        ),
        packet_loss=PacketLossConfig(
            enabled=cfg.packet_loss.enabled,
            probability=cfg.packet_loss.probability,
            min_interval=cfg.packet_loss.min_interval,
            max_interval=cfg.packet_loss.max_interval,
            min_loss=cfg.packet_loss.min_loss,
            max_loss=cfg.packet_loss.max_loss,
            revert_after=cfg.packet_loss.revert_after,
        ),
        congestion=CongestionConfig(
            enabled=cfg.congestion.enabled,
            probability=cfg.congestion.probability,
            min_interval=cfg.congestion.min_interval,
            max_interval=cfg.congestion.max_interval,
            min_duration=cfg.congestion.min_duration,
            max_duration=cfg.congestion.max_duration,
            bandwidth_reduction=cfg.congestion.bandwidth_reduction,
            additional_delay_ms=cfg.congestion.additional_delay_ms,
            loss_percent=cfg.congestion.loss_percent,
        ),
    )


@hydra.main(
    version_base="1.2",
    config_path="../config",
    config_name="simulation_with_variability",
)
def main(cfg: DictConfig) -> None:
    """Main entry point with Hydra configuration."""
    # Convert OmegaConf to the dataclass
    schema = OmegaConf.structured(VoIPSimulationConfig)
    cfg = OmegaConf.merge(schema, cfg)

    logger.info("Configuration:")
    logger.info(OmegaConf.to_yaml(cfg))

    # Convert to dataclass for type safety
    config = VoIPSimulationConfig(
        topology=cfg.topology,
        traffic=cfg.traffic,
        simulation=cfg.simulation,
        variability=cfg.variability,
    )

    logger.info("Cleaning previous mininet run state")
    cleanup_network()

    run_simulation(config)


if __name__ == "__main__":
    main()
