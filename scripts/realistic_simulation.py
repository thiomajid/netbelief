import time
import typing as tp

import hydra
from hydra.core.config_store import ConfigStore
from loguru import logger
from mininet.link import TCLink
from mininet.net import Mininet
from mininet.node import OVSController, OVSKernelSwitch
from omegaconf import DictConfig, OmegaConf
from tqdm import tqdm

from src.simulation.collectors import DataCollectorManager
from src.simulation.network import cleanup_network, ensure_dir, load_topology_class
from src.simulation.qos_processor import process_qos_metrics
from src.simulation.realistic_processor import process_and_export_realistic
from src.simulation.realistic_traffic import RealisticTrafficGenerator
from src.simulation.variability import (
    BandwidthVariationConfig as BandwidthVariationConfigInternal,
)
from src.simulation.variability import (
    CongestionConfig as CongestionConfigInternal,
)
from src.simulation.variability import (
    DelayVariationConfig as DelayVariationConfigInternal,
)
from src.simulation.variability import (
    LinkFailureConfig as LinkFailureConfigInternal,
)
from src.simulation.variability import NetworkVariabilityManager
from src.simulation.variability import (
    PacketLossConfig as PacketLossConfigInternal,
)
from src.simulation.variability import VariabilityConfig as VariabilityConfigInternal
from src.simulation_config import (
    RealisticSimulationConfig,
    RealisticTrafficConfig,
    SimulationConfig,
    TopologyConfig,
    VariabilityConfig,
)

cs = ConfigStore.instance()
cs.store(name="realistic_simulation_config", node=RealisticSimulationConfig)


def _convert_variability_config(cfg: VariabilityConfig) -> VariabilityConfigInternal:
    """Convert dataclass VariabilityConfig to internal VariabilityConfig."""
    excluded = [tuple(link) for link in (cfg.excluded_links or [])]
    target = [tuple(link) for link in (cfg.target_links or [])]

    return VariabilityConfigInternal(
        enabled=cfg.enabled,
        seed=cfg.seed,
        link_scope=cfg.link_scope,
        excluded_links=excluded,
        target_links=target,
        default_bandwidth=cfg.default_bandwidth,
        default_delay=cfg.default_delay,
        link_failure=LinkFailureConfigInternal(
            enabled=cfg.link_failure.enabled,
            probability=cfg.link_failure.probability,
            min_interval=cfg.link_failure.min_interval,
            max_interval=cfg.link_failure.max_interval,
            min_duration=cfg.link_failure.min_duration,
            max_duration=cfg.link_failure.max_duration,
            max_concurrent_failures=cfg.link_failure.max_concurrent_failures,
        ),
        delay_variation=DelayVariationConfigInternal(
            enabled=cfg.delay_variation.enabled,
            probability=cfg.delay_variation.probability,
            min_interval=cfg.delay_variation.min_interval,
            max_interval=cfg.delay_variation.max_interval,
            min_multiplier=cfg.delay_variation.min_multiplier,
            max_multiplier=cfg.delay_variation.max_multiplier,
            revert_after=cfg.delay_variation.revert_after,
        ),
        bandwidth_variation=BandwidthVariationConfigInternal(
            enabled=cfg.bandwidth_variation.enabled,
            probability=cfg.bandwidth_variation.probability,
            min_interval=cfg.bandwidth_variation.min_interval,
            max_interval=cfg.bandwidth_variation.max_interval,
            min_fraction=cfg.bandwidth_variation.min_fraction,
            max_fraction=cfg.bandwidth_variation.max_fraction,
            revert_after=cfg.bandwidth_variation.revert_after,
        ),
        packet_loss=PacketLossConfigInternal(
            enabled=cfg.packet_loss.enabled,
            probability=cfg.packet_loss.probability,
            min_interval=cfg.packet_loss.min_interval,
            max_interval=cfg.packet_loss.max_interval,
            min_loss=cfg.packet_loss.min_loss,
            max_loss=cfg.packet_loss.max_loss,
            revert_after=cfg.packet_loss.revert_after,
        ),
        congestion=CongestionConfigInternal(
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


def run_realistic_simulation(config: RealisticSimulationConfig) -> None:
    """
    Run the realistic network simulation.

    Creates a Mininet network, starts infrastructure services,
    generates realistic traffic, and collects telemetry data.
    """
    simulation = config.simulation
    topology = config.topology
    realistic_traffic = config.realistic_traffic
    variability = config.variability

    ensure_dir(simulation.output_dir)

    logger.info("=" * 60)
    logger.info("REALISTIC NETWORK SIMULATION")
    logger.info("=" * 60)
    logger.info(f"Duration: {simulation.duration}s")
    logger.info(f"Polling interval: {simulation.polling_interval}s")
    logger.info(f"Output directory: {simulation.output_dir}")
    logger.info(f"Topology: {topology.module_name}.{topology.class_name}")

    TopoClass = load_topology_class(
        topology.module_name,
        topology.class_name,
    )

    topo = TopoClass()
    net = Mininet(
        topo=topo,
        controller=OVSController,
        link=TCLink,
        switch=OVSKernelSwitch,
    )

    variability_manager: tp.Optional[NetworkVariabilityManager] = None
    traffic_generator: tp.Optional[RealisticTrafficGenerator] = None

    try:
        net.start()
        logger.info(
            f"Network started with {len(net.hosts)} hosts and {len(net.switches)} switches"
        )

        logger.info("Enabling STP on switches")
        for sw in net.switches:
            sw.cmd(f"ovs-vsctl set bridge {sw.name} stp_enable=true")

        logger.info("Waiting for switch connections")
        net.waitConnected(timeout=simulation.stp_convergence_time)

        logger.info(
            f"Waiting for STP convergence ({simulation.stp_convergence_time}s)..."
        )
        time.sleep(simulation.stp_convergence_time)

        if len(net.hosts) < 2:
            logger.error("Not enough hosts for simulation (minimum 2 required)")
            net.stop()
            return

        logger.info("Initializing data collectors")
        collector = DataCollectorManager(
            net=net,
            output_dir=simulation.output_dir,
            polling_interval=simulation.polling_interval,
        )

        all_host_pairs = [(h, h) for h in net.hosts]
        collector.start_all(all_host_pairs)

        if variability.enabled:
            logger.info("Initializing network variability simulation")
            variability_internal = _convert_variability_config(variability)
            variability_manager = NetworkVariabilityManager(
                net=net,
                config=variability_internal,
                output_dir=simulation.output_dir,
            )
            variability_manager.start()

        logger.info("Initializing realistic traffic generator")
        traffic_generator = RealisticTrafficGenerator(
            net=net,
            output_dir=simulation.output_dir,
            seed=simulation.seed,
            server_host=realistic_traffic.server_host,
        )

        traffic_generator.start(duration=simulation.duration)

        logger.info("=" * 60)
        logger.info(f"SIMULATION RUNNING FOR {simulation.duration} SECONDS")
        logger.info("=" * 60)

        for elapsed in tqdm(
            range(simulation.duration),
            desc="Simulation Progress",
            unit="s",
        ):
            time.sleep(1)

            if elapsed > 0 and elapsed % 60 == 0:
                stats = traffic_generator.get_statistics()
                logger.info(f"[{elapsed}s] Total sessions: {stats['total_sessions']}")

        logger.info("Simulation complete. Stopping components...")

        if variability_manager is not None:
            variability_manager.stop()
            logger.info("Network variability stopped")

        if traffic_generator is not None:
            traffic_generator.stop()
            stats = traffic_generator.get_statistics()
            logger.info(
                f"Final traffic stats: {stats['total_sessions']} total sessions"
            )

        collector.stop_all()
        logger.info("Data collection stopped")

    except KeyboardInterrupt:
        logger.warning("Simulation interrupted by user")
    except Exception as e:
        logger.error(f"Simulation error: {e}")
        raise
    finally:
        net.stop()
        cleanup_network()

    logger.info("Processing collected data...")

    try:
        process_and_export_realistic(
            output_dir=simulation.output_dir,
            polling_interval=simulation.polling_interval,
        )
    except Exception as e:
        logger.error(f"Data processing error: {e}")

    # Process QoS metrics for forecasting
    logger.info("Calculating QoS metrics for forecasting...")
    try:
        process_qos_metrics(
            output_dir=simulation.output_dir,
            polling_interval=simulation.polling_interval,
            link_capacity_mbps=simulation.link_capacity_bps / 1_000_000,
        )
    except Exception as e:
        logger.error(f"QoS metrics processing error: {e}")

    logger.info("=" * 60)
    logger.info("SIMULATION COMPLETE")
    logger.info(f"Results saved to: {simulation.output_dir}")
    logger.info("=" * 60)


def _build_config_from_dict(cfg: DictConfig) -> RealisticSimulationConfig:
    """Build RealisticSimulationConfig from Hydra DictConfig."""
    from src.simulation_config import (
        BackgroundTrafficConfig,
        BandwidthVariationConfig,
        CongestionConfig,
        DelayVariationConfig,
        LinkFailureConfig,
        PacketLossConfig,
        ProfileWeightsConfig,
    )

    topology_cfg = TopologyConfig(
        module_name=cfg.topology.module_name,
        class_name=cfg.topology.class_name,
    )

    simulation_cfg = SimulationConfig(
        duration=cfg.simulation.duration,
        polling_interval=cfg.simulation.polling_interval,
        output_dir=cfg.simulation.output_dir,
        stp_convergence_time=cfg.simulation.stp_convergence_time,
        seed=cfg.simulation.seed,
        link_capacity_bps=cfg.simulation.link_capacity_bps,
    )

    profile_weights_cfg = ProfileWeightsConfig(
        casual=cfg.realistic_traffic.profile_weights.casual,
        power=cfg.realistic_traffic.profile_weights.power,
        gamer=cfg.realistic_traffic.profile_weights.gamer,
        office=cfg.realistic_traffic.profile_weights.office,
        iot=cfg.realistic_traffic.profile_weights.iot,
    )

    background_traffic_cfg = BackgroundTrafficConfig(
        enabled=cfg.realistic_traffic.background_traffic.enabled,
        min_flows=cfg.realistic_traffic.background_traffic.min_flows,
        max_flows=cfg.realistic_traffic.background_traffic.max_flows,
    )

    realistic_traffic_cfg = RealisticTrafficConfig(
        enabled=cfg.realistic_traffic.enabled,
        server_host=cfg.realistic_traffic.server_host,
        profile_weights=profile_weights_cfg,
        background_traffic=background_traffic_cfg,
    )

    link_failure_cfg = LinkFailureConfig(
        enabled=cfg.variability.link_failure.enabled,
        probability=cfg.variability.link_failure.probability,
        min_interval=cfg.variability.link_failure.min_interval,
        max_interval=cfg.variability.link_failure.max_interval,
        min_duration=cfg.variability.link_failure.min_duration,
        max_duration=cfg.variability.link_failure.max_duration,
        max_concurrent_failures=cfg.variability.link_failure.max_concurrent_failures,
    )

    delay_variation_cfg = DelayVariationConfig(
        enabled=cfg.variability.delay_variation.enabled,
        probability=cfg.variability.delay_variation.probability,
        min_interval=cfg.variability.delay_variation.min_interval,
        max_interval=cfg.variability.delay_variation.max_interval,
        min_multiplier=cfg.variability.delay_variation.min_multiplier,
        max_multiplier=cfg.variability.delay_variation.max_multiplier,
        revert_after=cfg.variability.delay_variation.revert_after,
    )

    bandwidth_variation_cfg = BandwidthVariationConfig(
        enabled=cfg.variability.bandwidth_variation.enabled,
        probability=cfg.variability.bandwidth_variation.probability,
        min_interval=cfg.variability.bandwidth_variation.min_interval,
        max_interval=cfg.variability.bandwidth_variation.max_interval,
        min_fraction=cfg.variability.bandwidth_variation.min_fraction,
        max_fraction=cfg.variability.bandwidth_variation.max_fraction,
        revert_after=cfg.variability.bandwidth_variation.revert_after,
    )

    packet_loss_cfg = PacketLossConfig(
        enabled=cfg.variability.packet_loss.enabled,
        probability=cfg.variability.packet_loss.probability,
        min_interval=cfg.variability.packet_loss.min_interval,
        max_interval=cfg.variability.packet_loss.max_interval,
        min_loss=cfg.variability.packet_loss.min_loss,
        max_loss=cfg.variability.packet_loss.max_loss,
        revert_after=cfg.variability.packet_loss.revert_after,
    )

    congestion_cfg = CongestionConfig(
        enabled=cfg.variability.congestion.enabled,
        probability=cfg.variability.congestion.probability,
        min_interval=cfg.variability.congestion.min_interval,
        max_interval=cfg.variability.congestion.max_interval,
        min_duration=cfg.variability.congestion.min_duration,
        max_duration=cfg.variability.congestion.max_duration,
        bandwidth_reduction=cfg.variability.congestion.bandwidth_reduction,
        additional_delay_ms=cfg.variability.congestion.additional_delay_ms,
        loss_percent=cfg.variability.congestion.loss_percent,
    )

    variability_cfg = VariabilityConfig(
        enabled=cfg.variability.enabled,
        seed=cfg.variability.seed,
        link_scope=cfg.variability.link_scope,
        excluded_links=list(cfg.variability.excluded_links or []),
        target_links=list(cfg.variability.target_links or []),
        default_bandwidth=cfg.variability.default_bandwidth,
        default_delay=cfg.variability.default_delay,
        link_failure=link_failure_cfg,
        delay_variation=delay_variation_cfg,
        bandwidth_variation=bandwidth_variation_cfg,
        packet_loss=packet_loss_cfg,
        congestion=congestion_cfg,
    )

    return RealisticSimulationConfig(
        topology=topology_cfg,
        simulation=simulation_cfg,
        realistic_traffic=realistic_traffic_cfg,
        variability=variability_cfg,
    )


@hydra.main(
    version_base="1.2",
    config_path="../configs",
    config_name="realistic_simulation",
)
def main(cfg: DictConfig) -> None:
    """Main entry point with Hydra configuration."""
    logger.info("Configuration:")
    logger.info(OmegaConf.to_yaml(cfg))

    logger.info("Cleaning previous mininet state")
    cleanup_network()

    config = _build_config_from_dict(cfg)
    run_realistic_simulation(config)


if __name__ == "__main__":
    main()
