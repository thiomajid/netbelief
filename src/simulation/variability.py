"""
Network variability simulation for realistic traffic conditions.

This module provides configurable network perturbations to introduce
controlled randomness into simulations, creating realistic time series
data suitable for forecasting while avoiding excessive noise.

The key design principle is to create infrequent but meaningful events
that affect network metrics without making the data intractable for
time series analysis.
"""

import json
import os
import random
import threading
import time
import typing as tp
from dataclasses import asdict, dataclass, field
from enum import Enum

from loguru import logger
from mininet.net import Mininet


class EventType(Enum):
    """Types of network variability events."""

    LINK_FAILURE = "link_failure"
    LINK_RECOVERY = "link_recovery"
    DELAY_CHANGE = "delay_change"
    BANDWIDTH_CHANGE = "bandwidth_change"
    PACKET_LOSS_CHANGE = "packet_loss_change"
    CONGESTION_START = "congestion_start"
    CONGESTION_END = "congestion_end"
    EVENT_FAILED = "event_failed"  # Records failed event attempts


@dataclass
class NetworkEvent:
    """Record of a network variability event."""

    timestamp: float
    event_type: EventType
    source_node: str
    target_node: str
    old_value: tp.Optional[tp.Any] = None
    new_value: tp.Optional[tp.Any] = None
    duration: tp.Optional[float] = None
    success: bool = True  # Whether the event was applied successfully
    error_message: tp.Optional[str] = None  # Error details if failed


@dataclass
class LinkFailureConfig:
    """Configuration for link failure events."""

    # Whether link failures are enabled
    enabled: bool = False
    # Probability of a link failure check triggering an actual failure
    probability: float = 0.05
    # Minimum time between failure checks (seconds)
    min_interval: float = 60.0
    # Maximum time between failure checks (seconds)
    max_interval: float = 300.0
    # Minimum failure duration (seconds)
    min_duration: float = 10.0
    # Maximum failure duration (seconds)
    max_duration: float = 60.0
    # Maximum concurrent failed links (as fraction of total links)
    max_concurrent_failures: float = 0.1


@dataclass
class DelayVariationConfig:
    """Configuration for delay variation events."""

    # Whether delay variations are enabled
    enabled: bool = False
    # Probability of a delay change during each check
    probability: float = 0.1
    # Minimum time between checks (seconds)
    min_interval: float = 30.0
    # Maximum time between checks (seconds)
    max_interval: float = 120.0
    # Minimum delay multiplier (1.0 = no change)
    min_multiplier: float = 0.8
    # Maximum delay multiplier
    max_multiplier: float = 2.0
    # Duration before delay returns to normal (seconds)
    # Set to None for permanent changes until next event
    revert_after: tp.Optional[float] = None


@dataclass
class BandwidthVariationConfig:
    """Configuration for bandwidth variation events."""

    # Whether bandwidth variations are enabled
    enabled: bool = False
    # Probability of a bandwidth change during each check
    probability: float = 0.1
    # Minimum time between checks (seconds)
    min_interval: float = 30.0
    # Maximum time between checks (seconds)
    max_interval: float = 120.0
    # Minimum bandwidth as fraction of original (0.3 = 30%)
    min_fraction: float = 0.3
    # Maximum bandwidth as fraction of original
    max_fraction: float = 1.0
    # Duration before bandwidth returns to normal (seconds)
    revert_after: tp.Optional[float] = None


@dataclass
class PacketLossConfig:
    """Configuration for packet loss variation events."""

    # Whether packet loss variations are enabled
    enabled: bool = False
    # Probability of a loss change during each check
    probability: float = 0.08
    # Minimum time between checks (seconds)
    min_interval: float = 45.0
    # Maximum time between checks (seconds)
    max_interval: float = 180.0
    # Minimum packet loss percentage
    min_loss: float = 0.0
    # Maximum packet loss percentage
    max_loss: float = 5.0
    # Duration before loss returns to normal (seconds)
    revert_after: tp.Optional[float] = None


@dataclass
class CongestionConfig:
    """Configuration for network congestion simulation."""

    # Whether congestion events are enabled
    enabled: bool = False
    # Probability of congestion starting during each check
    probability: float = 0.05
    # Minimum time between congestion checks (seconds)
    min_interval: float = 120.0
    # Maximum time between congestion checks (seconds)
    max_interval: float = 600.0
    # Minimum congestion duration (seconds)
    min_duration: float = 30.0
    # Maximum congestion duration (seconds)
    max_duration: float = 180.0
    # Bandwidth reduction during congestion (fraction)
    bandwidth_reduction: float = 0.4
    # Additional delay during congestion (ms)
    additional_delay_ms: float = 50.0
    # Packet loss during congestion (%)
    loss_percent: float = 2.0


class LinkScope(Enum):
    """Scope of links to target for variability events."""

    ALL = "all"  # All links (backbone + access)
    BACKBONE = "backbone"  # Switch-to-switch links only
    ACCESS = "access"  # Switch-to-host links only


@dataclass
class VariabilityConfig:
    """Master configuration for all network variability features."""

    # Global enable/disable for all variability
    enabled: bool = False
    # Random seed for reproducibility (None for random)
    seed: tp.Optional[int] = None
    # Scope of links to affect: "all", "backbone", or "access"
    link_scope: str = "all"
    # Links to exclude from variability (list of (node1, node2) tuples)
    excluded_links: tp.List[tp.Tuple[str, str]] = field(default_factory=list)
    # Specific links to target (empty = use link_scope)
    target_links: tp.List[tp.Tuple[str, str]] = field(default_factory=list)

    # Default values for links without QoS parameters
    # Used when topology doesn't define bandwidth/delay for some links
    default_bandwidth: tp.Optional[float] = 10.0  # Mbps
    default_delay: tp.Optional[str] = "1ms"  # Latency

    # Sub-configurations
    link_failure: LinkFailureConfig = field(default_factory=LinkFailureConfig)
    delay_variation: DelayVariationConfig = field(default_factory=DelayVariationConfig)
    bandwidth_variation: BandwidthVariationConfig = field(
        default_factory=BandwidthVariationConfig
    )
    packet_loss: PacketLossConfig = field(default_factory=PacketLossConfig)
    congestion: CongestionConfig = field(default_factory=CongestionConfig)


class LinkState:
    """Tracks the current state of a network link."""

    def __init__(
        self,
        node1: str,
        node2: str,
        original_bw: tp.Optional[float] = None,
        original_delay: tp.Optional[str] = None,
        is_backbone: bool = True,
    ):
        self.node1 = node1
        self.node2 = node2
        self.original_bw = original_bw
        self.original_delay = original_delay
        self.is_backbone = is_backbone  # True for switch-switch, False for switch-host
        self.is_up = True
        self.current_bw = original_bw
        self.current_delay = original_delay
        self.current_loss = 0.0
        self.is_congested = False
        self.failure_time: tp.Optional[float] = None
        self.scheduled_recovery: tp.Optional[float] = None

    def key(self) -> tp.Tuple[str, str]:
        """Return a canonical key for this link."""
        return (min(self.node1, self.node2), max(self.node1, self.node2))


class NetworkVariabilityManager:
    """
    Manages network variability events during simulation.

    This class coordinates all network perturbations, ensuring they
    occur at appropriate intervals and don't overwhelm the network
    or make the collected data too noisy for analysis.
    """

    def __init__(
        self,
        net: Mininet,
        config: VariabilityConfig,
        output_dir: str,
    ):
        """
        Initialize the variability manager.

        Args:
            net: Mininet network instance
            config: Variability configuration
            output_dir: Directory for event logs
        """
        self.net = net
        self.config = config
        self.output_dir = output_dir

        self._stop_event = threading.Event()
        self._threads: tp.List[threading.Thread] = []
        self._events: tp.List[NetworkEvent] = []
        self._events_lock = threading.Lock()
        self._link_states: tp.Dict[tp.Tuple[str, str], LinkState] = {}
        self._link_states_lock = threading.Lock()

        # Initialize random generator
        if config.seed is not None:
            random.seed(config.seed)

        # Initialize link states
        self._initialize_link_states()

    def _is_switch(self, node_name: str) -> bool:
        """
        Determine if a node is a switch.

        Uses multiple detection methods:
        1. Name starts with 's' followed by digits (e.g., s0, s1, s10)
        2. Node type check via Mininet if available

        Args:
            node_name: Name of the node

        Returns:
            True if the node is a switch
        """
        import re

        # Pattern: 's' followed by one or more digits
        if re.match(r"^s\d+$", node_name):
            return True

        # Try to get the actual node and check its type
        try:
            node = self.net.get(node_name)
            if node is not None:
                # Check if it's an OVS switch or similar
                from mininet.node import OVSSwitch, Switch

                if isinstance(node, (OVSSwitch, Switch)):
                    return True
        except Exception:
            pass

        return False

    def _is_host(self, node_name: str) -> bool:
        """
        Determine if a node is a host.

        Uses multiple detection methods:
        1. Name starts with 'h' followed by digits (e.g., h0, h1, h10)
        2. Node type check via Mininet if available

        Args:
            node_name: Name of the node

        Returns:
            True if the node is a host
        """
        import re

        # Pattern: 'h' followed by one or more digits
        if re.match(r"^h\d+$", node_name):
            return True

        # Try to get the actual node and check its type
        try:
            node = self.net.get(node_name)
            if node is not None:
                from mininet.node import Host

                if isinstance(node, Host):
                    return True
        except Exception:
            pass

        return False

    def _initialize_link_states(self) -> None:
        """Initialize tracking for all network links based on configured scope."""
        scope = self.config.link_scope.lower()
        backbone_count = 0
        access_count = 0
        skipped_count = 0

        logger.debug(f"Initializing link states with scope: {scope}")
        logger.debug(f"Total links in network: {len(self.net.links)}")
        logger.debug(
            f"Default values: bw={self.config.default_bandwidth}Mbps, "
            f"delay={self.config.default_delay}"
        )

        for link in self.net.links:
            intf1, intf2 = link.intf1, link.intf2
            node1 = intf1.node.name
            node2 = intf2.node.name

            logger.debug(f"Processing link: {node1} <-> {node2}")

            # Determine node types using robust detection
            is_switch1 = self._is_switch(node1)
            is_switch2 = self._is_switch(node2)
            is_host1 = self._is_host(node1)
            is_host2 = self._is_host(node2)

            # Log unknown node types for debugging
            if not (is_switch1 or is_host1):
                logger.warning(
                    f"Unknown node type for '{node1}' - "
                    f"not detected as switch or host, skipping this link"
                )
            if not (is_switch2 or is_host2):
                logger.warning(
                    f"Unknown node type for '{node2}' - "
                    f"not detected as switch or host, skipping this link"
                )

            is_backbone = is_switch1 and is_switch2
            is_access = (is_switch1 and is_host2) or (is_host1 and is_switch2)

            # Filter based on configured scope
            if scope == "backbone" and not is_backbone:
                logger.debug(f"Skipping non-backbone link: {node1} <-> {node2}")
                skipped_count += 1
                continue
            elif scope == "access" and not is_access:
                logger.debug(f"Skipping non-access link: {node1} <-> {node2}")
                skipped_count += 1
                continue
            elif scope == "all" and not (is_backbone or is_access):
                logger.debug(
                    f"Skipping unknown link type: {node1} <-> {node2} "
                    f"(switch1={is_switch1}, switch2={is_switch2}, "
                    f"host1={is_host1}, host2={is_host2})"
                )
                skipped_count += 1
                continue

            # Extract original parameters from interface params dict
            # TCIntf stores config params in self.params, not as direct attributes
            intf1_params = getattr(intf1, "params", {})
            intf2_params = getattr(intf2, "params", {})

            # Try to get bw/delay from either interface's params
            original_bw = intf1_params.get("bw") or intf2_params.get("bw")
            original_delay = intf1_params.get("delay") or intf2_params.get("delay")

            # Apply defaults for links without QoS parameters
            used_default_bw = False
            used_default_delay = False

            if original_bw is None and self.config.default_bandwidth is not None:
                original_bw = self.config.default_bandwidth
                used_default_bw = True

            if original_delay is None and self.config.default_delay is not None:
                original_delay = self.config.default_delay
                used_default_delay = True

            if used_default_bw or used_default_delay:
                logger.debug(
                    f"Link {node1} <-> {node2}: applied defaults "
                    f"(bw={'default' if used_default_bw else 'original'}, "
                    f"delay={'default' if used_default_delay else 'original'})"
                )
            else:
                logger.debug(
                    f"Link {node1} <-> {node2}: using topology values "
                    f"(bw={original_bw}, delay={original_delay})"
                )

            state = LinkState(
                node1, node2, original_bw, original_delay, is_backbone=is_backbone
            )
            self._link_states[state.key()] = state

            if is_backbone:
                backbone_count += 1
                logger.debug(
                    f"Added backbone link: {node1} <-> {node2} "
                    f"(bw={original_bw}, delay={original_delay})"
                )
            else:
                access_count += 1
                logger.debug(
                    f"Added access link: {node1} <-> {node2} "
                    f"(bw={original_bw}, delay={original_delay})"
                )

        logger.info(
            f"Initialized state tracking for {len(self._link_states)} links "
            f"(scope={scope}, backbone={backbone_count}, access={access_count}, "
            f"skipped={skipped_count})"
        )

    def _get_eligible_links(self) -> tp.List[LinkState]:
        """Get links eligible for variability events."""
        eligible = []
        excluded = set((min(a, b), max(a, b)) for a, b in self.config.excluded_links)
        target = set((min(a, b), max(a, b)) for a, b in self.config.target_links)

        for key, state in self._link_states.items():
            if key in excluded:
                continue
            if target and key not in target:
                continue
            eligible.append(state)

        return eligible

    def _record_event(self, event: NetworkEvent) -> None:
        """Record an event for later export."""
        with self._events_lock:
            self._events.append(event)
        logger.info(
            f"Event: {event.event_type.value} on {event.source_node}<->{event.target_node}"
        )

    def _set_link_status(
        self, node1: str, node2: str, status: str
    ) -> tp.Tuple[bool, tp.Optional[str]]:
        """
        Set link status (up/down).

        Args:
            node1: First node name
            node2: Second node name
            status: 'up' or 'down'

        Returns:
            Tuple of (success, error_message)
        """
        try:
            self.net.configLinkStatus(node1, node2, status)
            return True, None
        except Exception as e:
            error_msg = f"Failed to set link status {node1}<->{node2} to {status}: {e}"
            logger.error(error_msg)
            return False, str(e)

    def _configure_link(
        self,
        node1: str,
        node2: str,
        bw: tp.Optional[float] = None,
        delay: tp.Optional[str] = None,
        loss: tp.Optional[float] = None,
    ) -> tp.Tuple[bool, tp.Optional[str]]:
        """
        Configure link parameters using tc.

        Args:
            node1: First node name
            node2: Second node name
            bw: Bandwidth in Mbps
            delay: Delay string (e.g., "10ms")
            loss: Packet loss percentage

        Returns:
            Tuple of (success, error_message)
        """
        try:
            n1 = self.net.get(node1)
            n2 = self.net.get(node2)

            if n1 is None or n2 is None:
                return False, f"Node not found: {node1 if n1 is None else node2}"

            links = self.net.linksBetween(n1, n2)
            if not links:
                return False, f"No link found between {node1} and {node2}"

            for link in links:
                for intf in [link.intf1, link.intf2]:
                    if hasattr(intf, "config"):
                        params = {}
                        if bw is not None:
                            params["bw"] = bw
                        if delay is not None:
                            params["delay"] = delay
                        if loss is not None:
                            params["loss"] = loss
                        if params:
                            intf.config(**params)

            return True, None
        except Exception as e:
            error_msg = f"Failed to configure link {node1}<->{node2}: {e}"
            logger.error(error_msg)
            return False, str(e)

    def _link_failure_loop(self) -> None:
        """Background thread for link failure events."""
        cfg = self.config.link_failure
        if not cfg.enabled:
            return

        logger.info("Link failure simulation started")

        while not self._stop_event.is_set():
            # Wait for next check interval
            interval = random.uniform(cfg.min_interval, cfg.max_interval)
            if self._stop_event.wait(interval):
                break

            with self._link_states_lock:
                eligible = [s for s in self._get_eligible_links() if s.is_up]

                # Check max concurrent failures
                failed_count = sum(1 for s in self._link_states.values() if not s.is_up)
                max_failures = int(len(self._link_states) * cfg.max_concurrent_failures)

                if failed_count >= max_failures or not eligible:
                    continue

                # Probabilistic failure
                if random.random() > cfg.probability:
                    continue

                # Select random link to fail
                state = random.choice(eligible)
                duration = random.uniform(cfg.min_duration, cfg.max_duration)
                event_time = time.time()

                success, error_msg = self._set_link_status(
                    state.node1, state.node2, "down"
                )

                if success:
                    state.is_up = False
                    state.failure_time = event_time
                    state.scheduled_recovery = event_time + duration

                    self._record_event(
                        NetworkEvent(
                            timestamp=event_time,
                            event_type=EventType.LINK_FAILURE,
                            source_node=state.node1,
                            target_node=state.node2,
                            duration=duration,
                            success=True,
                        )
                    )

                    # Schedule recovery
                    threading.Timer(
                        duration,
                        self._recover_link,
                        args=(state.node1, state.node2),
                    ).start()
                else:
                    # Record failed event for analysis
                    self._record_event(
                        NetworkEvent(
                            timestamp=event_time,
                            event_type=EventType.LINK_FAILURE,
                            source_node=state.node1,
                            target_node=state.node2,
                            duration=duration,
                            success=False,
                            error_message=error_msg,
                        )
                    )
                    logger.warning(
                        f"Link failure event failed on {state.node1}<->{state.node2}: {error_msg}"
                    )

    def _recover_link(self, node1: str, node2: str) -> None:
        """Recover a failed link."""
        # Check if simulation is still running
        if self._stop_event.is_set():
            logger.debug(f"Skipping recovery for {node1}<->{node2}: simulation stopped")
            return

        key = (min(node1, node2), max(node1, node2))

        with self._link_states_lock:
            state = self._link_states.get(key)
            if state is None:
                logger.warning(
                    f"Cannot recover link {node1}<->{node2}: state not found"
                )
                return
            if state.is_up:
                logger.debug(f"Link {node1}<->{node2} already up, skipping recovery")
                return

            recovery_time = time.time()
            success, error_msg = self._set_link_status(node1, node2, "up")

            if success:
                state.is_up = True

                self._record_event(
                    NetworkEvent(
                        timestamp=recovery_time,
                        event_type=EventType.LINK_RECOVERY,
                        source_node=node1,
                        target_node=node2,
                        duration=recovery_time - (state.failure_time or 0),
                        success=True,
                    )
                )

                state.failure_time = None
                state.scheduled_recovery = None
            else:
                # Record failed recovery event
                self._record_event(
                    NetworkEvent(
                        timestamp=recovery_time,
                        event_type=EventType.LINK_RECOVERY,
                        source_node=node1,
                        target_node=node2,
                        success=False,
                        error_message=error_msg,
                    )
                )
                logger.warning(
                    f"Link recovery failed on {node1}<->{node2}: {error_msg}"
                )

    def _delay_variation_loop(self) -> None:
        """Background thread for delay variation events."""
        cfg = self.config.delay_variation
        if not cfg.enabled:
            return

        logger.info("Delay variation simulation started")

        while not self._stop_event.is_set():
            interval = random.uniform(cfg.min_interval, cfg.max_interval)
            if self._stop_event.wait(interval):
                break

            if random.random() > cfg.probability:
                continue

            with self._link_states_lock:
                eligible = [s for s in self._get_eligible_links() if s.is_up]
                if not eligible:
                    continue

                state = random.choice(eligible)

                # Parse original delay
                original_delay_ms = self._parse_delay_ms(state.original_delay)
                if original_delay_ms is None:
                    logger.debug(
                        f"Skipping delay variation on {state.node1}<->{state.node2}: "
                        f"cannot parse original delay '{state.original_delay}'"
                    )
                    continue

                multiplier = random.uniform(cfg.min_multiplier, cfg.max_multiplier)
                new_delay_ms = original_delay_ms * multiplier
                new_delay = f"{new_delay_ms:.2f}ms"
                old_delay = state.current_delay
                event_time = time.time()

                success, error_msg = self._configure_link(
                    state.node1, state.node2, delay=new_delay
                )

                if success:
                    state.current_delay = new_delay

                    self._record_event(
                        NetworkEvent(
                            timestamp=event_time,
                            event_type=EventType.DELAY_CHANGE,
                            source_node=state.node1,
                            target_node=state.node2,
                            old_value=old_delay,
                            new_value=new_delay,
                            success=True,
                        )
                    )

                    # Schedule revert if configured
                    if cfg.revert_after is not None:
                        threading.Timer(
                            cfg.revert_after,
                            self._revert_delay,
                            args=(state.node1, state.node2, state.original_delay),
                        ).start()
                else:
                    self._record_event(
                        NetworkEvent(
                            timestamp=event_time,
                            event_type=EventType.DELAY_CHANGE,
                            source_node=state.node1,
                            target_node=state.node2,
                            old_value=old_delay,
                            new_value=new_delay,
                            success=False,
                            error_message=error_msg,
                        )
                    )
                    logger.warning(
                        f"Delay variation failed on {state.node1}<->{state.node2}: {error_msg}"
                    )

    def _revert_delay(
        self, node1: str, node2: str, original_delay: tp.Optional[str]
    ) -> None:
        """Revert delay to original value."""
        if self._stop_event.is_set():
            return
        if original_delay is None:
            return

        key = (min(node1, node2), max(node1, node2))
        with self._link_states_lock:
            state = self._link_states.get(key)
            if state is None:
                return

            success, error_msg = self._configure_link(
                node1, node2, delay=original_delay
            )
            if success:
                state.current_delay = original_delay
            else:
                logger.warning(
                    f"Failed to revert delay on {node1}<->{node2}: {error_msg}"
                )

    def _bandwidth_variation_loop(self) -> None:
        """Background thread for bandwidth variation events."""
        cfg = self.config.bandwidth_variation
        if not cfg.enabled:
            return

        logger.info("Bandwidth variation simulation started")

        while not self._stop_event.is_set():
            interval = random.uniform(cfg.min_interval, cfg.max_interval)
            if self._stop_event.wait(interval):
                break

            if random.random() > cfg.probability:
                continue

            with self._link_states_lock:
                eligible = [s for s in self._get_eligible_links() if s.is_up]
                if not eligible:
                    continue

                state = random.choice(eligible)

                if state.original_bw is None:
                    logger.debug(
                        f"Skipping bandwidth variation on {state.node1}<->{state.node2}: "
                        "no original bandwidth defined"
                    )
                    continue

                fraction = random.uniform(cfg.min_fraction, cfg.max_fraction)
                new_bw = state.original_bw * fraction
                old_bw = state.current_bw
                event_time = time.time()

                success, error_msg = self._configure_link(
                    state.node1, state.node2, bw=new_bw
                )

                if success:
                    state.current_bw = new_bw

                    self._record_event(
                        NetworkEvent(
                            timestamp=event_time,
                            event_type=EventType.BANDWIDTH_CHANGE,
                            source_node=state.node1,
                            target_node=state.node2,
                            old_value=old_bw,
                            new_value=new_bw,
                            success=True,
                        )
                    )

                    if cfg.revert_after is not None:
                        threading.Timer(
                            cfg.revert_after,
                            self._revert_bandwidth,
                            args=(state.node1, state.node2, state.original_bw),
                        ).start()
                else:
                    self._record_event(
                        NetworkEvent(
                            timestamp=event_time,
                            event_type=EventType.BANDWIDTH_CHANGE,
                            source_node=state.node1,
                            target_node=state.node2,
                            old_value=old_bw,
                            new_value=new_bw,
                            success=False,
                            error_message=error_msg,
                        )
                    )
                    logger.warning(
                        f"Bandwidth variation failed on {state.node1}<->{state.node2}: {error_msg}"
                    )

    def _revert_bandwidth(
        self, node1: str, node2: str, original_bw: tp.Optional[float]
    ) -> None:
        """Revert bandwidth to original value."""
        if self._stop_event.is_set():
            return
        if original_bw is None:
            return

        key = (min(node1, node2), max(node1, node2))
        with self._link_states_lock:
            state = self._link_states.get(key)
            if state is None:
                return

            success, error_msg = self._configure_link(node1, node2, bw=original_bw)
            if success:
                state.current_bw = original_bw
            else:
                logger.warning(
                    f"Failed to revert bandwidth on {node1}<->{node2}: {error_msg}"
                )

    def _packet_loss_loop(self) -> None:
        """Background thread for packet loss variation events."""
        cfg = self.config.packet_loss
        if not cfg.enabled:
            return

        logger.info("Packet loss variation simulation started")

        while not self._stop_event.is_set():
            interval = random.uniform(cfg.min_interval, cfg.max_interval)
            if self._stop_event.wait(interval):
                break

            if random.random() > cfg.probability:
                continue

            with self._link_states_lock:
                eligible = [s for s in self._get_eligible_links() if s.is_up]
                if not eligible:
                    continue

                state = random.choice(eligible)

                new_loss = random.uniform(cfg.min_loss, cfg.max_loss)
                old_loss = state.current_loss
                event_time = time.time()

                success, error_msg = self._configure_link(
                    state.node1, state.node2, loss=new_loss
                )

                if success:
                    state.current_loss = new_loss

                    self._record_event(
                        NetworkEvent(
                            timestamp=event_time,
                            event_type=EventType.PACKET_LOSS_CHANGE,
                            source_node=state.node1,
                            target_node=state.node2,
                            old_value=old_loss,
                            new_value=new_loss,
                            success=True,
                        )
                    )

                    if cfg.revert_after is not None:
                        threading.Timer(
                            cfg.revert_after,
                            self._revert_loss,
                            args=(state.node1, state.node2),
                        ).start()
                else:
                    self._record_event(
                        NetworkEvent(
                            timestamp=event_time,
                            event_type=EventType.PACKET_LOSS_CHANGE,
                            source_node=state.node1,
                            target_node=state.node2,
                            old_value=old_loss,
                            new_value=new_loss,
                            success=False,
                            error_message=error_msg,
                        )
                    )
                    logger.warning(
                        f"Packet loss variation failed on {state.node1}<->{state.node2}: {error_msg}"
                    )

    def _revert_loss(self, node1: str, node2: str) -> None:
        """Revert packet loss to zero."""
        if self._stop_event.is_set():
            return

        key = (min(node1, node2), max(node1, node2))
        with self._link_states_lock:
            state = self._link_states.get(key)
            if state is None:
                return

            success, error_msg = self._configure_link(node1, node2, loss=0)
            if success:
                state.current_loss = 0.0
            else:
                logger.warning(
                    f"Failed to revert packet loss on {node1}<->{node2}: {error_msg}"
                )

    def _congestion_loop(self) -> None:
        """Background thread for congestion simulation."""
        cfg = self.config.congestion
        if not cfg.enabled:
            return

        logger.info("Congestion simulation started")

        while not self._stop_event.is_set():
            interval = random.uniform(cfg.min_interval, cfg.max_interval)
            if self._stop_event.wait(interval):
                break

            if random.random() > cfg.probability:
                continue

            with self._link_states_lock:
                eligible = [
                    s
                    for s in self._get_eligible_links()
                    if s.is_up and not s.is_congested
                ]
                if not eligible:
                    continue

                state = random.choice(eligible)
                duration = random.uniform(cfg.min_duration, cfg.max_duration)
                event_time = time.time()

                # Apply congestion effects
                new_bw = None
                if state.original_bw is not None:
                    new_bw = state.original_bw * cfg.bandwidth_reduction

                original_delay_ms = self._parse_delay_ms(state.original_delay)
                new_delay = None
                if original_delay_ms is not None:
                    new_delay = f"{original_delay_ms + cfg.additional_delay_ms:.2f}ms"

                success, error_msg = self._configure_link(
                    state.node1,
                    state.node2,
                    bw=new_bw,
                    delay=new_delay,
                    loss=cfg.loss_percent,
                )

                if success:
                    state.is_congested = True

                    self._record_event(
                        NetworkEvent(
                            timestamp=event_time,
                            event_type=EventType.CONGESTION_START,
                            source_node=state.node1,
                            target_node=state.node2,
                            new_value={
                                "bw": new_bw,
                                "delay": new_delay,
                                "loss": cfg.loss_percent,
                            },
                            duration=duration,
                            success=True,
                        )
                    )

                    threading.Timer(
                        duration,
                        self._end_congestion,
                        args=(state.node1, state.node2),
                    ).start()
                else:
                    self._record_event(
                        NetworkEvent(
                            timestamp=event_time,
                            event_type=EventType.CONGESTION_START,
                            source_node=state.node1,
                            target_node=state.node2,
                            new_value={
                                "bw": new_bw,
                                "delay": new_delay,
                                "loss": cfg.loss_percent,
                            },
                            duration=duration,
                            success=False,
                            error_message=error_msg,
                        )
                    )
                    logger.warning(
                        f"Congestion event failed on {state.node1}<->{state.node2}: {error_msg}"
                    )

    def _end_congestion(self, node1: str, node2: str) -> None:
        """End congestion on a link."""
        if self._stop_event.is_set():
            return

        key = (min(node1, node2), max(node1, node2))

        with self._link_states_lock:
            state = self._link_states.get(key)
            if state is None:
                logger.warning(
                    f"Cannot end congestion on {node1}<->{node2}: state not found"
                )
                return
            if not state.is_congested:
                logger.debug(f"Link {node1}<->{node2} not congested, skipping")
                return

            event_time = time.time()

            # Restore original parameters
            success, error_msg = self._configure_link(
                node1,
                node2,
                bw=state.original_bw,
                delay=state.original_delay,
                loss=0,
            )

            if success:
                state.is_congested = False
                state.current_bw = state.original_bw
                state.current_delay = state.original_delay
                state.current_loss = 0.0

                self._record_event(
                    NetworkEvent(
                        timestamp=event_time,
                        event_type=EventType.CONGESTION_END,
                        source_node=node1,
                        target_node=node2,
                        success=True,
                    )
                )
            else:
                self._record_event(
                    NetworkEvent(
                        timestamp=event_time,
                        event_type=EventType.CONGESTION_END,
                        source_node=node1,
                        target_node=node2,
                        success=False,
                        error_message=error_msg,
                    )
                )
                logger.warning(
                    f"Failed to end congestion on {node1}<->{node2}: {error_msg}"
                )

    @staticmethod
    def _parse_delay_ms(delay_str: tp.Optional[str]) -> tp.Optional[float]:
        """Parse delay string to milliseconds."""
        if delay_str is None:
            return None

        delay_str = delay_str.strip().lower()
        if delay_str.endswith("ms"):
            try:
                return float(delay_str[:-2])
            except ValueError:
                return None
        elif delay_str.endswith("s"):
            try:
                return float(delay_str[:-1]) * 1000
            except ValueError:
                return None

        try:
            return float(delay_str)
        except ValueError:
            return None

    def start(self) -> None:
        """Start all variability simulation threads."""
        if not self.config.enabled:
            logger.info("Network variability is disabled")
            return

        logger.info("Starting network variability simulation")

        # Start individual feature threads
        thread_configs = [
            ("link_failure", self._link_failure_loop),
            ("delay_variation", self._delay_variation_loop),
            ("bandwidth_variation", self._bandwidth_variation_loop),
            ("packet_loss", self._packet_loss_loop),
            ("congestion", self._congestion_loop),
        ]

        for name, target in thread_configs:
            thread = threading.Thread(target=target, name=f"variability_{name}")
            thread.daemon = True
            thread.start()
            self._threads.append(thread)

        logger.info(f"Started {len(self._threads)} variability threads")

    def stop(self) -> None:
        """Stop all variability simulation and export events."""
        self._stop_event.set()

        for thread in self._threads:
            thread.join(timeout=5)

        self._export_events()
        logger.info("Network variability simulation stopped")

    def _export_events(self) -> None:
        """Export recorded events to file."""

        filepath = os.path.join(self.output_dir, "network_events.json")

        with self._events_lock:
            events_data = []
            for event in self._events:
                event_dict = asdict(event)
                # Convert EventType enum to string
                event_dict["event_type"] = event.event_type.value
                events_data.append(event_dict)

        # Calculate summary statistics
        total_events = len(events_data)
        successful_events = sum(1 for e in events_data if e.get("success", True))
        failed_events = total_events - successful_events

        with open(filepath, "w") as f:
            json.dump(
                {
                    "summary": {
                        "total_events": total_events,
                        "successful_events": successful_events,
                        "failed_events": failed_events,
                    },
                    "events": events_data,
                },
                f,
                indent=2,
            )

        logger.info(
            f"Exported {total_events} network events to {filepath} "
            f"({successful_events} successful, {failed_events} failed)"
        )

    def get_events(self) -> tp.List[NetworkEvent]:
        """Get a copy of all recorded events."""
        with self._events_lock:
            return list(self._events)

    def get_current_link_states(self) -> tp.Dict[tp.Tuple[str, str], dict]:
        """Get current state of all tracked links."""
        with self._link_states_lock:
            return {
                key: {
                    "node1": state.node1,
                    "node2": state.node2,
                    "is_up": state.is_up,
                    "current_bw": state.current_bw,
                    "current_delay": state.current_delay,
                    "current_loss": state.current_loss,
                    "is_congested": state.is_congested,
                }
                for key, state in self._link_states.items()
            }
