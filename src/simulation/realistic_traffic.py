"""
Realistic traffic generation for network simulation.

This module provides human-like traffic generation that simulates realistic
user behavior patterns for time series forecasting data collection.
Traffic includes web browsing, video streaming, VoIP, gaming, and more.
"""

import os
import random
import subprocess
import threading
import time
import typing as tp
from dataclasses import dataclass

from loguru import logger
from mininet.net import Mininet
from mininet.node import Node

from src.simulation.traffic_profiles import (
    TRAFFIC_PROFILES,
    USER_PROFILES,
    ApplicationType,
    TrafficProfile,
    UserBehaviorProfile,
)


@dataclass
class ActiveSession:
    """Represents an active traffic session on a host."""

    host_name: str
    app_type: ApplicationType
    target_ip: str
    start_time: float
    expected_duration: float
    process: tp.Optional[subprocess.Popen] = None
    server_process: tp.Optional[subprocess.Popen] = None


@dataclass
class HostState:
    """Tracks the current state of a simulated host."""

    host: Node
    behavior_profile: UserBehaviorProfile
    current_session: tp.Optional[ActiveSession] = None
    total_sessions: int = 0
    last_activity_time: float = 0.0


class RealisticTrafficGenerator:
    """
    Generates realistic human-like network traffic patterns.

    Each host is assigned a user behavior profile that determines
    what types of applications they use and how they switch between them.
    """

    def __init__(
        self,
        net: Mininet,
        output_dir: str,
        seed: tp.Optional[int] = None,
        server_host: str = "h0",
    ):
        self.net = net
        self.output_dir = output_dir
        self.server_host_name = server_host
        self.rng = random.Random(seed)

        self._stop_event = threading.Event()
        self._threads: tp.List[threading.Thread] = []
        self._host_states: tp.Dict[str, HostState] = {}
        self._server_processes: tp.List[subprocess.Popen] = []
        self._lock = threading.Lock()

        base_dir = os.path.dirname(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        )
        self.agent_script = os.path.join(base_dir, "scripts", "realistic_agents.py")

        try:
            self.server = self.net.get(server_host)
            self.server_ip = self.server.IP()
        except Exception:
            self.server = self.net.hosts[0]
            self.server_ip = self.server.IP()
            logger.warning(f"Using {self.server.name} as server")

        self._initialize_host_states()

    def _initialize_host_states(self) -> None:
        """Initialize state tracking for all hosts."""
        profile_names = list(USER_PROFILES.keys())

        for host in self.net.hosts:
            if host.name == self.server.name:
                continue

            profile_name = self.rng.choice(profile_names)
            profile = USER_PROFILES[profile_name]

            self._host_states[host.name] = HostState(
                host=host,
                behavior_profile=profile,
                last_activity_time=time.time(),
            )

            logger.debug(f"Host {host.name} assigned profile: {profile_name}")

    def start(self, duration: int) -> None:
        """Start realistic traffic generation."""
        logger.info(f"Starting realistic traffic generation for {duration}s")

        self._start_infrastructure_servers()
        time.sleep(2)

        for host_name, state in self._host_states.items():
            thread = threading.Thread(
                target=self._host_traffic_loop,
                args=(host_name, duration),
                daemon=True,
            )
            thread.start()
            self._threads.append(thread)

        logger.info(f"Started {len(self._threads)} host traffic threads")

    def stop(self) -> None:
        """Stop all traffic generation."""
        logger.info("Stopping realistic traffic generation")
        self._stop_event.set()

        for thread in self._threads:
            thread.join(timeout=5)

        self._cleanup_all_sessions()
        self._stop_infrastructure_servers()

        logger.info("Realistic traffic generation stopped")

    def _start_infrastructure_servers(self) -> None:
        """Start all required server processes on the server host."""
        logger.info(f"Starting infrastructure servers on {self.server.name}")

        # HTTP server for web browsing and downloads
        self.server.cmd("mkdir -p /tmp/www")
        self.server.cmd(
            "dd if=/dev/urandom of=/tmp/www/file_1mb bs=1M count=1 2>/dev/null"
        )
        self.server.cmd(
            "dd if=/dev/urandom of=/tmp/www/file_10mb bs=1M count=10 2>/dev/null"
        )
        self.server.cmd(
            "dd if=/dev/urandom of=/tmp/www/file_100mb bs=1M count=100 2>/dev/null"
        )

        # Create sample HTML pages
        self.server.cmd("""cat > /tmp/www/index.html << 'EOF'
<!DOCTYPE html>
<html><head><title>Test Page</title></head>
<body><h1>Network Simulation Test Server</h1>
<p>This is a test page for realistic traffic simulation.</p>
<img src="image.jpg" alt="test">
<script src="script.js"></script>
</body></html>
EOF""")
        self.server.cmd(
            "dd if=/dev/urandom of=/tmp/www/image.jpg bs=100K count=1 2>/dev/null"
        )
        self.server.cmd(
            "dd if=/dev/urandom of=/tmp/www/script.js bs=50K count=1 2>/dev/null"
        )

        http_proc = self.server.popen(
            "python3 -m http.server 80 --directory /tmp/www",
            shell=True,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        self._server_processes.append(http_proc)

        # HTTPS-like server on port 443
        https_proc = self.server.popen(
            "python3 -m http.server 443 --directory /tmp/www",
            shell=True,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        self._server_processes.append(https_proc)

        # iperf3 TCP server for large transfers
        iperf_tcp = self.server.popen(
            "iperf3 -s -p 5201",
            shell=True,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        self._server_processes.append(iperf_tcp)

        # iperf3 UDP server for streaming/VoIP
        iperf_udp = self.server.popen(
            "iperf3 -s -p 5202",
            shell=True,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        self._server_processes.append(iperf_udp)

        # Multiple UDP receivers for different application types
        for port in [5060, 5061, 5062, 27015, 3478]:
            receiver = self.server.popen(
                f"python3 {self.agent_script} --mode receiver --port {port}",
                shell=True,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
            self._server_processes.append(receiver)

        logger.info("Infrastructure servers started")

    def _stop_infrastructure_servers(self) -> None:
        """Stop all infrastructure server processes."""
        for proc in self._server_processes:
            try:
                proc.terminate()
                proc.wait(timeout=2)
            except Exception:
                try:
                    proc.kill()
                except Exception:
                    pass

        self.server.cmd("pkill -f 'http.server'")
        self.server.cmd("pkill -f 'iperf3'")
        self.server.cmd("pkill -f 'realistic_agents.py'")
        self.server.cmd("rm -rf /tmp/www")

    def _host_traffic_loop(self, host_name: str, duration: int) -> None:
        """Main traffic generation loop for a single host."""
        state = self._host_states[host_name]
        start_time = time.time()

        initial_delay = self.rng.uniform(0, 10)
        time.sleep(initial_delay)

        while not self._stop_event.is_set():
            elapsed = time.time() - start_time
            if elapsed >= duration:
                break

            remaining = duration - elapsed
            if remaining < 5:
                break

            app_type = self._select_application(state)
            profile = TRAFFIC_PROFILES.get(app_type)

            if profile is None or app_type == ApplicationType.IDLE:
                idle_time = (
                    self.rng.uniform(5, 30) * state.behavior_profile.activity_level
                )
                idle_time = min(idle_time, remaining)
                self._interruptible_sleep(idle_time)
                continue

            session_duration = self._calculate_session_duration(
                profile, state, remaining
            )

            logger.debug(
                f"{host_name}: Starting {app_type.value} session for {session_duration:.1f}s"
            )

            self._run_application_session(state, app_type, profile, session_duration)

            state.total_sessions += 1
            state.last_activity_time = time.time()

            if profile.timing.think_time_max_sec > 0:
                think_time = self.rng.uniform(
                    profile.timing.think_time_min_sec,
                    profile.timing.think_time_max_sec,
                )
                think_time *= state.behavior_profile.activity_level
                self._interruptible_sleep(think_time)

        logger.debug(
            f"{host_name}: Traffic loop completed, {state.total_sessions} sessions"
        )

    def _select_application(self, state: HostState) -> ApplicationType:
        """Select the next application based on user behavior profile."""
        profile = state.behavior_profile

        if self.rng.random() > profile.activity_level:
            return ApplicationType.IDLE

        if not profile.app_weights:
            return ApplicationType.WEB_BROWSING

        apps = list(profile.app_weights.keys())
        weights = list(profile.app_weights.values())

        return self.rng.choices(apps, weights=weights, k=1)[0]

    def _calculate_session_duration(
        self,
        profile: TrafficProfile,
        state: HostState,
        max_duration: float,
    ) -> float:
        """Calculate session duration with human variability."""
        timing = profile.timing

        base_duration = self.rng.gauss(
            timing.avg_duration_sec, timing.avg_duration_sec * 0.3
        )
        base_duration = max(
            timing.min_duration_sec, min(timing.max_duration_sec, base_duration)
        )

        variability = 1.0 + self.rng.uniform(
            -profile.human_variability,
            profile.human_variability,
        )
        adjusted_duration = (
            base_duration * variability * state.behavior_profile.session_multiplier
        )

        return min(adjusted_duration, max_duration)

    def _run_application_session(
        self,
        state: HostState,
        app_type: ApplicationType,
        profile: TrafficProfile,
        duration: float,
    ) -> None:
        """Run a single application session."""
        host = state.host

        session = ActiveSession(
            host_name=host.name,
            app_type=app_type,
            target_ip=self.server_ip,
            start_time=time.time(),
            expected_duration=duration,
        )
        state.current_session = session

        try:
            if app_type == ApplicationType.WEB_BROWSING:
                self._simulate_web_browsing(host, session, duration)
            elif app_type == ApplicationType.VIDEO_STREAMING:
                self._simulate_video_streaming(host, session, duration)
            elif app_type == ApplicationType.VOIP_CALL:
                self._simulate_voip_call(host, session, duration)
            elif app_type == ApplicationType.ONLINE_GAMING:
                self._simulate_gaming(host, session, duration)
            elif app_type == ApplicationType.FILE_DOWNLOAD:
                self._simulate_file_download(host, session, duration)
            elif app_type == ApplicationType.VIDEO_CONFERENCE:
                self._simulate_video_conference(host, session, duration)
            elif app_type == ApplicationType.MUSIC_STREAMING:
                self._simulate_music_streaming(host, session, duration)
            elif app_type == ApplicationType.SOCIAL_MEDIA:
                self._simulate_social_media(host, session, duration)
            elif app_type == ApplicationType.EMAIL:
                self._simulate_email(host, session, duration)
            elif app_type == ApplicationType.CLOUD_SYNC:
                self._simulate_cloud_sync(host, session, duration)
            elif app_type == ApplicationType.IOT_TELEMETRY:
                self._simulate_iot_telemetry(host, session, duration)
        except Exception as e:
            logger.warning(f"{host.name}: Session error for {app_type.value}: {e}")
        finally:
            self._cleanup_session(session)
            state.current_session = None

    def _simulate_web_browsing(
        self,
        host: Node,
        session: ActiveSession,
        duration: float,
    ) -> None:
        """Simulate realistic web browsing with page loads and think times."""
        end_time = time.time() + duration
        pages = ["index.html", "image.jpg", "script.js", "file_1mb"]

        while time.time() < end_time and not self._stop_event.is_set():
            page = self.rng.choice(pages)

            session.process = host.popen(
                f"curl -s -o /dev/null --max-time 30 http://{self.server_ip}/{page}",
                shell=True,
            )

            try:
                session.process.wait(timeout=35)
            except subprocess.TimeoutExpired:
                session.process.kill()

            if time.time() >= end_time:
                break

            think_time = self.rng.uniform(1, 15)
            self._interruptible_sleep(min(think_time, end_time - time.time()))

    def _simulate_video_streaming(
        self,
        host: Node,
        session: ActiveSession,
        duration: float,
    ) -> None:
        """Simulate video streaming with adaptive bitrate behavior."""
        bitrates = ["500k", "1M", "2M", "4M", "8M"]
        initial_bitrate = self.rng.choice(bitrates[:3])

        session.process = host.popen(
            f"iperf3 -c {self.server_ip} -p 5201 -t {int(duration)} -b {initial_bitrate} "
            f"--logfile /tmp/stream_{host.name}.log",
            shell=True,
        )

        start = time.time()
        while time.time() - start < duration and not self._stop_event.is_set():
            if self.rng.random() < 0.1:
                new_bitrate = self.rng.choice(bitrates)
                logger.debug(f"{host.name}: Bitrate change to {new_bitrate}")

            self._interruptible_sleep(5)

            if session.process.poll() is not None:
                break

        self._terminate_process(session.process)

    def _simulate_voip_call(
        self,
        host: Node,
        session: ActiveSession,
        duration: float,
    ) -> None:
        """Simulate VoIP call with G.711 codec characteristics."""
        session.process = host.popen(
            f"python3 {self.agent_script} --mode voip "
            f"--target {self.server_ip} --port 5060 "
            f"--bandwidth 96000 --duration {int(duration)}",
            shell=True,
        )

        start = time.time()
        while time.time() - start < duration and not self._stop_event.is_set():
            self._interruptible_sleep(1)
            if session.process.poll() is not None:
                break

        self._terminate_process(session.process)

    def _simulate_gaming(
        self,
        host: Node,
        session: ActiveSession,
        duration: float,
    ) -> None:
        """Simulate online gaming traffic with small frequent packets."""
        session.process = host.popen(
            f"python3 {self.agent_script} --mode gaming "
            f"--target {self.server_ip} --port 27015 "
            f"--duration {int(duration)}",
            shell=True,
        )

        start = time.time()
        while time.time() - start < duration and not self._stop_event.is_set():
            self._interruptible_sleep(1)
            if session.process.poll() is not None:
                break

        self._terminate_process(session.process)

    def _simulate_file_download(
        self,
        host: Node,
        session: ActiveSession,
        duration: float,
    ) -> None:
        """Simulate file download with varying file sizes."""
        files = ["file_1mb", "file_10mb", "file_100mb"]
        weights = [0.5, 0.35, 0.15]
        selected_file = self.rng.choices(files, weights=weights, k=1)[0]

        session.process = host.popen(
            f"wget -q -O /dev/null --timeout=300 "
            f"http://{self.server_ip}/{selected_file}",
            shell=True,
        )

        start = time.time()
        while time.time() - start < duration and not self._stop_event.is_set():
            self._interruptible_sleep(1)
            if session.process.poll() is not None:
                break

        self._terminate_process(session.process)

    def _simulate_video_conference(
        self,
        host: Node,
        session: ActiveSession,
        duration: float,
    ) -> None:
        """Simulate video conference with bidirectional traffic."""
        session.process = host.popen(
            f"python3 {self.agent_script} --mode video_conference "
            f"--target {self.server_ip} --port 3478 "
            f"--bandwidth 1500000 --duration {int(duration)}",
            shell=True,
        )

        start = time.time()
        while time.time() - start < duration and not self._stop_event.is_set():
            self._interruptible_sleep(1)
            if session.process.poll() is not None:
                break

        self._terminate_process(session.process)

    def _simulate_music_streaming(
        self,
        host: Node,
        session: ActiveSession,
        duration: float,
    ) -> None:
        """Simulate music streaming at constant bitrate."""
        bitrate = self.rng.choice(["128k", "160k", "256k", "320k"])

        session.process = host.popen(
            f"iperf3 -c {self.server_ip} -p 5201 -t {int(duration)} -b {bitrate}",
            shell=True,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )

        start = time.time()
        while time.time() - start < duration and not self._stop_event.is_set():
            self._interruptible_sleep(5)
            if session.process.poll() is not None:
                break

        self._terminate_process(session.process)

    def _simulate_social_media(
        self,
        host: Node,
        session: ActiveSession,
        duration: float,
    ) -> None:
        """Simulate social media with bursty traffic pattern."""
        end_time = time.time() + duration

        while time.time() < end_time and not self._stop_event.is_set():
            if self.rng.random() < 0.3:
                burst_size = self.rng.choice(["100k", "500k", "1M", "2M"])
                session.process = host.popen(
                    f"iperf3 -c {self.server_ip} -p 5201 -n {burst_size}",
                    shell=True,
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                )
                session.process.wait(timeout=30)
            else:
                session.process = host.popen(
                    f"curl -s -o /dev/null http://{self.server_ip}/index.html",
                    shell=True,
                )
                session.process.wait(timeout=10)

            think_time = self.rng.uniform(2, 30)
            self._interruptible_sleep(min(think_time, max(0, end_time - time.time())))

    def _simulate_email(
        self,
        host: Node,
        session: ActiveSession,
        duration: float,
    ) -> None:
        """Simulate email checking with small bursty transfers."""
        end_time = time.time() + duration

        while time.time() < end_time and not self._stop_event.is_set():
            size = self.rng.choice(["10k", "50k", "200k", "1M"])
            session.process = host.popen(
                f"iperf3 -c {self.server_ip} -p 5201 -n {size}",
                shell=True,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )

            try:
                session.process.wait(timeout=30)
            except subprocess.TimeoutExpired:
                session.process.kill()

            think_time = self.rng.uniform(5, 60)
            self._interruptible_sleep(min(think_time, max(0, end_time - time.time())))

    def _simulate_cloud_sync(
        self,
        host: Node,
        session: ActiveSession,
        duration: float,
    ) -> None:
        """Simulate cloud sync with periodic uploads and downloads."""
        end_time = time.time() + duration

        while time.time() < end_time and not self._stop_event.is_set():
            is_upload = self.rng.random() < 0.4
            size = self.rng.choice(["100k", "500k", "2M", "10M"])

            if is_upload:
                session.process = host.popen(
                    f"iperf3 -c {self.server_ip} -p 5201 -n {size}",
                    shell=True,
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                )
            else:
                session.process = host.popen(
                    f"iperf3 -c {self.server_ip} -p 5201 -n {size} -R",
                    shell=True,
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                )

            try:
                session.process.wait(timeout=120)
            except subprocess.TimeoutExpired:
                session.process.kill()

            sync_interval = self.rng.uniform(10, 60)
            self._interruptible_sleep(
                min(sync_interval, max(0, end_time - time.time()))
            )

    def _simulate_iot_telemetry(
        self,
        host: Node,
        session: ActiveSession,
        duration: float,
    ) -> None:
        """Simulate IoT telemetry with small periodic packets."""
        session.process = host.popen(
            f"python3 {self.agent_script} --mode iot "
            f"--target {self.server_ip} --port 1883 "
            f"--duration {int(duration)}",
            shell=True,
        )

        start = time.time()
        while time.time() - start < duration and not self._stop_event.is_set():
            self._interruptible_sleep(1)
            if session.process.poll() is not None:
                break

        self._terminate_process(session.process)

    def _cleanup_session(self, session: ActiveSession) -> None:
        """Clean up resources for a completed session."""
        self._terminate_process(session.process)
        self._terminate_process(session.server_process)

    def _cleanup_all_sessions(self) -> None:
        """Clean up all active sessions."""
        for state in self._host_states.values():
            if state.current_session:
                self._cleanup_session(state.current_session)

        for host in self.net.hosts:
            host.cmd("pkill -f realistic_agents.py 2>/dev/null")
            host.cmd("pkill -f iperf3 2>/dev/null")
            host.cmd("pkill -f wget 2>/dev/null")
            host.cmd("pkill -f curl 2>/dev/null")

    def _terminate_process(self, proc: tp.Optional[subprocess.Popen]) -> None:
        """Safely terminate a subprocess."""
        if proc is None:
            return
        try:
            proc.terminate()
            proc.wait(timeout=2)
        except subprocess.TimeoutExpired:
            proc.kill()
        except Exception:
            pass

    def _interruptible_sleep(self, duration: float) -> None:
        """Sleep that can be interrupted by stop event."""
        end_time = time.time() + duration
        while time.time() < end_time:
            if self._stop_event.is_set():
                break
            time.sleep(min(0.5, end_time - time.time()))

    def get_statistics(self) -> tp.Dict[str, tp.Any]:
        """Get traffic generation statistics."""
        stats = {
            "total_sessions": sum(s.total_sessions for s in self._host_states.values()),
            "hosts": {},
        }

        for name, state in self._host_states.items():
            stats["hosts"][name] = {
                "profile": state.behavior_profile.name,
                "sessions": state.total_sessions,
            }

        return stats
