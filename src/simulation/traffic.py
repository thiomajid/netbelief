"""
Advanced traffic generation for realistic network simulation.
"""

import os
import typing as tp
from enum import Enum

from loguru import logger
from mininet.net import Mininet


class TrafficType(Enum):
    IDLE = "idle"
    VOIP = "voip"
    VIDEO_STREAMING = "video_streaming"
    DOWNLOAD = "download"
    WEB_BROWSING = "web_browsing"
    GAMING = "gaming"


class TrafficManager:
    """
    Manages dynamic traffic generation across Mininet hosts.

    Each host is assigned a single traffic type at a time and can switch
    between different types periodically.
    """

    def __init__(self, net: Mininet, config: tp.Any, output_dir: str):
        self.net = net
        self.config = config
        self.output_dir = output_dir
        self.host_tasks: tp.Dict[
            str, tp.Tuple[TrafficType, tp.Any]
        ] = {}  # host_name -> (type, process)

        # Find scripts directory relative to this file
        base_dir = os.path.dirname(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        )
        self.script_path = os.path.join(base_dir, "scripts", "traffic_generators.py")

        # Get server host
        try:
            self.server = self.net.get(self.config.server_host)
            self.server_ip = self.server.IP()
        except Exception as e:
            logger.error(f"Could not find server host {self.config.server_host}: {e}")
            # Fallback to first host if not found
            self.server = self.net.hosts[0]
            self.server_ip = self.server.IP()
            logger.warning(f"Falling back to {self.server.name} as server")

    def start(self, total_duration: int = 3600) -> None:
        """Start the traffic manager and servers."""
        logger.info(
            f"Starting Advanced Traffic Manager. Server: {self.server.name} ({self.server_ip})"
        )
        self._start_servers()
        self._start_agents(total_duration)

    def stop(self) -> None:
        """Stop all traffic and the management thread."""
        logger.info("Stopping Advanced Traffic Manager")
        self._cleanup_all()

    def _start_servers(self) -> None:
        """Start necessary server processes on the designated server host."""
        logger.info(f"Initializing servers on {self.server.name}")

        # HTTP server for downloads and web browsing
        self.server.cmd("python3 -m http.server 80 &")

        # Create a dummy large file for downloads if it doesn't exist
        self.server.cmd("dd if=/dev/zero of=large_file bs=1M count=100")

        # VLC stream server (using a dummy source)
        self.server.cmd(
            "cvlc -I dummy --ttl 12 --repeat /usr/share/vlc/lua/http/index.html --sout '#transcode{vcodec=h264,vb=800,scale=0.25,acodec=mpga,ab=128,channels=2,samplerate=44100}:std{access=http,mux=ts,dst=:8080}' &"
        )

        # UDP receiver for VoIP and Gaming
        self.server.cmd(
            f"python3 {self.script_path} --mode receiver --port 5005 --duration 7200 &"
        )

    def _start_agents(self, total_duration: int = 3600) -> None:
        """Start the host agent on each client host."""
        # All hosts except the server
        hosts = [h for h in self.net.hosts if h.name != self.server.name]

        if not hosts:
            logger.error("No client hosts available for traffic generation")
            return

        for host in hosts:
            logger.info(f"Starting host agent on {host.name}")
            cmd = (
                f"python3 {self.script_path} --mode agent "
                f"--target {self.server_ip} "
                f"--duration {total_duration} "
                f"--min_dur {self.config.min_duration} "
                f"--max_dur {self.config.max_duration} "
                f"--idle_prob {self.config.idle_probability} "
                f"--bw {self.config.voip_bandwidth} &"
            )
            host.cmd(cmd)

    def _cleanup_all(self) -> None:
        """Terminate all running traffic processes and servers."""
        # Kill agents and their children on all hosts
        for host in self.net.hosts:
            host.cmd("pkill -f traffic_generators.py")
            host.cmd("pkill -f cvlc")
            host.cmd("pkill -f wget")
            host.cmd("pkill -f curl")

        # Kill servers on the server host
        self.server.cmd("pkill -f 'http.server'")

        # Clean up dummy file
        self.server.cmd("rm -f large_file")

    def _run(self) -> None:
        """Main management loop (No longer used as agents manage themselves)."""
        pass
