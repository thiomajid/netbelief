"""
Traffic profile definitions for realistic network simulation.

Each profile defines the characteristics of a specific application type,
including bandwidth patterns, timing, and behavior parameters that mimic
real-world usage patterns.
"""

import typing as tp
from dataclasses import dataclass, field
from enum import Enum


class ApplicationType(Enum):
    """Types of simulated network applications."""

    WEB_BROWSING = "web_browsing"
    VIDEO_STREAMING = "video_streaming"
    VOIP_CALL = "voip_call"
    ONLINE_GAMING = "online_gaming"
    FILE_DOWNLOAD = "file_download"
    FILE_UPLOAD = "file_upload"
    VIDEO_CONFERENCE = "video_conference"
    MUSIC_STREAMING = "music_streaming"
    SOCIAL_MEDIA = "social_media"
    EMAIL = "email"
    CLOUD_SYNC = "cloud_sync"
    IOT_TELEMETRY = "iot_telemetry"
    IDLE = "idle"


@dataclass
class BandwidthProfile:
    """Bandwidth characteristics for an application."""

    min_kbps: float
    max_kbps: float
    avg_kbps: float
    burst_probability: float = 0.0
    burst_multiplier: float = 1.0


@dataclass
class TimingProfile:
    """Timing characteristics for an application session."""

    min_duration_sec: float
    max_duration_sec: float
    avg_duration_sec: float
    think_time_min_sec: float = 0.0
    think_time_max_sec: float = 0.0


@dataclass
class PacketProfile:
    """Packet-level characteristics."""

    min_size_bytes: int
    max_size_bytes: int
    avg_size_bytes: int
    packets_per_second_min: float
    packets_per_second_max: float


@dataclass
class TrafficProfile:
    """Complete traffic profile for an application type."""

    app_type: ApplicationType
    protocol: str
    port_range: tp.Tuple[int, int]
    bandwidth: BandwidthProfile
    timing: TimingProfile
    packets: PacketProfile
    bidirectional: bool = False
    server_required: bool = True
    human_variability: float = 0.3


# Pre-defined traffic profiles based on real-world measurements
WEB_BROWSING_PROFILE = TrafficProfile(
    app_type=ApplicationType.WEB_BROWSING,
    protocol="tcp",
    port_range=(80, 443),
    bandwidth=BandwidthProfile(
        min_kbps=50,
        max_kbps=5000,
        avg_kbps=500,
        burst_probability=0.7,
        burst_multiplier=3.0,
    ),
    timing=TimingProfile(
        min_duration_sec=2,
        max_duration_sec=30,
        avg_duration_sec=8,
        think_time_min_sec=1,
        think_time_max_sec=60,
    ),
    packets=PacketProfile(
        min_size_bytes=64,
        max_size_bytes=1460,
        avg_size_bytes=800,
        packets_per_second_min=5,
        packets_per_second_max=200,
    ),
    human_variability=0.4,
)

VIDEO_STREAMING_PROFILE = TrafficProfile(
    app_type=ApplicationType.VIDEO_STREAMING,
    protocol="tcp",
    port_range=(443, 8080),
    bandwidth=BandwidthProfile(
        min_kbps=1000,
        max_kbps=25000,
        avg_kbps=5000,
        burst_probability=0.2,
        burst_multiplier=2.0,
    ),
    timing=TimingProfile(
        min_duration_sec=120,
        max_duration_sec=7200,
        avg_duration_sec=1800,
        think_time_min_sec=0,
        think_time_max_sec=5,
    ),
    packets=PacketProfile(
        min_size_bytes=500,
        max_size_bytes=1460,
        avg_size_bytes=1200,
        packets_per_second_min=100,
        packets_per_second_max=2000,
    ),
    human_variability=0.2,
)

VOIP_CALL_PROFILE = TrafficProfile(
    app_type=ApplicationType.VOIP_CALL,
    protocol="udp",
    port_range=(5060, 5080),
    bandwidth=BandwidthProfile(
        min_kbps=64,
        max_kbps=128,
        avg_kbps=96,
        burst_probability=0.05,
        burst_multiplier=1.2,
    ),
    timing=TimingProfile(
        min_duration_sec=60,
        max_duration_sec=1800,
        avg_duration_sec=300,
        think_time_min_sec=0,
        think_time_max_sec=0,
    ),
    packets=PacketProfile(
        min_size_bytes=160,
        max_size_bytes=200,
        avg_size_bytes=180,
        packets_per_second_min=30,
        packets_per_second_max=50,
    ),
    bidirectional=True,
    human_variability=0.1,
)

ONLINE_GAMING_PROFILE = TrafficProfile(
    app_type=ApplicationType.ONLINE_GAMING,
    protocol="udp",
    port_range=(27015, 27030),
    bandwidth=BandwidthProfile(
        min_kbps=30,
        max_kbps=500,
        avg_kbps=100,
        burst_probability=0.3,
        burst_multiplier=2.0,
    ),
    timing=TimingProfile(
        min_duration_sec=600,
        max_duration_sec=14400,
        avg_duration_sec=3600,
        think_time_min_sec=0,
        think_time_max_sec=0,
    ),
    packets=PacketProfile(
        min_size_bytes=40,
        max_size_bytes=256,
        avg_size_bytes=64,
        packets_per_second_min=20,
        packets_per_second_max=120,
    ),
    bidirectional=True,
    human_variability=0.15,
)

FILE_DOWNLOAD_PROFILE = TrafficProfile(
    app_type=ApplicationType.FILE_DOWNLOAD,
    protocol="tcp",
    port_range=(80, 443),
    bandwidth=BandwidthProfile(
        min_kbps=500,
        max_kbps=100000,
        avg_kbps=10000,
        burst_probability=0.1,
        burst_multiplier=1.5,
    ),
    timing=TimingProfile(
        min_duration_sec=5,
        max_duration_sec=600,
        avg_duration_sec=60,
        think_time_min_sec=0,
        think_time_max_sec=0,
    ),
    packets=PacketProfile(
        min_size_bytes=1000,
        max_size_bytes=1460,
        avg_size_bytes=1400,
        packets_per_second_min=50,
        packets_per_second_max=8000,
    ),
    human_variability=0.1,
)

VIDEO_CONFERENCE_PROFILE = TrafficProfile(
    app_type=ApplicationType.VIDEO_CONFERENCE,
    protocol="udp",
    port_range=(3478, 3481),
    bandwidth=BandwidthProfile(
        min_kbps=500,
        max_kbps=4000,
        avg_kbps=1500,
        burst_probability=0.15,
        burst_multiplier=1.5,
    ),
    timing=TimingProfile(
        min_duration_sec=300,
        max_duration_sec=7200,
        avg_duration_sec=1800,
        think_time_min_sec=0,
        think_time_max_sec=0,
    ),
    packets=PacketProfile(
        min_size_bytes=200,
        max_size_bytes=1200,
        avg_size_bytes=600,
        packets_per_second_min=30,
        packets_per_second_max=200,
    ),
    bidirectional=True,
    human_variability=0.2,
)

MUSIC_STREAMING_PROFILE = TrafficProfile(
    app_type=ApplicationType.MUSIC_STREAMING,
    protocol="tcp",
    port_range=(443, 443),
    bandwidth=BandwidthProfile(
        min_kbps=96,
        max_kbps=320,
        avg_kbps=160,
        burst_probability=0.3,
        burst_multiplier=2.0,
    ),
    timing=TimingProfile(
        min_duration_sec=180,
        max_duration_sec=7200,
        avg_duration_sec=1800,
        think_time_min_sec=0,
        think_time_max_sec=30,
    ),
    packets=PacketProfile(
        min_size_bytes=500,
        max_size_bytes=1460,
        avg_size_bytes=1000,
        packets_per_second_min=15,
        packets_per_second_max=40,
    ),
    human_variability=0.25,
)

SOCIAL_MEDIA_PROFILE = TrafficProfile(
    app_type=ApplicationType.SOCIAL_MEDIA,
    protocol="tcp",
    port_range=(443, 443),
    bandwidth=BandwidthProfile(
        min_kbps=100,
        max_kbps=8000,
        avg_kbps=1000,
        burst_probability=0.5,
        burst_multiplier=4.0,
    ),
    timing=TimingProfile(
        min_duration_sec=5,
        max_duration_sec=120,
        avg_duration_sec=30,
        think_time_min_sec=2,
        think_time_max_sec=120,
    ),
    packets=PacketProfile(
        min_size_bytes=64,
        max_size_bytes=1460,
        avg_size_bytes=600,
        packets_per_second_min=10,
        packets_per_second_max=500,
    ),
    human_variability=0.5,
)

EMAIL_PROFILE = TrafficProfile(
    app_type=ApplicationType.EMAIL,
    protocol="tcp",
    port_range=(993, 993),
    bandwidth=BandwidthProfile(
        min_kbps=10,
        max_kbps=2000,
        avg_kbps=200,
        burst_probability=0.6,
        burst_multiplier=5.0,
    ),
    timing=TimingProfile(
        min_duration_sec=1,
        max_duration_sec=60,
        avg_duration_sec=10,
        think_time_min_sec=30,
        think_time_max_sec=600,
    ),
    packets=PacketProfile(
        min_size_bytes=64,
        max_size_bytes=1460,
        avg_size_bytes=500,
        packets_per_second_min=5,
        packets_per_second_max=100,
    ),
    human_variability=0.4,
)

CLOUD_SYNC_PROFILE = TrafficProfile(
    app_type=ApplicationType.CLOUD_SYNC,
    protocol="tcp",
    port_range=(443, 443),
    bandwidth=BandwidthProfile(
        min_kbps=50,
        max_kbps=50000,
        avg_kbps=5000,
        burst_probability=0.4,
        burst_multiplier=3.0,
    ),
    timing=TimingProfile(
        min_duration_sec=10,
        max_duration_sec=300,
        avg_duration_sec=60,
        think_time_min_sec=60,
        think_time_max_sec=900,
    ),
    packets=PacketProfile(
        min_size_bytes=64,
        max_size_bytes=1460,
        avg_size_bytes=1200,
        packets_per_second_min=10,
        packets_per_second_max=3000,
    ),
    human_variability=0.3,
)

IOT_TELEMETRY_PROFILE = TrafficProfile(
    app_type=ApplicationType.IOT_TELEMETRY,
    protocol="udp",
    port_range=(1883, 1883),
    bandwidth=BandwidthProfile(
        min_kbps=0.5,
        max_kbps=10,
        avg_kbps=2,
        burst_probability=0.1,
        burst_multiplier=2.0,
    ),
    timing=TimingProfile(
        min_duration_sec=0.1,
        max_duration_sec=2,
        avg_duration_sec=0.5,
        think_time_min_sec=5,
        think_time_max_sec=60,
    ),
    packets=PacketProfile(
        min_size_bytes=32,
        max_size_bytes=256,
        avg_size_bytes=64,
        packets_per_second_min=0.1,
        packets_per_second_max=10,
    ),
    human_variability=0.1,
)


TRAFFIC_PROFILES: tp.Dict[ApplicationType, TrafficProfile] = {
    ApplicationType.WEB_BROWSING: WEB_BROWSING_PROFILE,
    ApplicationType.VIDEO_STREAMING: VIDEO_STREAMING_PROFILE,
    ApplicationType.VOIP_CALL: VOIP_CALL_PROFILE,
    ApplicationType.ONLINE_GAMING: ONLINE_GAMING_PROFILE,
    ApplicationType.FILE_DOWNLOAD: FILE_DOWNLOAD_PROFILE,
    ApplicationType.VIDEO_CONFERENCE: VIDEO_CONFERENCE_PROFILE,
    ApplicationType.MUSIC_STREAMING: MUSIC_STREAMING_PROFILE,
    ApplicationType.SOCIAL_MEDIA: SOCIAL_MEDIA_PROFILE,
    ApplicationType.EMAIL: EMAIL_PROFILE,
    ApplicationType.CLOUD_SYNC: CLOUD_SYNC_PROFILE,
    ApplicationType.IOT_TELEMETRY: IOT_TELEMETRY_PROFILE,
}


@dataclass
class UserBehaviorProfile:
    """Defines a user's behavior pattern throughout the day."""

    name: str
    app_weights: tp.Dict[ApplicationType, float] = field(default_factory=dict)
    activity_level: float = 1.0
    session_multiplier: float = 1.0
    switch_probability: float = 0.3


# Pre-defined user behavior profiles
CASUAL_USER = UserBehaviorProfile(
    name="casual",
    app_weights={
        ApplicationType.WEB_BROWSING: 0.3,
        ApplicationType.SOCIAL_MEDIA: 0.25,
        ApplicationType.VIDEO_STREAMING: 0.2,
        ApplicationType.MUSIC_STREAMING: 0.15,
        ApplicationType.EMAIL: 0.1,
    },
    activity_level=0.6,
    session_multiplier=0.8,
    switch_probability=0.4,
)

POWER_USER = UserBehaviorProfile(
    name="power",
    app_weights={
        ApplicationType.FILE_DOWNLOAD: 0.2,
        ApplicationType.VIDEO_STREAMING: 0.2,
        ApplicationType.CLOUD_SYNC: 0.15,
        ApplicationType.WEB_BROWSING: 0.15,
        ApplicationType.VIDEO_CONFERENCE: 0.15,
        ApplicationType.SOCIAL_MEDIA: 0.1,
        ApplicationType.ONLINE_GAMING: 0.05,
    },
    activity_level=0.9,
    session_multiplier=1.2,
    switch_probability=0.25,
)

GAMER_USER = UserBehaviorProfile(
    name="gamer",
    app_weights={
        ApplicationType.ONLINE_GAMING: 0.5,
        ApplicationType.VIDEO_STREAMING: 0.2,
        ApplicationType.VOIP_CALL: 0.15,
        ApplicationType.FILE_DOWNLOAD: 0.1,
        ApplicationType.WEB_BROWSING: 0.05,
    },
    activity_level=0.85,
    session_multiplier=1.5,
    switch_probability=0.15,
)

OFFICE_USER = UserBehaviorProfile(
    name="office",
    app_weights={
        ApplicationType.VIDEO_CONFERENCE: 0.3,
        ApplicationType.EMAIL: 0.25,
        ApplicationType.CLOUD_SYNC: 0.2,
        ApplicationType.WEB_BROWSING: 0.15,
        ApplicationType.VOIP_CALL: 0.1,
    },
    activity_level=0.8,
    session_multiplier=1.0,
    switch_probability=0.2,
)

IOT_DEVICE = UserBehaviorProfile(
    name="iot",
    app_weights={
        ApplicationType.IOT_TELEMETRY: 1.0,
    },
    activity_level=0.95,
    session_multiplier=0.1,
    switch_probability=0.0,
)


USER_PROFILES: tp.Dict[str, UserBehaviorProfile] = {
    "casual": CASUAL_USER,
    "power": POWER_USER,
    "gamer": GAMER_USER,
    "office": OFFICE_USER,
    "iot": IOT_DEVICE,
}
