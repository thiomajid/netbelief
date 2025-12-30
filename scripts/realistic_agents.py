#!/usr/bin/env python3
"""
Realistic network traffic agent implementations.

This script provides various traffic generator modes that simulate
real application traffic patterns for network telemetry collection.
Each mode mimics the network behavior of specific application types.
"""

import argparse
import random
import socket
import struct
import time
from dataclasses import dataclass


@dataclass
class PacketStats:
    """Statistics for packet transmission."""

    sent: int = 0
    bytes_sent: int = 0
    start_time: float = 0.0

    def report(self) -> str:
        elapsed = time.time() - self.start_time
        rate = self.bytes_sent * 8 / elapsed / 1000 if elapsed > 0 else 0
        return f"Sent {self.sent} packets, {self.bytes_sent} bytes, {rate:.2f} kbps"


def create_udp_socket() -> socket.socket:
    """Create a non-blocking UDP socket."""
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.setblocking(False)
    return sock


def create_tcp_socket() -> socket.socket:
    """Create a TCP socket with reasonable timeouts."""
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.settimeout(30)
    return sock


def run_voip_traffic(
    target_ip: str,
    port: int,
    bandwidth_bps: float,
    duration: float,
) -> None:
    """
    Simulate VoIP traffic using G.711 codec characteristics.

    G.711 produces 64 kbps audio with 20ms frames.
    Packet size: 160 bytes payload + 40 bytes headers = 200 bytes
    Packets per second: 50 (one every 20ms)
    """
    sock = create_udp_socket()

    # RTP-like packet structure: 12 byte header + payload
    rtp_header_size = 12
    payload_size = 160
    packet_size = rtp_header_size + payload_size

    # Calculate inter-packet interval
    packets_per_sec = bandwidth_bps / (packet_size * 8)
    interval = 1.0 / packets_per_sec if packets_per_sec > 0 else 0.02

    stats = PacketStats(start_time=time.time())
    sequence_number = 0
    timestamp = 0
    ssrc = random.randint(0, 0xFFFFFFFF)

    end_time = time.time() + duration

    while time.time() < end_time:
        # Build RTP-like header
        header = struct.pack(
            "!BBHII",
            0x80,  # Version 2, no padding, no extension
            0,  # Payload type (dynamic)
            sequence_number & 0xFFFF,
            timestamp,
            ssrc,
        )

        # Generate payload with slight variation (simulates speech patterns)
        if random.random() < 0.3:  # Silence periods
            payload = b"\x00" * payload_size
        else:
            payload = bytes(random.randint(0, 255) for _ in range(payload_size))

        packet = header + payload

        try:
            sock.sendto(packet, (target_ip, port))
            stats.sent += 1
            stats.bytes_sent += len(packet)
        except BlockingIOError:
            pass
        except Exception:
            pass

        sequence_number += 1
        timestamp += 160  # G.711: 160 samples per frame at 8kHz

        # Add jitter to simulate real-world timing
        jitter = random.gauss(0, interval * 0.1)
        time.sleep(max(0.001, interval + jitter))

    sock.close()
    print(f"VoIP: {stats.report()}")


def run_gaming_traffic(
    target_ip: str,
    port: int,
    duration: float,
) -> None:
    """
    Simulate online gaming traffic patterns.

    Characteristics:
    - Small packets (40-100 bytes)
    - High frequency (30-60 packets/sec)
    - Variable timing based on game state
    """
    sock = create_udp_socket()

    stats = PacketStats(start_time=time.time())
    tick_rate = random.choice([20, 30, 60, 128])  # Common tick rates
    base_interval = 1.0 / tick_rate

    end_time = time.time() + duration
    game_state = "idle"
    state_change_time = time.time() + random.uniform(5, 30)

    while time.time() < end_time:
        # Periodically change game state
        if time.time() > state_change_time:
            game_state = random.choice(["idle", "moving", "combat", "menu"])
            state_change_time = time.time() + random.uniform(5, 60)

        # Packet size varies by game state
        if game_state == "combat":
            packet_size = random.randint(80, 150)
            interval = base_interval * 0.7
        elif game_state == "moving":
            packet_size = random.randint(50, 80)
            interval = base_interval
        elif game_state == "menu":
            packet_size = random.randint(20, 40)
            interval = base_interval * 5
        else:
            packet_size = random.randint(30, 50)
            interval = base_interval * 2

        # Build game packet with sequence and timestamp
        seq_bytes = struct.pack("!I", stats.sent)
        time_bytes = struct.pack("!d", time.time())
        payload = bytes(random.randint(0, 255) for _ in range(packet_size - 12))
        packet = seq_bytes + time_bytes + payload

        try:
            sock.sendto(packet, (target_ip, port))
            stats.sent += 1
            stats.bytes_sent += len(packet)
        except BlockingIOError:
            pass
        except Exception:
            pass

        # Add realistic jitter
        jitter = random.gauss(0, interval * 0.15)
        time.sleep(max(0.001, interval + jitter))

    sock.close()
    print(f"Gaming: {stats.report()}")


def run_video_conference_traffic(
    target_ip: str,
    port: int,
    bandwidth_bps: float,
    duration: float,
) -> None:
    """
    Simulate video conference traffic.

    Combines audio (continuous, low bandwidth) with video
    (bursty, variable bandwidth based on motion).
    """
    sock = create_udp_socket()
    stats = PacketStats(start_time=time.time())

    # Audio: ~32kbps Opus
    audio_packet_size = 80
    audio_interval = 0.02  # 50 packets/sec

    # Video: Variable bitrate 300kbps - 2Mbps
    video_frame_interval = 1.0 / 30  # 30 fps
    base_video_bitrate = bandwidth_bps * 0.9

    end_time = time.time() + duration
    next_audio = time.time()
    next_video = time.time()

    motion_level = 0.3  # 0-1, affects video bitrate
    motion_change_time = time.time() + random.uniform(2, 10)

    while time.time() < end_time:
        current_time = time.time()

        # Update motion level periodically
        if current_time > motion_change_time:
            motion_level = random.uniform(0.1, 1.0)
            motion_change_time = current_time + random.uniform(2, 15)

        # Send audio packet
        if current_time >= next_audio:
            audio_packet = struct.pack("!BH", 0x01, stats.sent) + bytes(
                audio_packet_size - 3
            )
            try:
                sock.sendto(audio_packet, (target_ip, port))
                stats.sent += 1
                stats.bytes_sent += len(audio_packet)
            except Exception:
                pass
            next_audio = current_time + audio_interval + random.gauss(0, 0.002)

        # Send video frame
        if current_time >= next_video:
            # Frame size varies with motion
            frame_bits = base_video_bitrate * video_frame_interval * motion_level
            frame_size = int(frame_bits / 8)
            frame_size = max(500, min(frame_size, 1400))

            # Fragment large frames
            fragments = (frame_size + 1399) // 1400
            for frag in range(fragments):
                frag_size = min(1400, frame_size - frag * 1400)
                video_packet = struct.pack("!BHB", 0x02, stats.sent, frag) + bytes(
                    frag_size
                )
                try:
                    sock.sendto(video_packet, (target_ip, port))
                    stats.sent += 1
                    stats.bytes_sent += len(video_packet)
                except Exception:
                    pass

            next_video = current_time + video_frame_interval + random.gauss(0, 0.003)

        time.sleep(0.001)

    sock.close()
    print(f"Video Conference: {stats.report()}")


def run_iot_traffic(
    target_ip: str,
    port: int,
    duration: float,
) -> None:
    """
    Simulate IoT telemetry traffic.

    Characteristics:
    - Very small packets (32-128 bytes)
    - Periodic with long intervals (1-60 seconds)
    - Occasional bursts for events
    """
    sock = create_udp_socket()
    stats = PacketStats(start_time=time.time())

    # Base reporting interval
    report_interval = random.uniform(5, 30)

    end_time = time.time() + duration
    next_report = time.time()
    device_id = random.randint(0, 0xFFFF)

    while time.time() < end_time:
        current_time = time.time()

        if current_time >= next_report:
            # Regular telemetry packet
            sensor_value = random.uniform(0, 100)
            packet = struct.pack(
                "!HHfI",
                device_id,
                0x0001,  # Message type: telemetry
                sensor_value,
                int(current_time),
            )

            # Add some padding for realistic size
            padding_size = random.randint(0, 64)
            packet += bytes(padding_size)

            try:
                sock.sendto(packet, (target_ip, port))
                stats.sent += 1
                stats.bytes_sent += len(packet)
            except Exception:
                pass

            # Random event burst (alarms, state changes)
            if random.random() < 0.05:
                burst_count = random.randint(2, 5)
                for _ in range(burst_count):
                    event_packet = struct.pack(
                        "!HHI",
                        device_id,
                        0x0002,  # Message type: event
                        random.randint(1, 100),
                    )
                    try:
                        sock.sendto(event_packet, (target_ip, port))
                        stats.sent += 1
                        stats.bytes_sent += len(event_packet)
                    except Exception:
                        pass
                    time.sleep(0.01)

            next_report = current_time + report_interval + random.uniform(-2, 2)

        time.sleep(0.1)

    sock.close()
    print(f"IoT: {stats.report()}")


def run_streaming_traffic(
    target_ip: str,
    port: int,
    bandwidth_bps: float,
    duration: float,
) -> None:
    """
    Simulate adaptive streaming traffic (like HLS/DASH).

    Characteristics:
    - Chunk-based delivery
    - Adaptive bitrate switching
    - Buffering behavior
    """
    sock = create_udp_socket()
    stats = PacketStats(start_time=time.time())

    # Streaming segment duration (2-10 seconds typically)
    segment_duration = random.uniform(2, 6)

    # Available quality levels (kbps)
    quality_levels = [500, 1000, 2000, 4000, 8000]
    current_quality_idx = 2  # Start at middle quality

    end_time = time.time() + duration
    segment_start = time.time()
    buffer_level = 3.0  # Seconds of buffer

    while time.time() < end_time:
        current_time = time.time()
        segment_elapsed = current_time - segment_start

        # Simulate segment download
        if segment_elapsed < segment_duration:
            # Calculate bytes needed for this segment
            segment_bytes = (
                quality_levels[current_quality_idx] * 1000 * segment_duration / 8
            )
            packets_needed = int(segment_bytes / 1400)
            packet_interval = (
                segment_duration / packets_needed if packets_needed > 0 else 0.01
            )

            packet = bytes(random.randint(0, 255) for _ in range(1400))
            try:
                sock.sendto(packet, (target_ip, port))
                stats.sent += 1
                stats.bytes_sent += len(packet)
            except Exception:
                pass

            time.sleep(max(0.001, packet_interval))
        else:
            # Segment complete, decide on next quality
            buffer_level += segment_duration - (current_time - segment_start)
            buffer_level = min(buffer_level, 30)  # Max 30s buffer

            # Adaptive bitrate logic
            if buffer_level < 2 and current_quality_idx > 0:
                current_quality_idx -= 1
            elif buffer_level > 10 and current_quality_idx < len(quality_levels) - 1:
                if random.random() < 0.3:
                    current_quality_idx += 1

            segment_start = current_time

            # Small gap between segments
            time.sleep(random.uniform(0.1, 0.5))

        # Simulate playback consuming buffer
        buffer_level -= 0.001

    sock.close()
    print(f"Streaming: {stats.report()}")


def run_receiver(port: int, duration: float = 7200) -> None:
    """
    Generic UDP receiver for handling traffic from senders.
    Runs indefinitely or for specified duration.
    """
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    sock.bind(("0.0.0.0", port))
    sock.settimeout(1.0)

    stats = PacketStats(start_time=time.time())
    end_time = time.time() + duration

    print(f"Receiver started on port {port}")

    while time.time() < end_time:
        try:
            data, addr = sock.recvfrom(2048)
            stats.sent += 1
            stats.bytes_sent += len(data)
        except socket.timeout:
            continue
        except Exception as e:
            print(f"Receiver error: {e}")
            break

    sock.close()
    print(f"Receiver: Received {stats.sent} packets, {stats.bytes_sent} bytes")


def run_bulk_transfer(
    target_ip: str,
    port: int,
    size_bytes: int,
    upload: bool = True,
) -> None:
    """
    Simulate bulk file transfer over TCP.
    """
    try:
        sock = create_tcp_socket()
        sock.connect((target_ip, port))

        stats = PacketStats(start_time=time.time())
        chunk_size = 65536

        if upload:
            remaining = size_bytes
            while remaining > 0:
                chunk = bytes(
                    random.randint(0, 255) for _ in range(min(chunk_size, remaining))
                )
                sent = sock.send(chunk)
                stats.sent += 1
                stats.bytes_sent += sent
                remaining -= sent
        else:
            received = 0
            while received < size_bytes:
                data = sock.recv(chunk_size)
                if not data:
                    break
                received += len(data)
                stats.sent += 1
                stats.bytes_sent += len(data)

        sock.close()
        print(f"Bulk transfer: {stats.report()}")

    except Exception as e:
        print(f"Bulk transfer error: {e}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Realistic network traffic generator")
    parser.add_argument(
        "--mode",
        choices=[
            "voip",
            "gaming",
            "video_conference",
            "iot",
            "streaming",
            "receiver",
            "bulk_upload",
            "bulk_download",
        ],
        required=True,
        help="Traffic generation mode",
    )
    parser.add_argument("--target", type=str, help="Target IP address")
    parser.add_argument("--port", type=int, default=5060, help="Target port")
    parser.add_argument(
        "--bandwidth", type=float, default=96000, help="Target bandwidth in bps"
    )
    parser.add_argument(
        "--duration", type=float, default=60, help="Duration in seconds"
    )
    parser.add_argument(
        "--size", type=int, default=1048576, help="Transfer size in bytes"
    )

    args = parser.parse_args()

    if args.mode == "receiver":
        run_receiver(args.port, args.duration)
    elif args.mode == "voip":
        run_voip_traffic(args.target, args.port, args.bandwidth, args.duration)
    elif args.mode == "gaming":
        run_gaming_traffic(args.target, args.port, args.duration)
    elif args.mode == "video_conference":
        run_video_conference_traffic(
            args.target, args.port, args.bandwidth, args.duration
        )
    elif args.mode == "iot":
        run_iot_traffic(args.target, args.port, args.duration)
    elif args.mode == "streaming":
        run_streaming_traffic(args.target, args.port, args.bandwidth, args.duration)
    elif args.mode == "bulk_upload":
        run_bulk_transfer(args.target, args.port, args.size, upload=True)
    elif args.mode == "bulk_download":
        run_bulk_transfer(args.target, args.port, args.size, upload=False)


if __name__ == "__main__":
    main()
