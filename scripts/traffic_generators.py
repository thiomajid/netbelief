import argparse
import random
import socket
import subprocess
import time

from loguru import logger


def run_voip_sender(
    target_ip: str, port: int, bandwidth_bps: float, duration: float
) -> None:
    """Simulate VoIP sender (UDP)."""
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    # G.711: 64kbps payload + overhead ~= 80-100kbps
    # Packet size: 160 bytes payload + 40 bytes (IP/UDP/RTP) = 200 bytes
    packet_size = 200
    interval = (packet_size * 8) / bandwidth_bps

    start_time = time.time()
    while time.time() - start_time < duration:
        sock.sendto(b"x" * packet_size, (target_ip, port))
        time.sleep(interval)


def run_gaming_sender(target_ip: str, port: int, duration: float) -> None:
    """Simulate Gaming traffic (small UDP packets, low latency)."""
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    # Small packets, frequent updates
    packet_size = 64
    interval = 0.02  # 50 packets per second (20ms interval)

    start_time = time.time()
    while time.time() - start_time < duration:
        sock.sendto(b"g" * packet_size, (target_ip, port))
        time.sleep(interval)


def run_receiver(port: int, duration: float) -> None:
    """Simple UDP receiver to keep the port open."""
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.bind(("0.0.0.0", port))
    sock.settimeout(1.0)
    start_time = time.time()
    while time.time() - start_time < duration:
        try:
            data, addr = sock.recvfrom(2048)
        except socket.timeout:
            continue


def run_host_agent(
    server_ip: str,
    total_duration: float,
    min_task_dur: float,
    max_task_dur: float,
    idle_prob: float,
    voip_bw: str,
) -> None:
    """
    Main agent loop running on each host.
    Manages its own traffic choreography.
    """
    start_time = time.time()
    current_proc = None

    traffic_types = ["voip", "video", "download", "web", "gaming"]

    logger.info(f"Host agent started. Total duration: {total_duration}s")

    while time.time() - start_time < total_duration:
        # Pick new task
        if random.random() < idle_prob:
            task = "idle"
        else:
            task = random.choice(traffic_types)

        task_duration = random.uniform(min_task_dur, max_task_dur)
        # Ensure we don't exceed total simulation time
        remaining = total_duration - (time.time() - start_time)
        task_duration = min(task_duration, remaining)

        if task_duration <= 0:
            break

        logger.info(
            f"Switching activity to: {task.upper()} (duration: {task_duration:.1f}s)"
        )

        if task == "voip":
            bw = voip_bw.replace("k", "000")
            cmd = f"python3 {__file__} --mode voip --target {server_ip} --bw {bw} --duration {task_duration}"
            current_proc = subprocess.Popen(cmd, shell=True)
        elif task == "video":
            cmd = f"cvlc -I dummy http://{server_ip}:8080"
            current_proc = subprocess.Popen(cmd, shell=True)
        elif task == "download":
            # Download a large file, then wait if finished early
            cmd = f"wget -O /dev/null http://{server_ip}/large_file"
            current_proc = subprocess.Popen(cmd, shell=True)
        elif task == "web":
            # Periodic small HTTP GETs
            cmd = f"while true; do curl -s http://{server_ip}/ > /dev/null; sleep {random.uniform(1, 5)}; done"
            current_proc = subprocess.Popen(cmd, shell=True)
        elif task == "gaming":
            cmd = f"python3 {__file__} --mode gaming --target {server_ip} --duration {task_duration}"
            current_proc = subprocess.Popen(cmd, shell=True)
        else:
            current_proc = None

        # Wait for task duration
        task_start = time.time()
        while time.time() - task_start < task_duration:
            if current_proc and current_proc.poll() is not None:
                # Process finished early (e.g. download done)
                break
            time.sleep(0.5)

        # Cleanup task
        if current_proc:
            try:
                # Kill the process and its children (for the shell commands)
                subprocess.run(f"pkill -P {current_proc.pid}", shell=True)
                current_proc.terminate()
                current_proc.wait(timeout=2)
            except Exception as e:
                logger.warning(f"Error cleaning up task {task}: {e}")
                if current_proc:
                    current_proc.kill()
            current_proc = None

        # If it was a download or web that finished early, wait out the rest of the task duration
        remaining_task = task_duration - (time.time() - task_start)
        if remaining_task > 0:
            time.sleep(remaining_task)

    logger.info("Host agent finished.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--mode", choices=["voip", "gaming", "receiver", "agent"], required=True
    )
    parser.add_argument("--target", type=str)
    parser.add_argument("--port", type=int, default=5005)
    parser.add_argument("--bw", type=str, default="100k")
    parser.add_argument("--duration", type=float, default=60)
    parser.add_argument("--min_dur", type=float, default=30)
    parser.add_argument("--max_dur", type=float, default=120)
    parser.add_argument("--idle_prob", type=float, default=0.1)

    args = parser.parse_args()

    if args.mode == "voip":
        bw_val = float(args.bw.replace("k", "000"))
        run_voip_sender(args.target, args.port, bw_val, args.duration)
    elif args.mode == "gaming":
        run_gaming_sender(args.target, args.port, args.duration)
    elif args.mode == "receiver":
        run_receiver(args.port, args.duration)
    elif args.mode == "agent":
        run_host_agent(
            args.target,
            args.duration,
            args.min_dur,
            args.max_dur,
            args.idle_prob,
            args.bw,
        )
