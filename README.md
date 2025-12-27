# VoIP Simulation Documentation

## Overview

This simulation uses Mininet to emulate a network based on the Airtel topology. It generates synthetic VoIP traffic alongside background TCP traffic to simulate a realistic network environment. The goal is to collect network performance metrics (Jitter, Packet Loss, Bandwidth) suitable for training time-series forecasting models.

## Simulation Logic

### 1. Topology Initialization

The script imports the custom `GeneratedTopo` class (from `airtel.py`) which defines the switches and links.

```python
# Initialize Topology
topo = GeneratedTopo()
net = Mininet(
    topo=topo, controller=OVSController, link=TCLink, switch=OVSKernelSwitch
)
net.start()
```

**Spanning Tree Protocol (STP)** is enabled on all switches to prevent routing loops in the mesh topology. This is crucial because the Airtel topology contains cycles.

```python
# Enable STP on all switches
for sw in net.switches:
    sw.cmd("ovs-vsctl set bridge {} stp_enable=true".format(sw.name))
```

### 2. Traffic Generation

The simulation generates two types of traffic to create a realistic network load.

#### A. VoIP Traffic (UDP)

- **Protocol**: UDP (User Datagram Protocol), which is the standard transport for real-time voice applications.
- **Characteristics**:
  - **Bandwidth**: `100k` (100 Kbps). This approximates a G.711 codec stream with IP/UDP/RTP headers.
  - **Bidirectional**: Voice calls are two-way. If Host A calls Host B, traffic flows A &rarr; B and B &rarr; A simultaneously.
- **Tool**: `iperf` in UDP mode (`-u`).

```python
# Start VoIP Clients (Bidirectional)
h1.cmd(
    "iperf -c {} -u -b {} -t {} -i 1 > {}/iperf_client_{}_to_{}.log &".format(
        h2.IP(), VOIP_BW, DURATION, OUTPUT_DIR, h1.name, h2.name
    )
)
```

#### B. Background Traffic (TCP)

- **Protocol**: TCP.
- **Purpose**: To introduce congestion, queueing delays, and variability in the network. Without this, the VoIP flows might experience near-perfect conditions, resulting in trivial data (0 jitter, 0 loss).
- **Tool**: `iperf` (default TCP mode).

```python
# Background traffic servers (TCP)
bg_hosts = random.sample(hosts, min(len(hosts), BG_TRAFFIC_COUNT * 2))
# ...
c.cmd(
    "iperf -c {} -t {} -i 1 > {}/iperf_bg_client_{}_to_{}.log &".format(
        s.IP(), DURATION, OUTPUT_DIR, c.name, s.name
    )
)
```

## Log Files Explained

The simulation produces several types of log files in the `dumps/` directory. Understanding these is key to interpreting the data.

| File Pattern                         | Type                    | Description                                                                                                                                                                                            |
| :----------------------------------- | :---------------------- | :----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `iperf_server_{host}.log`            | **VoIP Receiver**       | **(Primary Data Source)** Contains the critical metrics for VoIP. In UDP `iperf`, the _server_ (receiver) calculates Jitter and Packet Loss. These files are parsed to generate the final CSV dataset. |
| `iperf_client_{src}_to_{dst}.log`    | **VoIP Sender**         | Logs from the sender side. These are less useful for quality analysis because the sender does not know about packet loss or jitter that occurred during transit.                                       |
| `iperf_bg_server_{host}.log`         | **Background Receiver** | Logs for the background TCP traffic receivers. Useful for debugging network congestion levels but not used in the final VoIP dataset.                                                                  |
| `iperf_bg_client_{src}_to_{dst}.log` | **Background Sender**   | Logs for the background TCP traffic senders.                                                                                                                                                           |

## Data Collection & Parsing

The script collects data from three sources:

1.  **VoIP Server Logs (`iperf`)**: Provides Bandwidth, Jitter, and Packet Loss.
2.  **Ping Logs (`ping`)**: Provides Round Trip Time (RTT) / Latency.
3.  **Switch Statistics (`ovs-ofctl`)**: Provides low-level port counters (packets, bytes, drops, errors) from all switches.

### Data Merging

The `process_logs` function merges these disparate data sources into a single CSV file:

- **Synchronization**: Data points are aligned based on timestamps.
- **MOS Calculation**: The Mean Opinion Score (MOS) is calculated for each second using the E-model formula, based on the measured RTT and Packet Loss.
- **Switch Aggregation**: Switch statistics are aggregated across the entire network to provide a global view of network congestion (e.g., total packets dropped per second).

The final output is `dumps/voip_simulation_data.csv`.

## Metrics Explained

The dataset (`dumps/voip_simulation_data.csv`) contains the following features for each 1-second interval of a VoIP flow. These metrics are used as input features or targets for the forecasting model.

### VoIP Flow Metrics (Target Variables)

These metrics directly measure the quality of the VoIP call.

| Metric                | Description                                                                                                                                               | Unit     | Source     |
| :-------------------- | :-------------------------------------------------------------------------------------------------------------------------------------------------------- | :------- | :--------- |
| `bandwidth_bps`       | The amount of data successfully transferred per second. For VoIP, this should be stable around the codec rate (e.g., 100kbps). Drops indicate congestion. | bits/sec | `iperf`    |
| `jitter_ms`           | The variation in packet arrival time. High jitter causes voice distortion.                                                                                | ms       | `iperf`    |
| `packet_loss_percent` | The percentage of packets lost during transmission. >1% is noticeable; >5% is unacceptable.                                                               | %        | `iperf`    |
| `mos`                 | **Mean Opinion Score**. A composite score derived from Latency and Packet Loss using the E-model. Ranges from 1 (Bad) to 5 (Excellent).                   | 1-5      | Calculated |
| `rtt_ms`              | **Round Trip Time**. The time it takes for a packet to go from source to destination and back. High latency causes "talk-over" in conversations.          | ms       | `ping`     |

### Network-Wide Switch Metrics (Input Features)

These metrics represent the global state of the network infrastructure. They are aggregated (summed) across **all switches** in the topology for that specific second. They serve as leading indicators for congestion.

| Metric         | Description                                                                                                                                     | Source |
| :------------- | :---------------------------------------------------------------------------------------------------------------------------------------------- | :----- |
| `net_rx_pkts`  | Total number of packets received by all switch ports. Indicates total network load.                                                             | OVS    |
| `net_rx_bytes` | Total number of bytes received by all switch ports.                                                                                             | OVS    |
| `net_rx_drop`  | **Critical**. Total number of packets dropped by switches upon reception (e.g., due to full buffers). High values predict packet loss in flows. | OVS    |
| `net_rx_errs`  | Total number of receive errors (e.g., CRC errors, alignment errors). Indicates physical link issues.                                            | OVS    |
| `net_tx_pkts`  | Total number of packets transmitted by all switch ports.                                                                                        | OVS    |
| `net_tx_bytes` | Total number of bytes transmitted by all switch ports.                                                                                          | OVS    |
| `net_tx_drop`  | Total number of packets dropped during transmission (e.g., queue overflow).                                                                     | OVS    |
| `net_tx_errs`  | Total number of transmission errors.                                                                                                            | OVS    |

### Metadata

| Metric            | Description                                                  |
| :---------------- | :----------------------------------------------------------- |
| `timestamp`       | The simulation time in seconds (relative to start).          |
| `flow_id`         | Unique identifier for the VoIP flow (e.g., `h1->h2`).        |
| `interval_start`  | Start time of the measurement interval.                      |
| `interval_end`    | End time of the measurement interval.                        |
| `packet_loss_cnt` | Absolute count of lost packets (numerator for percentage).   |
| `total_packets`   | Total packets sent in interval (denominator for percentage). |

---

## Network Variability Simulation

The simulation includes a configurable variability system that introduces controlled network perturbations. This creates more realistic time series data by simulating real-world network events like link failures, congestion, and quality degradation.

### Design Philosophy

The variability system is designed to create **meaningful patterns** in the data without excessive noise. Events are:

- **Infrequent enough** to be distinguishable in the time series
- **Impactful enough** to create visible changes in metrics
- **Configurable** to allow tuning for different analysis requirements

### Event Types

| Event Type              | Description                                  | Effect on Metrics                                     |
| :---------------------- | :------------------------------------------- | :---------------------------------------------------- |
| **Link Failure**        | Complete link outage with automatic recovery | Sudden spike in packet loss, possible routing changes |
| **Delay Variation**     | Changes in link propagation delay            | Increased RTT, jitter fluctuations                    |
| **Bandwidth Variation** | Capacity fluctuations                        | Bandwidth drops, potential queue buildup              |
| **Packet Loss**         | Random packet drops                          | Direct packet loss increase                           |
| **Congestion**          | Combined bandwidth/delay/loss effects        | Multiple metrics affected simultaneously              |

### Configuration

Variability is configured in `config/simulation.yaml` under the `variability` section:

```yaml
variability:
  enabled: true
  seed: 42 # For reproducibility

  link_failure:
    enabled: true
    probability: 0.05 # 5% chance per check
    min_interval: 60.0 # Seconds between checks
    max_interval: 300.0
    min_duration: 10.0 # Failure duration
    max_duration: 60.0
    max_concurrent_failures: 0.1 # Max 10% of links
```

### Preset Configurations

Two preset configurations are provided:

1. **`simulation_with_variability.yaml`**: Moderate variability for typical time series analysis
2. **`simulation_minimal_variability.yaml`**: Subtle perturbations for longer simulations

### Event Logging

All network events are logged to `dumps/network_events.json`:

```json
[
  {
    "timestamp": 125.45,
    "event_type": "link_failure",
    "source_node": "s0",
    "target_node": "s7",
    "duration": 45.2
  },
  {
    "timestamp": 170.65,
    "event_type": "link_recovery",
    "source_node": "s0",
    "target_node": "s7"
  }
]
```

### Tuning Guidelines

For time series forecasting, consider:

| Goal                     | Recommended Settings                                   |
| :----------------------- | :----------------------------------------------------- |
| **More stable baseline** | Lower probabilities, longer intervals                  |
| **More variation**       | Higher probabilities, shorter intervals                |
| **Distinct events**      | Enable `revert_after` to create clear event boundaries |
| **Gradual changes**      | Disable `revert_after` for persistent state changes    |

### Implementation Details

The variability system runs in background threads, with each feature type having its own independent loop:

1. Each loop waits for a random interval within its configured range
2. A probability check determines if an event occurs
3. If triggered, the event is applied and logged
4. Automatic reversal is scheduled if `revert_after` is configured

Link modifications use Mininet's `configLinkStatus()` for failures and the TCLink interface for parameter changes (delay, bandwidth, loss).
