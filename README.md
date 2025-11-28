# Network Visualization - Best Server Selection

A high-performance real-time visualization tool that demonstrates how to select the optimal server for file downloads in a network by analyzing latency and bandwidth constraints.

## 📋 Table of Contents

- [Overview](#overview)
- [How It Works](#how-it-works)
- [Network Model](#network-model)
- [Core Algorithms](#core-algorithms)
- [Code Structure](#code-structure)
- [Server Selection Logic](#server-selection-logic)
- [Visualization](#visualization)
- [Usage](#usage)

---

## Overview

This project simulates a network with:
- **1 client** requesting a file
- **8 servers** that can provide the file
- **40 nodes** total (client + servers + intermediate routers)
- **Edges** with varying latency and bandwidth

The goal is to **find which server can deliver the file fastest** by considering both network latency and available bandwidth.

---

## How It Works

### Step-by-Step Process

1. **Generate a random network graph** with nodes and edges
2. **Assign random properties** to edges (latency: 5-250ms, bandwidth: 1-300 Mbps)
3. **For each server**, calculate the download time to the client
4. **Select the server** with the minimum download time
5. **Visualize** the process in real-time, showing paths and metrics

### Download Time Formula

```
Download Time = Network Latency + Transfer Time
              = Latency (seconds) + (File Size in Mbits) / (Path Bandwidth in Mbps)
```

**Example:**
- File size: 80 MB = 640 Mbits
- Path latency: 150 ms = 0.15 seconds
- Path bandwidth: 100 Mbps
- **Download time** = 0.15 + (640 / 100) = **6.55 seconds**

---

## Network Model

### Network Parameters

```python
N_NODES = 40        # Total number of nodes
P_EDGE = 0.10       # Probability of edge between any two nodes (10%)
N_SERVERS = 8       # Number of servers
FILE_MB = 80.0      # File size in megabytes
```

### Node Types

1. **Client (Node 0)**: Green node, requests the file
2. **Servers (Nodes 1-8)**: Blue nodes with 🖥️ icon, all have capacity = 640 Mbps
3. **Intermediate Nodes**: Tiny gray nodes (routers/switches in the network)

### Edge Properties

Each edge has two properties:
- **Latency** (`latency_ms`): Time delay in milliseconds (5-250ms)
- **Bandwidth** (`bandwidth_mbps`): Maximum data rate in Mbps (1-300 Mbps)

### Server Capacity

All servers have **uniform upload capacity**:
```python
SERVER_CAPACITY_MBPS = FILE_MB * 8  # = 80 * 8 = 640 Mbps
```

This means servers themselves are **not the bottleneck**—the network path is!

---

## Core Algorithms

### 1. Dijkstra's Algorithm (Shortest Path by Latency)

**Purpose:** Find the path with minimum total latency from server to client.

**Function:** `dijkstra_latencies_nx(G, source)`

**How it works:**
```
1. Start at source node (a server)
2. Initialize all distances to infinity, except source = 0
3. Use priority queue to explore nodes by shortest distance
4. For each neighbor:
   - Calculate new distance = current_distance + edge_latency
   - If new distance < known distance, update it
5. Return dictionary of {node: minimum_latency}
```

**Code location:** Lines 52-59

**Example:**
```
Path: Server 3 → Node 15 → Node 8 → Client
Latencies: 50ms + 30ms + 70ms = 150ms total
```

### 2. Widest Path Algorithm (Maximum Bottleneck Bandwidth)

**Purpose:** Find the path with maximum minimum bandwidth (widest bottleneck) from server to client.

**Function:** `widest_path_nx(G, source)`

**How it works:**
```
1. Start at source node (a server) with infinite bandwidth
2. For each node, track the maximum bottleneck bandwidth reachable
3. Use max-heap (negated min-heap) to explore paths by best bandwidth
4. For each neighbor:
   - Bottleneck = min(current_bandwidth, edge_bandwidth)
   - If bottleneck > known_bandwidth to neighbor, update it
5. Return dictionary of {node: max_bottleneck_bandwidth} and parent pointers
```

**Code location:** Lines 58-81

**Example:**
```
Path: Server 3 → Node 15 → Node 8 → Client
Edge BWs: 200Mbps → 50Mbps → 150Mbps
Bottleneck = min(200, 50, 150) = 50 Mbps
```

**Why Widest Path?**
- We want the path that can sustain the **highest data rate**
- The weakest link (minimum bandwidth edge) determines the speed
- Widest path maximizes this minimum bandwidth

### 3. Path Reconstruction

**Function:** `reconstruct_path(parents, target, source)`

**How it works:**
```
1. Start at target node (client)
2. Follow parent pointers backwards
3. Stop when reaching source (server)
4. Reverse the path to get source → target direction
5. Return list of nodes in path
```

**Code location:** Lines 83-93

---

## Code Structure

### Helper Functions

#### `_edge_attr(edge_data, names)`
- **Purpose:** Extract an attribute from edge data, trying multiple possible names
- **Example:** Try 'bandwidth_mbps' first, then 'bandwidth'

#### `_get_edge_bandwidth(G, u, v)`
- **Purpose:** Get bandwidth of edge between nodes u and v
- **Handles:** Both regular graphs and multigraphs
- **Returns:** Maximum bandwidth if multiple edges exist

#### `_get_edge_latency(G, u, v)`
- **Purpose:** Get latency of edge between nodes u and v
- **Handles:** Both regular graphs and multigraphs
- **Returns:** Minimum latency if multiple edges exist

### Main Computation

#### `estimate_times_to_servers(G, client, servers, x_MB)`

**Purpose:** Calculate download time for file from each server to client.

**Parameters:**
- `G`: Network graph
- `client`: Client node ID
- `servers`: List of server node IDs
- `x_MB`: File size in megabytes

**Algorithm:**
```python
for each server:
    1. Run Dijkstra from server to get latencies
    2. Run Widest Path from server to get bandwidths
    3. Extract latency to client: lat_ms
    4. Extract bandwidth to client: path_bw
    5. Convert latency to seconds: lat_s = lat_ms / 1000
    6. Convert file to megabits: x_Mb = x_MB * 8
    7. Calculate: time[server] = lat_s + (x_Mb / path_bw)
```

**Returns:** 
- `times`: Dictionary {server: download_time_seconds}
- `all_widths`: Dictionary {server: path_bandwidth}
- `all_parents`: Dictionary {server: parent_pointers}

**Code location:** Lines 122-150

---

## Server Selection Logic

### How the Best Server is Chosen

The algorithm evaluates each server **sequentially** and keeps track of the current best:

```python
# Step 1: Calculate download time for all servers
all_times = estimate_times_to_servers(G, client, servers, FILE_MB)

# Step 2: During animation, evaluate one server per frame
for frame_idx in range(len(servers)):
    current_server = servers[frame_idx]
    download_time = all_times[current_server]
    
    # Step 3: Track evaluated times
    evaluated_times[current_server] = download_time

# Step 4: Find minimum among evaluated servers
good_servers = {s: time for s, time in evaluated_times.items() 
                if time is not infinity and time is not NaN}

if good_servers:
    best_server = min(good_servers, key=lambda s: good_servers[s])
```

### Selection Criteria

A server is chosen as **best** if:
1. ✅ Path exists from server to client (not infinity)
2. ✅ Has the **minimum download time** among all evaluated servers

Download time depends on:
- **Path latency** (lower is better)
- **Path bandwidth** (higher is better)
- **File size** (larger files amplify bandwidth differences)

### Example Comparison

```
Server 1: 150ms latency, 100 Mbps → 0.15 + (640/100) = 6.55s
Server 2: 50ms latency, 50 Mbps  → 0.05 + (640/50)  = 12.85s
Server 3: 200ms latency, 200 Mbps → 0.20 + (640/200) = 3.40s ⭐ BEST

Winner: Server 3 (lowest total time despite higher latency)
```

This shows **bandwidth matters more** for large files!

---

## Visualization

### Technology: Vispy (OpenGL)

The visualization uses **Vispy** for high-performance, GPU-accelerated rendering:
- Much faster than Matplotlib
- Smooth 60 FPS animations
- Interactive pan/zoom
- Real-time updates

### Visual Elements

#### Nodes
- **Client:** 30px bright green circle with 👤 icon
- **Servers:** 25px bright blue circles with 🖥️ icon
- **Intermediate:** 3px gray dots (minimized to reduce clutter)

#### Edges
- **Base network:** Very faint gray lines (15% opacity)
- **Active path:** Thick red lines (6px width)

#### Labels
- **Edge labels:** Show latency (ms) and bandwidth (Mbps) on active path
- **Server labels:** Show server ID and "✓ Ready" status
- **Client label:** Shows "CLIENT" with node ID

#### Information Panels

**Status Panel (Top-left):**
- Current frame number
- Server being evaluated
- Download time estimate
- Current best server

**Info Panel (Top-right):**
- Network statistics (nodes, edges)
- File size
- Server capacity (all servers)
- Keyboard controls

**Path Info Panel (Bottom-left):**
- Path length (number of hops)
- Total latency (sum of all edges)
- Path bandwidth (bottleneck)

### Animation Flow

```
Frame 0: Show base network
Frame 1: Evaluate Server 1, highlight path, show time
Frame 2: Evaluate Server 2, highlight path, show time, update best
...
Frame 8: Evaluate Server 8, final best server shown
Frame 9-18: Hold final state for viewing
```

---

## Usage

### Running the Visualization

```bash
# Run the live visualization
uv run main_live.py
```

### Keyboard Controls

| Key | Action |
|-----|--------|
| `SPACE` | Play/Pause animation |
| `R` | Reset to beginning |
| `Q` | Quit application |
| Mouse drag | Pan the view |
| Mouse scroll | Zoom in/out |

### Understanding the Output

1. **Press SPACE** to start the animation
2. **Watch the red path** highlight from each server to client
3. **Read edge labels** to see latency and bandwidth values
4. **Check path info** to see the bottleneck bandwidth
5. **See status panel** to know current best server
6. **Final frame** shows the winning server

### Modifying Parameters

You can change these values in the code to experiment:

```python
N_NODES = 40        # More nodes = more complex network
P_EDGE = 0.10       # Higher = denser network
N_SERVERS = 8       # Number of server options
FILE_MB = 80.0      # Larger file = bandwidth matters more
```

Edge properties (lines 109-111):
```python
latency_ms = round(random.uniform(5.0, 250.0), 1)      # Adjust range
bandwidth_mbps = round(random.uniform(1.0, 300.0), 1)  # Adjust range
```

---

## Key Insights

### 1. Bandwidth vs Latency Trade-off

For **small files**: Latency dominates
```
10 MB file (80 Mbits):
- Path A: 50ms, 50 Mbps  → 0.05 + 1.6  = 1.65s
- Path B: 150ms, 200 Mbps → 0.15 + 0.4  = 0.55s ⭐ BEST
```

For **large files**: Bandwidth dominates
```
1000 MB file (8000 Mbits):
- Path A: 50ms, 50 Mbps   → 0.05 + 160 = 160.05s
- Path B: 150ms, 200 Mbps → 0.15 + 40  = 40.15s ⭐ BEST
```

### 2. Path Quality Matters

A path through many low-bandwidth edges is slower than a direct high-bandwidth path, even if slightly longer in latency.

### 3. Uniform Server Capacity

Since all servers have sufficient capacity (640 Mbps), the **network path is always the bottleneck**, not the server itself. This reflects real-world CDN scenarios where servers are powerful but network quality varies.

---

## File Structure

```
network_project/
├── main.py              # Original matplotlib version (exports GIF/MP4)
├── main_live.py         # Vispy real-time version (this README)
├── pyproject.toml       # Dependencies
└── README.md           # This file
```

---

## Dependencies

```toml
ipython>=9.7.0
matplotlib>=3.10.7
networkx>=3.6
vispy>=0.14.0
PyQt6>=6.8.0
```

Install with:
```bash
uv sync
```

---

## Algorithm Complexity

- **Dijkstra's Algorithm:** O((V + E) log V) where V = nodes, E = edges
- **Widest Path Algorithm:** O((V + E) log V) 
- **Path Reconstruction:** O(path_length) ≈ O(V) worst case
- **Total per server:** O((V + E) log V)
- **For all servers:** O(S × (V + E) log V) where S = number of servers

With V=40, E≈78, S=8: Very fast, runs in milliseconds.

---

## Educational Value

This project demonstrates:
- ✅ Graph algorithms (Dijkstra, Widest Path)
- ✅ Network optimization
- ✅ Real-time visualization
- ✅ Trade-offs in system design
- ✅ Priority queue usage
- ✅ GPU-accelerated graphics with Vispy

Perfect for learning about:
- Computer networks
- CDN server selection
- Graph theory
- Algorithm visualization

---

## Author & License

Created as an educational tool for understanding network path selection algorithms.

Enjoy exploring how networks choose the best path! 🚀
