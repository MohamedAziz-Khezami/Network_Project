# Real-time network visualization using Vispy with curved edges
# Run directly from terminal: uv run main_live.py
# Press SPACE to start/pause, R to reset, Q to quit

import networkx as nx
from heapq import heappush, heappop
from math import inf
from typing import Any, Dict, List, Optional, Tuple
import random
import numpy as np
from vispy import app, scene
from vispy.scene import visuals
from vispy.color import Color

# ---------------- Helpers ----------------
def _edge_attr(edge_data: dict, names):
    for n in names:
        if n in edge_data:
            return edge_data[n]
    return None

def _get_edge_bandwidth(G: nx.Graph, u, v) -> float:
    if G.is_multigraph():
        ed = G.get_edge_data(u, v, default={})
        if not ed: return 0.0
        max_bw = 0.0
        for key, attr in ed.items():
            bw = _edge_attr(attr, ('bandwidth_mbps', 'bandwidth'))
            if bw is None: continue
            max_bw = max(max_bw, float(bw))
        return max_bw
    else:
        attr = G.get_edge_data(u, v, default={})
        bw = _edge_attr(attr, ('bandwidth_mbps', 'bandwidth'))
        return float(bw) if bw is not None else 0.0

def _get_edge_latency(G: nx.Graph, u, v) -> float:
    if G.is_multigraph():
        ed = G.get_edge_data(u, v, default={})
        if not ed: return 0.0
        min_lat = inf
        for key, attr in ed.items():
            lat = _edge_attr(attr, ('latency_ms', 'latency'))
            if lat is None: continue
            min_lat = min(min_lat, float(lat))
        return 0.0 if min_lat == inf else min_lat
    else:
        attr = G.get_edge_data(u, v, default={})
        lat = _edge_attr(attr, ('latency_ms', 'latency'))
        return float(lat) if lat is not None else 0.0

def dijkstra_latencies_nx(G: nx.Graph, source) -> Dict[Any, float]:
    """
    Calculates the shortest path latencies from a source node to all other nodes in a graph
    using Dijkstra's algorithm (manual implementation).

    Args:
        G (nx.Graph): The input graph.
        source: The source node from which to calculate latencies.

    Returns:
        Dict[Any, float]: A dictionary where keys are nodes and values are the
                         shortest path latencies from the source to that node.
                         Returns 0.0 for the source node and infinity for unreachable nodes.
    """
    # Initialize distances to all nodes as infinity
    distances = {node: inf for node in G.nodes()}
    distances[source] = 0.0
    
    # Priority queue: stores (distance, node) tuples
    # Using negative distance would give max heap, but we want min heap for Dijkstra
    pq = [(0.0, source)]
    
    # Set to track visited nodes
    visited = set()
    
    while pq:
        # Get node with minimum distance
        current_dist, current_node = heappop(pq)
        
        # Skip if already visited
        if current_node in visited:
            continue
        
        visited.add(current_node)
        
        # If the distance we popped is worse than what we have, skip
        if current_dist > distances[current_node]:
            continue
        
        # Explore neighbors
        for neighbor in G.neighbors(current_node):
            # Get edge latency
            edge_latency = _get_edge_latency(G, current_node, neighbor)
            
            # Skip edges with no latency
            if edge_latency <= 0:
                continue
            
            # Calculate new distance through current node
            new_distance = current_dist + edge_latency
            
            # If we found a shorter path, update it
            if new_distance < distances[neighbor]:
                distances[neighbor] = new_distance
                heappush(pq, (new_distance, neighbor))
    
    return distances

def widest_path_nx(G: nx.Graph, source) -> Tuple[Dict[Any, float], Dict[Any, Optional[Any]]]:
    # Initialize bandwidths to all nodes as 0, except source which is infinity
    bw = {n: 0.0 for n in G.nodes()}
    # Initialize parent pointers for path reconstruction
    parent: Dict[Any, Optional[Any]] = {n: None for n in G.nodes()}
    bw[source] = inf
    # Priority queue stores (negative_bottleneck_bandwidth, node)
    # We use negative bandwidth because heappop retrieves the smallest item,
    # and we want the largest bandwidth.
    pq = [(-bw[source], source)]

    # Process nodes while the priority queue is not empty
    while pq:
        # Get the node with the current widest path to it
        negb, u = heappop(pq)
        cur_b = -negb

        # If we found a wider path to 'u' already, skip this entry
        if cur_b < bw[u]:
            continue

        # Explore neighbors of the current node 'u'
        for v in G.neighbors(u):
            # Get the bandwidth of the edge (u, v)
            edge_bw = _get_edge_bandwidth(G, u, v)
            # Skip if edge has no bandwidth or invalid
            if edge_bw <= 0:
                continue

            # Calculate the bottleneck bandwidth for the path through 'u' to 'v'
            bottleneck = min(cur_b, edge_bw)

            # If this path offers a wider bottleneck to 'v' than previously found
            if bottleneck > bw.get(v, 0.0):
                # Update 'v's widest path bandwidth and parent
                bw[v] = bottleneck
                parent[v] = u
                # Add 'v' to the priority queue with its new bottleneck bandwidth
                heappush(pq, (-bottleneck, v))
    # Return the widest path bandwidths to all nodes and their parent pointers
    return bw, parent

def reconstruct_path(parents: Dict[Any, Optional[Any]], target, source) -> List[Any]:
    path = []
    cur = target
    while cur is not None:
        path.append(cur)
        if cur == source:
            break
        cur = parents.get(cur)
    if not path or path[-1] != source:
        return []
    return list(reversed(path))

def create_curved_edge(p1, p2, curve_offset=30, num_points=20):
    """Create a curved edge using quadratic Bézier curve"""
    p1 = np.array(p1[:2])
    p2 = np.array(p2[:2])
    
    # Calculate perpendicular offset for curve
    mid = (p1 + p2) / 2
    direction = p2 - p1
    length = np.linalg.norm(direction)
    
    if length < 0.001:
        return np.column_stack([np.linspace(p1[0], p2[0], num_points),
                               np.linspace(p1[1], p2[1], num_points),
                               np.zeros(num_points)])
    
    # Perpendicular vector
    perp = np.array([-direction[1], direction[0]]) / length
    
    # Control point for quadratic Bézier
    control = mid + perp * curve_offset
    
    # Generate curve points
    t = np.linspace(0, 1, num_points)
    curve = np.outer((1-t)**2, p1) + np.outer(2*(1-t)*t, control) + np.outer(t**2, p2)
    
    # Add z-coordinate
    return np.column_stack([curve, np.zeros(num_points)])

def create_arrow_at_point(position, tangent, size=20):
    """Create arrow head at a specific point pointing in tangent direction"""
    pos = np.array(position[:2])
    tangent = np.array(tangent[:2])
    
    length = np.linalg.norm(tangent)
    if length < 0.001:
        return np.array([[0, 0, 0], [0, 0, 0], [0, 0, 0]])
    
    direction = tangent / length
    
    # Perpendicular vector
    perp = np.array([-direction[1], direction[0]])
    
    # Arrow head triangle
    tip = pos + direction * size * 1.5
    left = pos - direction * size * 0.5 + perp * size * 1.0
    right = pos - direction * size * 0.5 - perp * size * 1.0
    
    return np.array([[left[0], left[1], 0],
                     [tip[0], tip[1], 0],
                     [right[0], right[1], 0]])

# ---------------- Build SMALLER network ----------------
random.seed(42)
np.random.seed(42)

N_NODES = 12 # Reduced from 20
P_EDGE = 0.19  # Slightly increased for connectivity
N_SERVERS = 5  # Reduced from 5
FILE_MB = 80.0

G = nx.fast_gnp_random_graph(N_NODES, P_EDGE, seed=2)
if not nx.is_connected(G):
    # Instead of removing nodes, connect the components!
    components = list(nx.connected_components(G))
    for i in range(len(components) - 1):
        # Connect a node from current component to a node in the next component
        u = list(components[i])[0]
        v = list(components[i+1])[0]
        G.add_edge(u, v)

G = G.to_directed()

nodes = list(G.nodes())
client = nodes[0]
servers = nodes[1:1+N_SERVERS]

# Assign random latency and bandwidth to each directed edge
# Each edge direction is independent - (u,v) and (v,u) have different values
for u, v in G.edges():
    G.edges[u, v]['latency_ms'] = round(random.uniform(5.0, 250.0), 1)
    G.edges[u, v]['bandwidth_mbps'] = round(random.uniform(1.0, 300.0), 1)

# Ensure bidirectional edges for better asymmetry demonstration
# This creates reverse edges with DIFFERENT latency/bandwidth values
edges_to_add = []
for u, v in list(G.edges()):
    if not G.has_edge(v, u):
        edges_to_add.append((v, u))

for u, v in edges_to_add:
    G.add_edge(u, v)
    # Note: These get INDEPENDENT values - demonstrating asymmetric network
    G.edges[u, v]['latency_ms'] = round(random.uniform(5.0, 250.0), 1)
    G.edges[u, v]['bandwidth_mbps'] = round(random.uniform(1.0, 300.0), 1)

SERVER_CAPACITY_MBPS = FILE_MB * 8

# Better layout with more spacing
pos = nx.spring_layout(G, seed=42, k=3.5, iterations=150)
pos_array = np.array([pos[n] for n in nodes])
pos_array = (pos_array - pos_array.mean(axis=0)) * 600  # Larger scale
pos = {n: pos_array[i] for i, n in enumerate(nodes)}

# ---------------- Compute metrics ----------------
def estimate_times_to_servers(G, client, servers, x_MB):
    # Convert file size from MB to Mb (megabits) for bandwidth calculations
    x_Mb = x_MB * 8.0
    times = {}          # Dictionary to store estimated total time for each server
    all_widths = {}     # Dictionary to store widest path bandwidth for each server
    all_parents = {}    # Dictionary to store parent pointers for widest path reconstruction for each server

    # Iterate through each server to estimate the time to download the file
    for s in servers:
        # Calculate shortest path latencies from the current server 's' to all other nodes
        latencies = dijkstra_latencies_nx(G, s)
        # Calculate widest path bandwidths and parent pointers from the current server 's' to all other nodes
        widths, parents = widest_path_nx(G, s)

        # Get the latency from the server 's' to the client
        lat_ms = latencies.get(client, float('inf'))
        # If there's no path (latency is infinity), set time to infinity and continue
        if lat_ms == float('inf'):
            times[s] = float('inf')
            all_widths[s] = 0.0
            all_parents[s] = parents
            continue

        # Convert latency from milliseconds to seconds
        lat_s = lat_ms / 1000.0
        # Get the widest path bandwidth from the server 's' to the client
        path_bw = widths.get(client, 0.0)

        # If the path bandwidth is zero or less (no path or bottleneck), set time to infinity
        if path_bw <= 0.0:
            times[s] = float('inf')
        else:
            # Calculate total time: latency + (file size in Mb / bandwidth in Mbps)
            times[s] = lat_s + x_Mb / path_bw

        # Store the calculated widest path bandwidth and parent pointers for the current server
        all_widths[s] = path_bw
        all_parents[s] = parents

    return times, all_widths, all_parents

# Estimate download times, widest path bandwidths, and parent paths for all servers
all_times, all_widths, all_parents = estimate_times_to_servers(G, client, servers, FILE_MB)
# Create a copy of the servers list to define the evaluation order for visualization
eval_order = servers.copy()

# ---------------- Vispy Visualization ----------------
class NetworkVisualization:
    def __init__(self):
        self.canvas = scene.SceneCanvas(keys='interactive', size=(1600, 900), 
                                       title='Network Visualization - [SPACE] play/pause, [R] reset, [Q] quit',
                                       show=True, bgcolor='#0d1117')
        self.view = self.canvas.central_widget.add_view()
        self.view.camera = scene.PanZoomCamera(aspect=1)
        self.view.camera.set_range(x=(-500, 500), y=(-400, 400))
        
        self.current_frame = 0
        self.max_frames = len(eval_order) + 10
        self.playing = False
        self.evaluated_times = {s: np.nan for s in servers}
        self.current_best = None
        self.current_path = []
        
        # Store edge curves and arrow visuals for reuse
        self.edge_curves = {}
        self.edge_info = {}
        self.edge_arrow_visuals = {}  # Store individual arrow visuals
        
        self._create_visuals()
        
        self.timer = app.Timer(interval=1.5, connect=self.update_frame, start=False)
        self.canvas.events.key_press.connect(self.on_key_press)
        
        self.update_visuals()
        
    def _create_visuals(self):
        """Create all visual elements with curved edges"""
        # Determine bidirectional edges for offset calculation
        bidirectional = set()
        for u, v in G.edges():
            if G.has_edge(v, u):
                bidirectional.add((min(u, v), max(u, v)))
        
        # Create curved edges and individual arrow heads
        all_curves = []
        self.edge_to_curve_indices = {}
        
        current_idx = 0
        for u, v in G.edges():
            # Determine curve offset based on bidirectionality
            is_bidir = (min(u, v), max(u, v)) in bidirectional
            
            if is_bidir:
                # Use CONSTANT positive offset so they curve in opposite directions (forming an eye)
                # u->v curves 'left', v->u curves 'left' (which is 'right' of u->v)
                offset = 80
            else:
                offset = 30
            
            curve = create_curved_edge(pos[u], pos[v], curve_offset=offset, num_points=50)
            self.edge_curves[(u, v)] = curve
            
            # --- ARROW at 70% ---
            arrow_idx = int(len(curve) * 0.70)
            arrow_pos = curve[arrow_idx]
            
            # Calculate tangent at arrow position
            p_prev = curve[arrow_idx - 1]
            p_next = curve[arrow_idx + 1]
            tangent = p_next - p_prev
            
            # Create arrow mesh
            arrow_verts = create_arrow_at_point(arrow_pos, tangent, size=30)
            
            # Create arrow visual
            arrow_visual = visuals.Mesh(
                vertices=arrow_verts,
                faces=np.array([[0, 1, 2]]),
                color=(1.0, 1.0, 0.2, 1.0),  # Bright neon yellow
                parent=self.view.scene
            )
            
            # Add white outline for contrast
            arrow_outline = visuals.Line(
                pos=np.vstack([arrow_verts, arrow_verts[0:1]]),  # Close the triangle
                color=(1.0, 1.0, 1.0, 1.0),  # White outline
                width=2,
                parent=self.view.scene
            )
            
            self.edge_arrow_visuals[(u, v)] = arrow_visual
            
            # --- LABEL at 40% (Slightly before middle to balance arrow) ---
            lat = _get_edge_latency(G, u, v)
            bw = _get_edge_bandwidth(G, u, v)
            
            label_idx = int(len(curve) * 0.40)
            label_base_pos = curve[label_idx]
            
            # Calculate tangent at label position for perpendicular shift
            p_l_prev = curve[label_idx - 1]
            p_l_next = curve[label_idx + 1]
            tan_l = p_l_next - p_l_prev
            tan_l_norm = tan_l / (np.linalg.norm(tan_l) + 1e-6)
            
            # Perpendicular vector (in XY plane)
            perp_vec = np.array([-tan_l_norm[1], tan_l_norm[0], 0])
            
            # Push label OUTWARD
            # Since offset is always positive, we always push in +Perp direction
            shift_dist = 40
            label_pos = label_base_pos + perp_vec * shift_dist

            self.edge_info[(u, v)] = {
                'latency': lat,
                'bandwidth': bw,
                'label_pos': label_pos,
                'curve': curve
            }
            
            # Add to all curves
            all_curves.append(curve)
            # Add NaN separator to prevent connecting separate edges
            all_curves.append(np.array([[np.nan, np.nan, np.nan]]))
            
            self.edge_to_curve_indices[(u, v)] = (current_idx, current_idx + len(curve))
            current_idx += len(curve) + 1
        
        # Concatenate all curves
        if all_curves:
            all_curve_points = np.vstack(all_curves)
        else:
            all_curve_points = np.zeros((0, 3), dtype=np.float32)
        
        # Base network edges - MUCH MORE VISIBLE
        self.base_edges = visuals.Line(pos=all_curve_points, 
                                       color=(0.6, 0.7, 0.9, 0.5), 
                                       width=4, 
                                       parent=self.view.scene)
        
        # Highlighted path edges - VERY BOLD
        self.path_edges = visuals.Line(pos=np.array([[0,0,0], [1,1,0]], dtype=np.float32),
                                       color='#ff1a66', width=12, 
                                       parent=self.view.scene)
        
        # Node markers - MUCH LARGER
        node_positions = np.array([list(pos[n]) + [0] for n in nodes], dtype=np.float32)
        
        # Use dictionaries indexed by node ID to avoid IndexError
        node_colors_dict = {n: [0.6, 0.7, 0.9, 1.0] for n in nodes}
        node_sizes_dict = {n: 35 for n in nodes}  # Bigger intermediate nodes
        
        node_colors_dict[client] = [0.2, 1.0, 0.3, 1.0]
        node_sizes_dict[client] = 100  # HUGE client node
        
        for s in servers:
            node_colors_dict[s] = [0.3, 0.8, 1.0, 1.0]
            node_sizes_dict[s] = 90  # HUGE server nodes
        
        # Convert to arrays in the same order as node_positions
        node_colors = np.array([node_colors_dict[n] for n in nodes], dtype=np.float32)
        node_sizes = np.array([node_sizes_dict[n] for n in nodes], dtype=np.float32)
        
        self.nodes = visuals.Markers(pos=node_positions, size=node_sizes,
                                     face_color=node_colors, edge_color='white',
                                     edge_width=3, parent=self.view.scene)
        
        # Text labels - LARGER AND CLEARER
        self.text_items = []
        
        t = visuals.Text(f'CLIENT\n#{client}', pos=list(pos[client]) + [0], 
                        color='#00ff66', font_size=16, anchor_x='center', anchor_y='top',
                        parent=self.view.scene, bold=True)
        self.text_items.append(t)
        
        for s in servers:
            t = visuals.Text(f'SERVER\n#{s}', pos=list(pos[s]) + [0], 
                           color='#66ddff', font_size=14, anchor_x='center', anchor_y='bottom',
                           parent=self.view.scene, bold=True)
            self.text_items.append(t)
        
        # Intermediate node labels - show node numbers
        for n in nodes:
            if n != client and n not in servers:
                t = visuals.Text(f'{n}', pos=list(pos[n]) + [0], 
                               color='white', font_size=10, anchor_x='center', anchor_y='center',
                               parent=self.view.scene, bold=True)
                self.text_items.append(t)
        
        # Edge labels - ENHANCED VISIBILITY
        self.edge_labels = []
        for (u, v), info in self.edge_info.items():
            label_pos = info['label_pos']
            lat = info['latency']
            bw = info['bandwidth']
            
            # Create highly visible edge labels
            t = visuals.Text(f'{int(lat)}ms\n{int(bw)}Mb/s', 
                           pos=label_pos, 
                           color='#ffeb3b',  # Bright yellow
                           font_size=15,  # Larger font
                           anchor_x='right', 
                           anchor_y='bottom',
                           parent=self.view.scene,
                           bold=True)
            self.edge_labels.append(t)
        
        # Info panels - LARGER TEXT
        self.path_info_text = visuals.Text('', 
                                          pos=(30, 860), color='#ffcc44', font_size=13,
                                          anchor_x='left', anchor_y='bottom',
                                          parent=self.view, bold=True)
        
        self.server_table_text = visuals.Text('', 
                                             pos=(1570, 860), color='#ffee88', font_size=12,
                                             anchor_x='right', anchor_y='bottom',
                                             parent=self.view, bold=True)
        
        algo_lines = [
            "🧮 ALGORITHMS",
            "═════════════",
            "1. Dijkstra",
            "   Min Latency",
            "",
            "2. Widest Path",
            "   Max Bandwidth",
            "",
            "3. T = L + S/B",
        ]
        self.algo_info_text = visuals.Text('\n'.join(algo_lines),
                                          pos=(1570, 500), color='#ffaa66', font_size=11,
                                          anchor_x='right', anchor_y='top',
                                          parent=self.view, bold=True)
        
        self.stats_text = visuals.Text('', 
                                      pos=(30, 500), color='#99ffaa', font_size=12,
                                      anchor_x='left', anchor_y='top',
                                      parent=self.view, bold=True)
        
        self.status_text = visuals.Text('Press SPACE to start', 
                                       pos=(30, 40), color='white', font_size=16,
                                       anchor_x='left', anchor_y='top',
                                       parent=self.view, bold=True)
        
        info_lines = [
            f"📊 NETWORK INFO",
            f"═══════════════",
            f"Directed Graph ➡️",
            f"Nodes: {len(nodes)}",
            f"Edges: {len(G.edges())}",
            f"File: {FILE_MB} MB",
            f"Servers: {N_SERVERS}",
            "",
            "⌨️  CONTROLS",
            "═══════════",
            "SPACE - Play",
            "R - Reset",
            "Q - Quit",
        ]
        self.info_text = visuals.Text('\n'.join(info_lines),
                                     pos=(1570, 40), color='#88ddff', font_size=12,
                                     anchor_x='right', anchor_y='bottom',
                                     parent=self.view, bold=True)
        
    def update_visuals(self):
        """Update visuals based on current frame"""
        # Reset all edge arrows to bright neon yellow
        for (u, v), arrow_visual in self.edge_arrow_visuals.items():
            arrow_visual.set_data(color=(1.0, 1.0, 0.2, 1.0))  # Bright neon yellow
        
        if self.current_frame < len(eval_order):
            s = eval_order[self.current_frame]
            
            widths_now, parents_now = widest_path_nx(G, s)
            path = reconstruct_path(parents_now, client, s)
            self.current_path = path
            
            if path and len(path) >= 2:
                # Build curved path
                path_curves = []
                total_latency = 0
                min_bandwidth = inf
                
                # Highlight edges and arrows in the active path
                for i in range(len(path) - 1):
                    u, v = path[i], path[i+1]
                    curve = self.edge_curves[(u, v)]
                    path_curves.append(curve)
                    
                    # Highlight this arrow in BRIGHT NEON LIME for maximum visibility
                    if (u, v) in self.edge_arrow_visuals:
                        self.edge_arrow_visuals[(u, v)].set_data(
                            color=(0.5, 1.0, 0.0, 1.0)  # Bright neon lime green - stands out!
                        )
                    
                    lat = _get_edge_latency(G, u, v)
                    bw = _get_edge_bandwidth(G, u, v)
                    total_latency += lat
                    min_bandwidth = min(min_bandwidth, bw)
                
                path_pos = np.vstack(path_curves)
                self.path_edges.set_data(pos=path_pos, color='#ff1a66', width=12)
                
                # path_info_lines = [
                #     f"🔗 ACTIVE PATH: Server #{s} → Client",
                #     f"═════════════════════════════",
                #     f"Hops: {len(path)-1}",
                #     f"Total Latency: {total_latency:.1f} ms",
                #     f"Bottleneck BW: {min_bandwidth:.1f} Mb/s",
                #     f"Download Time: {total_latency/1000 + (FILE_MB*8)/min_bandwidth:.2f} sec"
                # ]
                # self.path_info_text.text = '\n'.join(path_info_lines)
            else:
                self.path_edges.set_data(pos=np.array([[0,0,0], [0,0,0]], dtype=np.float32))
                self.path_info_text.text = ''
            
            # Compute time
            x_Mb = FILE_MB * 8.0
            latencies = dijkstra_latencies_nx(G, s)
            lat_ms = latencies.get(client, inf)
            if lat_ms == inf:
                tval = np.nan
            else:
                lat_s = lat_ms / 1000.0
                path_bw = widths_now.get(client, 0.0)
                tval = lat_s + x_Mb / path_bw if path_bw > 0 else np.nan
            self.evaluated_times[s] = tval
            
            good = {sv: t for sv, t in self.evaluated_times.items() if not np.isnan(t)}
            if good:
                self.current_best = min(good, key=good.get)
            
            # Update tables
            table_lines = ["📋 SERVER RESULTS", "═════════════════"]
            for srv in servers:
                if not np.isnan(self.evaluated_times[srv]):
                    time_val = self.evaluated_times[srv]
                    status = " ⭐ BEST" if srv == self.current_best else " ✓"
                    table_lines.append(f"Server {srv}: {time_val:6.2f}s{status}")
                elif srv == s:
                    table_lines.append(f"Server {srv}: ... ⏳")
                else:
                    table_lines.append(f"Server {srv}: -------")
            self.server_table_text.text = '\n'.join(table_lines)
            
            evaluated_count = sum(1 for t in self.evaluated_times.values() if not np.isnan(t))
            avg_time = np.nanmean([t for t in self.evaluated_times.values() if not np.isnan(t)]) if evaluated_count > 0 else 0
            
            stats_lines = [
                "📈 STATISTICS",
                "═════════════",
                f"Evaluated: {evaluated_count}/{N_SERVERS}",
                f"Average: {avg_time:.2f}s" if evaluated_count > 0 else "Average: N/A",
                f"Best: {self.evaluated_times[self.current_best]:.2f}s" if self.current_best else "Best: N/A",
            ]
            self.stats_text.text = '\n'.join(stats_lines)
            
            status_lines = [
                f"⏱️  Frame {self.current_frame + 1} / {len(eval_order)}",
                f"═════════════════════",
                f"Testing: Server #{s}",
                f"Time: {tval:.2f}s" if not np.isnan(tval) else "Time: ∞",
                "",
                f"🏆 Current Best: Server #{self.current_best}" if self.current_best else "🏆 No winner yet"
            ]
            self.status_text.text = '\n'.join(status_lines)
        else:
            # Complete - Show optimal path in GREEN
            # self.status_text.text = f"✅ COMPLETE!\n═══════════\n🏆 Winner: Server #{self.current_best}\n⏱️  Time: {self.evaluated_times[self.current_best]:.2f}s"
            
            # Reconstruct and display the optimal path in GREEN
            if self.current_best is not None:
                widths_best, parents_best = widest_path_nx(G, self.current_best)
                optimal_path = reconstruct_path(parents_best, client, self.current_best)
                
                if optimal_path and len(optimal_path) >= 2:
                    # Build curved path for the optimal route
                    path_curves = []
                    total_latency = 0
                    min_bandwidth = inf
                    
                    # Highlight edges and arrows in the optimal path with GREEN
                    for i in range(len(optimal_path) - 1):
                        u, v = optimal_path[i], optimal_path[i+1]
                        curve = self.edge_curves[(u, v)]
                        path_curves.append(curve)
                        
                        # Highlight this arrow in BRIGHT GREEN for the optimal path
                        if (u, v) in self.edge_arrow_visuals:
                            self.edge_arrow_visuals[(u, v)].set_data(
                                color=(0.0, 1.0, 0.0, 1.0)  # Bright green
                            )
                        
                        lat = _get_edge_latency(G, u, v)
                        bw = _get_edge_bandwidth(G, u, v)
                        total_latency += lat
                        min_bandwidth = min(min_bandwidth, bw)
                    
                    path_pos = np.vstack(path_curves)
                    # Display optimal path in BRIGHT GREEN
                    self.path_edges.set_data(pos=path_pos, color='#00ff66', width=12)
                    
                    # Update path info with GREEN theme
                    # path_info_lines = [
                    #     f"🏆 OPTIMAL PATH: Server #{self.current_best} → Client",
                    #     f"═════════════════════════════",
                    #     f"Hops: {len(optimal_path)-1}",
                    #     f"Total Latency: {total_latency:.1f} ms",
                    #     f"Bottleneck BW: {min_bandwidth:.1f} Mb/s",
                    #     f"Download Time: {total_latency/1000 + (FILE_MB*8)/min_bandwidth:.2f} sec"
                    # ]
                    # self.path_info_text.text = '\n'.join(path_info_lines)
                else:
                    self.path_edges.set_data(pos=np.array([[0,0,0], [0,0,0]], dtype=np.float32))
            else:
                self.path_edges.set_data(pos=np.array([[0,0,0], [0,0,0]], dtype=np.float32))
            
            good = {sv: t for sv, t in self.evaluated_times.items() if not np.isnan(t)}
            ranked = sorted(good.items(), key=lambda x: x[1])
            
            medals = {0: "🥇", 1: "🥈", 2: "🥉"}
            table_lines = ["🏅 FINAL RANKING", "═════════════════"]
            for rank, (srv, time_val) in enumerate(ranked):
                medal = medals.get(rank, f"{rank+1}.")
                table_lines.append(f"{medal} Server {srv}: {time_val:.2f}s")
            self.server_table_text.text = '\n'.join(table_lines)
        
        self.canvas.update()
    
    def update_frame(self, event):
        if self.current_frame < self.max_frames:
            self.update_visuals()
            self.current_frame += 1
        else:
            self.timer.stop()
            self.playing = False
    
    def on_key_press(self, event):
        if event.key == ' ':
            if self.playing:
                self.timer.stop()
                self.playing = False
            else:
                if self.current_frame >= self.max_frames:
                    self.current_frame = 0
                    self.evaluated_times = {s: np.nan for s in servers}
                    self.current_best = None
                self.timer.start()
                self.playing = True
        elif event.key == 'R':
            self.timer.stop()
            self.playing = False
            self.current_frame = 0
            self.evaluated_times = {s: np.nan for s in servers}
            self.current_best = None
            self.update_visuals()
        elif event.key == 'Q':
            self.canvas.close()
            app.quit()
    
    def run(self):
        print("\n" + "="*60)
        print("🚀 Enhanced Network Visualization")
        print("="*60)
        print("\nFeatures:")
        print("  ✨ Smaller, clearer graph (12 nodes)")
        print("  ➡️  Highly visible curved edges")
        print("  🔢 Clear edge labels with values")
        print("  💡 Bold colors and larger elements")
        print("\nControls:")
        print("  SPACE - Play/Pause")
        print("  R     - Reset")
        print("  Q     - Quit")
        print("="*60 + "\n")
        app.run()

if __name__ == '__main__':
    viz = NetworkVisualization()
    viz.run()