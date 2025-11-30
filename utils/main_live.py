# Real-time network visualization using Vispy (high-performance OpenGL rendering)
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

# ---------------- Helpers (same as main.py) ----------------
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
    weight_attr = 'latency_ms' if any('latency_ms' in d for _,_,d in G.edges(data=True)) else 'latency'
    if not any(weight_attr in d for _,_,d in G.edges(data=True)):
        return {n: (0.0 if n in G else inf) for n in G.nodes()}
    return nx.single_source_dijkstra_path_length(G, source, weight=weight_attr)

def widest_path_nx(G: nx.Graph, source) -> Tuple[Dict[Any, float], Dict[Any, Optional[Any]]]:
    bw = {n: 0.0 for n in G.nodes()}
    parent: Dict[Any, Optional[Any]] = {n: None for n in G.nodes()}
    bw[source] = inf
    pq = [(-bw[source], source)]
    while pq:
        negb, u = heappop(pq)
        cur_b = -negb
        if cur_b < bw[u]:
            continue
        for v in G.neighbors(u):
            edge_bw = _get_edge_bandwidth(G, u, v)
            if edge_bw <= 0:
                continue
            bottleneck = min(cur_b, edge_bw)
            if bottleneck > bw.get(v, 0.0):
                bw[v] = bottleneck
                parent[v] = u
                heappush(pq, (-bottleneck, v))
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

# ---------------- Build network (same parameters) ----------------
random.seed(42)
np.random.seed(42)

N_NODES = 20
P_EDGE = 0.15  # Slightly higher probability to ensure connectivity with fewer nodes
N_SERVERS = 5
FILE_MB = 80.0

G = nx.fast_gnp_random_graph(N_NODES, P_EDGE, seed=2)
if not nx.is_connected(G):
    comp = max(nx.connected_components(G), key=len)
    G = G.subgraph(comp).copy()

# Convert to directed graph for asymmetric paths
G = G.to_directed()

nodes = list(G.nodes())
client = nodes[0]
servers = nodes[1:1+N_SERVERS]

# Assign random weights to directed edges (u->v and v->u are independent)
for u, v in G.edges():
    G.edges[u, v]['latency_ms'] = round(random.uniform(5.0, 250.0), 1)
    G.edges[u, v]['bandwidth_mbps'] = round(random.uniform(1.0, 300.0), 1)

# All servers have the same upload capacity (sufficient to serve the file)
SERVER_CAPACITY_MBPS = FILE_MB * 8  # Capacity equals file bitrate

# Layout nodes - spread them out more
pos = nx.spring_layout(G, seed=42, k=2.5, iterations=100)  # Increased k for more spread
# Scale positions to fit nicely in view
pos_array = np.array([pos[n] for n in nodes])
pos_array = (pos_array - pos_array.mean(axis=0)) * 800  # Scale to 800 units
pos = {n: pos_array[i] for i, n in enumerate(nodes)}

# ---------------- Compute metrics ----------------
def estimate_times_to_servers(G, client, servers, x_MB):
    x_Mb = x_MB * 8.0
    times = {}
    all_widths = {}
    all_parents = {}
    
    for s in servers:
        latencies = dijkstra_latencies_nx(G, s)
        widths, parents = widest_path_nx(G, s)
        
        lat_ms = latencies.get(client, inf)
        if lat_ms == inf:
            times[s] = inf
            all_widths[s] = 0.0
            all_parents[s] = parents
            continue
        
        lat_s = lat_ms / 1000.0
        path_bw = widths.get(client, 0.0)  # Only edge bandwidth matters
        if path_bw <= 0.0:
            times[s] = inf
        else:
            times[s] = lat_s + x_Mb / path_bw  # Download time based on path bandwidth only
        
        all_widths[s] = path_bw
        all_parents[s] = parents
    
    return times, all_widths, all_parents

all_times, all_widths, all_parents = estimate_times_to_servers(G, client, servers, FILE_MB)
eval_order = servers.copy()

# ---------------- Vispy Visualization ----------------
class NetworkVisualization:
    def __init__(self):
        # Create canvas
        self.canvas = scene.SceneCanvas(keys='interactive', size=(1400, 800), 
                                       title='Network Visualization - [SPACE] play/pause, [R] reset, [Q] quit',
                                       show=True, bgcolor='#1a1a2e')
        self.view = self.canvas.central_widget.add_view()
        self.view.camera = scene.PanZoomCamera(aspect=1)
        self.view.camera.set_range(x=(-500, 500), y=(-500, 500))
        
        # Animation state
        self.current_frame = 0
        self.max_frames = len(eval_order) + 10
        self.playing = False
        self.evaluated_times = {s: np.nan for s in servers}
        self.current_best = None
        self.current_path = []
        
        # Create visual elements
        self._create_visuals()
        
        # Set up timer for animation (2.5 seconds per frame for better viewing)
        self.timer = app.Timer(interval=1.5, connect=self.update_frame, start=False)
        
        # Key bindings
        self.canvas.events.key_press.connect(self.on_key_press)
        
        # Initial draw
        self.update_visuals()
        
    def _create_visuals(self):
        """Create all visual elements"""
        # Edge lines (base network) - make them very faint
        edge_pos = []
        for u, v in G.edges():
            edge_pos.append(list(pos[u]) + [0])
            edge_pos.append(list(pos[v]) + [0])
        edge_pos = np.array(edge_pos, dtype=np.float32)
        
        # Edge lines (base network) - make them thicker and more visible
        edge_pos = []
        for u, v in G.edges():
            edge_pos.append(list(pos[u]) + [0])
            edge_pos.append(list(pos[v]) + [0])
        edge_pos = np.array(edge_pos, dtype=np.float32)
        
        self.base_edges = visuals.Line(pos=edge_pos, color=(0.5, 0.5, 0.6, 0.3), 
                                       width=3, connect='segments', parent=self.view.scene)
        
        # Highlighted path edges (will be updated)
        self.path_edges = visuals.Line(pos=np.array([[0,0,0], [1,1,0]], dtype=np.float32),
                                       color='#ff3333', width=12, connect='segments', 
                                       parent=self.view.scene)
        
        # Node markers - make intermediate nodes bigger
        node_positions = np.array([list(pos[n]) + [0] for n in nodes], dtype=np.float32)
        node_colors = np.array([[0.5, 0.5, 0.6, 0.8] for _ in nodes], dtype=np.float32)
        node_sizes = np.array([15 for _ in nodes], dtype=np.float32)  # Bigger intermediate nodes
        
        # Highlight client - large green circle
        node_colors[client] = [0.2, 1.0, 0.2, 1.0]  # Bright green for client
        node_sizes[client] = 70
        
        # Highlight servers - large blue circles
        for s in servers:
            node_colors[s] = [0.3, 0.7, 1.0, 1.0]  # Bright blue for servers
            node_sizes[s] = 70
        
        self.nodes = visuals.Markers(pos=node_positions, size=node_sizes,
                                     face_color=node_colors, edge_color='white',
                                     edge_width=2, parent=self.view.scene)
        
        # Text labels for nodes (showing only key nodes to reduce clutter)
        self.text_items = []
        
        # Label client with icon
        t = visuals.Text(f'👤 CLIENT\nNode {client}', pos=list(pos[client]) + [0], 
                        color='lime', font_size=12, anchor_x='center', anchor_y='top',
                        parent=self.view.scene, bold=True)
        self.text_items.append(t)
        
        # Label servers with server icons
        for s in servers:
            t = visuals.Text(f'🖥️ SERVER {s}\n✓ Ready', pos=list(pos[s]) + [0], 
                           color='cyan', font_size=10, anchor_x='center', anchor_y='bottom',
                           parent=self.view.scene, bold=True)
            self.text_items.append(t)
        
        # Permanent edge labels for ALL edges
        self.edge_labels = []
        for u, v in G.edges():
            lat = _get_edge_latency(G, u, v)
            bw = _get_edge_bandwidth(G, u, v)
            
            # Position labels
            mid_x = (pos[u][0] + pos[v][0]) / 2
            mid_y = (pos[u][1] + pos[v][1]) / 2
            
            # Add small offset based on edge direction to separate u->v and v->u labels
            # Simple heuristic: shift slightly perpendicular to the edge
            dx = pos[v][0] - pos[u][0]
            dy = pos[v][1] - pos[u][1]
            length = np.sqrt(dx*dx + dy*dy)
            if length > 0:
                off_x = -dy / length * 15  # 15 units offset
                off_y = dx / length * 15
            else:
                off_x, off_y = 0, 0
            
            t = visuals.Text(f'{int(lat)}ms\n{int(bw)}Mbps', pos=[mid_x + off_x, mid_y + off_y, 0], 
                           color='white', font_size=8, anchor_x='center', anchor_y='center',
                           parent=self.view.scene, bold=True)
            self.edge_labels.append(t)
        
        # Path info box (bottom-left)
        self.path_info_text = visuals.Text('', 
                                          pos=(20, 720), color='orange', font_size=10,
                                          anchor_x='left', anchor_y='bottom',
                                          parent=self.view, bold=True)
        
        # Server evaluation table (bottom-right)
        self.server_table_text = visuals.Text('', 
                                             pos=(1380, 720), color='yellow', font_size=9,
                                             anchor_x='right', anchor_y='bottom',
                                             parent=self.view, bold=False)
        
        # Algorithm info panel (middle-right)
        algo_lines = [
            "🧮 ALGORITHMS USED",
            "━━━━━━━━━━━━━━━━━━",
            "1. Dijkstra's Algorithm",
            "   → Finds minimum latency path",
            "   → Complexity: O((V+E)logV)",
            "",
            "2. Widest Path Algorithm",
            "   → Finds max bottleneck BW",
            "   → Uses max-heap approach",
            "",
            "3. Time Calculation",
            "   → T = latency + size/BW",
        ]
        self.algo_info_text = visuals.Text('\n'.join(algo_lines),
                                          pos=(1380, 400), color='#FFA500', font_size=9,
                                          anchor_x='right', anchor_y='top',
                                          parent=self.view)
        
        # Statistics panel (middle-left)
        self.stats_text = visuals.Text('', 
                                      pos=(20, 400), color='#90EE90', font_size=9,
                                      anchor_x='left', anchor_y='top',
                                      parent=self.view, bold=False)
        
        # Status text (top-left) - larger and more prominent
        self.status_text = visuals.Text('Press SPACE to start animation', 
                                       pos=(20, 30), color='white', font_size=13,
                                       anchor_x='left', anchor_y='top',
                                       parent=self.view, bold=True)
        
        # Info panel text (top-right)
        info_lines = [
            f"📊 NETWORK INFO",
            f"━━━━━━━━━━━━━━━━",
            f"Type: Directed Graph ➡️",
            f"Nodes: {len(nodes)}",
            f"Edges: {len(G.edges())} (Asymmetric)",
            f"Density: {len(G.edges())/(len(nodes)*(len(nodes)-1)/2)*100:.1f}%",
            f"File: {FILE_MB} MB = {FILE_MB*8:.0f} Mb",
            f"Servers: {N_SERVERS}",
            f"Server cap: {SERVER_CAPACITY_MBPS:.0f} Mbps (all)",
            "",
            "⌨️  CONTROLS",
            "━━━━━━━━━━━━━━",
            "  SPACE - Play/Pause",
            "  R - Reset",
            "  Q - Quit",
            "  Mouse - Pan/Zoom"
        ]
        self.info_text = visuals.Text('\n'.join(info_lines),
                                     pos=(1380, 30), color='cyan', font_size=10,
                                     anchor_x='right', anchor_y='top',
                                     parent=self.view)
        
    def update_visuals(self):
        """Update visuals based on current frame"""
        # Note: Edge labels are now permanent, so we don't clear them
        
        if self.current_frame < len(eval_order):
            # Evaluate current server
            s = eval_order[self.current_frame]
            
            # Compute path from server to client
            widths_now, parents_now = widest_path_nx(G, s)
            path = reconstruct_path(parents_now, client, s)
            self.current_path = path
            
            # Update path visualization
            if path and len(path) >= 2:
                path_pos = []
                total_latency = 0
                min_bandwidth = inf
                
                for i in range(len(path) - 1):
                    u, v = path[i], path[i+1]
                    path_pos.append(list(pos[u]) + [0])
                    path_pos.append(list(pos[v]) + [0])
                    # Get edge info
                    lat = _get_edge_latency(G, u, v)
                    bw = _get_edge_bandwidth(G, u, v)
                    total_latency += lat
                    min_bandwidth = min(min_bandwidth, bw)
                    
                    # Highlight edge label on path
                    # We could make the text bigger/brighter for active path, 
                    # but for now the red line is enough indication
                    pass
                
                path_pos = np.array(path_pos, dtype=np.float32)
                self.path_edges.set_data(pos=path_pos, color='red', width=8)
                
                # Update path info box
                path_info_lines = [
                    f"🔗 PATH: Server {s} → Client",
                    f"━━━━━━━━━━━━━━━━━━━━━━",
                    f"Hops: {len(path)}",
                    f"Total latency: {total_latency:.1f} ms",
                    f"Bottleneck BW: {min_bandwidth:.1f} Mbps",
                    f"Download time: {total_latency/1000 + (FILE_MB*8)/min_bandwidth:.2f}s"
                ]
                self.path_info_text.text = '\n'.join(path_info_lines)
            else:
                # No path, hide the line
                self.path_edges.set_data(pos=np.array([[0,0,0], [0,0,0]], dtype=np.float32))
                self.path_info_text.text = ''
            
            # Compute time for this server
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
            
            # Find current best
            good = {sv: t for sv, t in self.evaluated_times.items() if not np.isnan(t)}
            if good:
                self.current_best = min(good, key=good.get)
            
            # Update server evaluation table
            table_lines = ["📋 SERVER EVALUATION", "━━━━━━━━━━━━━━━━━━━"]
            for srv in servers:
                if not np.isnan(self.evaluated_times[srv]):
                    time_val = self.evaluated_times[srv]
                    status = "⭐ BEST" if srv == self.current_best else "✓ Ready"
                    table_lines.append(f"S{srv}: {time_val:6.2f}s {status}")
                elif srv == s:
                    table_lines.append(f"S{srv}: ... evaluating")
                else:
                    table_lines.append(f"S{srv}: -------")
            self.server_table_text.text = '\n'.join(table_lines)
            
            # Update statistics panel
            evaluated_count = sum(1 for t in self.evaluated_times.values() if not np.isnan(t))
            avg_time = np.nanmean([t for t in self.evaluated_times.values() if not np.isnan(t)]) if evaluated_count > 0 else 0
            stats_lines = [
                "📈 STATISTICS",
                "━━━━━━━━━━━━━━",
                f"Evaluated: {evaluated_count}/{N_SERVERS}",
                f"Avg time: {avg_time:.2f}s" if evaluated_count > 0 else "Avg time: N/A",
                f"Best time: {self.evaluated_times[self.current_best]:.2f}s" if self.current_best else "Best: N/A",
                f"Worst time: {max([t for t in self.evaluated_times.values() if not np.isnan(t)]):.2f}s" if good else "Worst: N/A",
                "",
                "🎯 GOAL",
                "━━━━━━━━━━━━━━",
                "Find server with",
                "minimum download",
                "time considering:",
                "• Path latency ⏱️",
                "• Path bandwidth 📡",
            ]
            self.stats_text.text = '\n'.join(stats_lines)
            
            # Update status text
            status_lines = [
                f"⏱️  Frame {self.current_frame + 1}/{len(eval_order)}",
                f"━━━━━━━━━━━━━━━━━━",
                f"Evaluating: Server {s}",
                f"Time: {tval:.2f}s" if not np.isnan(tval) else "Time: ∞",
                "",
                f"🏆 Current best: Server {self.current_best}" if self.current_best else "🏆 No best yet",
                f"   Best time: {self.evaluated_times[self.current_best]:.2f}s" if self.current_best else ""
            ]
            self.status_text.text = '\n'.join(status_lines)
        else:
            # Animation complete - show final summary
            final_lines = [
                "✅ EVALUATION COMPLETE!",
                "━━━━━━━━━━━━━━━━━━━━",
                f"🏆 Winner: Server {self.current_best}",
                f"⏱️  Best time: {self.evaluated_times[self.current_best]:.2f}s",
                "",
                "All servers evaluated.",
                "Winner minimizes:",
                "  latency + size/bandwidth"
            ]
            self.status_text.text = '\n'.join(final_lines)
            self.path_edges.set_data(pos=np.array([[0,0,0], [0,0,0]], dtype=np.float32))
            self.path_info_text.text = ''
            
            # Create full ranking
            good = {sv: t for sv, t in self.evaluated_times.items() if not np.isnan(t)}
            ranked_servers = sorted(good.items(), key=lambda x: x[1])  # Sort by time (ascending)
            
            # Update server table with full ranking
            medals = {0: "🥇", 1: "🥈", 2: "🥉"}
            table_lines = ["🏅 FINAL RANKING", "━━━━━━━━━━━━━━━━━━━"]
            for rank, (srv, time_val) in enumerate(ranked_servers):
                medal = medals.get(rank, f"{rank+1}.")
                table_lines.append(f"{medal} S{srv}: {time_val:6.2f}s")
            
            # Add performance metrics
            if len(ranked_servers) > 1:
                best_time = ranked_servers[0][1]
                worst_time = ranked_servers[-1][1]
                table_lines.append("")
                table_lines.append("📊 Performance Gap:")
                table_lines.append(f"  {worst_time - best_time:.2f}s spread")
                table_lines.append(f"  {(worst_time/best_time - 1)*100:.1f}% slower")
            
            self.server_table_text.text = '\n'.join(table_lines)
            
            # Update statistics with ranking details
            evaluated_count = len(good)
            avg_time = np.mean([t for t in good.values()])
            median_time = np.median([t for t in good.values()])
            
            stats_lines = [
                "📈 FINAL STATISTICS",
                "━━━━━━━━━━━━━━━━",
                f"Total evaluated: {evaluated_count}",
                f"Average time: {avg_time:.2f}s",
                f"Median time: {median_time:.2f}s",
                f"Best time: {self.evaluated_times[self.current_best]:.2f}s",
                f"Worst time: {max(good.values()):.2f}s",
                f"Range: {max(good.values()) - min(good.values()):.2f}s",
                "",
                "💡 WHY THIS SERVER?",
                "━━━━━━━━━━━━━━━━",
                "Best combination of:",
                "• Low latency path",
                "• High bandwidth path",
                "",
                f"Winner is {(max(good.values())/min(good.values()) - 1)*100:.1f}%",
                "faster than worst!"
            ]
            self.stats_text.text = '\n'.join(stats_lines)
            
            # Show winner's path details in path info
            winner_path_lines = [
                f"🏆 WINNER: Server {self.current_best}",
                "━━━━━━━━━━━━━━━━━━━━━━",
                f"Rank: #1 out of {evaluated_count}",
                f"Time: {self.evaluated_times[self.current_best]:.2f}s",
                "",
                "Outperformed all other",
                "servers in download time!"
            ]
            self.path_info_text.text = '\n'.join(winner_path_lines)

        
        self.canvas.update()
    
    def update_frame(self, event):
        """Timer callback for animation"""
        if self.current_frame < self.max_frames:
            self.update_visuals()
            self.current_frame += 1
        else:
            self.timer.stop()
            self.playing = False
    
    def on_key_press(self, event):
        """Handle keyboard input"""
        if event.key == ' ':  # Space bar
            if self.playing:
                self.timer.stop()
                self.playing = False
                print("⏸  Paused")
            else:
                if self.current_frame >= self.max_frames:
                    # Reset if at end
                    self.current_frame = 0
                    self.evaluated_times = {s: np.nan for s in servers}
                    self.current_best = None
                self.timer.start()
                self.playing = True
                print("▶  Playing")
        elif event.key == 'R':  # Reset
            self.timer.stop()
            self.playing = False
            self.current_frame = 0
            self.evaluated_times = {s: np.nan for s in servers}
            self.current_best = None
            self.update_visuals()
            print("🔄 Reset")
        elif event.key == 'Q':  # Quit
            self.canvas.close()
            app.quit()
    
    def run(self):
        """Start the application"""
        print("\n" + "="*60)
        print("🚀 Network Visualization Started!")
        print("="*60)
        print("\nControls:")
        print("  SPACE - Play/Pause animation")
        print("  R     - Reset to beginning")
        print("  Q     - Quit")
        print("  Mouse - Pan and zoom the view")
        print("\n" + "="*60 + "\n")
        app.run()

# ---------------- Main ----------------
if __name__ == '__main__':
    viz = NetworkVisualization()
    viz.run()
