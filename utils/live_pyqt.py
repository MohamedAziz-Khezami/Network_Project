# Real-time network visualization using PyQtGraph
# Run directly from terminal: uv run live_pyqt.py
# Press SPACE to start/pause, R to reset, ESC to quit

import sys
import networkx as nx
from heapq import heappush, heappop
from math import inf
from typing import Any, Dict, List, Optional, Tuple
import random
import numpy as np
import pyqtgraph as pg
from pyqtgraph.Qt import QtCore, QtGui, QtWidgets

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
                               np.linspace(p1[1], p2[1], num_points)])
    
    # Perpendicular vector
    perp = np.array([-direction[1], direction[0]]) / length
    
    # Control point for quadratic Bézier
    control = mid + perp * curve_offset
    
    # Generate curve points
    t = np.linspace(0, 1, num_points)
    curve = np.outer((1-t)**2, p1) + np.outer(2*(1-t)*t, control) + np.outer(t**2, p2)
    
    return curve

def create_arrow_polygon(p1, p2, size=28):
    """Create arrow head polygon pointing from p1 to p2"""
    p1 = np.array(p1[:2])
    p2 = np.array(p2[:2])
    
    direction = p2 - p1
    length = np.linalg.norm(direction)
    
    if length < 0.001:
        return np.array([[0, 0], [0, 0], [0, 0]])
    
    direction = direction / length
    
    # Arrow head position (70% along the edge)
    arrow_pos = p1 + direction * length * 0.70
    
    # Perpendicular vector
    perp = np.array([-direction[1], direction[0]])
    
    # Arrow head triangle
    tip = arrow_pos + direction * size * 1.2
    left = arrow_pos - direction * size * 0.5 + perp * size * 0.8
    right = arrow_pos - direction * size * 0.5 - perp * size * 0.8
    
    return np.array([left, tip, right])

# ---------------- Build network ----------------
random.seed(42)
np.random.seed(42)

N_NODES = 20
P_EDGE = 0.20
N_SERVERS = 6
FILE_MB = 80.0

G = nx.fast_gnp_random_graph(N_NODES, P_EDGE, seed=2)
if not nx.is_connected(G):
    comp = max(nx.connected_components(G), key=len)
    G = G.subgraph(comp).copy()

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

# Layout
pos = nx.spring_layout(G, seed=42, k=3.5, iterations=150)
pos_array = np.array([pos[n] for n in nodes])
pos_array = (pos_array - pos_array.mean(axis=0)) * 600
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
        path_bw = widths.get(client, 0.0)
        if path_bw <= 0.0:
            times[s] = inf
        else:
            times[s] = lat_s + x_Mb / path_bw
        
        all_widths[s] = path_bw
        all_parents[s] = parents
    
    return times, all_widths, all_parents

all_times, all_widths, all_parents = estimate_times_to_servers(G, client, servers, FILE_MB)
eval_order = servers.copy()

# ---------------- PyQtGraph Visualization ----------------
class NetworkVisualization(QtWidgets.QMainWindow):
    def __init__(self):
        super().__init__()
        
        self.setWindowTitle('Network Visualization - [SPACE] play/pause, [R] reset, [ESC] quit')
        self.setGeometry(100, 100, 1600, 900)
        
        # Central widget
        central_widget = QtWidgets.QWidget()
        self.setCentralWidget(central_widget)
        layout = QtWidgets.QVBoxLayout(central_widget)
        
        # Graphics view
        self.graphics_view = pg.GraphicsLayoutWidget()
        self.graphics_view.setBackground('#0d1117')
        layout.addWidget(self.graphics_view)
        
        # Create plot
        self.plot = self.graphics_view.addPlot()
        self.plot.setAspectLocked(True)
        self.plot.setXRange(-700, 700)
        self.plot.setYRange(-500, 500)
        self.plot.hideAxis('left')
        self.plot.hideAxis('bottom')
        self.plot.setMouseEnabled(x=True, y=True)
        
        # State
        self.current_frame = 0
        self.max_frames = len(eval_order) + 10
        self.playing = False
        self.evaluated_times = {s: np.nan for s in servers}
        self.current_best = None
        self.current_path = []
        
        # Visual elements storage
        self.edge_curves = {}
        self.edge_info = {}
        self.edge_items = {}
        self.arrow_items = {}
        self.edge_label_items = {}
        self.node_items = {}
        self.text_items = []
        
        self._create_visuals()
        
        # Timer for animation
        self.timer = QtCore.QTimer()
        self.timer.timeout.connect(self.update_frame)
        self.timer.setInterval(1500)  # 1.5 seconds
        
        self.update_visuals()
    
    def _create_visuals(self):
        """Create all visual elements"""
        # Determine bidirectional edges
        bidirectional = set()
        for u, v in G.edges():
            if G.has_edge(v, u):
                bidirectional.add((min(u, v), max(u, v)))
        
        # Create edges and arrows
        for u, v in G.edges():
            # Determine curve offset
            is_bidir = (min(u, v), max(u, v)) in bidirectional
            if is_bidir:
                offset = 50 if u < v else -50
            else:
                offset = 0
            
            # Create curved edge
            curve = create_curved_edge(pos[u], pos[v], curve_offset=offset, num_points=30)
            self.edge_curves[(u, v)] = curve
            
            # Create edge line item
            edge_item = pg.PlotCurveItem(
                x=curve[:, 0], y=curve[:, 1],
                pen=pg.mkPen(color=(153, 179, 230, 128), width=4)
            )
            self.plot.addItem(edge_item)
            self.edge_items[(u, v)] = edge_item
            
            # Store edge info
            lat = _get_edge_latency(G, u, v)
            bw = _get_edge_bandwidth(G, u, v)
            mid_idx = len(curve) // 2
            label_pos = curve[mid_idx]
            self.edge_info[(u, v)] = {
                'latency': lat,
                'bandwidth': bw,
                'label_pos': label_pos,
                'curve': curve
            }
            
            # Create arrow head
            arrow_poly = create_arrow_polygon(pos[u], pos[v], size=28)
            arrow_item = pg.QtWidgets.QGraphicsPolygonItem(
                QtGui.QPolygonF([QtCore.QPointF(p[0], p[1]) for p in arrow_poly])
            )
            arrow_item.setPen(pg.mkPen(None))
            arrow_item.setBrush(pg.mkBrush(color=(179, 204, 255, 255)))
            self.plot.addItem(arrow_item)
            self.arrow_items[(u, v)] = arrow_item
            
            # Create edge label
            label_text = pg.TextItem(
                text=f'{int(lat)}ms\n{int(bw)}Mb/s',
                color='#ffeb3b',
                anchor=(0.5, 0.5)
            )
            label_text.setFont(QtGui.QFont('Arial', 10, weight=75))  # 75 = Bold
            label_text.setPos(label_pos[0], label_pos[1])
            self.plot.addItem(label_text)
            self.edge_label_items[(u, v)] = label_text
        
        # Create nodes
        for n in nodes:
            if n == client:
                color = (51, 255, 102, 255)
                size = 100
            elif n in servers:
                color = (51, 204, 255, 255)
                size = 90
            else:
                color = (153, 179, 230, 255)
                size = 35
            
            node_item = pg.ScatterPlotItem(
                pos=[[pos[n][0], pos[n][1]]],
                size=size,
                pen=pg.mkPen(color='white', width=3),
                brush=pg.mkBrush(color)
            )
            self.plot.addItem(node_item)
            self.node_items[n] = node_item
        
        # Create node labels
        for n in nodes:
            if n == client:
                label = f'CLIENT\n#{n}'
                color = '#00ff66'
                offset = -60
            elif n in servers:
                label = f'SERVER\n#{n}'
                color = '#66ddff'
                offset = 50
            else:
                label = f'{n}'
                color = 'white'
                offset = 0
            
            text_item = pg.TextItem(text=label, color=color, anchor=(0.5, 0.5))
            text_item.setFont(QtGui.QFont('Arial', 12 if n == client else 10, weight=75))
            text_item.setPos(pos[n][0], pos[n][1] + (offset if offset else 0))
            self.plot.addItem(text_item)
            self.text_items.append(text_item)
        
        # Info panels
        self.path_info = pg.TextItem(text='', color='#ffcc44', anchor=(0, 1))
        self.path_info.setFont(QtGui.QFont('Courier', 11, weight=75))
        self.path_info.setPos(-680, 480)
        self.plot.addItem(self.path_info)
        
        self.server_table = pg.TextItem(text='', color='#ffee88', anchor=(1, 1))
        self.server_table.setFont(QtGui.QFont('Courier', 10, weight=75))
        self.server_table.setPos(680, 480)
        self.plot.addItem(self.server_table)
        
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
        self.algo_info = pg.TextItem(text='\n'.join(algo_lines), color='#ffaa66', anchor=(1, 0))
        self.algo_info.setFont(QtGui.QFont('Courier', 10, weight=75))
        self.algo_info.setPos(680, -480)
        self.plot.addItem(self.algo_info)
        
        self.stats_text = pg.TextItem(text='', color='#99ffaa', anchor=(0, 0))
        self.stats_text.setFont(QtGui.QFont('Courier', 10, weight=75))
        self.stats_text.setPos(-680, -480)
        self.plot.addItem(self.stats_text)
        
        self.status_text = pg.TextItem(text='Press SPACE to start', color='white', anchor=(0, 0))
        self.status_text.setFont(QtGui.QFont('Arial', 14, weight=75))
        self.status_text.setPos(-680, -420)
        self.plot.addItem(self.status_text)
        
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
            "ESC - Quit",
        ]
        self.info_text = pg.TextItem(text='\n'.join(info_lines), color='#88ddff', anchor=(1, 0))
        self.info_text.setFont(QtGui.QFont('Courier', 10, weight=75))
        self.info_text.setPos(680, -320)
        self.plot.addItem(self.info_text)
    
    def update_visuals(self):
        """Update visuals based on current frame"""
        # Reset all edges and arrows to base color
        for (u, v), edge_item in self.edge_items.items():
            edge_item.setPen(pg.mkPen(color=(153, 179, 230, 128), width=4))
        for (u, v), arrow_item in self.arrow_items.items():
            arrow_item.setBrush(pg.mkBrush(color=(179, 204, 255, 255)))
        
        if self.current_frame < len(eval_order):
            s = eval_order[self.current_frame]
            
            widths_now, parents_now = widest_path_nx(G, s)
            path = reconstruct_path(parents_now, client, s)
            self.current_path = path
            
            if path and len(path) >= 2:
                total_latency = 0
                min_bandwidth = inf
                
                # Highlight edges and arrows in the active path
                for i in range(len(path) - 1):
                    u, v = path[i], path[i+1]
                    
                    # Highlight edge
                    if (u, v) in self.edge_items:
                        self.edge_items[(u, v)].setPen(
                            pg.mkPen(color=(255, 26, 102), width=12)
                        )
                    
                    # Highlight arrow
                    if (u, v) in self.arrow_items:
                        self.arrow_items[(u, v)].setBrush(
                            pg.mkBrush(color=(255, 0, 153, 255))
                        )
                    
                    lat = _get_edge_latency(G, u, v)
                    bw = _get_edge_bandwidth(G, u, v)
                    total_latency += lat
                    min_bandwidth = min(min_bandwidth, bw)
                
                path_info_lines = [
                    f"🔗 ACTIVE PATH: Server #{s} → Client",
                    f"═════════════════════════════",
                    f"Hops: {len(path)-1}",
                    f"Total Latency: {total_latency:.1f} ms",
                    f"Bottleneck BW: {min_bandwidth:.1f} Mb/s",
                    f"Download Time: {total_latency/1000 + (FILE_MB*8)/min_bandwidth:.2f} sec"
                ]
                self.path_info.setText('\n'.join(path_info_lines))
            else:
                self.path_info.setText('')
            
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
            self.server_table.setText('\n'.join(table_lines))
            
            evaluated_count = sum(1 for t in self.evaluated_times.values() if not np.isnan(t))
            avg_time = np.nanmean([t for t in self.evaluated_times.values() if not np.isnan(t)]) if evaluated_count > 0 else 0
            
            stats_lines = [
                "📈 STATISTICS",
                "═════════════",
                f"Evaluated: {evaluated_count}/{N_SERVERS}",
                f"Average: {avg_time:.2f}s" if evaluated_count > 0 else "Average: N/A",
                f"Best: {self.evaluated_times[self.current_best]:.2f}s" if self.current_best else "Best: N/A",
            ]
            self.stats_text.setText('\n'.join(stats_lines))
            
            status_lines = [
                f"⏱️  Frame {self.current_frame + 1} / {len(eval_order)}",
                f"═════════════════════",
                f"Testing: Server #{s}",
                f"Time: {tval:.2f}s" if not np.isnan(tval) else "Time: ∞",
                "",
                f"🏆 Current Best: Server #{self.current_best}" if self.current_best else "🏆 No winner yet"
            ]
            self.status_text.setText('\n'.join(status_lines))
        else:
            # Complete
            self.status_text.setText(f"✅ COMPLETE!\n═══════════\n🏆 Winner: Server #{self.current_best}\n⏱️  Time: {self.evaluated_times[self.current_best]:.2f}s")
            
            good = {sv: t for sv, t in self.evaluated_times.items() if not np.isnan(t)}
            ranked = sorted(good.items(), key=lambda x: x[1])
            
            medals = {0: "🥇", 1: "🥈", 2: "🥉"}
            table_lines = ["🏅 FINAL RANKING", "═════════════════"]
            for rank, (srv, time_val) in enumerate(ranked):
                medal = medals.get(rank, f"{rank+1}.")
                table_lines.append(f"{medal} Server {srv}: {time_val:.2f}s")
            self.server_table.setText('\n'.join(table_lines))
            
            self.path_info.setText(f"🏆 WINNER: Server #{self.current_best}\n═══════════════════════\nBest time: {self.evaluated_times[self.current_best]:.2f}s")
    
    def update_frame(self):
        if self.current_frame < self.max_frames:
            self.update_visuals()
            self.current_frame += 1
        else:
            self.timer.stop()
            self.playing = False
    
    def keyPressEvent(self, event):
        key = event.key()
        if key == QtCore.Qt.Key.Key_Space or key == 32:  # Space
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
        elif key == QtCore.Qt.Key.Key_R or key == 82:  # R
            self.timer.stop()
            self.playing = False
            self.current_frame = 0
            self.evaluated_times = {s: np.nan for s in servers}
            self.current_best = None
            self.update_visuals()
        elif key == QtCore.Qt.Key.Key_Escape or key == 16777216:  # Escape
            self.close()

if __name__ == '__main__':
    app = QtWidgets.QApplication(sys.argv)
    
    print("\n" + "="*60)
    print("🚀 PyQtGraph Network Visualization")
    print("="*60)
    print("\nFeatures:")
    print("  ✨ Interactive graph with pan/zoom")
    print("  ➡️  Curved edges with arrow heads")
    print("  🔢 Clear edge labels with values")
    print("  💡 Bold colors and smooth animation")
    print("\nControls:")
    print("  SPACE - Play/Pause")
    print("  R     - Reset")
    print("  ESC   - Quit")
    print("="*60 + "\n")
    
    viz = NetworkVisualization()
    viz.show()
    sys.exit(app.exec())
