# Dynamic network -> video-like animation (MP4/GIF) and inline playback.
# Run in Jupyter. Requires: networkx, matplotlib, numpy, pillow.
# For MP4 output you need ffmpeg installed; if not found the code will save a GIF.

import networkx as nx
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, FFMpegWriter, PillowWriter
from heapq import heappush, heappop
from math import inf
from typing import Any, Dict, List, Optional, Iterable, Tuple
import random
import numpy as np
from IPython.display import HTML, display, Video
import os
import shutil

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

# Dijkstra for latencies (returns ms)
def dijkstra_latencies_nx(G: nx.Graph, source) -> Dict[Any, float]:
    weight_attr = 'latency_ms' if any('latency_ms' in d for _,_,d in G.edges(data=True)) else 'latency'
    if not any(weight_attr in d for _,_,d in G.edges(data=True)):
        return {n: (0.0 if n in G else inf) for n in G.nodes()}
    return nx.single_source_dijkstra_path_length(G, source, weight=weight_attr)

# Widest-path (maximin) returns widths and parents
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

# ---------------- Build a larger random network (customize these) ----------------
random.seed(42)
np.random.seed(42)

N_NODES = 40        # larger network
P_EDGE = 0.10       # probability to form an edge
N_SERVERS = 8
FILE_MB = 80.0      # file size (MB)

G = nx.fast_gnp_random_graph(N_NODES, P_EDGE, seed=2)
if not nx.is_connected(G):
    comp = max(nx.connected_components(G), key=len)
    G = G.subgraph(comp).copy()

nodes = list(G.nodes())
client = nodes[0]
servers = nodes[1:1+N_SERVERS]

for u, v in G.edges():
    G.edges[u, v]['latency_ms'] = round(random.uniform(5.0, 250.0), 1)
    G.edges[u, v]['bandwidth_mbps'] = round(random.uniform(1.0, 300.0), 1)

server_bw = {s: round(random.uniform(5.0, 500.0), 1) for s in servers}

pos = nx.spring_layout(G, seed=42)  # fixed layout for stable animation

# ---------------- Precompute metrics ----------------
def estimate_times_to_servers(G, client, servers, x_MB, server_bw):
    x_Mb = x_MB * 8.0
    times = {}
    all_widths = {}
    all_parents = {}
    
    # Compute from each server TO the client
    for s in servers:
        latencies = dijkstra_latencies_nx(G, s)  # Start from server
        widths, parents = widest_path_nx(G, s)   # Start from server
        
        lat_ms = latencies.get(client, inf)
        if lat_ms == inf:
            times[s] = inf
            all_widths[s] = 0.0
            all_parents[s] = parents
            continue
        
        lat_s = lat_ms / 1000.0
        path_bw = widths.get(client, 0.0)  # Get bandwidth to client
        eff_bw = min(path_bw, float(server_bw.get(s, 0.0)))
        if eff_bw <= 0.0:
            times[s] = inf
        else:
            times[s] = lat_s + x_Mb / eff_bw
        
        all_widths[s] = path_bw
        all_parents[s] = parents
    
    return times, all_widths, all_parents

all_times, all_widths, all_parents = estimate_times_to_servers(G, client, servers, FILE_MB, server_bw)
eval_order = servers.copy()
frames = len(eval_order) + 20  # evaluate each server, then extra frames for final pause

# ---------------- Create plotting canvas (single figure with subplots) ----------------
fig, (ax_net, ax_bar) = plt.subplots(1, 2, figsize=(14, 6), gridspec_kw={'width_ratios':[2,1]})
plt.tight_layout()

# plotting helpers
server_labels = [str(s) for s in servers]
ymax = max(t for t in all_times.values() if t < inf) * 1.6 if any(t < inf for t in all_times.values()) else 1.0
evaluated_times = {s: np.nan for s in servers}
current_best = None

def draw_frame(frame_idx):
    ax_net.clear(); ax_bar.clear()
    ax_net.set_title("Network (path from server to client highlighted)")
    ax_net.axis('off')

    # draw base nodes and tiny edges
    nx.draw_networkx_nodes(G, pos, ax=ax_net, node_size=100)
    nx.draw_networkx_labels(G, pos, ax=ax_net, font_size=7)

    all_edges = list(G.edges())
    nx.draw_networkx_edges(G, pos, edgelist=all_edges, ax=ax_net, width=0.6)

    # partial edge labeling to reduce clutter
    label_candidates = all_edges.copy()
    random.seed(100 + frame_idx)  # deterministic subset per frame
    label_sample = random.sample(label_candidates, min(len(label_candidates), 40))
    edge_labels = {}
    for u, v in label_sample:
        lat = _get_edge_latency(G, u, v)
        bw = _get_edge_bandwidth(G, u, v)
        edge_labels[(u, v)] = f"{int(lat)}ms/{int(bw)}Mbps"
        edge_labels[(v, u)] = edge_labels[(u, v)]
    nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_size=6, ax=ax_net)

    # highlight client and servers
    nx.draw_networkx_nodes(G, pos, nodelist=[client], ax=ax_net, node_size=320)
    nx.draw_networkx_nodes(G, pos, nodelist=list(servers), ax=ax_net, node_size=260)

    # evaluate server for this frame if within eval_order
    if frame_idx < len(eval_order):
        s = eval_order[frame_idx]
        # recompute parents/widths starting FROM the server
        widths_now, parents_now = widest_path_nx(G, s)  # Start from server
        path = reconstruct_path(parents_now, client, s)  # Path from server to client
        if path and len(path) >= 2:
            path_edges = [(path[i], path[i+1]) for i in range(len(path)-1)]
            nx.draw_networkx_edges(G, pos, edgelist=path_edges, ax=ax_net, width=3.5, edge_color='red', arrows=True, arrowsize=20)
        # annotate server under evaluation
        if s in pos:
            x,y = pos[s]; ax_net.annotate("evaluating", (x,y), textcoords="offset points", xytext=(0,10), ha='center', fontsize=8, color='red', weight='bold')
        # compute and store time for s
        x_Mb = FILE_MB * 8.0
        latencies = dijkstra_latencies_nx(G, s)  # Start from server
        lat_ms = latencies.get(client, inf)  # Get latency to client
        if lat_ms == inf:
            tval = np.nan
        else:
            lat_s = lat_ms / 1000.0
            path_bw = widths_now.get(client, 0.0)  # Get bandwidth to client
            eff_bw = min(path_bw, float(server_bw.get(s, 0.0)))
            tval = lat_s + x_Mb / eff_bw if eff_bw > 0 else np.nan
        evaluated_times[s] = tval

    # after evaluations, determine current best among evaluated ones
    good = {sv: t for sv, t in evaluated_times.items() if not np.isnan(t)}
    global current_best
    if good:
        current_best = min(good, key=good.get)
    else:
        current_best = None

    # annotate client and current best
    if client in pos:
        x,y = pos[client]; ax_net.annotate("client", (x,y), textcoords="offset points", xytext=(0,-12), ha='center', fontsize=8)
    if current_best is not None and current_best in pos:
        x,y = pos[current_best]; ax_net.annotate("current best", (x,y), textcoords="offset points", xytext=(0,-12), ha='center', fontsize=8)

    # ---- bar chart
    ax_bar.set_title("Estimated time per server (s)")
    ax_bar.set_ylim(0, ymax)
    heights = [(evaluated_times[s] if not np.isnan(evaluated_times[s]) else 0.0) for s in servers]
    bars = ax_bar.bar(server_labels, heights)
    # annotate numeric values for evaluated bars
    for idx, s in enumerate(servers):
        val = evaluated_times[s]
        if not np.isnan(val):
            ax_bar.text(idx, heights[idx] + ymax*0.02, f"{val:.1f}", ha='center', va='bottom', fontsize=8)

    # side-panel info (server capacities)
    info_lines = [f"file = {FILE_MB} MB", "server caps (Mbps):"]
    info_lines += [f"{s}: {server_bw[s]:.0f}" for s in servers]
    ax_bar.text(1.02, 0.5, "\n".join(info_lines), transform=ax_bar.transAxes, fontsize=8, va='center')
    ax_bar.set_xticklabels(server_labels, rotation=45)

# ---------------- Build animation ----------------
anim = FuncAnimation(fig, lambda i: draw_frame(i), frames=frames, interval=700, repeat=False)

# ---------------- Save animation to file (MP4 if ffmpeg available, else GIF fallback) ----------------
out_mp4 = "animation.mp4"
out_gif = "animation.gif"

ffmpeg_exists = shutil.which("ffmpeg") is not None
saved_fname = None

print("Saving animation to file (this may take a few seconds)...")
try:
    if ffmpeg_exists:
        print("ffmpeg found -> saving MP4:", out_mp4)
        writer = FFMpegWriter(fps=1000/700)  # approximate fps from interval
        anim.save(out_mp4, writer=writer, dpi=150)
        saved_fname = out_mp4
    else:
        print("ffmpeg not found -> saving GIF fallback:", out_gif)
        writer = PillowWriter(fps=1000/700)
        anim.save(out_gif, writer=writer)
        saved_fname = out_gif
except Exception as e:
    print("Error while saving animation:", e)
    print("Attempting to save GIF with PillowWriter as fallback...")
    try:
        writer = PillowWriter(fps=1000/700)
        anim.save(out_gif, writer=writer)
        saved_fname = out_gif
    except Exception as e2:
        print("Fallback GIF save failed:", e2)
        saved_fname = None

# ---------------- Display inline (if saved) ----------------
plt.close(fig)  # close the interactive figure to avoid duplicate display
if saved_fname and os.path.exists(saved_fname):
    print("Saved animation to:", saved_fname)
    # show inline using IPython.display.Video if MP4, otherwise HTML for GIF
    if saved_fname.endswith(".mp4"):
        display(Video(saved_fname, embed=True, width=900))
    else:
        # GIF
        display(HTML(f'<img src="{saved_fname}" width="900">'))
else:
    # If saving failed, fall back to inline HTML5 video from the animation object (may be slower)
    try:
        display(HTML(anim.to_html5_video()))
    except Exception as e:
        print("Unable to display animation inline:", e)
        print("You can try running this notebook locally and ensure ffmpeg is installed for MP4 export.")
