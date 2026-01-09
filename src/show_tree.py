from pathlib import Path
from collections import deque

import networkx as nx
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap


def build_graph(data, max_depth=None, max_children=None):
    """
    Build a DiGraph where node ids are path-like strings ("0", "0.1", "0.1.2"...).
    Each node has attr 'q_mean' (float).
    """
    G = nx.DiGraph()

    # BFS so we can easily respect max_depth and max_children
    q = deque([("0", 0, data)])
    while q:
        node_id, depth, nd = q.popleft()
        q_mean = float(nd.get("q_mean", 0.0))
        G.add_node(node_id, q_mean=q_mean)

        if max_depth is not None and depth >= max_depth:
            continue

        children = nd.get("children", []) or []
        if max_children is not None:
            children = children[:max_children]

        for i, ch in enumerate(children):
            child_id = f"{node_id}.{i}"
            G.add_edge(node_id, child_id)
            q.append((child_id, depth + 1, ch))

    return G


def _leaf_counts(G, root="0"):
    """
    Return dict node -> number of descendant leaves (counting itself if leaf).
    """
    counts = {}

    def dfs(n):
        children = list(G.successors(n))
        if not children:
            counts[n] = 1
            return 1
        s = 0
        for c in children:
            s += dfs(c)
        counts[n] = max(1, s)
        return counts[n]

    dfs(root)
    return counts


def hierarchy_pos(G, root="0", vert_gap=1.0, vert_loc=0.0, xcenter=0.0, leaf_pad=0.4):
    """
    Top-down tree layout. Horizontal span for a node equals the number of leaves
    under it, so leaves are evenly separated. `leaf_pad` adds extra spacing
    between adjacent leaves (in leaf-width units).
    Returns: dict node -> (x, y)
    """
    lc = _leaf_counts(G, root)
    total_leaves = lc.get(root, 1)

    # The total width is number of leaves plus padding slots between leaves
    width = total_leaves + (total_leaves - 1) * leaf_pad

    # Map each leaf to a consecutive "slot" along x
    pos = {}

    def assign(n, x_left, x_right, y):
        children = list(G.successors(n))
        if not children:
            pos[n] = ((x_left + x_right) / 2.0, y)
            return

        # Each child gets width proportional to its leaf count, plus padding between children
        child_leafs = [lc[c] for c in children]
        # Convert leaf counts to leaf+pad units
        # For k children, there are (k-1) internal pads between their blocks.
        total_units = sum(child_leafs) + (len(children) - 1) * leaf_pad
        run = x_left
        child_centers = []
        for i, c in enumerate(children):
            w_units = child_leafs[i]
            w = (w_units / total_units) * (x_right - x_left)
            # remaining padding width in this segment:
            pad = 0.0
            if i < len(children) - 1:
                pad = (leaf_pad / total_units) * (x_right - x_left)

            cx_left = run
            cx_right = run + w
            assign(c, cx_left, cx_right, y - vert_gap)
            child_centers.append(pos[c][0])
            run = cx_right + pad

        pos[n] = (sum(child_centers) / len(child_centers), y)

    # Lay out with the computed width centered at xcenter
    x_min = xcenter - width / 2.0
    x_max = xcenter + width / 2.0
    assign(root, x_min, x_max, vert_loc)
    return pos


def draw_tree(G, out_path: Path, dpi=200, font_size=9):
    if "0" not in G:
        raise ValueError("Root node '0' missing. Is the JSON from node_to_dict?")

    # More space between levels + better leaf separation
    pos = hierarchy_pos(G, root="0", vert_gap=1.4, vert_loc=0.0, xcenter=0.0, leaf_pad=0.6)

    # Build labels: only q_mean
    labels = {n: f"{G.nodes[n].get('q_mean', 0.0):.3f}" for n in G.nodes}

    # Heuristic figure size: scale by depth & breadth
    depths = [len(n.split(".")) - 1 for n in G.nodes]
    max_depth = max(depths) if depths else 0
    by_depth = {}
    for n in G.nodes:
        d = len(n.split(".")) - 1
        by_depth[d] = by_depth.get(d, 0) + 1
    max_breadth = max(by_depth.values()) if by_depth else 1
    w = max(6.0, min(30.0, 1.2 * max_breadth))
    h = max(4.0, min(30.0, 1.6 * (max_depth + 1)))

    plt.figure(figsize=(w, h), dpi=dpi)
    ax = plt.gca()
    ax.set_axis_off()

    # --- Colors: map q_mean in [-1, 1] to red->yellow->green with a colorbar ---
    values = [G.nodes[n].get("q_mean", 0.0) for n in G.nodes]
    cmap = LinearSegmentedColormap.from_list("red2green", ["red", "yellow", "green"])
    norm = plt.Normalize(-1.0, 1.0)

    # Edges
    nx.draw_networkx_edges(G, pos, arrows=False, width=0.9, alpha=0.7)

    # Nodes (colored by q_mean)
    nx.draw_networkx_nodes(
        G,
        pos,
        node_size=900,
        linewidths=0.9,
        edgecolors="black",
        node_color=values,
        cmap=cmap,
        vmin=-1.0,
        vmax=1.0,
    )

    # Labels (q_mean only)
    nx.draw_networkx_labels(
        G,
        pos,
        labels=labels,
        font_size=font_size,
        bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="none", alpha=0.7),
    )

    # Colorbar
    sm = plt.cm.ScalarMappable(norm=norm, cmap=cmap)
    sm.set_array([])
    cbar = plt.colorbar(sm, ax=ax, fraction=0.028, pad=0.02)
    cbar.set_label("q_mean")
    cbar.set_ticks([-1.0, 0.0, 1.0])

    plt.tight_layout()
    plt.savefig(out_path, bbox_inches="tight")
    plt.close()
