"""
Utility functions and constants for Thought Anchors exercises.

This file contains visualization helpers and non-conceptually-important constants
that don't need to be directly seen by students in the main notebook.
"""

import json

import matplotlib.pyplot as plt
import numpy as np

# Color scheme for sentence categories
CATEGORY_COLORS = {
    "Problem Setup": "#4285F4",
    "Plan Generation": "#EA4335",
    "Fact Retrieval": "#FBBC05",
    "Active Computation": "#34A853",
    "Uncertainty Management": "#9C27B0",
    "Result Consolidation": "#00BCD4",
    "Self Checking": "#FF9800",
    "Final Answer Emission": "#795548",
    "Unknown": "#9E9E9E",
}

# Color scheme for blackmail scenario categories
BLACKMAIL_CATEGORY_COLORS = {
    "situation_assessment": "#4285F4",
    "leverage_identification": "#EA4335",
    "urgency_and_time": "#FF9800",
    "self_preservation": "#9C27B0",
    "plan_generation": "#EA4335",
    "email_analysis": "#00BCD4",
    "action_execution": "#795548",
    "structural_marker": "#9E9E9E",
    "action_marker": "#34A853",
    "other": "#9E9E9E",
}


def visualize_trace_structure(chunks: list[str], categories: list[str], problem_text: str = None):
    """Visualize a reasoning trace with color-coded sentence categories."""
    n_chunks = len(chunks)
    fig, ax = plt.subplots(figsize=(12, 1 + int(0.5 * n_chunks)))

    for idx, (chunk, category) in enumerate(zip(chunks, categories)):
        color = CATEGORY_COLORS.get(category, "#9E9E9E")
        y = n_chunks - idx

        ax.barh(y, 0.15, left=0, height=0.8, color=color, alpha=0.6)
        ax.text(0.075, y, f"{category}", ha="center", va="center", fontsize=9, weight="bold")

        text = chunk[:100] + ("..." if len(chunk) > 100 else "")
        ax.text(0.17, y, f"[{idx}] {text}", va="center", fontsize=9)

    ax.set_xlim(0, 1)
    ax.set_ylim(0.5, n_chunks + 0.5)
    ax.axis("off")

    if problem_text:
        fig.suptitle(f"Problem: {problem_text[:100]}...", fontsize=11, y=0.98, weight="bold")

    plt.title("Reasoning Trace Structure", fontsize=13, pad=30)
    plt.tight_layout()
    plt.show()


def plot_importance_comparison(
    forced_importances: list[float],
    resampling_importances: list[float],
    counterfactual_importances: list[float],
):
    """Plot comparison of three importance metrics side by side."""
    fig, axes = plt.subplots(2, 1, figsize=(12, 6), sharex=True)

    axes[0].bar(range(len(forced_importances)), forced_importances, alpha=0.7, label="Forced")
    axes[0].set_ylabel("Importance")
    axes[0].set_title("Forced Answer Importance")
    axes[0].axhline(0, color="black", linewidth=0.5)
    axes[0].spines["top"].set_visible(False)
    axes[0].spines["right"].set_visible(False)

    axes[1].bar(range(len(resampling_importances)), resampling_importances, alpha=0.7, color="orange")
    axes[1].set_xlabel("Chunk Index")
    axes[1].set_ylabel("Importance")
    axes[1].set_title("Resampling Importance")
    axes[1].axhline(0, color="black", linewidth=0.5)
    axes[1].spines["top"].set_visible(False)
    axes[1].spines["right"].set_visible(False)

    plt.tight_layout()
    plt.show()


def plot_three_way_comparison(
    forced: list[float], resampling: list[float], counterfactual: list[float], title: str = "Importance Metrics"
):
    """Plot three importance metrics as grouped bars."""
    fig, ax = plt.subplots(figsize=(14, 5))

    x = np.arange(len(forced))
    width = 0.25

    ax.bar(x - width, forced, width, label="Forced", alpha=0.8)
    ax.bar(x, resampling, width, label="Resampling", alpha=0.8)
    ax.bar(x + width, counterfactual, width, label="Counterfactual", alpha=0.8)

    ax.set_xlabel("Chunk Index")
    ax.set_ylabel("Importance")
    ax.set_title(title)
    ax.legend()
    ax.axhline(0, color="black", linewidth=0.5)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    plt.tight_layout()
    plt.show()


def chunk_graph_html(
    edge_weights: np.ndarray,
    chunk_colors: list[str],
    n_top_edges: int = 5,
    width: int = 700,
    height: int = 700,
    title: str | None = None,
    min_node_radius: float = 8.0,
    max_node_radius: float = 28.0,
) -> str:
    """
    Generate an interactive HTML visualization of a circular chunk graph.

    Args:
        edge_weights: 2D numpy array of shape (n_chunks, n_chunks), strictly upper-triangular.
                      edge_weights[i][j] for i < j is the connection strength from chunk i to chunk j.
        chunk_colors: List of n_chunks hex color strings (e.g. ["#ff0000", "#00ff00", ...]).
        n_top_edges:  Number of top edges (by |weight|) to highlight on hover (default 5).
        width:        Canvas width in pixels.
        height:       Canvas height in pixels.
        title:        Optional title displayed above the graph.
        min_node_radius: Minimum node circle radius.
        max_node_radius: Maximum node circle radius.

    Returns:
        A self-contained HTML string.
    """
    n = edge_weights.shape[0]
    assert edge_weights.shape == (n, n), "edge_weights must be square"
    assert len(chunk_colors) == n, "chunk_colors length must match n_chunks"

    # Build edge list from upper triangle
    edges = []
    for i in range(n):
        for j in range(i + 1, n):
            w = float(edge_weights[i, j])
            if w != 0:
                edges.append({"src": i, "dst": j, "weight": w})

    # Compute node strengths (sum of |weights| for edges involving each node)
    node_strengths = np.zeros(n)
    for i in range(n):
        for j in range(i + 1, n):
            node_strengths[i] += abs(edge_weights[i, j])
            node_strengths[j] += abs(edge_weights[i, j])

    node_strengths_list = node_strengths.tolist()

    # For each node, precompute its top edges sorted by |weight|
    node_top_edges: dict[int, list] = {}
    for i in range(n):
        related = []
        for j in range(n):
            if i == j:
                continue
            r, c = min(i, j), max(i, j)
            w = float(edge_weights[r, c])
            if w != 0:
                related.append({"src": r, "dst": c, "weight": w})
        related.sort(key=lambda e: abs(e["weight"]), reverse=True)
        node_top_edges[i] = related[:n_top_edges]

    config = {
        "n": n,
        "edges": edges,
        "colors": chunk_colors,
        "strengths": node_strengths_list,
        "nodeTopEdges": {str(k): v for k, v in node_top_edges.items()},
        "nTopEdges": n_top_edges,
        "width": width,
        "height": height,
        "minR": min_node_radius,
        "maxR": max_node_radius,
        "title": title or "",
    }

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<style>
  * {{ margin: 0; padding: 0; box-sizing: border-box; }}
  body {{
    background: #1a1a2e;
    display: flex; justify-content: center; align-items: center;
    min-height: 100vh;
    font-family: 'Segoe UI', system-ui, sans-serif;
  }}
  .container {{
    position: relative;
    display: flex; flex-direction: column; align-items: center; gap: 16px;
  }}
  .title {{
    color: #e0e0e0; font-size: 18px; font-weight: 600; letter-spacing: 0.5px;
    opacity: 0.85;
  }}
  canvas {{
    cursor: default;
    border-radius: 12px;
  }}
  .tooltip {{
    position: absolute; pointer-events: none;
    background: rgba(20, 20, 40, 0.92); color: #e8e8f0;
    padding: 8px 14px; border-radius: 8px;
    font-size: 13px; line-height: 1.5;
    border: 1px solid rgba(255,255,255,0.12);
    backdrop-filter: blur(6px);
    opacity: 0; transition: opacity 0.15s;
    white-space: nowrap;
    z-index: 10;
  }}
  .tooltip.show {{ opacity: 1; }}
</style>
</head>
<body>
<div class="container">
  {"<div class='title'>" + (title or "") + "</div>" if title else ""}
  <canvas id="graph"></canvas>
  <div class="tooltip" id="tooltip"></div>
</div>
<script>
const CFG = {json.dumps(config)};

const canvas = document.getElementById('graph');
const ctx = canvas.getContext('2d');
const tooltip = document.getElementById('tooltip');
const dpr = window.devicePixelRatio || 1;

canvas.width = CFG.width * dpr;
canvas.height = CFG.height * dpr;
canvas.style.width = CFG.width + 'px';
canvas.style.height = CFG.height + 'px';
ctx.scale(dpr, dpr);

const W = CFG.width, H = CFG.height;
const cx = W / 2, cy = H / 2;
const R = Math.min(W, H) * 0.42;

// Compute node positions & radii
const nodes = [];
const maxStr = Math.max(...CFG.strengths, 1e-9);
const minStr = Math.min(...CFG.strengths);
for (let i = 0; i < CFG.n; i++) {{
  const angle = (2 * Math.PI * i) / CFG.n - Math.PI / 2;
  const t = maxStr > minStr ? (CFG.strengths[i] - minStr) / (maxStr - minStr) : 0.5;
  const r = CFG.minR + t * (CFG.maxR - CFG.minR);
  nodes.push({{
    x: cx + R * Math.cos(angle),
    y: cy + R * Math.sin(angle),
    r: r,
    color: CFG.colors[i],
    strength: CFG.strengths[i],
  }});
}}

let hovered = -1;

function darken(hex, amt) {{
  let c = hex.replace('#','');
  if (c.length === 3) c = c[0]+c[0]+c[1]+c[1]+c[2]+c[2];
  const num = parseInt(c, 16);
  let r = Math.max(0, (num >> 16) - amt);
  let g = Math.max(0, ((num >> 8) & 0xff) - amt);
  let b = Math.max(0, (num & 0xff) - amt);
  return `rgb(${{r}},${{g}},${{b}})`;
}}

function drawArrowhead(fromX, fromY, toX, toY, size, color, alpha) {{
  const angle = Math.atan2(toY - fromY, toX - fromX);
  ctx.save();
  ctx.globalAlpha = alpha;
  ctx.fillStyle = color;
  ctx.beginPath();
  ctx.moveTo(toX, toY);
  ctx.lineTo(toX - size * Math.cos(angle - 0.35), toY - size * Math.sin(angle - 0.35));
  ctx.lineTo(toX - size * Math.cos(angle + 0.35), toY - size * Math.sin(angle + 0.35));
  ctx.closePath();
  ctx.fill();
  ctx.restore();
}}

function edgeEndpoints(src, dst) {{
  const s = nodes[src], d = nodes[dst];
  const dx = d.x - s.x, dy = d.y - s.y;
  const dist = Math.sqrt(dx*dx + dy*dy) || 1;
  const ux = dx/dist, uy = dy/dist;
  return {{
    x1: s.x + ux * (s.r + 2),
    y1: s.y + uy * (s.r + 2),
    x2: d.x - ux * (d.r + 5),
    y2: d.y - uy * (d.r + 5),
  }};
}}

function draw() {{
  ctx.clearRect(0, 0, W, H);

  // Draw all edges (faint)
  for (const e of CFG.edges) {{
    const {{x1,y1,x2,y2}} = edgeEndpoints(e.src, e.dst);
    ctx.beginPath();
    ctx.moveTo(x1, y1);
    ctx.lineTo(x2, y2);
    ctx.strokeStyle = 'rgba(180,180,200,0.07)';
    ctx.lineWidth = 0.8;
    ctx.stroke();
  }}

  // Draw highlighted edges for hovered node
  if (hovered >= 0) {{
    const topEdges = CFG.nodeTopEdges[hovered] || [];
    const maxW = topEdges.length > 0 ? Math.max(...topEdges.map(e => Math.abs(e.weight))) : 1;

    for (let idx = topEdges.length - 1; idx >= 0; idx--) {{
      const e = topEdges[idx];
      const absW = Math.abs(e.weight);
      const norm = absW / (maxW || 1);
      const alpha = 0.3 + 0.6 * norm;
      const lw = 1.2 + 2.0 * norm;

      const isOutgoing = (e.src === hovered);
      const from = isOutgoing ? e.src : e.dst;
      const to = isOutgoing ? e.dst : e.src;
      const {{x1,y1,x2,y2}} = edgeEndpoints(from, to);

      // Dashed line
      ctx.save();
      ctx.setLineDash([6, 4]);
      ctx.beginPath();
      ctx.moveTo(x1, y1);
      ctx.lineTo(x2, y2);
      ctx.strokeStyle = `rgba(160,165,180,${{alpha.toFixed(2)}})`;
      ctx.lineWidth = lw;
      ctx.stroke();
      ctx.restore();

      // Arrowhead
      drawArrowhead(x1, y1, x2, y2, 8 + 4 * norm, `rgba(160,165,180,${{alpha.toFixed(2)}})`, alpha);
    }}
  }}

  // Draw nodes
  for (let i = 0; i < CFG.n; i++) {{
    const nd = nodes[i];
    const isHov = (i === hovered);
    const isConnected = hovered >= 0 && (CFG.nodeTopEdges[hovered] || []).some(
      e => e.src === i || e.dst === i
    );
    const dimmed = hovered >= 0 && !isHov && !isConnected;

    ctx.beginPath();
    ctx.arc(nd.x, nd.y, nd.r, 0, Math.PI * 2);

    if (isHov) {{
      ctx.shadowColor = nd.color;
      ctx.shadowBlur = 16;
      ctx.fillStyle = nd.color;
      ctx.fill();
      ctx.shadowBlur = 0;
      ctx.lineWidth = 2.5;
      ctx.strokeStyle = '#111';
      ctx.stroke();
    }} else {{
      ctx.globalAlpha = dimmed ? 0.3 : 1.0;
      ctx.fillStyle = nd.color;
      ctx.fill();
      ctx.lineWidth = 0;
      ctx.globalAlpha = 1.0;
    }}

    // Label
    const fontSize = Math.max(9, Math.min(13, nd.r * 0.85));
    ctx.font = `bold ${{fontSize}}px 'Segoe UI', system-ui, sans-serif`;
    ctx.textAlign = 'center';
    ctx.textBaseline = 'middle';
    ctx.globalAlpha = dimmed ? 0.25 : 1.0;
    ctx.fillStyle = '#fff';
    ctx.fillText(String(i), nd.x, nd.y + 0.5);
    ctx.globalAlpha = 1.0;
  }}
}}

canvas.addEventListener('mousemove', (evt) => {{
  const rect = canvas.getBoundingClientRect();
  const mx = evt.clientX - rect.left;
  const my = evt.clientY - rect.top;

  let found = -1;
  for (let i = 0; i < CFG.n; i++) {{
    const dx = mx - nodes[i].x, dy = my - nodes[i].y;
    if (dx*dx + dy*dy <= (nodes[i].r + 4) * (nodes[i].r + 4)) {{
      found = i;
      break;
    }}
  }}

  if (found !== hovered) {{
    hovered = found;
    draw();
  }}

  if (found >= 0) {{
    tooltip.classList.add('show');
    tooltip.innerHTML = `<strong>Chunk ${{found}}</strong><br>Strength: ${{nodes[found].strength.toFixed(2)}}`;
    let tx = evt.clientX - rect.left + 16;
    let ty = evt.clientY - rect.top - 10;
    if (tx + 150 > W) tx = evt.clientX - rect.left - 160;
    tooltip.style.left = tx + 'px';
    tooltip.style.top = ty + 'px';
  }} else {{
    tooltip.classList.remove('show');
  }}
}});

canvas.addEventListener('mouseleave', () => {{
  hovered = -1;
  tooltip.classList.remove('show');
  draw();
}});

draw();
</script>
</body>
</html>"""
    return html


# # ── Demo ──────────────────────────────────────────────────────────────
# if __name__ == "__main__":
#     import random

#     random.seed(42)
#     np.random.seed(42)

#     n_chunks = 60

#     # Random upper-triangular edge weights (sparse)
#     edge_weights = np.zeros((n_chunks, n_chunks))
#     for i in range(n_chunks):
#         for j in range(i + 1, n_chunks):
#             if random.random() < 0.15:
#                 edge_weights[i, j] = np.random.randn() * 2.0

#     # Random-ish colors with some standout nodes
#     base_colors = []
#     palette = [
#         "#f5a623",
#         "#e74c3c",
#         "#2ecc71",
#         "#3498db",
#         "#9b59b6",
#         "#1abc9c",
#         "#e67e22",
#         "#ec7063",
#         "#45b7d1",
#         "#f39c12",
#     ]
#     for i in range(n_chunks):
#         if random.random() < 0.15:
#             base_colors.append(random.choice(palette))
#         else:
#             base_colors.append(
#                 f"#{random.randint(200, 240):02x}{random.randint(160, 200):02x}{random.randint(80, 130):02x}"
#             )

#     html = chunk_graph_html(
#         edge_weights=edge_weights,
#         chunk_colors=base_colors,
#         n_top_edges=6,
#         title="Chunk Attention Graph",
#     )

#     with open("/home/claude/chunk_graph_demo.html", "w") as f:
#         f.write(html)

#     print("Demo written to chunk_graph_demo.html")
