import json
import shutil
from pathlib import Path

import numpy as np
import torch as t
from torch import Tensor

Arr = np.ndarray


# ==================================================
# PART 3.2 UTILS (memory profiling, shared helpers)
# ==================================================


def to_numpy(tensor: t.Tensor | Arr) -> Arr:
    """Convert a tensor or array to a numpy array."""
    if isinstance(tensor, np.ndarray):
        return tensor
    return tensor.detach().cpu().numpy()


def get_tensor_size(obj: t.Tensor) -> int:
    """Get the memory size of a tensor in bytes."""
    return obj.element_size() * obj.nelement()


def get_tensors_size(obj: t.nn.Module | t.Tensor) -> int:
    """Get the total memory size of a module's parameters and buffers (or a single tensor) in bytes."""
    if isinstance(obj, t.Tensor):
        return get_tensor_size(obj)
    total = 0
    for param in obj.parameters():
        total += get_tensor_size(param)
    for buffer in obj.buffers():
        total += get_tensor_size(buffer)
    return total


def get_device(obj: t.nn.Module | t.Tensor) -> str:
    """Get the device of a module or tensor."""
    if isinstance(obj, t.Tensor):
        return str(obj.device)
    try:
        return str(next(obj.parameters()).device)
    except StopIteration:
        return "N/A"


def print_memory_status() -> None:
    """Print current CUDA memory allocation info."""
    if t.cuda.is_available():
        allocated = t.cuda.memory_allocated() / 1024**3
        reserved = t.cuda.memory_reserved() / 1024**3
        free = reserved - allocated
        print(f"Allocated = {allocated:.2f} GB")
        print(f"Reserved = {reserved:.2f} GB")
        print(f"Free = {free:.2f}")


def profile_pytorch_memory(
    namespace: dict | None = None,
    n_top: int = 10,
    filter_device: str | None = None,
) -> None:
    """Profile memory usage of PyTorch objects in the given namespace."""
    if namespace is None:
        return

    objs = []
    for name, obj in namespace.items():
        if name.startswith("_"):
            continue
        if isinstance(obj, (t.Tensor, t.nn.Module)):
            size = get_tensors_size(obj) / 1024**3
            device = get_device(obj)
            if filter_device and device != filter_device:
                continue
            objs.append((name, type(obj).__name__, device, size))

    objs.sort(key=lambda x: x[3], reverse=True)
    objs = objs[:n_top]

    if t.cuda.is_available():
        allocated = t.cuda.memory_allocated() / 1024**3
        total = t.cuda.memory_reserved() / 1024**3
        free = total - allocated
        print(f"Allocated = {allocated:.2f} GB")
        print(f"Total = {total:.2f} GB")
        print(f"Free = {free:.2f} GB")

    from tabulate import tabulate

    headers = ["Name", "Object", "Device", "Size (GB)"]
    rows = [(name, obj_type, device, f"{size:.2f}") for name, obj_type, device, size in objs]
    print(tabulate(rows, headers=headers, tablefmt="simple_outline"))


# ==================================================
# ATTRIBUTION GRAPH UTILS (for section 3)
# ==================================================


def normalize_matrix(matrix: Tensor) -> Tensor:
    """Row-normalize a matrix by absolute values.

    Takes the elementwise absolute value, then divides each row by its sum. Rows that sum to zero
    (or near-zero) are left as zeros thanks to the clamp on the denominator.
    """
    normalized = matrix.abs()
    return normalized / normalized.sum(dim=1, keepdim=True).clamp(min=1e-10)


def compute_influence(A: Tensor, logit_weights: Tensor, max_iter: int = 1000) -> Tensor:
    """Compute total influence of each node on the output logits.

    Uses iterative matrix-vector products: influence = w @ A + w @ A^2 + w @ A^3 + ...

    Because of the attribution graph's causal structure (features in layer i can only have edges to
    features in layers < i), A is nilpotent: A^L = 0 for L = number of layers. This guarantees
    convergence in at most L iterations.

    Args:
        A: Normalized adjacency matrix (n_nodes, n_nodes)
        logit_weights: (n_nodes,) vector with logit probabilities at the logit node positions

    Returns:
        influence: (n_nodes,) total influence of each node
    """
    current = logit_weights @ A
    influence = current.clone()
    iterations = 0
    while current.any():
        if iterations >= max_iter:
            raise RuntimeError(f"Influence computation failed to converge after {iterations} iterations")
        current = current @ A
        influence += current
        iterations += 1
    return influence


def compute_node_influence(adjacency_matrix: Tensor, logit_weights: Tensor) -> Tensor:
    """Compute node influence by normalizing the adjacency matrix then running power iteration."""
    return compute_influence(normalize_matrix(adjacency_matrix), logit_weights)


def compute_edge_influence(pruned_matrix: Tensor, logit_weights: Tensor) -> Tensor:
    """Compute per-edge influence scores.

    For each edge (i, j), the score is: normalized_A[i,j] * (influence[i] + logit_weight[i]),
    i.e. the edge weight times the total outgoing influence of the source node.
    """
    normalized_pruned = normalize_matrix(pruned_matrix)
    pruned_influence = compute_influence(normalized_pruned, logit_weights)
    pruned_influence += logit_weights
    edge_scores = normalized_pruned * pruned_influence[:, None]
    return edge_scores


def find_threshold(scores: Tensor, threshold: float) -> Tensor:
    """Find the score value such that keeping all scores above it retains `threshold` fraction of total."""
    sorted_scores = t.sort(scores, descending=True).values
    cumulative_score = t.cumsum(sorted_scores, dim=0) / t.sum(sorted_scores)
    threshold_index = int(t.searchsorted(cumulative_score, threshold).item())
    threshold_index = min(threshold_index, len(cumulative_score) - 1)
    return sorted_scores[threshold_index]


# ==================================================
# ATTRIBUTION GRAPH DASHBOARD (for section 3)
# ==================================================

# These are the node_type values expected by the frontend JS
NODE_TYPE_JS_MAP = {
    "embedding": "embedding",
    "latent": "latent",
    "mlp_error": "mlp_error",
    "logit": "logit",
}

STR_TOKENS_MAP = {
    "<start_of_turn>": "<ctrl99>",
    "<end_of_turn>": "<ctrl100>",
    "\n": "⏎",
}


def create_attribution_dashboard(
    result: "AttributionResult",
    output_dir: str | Path = "attribution_dashboards",
) -> Path:
    """
    Create an interactive attribution graph dashboard from an AttributionResult.

    Copies the JS/CSS template files and generates a data.js file containing the graph
    data in the format expected by the frontend.

    Args:
        result: The AttributionResult from the `attribute()` function.
        output_dir: Directory to save the dashboard files.

    Returns:
        Path to the generated index.html file.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Get the template directory (relative to this file)
    templates_dir = Path(__file__).parent / "attribution_graphs" / "templates"

    # Copy all template files
    for suffix in [".js", ".css"]:
        for template_file in templates_dir.glob(f"*{suffix}"):
            shutil.copy2(template_file, output_dir / template_file.name)

    # Extract kept nodes and pruned matrix
    kept_indices = result.kept_indices
    pruned_matrix = result.pruned_matrix
    graph = result.graph
    str_tokens = result.str_tokens

    # Clean up str_tokens for display
    display_tokens = []
    for tok in str_tokens:
        for k, v in STR_TOKENS_MAP.items():
            tok = tok.replace(k, v)
        display_tokens.append(tok)

    kept_nodes = [graph.nodes[i] for i in kept_indices]
    n_layers = graph.n_layers
    seq_len = graph.seq_len

    slug = f"attribution-graph-{n_layers}l"

    # Build nodes list for the frontend
    nodes_json = []
    for node in kept_nodes:
        # Build clerp (human-readable label)
        if node.node_type.value == "logit":
            clerp = f'output: "{node.str_token}" (p={node.token_prob:.3f})'
        elif node.node_type.value == "embedding":
            tok_display = node.str_token
            for k, v in STR_TOKENS_MAP.items():
                tok_display = tok_display.replace(k, v)
            clerp = f'Emb: "{tok_display}"'
        elif node.node_type.value == "mlp_error":
            clerp = f"MLP error L{node.layer}"
        else:
            clerp = node.label or f"L{node.layer} F{node.feature}"

        # Compute reverse_ctx_idx
        reverse_ctx_idx = seq_len - node.ctx_idx

        # Map layer to what the frontend expects
        if node.node_type.value == "embedding":
            display_layer = "E"
        elif node.node_type.value == "logit":
            display_layer = n_layers
        else:
            display_layer = node.layer + 1  # 1-indexed for display

        node_id = f"{display_layer}_{node.feature}_{node.ctx_idx}"
        js_node_id = f"{display_layer}_{node.feature}_-{reverse_ctx_idx}"

        nodes_json.append({
            "clerp": clerp,
            "ctx_idx": node.ctx_idx,
            "reverse_ctx_idx": reverse_ctx_idx,
            "feature": node.feature,
            "is_target_logit": node.node_type.value == "logit" and node.feature == 0,
            "node_type": NODE_TYPE_JS_MAP.get(node.node_type.value, node.node_type.value),
            "token_prob": node.token_prob,
            "feature_density": 0.0,
            "layer": display_layer,
            "node_id": node_id,
            "js_node_id": js_node_id,
            "run_idx": 0,
        })

    # Build links list from pruned matrix
    links_json = []
    edges_mask = pruned_matrix.abs() > 1e-8
    nonzero_indices = t.nonzero(edges_mask)
    for idx in range(len(nonzero_indices)):
        j, i = nonzero_indices[idx]  # j=target, i=source
        weight = pruned_matrix[j, i].item()

        # Get node IDs from the kept_nodes list
        source_node = nodes_json[i.item()]
        target_node = nodes_json[j.item()]

        links_json.append({
            "source": source_node["node_id"],
            "target": target_node["node_id"],
            "weight": round(weight, 5),
        })

    # Prepare metadata
    prompt_formatted = result.prompt
    for k, v in STR_TOKENS_MAP.items():
        prompt_formatted = prompt_formatted.replace(k, v)

    metadata = [{
        "prompt": prompt_formatted,
        "prompt_tokens": display_tokens[1:],  # Skip BOS
        "scan": slug,
        "slug": slug,
        "n_layers": n_layers,
    }]

    # Build the case study data
    case_study_data = {
        "metadata": metadata[0] | {"title_prefix": ""},
        "qParams": {
            "linkType": "both",
            "pinnedIds": [],
            "clickedId": "",
            "supernodes": [],
            "sg_pos": "",
        },
        "nodes": nodes_json,
        "links": links_json,
    }

    graph_data_all = {slug: case_study_data}

    # Generate data.js
    js_str = f"""
window.graphMetadata = {json.dumps(metadata)};
window.graphData = {json.dumps(graph_data_all)};
window.featureData = {json.dumps({})};
"""

    (output_dir / "data.js").write_text(js_str)

    # Generate index.html that loads everything
    script_tags = []
    for js_file in sorted(output_dir.glob("*.js")):
        script_tags.append(f'<script src="{js_file.name}"></script>')

    css_tags = []
    for css_file in sorted(output_dir.glob("*.css")):
        css_tags.append(f'<link rel="stylesheet" href="{css_file.name}">')

    html = f"""<!DOCTYPE html>
<html>
<head>
    <meta charset="utf-8">
    <title>Attribution Graph</title>
    <script src="https://cdn.jsdelivr.net/npm/d3@7"></script>
    <script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
    {chr(10).join(css_tags)}
</head>
<body>
    <div id="container"></div>
    {chr(10).join(script_tags)}
    <script>
        window.rootData = {{}};
        const slug = "{slug}";
        const sel = d3.select('#container');
        window.initCg(sel, slug, {{
            clickedId: null,
            clickedIdCb: () => {{}},
            isModal: false,
            isGridsnap: true,
        }});
    </script>
</body>
</html>"""

    index_path = output_dir / "index.html"
    index_path.write_text(html)

    return index_path
