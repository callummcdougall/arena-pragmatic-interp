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
