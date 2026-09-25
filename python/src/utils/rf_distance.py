from nexwick_py import CompactTree


def _clade_masks(tree: CompactTree) -> tuple[set[int], int]:
    """Return non-trivial rooted clade masks and the full leaf mask."""
    root_index = tree.root_index()
    masks_by_vertex = [0] * tree.num_vertices()
    clades: set[int] = set()
    stack = [(root_index, False)]

    while stack:
        vertex_index, visited = stack.pop()
        vertex = tree.vertex(vertex_index)

        if visited:
            if vertex.is_leaf():
                if vertex.label_index is None:
                    raise ValueError(f"leaf vertex {vertex.index} has no label index")
                if vertex.label_index < 0:
                    raise ValueError(
                        f"leaf vertex {vertex.index} has a negative label index"
                    )
                masks_by_vertex[vertex_index] = 1 << vertex.label_index
            else:
                children = vertex.children
                if children is None:
                    raise ValueError(f"internal vertex {vertex.index} has no children")

                mask = 0
                for child in children:
                    mask |= masks_by_vertex[child]
                masks_by_vertex[vertex_index] = mask
                if vertex_index != root_index and not _is_single_leaf(mask):
                    clades.add(mask)
        else:
            stack.append((vertex_index, True))
            children = vertex.children
            if children is not None:
                for child in children:
                    stack.append((child, False))

    full_mask = masks_by_vertex[root_index]
    return clades, full_mask


def _is_single_leaf(mask: int) -> bool:
    """Return True when a clade mask contains exactly one leaf."""
    return mask != 0 and mask & (mask - 1) == 0


def get_rf_distance(tree1: CompactTree, tree2: CompactTree) -> float:
    """Return the rooted Robinson-Foulds distance between two trees."""
    tree1_clades, tree1_leaves = _clade_masks(tree1)
    tree2_clades, tree2_leaves = _clade_masks(tree2)

    if tree1_leaves != tree2_leaves:
        raise ValueError("RF distance requires both trees to have the same leaf labels")

    return float(len(tree1_clades.symmetric_difference(tree2_clades)))
