package emat.helper;

import beast.base.evolution.tree.Node;
import emat.state.Mutation;
import emat.state.Mutations;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.HashSet;
import java.util.List;
import java.util.Map;
import java.util.Set;

/**
 * Reconstructs states along paths of an EMAT: the sites that differ between two points of
 * the tree, found from the mutations on the path between them, and full sequences.
 */
public final class MutationPaths {

    private MutationPaths() {
    }

    /**
     * Returns the most recent common ancestor of the two given nodes, which may be one of
     * them. This steps up the lower of the two lineages until they meet, so its cost is the
     * length of the path between the nodes rather than their depth.
     */
    public static Node findMrca(Node node, Node otherNode) {
        while (node != otherNode) {
            boolean isNodeLower = node.getHeight() < otherNode.getHeight()
                    || (node.getHeight() == otherNode.getHeight() && !isAncestorAtSameHeight(node, otherNode));

            if (isNodeLower) {
                node = node.getParent();
            } else {
                otherNode = otherNode.getParent();
            }
        }
        return node;
    }

    /**
     * Checks whether the given ancestor lies above the given node across zero-length
     * branches only. On a height tie, this tells which node must not be stepped up.
     */
    private static boolean isAncestorAtSameHeight(Node ancestor, Node node) {
        for (Node parent = node.getParent(); parent != null && parent.getHeight() == node.getHeight(); parent = parent.getParent()) {
            if (parent == ancestor) {
                return true;
            }
        }
        return false;
    }

    /**
     * Collects the changes on the path from the given ancestor down to the point at the
     * given height on the branch above the given node, ignoring mutations below that point.
     * The changes map every site that mutates on the path to its state at the ancestor and
     * at the point.
     */
    public static Map<Integer, int[]> collectChanges(Mutations mutations, Node ancestor, Node node, double height) {
        Map<Integer, int[]> changes = new HashMap<>();

        for (Node branchNode = node; branchNode != ancestor; branchNode = branchNode.getParent()) {
            List<Mutation> branchMutations = mutations.getMutations(branchNode);

            // walk upwards, so the first mutation seen at a site sets its state at the point and the last one at the ancestor
            for (int i = branchMutations.size() - 1; i >= 0; i--) {
                Mutation mutation = branchMutations.get(i);
                if (mutation.time() < height) {
                    continue;
                }

                int[] states = changes.computeIfAbsent(mutation.site(), site -> new int[]{mutation.oldState(), mutation.newState()});
                states[0] = mutation.oldState();
            }
        }

        return changes;
    }

    /**
     * Combines the changes along the two paths from a common ancestor down to the start and
     * down to the end into the sites whose states differ between start and end, mapped to
     * their states at the start and at the end. A site that changes on only one path keeps
     * the state of the common ancestor on the other.
     */
    public static Map<Integer, int[]> combineChanges(Map<Integer, int[]> startChanges, Map<Integer, int[]> endChanges) {
        Set<Integer> sites = new HashSet<>(startChanges.keySet());
        sites.addAll(endChanges.keySet());

        Map<Integer, int[]> differingSites = new HashMap<>();

        for (int site : sites) {
            int[] startStates = startChanges.get(site);
            int[] endStates = endChanges.get(site);

            int ancestorState = startStates != null ? startStates[0] : endStates[0];
            int startState = startStates != null ? startStates[1] : ancestorState;
            int endState = endStates != null ? endStates[1] : ancestorState;

            if (startState != endState) {
                differingSites.put(site, new int[]{startState, endState});
            }
        }

        return differingSites;
    }

    /**
     * Returns the full sequence at the given node, starting from the reference sequence and
     * applying all mutations from the root down to the node.
     */
    public static int[] getSequence(Mutations mutations, Node node) {
        List<Node> lineage = new ArrayList<>();
        for (Node ancestor = node; ancestor != null; ancestor = ancestor.getParent()) {
            lineage.add(ancestor);
        }

        int[] sequence = mutations.getReferenceSequence().clone();

        for (int i = lineage.size() - 1; i >= 0; i--) {
            for (Mutation mutation : mutations.getMutations(lineage.get(i))) {
                sequence[mutation.site()] = mutation.newState();
            }
        }

        return sequence;
    }

    /**
     * Returns the full sequence at the start of a path, given the full sequence at its end
     * and the sites that differ between both, mapped to their states at the start and end.
     */
    public static int[] getStartSequence(int[] endSequence, Map<Integer, int[]> differingSites) {
        int[] startSequence = endSequence.clone();
        for (Map.Entry<Integer, int[]> entry : differingSites.entrySet()) {
            startSequence[entry.getKey()] = entry.getValue()[0];
        }
        return startSequence;
    }

}
