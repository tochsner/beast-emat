package emat.helper;

import beast.base.evolution.tree.Node;
import emat.state.Mutation;
import emat.state.Mutations;

import java.util.ArrayList;
import java.util.List;
import java.util.function.Function;

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
     * The changes are written into the given map, which is cleared first, and map every site
     * that mutates on the path to its state at the ancestor (start) and at the point (end).
     */
    public static void collectChanges(Mutations mutations, Node ancestor, Node node, double height, SiteChanges changes) {
        collectChanges(mutations::getMutations, ancestor, node, height, changes);
    }

    /**
     * Collects the changes like collectChanges(Mutations, ...), with the mutations above
     * every node given by the given function.
     */
    public static void collectChanges(Function<Node, List<Mutation>> mutationsAboveNode, Node ancestor, Node node, double height,
                                      SiteChanges changes) {
        changes.clear();

        for (Node branchNode = node; branchNode != ancestor; branchNode = branchNode.getParent()) {
            List<Mutation> branchMutations = mutationsAboveNode.apply(branchNode);

            // the mutations are sorted by descending height, so the ones below the point are at the end
            int i = branchMutations.size() - 1;
            while (i >= 0 && branchMutations.get(i).time() < height) {
                i--;
            }

            // walk upwards, so the first mutation seen at a site sets its state at the point and the last one at the ancestor
            for (; i >= 0; i--) {
                Mutation mutation = branchMutations.get(i);
                int slot = changes.getSlot(mutation.site());
                if (slot < 0) {
                    changes.addSite(mutation.site(), mutation.oldState(), mutation.newState());
                } else {
                    changes.setStartState(slot, mutation.oldState());
                }
            }
        }
    }

    /**
     * Combines the changes along the two paths from a common ancestor down to the start and
     * down to the end into the sites whose states differ between start and end, which are
     * written into the given map with their states at the start and at the end. The map is
     * cleared first and must differ from both inputs. A site that changes on only one path
     * keeps the state of the common ancestor on the other.
     */
    public static void combineChanges(SiteChanges startChanges, SiteChanges endChanges, SiteChanges differingSites) {
        differingSites.clear();

        // sites that change on the start path, and possibly on the end path

        for (int slot = 0; slot < startChanges.getSize(); slot++) {
            int site = startChanges.getSite(slot);
            int startState = startChanges.getEndState(slot);

            int endSlot = endChanges.getSlot(site);
            int endState = endSlot >= 0 ? endChanges.getEndState(endSlot) : startChanges.getStartState(slot);

            if (startState != endState) {
                differingSites.addSite(site, startState, endState);
            }
        }

        // sites that change on the end path only

        for (int slot = 0; slot < endChanges.getSize(); slot++) {
            int site = endChanges.getSite(slot);
            if (startChanges.containsSite(site)) {
                continue;
            }

            int startState = endChanges.getStartState(slot);
            int endState = endChanges.getEndState(slot);

            if (startState != endState) {
                differingSites.addSite(site, startState, endState);
            }
        }
    }

    /** Counts the sites whose states differ between the start and the end of the given changes. */
    public static int countDifferingSites(SiteChanges changes) {
        int numDifferingSites = 0;
        for (int slot = 0; slot < changes.getSize(); slot++) {
            if (changes.getStartState(slot) != changes.getEndState(slot)) {
                numDifferingSites++;
            }
        }
        return numDifferingSites;
    }

    /**
     * Collects the sequence at the given node as the sites that mutate on the path from the
     * root down to it, with their state at the node as end state. Sites without an entry
     * keep the state of the reference sequence. The changes are written into the given map,
     * which is cleared first, and their start states are not meaningful. Its cost is the
     * number of mutations on the path rather than the genome length.
     */
    public static void collectSequence(Mutations mutations, Node node, SiteChanges sequence) {
        sequence.clear();

        for (Node branchNode = node; branchNode != null; branchNode = branchNode.getParent()) {
            List<Mutation> branchMutations = mutations.getMutations(branchNode);

            // walk upwards, so the first mutation seen at a site sets its state at the node
            for (int i = branchMutations.size() - 1; i >= 0; i--) {
                Mutation mutation = branchMutations.get(i);
                if (!sequence.containsSite(mutation.site())) {
                    sequence.addSite(mutation.site(), mutation.oldState(), mutation.newState());
                }
            }
        }
    }

    /**
     * Returns the state of the given site at the given node, which is set by the lowest
     * mutation at the site on the path from the node up to the root, or by the reference
     * sequence if there is none.
     */
    public static int getState(Mutations mutations, Node node, int site) {
        for (Node branchNode = node; branchNode != null; branchNode = branchNode.getParent()) {
            List<Mutation> branchMutations = mutations.getMutations(branchNode);

            // the mutations are sorted by descending height, so walk upwards from the end
            for (int i = branchMutations.size() - 1; i >= 0; i--) {
                Mutation mutation = branchMutations.get(i);
                if (mutation.site() == site) {
                    return mutation.newState();
                }
            }
        }

        return mutations.getReferenceSequence()[site];
    }

}
