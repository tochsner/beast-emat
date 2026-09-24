package emat.operators;

import beast.base.core.Description;
import beast.base.evolution.tree.Node;
import beast.base.util.Randomizer;

import java.util.ArrayList;
import java.util.List;

@Description("Regrafts a random subtree onto a uniformly chosen branch at a uniform height " +
        "(Wilson–Balding, docs/mcmc-moves.md §8.2). The root is never changed.")
public class WilsonBaldingOperator extends SubtreePruneRegraftOperator {

    /**
     * Picks X uniformly among all nodes where neither X nor its parent is the root. This set
     * does not change under moves that keep the root, so the choice is symmetric.
     */
    @Override
    protected Node pickSubtreeRoot() {
        List<Node> candidates = new ArrayList<>();
        for (Node node : this.tree.getNodesAsArray()) {
            if (!node.isRoot() && !node.getParent().isRoot()) {
                candidates.add(node);
            }
        }

        if (candidates.isEmpty()) {
            return null;
        }
        return candidates.get(Randomizer.nextInt(candidates.size()));
    }

    /**
     * Picks S' uniformly among all nodes and draws t_P' uniformly between max(t_X, t_S') and
     * the height of G', the parent of S' in the pruned tree. Invalid choices, i.e. P, the
     * root, a node in the subtree below X, or a branch that ends before X, are rejected.
     * The reverse move picks S among the same number of nodes, so the Hastings ratio is the
     * ratio of the new to the old height range.
     */
    @Override
    protected GraftingPoint proposeGraftingPoint(Node x) {
        Node parent = x.getParent();
        Node sibling = this.getOtherChild(parent, x);

        Node[] nodes = this.tree.getNodesAsArray();
        Node newSibling = nodes[Randomizer.nextInt(nodes.length)];

        if (newSibling == parent || newSibling.isRoot() || newSibling == x) {
            return null;
        }

        double newMinHeight = Math.max(x.getHeight(), newSibling.getHeight());
        double newRange = this.getPrunedParent(x, newSibling).getHeight() - newMinHeight;

        double oldMinHeight = Math.max(x.getHeight(), sibling.getHeight());
        double oldRange = parent.getParent().getHeight() - oldMinHeight;

        if (newRange <= 0.0 || oldRange <= 0.0) {
            // this also rejects every node below X, as their branches end before X
            return null;
        }

        double newParentHeight = newMinHeight + Randomizer.nextDouble() * newRange;
        return new GraftingPoint(newSibling, newParentHeight, Math.log(newRange / oldRange));
    }

}
