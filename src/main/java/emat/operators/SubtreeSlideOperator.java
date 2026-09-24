package emat.operators;

import beast.base.core.Description;
import beast.base.core.Input;
import beast.base.evolution.tree.Node;
import beast.base.util.Randomizer;

import java.util.ArrayList;
import java.util.List;

@Description("Slides the parent of a random subtree up or down the tree by a Gaussian step, regrafting the " +
        "subtree where it lands (docs/mcmc-moves.md §8.1). The root is never changed.")
public class SubtreeSlideOperator extends SubtreePruneRegraftOperator {

    final public Input<Double> sizeInput = new Input<>("size", "the standard deviation of the height step", 1.0);
    final public Input<Boolean> optimiseInput = new Input<>("optimise", "whether to tune the size towards the target acceptance probability", true);

    double size;

    @Override
    public void initAndValidate() {
        super.initAndValidate();
        this.size = this.sizeInput.get();
    }

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
     * Displaces the height of P by δ ~ N(0, size²) along the pruned tree. Going up, the new
     * sibling is the unique lineage above S that is active at the new height, and the move
     * is rejected if it passes the root. Going down, it is chosen uniformly among the K
     * lineages below P that are active at the new height, and the move is rejected if the
     * new height is not above X or K = 0. The reverse of a move up chooses among the K'
     * lineages below P' at the old height, so the Hastings ratio is K going down and 1/K'
     * going up.
     */
    @Override
    protected GraftingPoint proposeGraftingPoint(Node x) {
        Node parent = x.getParent();
        Node sibling = this.getOtherChild(parent, x);

        double oldParentHeight = parent.getHeight();
        double newParentHeight = oldParentHeight + Randomizer.nextGaussian() * this.size;

        if (newParentHeight > oldParentHeight) {
            // walk up the lineage above S, where the old grandparent is S's parent in the pruned tree

            Node newSibling = sibling;
            Node newGrandparent = parent.getParent();

            while (newGrandparent != null && newGrandparent.getHeight() < newParentHeight) {
                newSibling = newGrandparent;
                newGrandparent = newGrandparent.getParent();
            }

            if (newGrandparent == null) {
                // P' would become the root
                return null;
            }

            List<Node> reverseLineages = new ArrayList<>();
            this.collectLineages(newSibling, oldParentHeight, x, reverseLineages);

            return new GraftingPoint(newSibling, newParentHeight, -Math.log(reverseLineages.size()));
        }

        if (newParentHeight <= x.getHeight()) {
            return null;
        }

        List<Node> lineages = new ArrayList<>();
        this.collectLineages(sibling, newParentHeight, x, lineages);

        if (lineages.isEmpty()) {
            return null;
        }

        Node newSibling = lineages.get(Randomizer.nextInt(lineages.size()));
        return new GraftingPoint(newSibling, newParentHeight, Math.log(lineages.size()));
    }

    /**
     * Collects the lineages of the pruned tree below the given node that are active at the
     * given height, assuming that the branch above the node starts above that height. In the
     * pruned tree, X and its parent P are removed and S takes the place of P.
     */
    private void collectLineages(Node node, double height, Node x, List<Node> lineages) {
        if (node.getHeight() <= height) {
            lineages.add(node);
            return;
        }

        Node parent = x.getParent();
        for (Node child : node.getChildren()) {
            // skip over P to its other child, which drops the subtree below X
            Node prunedChild = child == parent ? this.getOtherChild(parent, x) : child;
            this.collectLineages(prunedChild, height, x, lineages);
        }
    }

    /* Tuning */

    @Override
    public void optimize(double logAlpha) {
        if (this.optimiseInput.get()) {
            this.size = Math.exp(this.calcDelta(logAlpha) + Math.log(this.size));
        }
    }

    @Override
    public double getCoercableParameterValue() {
        return this.size;
    }

    @Override
    public void setCoercableParameterValue(double value) {
        this.size = value;
    }

}
