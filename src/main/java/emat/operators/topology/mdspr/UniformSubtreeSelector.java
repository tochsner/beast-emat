package emat.operators.topology.mdspr;

import beast.base.evolution.tree.Node;
import beast.base.evolution.tree.Tree;

/** Picks every candidate with the same probability. */
public class UniformSubtreeSelector extends SubtreeSelector {

    public UniformSubtreeSelector(Tree tree) {
        super(tree);
    }

    @Override
    protected double computeWeight(Node node, Node parent) {
        return 1.0;
    }

    /** The number of candidates stays the same under moves that keep the root, so the choice is symmetric. */
    @Override
    public double computeLogProbability(Node x) {
        return 0.0;
    }
}
