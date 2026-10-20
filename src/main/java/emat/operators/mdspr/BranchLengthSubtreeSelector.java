package emat.operators.mdspr;

import beast.base.evolution.tree.Node;
import beast.base.evolution.tree.Tree;

/**
 * Picks candidates in proportion to the length of the branch above them. A move changes the
 * lengths of the branches around the old and the new attachment point.
 */
public class BranchLengthSubtreeSelector extends SubtreeSelector {

    public BranchLengthSubtreeSelector(Tree tree) {
        super(tree);
    }

    @Override
    protected double computeWeight(Node node) {
        return node.getLength();
    }
}
