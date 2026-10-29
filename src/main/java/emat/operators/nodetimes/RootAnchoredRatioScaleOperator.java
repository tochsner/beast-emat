package emat.operators.nodetimes;

import beast.base.core.Description;
import beast.base.evolution.tree.Node;

@Description("Moves a random internal node in the parameterisation by the ratio (h - τ) / (h_r - τ) of its height and the " +
        "height of the root, both above the oldest tip τ below the node. The root is parameterised by its height above " +
        "its oldest tip. A move of any other node changes only that node, and a move of the root rescales all nodes.")
public class RootAnchoredRatioScaleOperator extends AnchoredRatioScaleOperator {

    @Override
    protected double getReferenceHeight(Node node) {
        return this.oldestTipHeights[node.getNr()];
    }

    @Override
    protected boolean isAnchoredAtRoot() {
        return true;
    }

}
