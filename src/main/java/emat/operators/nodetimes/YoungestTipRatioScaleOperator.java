package emat.operators.nodetimes;

import beast.base.core.Description;
import beast.base.evolution.tree.Node;

@Description("Moves a random internal node in the parameterisation by the ratio (h - σ) / (h_p - σ) of its height and the " +
        "height of its parent, both above the youngest tip σ below the node. The root is parameterised by its height " +
        "above its youngest tip.")
public class YoungestTipRatioScaleOperator extends AnchoredRatioScaleOperator {

    @Override
    protected double getReferenceHeight(Node node) {
        return this.youngestTipHeights[node.getNr()];
    }

    @Override
    protected boolean isAnchoredAtRoot() {
        return false;
    }

}
