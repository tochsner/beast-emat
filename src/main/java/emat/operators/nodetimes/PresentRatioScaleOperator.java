package emat.operators.nodetimes;

import beast.base.core.Description;
import beast.base.evolution.tree.Node;

@Description("Moves a random internal node in the parameterisation by the ratio h / h_p of its height and the height of " +
        "its parent, both measured from the most recent time, which is independent of the sampling times of the tips. " +
        "The root is parameterised by its height.")
public class PresentRatioScaleOperator extends AnchoredRatioScaleOperator {

    @Override
    protected double getReferenceHeight(Node node) {
        return 0.0;
    }

    @Override
    protected boolean isAnchoredAtRoot() {
        return false;
    }

}
