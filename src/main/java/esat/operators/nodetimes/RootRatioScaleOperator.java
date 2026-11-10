package esat.operators.nodetimes;

import beast.base.core.Description;
import beast.base.evolution.tree.Node;

@Description("Moves the root in the ratio parameterisation of the node heights like the ratio scale operator: it " +
        "scales the height of the root above its oldest tip and recomputes the heights of all other internal nodes " +
        "from their unchanged ratios, so that every node height moves along with the root.")
public class RootRatioScaleOperator extends RatioScaleOperator {

    /** Returns the root. */
    @Override
    protected Node selectNode() {
        return this.tree.getRoot();
    }

}
