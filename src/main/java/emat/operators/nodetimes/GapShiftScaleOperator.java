package emat.operators.nodetimes;

import beast.base.core.Description;
import beast.base.evolution.tree.Node;

@Description("Moves a random internal node in the gap parameterisation, where every internal node other than the root " +
        "is parameterised by the gap h_p - h to its parent, and the root by its height above its oldest tip. The gap " +
        "of the selected node is scaled by a factor from a Bactrian kernel. The gaps below it stay the same, so the " +
        "internal nodes below it shift by the same amount in time.")
public class GapShiftScaleOperator extends ReparamScaleOperator {

    /**
     * Scales the gap g = h_p - h of the selected node by the given factor, which has the Jacobian s, and
     * shifts the internal nodes below it by the change of its height. The map from the gaps to the heights
     * is a translation of the subtree, so it has no further Jacobian.
     */
    @Override
    protected double proposeHeights(Node selectedNode, double scaler) {
        double oldHeight = selectedNode.getHeight();
        double newHeight;
        if (selectedNode.isRoot()) {
            double oldestTipHeight = this.oldestTipHeights[selectedNode.getNr()];
            newHeight = oldestTipHeight + (oldHeight - oldestTipHeight) * scaler;
        } else {
            double parentHeight = selectedNode.getParent().getHeight();
            newHeight = parentHeight - (parentHeight - oldHeight) * scaler;
        }

        double shift = newHeight - oldHeight;
        for (Node node : this.affectedNodes) {
            this.newHeights[node.getNr()] = node.getHeight() + shift;
        }

        return Math.log(scaler);
    }

}
