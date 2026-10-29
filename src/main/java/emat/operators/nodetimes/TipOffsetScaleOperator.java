package emat.operators.nodetimes;

import beast.base.core.Description;
import beast.base.evolution.tree.Node;

@Description("Moves a random internal node in the offset parameterisation, where every internal node is parameterised " +
        "by its height above the oldest tip below it. The offset of the selected node is scaled by a factor from a " +
        "Bactrian kernel. The offsets of the other nodes stay the same, so they keep their heights and only the " +
        "selected node moves.")
public class TipOffsetScaleOperator extends ReparamScaleOperator {

    @Override
    protected void selectAffectedNodes(Node selectedNode) {
        this.addAffectedNode(selectedNode);
    }

    /** Scales the offset x = h - τ by the given factor, which has the Jacobian s. */
    @Override
    protected double proposeHeights(Node selectedNode, double scaler) {
        double oldestTipHeight = this.oldestTipHeights[selectedNode.getNr()];
        double newHeight = oldestTipHeight + (selectedNode.getHeight() - oldestTipHeight) * scaler;
        if (!(newHeight > oldestTipHeight)) {
            return Double.NEGATIVE_INFINITY;
        }

        this.newHeights[selectedNode.getNr()] = newHeight;
        return Math.log(scaler);
    }

}
