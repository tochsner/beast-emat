package emat.operators.nodetimes;

import beast.base.core.Description;
import beast.base.core.Input;
import beast.base.evolution.tree.Node;
import beast.base.inference.Scalable;

import java.util.ArrayList;
import java.util.List;

@Description("Scales the intervals between every internal node and its older child by a common factor, " +
        "and maps every mutation linearly within its branch. Unlike scaling all heights, this keeps the tree " +
        "and its mutations valid even for tips sampled through time " +
        "(https://www.biorxiv.org/content/10.1101/2025.06.18.660471v1).")
public class IntervalScaleOperator extends ScaleTunedOperator {

    final public Input<Double> scaleUpperLimit = new Input<>("upper", "upper limit of the scale factor", 1.0 - 1e-8);
    final public Input<Double> scaleLowerLimit = new Input<>("lower", "lower limit of the scale factor", 1e-8);
    final public Input<Double> scaleFactorInput = new Input<>("scaleFactor", "scaling factor: range from 0 to 1. Close to zero is very large jumps, close to 1.0 is very small jumps.", 0.1);
    final public Input<List<Scalable>> downInput = new Input<>("down", "parameters to scale inversely to the tree length", new ArrayList<>());
    final public Input<List<Scalable>> upInput = new Input<>("up", "parameters to scale with the tree length", new ArrayList<>());

    @Override
    public void initAndValidate() {
        super.initAndValidate();

        this.scaleFactor = this.scaleFactorInput.get();
        this.upper = this.scaleUpperLimit.get();
        this.lower = this.scaleLowerLimit.get();
    }

    /**
     * Scales the interval between every internal node and its older child by the same
     * factor s, which is a map with Jacobian s^k for k internal nodes. Every mutation then
     * keeps its relative position on its branch, which scales its time by the ratio of the
     * new to the old branch length and adds that ratio to the Jacobian once per mutation.
     */
    @Override
    public double proposal() {
        double scaler = this.kernelDistribution.getScaler(0, this.getCoercableParameterValue());

        // remember the old heights to map the mutations of every branch, and the old tree length

        Node[] nodes = this.tree.getNodesAsArray();
        double[] oldHeights = new double[nodes.length];
        double lengthBefore = 0.0;
        for (Node node : nodes) {
            oldHeights[node.getNr()] = node.getHeight();
            lengthBefore += node.getLength();
        }

        int numScaledNodes = this.scaleIntervals(this.tree.getRoot(), scaler);
        double lengthAfter = NodeTimes.computeTreeLength(this.tree);

        double logHastingsRatio = numScaledNodes * Math.log(scaler);

        // move the mutations of every non-root branch along with its ends

        logHastingsRatio += BranchTimeMapper.mapMutations(this.mutations, nodes, oldHeights, this);

        // scale the other parameters with the tree length

        double actualScaler = lengthAfter / lengthBefore;

        // scale returns the log Jacobian, i.e. dimension * log(factor)

        try {
            for (Scalable down : this.downInput.get()) {
                logHastingsRatio += down.scale(1.0 / actualScaler);
            }
            for (Scalable up : this.upInput.get()) {
                logHastingsRatio += up.scale(actualScaler);
            }
        } catch (IllegalArgumentException e) {
            // a scalable rejected the move
            return Double.NEGATIVE_INFINITY;
        }

        return logHastingsRatio;
    }

    /**
     * Scales the interval between the given node and its older child by the given factor,
     * after scaling everything below it. Returns the number of scaled nodes.
     */
    private int scaleIntervals(Node node, double scaler) {
        if (node.isLeaf()) {
            return 0;
        }

        double oldInterval = node.getHeight() - Math.max(node.getLeft().getHeight(), node.getRight().getHeight());

        int numScaledNodes = 1;
        numScaledNodes += this.scaleIntervals(node.getLeft(), scaler);
        numScaledNodes += this.scaleIntervals(node.getRight(), scaler);

        double minHeight = Math.max(node.getLeft().getHeight(), node.getRight().getHeight());
        node.setHeight(minHeight + oldInterval * scaler);

        return numScaledNodes;
    }

}
