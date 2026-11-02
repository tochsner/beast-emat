package emat.operators.nodetimes;

import beast.base.core.Description;
import beast.base.core.Input;
import beast.base.evolution.tree.Node;
import emat.diagnostics.DiagnosedOperator;
import emat.diagnostics.StepAnnotations;

@Description("Scales the intervals between every internal node and its older child by a common factor, " +
        "and maps every mutation linearly within its branch. Unlike scaling all heights, this keeps the tree " +
        "and its mutations valid even for tips sampled through time " +
        "(https://www.biorxiv.org/content/10.1101/2025.06.18.660471v1).")
public class IntervalScaleOperator extends CompensatedScaleOperator implements DiagnosedOperator {

    final public Input<Double> scaleUpperLimit = new Input<>("upper", "upper limit of the scale factor", 1.0 - 1e-8);
    final public Input<Double> scaleLowerLimit = new Input<>("lower", "lower limit of the scale factor", 1e-8);
    final public Input<Double> scaleFactorInput = new Input<>("scaleFactor", "scaling factor: range from 0 to 1. Close to zero is very large jumps, close to 1.0 is very small jumps.", 0.1);

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
        double scaler = this.drawScaler();

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
        if (!this.isTreeLengthAllowed(lengthAfter)) {
            return Double.NEGATIVE_INFINITY;
        }

        double logHastingsRatio = numScaledNodes * Math.log(scaler);
        StepAnnotations.set(0, Math.log(scaler));
        StepAnnotations.set(1, numScaledNodes);
        StepAnnotations.set(2, Math.log(lengthAfter / lengthBefore));
        StepAnnotations.set(3, lengthBefore);
        StepAnnotations.set(4, this.getCoercableParameterValue());

        // move the mutations of every non-root branch along with its ends

        logHastingsRatio += BranchTimeMapper.mapMutations(this.mutations, nodes, oldHeights, this);

        // scale the other parameters with the tree length

        double logJacobian = this.scaleCompensatedParameters(lengthBefore, lengthAfter);
        if (logJacobian == Double.NEGATIVE_INFINITY) {
            return Double.NEGATIVE_INFINITY;
        }
        logHastingsRatio += logJacobian;

        return logHastingsRatio;
    }

    /** Draws the factor by which the intervals are scaled; its logarithm is symmetric around zero. */
    protected double drawScaler() {
        return this.kernelDistribution.getScaler(0, this.getCoercableParameterValue());
    }

    /** Returns whether the tree length after the scaling lies in the support of the move; the proposal is rejected otherwise. */
    protected boolean isTreeLengthAllowed(double treeLength) {
        return true;
    }

    @Override
    public String[] getAnnotationNames() {
        return new String[]{"logScaler", "numScaledNodes", "logTreeLengthScaler", "treeLengthBefore", "width"};
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
