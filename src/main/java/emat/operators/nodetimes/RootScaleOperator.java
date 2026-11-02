package emat.operators.nodetimes;

import beast.base.core.Description;
import beast.base.core.Input;
import beast.base.evolution.tree.Node;
import emat.diagnostics.DiagnosedOperator;
import emat.diagnostics.StepAnnotations;

@Description("Scales the shorter of the two root branches by a factor drawn from a Bactrian kernel, which " +
        "moves the root height. The mutations on the branches below the root move linearly along with the root. " +
        "Missations are not supported.")
public class RootScaleOperator extends ScaleTunedOperator implements DiagnosedOperator {

    final public Input<Double> scaleUpperLimit = new Input<>("upper", "upper limit of the scale factor", 10.0);
    final public Input<Double> scaleLowerLimit = new Input<>("lower", "lower limit of the scale factor", 1e-8);
    final public Input<Double> scaleFactorInput = new Input<>("scaleFactor", "width of the kernel in log space: close to zero is very small jumps, larger is larger jumps.", 0.1);

    @Override
    public void initAndValidate() {
        super.initAndValidate();

        this.scaleFactor = this.scaleFactorInput.get();
        this.upper = this.scaleUpperLimit.get();
        this.lower = this.scaleLowerLimit.get();
    }

    /**
     * Scales the shorter root branch h - h_m by a factor s, where h is the root height and h_m
     * the height of the higher child, which maps h to h_m + (h - h_m) s with Jacobian s. Every
     * mutation on the branch above a child c keeps its relative position on the branch.
     */
    @Override
    public double proposal() {
        Node root = this.tree.getRoot();
        if (root.isLeaf()) {
            return Double.NEGATIVE_INFINITY;
        }

        double oldHeight = root.getHeight();
        double minHeight = this.computeMinHeight(root);
        double scaler = this.kernelDistribution.getScaler(root.getNr(), oldHeight, this.getCoercableParameterValue());

        // every positive scaler keeps the root above its children
        double newHeight = minHeight + (oldHeight - minHeight) * scaler;
        StepAnnotations.set(0, Math.log(scaler));
        StepAnnotations.set(1, oldHeight);
        StepAnnotations.set(2, minHeight);
        StepAnnotations.set(3, this.getCoercableParameterValue());
        root.setHeight(newHeight);

        double logHastingsRatio = Math.log(scaler);
        for (Node child : root.getChildren()) {
            logHastingsRatio += BranchTimeMapper.mapBranch(
                    this.mutations, child, oldHeight, child.getHeight(), newHeight, child.getHeight(), this
            );
        }

        return logHastingsRatio;
    }

    @Override
    public String[] getAnnotationNames() {
        return new String[]{"logScaler", "oldHeight", "minHeight", "width"};
    }

    /** Returns the lowest height the root can take, which is above each child. */
    private double computeMinHeight(Node root) {
        double minHeight = Double.NEGATIVE_INFINITY;
        for (Node child : root.getChildren()) {
            minHeight = Math.max(minHeight, child.getHeight());
        }
        return minHeight;
    }

}
