package esat.operators.nodetimes;

import beast.base.core.Description;
import beast.base.core.Input;
import beast.base.evolution.operator.TreeOperator;
import beast.base.evolution.tree.Node;
import beast.base.evolution.tree.Tree;
import beast.base.inference.operator.kernel.KernelDistribution;

import java.text.DecimalFormat;

@Description("Scales the shorter of the two root branches by a factor drawn from a Bactrian kernel, which " +
        "moves the root height. The node states do not depend on the heights, so nothing else changes.")
public class RootScaleOperator extends TreeOperator {

    final public Input<KernelDistribution> kernelDistributionInput = new Input<>("kernelDistribution", "provides sample distribution for proposals",
            KernelDistribution.newDefaultKernelDistribution());
    final public Input<Boolean> optimiseInput = new Input<>("optimise", "whether to tune the scale factor towards the target acceptance probability", true);
    final public Input<Double> scaleUpperLimit = new Input<>("upper", "upper limit of the scale factor", 10.0);
    final public Input<Double> scaleLowerLimit = new Input<>("lower", "lower limit of the scale factor", 1e-8);
    final public Input<Double> scaleFactorInput = new Input<>("scaleFactor", "width of the kernel in log space: close to zero is very small jumps, larger is larger jumps.", 0.1);

    private Tree tree;
    private KernelDistribution kernelDistribution;

    private double scaleFactor;
    private double upper;
    private double lower;

    @Override
    public void initAndValidate() {
        this.tree = this.treeInput.get();
        this.kernelDistribution = this.kernelDistributionInput.get();

        this.scaleFactor = this.scaleFactorInput.get();
        this.upper = this.scaleUpperLimit.get();
        this.lower = this.scaleLowerLimit.get();
    }

    /**
     * Scales the shorter root branch h - h_m by a factor s, where h is the root height and h_m
     * the height of the higher child, which maps h to h_m + (h - h_m) s with Jacobian s.
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
        root.setHeight(minHeight + (oldHeight - minHeight) * scaler);

        return Math.log(scaler);
    }

    /** Returns the lowest height the root can take, which is above each child. */
    private double computeMinHeight(Node root) {
        double minHeight = Double.NEGATIVE_INFINITY;
        for (Node child : root.getChildren()) {
            minHeight = Math.max(minHeight, child.getHeight());
        }
        return minHeight;
    }

    /* Tuning */

    @Override
    public void optimize(double logAlpha) {
        if (this.optimiseInput.get()) {
            double delta = this.calcDelta(logAlpha) + Math.log(this.getCoercableParameterValue());
            this.setCoercableParameterValue(Math.exp(delta));
        }
    }

    @Override
    public double getTargetAcceptanceProbability() {
        return 0.3;
    }

    @Override
    public double getCoercableParameterValue() {
        return this.scaleFactor;
    }

    @Override
    public void setCoercableParameterValue(double value) {
        this.scaleFactor = Math.max(Math.min(value, this.upper), this.lower);
    }

    @Override
    public String getPerformanceSuggestion() {
        double acceptanceProbability = this.m_nNrAccepted / (this.m_nNrAccepted + this.m_nNrRejected + 0.0);
        double ratio = acceptanceProbability / this.getTargetAcceptanceProbability();
        ratio = Math.max(Math.min(ratio, 2.0), 0.5);

        DecimalFormat formatter = new DecimalFormat("#.###");
        if (acceptanceProbability < 0.10 || acceptanceProbability > 0.40) {
            return "Try setting scale factor to about " + formatter.format(this.getCoercableParameterValue() * ratio);
        }
        return "";
    }

}
