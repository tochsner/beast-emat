package emat.operators;

import beast.base.core.Description;
import beast.base.core.Input;
import beast.base.evolution.operator.TreeOperator;
import beast.base.evolution.tree.Node;
import beast.base.evolution.tree.Tree;
import beast.base.inference.operator.kernel.KernelDistribution;
import emat.state.Mutation;
import emat.state.Mutations;

import java.text.DecimalFormat;
import java.util.ArrayList;
import java.util.List;

@Description("Scales the root height by a factor drawn from a Bactrian kernel, like a BactrianScaleOperator " +
        "with rootOnly set to true. The mutations on the branches below the root move linearly along with the " +
        "root, so they neither block the root nor have to follow it one at a time.")
public class RootScaleOperator extends TreeOperator {

    final public Input<Mutations> mutationsInput = new Input<>("mutations", "the mutations to operate on", Input.Validate.REQUIRED);
    final public Input<KernelDistribution> kernelDistributionInput = new Input<>("kernelDistribution", "provides sample distribution for proposals",
            KernelDistribution.newDefaultKernelDistribution());
    final public Input<Double> scaleUpperLimit = new Input<>("upper", "upper limit of the scale factor", 10.0);
    final public Input<Double> scaleLowerLimit = new Input<>("lower", "lower limit of the scale factor", 1e-8);
    final public Input<Double> scaleFactorInput = new Input<>("scaleFactor", "scaling factor: range from 0 to 1. Close to zero is very large jumps, close to 1.0 is very small jumps.", 0.1);
    final public Input<Boolean> optimiseInput = new Input<>("optimise", "whether to tune the scale factor towards the target acceptance probability", true);

    Mutations mutations;
    Tree tree;
    KernelDistribution kernelDistribution;

    double scaleFactor;
    double upper;
    double lower;

    @Override
    public void initAndValidate() {
        this.mutations = this.mutationsInput.get();
        this.tree = this.treeInput.get();
        this.kernelDistribution = this.kernelDistributionInput.get();

        this.scaleFactor = this.scaleFactorInput.get();
        this.upper = this.scaleUpperLimit.get();
        this.lower = this.scaleLowerLimit.get();
    }

    /**
     * Scales the root height h by a factor s, which is a map with Jacobian s. Every mutation
     * on the branch above a child c keeps its relative position on the branch, which scales
     * its height above h_c by the ratio of the new to the old branch length and adds that
     * ratio to the Jacobian once per mutation.
     *
     * Keeping the mutation times instead would pin the root just above the highest of them,
     * as the genetic prior decays like exp(-2 λ h) above it. The root and that mutation
     * could then only move together in steps of about 1 / (2 λ).
     */
    @Override
    public double proposal() {
        Node root = this.tree.getRoot();
        if (root.isLeaf()) {
            return Double.NEGATIVE_INFINITY;
        }

        double oldHeight = root.getHeight();
        double scaler = this.kernelDistribution.getScaler(root.getNr(), oldHeight, this.getCoercableParameterValue());
        double newHeight = oldHeight * scaler;

        if (newHeight <= this.computeMinHeight(root)) {
            return Double.NEGATIVE_INFINITY;
        }

        root.setHeight(newHeight);

        double logHastingsRatio = Math.log(scaler);
        for (Node child : root.getChildren()) {
            logHastingsRatio += this.scaleBranch(child, oldHeight, newHeight);
        }

        return logHastingsRatio;
    }

    /** Returns the lowest height the root can take, which is above each child. */
    private double computeMinHeight(Node root) {
        double minHeight = Double.NEGATIVE_INFINITY;
        for (Node child : root.getChildren()) {
            minHeight = Math.max(minHeight, child.getHeight());
        }
        return minHeight;
    }

    /**
     * Maps the mutations on the branch above the given node linearly from the old to the
     * new branch start. Returns the log Jacobian of the map, which is the log of the ratio
     * of the new to the old branch length once per mutation.
     */
    private double scaleBranch(Node node, double oldStartHeight, double newStartHeight) {
        List<Mutation> branchMutations = this.mutations.getMutations(node);
        if (branchMutations.isEmpty()) {
            return 0.0;
        }

        double endHeight = node.getHeight();
        double lengthRatio = (newStartHeight - endHeight) / (oldStartHeight - endHeight);

        List<Mutation> newBranchMutations = new ArrayList<>(branchMutations.size());
        for (Mutation mutation : branchMutations) {
            newBranchMutations.add(new Mutation(
                    mutation.nodeNr(),
                    this.mapHeight(mutation.time(), oldStartHeight, newStartHeight, endHeight, lengthRatio),
                    this.mapHeight(mutation.timeOfPreviousMutation(), oldStartHeight, newStartHeight, endHeight, lengthRatio),
                    mutation.site(), mutation.oldState(), mutation.newState()
            ));
        }

        this.mutations.applyMutations(node, newBranchMutations, this);
        return branchMutations.size() * Math.log(lengthRatio);
    }

    /**
     * Maps a height on a branch linearly from its old to its new start, keeping its end. The
     * start maps exactly onto the new start, which keeps the times of previous mutations at
     * the branch start consistent, and rounding cannot push a height off the branch.
     */
    private double mapHeight(double height, double oldStartHeight, double newStartHeight, double endHeight, double lengthRatio) {
        if (height == oldStartHeight) {
            return newStartHeight;
        }

        double newHeight = endHeight + (height - endHeight) * lengthRatio;
        return Math.min(Math.max(newHeight, endHeight), newStartHeight);
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
