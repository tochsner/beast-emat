package emat.operators;

import beast.base.core.Description;
import beast.base.core.Input;
import beast.base.evolution.operator.TreeOperator;
import beast.base.evolution.tree.Node;
import beast.base.evolution.tree.Tree;
import beast.base.inference.Scalable;
import beast.base.inference.operator.kernel.KernelDistribution;
import emat.state.Mutation;
import emat.state.Mutations;

import java.text.DecimalFormat;
import java.util.ArrayList;
import java.util.List;

@Description("Scales the intervals between every internal node and its older child by a common factor, " +
        "and maps every mutation linearly within its branch. Unlike scaling all heights, this keeps the tree " +
        "and its mutations valid even for tips sampled through time " +
        "(https://www.biorxiv.org/content/10.1101/2025.06.18.660471v1).")
public class IntervalScaleOperator extends TreeOperator {

    final public Input<Mutations> mutationsInput = new Input<>("mutations", "the mutations to operate on", Input.Validate.REQUIRED);
    final public Input<KernelDistribution> kernelDistributionInput = new Input<>("kernelDistribution", "provides sample distribution for proposals",
            KernelDistribution.newDefaultKernelDistribution());
    final public Input<Double> scaleUpperLimit = new Input<>("upper", "upper limit of the scale factor", 1.0 - 1e-8);
    final public Input<Double> scaleLowerLimit = new Input<>("lower", "lower limit of the scale factor", 1e-8);
    final public Input<Double> scaleFactorInput = new Input<>("scaleFactor", "scaling factor: range from 0 to 1. Close to zero is very large jumps, close to 1.0 is very small jumps.", 0.1);
    final public Input<Boolean> optimiseInput = new Input<>("optimise", "whether to tune the scale factor towards the target acceptance probability", true);
    final public Input<List<Scalable>> downInput = new Input<>("down", "parameters to scale inversely to the tree length", new ArrayList<>());
    final public Input<List<Scalable>> upInput = new Input<>("up", "parameters to scale with the tree length", new ArrayList<>());

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
     * Scales the interval between every internal node and its older child by the same
     * factor s, which is a map with Jacobian s^k for k internal nodes. Every mutation then
     * keeps its relative position on its branch, which scales its time by the ratio of the
     * new to the old branch length and adds that ratio to the Jacobian once per mutation.
     */
    @Override
    public double proposal() {
        double scaler = this.kernelDistribution.getScaler(0, this.getCoercableParameterValue());

        // remember the old heights to map the mutations of every branch

        Node[] nodes = this.tree.getNodesAsArray();
        double[] oldHeights = new double[nodes.length];
        for (Node node : nodes) {
            oldHeights[node.getNr()] = node.getHeight();
        }

        double lengthBefore = this.computeTreeLength();
        int numScaledNodes = this.scaleIntervals(this.tree.getRoot(), scaler);
        double lengthAfter = this.computeTreeLength();

        double logHastingsRatio = numScaledNodes * Math.log(scaler);

        // move the mutations of every non-root branch along with its ends

        for (Node node : nodes) {
            List<Mutation> branchMutations = this.mutations.getMutations(node);
            if (node.isRoot() || branchMutations.isEmpty()) {
                // the times of the root mutations are meaningless
                continue;
            }

            double oldStartHeight = oldHeights[node.getParent().getNr()];
            double oldEndHeight = oldHeights[node.getNr()];
            double newStartHeight = node.getParent().getHeight();
            double newEndHeight = node.getHeight();
            double lengthRatio = (newStartHeight - newEndHeight) / (oldStartHeight - oldEndHeight);

            List<Mutation> newBranchMutations = new ArrayList<>();
            for (Mutation mutation : branchMutations) {
                newBranchMutations.add(new Mutation(
                        mutation.nodeNr(),
                        this.mapHeight(mutation.time(), oldStartHeight, oldEndHeight, newStartHeight, newEndHeight),
                        this.mapHeight(mutation.timeOfPreviousMutation(), oldStartHeight, oldEndHeight, newStartHeight, newEndHeight),
                        mutation.site(), mutation.oldState(), mutation.newState()
                ));
            }

            this.mutations.applyMutations(node, newBranchMutations, this);
            logHastingsRatio += branchMutations.size() * Math.log(lengthRatio);
        }

        // scale the other parameters with the tree length

        double actualScaler = lengthAfter / lengthBefore;

        for (Scalable down : this.downInput.get()) {
            int dimension = down.scale(1.0 / actualScaler);
            logHastingsRatio -= dimension * Math.log(actualScaler);
        }
        for (Scalable up : this.upInput.get()) {
            int dimension = up.scale(actualScaler);
            logHastingsRatio += dimension * Math.log(actualScaler);
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

    /**
     * Maps a height on a branch linearly from its old to its new ends. The ends map exactly
     * onto the new ends, which keeps the times of previous mutations at the branch start
     * consistent, and rounding cannot push a height off the branch.
     */
    private double mapHeight(double height, double oldStartHeight, double oldEndHeight, double newStartHeight, double newEndHeight) {
        if (height == oldStartHeight) {
            return newStartHeight;
        }

        double fraction = (height - oldEndHeight) / (oldStartHeight - oldEndHeight);
        double newHeight = newEndHeight + fraction * (newStartHeight - newEndHeight);
        return Math.min(Math.max(newHeight, newEndHeight), newStartHeight);
    }

    private double computeTreeLength() {
        double length = 0.0;
        for (Node node : this.tree.getNodesAsArray()) {
            length += node.getLength();
        }
        return length;
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
