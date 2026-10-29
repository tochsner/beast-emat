package emat.operators.nodetimes;

import beast.base.core.Description;
import beast.base.core.Input;
import beast.base.evolution.tree.Node;
import beast.base.util.Randomizer;

import java.text.DecimalFormat;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

@Description("Base of the operators that move a random internal node in a reparameterisation of the node heights. " +
        "A subclass defines the parameterisation: which internal nodes change their height when the parameter of the " +
        "selected node is scaled, and how their new heights follow from their unchanged parameters. The mutations on " +
        "the changed branches move linearly along with their ends. Every clade size class, the base-2 logarithm of " +
        "the number of tips below the node rounded down, has a scale factor of its own that is tuned separately.")
public abstract class ReparamScaleOperator extends ScaleTunedOperator {

    final public Input<Double> scaleUpperLimit = new Input<>("upper", "upper limit of the scale factor", 10.0);
    final public Input<Double> scaleLowerLimit = new Input<>("lower", "lower limit of the scale factor", 1e-8);
    final public Input<Double> scaleFactorInput = new Input<>("scaleFactor", "initial width of the kernel in log space of every clade size class: close to zero is very small jumps, larger is larger jumps.", 0.1);
    final public Input<Boolean> scaleWeightByLogNumTipsInput = new Input<>("scaleWeightByLogNumTips", "whether to scale the weight by the natural logarithm of the number of tips n, so that the operator is selected with weight * ln(n). Otherwise, the weight is used as given", false);

    // the factor ln(n) or one by which the weight of the operator is scaled
    private double weightFactor;

    // per node number, the heights of the oldest and youngest tip below the node, and its old and new height
    double[] oldestTipHeights;
    double[] youngestTipHeights;
    double[] oldHeights;
    double[] newHeights;

    // the nodes of the subtree of the last proposal in pre-order, starting with the selected node
    final List<Node> subtreeNodes = new ArrayList<>();

    // the internal nodes that change their height in the last proposal in pre-order, starting with the selected node
    final List<Node> affectedNodes = new ArrayList<>();
    private boolean[] isAffected;

    // the scale factor and the number of tuning steps of every clade size class, and the class of the last proposal
    private double[] classScaleFactors;
    private long[] classTuningSteps;
    int lastSizeClass;

    @Override
    public void initAndValidate() {
        super.initAndValidate();

        this.scaleFactor = this.scaleFactorInput.get();
        this.upper = this.scaleUpperLimit.get();
        this.lower = this.scaleLowerLimit.get();

        int numNodes = this.tree.getNodeCount();
        this.oldestTipHeights = new double[numNodes];
        this.youngestTipHeights = new double[numNodes];
        this.oldHeights = new double[numNodes];
        this.newHeights = new double[numNodes];
        this.isAffected = new boolean[numNodes];

        int numSizeClasses = this.computeSizeClass(this.tree.getLeafNodeCount()) + 1;
        this.classScaleFactors = new double[numSizeClasses];
        this.classTuningSteps = new long[numSizeClasses];
        Arrays.fill(this.classScaleFactors, this.scaleFactor);

        this.weightFactor = this.scaleWeightByLogNumTipsInput.get() ? Math.log(this.tree.getLeafNodeCount()) : 1.0;
    }

    /**
     * Returns the weight, scaled by the logarithm of the number of tips if requested, which
     * lets the share of the proposals grow with the size of the tree.
     */
    @Override
    public double getWeight() {
        return super.getWeight() * this.weightFactor;
    }

    /**
     * Picks an internal node uniformly, including the root, and lets the subclass scale its parameter
     * and recompute the heights of the other affected nodes. Every mutation keeps its relative position
     * on its branch, which adds the log ratio of the new to the old branch length once per mutation.
     */
    @Override
    public double proposal() {
        int numLeaves = this.tree.getLeafNodeCount();
        int numInternalNodes = this.tree.getInternalNodeCount();
        if (numInternalNodes == 0) {
            return Double.NEGATIVE_INFINITY;
        }

        // BEAST numbers the internal nodes after the leaves
        Node node = this.tree.getNode(numLeaves + Randomizer.nextInt(numInternalNodes));

        this.clearAffectedNodes();
        this.subtreeNodes.clear();
        this.collectSubtree(node);
        this.selectAffectedNodes(node);

        this.lastSizeClass = this.computeSizeClass(this.countTips());
        double scaler = this.kernelDistribution.getScaler(node.getNr(), node.getHeight(), this.classScaleFactors[this.lastSizeClass]);
        double logHastingsRatio = this.proposeHeights(node, scaler);
        if (logHastingsRatio == Double.NEGATIVE_INFINITY || !this.isOrdered()) {
            return Double.NEGATIVE_INFINITY;
        }

        for (Node affectedNode : this.affectedNodes) {
            affectedNode.setHeight(this.newHeights[affectedNode.getNr()]);
        }

        return logHastingsRatio + this.mapMutations();
    }

    /**
     * Fills the new heights of the affected nodes by scaling the parameter of the selected node by the
     * given factor. Returns the log Hastings ratio of the step in the parameters plus the log ratio of
     * the new to the old Jacobian of the parameterisation, or negative infinity if the step is invalid.
     */
    protected abstract double proposeHeights(Node selectedNode, double scaler);

    /**
     * Marks the internal nodes whose height changes when the parameter of the selected node is
     * scaled. By default, these are all internal nodes of the subtree of the selected node.
     */
    protected void selectAffectedNodes(Node selectedNode) {
        for (Node node : this.subtreeNodes) {
            if (!node.isLeaf()) {
                this.addAffectedNode(node);
            }
        }
    }

    protected final void addAffectedNode(Node node) {
        this.affectedNodes.add(node);
        this.isAffected[node.getNr()] = true;
    }

    private void clearAffectedNodes() {
        for (Node node : this.affectedNodes) {
            this.isAffected[node.getNr()] = false;
        }
        this.affectedNodes.clear();
    }

    /** Returns the height that the node has after the last proposal. */
    protected final double getNewHeight(Node node) {
        return this.isAffected[node.getNr()] ? this.newHeights[node.getNr()] : node.getHeight();
    }

    /** Returns the height that the node had before the last proposal. */
    protected final double getOldHeight(Node node) {
        return this.isAffected[node.getNr()] ? this.oldHeights[node.getNr()] : node.getHeight();
    }

    /**
     * Adds the subtree of the given node to the subtree nodes in pre-order, and records the
     * old height and the heights of the oldest and youngest tip below every node of it.
     */
    private void collectSubtree(Node node) {
        int nodeNr = node.getNr();
        this.subtreeNodes.add(node);
        this.oldHeights[nodeNr] = node.getHeight();
        this.newHeights[nodeNr] = node.getHeight();

        if (node.isLeaf()) {
            this.oldestTipHeights[nodeNr] = node.getHeight();
            this.youngestTipHeights[nodeNr] = node.getHeight();
            return;
        }

        double oldestTipHeight = Double.NEGATIVE_INFINITY;
        double youngestTipHeight = Double.POSITIVE_INFINITY;
        for (Node child : node.getChildren()) {
            this.collectSubtree(child);
            oldestTipHeight = Math.max(oldestTipHeight, this.oldestTipHeights[child.getNr()]);
            youngestTipHeight = Math.min(youngestTipHeight, this.youngestTipHeights[child.getNr()]);
        }
        this.oldestTipHeights[nodeNr] = oldestTipHeight;
        this.youngestTipHeights[nodeNr] = youngestTipHeight;
    }

    /** Returns the number of tips in the subtree of the last proposal. */
    private int countTips() {
        int numTips = 0;
        for (Node node : this.subtreeNodes) {
            if (node.isLeaf()) {
                numTips++;
            }
        }
        return numTips;
    }

    /** Returns the clade size class of a clade with the given number of tips, ⌊log₂ n⌋. */
    private int computeSizeClass(int numTips) {
        return 31 - Integer.numberOfLeadingZeros(Math.max(numTips, 1));
    }

    /** Returns whether every affected node is strictly below its parent and strictly above its children. */
    private boolean isOrdered() {
        for (Node node : this.affectedNodes) {
            double height = this.newHeights[node.getNr()];
            if (!node.isRoot() && !(height < this.getNewHeight(node.getParent()))) {
                return false;
            }
            for (Node child : node.getChildren()) {
                if (!(this.getNewHeight(child) < height)) {
                    return false;
                }
            }
        }
        return true;
    }

    /**
     * Moves the mutations of every branch with a changed end linearly along with its ends, once the new
     * heights are applied. Returns the sum of the log Jacobians of the branches.
     */
    private double mapMutations() {
        double logJacobian = 0.0;

        for (Node node : this.affectedNodes) {
            logJacobian += this.mapBranch(node);
            for (Node child : node.getChildren()) {
                // the branch between two affected nodes is mapped once, from its lower end
                if (!this.isAffected[child.getNr()]) {
                    logJacobian += this.mapBranch(child);
                }
            }
        }

        return logJacobian;
    }

    private double mapBranch(Node node) {
        // the times of the root mutations are meaningless
        if (node.isRoot()) {
            return 0.0;
        }

        Node parent = node.getParent();
        double oldStartHeight = this.getOldHeight(parent);
        double oldEndHeight = this.getOldHeight(node);
        double newStartHeight = parent.getHeight();
        double newEndHeight = node.getHeight();
        if (oldStartHeight == newStartHeight && oldEndHeight == newEndHeight) {
            return 0.0;
        }

        return BranchTimeMapper.mapBranch(this.mutations, node, oldStartHeight, oldEndHeight, newStartHeight, newEndHeight, this);
    }

    /* Tuning */

    /**
     * Tunes the scale factor of the clade size class of the last proposal by the Robbins-Monro
     * rule of the operator schedule, log s += (α - α*) / √(n + 1), where n counts the tuning
     * steps of that class only.
     */
    @Override
    public void optimize(double logAlpha) {
        if (!this.optimiseInput.get()) {
            return;
        }

        int sizeClass = this.lastSizeClass;
        double gain = 1.0 / Math.sqrt(this.classTuningSteps[sizeClass] + 1.0);
        double delta = gain * (Math.exp(Math.min(logAlpha, 0.0)) - this.getTargetAcceptanceProbability());
        if (Double.isFinite(delta)) {
            double scaleFactor = Math.exp(Math.log(this.classScaleFactors[sizeClass]) + delta);
            this.classScaleFactors[sizeClass] = Math.max(Math.min(scaleFactor, this.upper), this.lower);
        }
        this.classTuningSteps[sizeClass]++;
    }

    /** Lists the tuned scale factor of every clade size class that was proposed. */
    @Override
    public String getPerformanceSuggestion() {
        DecimalFormat formatter = new DecimalFormat("#.###");
        StringBuilder suggestion = new StringBuilder("scale factors by clade size class:");
        for (int sizeClass = 0; sizeClass < this.classScaleFactors.length; sizeClass++) {
            if (this.classTuningSteps[sizeClass] > 0) {
                suggestion.append(" ").append(1 << sizeClass).append("+ tips ").append(formatter.format(this.classScaleFactors[sizeClass]));
            }
        }
        return suggestion.toString();
    }

}
