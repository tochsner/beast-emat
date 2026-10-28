package emat.operators.nodetimes;

import beast.base.core.Description;
import beast.base.core.Input;
import beast.base.evolution.tree.Node;
import beast.base.util.Randomizer;

import java.text.DecimalFormat;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

@Description("Moves a random internal node in the ratio parameterisation of the node heights. The root is " +
        "parameterised by its height above its oldest tip, and every other internal node by the ratio " +
        "r = (h - τ) / (h_p - τ) of its height h and the height h_p of its parent, both above the oldest tip τ " +
        "below the node. The parameter of the node is scaled by a factor from a Bactrian kernel, the height above " +
        "the oldest tip for the root and the odds r / (1 - r) for any other node, and the subtree below is " +
        "rescaled so that all other parameters stay the same. The mutations on the changed branches move " +
        "linearly along with their ends. Every clade size class, the base-2 logarithm of the number of tips " +
        "below the node rounded down, has a scale factor of its own that is tuned separately, since a clade with " +
        "more nodes and mutations below it accepts only smaller steps.")
public class RatioScaleOperator extends ScaleTunedOperator {

    final public Input<Double> scaleUpperLimit = new Input<>("upper", "upper limit of the scale factor", 10.0);
    final public Input<Double> scaleLowerLimit = new Input<>("lower", "lower limit of the scale factor", 1e-8);
    final public Input<Double> scaleFactorInput = new Input<>("scaleFactor", "initial width of the kernel in log space of every clade size class: close to zero is very small jumps, larger is larger jumps.", 0.1);

    // per node number, the height of the oldest tip below the node, its ratio, and its old and new height
    private double[] oldestTipHeights;
    private double[] ratios;
    private double[] oldHeights;
    private double[] newHeights;

    // the nodes of the subtree of the last proposal in pre-order, starting with the selected node
    private final List<Node> subtreeNodes = new ArrayList<>();

    // the scale factor and the number of tuning steps of every clade size class, and the class of the last proposal
    private double[] classScaleFactors;
    private long[] classTuningSteps;
    private int lastSizeClass;

    @Override
    public void initAndValidate() {
        super.initAndValidate();

        this.scaleFactor = this.scaleFactorInput.get();
        this.upper = this.scaleUpperLimit.get();
        this.lower = this.scaleLowerLimit.get();

        int numNodes = this.tree.getNodeCount();
        this.oldestTipHeights = new double[numNodes];
        this.ratios = new double[numNodes];
        this.oldHeights = new double[numNodes];
        this.newHeights = new double[numNodes];

        int numSizeClasses = this.computeSizeClass(this.tree.getLeafNodeCount()) + 1;
        this.classScaleFactors = new double[numSizeClasses];
        this.classTuningSteps = new long[numSizeClasses];
        Arrays.fill(this.classScaleFactors, this.scaleFactor);
    }

    /**
     * Picks an internal node X uniformly, including the root, scales its parameter and
     * recomputes the heights below X from their unchanged ratios. In the ratio
     * parameterisation, h_v = τ_v + r_v (h_p - τ_v), so the map from the ratios to the
     * heights has the Jacobian Π_v (h_p - τ_v) over the internal nodes v below the root.
     * Only the factors of the internal nodes strictly below X change. Every mutation keeps
     * its relative position on its branch, which adds the log ratio of the new to the old
     * branch length once per mutation.
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

        this.subtreeNodes.clear();
        this.collectSubtree(node);
        this.computeRatios();

        this.lastSizeClass = this.computeSizeClass(this.countTips());
        double scaler = this.kernelDistribution.getScaler(node.getNr(), node.getHeight(), this.classScaleFactors[this.lastSizeClass]);
        double logHastingsRatio = node.isRoot()
                ? this.proposeRootHeight(node, scaler)
                : this.proposeInnerHeight(node, scaler);
        if (logHastingsRatio == Double.NEGATIVE_INFINITY) {
            return Double.NEGATIVE_INFINITY;
        }

        double logJacobian = this.computeSubtreeHeights();
        if (logJacobian == Double.NEGATIVE_INFINITY) {
            return Double.NEGATIVE_INFINITY;
        }
        logHastingsRatio += logJacobian;

        // apply the new heights, then move the mutations of every changed branch along with its ends

        for (Node subtreeNode : this.subtreeNodes) {
            if (!subtreeNode.isLeaf()) {
                subtreeNode.setHeight(this.newHeights[subtreeNode.getNr()]);
            }
        }

        for (Node subtreeNode : this.subtreeNodes) {
            // the times of the root mutations are meaningless
            if (!subtreeNode.isRoot()) {
                int nodeNr = subtreeNode.getNr();
                int parentNr = subtreeNode.getParent().getNr();
                logHastingsRatio += BranchTimeMapper.mapBranch(
                        this.mutations, subtreeNode, this.oldHeights[parentNr], this.oldHeights[nodeNr],
                        this.newHeights[parentNr], this.newHeights[nodeNr], this
                );
            }
        }

        return logHastingsRatio;
    }

    /**
     * Adds the subtree of the given node to the subtree nodes in pre-order, and records the
     * old height and the height of the oldest tip below every node of it.
     */
    private void collectSubtree(Node node) {
        int nodeNr = node.getNr();
        this.subtreeNodes.add(node);
        this.oldHeights[nodeNr] = node.getHeight();
        this.newHeights[nodeNr] = node.getHeight();

        if (node.isLeaf()) {
            this.oldestTipHeights[nodeNr] = node.getHeight();
            return;
        }

        double oldestTipHeight = Double.NEGATIVE_INFINITY;
        for (Node child : node.getChildren()) {
            this.collectSubtree(child);
            oldestTipHeight = Math.max(oldestTipHeight, this.oldestTipHeights[child.getNr()]);
        }
        this.oldestTipHeights[nodeNr] = oldestTipHeight;
    }

    /**
     * Computes the ratio of every internal node of the subtree, including the selected node
     * unless it is the root. The parent of the selected node lies outside the subtree, so its
     * height is recorded as well, which stays the same.
     */
    private void computeRatios() {
        Node selectedNode = this.subtreeNodes.getFirst();
        if (!selectedNode.isRoot()) {
            Node parent = selectedNode.getParent();
            this.oldHeights[parent.getNr()] = parent.getHeight();
            this.newHeights[parent.getNr()] = parent.getHeight();
        }

        for (Node node : this.subtreeNodes) {
            if (!node.isLeaf() && !node.isRoot()) {
                double oldestTipHeight = this.oldestTipHeights[node.getNr()];
                this.ratios[node.getNr()] = (node.getHeight() - oldestTipHeight) / (node.getParent().getHeight() - oldestTipHeight);
            }
        }
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

    /**
     * Scales the height x = h - τ of the root above its oldest tip by the given factor, which
     * has the Jacobian s. Returns the log Hastings ratio of this step.
     */
    private double proposeRootHeight(Node root, double scaler) {
        double oldestTipHeight = this.oldestTipHeights[root.getNr()];
        double newHeight = oldestTipHeight + (root.getHeight() - oldestTipHeight) * scaler;
        if (!(newHeight > oldestTipHeight)) {
            return Double.NEGATIVE_INFINITY;
        }

        this.newHeights[root.getNr()] = newHeight;
        return Math.log(scaler);
    }

    /**
     * Scales the odds r / (1 - r) of the ratio r of the given non-root node by the given
     * factor, which is a symmetric step in logit(r) and has the Hastings ratio
     * r' (1 - r') / (r (1 - r)) in the ratio. The parent keeps its height, so the factor of the
     * node in the Jacobian of the parameterisation stays the same.
     */
    private double proposeInnerHeight(Node node, double scaler) {
        int nodeNr = node.getNr();
        double ratio = this.ratios[nodeNr];
        double odds = ratio / (1.0 - ratio) * scaler;
        double newRatio = odds / (1.0 + odds);
        if (!(newRatio > 0.0 && newRatio < 1.0)) {
            return Double.NEGATIVE_INFINITY;
        }

        double oldestTipHeight = this.oldestTipHeights[nodeNr];
        double parentHeight = node.getParent().getHeight();
        this.newHeights[nodeNr] = oldestTipHeight + newRatio * (parentHeight - oldestTipHeight);
        this.ratios[nodeNr] = newRatio;

        return Math.log(newRatio * (1.0 - newRatio)) - Math.log(ratio * (1.0 - ratio));
    }

    /**
     * Computes the new heights of the internal nodes strictly below the selected node from
     * their unchanged ratios and the new heights of their parents. Returns the log ratio of
     * the new to the old Jacobian of the parameterisation, or negative infinity if rounding
     * would put a node at or below one of its children.
     */
    private double computeSubtreeHeights() {
        double logJacobian = 0.0;

        // the pre-order visits every parent before its children
        for (int i = 1; i < this.subtreeNodes.size(); i++) {
            Node node = this.subtreeNodes.get(i);
            int nodeNr = node.getNr();
            int parentNr = node.getParent().getNr();

            if (!node.isLeaf()) {
                double oldestTipHeight = this.oldestTipHeights[nodeNr];
                double newParentInterval = this.newHeights[parentNr] - oldestTipHeight;
                double oldParentInterval = this.oldHeights[parentNr] - oldestTipHeight;

                this.newHeights[nodeNr] = oldestTipHeight + this.ratios[nodeNr] * newParentInterval;
                logJacobian += Math.log(newParentInterval / oldParentInterval);
            }

            if (!(this.newHeights[nodeNr] < this.newHeights[parentNr])) {
                return Double.NEGATIVE_INFINITY;
            }
        }

        return logJacobian;
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
