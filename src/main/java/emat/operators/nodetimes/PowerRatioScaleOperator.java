package emat.operators.nodetimes;

import beast.base.core.Description;
import beast.base.core.Input;
import beast.base.evolution.tree.Node;
import emat.diagnostics.DiagnosedOperator;
import emat.diagnostics.StepAnnotations;

import java.util.ArrayList;
import java.util.List;

@Description("Keeps the height of the root and moves all other internal nodes in the ratio parameterisation, " +
        "in which every internal node has the ratio r = (h - τ) / (h_p - τ) of its height h and the height h_p of " +
        "its parent, both above the oldest tip τ below the node. Every ratio r is replaced by r^k for a factor k " +
        "drawn from a Bactrian kernel in log space, and the heights are recomputed from the root down. A k below " +
        "one moves every node towards its parent and lengthens the tree, a k above one shortens it. All ratios share " +
        "the same k, which moves the tree along one direction only, so the operator needs other moves that change " +
        "the ratios. The mutations on the changed branches move linearly along with their ends.")
public class PowerRatioScaleOperator extends CompensatedScaleOperator implements DiagnosedOperator {

    final public Input<Double> scaleUpperLimit = new Input<>("upper", "upper limit of the scale factor", 10.0);
    final public Input<Double> scaleLowerLimit = new Input<>("lower", "lower limit of the scale factor", 1e-8);
    final public Input<Double> scaleFactorInput = new Input<>("scaleFactor", "initial width of the kernel of log k: close to zero is very small jumps, larger is larger jumps.", 0.02);

    // per node number, the height of the oldest tip below the node, its ratio, and its old and new height
    private double[] oldestTipHeights;
    private double[] ratios;
    private double[] oldHeights;
    private double[] newHeights;

    // the nodes of the tree of the last proposal in pre-order, starting with the root
    private final List<Node> nodes = new ArrayList<>();

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
    }

    @Override
    public String[] getAnnotationNames() {
        return new String[]{"logScaler", "width"};
    }

    /**
     * Raises the ratio of every internal node below the root to the power k and recomputes the
     * heights from the root down. The ratios are moved by r' = r^k with k symmetric in log
     * space, so the Hastings ratio of the ratios is the Jacobian Π_v k r_v^(k-1) of the map
     * r -> r^k. The map from the ratios to the heights has the Jacobian Π_v (h_p - τ_v), so
     * the log ratio of the new to the old Jacobian is added as well. Every mutation keeps
     * its relative position on its branch, which adds the log ratio of the new to the old
     * branch length once per mutation.
     */
    @Override
    public double proposal() {
        // the root is kept, so there must be another internal node to move
        if (this.tree.getInternalNodeCount() < 2) {
            return Double.NEGATIVE_INFINITY;
        }

        double lengthBefore = this.hasCompensatedParameters() ? NodeTimes.computeTreeLength(this.tree) : 0.0;

        this.nodes.clear();
        this.collectNodes(this.tree.getRoot());
        this.computeRatios();

        double power = this.kernelDistribution.getScaler(0, 1.0, this.scaleFactor);
        StepAnnotations.set(0, Math.log(power));
        StepAnnotations.set(1, this.scaleFactor);

        double logHastingsRatio = this.computeNewHeights(power);
        if (logHastingsRatio == Double.NEGATIVE_INFINITY) {
            return Double.NEGATIVE_INFINITY;
        }

        // apply the new heights, then move the mutations of every changed branch along with its ends

        for (Node node : this.nodes) {
            if (!node.isLeaf()) {
                node.setHeight(this.newHeights[node.getNr()]);
            }
        }

        for (Node node : this.nodes) {
            // the times of the root mutations are meaningless
            if (!node.isRoot()) {
                int nodeNr = node.getNr();
                int parentNr = node.getParent().getNr();
                logHastingsRatio += BranchTimeMapper.mapBranch(
                        this.mutations, node, this.oldHeights[parentNr], this.oldHeights[nodeNr],
                        this.newHeights[parentNr], this.newHeights[nodeNr], this
                );
            }
        }

        // scale the other parameters with the tree length

        if (this.hasCompensatedParameters()) {
            double logJacobian = this.scaleCompensatedParameters(lengthBefore, NodeTimes.computeTreeLength(this.tree));
            if (logJacobian == Double.NEGATIVE_INFINITY) {
                return Double.NEGATIVE_INFINITY;
            }
            logHastingsRatio += logJacobian;
        }

        return logHastingsRatio;
    }

    /** Adds the subtree of the given node to the nodes in pre-order, and records the old height and the height of the oldest tip below every node of it. */
    private void collectNodes(Node node) {
        int nodeNr = node.getNr();
        this.nodes.add(node);
        this.oldHeights[nodeNr] = node.getHeight();
        this.newHeights[nodeNr] = node.getHeight();

        if (node.isLeaf()) {
            this.oldestTipHeights[nodeNr] = node.getHeight();
            return;
        }

        double oldestTipHeight = Double.NEGATIVE_INFINITY;
        for (Node child : node.getChildren()) {
            this.collectNodes(child);
            oldestTipHeight = Math.max(oldestTipHeight, this.oldestTipHeights[child.getNr()]);
        }
        this.oldestTipHeights[nodeNr] = oldestTipHeight;
    }

    /** Computes the ratio of every internal node below the root. */
    private void computeRatios() {
        for (Node node : this.nodes) {
            if (!node.isLeaf() && !node.isRoot()) {
                double oldestTipHeight = this.oldestTipHeights[node.getNr()];
                this.ratios[node.getNr()] = (node.getHeight() - oldestTipHeight) / (node.getParent().getHeight() - oldestTipHeight);
            }
        }
    }

    /**
     * Computes the new heights of the internal nodes below the root from the ratios raised to the given power and the new heights of their parents. Returns the log Hastings ratio of the ratios and the log ratio of the new to the
     * old Jacobian of the parameterisation, or negative infinity if rounding would put a node at or above its parent.
     */
    private double computeNewHeights(double power) {
        double logHastingsRatio = 0.0;

        // the pre-order visits every parent before its children
        for (int i = 1; i < this.nodes.size(); i++) {
            Node node = this.nodes.get(i);
            int nodeNr = node.getNr();
            int parentNr = node.getParent().getNr();

            if (!node.isLeaf()) {
                double ratio = this.ratios[nodeNr];
                double oldestTipHeight = this.oldestTipHeights[nodeNr];
                double newParentInterval = this.newHeights[parentNr] - oldestTipHeight;
                double oldParentInterval = this.oldHeights[parentNr] - oldestTipHeight;

                this.newHeights[nodeNr] = oldestTipHeight + Math.pow(ratio, power) * newParentInterval;
                logHastingsRatio += Math.log(power) + (power - 1.0) * Math.log(ratio) + Math.log(newParentInterval / oldParentInterval);
            }

            if (!(this.newHeights[nodeNr] < this.newHeights[parentNr])) {
                return Double.NEGATIVE_INFINITY;
            }
        }

        return logHastingsRatio;
    }

}
