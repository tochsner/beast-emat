package emat.operators.nodetimes;

import beast.base.core.Description;
import beast.base.evolution.tree.Node;

@Description("Base of the operators that move a random internal node in a ratio parameterisation of the node heights. " +
        "Every internal node v other than the root is parameterised by the ratio r = (h - ρ) / (h_a - ρ) of its height " +
        "h and the height h_a of its anchor node, both above a reference height ρ below the node. The root is " +
        "parameterised by its height above its reference height. The parameter of the selected node is scaled, the odds " +
        "r / (1 - r) for any other node and the height above the reference for the root, and the nodes anchored below it " +
        "are rescaled so that all other parameters stay the same.")
public abstract class AnchoredRatioScaleOperator extends ReparamScaleOperator {

    // per node number, the ratio of the node
    private double[] ratios;

    @Override
    public void initAndValidate() {
        super.initAndValidate();
        this.ratios = new double[this.tree.getNodeCount()];
    }

    /** Returns the reference height ρ of the node, which lies below the node. */
    protected abstract double getReferenceHeight(Node node);

    /** Returns whether the anchor of every node is the root, otherwise it is the parent of the node. */
    protected abstract boolean isAnchoredAtRoot();

    /**
     * Without a parent anchor, the nodes below the selected node keep their anchor, so only the
     * selected node changes unless it is the root itself.
     */
    @Override
    protected void selectAffectedNodes(Node selectedNode) {
        if (this.isAnchoredAtRoot() && !selectedNode.isRoot()) {
            this.addAffectedNode(selectedNode);
        } else {
            super.selectAffectedNodes(selectedNode);
        }
    }

    /**
     * Scales the parameter of the selected node and recomputes the heights of the nodes anchored below it
     * from their unchanged ratios. In the ratio parameterisation, h_v = ρ_v + r_v (h_a - ρ_v), so the map
     * from the ratios to the heights has the Jacobian Π_v (h_a - ρ_v) over the internal nodes v below the
     * root. Only the factors of the nodes strictly below the selected node change.
     */
    @Override
    protected double proposeHeights(Node selectedNode, double scaler) {
        this.computeRatios();

        double logHastingsRatio = selectedNode.isRoot()
                ? this.proposeRootHeight(selectedNode, scaler)
                : this.proposeInnerHeight(selectedNode, scaler);
        if (logHastingsRatio == Double.NEGATIVE_INFINITY) {
            return Double.NEGATIVE_INFINITY;
        }

        return logHastingsRatio + this.computeAnchoredHeights();
    }

    private Node getAnchor(Node node) {
        return this.isAnchoredAtRoot() ? this.tree.getRoot() : node.getParent();
    }

    private void computeRatios() {
        for (Node node : this.affectedNodes) {
            if (!node.isRoot()) {
                double referenceHeight = this.getReferenceHeight(node);
                double anchorHeight = this.getOldHeight(this.getAnchor(node));
                this.ratios[node.getNr()] = (node.getHeight() - referenceHeight) / (anchorHeight - referenceHeight);
            }
        }
    }

    /**
     * Scales the height x = h - ρ of the root above its reference height by the given factor,
     * which has the Jacobian s. Returns the log Hastings ratio of this step.
     */
    private double proposeRootHeight(Node root, double scaler) {
        double referenceHeight = this.getReferenceHeight(root);
        double newHeight = referenceHeight + (root.getHeight() - referenceHeight) * scaler;
        if (!(newHeight > referenceHeight)) {
            return Double.NEGATIVE_INFINITY;
        }

        this.newHeights[root.getNr()] = newHeight;
        return Math.log(scaler);
    }

    /**
     * Moves the ratio r of the given non-root node by the given scaler. The anchor keeps its height, so the
     * factor of the node in the Jacobian of the parameterisation stays the same. Returns the log Hastings
     * ratio of the step in the ratio, or negative infinity if the new ratio leaves the unit interval.
     */
    private double proposeInnerHeight(Node node, double scaler) {
        int nodeNr = node.getNr();
        double ratio = this.ratios[nodeNr];
        double newRatio = this.proposeRatio(ratio, scaler);
        if (!(newRatio > 0.0 && newRatio < 1.0)) {
            return Double.NEGATIVE_INFINITY;
        }

        double logHastingsRatio = this.computeLogHastingsRatio(ratio, newRatio);
        if (!Double.isFinite(logHastingsRatio)) {
            return Double.NEGATIVE_INFINITY;
        }

        double referenceHeight = this.getReferenceHeight(node);
        double anchorHeight = this.getOldHeight(this.getAnchor(node));
        this.newHeights[nodeNr] = referenceHeight + newRatio * (anchorHeight - referenceHeight);
        this.ratios[nodeNr] = newRatio;

        return logHastingsRatio;
    }

    /**
     * Scales the odds r / (1 - r) by the given factor, which is a symmetric step in logit(r). Subclasses
     * override this together with {@link #computeLogHastingsRatio} to step in another transformed space.
     */
    protected double proposeRatio(double ratio, double scaler) {
        double odds = ratio / (1.0 - ratio) * scaler;
        return odds / (1.0 + odds);
    }

    /** Returns the log Hastings ratio r' (1 - r') / (r (1 - r)) in the ratio of the step in logit(r). */
    protected double computeLogHastingsRatio(double ratio, double newRatio) {
        return Math.log(newRatio * (1.0 - newRatio)) - Math.log(ratio * (1.0 - ratio));
    }

    /**
     * Computes the new heights of the affected nodes strictly below the selected node from their unchanged
     * ratios and the new heights of their anchors. Returns the log ratio of the new to the old Jacobian of
     * the parameterisation, or negative infinity if an anchor lies at or below a reference height.
     */
    private double computeAnchoredHeights() {
        double logJacobian = 0.0;

        // the pre-order visits every anchor before the nodes anchored to it
        for (int i = 1; i < this.affectedNodes.size(); i++) {
            Node node = this.affectedNodes.get(i);
            Node anchor = this.getAnchor(node);
            double referenceHeight = this.getReferenceHeight(node);
            double newAnchorInterval = this.getNewHeight(anchor) - referenceHeight;
            double oldAnchorInterval = this.getOldHeight(anchor) - referenceHeight;
            if (!(newAnchorInterval > 0.0)) {
                return Double.NEGATIVE_INFINITY;
            }

            this.newHeights[node.getNr()] = referenceHeight + this.ratios[node.getNr()] * newAnchorInterval;
            logJacobian += Math.log(newAnchorInterval / oldAnchorInterval);
        }

        return logJacobian;
    }

}
