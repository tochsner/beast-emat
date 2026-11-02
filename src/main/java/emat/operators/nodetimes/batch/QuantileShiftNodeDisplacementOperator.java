package emat.operators.nodetimes.batch;

import beast.base.core.Description;
import beast.base.core.Input;
import beast.base.evolution.tree.Node;
import beast.base.util.Randomizer;
import emat.operators.nodetimes.NodeTimes;
import emat.operators.nodetimes.batch.TruncatedDensityTransport.Tails;

@Description("Shifts the heights of all internal nodes below the root together in the space of their conditional " +
        "quantiles. Every node has a conditional density for its height (the genetic prior times a learned quadratic " +
        "approximation of the tree prior) within the range that keeps the topology and the mutation times, and a height " +
        "is mapped to its cumulative probability under it. All probabilities are shifted by the same amount, modulo one, " +
        "and mapped back. A shift leaves the uniform distribution of the probabilities invariant, so the move is accepted " +
        "unless the tree prior differs from its approximation, while all nodes move in the same direction and the tree length " +
        "changes coherently. The nodes are moved one after the other in random order, and the reverse move shifts back in the " +
        "opposite order, which sees the same ranges. Optionally the sign of the shift is persistent: it stays after an " +
        "accepted proposal and flips after a rejected one, which makes the chain non-reversible but leaves the posterior " +
        "invariant.")
public class QuantileShiftNodeDisplacementOperator extends TransportedNodeDisplacementOperator {

    final public Input<Boolean> persistentInput = new Input<>("persistent", "whether the sign of the shift persists until a proposal is rejected. Otherwise it is symmetric", false);

    // the shift is a rotation of the circle of the probabilities, so half a turn is the largest useful shift
    private static final double MAX_STEP_SIZE = 0.5;

    private boolean isPersistent;

    // the sign of the shift of a persistent chain
    private int direction;

    // the numbers of the internal nodes below the root, in the order they were visited for the last proposal
    private int[] internalNodeNrs;

    @Override
    public void initAndValidate() {
        super.initAndValidate();

        this.isPersistent = this.persistentInput.get();
        this.direction = Randomizer.nextBoolean() ? 1 : -1;

        // BEAST numbers the internal nodes after the leaves
        int numLeaves = this.tree.getLeafNodeCount();
        this.internalNodeNrs = new int[this.tree.getInternalNodeCount()];
        for (int i = 0; i < this.internalNodeNrs.length; i++) {
            this.internalNodeNrs[i] = numLeaves + i;
        }
    }

    /**
     * Shifts the probabilities of the nodes one after the other in random order, also the nodes that have no room
     * and are skipped. The node at each step sees the nodes before it at their new heights and the nodes after it at
     * their old heights, in both directions, so its range and its conditional are the same in the move and its reverse.
     * The visiting order does not enter the Hastings ratio.
     */
    @Override
    protected double moveNodes() {
        double shift = this.drawShift();

        double logHastingsRatio = 0.0;
        int numInternalNodes = this.internalNodeNrs.length;
        for (int i = 0; i < numInternalNodes; i++) {
            // shuffle the next node into place, so that every node is visited once
            int j = i + Randomizer.nextInt(numInternalNodes - i);
            int nodeNr = this.internalNodeNrs[j];
            this.internalNodeNrs[j] = this.internalNodeNrs[i];
            this.internalNodeNrs[i] = nodeNr;

            Node node = this.tree.getNode(nodeNr);
            if (node.isRoot()) {
                continue;
            }

            logHastingsRatio += this.shiftNode(node, shift);
        }

        return logHastingsRatio;
    }

    /**
     * Draws the shift of the probabilities. A persistent chain shifts in its direction by a size that is uniform
     * on [0, step size], and otherwise the shift is uniform on [-step size, step size]. The reverse of a shift is the
     * shift of the same size in the opposite direction, so that the proposal densities cancel.
     */
    private double drawShift() {
        if (this.isPersistent) {
            return this.direction * this.getCoercableParameterValue() * Randomizer.nextDouble();
        }
        return this.getCoercableParameterValue() * (2.0 * Randomizer.nextDouble() - 1.0);
    }

    /** Shifts the probability of the given node within the range it has now, and returns its part of the log Hastings ratio. */
    private double shiftNode(Node node, double shift) {
        // the range of a node depends on the nodes that were moved before it
        double minHeight = NodeTimes.computeMinHeight(this.mutations, node);
        double maxHeight = NodeTimes.computeMaxHeight(this.mutations, node);
        if (maxHeight <= minHeight) {
            return 0.0;
        }

        Conditional conditional = this.createConditional(node, minHeight, maxHeight);

        double probability = conditional.computeTails(node.getHeight()).lower() + shift;
        probability -= Math.floor(probability);

        double newHeight = conditional.computeHeight(Tails.ofProbability(probability));
        return this.displace(node, conditional, newHeight);
    }

    /**
     * Reverses the direction of a persistent chain after a rejected proposal, whatever the reason of the rejection.
     * The direction stays after an accepted proposal.
     */
    @Override
    public void reject(int reason) {
        super.reject(reason);
        if (this.isPersistent) {
            this.direction = -this.direction;
        }
    }

    @Override
    protected double getMaxStepSize() {
        return MAX_STEP_SIZE;
    }

}
