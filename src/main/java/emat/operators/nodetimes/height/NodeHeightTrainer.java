package emat.operators.nodetimes.height;

import beast.base.core.Description;
import beast.base.core.Input;
import beast.base.evolution.tree.Node;
import beast.base.evolution.tree.Tree;
import beast.base.inference.State;
import beast.base.util.Randomizer;
import emat.helper.ThinnedEssTracker;
import emat.operators.nodetimes.NodeTimes;
import emat.state.Mutations;
import slice.SliceOperator;

import java.util.function.Supplier;

@Description("Collects training data for a NodeHeightLearner. Each proposal picks a random internal node below the root, " +
        "moves it over a grid of heights between its parent and its older child together with the mutations on the " +
        "three branches around it, and records the log posterior at every grid point. The state is always restored, so " +
        "the proposal never changes the chain, but the operator has to run in an MCMC that evaluates the posterior inside " +
        "a proposal, such as mcmc.JumpMCMC. How often it runs is set by its weight. It does nothing until the effective " +
        "sample size of the tree length, as seen at its own proposals, exceeds a threshold, so that the chain has left its " +
        "initial transient, and nothing once the budget of the learner is used up.")
public class NodeHeightTrainer extends SliceOperator {

    final public Input<Tree> treeInput = new Input<>("tree", "the tree to operate on", Input.Validate.REQUIRED);
    final public Input<Mutations> mutationsInput = new Input<>("mutations", "the mutations to operate on", Input.Validate.REQUIRED);
    final public Input<NodeHeightLearner> learnerInput = new Input<>("learner", "the learner that receives the training batches", Input.Validate.REQUIRED);
    final public Input<Integer> numGridPointsInput = new Input<>("numGridPoints", "the number of heights per training batch", 20);

    final public Input<Double> startEssInput = new Input<>("startEss", "the effective sample size of the tree length that has to be exceeded before the training starts", 25.0);

    private Tree tree;
    private Mutations mutations;
    private NodeHeightLearner learner;
    private int numGridPoints;
    private double startEss;

    // the tree lengths seen at the proposals, which is null once the training has started
    private ThinnedEssTracker treeLengthTracker = new ThinnedEssTracker(1);

    @Override
    public void initAndValidate() {
        this.tree = this.treeInput.get();
        this.mutations = this.mutationsInput.get();
        this.learner = this.learnerInput.get();
        this.numGridPoints = this.numGridPointsInput.get();
        this.startEss = this.startEssInput.get();
    }

    /**
     * Collects a training batch and then rejects the proposal, which makes the MCMC restore the state it had before
     * the grid walk.
     */
    @Override
    public double proposal(Supplier<Double> computeLogPosterior, State state) {
        if (!this.learner.isTrainingComplete() && this.hasStarted()) {
            this.collectBatch(computeLogPosterior);
        }
        return Double.NEGATIVE_INFINITY;
    }

    /**
     * Returns whether the training has started, which it does once the ESS of the tree length at the proposals of this
     * operator exceeds the threshold. Until then the tree length of the current state is recorded.
     */
    private boolean hasStarted() {
        if (this.treeLengthTracker == null) {
            return true;
        }

        this.treeLengthTracker.record(NodeTimes.computeTreeLength(this.tree));
        if (!this.treeLengthTracker.hasReached(this.startEss)) {
            return false;
        }

        System.out.printf("%s: training starts at an ESS of the tree length of %.1f%n", this.getID(), this.treeLengthTracker.getEss());
        this.treeLengthTracker = null;
        return true;
    }

    /**
     * Moves a random node over a grid and adds the log densities of its height to the learner. The state is left at
     * the last grid point, so the caller has to restore it.
     */
    void collectBatch(Supplier<Double> computeLogPosterior) {
        Node node = this.pickNode();
        if (node == null) {
            return;
        }

        double lowerHeight = NodeHeightMover.getOlderChild(node).getHeight();
        double upperHeight = node.getParent().getHeight();
        if (!(node.getHeight() > lowerHeight && node.getHeight() < upperHeight)) {
            return;
        }

        this.learner.refreshParameters();
        NodeHeightContext context = this.learner.createContext(node, this.mutations);

        // the grid is shifted randomly, so that the heights of different batches cover the range evenly
        double shift = Randomizer.nextDouble();
        double[] heights = new double[this.numGridPoints];
        double[] logDensities = new double[this.numGridPoints];

        // the density of the height includes the Jacobian of the map of the mutations, which accumulates over the grid
        double logJacobian = 0.0;
        for (int i = 0; i < this.numGridPoints; i++) {
            heights[i] = lowerHeight + (i + shift) / this.numGridPoints * (upperHeight - lowerHeight);
            logJacobian += NodeHeightMover.moveNode(this.mutations, node, heights[i], this);
            logDensities[i] = computeLogPosterior.get() + logJacobian;

            if (!Double.isFinite(logDensities[i])) {
                return;
            }
        }

        this.learner.addBatch(new NodeHeightPredictor.Batch(context, heights, logDensities));
    }

    /** Returns a random internal node below the root, or null if there is none. */
    private Node pickNode() {
        int numInternalNodes = this.tree.getInternalNodeCount();
        if (numInternalNodes < 2) {
            return null;
        }

        // BEAST numbers the internal nodes after the leaves
        Node node;
        do {
            node = this.tree.getNode(this.tree.getLeafNodeCount() + Randomizer.nextInt(numInternalNodes));
        } while (node.isRoot());

        return node;
    }

}
