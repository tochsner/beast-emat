package emat.operators.nodetimes.batch;

import adapters.Adapter;
import beast.base.core.Description;
import beast.base.core.Input;
import beast.base.evolution.operator.TreeOperator;
import beast.base.evolution.tree.Node;
import beast.base.inference.StateNode;
import beast.base.util.Randomizer;
import emat.helper.ThinnedEssTracker;
import emat.operators.nodetimes.NodeTimes;
import emat.operators.nodetimes.height.NodeHeightMover;
import emat.prior.GeneticPrior;
import emat.state.Mutations;

import java.util.ArrayList;
import java.util.List;

@Description("Displaces the heights of a batch of internal nodes one after the other, each anywhere between its parent " +
    "and its older child, and maps the mutations on the three branches around it linearly along with it. Every node draws " +
    "its new height from a piecewise constant density over its range, whose cells follow the exact genetic prior of the " +
    "stretched branches, including the Jacobian of the map of the mutations, times a learned global spline approximation " +
    "of the tree prior. So only the error of that approximation and of the piecewise constant density enters the acceptance " +
    "probability. The operator does nothing until the effective sample size of the tree length, as seen at its own " +
    "proposals, exceeds a threshold, so that the chain has left its initial transient, in which moves over the whole range of " +
    "a node can stretch the tree and its mutations without bound.")
public class MappedBatchNodeDisplacementOperator extends TreeOperator {

    final public Input<Mutations> mutationsInput = new Input<>("mutations", "the mutations to operate on", Input.Validate.REQUIRED);
    final public Input<GeneticPrior> geneticPriorInput = new Input<>("geneticPrior", "the genetic prior that defines the evolutionary model", Input.Validate.REQUIRED);
    final public Input<Integer> batchSizeInput = new Input<>("batchSize", "the initial number of nodes that are displaced together, which is tuned if optimise is set, and which is capped at the number of internal nodes below the root", 128);
    final public Input<Boolean> optimiseInput = new Input<>("optimise", "whether to tune the batch size", true);
    final public Input<Integer> numCellsInput = new Input<>("numCells", "the number of cells of equal width of the proposal density of a node height", 100);
    final public Input<Boolean> learnTreePriorInput = new Input<>("learnTreePrior", "whether to learn a global spline approximation of the tree prior, instead of drawing from the genetic prior only", true);
    final public Input<Integer> treePriorKnotsInput = new Input<>("treePriorKnots", "the number of knots along the height between which the slope of the approximation of the tree prior is interpolated", 20);
    final public Input<List<StateNode>> treePriorParametersInput = new Input<>("treePriorParameter", "real scalar, real vector or integer vector parameters of the tree prior, which the approximation of the tree prior may depend on", new ArrayList<>());
    final public Input<List<Adapter>> adaptersInput = new Input<>("adapter", "adapters of further parameters of the tree prior, whose mutable and immutable values the approximation of the tree prior may depend on. The operator never moves them", new ArrayList<>());
    final public Input<Integer> treePriorWarmUpInput = new Input<>("treePriorWarmUp", "the number of proposals after which the approximation of the tree prior is fitted for the first time", 500);
    final public Input<Boolean> scaleWeightByLogNumTipsInput = new Input<>("scaleWeightByLogNumTips", "whether to scale the weight by the natural logarithm of the number of tips n, so that the operator is selected with weight * ln(n). Otherwise, the weight is used as given", false);

    final public Input<Double> startEssInput = new Input<>("startEss", "the effective sample size of the tree length that has to be exceeded before the operator starts to move nodes", 25.0);

    // the share of the proposal density that is uniform, which keeps the density positive whatever the cells say
    private static final double UNIFORM_SHARE = 0.01;

    // the gain of the batch size tuning is the number of tuning steps to this negative power, which decays to zero
    // but slower than 1 / n, as the batch size has to cover a wide range
    private static final double TUNING_GAIN_EXPONENT = -0.6;

    private Mutations mutations;
    private GeneticPrior geneticPrior;
    private int numCells;
    private double batchSize;

    // the factor ln(n) or one by which the weight of the operator is scaled
    private double weightFactor;

    // the numbers of the internal nodes, in the order they were visited for the last batch
    private int[] internalNodeNrs;

    // the nodes of the last batch, in the order they were displaced in
    private Node[] batchNodes;
    private int batchCount;

    // the probabilities of the cells of the node that is displaced
    private double[] cellWeights;

    // the approximation of the tree prior, or null to draw from the genetic prior only
    private GlobalSplineTreePriorApproximation treePriorApproximation;

    // the values of the parameters of the tree prior, read at the start of each proposal
    private TreePriorParameterValues treePriorParameterValues;

    // the change of the approximate log tree prior over the nodes of the last batch
    private double approximateTreePriorChange;

    // the tree lengths seen at the proposals, which is null once the operator has started
    private ThinnedEssTracker treeLengthTracker = new ThinnedEssTracker(1);

    // the number of proposals the batch size was tuned on
    private long numTuningSteps;

    // whether the first fit of the approximation of the tree prior has been announced
    private boolean hasReportedFit;

    @Override
    public void initAndValidate() {
        this.mutations = this.mutationsInput.get();
        this.geneticPrior = this.geneticPriorInput.get();
        this.numCells = this.numCellsInput.get();
        this.cellWeights = new double[this.numCells];

        // a threshold of zero or less starts the operator at once
        if (this.startEssInput.get() <= 0.0) {
            this.treeLengthTracker = null;
        }

        // BEAST numbers the internal nodes after the leaves
        int numLeaves = this.treeInput.get().getLeafNodeCount();
        this.internalNodeNrs = new int[this.treeInput.get().getInternalNodeCount()];
        for (int i = 0; i < this.internalNodeNrs.length; i++) {
            this.internalNodeNrs[i] = numLeaves + i;
        }

        this.batchNodes = new Node[this.internalNodeNrs.length];

        this.setCoercableParameterValue(this.batchSizeInput.get());
        this.weightFactor = this.scaleWeightByLogNumTipsInput.get() ? Math.log(numLeaves) : 1.0;

        if (this.learnTreePriorInput.get()) {
            this.treePriorParameterValues = new TreePriorParameterValues(this.treePriorParametersInput.get(), this.adaptersInput.get());
            this.treePriorApproximation = new GlobalSplineTreePriorApproximation(
                    this.treePriorKnotsInput.get(), this.treePriorParameterValues.getValues().length, this.treePriorWarmUpInput.get());
        }
    }

    /**
     * Returns the weight, scaled by the logarithm of the number of tips if requested, which
     * lets the share of the proposals grow with the size of the tree.
     */
    @Override
    public double getWeight() {
        return super.getWeight() * this.weightFactor;
    }

    @Override
    public double proposal() {
        if (!this.hasStarted()) {
            return Double.NEGATIVE_INFINITY;
        }

        this.selectBatch();

        this.approximateTreePriorChange = 0.0;
        if (this.treePriorApproximation != null) {
            this.treePriorParameterValues.read();
            this.treePriorApproximation.startObservation();
        }

        double logHastingsRatio = 0.0;
        for (int i = 0; i < this.batchCount; i++) {
            double logRatio = this.displace(this.batchNodes[i]);
            if (logRatio == Double.NEGATIVE_INFINITY) {
                return Double.NEGATIVE_INFINITY;
            }
            logHastingsRatio += logRatio;
        }

        return logHastingsRatio;
    }

    /**
     * Returns whether the operator has started, which it does once the ESS of the tree length at its proposals exceeds
     * the threshold. Until then the tree length of the current state is recorded.
     */
    private boolean hasStarted() {
        if (this.treeLengthTracker == null) {
            return true;
        }

        this.treeLengthTracker.record(NodeTimes.computeTreeLength(this.treeInput.get()));
        if (!this.treeLengthTracker.hasReached(this.startEssInput.get())) {
            return false;
        }

        System.out.printf("%s: starts at an ESS of the tree length of %.1f%n", this.getID(), this.treeLengthTracker.getEss());
        this.treeLengthTracker = null;
        return true;
    }

    /**
     * Selects a random set of non-root internal nodes in random order, without looking at heights or mutations.
     * Displacing them in this order and then in the opposite order are equally likely, and the node at each step
     * sees the same neighbour heights in both directions: the nodes before it at their new heights, and the nodes
     * after it at their old heights. Hence its range and its proposal density are the same in both directions, and a
     * node without room is skipped in both. The selection has no effect on the Hastings ratio.
     */
    private void selectBatch() {
        int targetSize = Math.toIntExact(Math.round(this.batchSize));
        this.batchCount = 0;

        int numInternalNodes = this.internalNodeNrs.length;
        for (int i = 0; i < numInternalNodes && this.batchCount < targetSize; i++) {
            // shuffle the next node into place, so that every node is visited at most once
            int j = i + Randomizer.nextInt(numInternalNodes - i);
            int nodeNr = this.internalNodeNrs[j];
            this.internalNodeNrs[j] = this.internalNodeNrs[i];
            this.internalNodeNrs[i] = nodeNr;

            Node node = this.treeInput.get().getNode(nodeNr);
            if (!node.isRoot()) {
                this.batchNodes[this.batchCount++] = node;
            }
        }
    }

    /**
     * Displaces the given node to a height between its parent and its older child and returns the log Hastings
     * ratio of the move. A node whose height is not inside its range is skipped. The height is drawn from the
     * cells, and the reverse move draws from the same cells, as the heights of the neighbours and the hazards of the
     * branches do not depend on the height of the node.
     */
    private double displace(Node node) {
        double lowerHeight = NodeHeightMover.getOlderChild(node).getHeight();
        double upperHeight = node.getParent().getHeight();
        double oldHeight = node.getHeight();
        if (!(oldHeight > lowerHeight && oldHeight < upperHeight)) {
            return 0.0;
        }

        if (!this.computeCellWeights(node, lowerHeight, upperHeight)) {
            return Double.NEGATIVE_INFINITY;
        }

        double cellWidth = (upperHeight - lowerHeight) / this.numCells;
        double newHeight = lowerHeight + (this.sampleCell() + Randomizer.nextDouble()) * cellWidth;
        if (!(newHeight > lowerHeight && newHeight < upperHeight)) {
            return Double.NEGATIVE_INFINITY;
        }

        // the cell width is the same in the reverse move and cancels
        int oldCell = this.findCell(oldHeight, lowerHeight, cellWidth);
        int newCell = this.findCell(newHeight, lowerHeight, cellWidth);
        double logProposalRatio = Math.log(this.cellWeights[oldCell]) - Math.log(this.cellWeights[newCell]);

        if (this.treePriorApproximation != null) {
            double[] parameters = this.treePriorParameterValues.getValues();
            double midpoint = (lowerHeight + upperHeight) / 2.0;
            this.approximateTreePriorChange += this.treePriorApproximation.getChange(lowerHeight, upperHeight, oldHeight, newHeight, parameters);
            this.treePriorApproximation.addNode(midpoint, parameters, oldHeight - midpoint, newHeight - midpoint);
        }

        return logProposalRatio + NodeHeightMover.moveNode(this.mutations, node, newHeight, this);
    }

    /**
     * Fills the probabilities of the cells of the range with the density ∝ exp(g(h) + a(h)) at the cell centres, mixed
     * with a uniform share. Here g is the exact log genetic prior of the three branches around the node, including the
     * Jacobian of the map of their mutations, and a is the approximation of the log tree prior. Returns whether the
     * densities are finite.
     */
    private boolean computeCellWeights(Node node, double lowerHeight, double upperHeight) {
        Node olderChild = NodeHeightMover.getOlderChild(node);
        Node youngerChild = NodeHeightMover.getYoungerChild(node);
        double youngerHeight = youngerChild.getHeight();

        int numParentMutations = this.mutations.getMutations(node).size();
        int numOlderMutations = this.mutations.getMutations(olderChild).size();
        int numYoungerMutations = this.mutations.getMutations(youngerChild).size();

        // the no-change terms of the three branches are linear in the height
        double decayRate = this.geneticPrior.getMutationHazardPerLength(olderChild)
                + this.geneticPrior.getMutationHazardPerLength(youngerChild)
                - this.geneticPrior.getMutationHazardPerLength(node);

        double[] parameters = this.treePriorApproximation == null ? null : this.treePriorParameterValues.getValues();
        double cellWidth = (upperHeight - lowerHeight) / this.numCells;

        double maxLogDensity = Double.NEGATIVE_INFINITY;
        for (int i = 0; i < this.numCells; i++) {
            double height = lowerHeight + (i + 0.5) * cellWidth;

            double logDensity = -decayRate * height
                    + numParentMutations * Math.log(upperHeight - height)
                    + numOlderMutations * Math.log(height - lowerHeight)
                    + numYoungerMutations * Math.log(height - youngerHeight);
            if (this.treePriorApproximation != null) {
                logDensity += this.treePriorApproximation.getChange(lowerHeight, upperHeight, lowerHeight, height, parameters);
            }

            if (Double.isNaN(logDensity)) {
                return false;
            }
            this.cellWeights[i] = logDensity;
            maxLogDensity = Math.max(maxLogDensity, logDensity);
        }

        if (!Double.isFinite(maxLogDensity)) {
            return false;
        }

        double sum = 0.0;
        for (int i = 0; i < this.numCells; i++) {
            this.cellWeights[i] = Math.exp(this.cellWeights[i] - maxLogDensity);
            sum += this.cellWeights[i];
        }
        for (int i = 0; i < this.numCells; i++) {
            this.cellWeights[i] = (1.0 - UNIFORM_SHARE) * this.cellWeights[i] / sum + UNIFORM_SHARE / this.numCells;
        }
        return true;
    }

    private int sampleCell() {
        double target = Randomizer.nextDouble();
        double cumulative = 0.0;
        for (int i = 0; i < this.numCells; i++) {
            cumulative += this.cellWeights[i];
            if (target < cumulative) {
                return i;
            }
        }
        return this.numCells - 1;
    }

    private int findCell(double height, double lowerHeight, double cellWidth) {
        return Math.min((int) ((height - lowerHeight) / cellWidth), this.numCells - 1);
    }

    /* Tuning */

    /**
     * Passes the last proposal on to the approximation of the tree prior, and tunes the batch size with a gain that
     * decays to zero, so that the adaptation diminishes. The genetic prior and the Jacobian cancel against the
     * density of the cells up to the error of the cells, so the log acceptance ratio is the change of the log tree
     * prior minus the change the approximation predicted. Adding the latter back gives the change of the log tree
     * prior over the whole batch.
     */
    @Override
    public void optimize(double logAlpha) {
        if (this.treeLengthTracker != null) {
            return;
        }

        if (this.treePriorApproximation != null) {
            this.treePriorApproximation.finishObservation(logAlpha + this.approximateTreePriorChange);

            if (this.treePriorApproximation.isFitted() && !this.hasReportedFit) {
                this.hasReportedFit = true;
                System.out.printf("%s: tree prior approximation fitted for the first time after %d proposals, batch size %.0f%n",
                        this.getID(), this.treePriorApproximation.getNumObservations(), this.batchSize);
            }
        }

        if (this.optimiseInput.get()) {
            this.numTuningSteps++;
            double gain = Math.pow(this.numTuningSteps + 1.0, TUNING_GAIN_EXPONENT);
            double acceptance = Double.isNaN(logAlpha) ? 0.0 : Math.min(1.0, Math.exp(logAlpha));

            double delta = gain * (acceptance - this.getTargetAcceptanceProbability()) + Math.log(this.getCoercableParameterValue());
            this.setCoercableParameterValue(Math.exp(delta));
        }
    }

    @Override
    public double getTargetAcceptanceProbability() {
        return 0.5;
    }

    @Override
    public double getCoercableParameterValue() {
        return this.batchSize;
    }

    @Override
    public void setCoercableParameterValue(double value) {
        this.batchSize = Math.max(1.0, Math.min(value, this.computeMaxBatchSize()));
    }

    /** Returns the largest batch size: a batch cannot hold more than the internal nodes below the root. */
    private int computeMaxBatchSize() {
        return Math.max(this.treeInput.get().getInternalNodeCount() - 1, 1);
    }

}
