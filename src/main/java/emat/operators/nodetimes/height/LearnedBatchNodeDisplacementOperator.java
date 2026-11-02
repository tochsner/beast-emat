package emat.operators.nodetimes.height;

import beast.base.core.Description;
import beast.base.core.Input;
import beast.base.evolution.operator.TreeOperator;
import beast.base.evolution.tree.Node;
import beast.base.util.Randomizer;
import emat.state.Mutations;

@Description("Displaces the heights of a batch of internal nodes one after the other, each anywhere between its parent " +
        "and its older child, and maps the mutations on the three branches around it along with it. Every node draws its " +
        "new height from a piecewise constant density over its range, which follows the conditional density that a " +
        "NodeHeightLearner has learned, mixed with a small uniform share. The learner's predictor is fixed during a proposal, " +
        "so the proposal density is exact and the acceptance probability corrects for the error of the prediction. " +
        "The operator does nothing until the learner has been fitted for the first time.")
public class LearnedBatchNodeDisplacementOperator extends TreeOperator {

    final public Input<Mutations> mutationsInput = new Input<>("mutations", "the mutations to operate on", Input.Validate.REQUIRED);
    final public Input<NodeHeightLearner> learnerInput = new Input<>("learner", "the learner of the conditional density of a node height", Input.Validate.REQUIRED);
    final public Input<Integer> batchSizeInput = new Input<>("batchSize", "the initial number of nodes that are displaced together, which is tuned if optimise is set, and which is capped at the number of internal nodes below the root", 128);
    final public Input<Integer> numCellsInput = new Input<>("numCells", "the number of cells of equal width of the proposal density of a node height", 50);
    final public Input<Boolean> optimiseInput = new Input<>("optimise", "whether to tune the batch size", true);
    final public Input<Boolean> scaleWeightByLogNumTipsInput = new Input<>("scaleWeightByLogNumTips", "whether to scale the weight by the natural logarithm of the number of tips n, so that the operator is selected with weight * ln(n). Otherwise, the weight is used as given", false);

    // the share of the proposal density that is uniform, which keeps the density positive whatever the predictor says
    private static final double UNIFORM_SHARE = 0.02;

    // the gain of the batch size tuning is the number of tuning steps to this negative power, which decays to zero
    // but slower than 1 / n, as the batch size has to cover a wide range
    private static final double TUNING_GAIN_EXPONENT = -0.6;

    private Mutations mutations;
    private NodeHeightLearner learner;
    private int numCells;
    private double batchSize;

    // the factor ln(n) or one by which the weight of the operator is scaled
    private double weightFactor;

    // the numbers of the internal nodes, in the order they were visited for the last batch
    private int[] internalNodeNrs;

    // the nodes of the last batch, in the order they were displaced in
    private Node[] batchNodes;
    private int batchCount;

    // the centres of the cells of the node that is displaced
    private double[] cellCentres;

    // the number of proposals the batch size was tuned on
    private long numTuningSteps;

    @Override
    public void initAndValidate() {
        this.mutations = this.mutationsInput.get();
        this.learner = this.learnerInput.get();
        this.numCells = this.numCellsInput.get();
        this.cellCentres = new double[this.numCells];

        // BEAST numbers the internal nodes after the leaves
        int numLeaves = this.treeInput.get().getLeafNodeCount();
        this.internalNodeNrs = new int[this.treeInput.get().getInternalNodeCount()];
        for (int i = 0; i < this.internalNodeNrs.length; i++) {
            this.internalNodeNrs[i] = numLeaves + i;
        }

        this.batchNodes = new Node[this.internalNodeNrs.length];

        this.setCoercableParameterValue(this.batchSizeInput.get());
        this.weightFactor = this.scaleWeightByLogNumTipsInput.get() ? Math.log(numLeaves) : 1.0;
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
        if (!this.learner.isFitted()) {
            return Double.NEGATIVE_INFINITY;
        }

        this.learner.refreshParameters();
        this.selectBatch();

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
     * Selects a random set of non-root internal nodes in random order, without looking at heights or mutations.
     * Displacing them in this order and then in the opposite order are equally likely, and the node at each step
     * sees the same neighbour heights in both directions: the nodes before it at their new heights, and the nodes
     * after it at their old heights. Hence its context, its range and its proposal density are the same in both
     * directions, and a node without room is skipped in both. The selection has no effect on the Hastings ratio.
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
     * ratio of the move. A node whose height is not inside its range is skipped.
     */
    private double displace(Node node) {
        double lowerHeight = NodeHeightMover.getOlderChild(node).getHeight();
        double upperHeight = node.getParent().getHeight();
        double oldHeight = node.getHeight();
        if (!(oldHeight > lowerHeight && oldHeight < upperHeight)) {
            return 0.0;
        }

        double[] cellWeights = this.computeCellWeights(node, lowerHeight, upperHeight);
        if (cellWeights == null) {
            return Double.NEGATIVE_INFINITY;
        }

        double cellWidth = (upperHeight - lowerHeight) / this.numCells;
        double newHeight = lowerHeight + (this.sampleCell(cellWeights) + Randomizer.nextDouble()) * cellWidth;
        if (!(newHeight > lowerHeight && newHeight < upperHeight)) {
            return Double.NEGATIVE_INFINITY;
        }

        // the cell width is the same in the reverse move and cancels
        int oldCell = this.findCell(oldHeight, lowerHeight, cellWidth);
        int newCell = this.findCell(newHeight, lowerHeight, cellWidth);
        double logProposalRatio = Math.log(cellWeights[oldCell]) - Math.log(cellWeights[newCell]);

        return logProposalRatio + NodeHeightMover.moveNode(this.mutations, node, newHeight, this);
    }

    /**
     * Returns the probabilities of the cells of the range, which are the predicted weights at the cell centres mixed
     * with a uniform share, or null if the prediction is not finite.
     */
    private double[] computeCellWeights(Node node, double lowerHeight, double upperHeight) {
        double cellWidth = (upperHeight - lowerHeight) / this.numCells;
        for (int i = 0; i < this.numCells; i++) {
            this.cellCentres[i] = lowerHeight + (i + 0.5) * cellWidth;
        }

        NodeHeightContext context = this.learner.createContext(node, this.mutations);
        double[] cellWeights = this.learner.getPredictor().predictWeights(context, this.cellCentres);

        for (int i = 0; i < this.numCells; i++) {
            if (!Double.isFinite(cellWeights[i])) {
                return null;
            }
            cellWeights[i] = (1.0 - UNIFORM_SHARE) * cellWeights[i] + UNIFORM_SHARE / this.numCells;
        }
        return cellWeights;
    }

    private int sampleCell(double[] cellWeights) {
        double target = Randomizer.nextDouble();
        double cumulative = 0.0;
        for (int i = 0; i < this.numCells; i++) {
            cumulative += cellWeights[i];
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
     * Tunes the batch size once the learner is fitted, with a gain that decays to zero, so that the adaptation diminishes.
     * The log batch size moves up if the acceptance probability is above the target and down if it is below.
     */
    @Override
    public void optimize(double logAlpha) {
        if (!this.optimiseInput.get() || !this.learner.isFitted()) {
            return;
        }

        this.numTuningSteps++;
        double gain = Math.pow(this.numTuningSteps + 1.0, TUNING_GAIN_EXPONENT);
        double acceptance = Double.isNaN(logAlpha) ? 0.0 : Math.min(1.0, Math.exp(logAlpha));

        double delta = gain * (acceptance - this.getTargetAcceptanceProbability()) + Math.log(this.getCoercableParameterValue());
        this.setCoercableParameterValue(Math.exp(delta));
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
