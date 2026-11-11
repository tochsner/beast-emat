package esat.operators.nodetimes;

import adapters.Adapter;
import beast.base.core.Description;
import beast.base.core.Input;
import beast.base.evolution.operator.TreeOperator;
import beast.base.evolution.tree.Node;
import beast.base.evolution.tree.Tree;
import beast.base.inference.StateNode;
import beast.base.spec.type.IntVector;
import beast.base.spec.type.RealScalar;
import beast.base.spec.type.RealVector;
import beast.base.util.Randomizer;
import emat.operators.nodetimes.batch.GlobalSplineTreePriorApproximation;
import emat.operators.nodetimes.batch.LearnedTreePrior;
import emat.operators.nodetimes.batch.TruncatedDensitySampler;
import esat.prior.GeneticPrior;

import java.util.ArrayList;
import java.util.List;

@Description("Displaces the heights of a batch of internal nodes one after the other, each between its oldest child and " +
    "its parent. Every node draws its new height from the exponential tilt of the genetic prior at the midpoint of its range " +
    "times a learned approximation of the tree prior, so only the errors of these approximations enter the acceptance probability.")
public class BatchNodeDisplacementOperator extends TreeOperator {

    final public Input<GeneticPrior> geneticPriorInput = new Input<>("geneticPrior", "");
    final public Input<Integer> batchSizeInput = new Input<>("batchSize", "the initial number of nodes that are displaced together, which is tuned if optimise is set, and which is capped at the number of internal nodes below the root", 128);
    final public Input<Boolean> optimiseInput = new Input<>("optimise", "", true);
    final public Input<Boolean> learnTreePriorInput = new Input<>("learnTreePrior", "whether to learn an approximation of the tree prior of its own, instead of drawing from the genetic prior only", true);
    final public Input<Integer> treePriorKnotsInput = new Input<>("treePriorKnots", "the number of knots along the height between which the slope of the approximation of the tree prior is interpolated", 20);
    final public Input<List<StateNode>> treePriorParametersInput = new Input<>("treePriorParameter", "real scalar, real vector or integer vector parameters of the tree prior, which the approximation of the tree prior may depend on", new ArrayList<>());
    final public Input<List<Adapter>> adaptersInput = new Input<>("adapter", "adapters of further parameters of the tree prior, whose mutable and immutable values the approximation of the tree prior may depend on. The operator never moves them", new ArrayList<>());
    final public Input<Integer> treePriorWarmUpInput = new Input<>("treePriorWarmUp", "the number of proposals after which the approximation of the tree prior is fitted for the first time", 500);
    final public Input<Boolean> interpolateGeneticPriorInput = new Input<>("interpolateGeneticPrior", "whether to draw from a piecewise exponential interpolation of the genetic prior between the oldest child and the parent of a node, instead of from its tangent at the midpoint", true);
    final public Input<Boolean> scaleWeightByLogNumTipsInput = new Input<>("scaleWeightByLogNumTips", "whether to scale the weight by the natural logarithm of the number of tips n, so that the operator is selected with weight * ln(n). Otherwise, the weight is used as given", false);

    // the adapters have no use for the node, as the parameters of the tree prior do not belong to one
    private static final int NO_NODE = -1;

    private Tree tree;
    private GeneticPrior geneticPrior;
    private double batchSize;

    // the factor ln(n) or one by which the weight of the operator is scaled
    private double weightFactor;

    // the gain of the batch size tuning decays with the number of proposals, but not below this value, as the
    // approximation of the tree prior makes large batches acceptable and the batch size has to cover a wide range
    private static final double MIN_TUNING_GAIN = 0.01;

    // the numbers of the internal nodes, in the order they were visited for the last batch
    private int[] internalNodeNrs;

    // the nodes of the last batch, in the order they were displaced in
    private Node[] batchNodes;
    private int batchCount;

    // the approximation of the tree prior, or null to draw from the genetic prior only
    private LearnedTreePrior treePriorApproximation;

    // the values of the parameters of the tree prior at the start of the last proposal
    private double[] treePriorParameterValues;

    // the number of proposals the batch size was tuned on
    private long numTuningSteps;

    // whether the first fit of the approximation of the tree prior has been announced
    private boolean hasReportedFit;

    // the fractions of the range of a node at which the genetic prior is evaluated if it is interpolated, which crowd
    // towards both ends of the range, where the genetic prior can fall off steeply
    private static final double[] KNOT_FRACTIONS = createKnotFractions();

    // the heights and the unnormalised log densities of the interpolation of the node that is displaced, and scratch space
    private double[] knotHeights;
    private double[] knotLogDensities;
    private double[] knotTreePriors;
    private double[] segmentMasses;

    // the change of the log density that the last node was drawn from, between its old and its new height
    private double logDensityChange;

    // over the nodes of the last batch, the change of the log density they were drawn from minus the change of the genetic prior,
    // which is the change of the tree prior that the draws explain
    private double treePriorChangeCorrection;

    @Override
    public void initAndValidate() {
        this.tree = this.treeInput.get();
        this.geneticPrior = this.geneticPriorInput.get();

        // BEAST numbers the internal nodes after the leaves
        int numLeaves = this.tree.getLeafNodeCount();
        this.internalNodeNrs = new int[this.tree.getInternalNodeCount()];
        for (int i = 0; i < this.internalNodeNrs.length; i++) {
            this.internalNodeNrs[i] = numLeaves + i;
        }

        this.batchNodes = new Node[this.internalNodeNrs.length];

        this.knotHeights = new double[KNOT_FRACTIONS.length + 2];
        this.knotLogDensities = new double[KNOT_FRACTIONS.length + 2];
        this.knotTreePriors = new double[KNOT_FRACTIONS.length + 2];
        this.segmentMasses = new double[KNOT_FRACTIONS.length + 1];

        this.setCoercableParameterValue(this.batchSizeInput.get());
        this.weightFactor = this.scaleWeightByLogNumTipsInput.get() ? Math.log(this.tree.getLeafNodeCount()) : 1.0;

        if (this.learnTreePriorInput.get()) {
            this.treePriorParameterValues = new double[this.countTreePriorParameterValues()];
            this.treePriorApproximation = new GlobalSplineTreePriorApproximation(
                    this.treePriorKnotsInput.get(), this.treePriorParameterValues.length, this.treePriorWarmUpInput.get());
        }
    }

    /**
     * Returns the number of values of the parameters of the tree prior, and checks that they are real or integer
     * vectors. The values of the adapters follow those of the parameters.
     */
    private int countTreePriorParameterValues() {
        int numValues = 0;
        for (Adapter adapter : this.adaptersInput.get()) {
            numValues += adapter.getNumMutable() + adapter.getNumImmutable();
        }
        for (StateNode parameter : this.treePriorParametersInput.get()) {
            if (parameter instanceof RealScalar<?>) {
                numValues++;
            } else if (parameter instanceof RealVector<?> vector) {
                numValues += vector.size();
            } else if (parameter instanceof IntVector<?> vector) {
                numValues += vector.size();
            } else {
                throw new IllegalArgumentException("The tree prior parameter " + parameter.getID() + " is neither a real scalar nor a real or integer vector.");
            }
        }
        return numValues;
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
        this.geneticPrior.refreshModel();
        this.selectBatch();

        this.treePriorChangeCorrection = 0.0;
        if (this.treePriorApproximation != null) {
            this.readTreePriorParameterValues();
            this.treePriorApproximation.startObservation();
        }

        double logHastingsRatio = 0.0;
        for (int i = 0; i < this.batchCount; i++) {
            Node node = this.batchNodes[i];

            // the range of a node depends on the nodes that were displaced before it
            double minHeight = this.computeMinHeight(node);
            double maxHeight = node.getParent().getHeight();
            if (maxHeight <= minHeight) {
                continue;
            }

            logHastingsRatio += this.displace(node, minHeight, maxHeight);
        }

        return logHastingsRatio;
    }

    /** Returns the height of the oldest child of the given node. */
    private double computeMinHeight(Node node) {
        double minHeight = Double.NEGATIVE_INFINITY;
        for (Node child : node.getChildren()) {
            minHeight = Math.max(minHeight, child.getHeight());
        }
        return minHeight;
    }

    /**
     * Reads the current values of the parameters of the tree prior, and then the mutable and immutable values of the
     * adapters, into the array that is handed to the approximation.
     */
    private void readTreePriorParameterValues() {
        int index = 0;
        for (StateNode parameter : this.treePriorParametersInput.get()) {
            if (parameter instanceof RealScalar<?> scalar) {
                this.treePriorParameterValues[index++] = scalar.get();
            } else if (parameter instanceof RealVector<?> vector) {
                for (int i = 0; i < vector.size(); i++) {
                    this.treePriorParameterValues[index++] = vector.get(i);
                }
            } else if (parameter instanceof IntVector<?> vector) {
                for (int i = 0; i < vector.size(); i++) {
                    this.treePriorParameterValues[index++] = vector.get(i);
                }
            }
        }

        for (Adapter adapter : this.adaptersInput.get()) {
            for (double value : adapter.getMutable(NO_NODE)) {
                this.treePriorParameterValues[index++] = value;
            }
            for (double value : adapter.getImmutable(NO_NODE)) {
                this.treePriorParameterValues[index++] = value;
            }
        }
    }

    /**
     * Selects a random set of non-root internal nodes in random order, without looking at heights or mutations.
     * Displacing them in this order and then in the opposite order are equally likely, and the node at each step
     * sees the same neighbour heights in both directions: the nodes before it at their new heights, and the nodes
     * after it at their old heights. Hence its range and its rate are the same in both directions, and a node
     * without room is skipped in both. The normalisations cancel, so the selection has no effect on the Hastings
     * ratio.
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

            Node node = this.tree.getNode(nodeNr);
            if (!node.isRoot()) {
                this.batchNodes[this.batchCount++] = node;
            }
        }
    }

    /**
     * Displaces the given node within the given range and returns the log Hastings ratio. The new height h is
     * drawn from a density q(h) that approximates the product of the genetic prior and the tree prior as a function
     * of the height. Neither the range nor q depend on the height of the node, so the reverse move draws from the
     * same density and the normalisation cancels, which leaves log q(old) - log q(new) as the Hastings ratio. The
     * deviation of q from the genetic prior stays in the acceptance probability.
     */
    private double displace(Node node, double minHeight, double maxHeight) {
        double midpoint = (minHeight + maxHeight) / 2.0;
        double oldHeight = node.getHeight();

        double newHeight = this.interpolateGeneticPriorInput.get()
                ? this.sampleInterpolatedHeight(node, minHeight, maxHeight, oldHeight)
                : this.sampleTangentHeight(node, minHeight, maxHeight, oldHeight, midpoint);

        // the change of q minus the change of the genetic prior is what q predicts for the tree prior, which the fit is corrected by

        this.treePriorChangeCorrection += this.logDensityChange - (this.computeGeneticLogP(node, newHeight) - this.computeGeneticLogP(node, oldHeight));

        node.setHeight(newHeight);

        // remember the move for the fit of the approximation

        if (this.treePriorApproximation != null) {
            this.treePriorApproximation.addNode(midpoint, this.treePriorParameterValues, oldHeight - midpoint, newHeight - midpoint);
        }

        return -this.logDensityChange;
    }

    /**
     * Draws a height from the density ∝ exp(-k h + a(h)), with the tangent of the log genetic prior at the midpoint
     * of the range as -k h and the learned approximation a of the log tree prior, and sets the change of its log density.
     */
    private double sampleTangentHeight(Node node, double minHeight, double maxHeight, double oldHeight, double midpoint) {
        double geneticDecayRate = -this.computeGeneticSlope(node, midpoint);
        double newHeight;
        double approximateChange = 0.0;

        if (this.treePriorApproximation == null) {
            newHeight = TruncatedDensitySampler.sampleExponential(geneticDecayRate, minHeight, maxHeight);
        } else {
            newHeight = this.treePriorApproximation.sampleHeight(geneticDecayRate, minHeight, maxHeight, this.treePriorParameterValues);
            approximateChange = this.treePriorApproximation.getChange(minHeight, maxHeight, oldHeight, newHeight, this.treePriorParameterValues);
        }

        this.logDensityChange = approximateChange - geneticDecayRate * (newHeight - oldHeight);
        return newHeight;
    }

    /**
     * Draws a height from the piecewise exponential density that interpolates the log of the genetic prior times the
     * learned approximation of the tree prior, which is evaluated at knots that crowd towards both ends of the range.
     * The ends of the range are not evaluated, as the genetic prior vanishes there if the branch has changed state, but
     * continue the tangent at the knot next to them. Sets the change of its log density. If a knot has no finite density, the draw is
     * uniform.
     */
    private double sampleInterpolatedHeight(Node node, double minHeight, double maxHeight, double oldHeight) {
        int numKnots = this.computeKnots(node, minHeight, maxHeight);
        if (numKnots == 0) {
            this.logDensityChange = 0.0;
            return minHeight + Randomizer.nextDouble() * (maxHeight - minHeight);
        }

        double newHeight = PiecewiseExponentialSampler.sample(this.knotHeights, this.knotLogDensities, numKnots, this.segmentMasses);

        // rounding must not push the value out of the range
        newHeight = Math.min(Math.max(newHeight, minHeight), maxHeight);

        this.logDensityChange = PiecewiseExponentialSampler.evaluate(this.knotHeights, this.knotLogDensities, numKnots, newHeight)
                - PiecewiseExponentialSampler.evaluate(this.knotHeights, this.knotLogDensities, numKnots, oldHeight);
        return newHeight;
    }

    /**
     * Fills the knots of the interpolation, with the ends of the range as the first and last knot, and returns their
     * number, or zero if a density is not finite.
     */
    private int computeKnots(Node node, double minHeight, double maxHeight) {
        int numInnerKnots = KNOT_FRACTIONS.length;
        double width = maxHeight - minHeight;

        for (int i = 0; i < numInnerKnots; i++) {
            double height = minHeight + KNOT_FRACTIONS[i] * width;
            this.knotHeights[i + 1] = height;

            double treePrior = this.treePriorApproximation == null ? 0.0
                    : this.treePriorApproximation.getChange(minHeight, maxHeight, minHeight, height, this.treePriorParameterValues);
            this.knotTreePriors[i + 1] = treePrior;

            double logDensity = this.computeGeneticLogP(node, height) + treePrior;
            if (!Double.isFinite(logDensity)) {
                return 0;
            }
            this.knotLogDensities[i + 1] = logDensity;
        }

        // the tangent at the first and last inner knot continues to the ends of the range

        double lowerSlope = this.computeGeneticSlope(node, this.knotHeights[1]) + this.computeTreePriorSlope(1);
        double upperSlope = this.computeGeneticSlope(node, this.knotHeights[numInnerKnots]) + this.computeTreePriorSlope(numInnerKnots - 1);
        this.knotHeights[0] = minHeight;
        this.knotLogDensities[0] = this.knotLogDensities[1] - lowerSlope * (this.knotHeights[1] - minHeight);
        this.knotHeights[numInnerKnots + 1] = maxHeight;
        this.knotLogDensities[numInnerKnots + 1] = this.knotLogDensities[numInnerKnots] + upperSlope * (maxHeight - this.knotHeights[numInnerKnots]);

        if (!Double.isFinite(this.knotLogDensities[0]) || !Double.isFinite(this.knotLogDensities[numInnerKnots + 1])) {
            return 0;
        }
        return numInnerKnots + 2;
    }

    /** Returns the slope of the learned tree prior between the given inner knot and the next, which is zero without the approximation. */
    private double computeTreePriorSlope(int knot) {
        return (this.knotTreePriors[knot + 1] - this.knotTreePriors[knot]) / (this.knotHeights[knot + 1] - this.knotHeights[knot]);
    }

    /** Returns the fractions of the range at which the genetic prior is evaluated: powers of a half towards both ends and a few in between. */
    private static double[] createKnotFractions() {
        double[] fractions = new double[21];
        for (int i = 0; i < 9; i++) {
            fractions[i] = Math.pow(0.5, 10 - i);
            fractions[20 - i] = 1.0 - fractions[i];
        }
        fractions[9] = 0.375;
        fractions[10] = 0.5;
        fractions[11] = 0.625;
        return fractions;
    }

    /**
     * Returns the slope of the genetic prior in the height of the given node, which is the sum of the slopes
     * of the branch above the node, which shortens as the node rises, and of the branches below, which lengthen.
     * A slope that is not finite is replaced by zero, which gives a uniform draw.
     */
    private double computeGeneticSlope(Node node, double height) {
        double slope = -this.geneticPrior.computeBranchLogPSlope(node, node.getParent().getHeight() - height);
        for (Node child : node.getChildren()) {
            slope += this.geneticPrior.computeBranchLogPSlope(child, height - child.getHeight());
        }
        return Double.isFinite(slope) ? slope : 0.0;
    }

    /** Returns the part of the genetic prior that depends on the height of the given node when it is at the given height. */
    private double computeGeneticLogP(Node node, double height) {
        double logP = this.geneticPrior.computeBranchLogP(node, node.getParent().getHeight() - height);
        for (Node child : node.getChildren()) {
            logP += this.geneticPrior.computeBranchLogP(child, height - child.getHeight());
        }
        return logP;
    }

    /* Tuning */

    /**
     * Passes the last proposal on to the approximation of the tree prior, and tunes the batch size with a gain
     * that decays with the number of proposals down to a floor. The log acceptance ratio is the change of the log
     * tree prior plus the change of the genetic prior minus the change of the log densities the heights were drawn
     * from. Adding the change of the log densities back and removing the change of the genetic prior gives the change
     * of the log tree prior over the whole batch.
     */
    @Override
    public void optimize(double logAlpha) {
        if (this.treePriorApproximation != null) {
            this.treePriorApproximation.finishObservation(logAlpha + this.treePriorChangeCorrection);
        }

        if (this.treePriorApproximation != null && this.treePriorApproximation.isFitted() && !this.hasReportedFit) {
            this.hasReportedFit = true;
            System.out.printf("%s: tree prior approximation fitted for the first time after %d proposals, batch size %.0f%n",
                    this.getID(), this.treePriorApproximation.getNumObservations(), this.batchSize);
        }

        if (this.optimiseInput.get()) {
            this.numTuningSteps++;
            double gain = Math.max(1.0 / (this.numTuningSteps + 1.0), MIN_TUNING_GAIN);
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
        return Math.max(this.tree.getInternalNodeCount() - 1, 1);
    }

}
