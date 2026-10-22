package emat.operators;

import beast.base.core.Description;
import beast.base.core.Input;
import beast.base.evolution.operator.TreeOperator;
import beast.base.evolution.tree.Node;
import beast.base.evolution.tree.Tree;
import beast.base.inference.StateNode;
import beast.base.spec.inference.parameter.RealScalarParam;
import beast.base.util.Randomizer;
import emat.state.Mutation;
import emat.state.Mutations;
import org.apache.commons.math4.legacy.linear.Array2DRowRealMatrix;
import org.apache.commons.math4.legacy.linear.ArrayRealVector;
import org.apache.commons.math4.legacy.linear.CholeskyDecomposition;
import org.apache.commons.math4.legacy.linear.RealMatrix;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

@Description("Proposes a joint move of the population size and the growth rate of the tree prior, the heights of all " +
        "internal nodes and the clock rate, like the CoalescentWarpOperator, but without the formula of the tree prior. " +
        "It learns from the chain how the quantiles of the heights of the internal nodes depend on the two parameters, " +
        "a regression of their logarithms. The fraction of internal nodes below a height, interpolated linearly between the " +
        "quantiles, is a time change, and every node keeps the interval to its older child in that time. Proposals are " +
        "rejected until the first fit.")
public class QuantileWarpOperator extends TreeOperator {

    // the fractions of the internal nodes below the quantiles that the map is built on
    private static final double[] LEVELS = {0.02, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50,
            0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90, 0.95, 0.98};

    // the number of features of an observation: the logarithm of the population size and the growth rate
    private static final int NUM_FEATURES = 2;

    final public Input<Mutations> mutationsInput = new Input<>("mutations", "the mutations to move along with the tree", Input.Validate.REQUIRED);
    final public Input<StateNode> popSizeInput = new Input<>("popSize", "the population size of the tree prior", Input.Validate.REQUIRED);
    final public Input<StateNode> growthRateInput = new Input<>("growthRate", "the growth rate of the tree prior", Input.Validate.REQUIRED);
    final public Input<StateNode> clockRateInput = new Input<>("clockRate", "the clock rate, which is set so that its product with the tree length stays the same", Input.Validate.REQUIRED);
    final public Input<Double> sdLogPopSizeInput = new Input<>("sdLogPopSize", "the standard deviation of the step in the logarithm of the population size, before scaling", Input.Validate.REQUIRED);
    final public Input<Double> sdGrowthRateInput = new Input<>("sdGrowthRate", "the standard deviation of the step in the growth rate, before scaling", Input.Validate.REQUIRED);
    final public Input<Double> correlationInput = new Input<>("correlation", "the correlation between the two steps", Input.Validate.REQUIRED);
    final public Input<Double> scaleFactorInput = new Input<>("scaleFactor", "the factor by which both standard deviations are scaled, which is tuned if optimise is set", 1.0);
    final public Input<Boolean> optimiseInput = new Input<>("optimise", "whether to tune the scale factor towards the target acceptance probability", true);
    final public Input<Integer> observeEveryInput = new Input<>("observeEvery", "the number of proposals between two observations of the state", 5);
    final public Input<Integer> minObservationsInput = new Input<>("minObservations", "the number of observations at which the map is fitted for the first time, after which the fit is repeated whenever the number of observations doubles", 400);

    private Mutations mutations;
    private Tree tree;
    private RealScalarParam<?> popSize;
    private RealScalarParam<?> growthRate;
    private RealScalarParam<?> clockRate;

    private double scaleFactor;

    // the observations of the state: the features followed by the logarithms of the quantiles
    private final List<double[]> observations = new ArrayList<>();
    private long numProposals;
    private int nextFitSize;

    // the fit: the means of the features and, per quantile, the intercept and the slopes on the centred features
    private boolean isFitted;
    private final double[] featureMeans = new double[NUM_FEATURES];
    private final double[] intercepts = new double[LEVELS.length];
    private final double[][] slopes = new double[LEVELS.length][NUM_FEATURES];

    // the fractions of the internal nodes below the knots of the time change, starting at zero
    private static final double[] KNOT_FRACTIONS = new double[LEVELS.length + 1];

    static {
        System.arraycopy(LEVELS, 0, KNOT_FRACTIONS, 1, LEVELS.length);
    }

    // the heights of the knots of the time change before and after the last move, starting at zero
    private final double[] oldKnots = new double[LEVELS.length + 1];
    private final double[] newKnots = new double[LEVELS.length + 1];

    // scratch space
    private double[] oldHeights;
    private double[] internalHeights;

    @Override
    public void initAndValidate() {
        this.tree = this.treeInput.get();
        this.mutations = this.mutationsInput.get();
        this.popSize = this.getScalarParameter(this.popSizeInput.get());
        this.growthRate = this.getScalarParameter(this.growthRateInput.get());
        this.clockRate = this.getScalarParameter(this.clockRateInput.get());
        this.scaleFactor = this.scaleFactorInput.get();
        this.nextFitSize = this.minObservationsInput.get();
        this.oldHeights = new double[this.tree.getNodeCount()];
        this.internalHeights = new double[this.tree.getInternalNodeCount()];
    }

    private RealScalarParam<?> getScalarParameter(StateNode parameter) {
        if (parameter instanceof RealScalarParam<?> scalar) {
            return scalar;
        }
        throw new IllegalArgumentException("The parameter " + parameter.getID() + " must be a real scalar parameter.");
    }

    @Override
    public double proposal() {
        this.observeState();
        if (!this.isFitted) {
            return Double.NEGATIVE_INFINITY;
        }

        double firstNormal = Randomizer.nextGaussian();
        double secondNormal = Randomizer.nextGaussian();
        double correlation = this.correlationInput.get();

        double logPopSizeShift = this.scaleFactor * this.sdLogPopSizeInput.get() * firstNormal;
        double growthRateShift = this.scaleFactor * this.sdGrowthRateInput.get()
                * (correlation * firstNormal + Math.sqrt(1.0 - correlation * correlation) * secondNormal);

        return this.warp(logPopSizeShift, growthRateShift);
    }

    /* Learning */

    /**
     * Records the state of every few proposals, and fits the map when enough observations are there for the first
     * time or have doubled since the last fit. The fit only changes between proposals, so a proposal and its reverse
     * use the same map.
     */
    private void observeState() {
        this.numProposals++;
        if (this.numProposals % this.observeEveryInput.get() != 0) {
            return;
        }

        this.observe(Math.log(this.popSize.get()), this.growthRate.get(), this.computeLogQuantiles());
        if (this.observations.size() >= this.nextFitSize) {
            this.fit();
            this.nextFitSize = 2 * this.observations.size();
        }
    }

    /** Adds an observation of the logarithm of the population size, the growth rate and the logarithms of the quantiles. */
    public void observe(double logPopSize, double growthRate, double[] logQuantiles) {
        double[] observation = new double[NUM_FEATURES + LEVELS.length];
        observation[0] = logPopSize;
        observation[1] = growthRate;
        System.arraycopy(logQuantiles, 0, observation, NUM_FEATURES, LEVELS.length);
        this.observations.add(observation);
    }

    /** Returns the logarithms of the quantiles of the heights of the internal nodes of the tree. */
    private double[] computeLogQuantiles() {
        int numInternalNodes = 0;
        for (Node node : this.tree.getNodesAsArray()) {
            if (!node.isLeaf()) {
                this.internalHeights[numInternalNodes++] = node.getHeight();
            }
        }
        Arrays.sort(this.internalHeights, 0, numInternalNodes);

        double[] logQuantiles = new double[LEVELS.length];
        for (int i = 0; i < LEVELS.length; i++) {
            double position = LEVELS[i] * (numInternalNodes - 1);
            int below = (int) Math.floor(position);
            int above = Math.min(below + 1, numInternalNodes - 1);
            double fraction = position - below;
            logQuantiles[i] = Math.log(this.internalHeights[below] * (1.0 - fraction) + this.internalHeights[above] * fraction);
        }
        return logQuantiles;
    }

    /**
     * Fits the logarithm of every quantile as a linear function of the features by least squares, on the most recent
     * half of the observations, which leaves out the early drift of the chain. A fit that is not finite is ignored.
     */
    public void fit() {
        int numUsed = Math.max(this.observations.size() / 2, Math.min(this.observations.size(), NUM_FEATURES + 2));
        List<double[]> used = this.observations.subList(this.observations.size() - numUsed, this.observations.size());

        double[] means = new double[NUM_FEATURES];
        for (double[] observation : used) {
            for (int f = 0; f < NUM_FEATURES; f++) {
                means[f] += observation[f] / numUsed;
            }
        }

        // the normal equations of the centred features with a constant
        int size = NUM_FEATURES + 1;
        double[][] gram = new double[size][size];
        double[][] moments = new double[size][LEVELS.length];
        for (double[] observation : used) {
            double[] row = {1.0, observation[0] - means[0], observation[1] - means[1]};
            for (int a = 0; a < size; a++) {
                for (int b = 0; b < size; b++) {
                    gram[a][b] += row[a] * row[b];
                }
                for (int q = 0; q < LEVELS.length; q++) {
                    moments[a][q] += row[a] * observation[NUM_FEATURES + q];
                }
            }
        }
        for (int a = 0; a < size; a++) {
            gram[a][a] += 1e-9 * (1.0 + gram[a][a]);
        }

        RealMatrix solution;
        try {
            solution = new CholeskyDecomposition(new Array2DRowRealMatrix(gram, false)).getSolver().solve(new Array2DRowRealMatrix(moments, false));
        } catch (RuntimeException exception) {
            return;
        }

        for (int q = 0; q < LEVELS.length; q++) {
            if (!Double.isFinite(solution.getEntry(0, q)) || !Double.isFinite(solution.getEntry(1, q)) || !Double.isFinite(solution.getEntry(2, q))) {
                return;
            }
        }

        System.arraycopy(means, 0, this.featureMeans, 0, NUM_FEATURES);
        for (int q = 0; q < LEVELS.length; q++) {
            this.intercepts[q] = solution.getEntry(0, q);
            this.slopes[q][0] = solution.getEntry(1, q);
            this.slopes[q][1] = solution.getEntry(2, q);
        }
        this.isFitted = true;
    }

    /* Move */

    /**
     * Shifts the logarithm of the population size and the growth rate by the given amounts, and maps the tree and
     * the clock rate accordingly. Returns the log Hastings ratio, which is the log Jacobian of the map of the heights
     * of the internal nodes and the clock rate, the shift of the logarithm of the population size, and one term for
     * every mutation. A step with the opposite shifts undoes the move and has the negated Hastings ratio.
     */
    public double warp(double logPopSizeShift, double growthRateShift) {
        double oldLogPopSize = Math.log(this.popSize.get());
        double newPopSize = this.popSize.get() * Math.exp(logPopSizeShift);
        double newGrowthRate = this.growthRate.get() + growthRateShift;
        if (!this.popSize.isValid(newPopSize) || !this.growthRate.isValid(newGrowthRate)
                || !this.buildMap(oldLogPopSize, this.growthRate.get(), Math.log(newPopSize), newGrowthRate)) {
            return Double.NEGATIVE_INFINITY;
        }

        Node[] nodes = this.tree.getNodesAsArray();
        for (Node node : nodes) {
            this.oldHeights[node.getNr()] = node.getHeight();
        }

        double lengthBefore = this.computeTreeLength();
        double logJacobian = this.warpHeights(this.tree.getRoot());
        if (!Double.isFinite(logJacobian)) {
            return Double.NEGATIVE_INFINITY;
        }

        double lengthAfter = this.computeTreeLength();
        double newClockRate = this.clockRate.get() * lengthBefore / lengthAfter;
        if (!this.clockRate.isValid(newClockRate)) {
            return Double.NEGATIVE_INFINITY;
        }

        double logHastingsRatio = logJacobian + Math.log(lengthBefore / lengthAfter) + logPopSizeShift;
        logHastingsRatio += this.moveMutations(nodes);

        this.popSize.set(newPopSize);
        this.growthRate.set(newGrowthRate);
        this.clockRate.set(newClockRate);

        return logHastingsRatio;
    }

    /**
     * Moves every internal node, after the nodes below it, to the height whose time above its older child under the new
     * time change equals the time it had under the old one. Returns the sum over the internal nodes of the log ratio
     * of the old to the new slope of the time change, which is the log Jacobian of the map, or a value that is not
     * finite if a node would not end up above its children.
     */
    private double warpHeights(Node node) {
        if (node.isLeaf()) {
            return 0.0;
        }

        double logJacobian = this.warpHeights(node.getLeft()) + this.warpHeights(node.getRight());

        double oldHeight = this.oldHeights[node.getNr()];
        double oldChildHeight = Math.max(this.oldHeights[node.getLeft().getNr()], this.oldHeights[node.getRight().getNr()]);
        double newChildHeight = Math.max(node.getLeft().getHeight(), node.getRight().getHeight());

        double newTime = this.computeTime(this.newKnots, newChildHeight)
                + this.computeTime(this.oldKnots, oldHeight)
                - this.computeTime(this.oldKnots, oldChildHeight);
        double newHeight = this.computeHeight(this.newKnots, newTime);
        if (!(newHeight > newChildHeight)) {
            return Double.NaN;
        }

        node.setHeight(newHeight);

        return logJacobian + this.computeLogSlope(this.oldKnots, oldHeight) - this.computeLogSlope(this.newKnots, newHeight);
    }

    /**
     * Fills the heights of the knots of the time change at the old and at the new parameters, and returns whether they
     * are valid: both must be finite and increase with the fraction of nodes below them.
     */
    private boolean buildMap(double oldLogPopSize, double oldGrowthRate, double newLogPopSize, double newGrowthRate) {
        this.oldKnots[0] = 0.0;
        this.newKnots[0] = 0.0;

        for (int q = 0; q < LEVELS.length; q++) {
            this.oldKnots[q + 1] = this.predictQuantile(q, oldLogPopSize, oldGrowthRate);
            this.newKnots[q + 1] = this.predictQuantile(q, newLogPopSize, newGrowthRate);
            if (!(this.oldKnots[q + 1] > this.oldKnots[q]) || !(this.newKnots[q + 1] > this.newKnots[q])
                    || Double.isInfinite(this.oldKnots[q + 1]) || Double.isInfinite(this.newKnots[q + 1])) {
                return false;
            }
        }
        return true;
    }

    private double predictQuantile(int level, double logPopSize, double growthRate) {
        return Math.exp(this.intercepts[level]
                + this.slopes[level][0] * (logPopSize - this.featureMeans[0])
                + this.slopes[level][1] * (growthRate - this.featureMeans[1]));
    }

    /**
     * Returns the time at the given height: the fraction of the internal nodes below it, interpolated linearly
     * between the knots and from zero to the first knot. Above the last knot it keeps the slope of the last segment.
     */
    private double computeTime(double[] knots, double height) {
        int segment = this.findSegment(knots, height);
        int last = LEVELS.length;
        if (segment == last) {
            return KNOT_FRACTIONS[last] + (height - knots[last]) * this.computeSlope(knots, last - 1);
        }
        return KNOT_FRACTIONS[segment] + (height - knots[segment]) * this.computeSlope(knots, segment);
    }

    /** Returns the height at the given time, which is the inverse of {@link #computeTime}. */
    private double computeHeight(double[] knots, double time) {
        int last = LEVELS.length;
        if (time >= KNOT_FRACTIONS[last]) {
            return knots[last] + (time - KNOT_FRACTIONS[last]) / this.computeSlope(knots, last - 1);
        }

        int low = 0;
        int high = last;
        while (high - low > 1) {
            int middle = (low + high) >>> 1;
            if (KNOT_FRACTIONS[middle] <= time) {
                low = middle;
            } else {
                high = middle;
            }
        }
        return knots[low] + (time - KNOT_FRACTIONS[low]) / this.computeSlope(knots, low);
    }

    /** Returns the logarithm of the slope of the time at the given height. */
    private double computeLogSlope(double[] knots, double height) {
        int segment = Math.min(this.findSegment(knots, height), LEVELS.length - 1);
        return Math.log(this.computeSlope(knots, segment));
    }

    /** Returns the slope of the time between the given knot and the next one. */
    private double computeSlope(double[] knots, int segment) {
        return (KNOT_FRACTIONS[segment + 1] - KNOT_FRACTIONS[segment]) / (knots[segment + 1] - knots[segment]);
    }

    /** Returns the index of the last knot at or below the height. */
    private int findSegment(double[] knots, double height) {
        int low = 0;
        int high = LEVELS.length;
        while (low < high) {
            int middle = (low + high + 1) >>> 1;
            if (knots[middle] <= height) {
                low = middle;
            } else {
                high = middle - 1;
            }
        }
        return low;
    }

    /**
     * Maps the mutations of every non-root branch linearly from the old to the new heights of its ends and returns
     * the sum over the mutations of the log ratio of the new to the old branch length.
     */
    private double moveMutations(Node[] nodes) {
        double logJacobian = 0.0;

        for (Node node : nodes) {
            List<Mutation> branchMutations = this.mutations.getMutations(node);
            if (node.isRoot() || branchMutations.isEmpty()) {
                // the times of the root mutations are meaningless
                continue;
            }

            double oldStartHeight = this.oldHeights[node.getParent().getNr()];
            double oldEndHeight = this.oldHeights[node.getNr()];
            double newStartHeight = node.getParent().getHeight();
            double newEndHeight = node.getHeight();
            double lengthRatio = (newStartHeight - newEndHeight) / (oldStartHeight - oldEndHeight);

            List<Mutation> newBranchMutations = new ArrayList<>();
            for (Mutation mutation : branchMutations) {
                newBranchMutations.add(new Mutation(
                        mutation.nodeNr(),
                        this.mapMutationHeight(mutation.time(), oldStartHeight, oldEndHeight, newStartHeight, newEndHeight),
                        this.mapMutationHeight(mutation.timeOfPreviousMutation(), oldStartHeight, oldEndHeight, newStartHeight, newEndHeight),
                        mutation.site(), mutation.oldState(), mutation.newState()
                ));
            }

            this.mutations.applyMutations(node, newBranchMutations, this);
            logJacobian += branchMutations.size() * Math.log(lengthRatio);
        }

        return logJacobian;
    }

    /**
     * Maps a height on a branch linearly from its old to its new ends. The ends map exactly
     * onto the new ends, which keeps the times of previous mutations at the branch start
     * consistent, and rounding cannot push a height off the branch.
     */
    private double mapMutationHeight(double height, double oldStartHeight, double oldEndHeight, double newStartHeight, double newEndHeight) {
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
        if (this.optimiseInput.get() && this.isFitted) {
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
        this.scaleFactor = value;
    }

}
