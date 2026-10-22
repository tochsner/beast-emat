package emat.operators.nodetimes.warp;

import beast.base.core.Description;
import beast.base.core.Input;
import beast.base.evolution.operator.TreeOperator;
import beast.base.evolution.tree.Node;
import beast.base.evolution.tree.Tree;
import beast.base.inference.StateNode;
import beast.base.inference.Distribution;
import beast.base.spec.inference.parameter.RealScalarParam;
import beast.base.util.Randomizer;
import emat.state.Mutation;
import emat.state.Mutations;

import java.util.ArrayList;
import java.util.List;

@Description("Proposes a joint move of the two parameters of the tree prior, the heights of all internal nodes and the " +
        "clock rate. The parameters take a correlated Gaussian step, and every node keeps the interval to its older child in " +
        "time change, which the operator learns from the chain. It fits the rate at which " +
        "the tree prior produces branching events, as a function of the height and of the two parameters, to the log density " +
        "that the tree prior gave the trees of the chain. The hazard says how the number of units at risk depends on the " +
        "number of lineages, and AUTO chooses between the two by the residual of their fits at the first fit. " +
        "The clock rate then follows from keeping the product of the clock rate and the tree length. Proposals are rejected until the first fit.")
public class LearnedWarpOperator extends TreeOperator {

    final public Input<Mutations> mutationsInput = new Input<>("mutations", "the mutations to move along with the tree", Input.Validate.REQUIRED);
    final public Input<StateNode> popSizeInput = new Input<>("popSize", "the population size at height zero of the exponential growth coalescent", Input.Validate.REQUIRED);
    final public Input<StateNode> growthRateInput = new Input<>("growthRate", "the growth rate of the exponential growth coalescent, with the population size exp(-rate * height) * popSize", Input.Validate.REQUIRED);
    final public Input<StateNode> clockRateInput = new Input<>("clockRate", "the clock rate, which is set so that its product with the tree length stays the same", Input.Validate.REQUIRED);
    final public Input<Double> sdLogPopSizeInput = new Input<>("sdLogPopSize", "the standard deviation of the step in the logarithm of the population size, before scaling", Input.Validate.REQUIRED);
    final public Input<Double> sdGrowthRateInput = new Input<>("sdGrowthRate", "the standard deviation of the step in the growth rate, before scaling", Input.Validate.REQUIRED);
    final public Input<Double> correlationInput = new Input<>("correlation", "the correlation between the two steps", Input.Validate.REQUIRED);
    final public Input<Double> scaleFactorInput = new Input<>("scaleFactor", "the factor by which both standard deviations are scaled, which is tuned if optimise is set", 1.0);
    final public Input<Boolean> optimiseInput = new Input<>("optimise", "whether to tune the scale factor towards the target acceptance probability", true);

    final public Input<Distribution> treePriorInput = new Input<>("treePrior", "the tree prior, whose log density is what the rate is fitted to", Input.Validate.REQUIRED);
    final public Input<TreeHazard> hazardInput = new Input<>("hazard", "how the number of units at risk depends on the number of lineages: every pair of lineages merges at the same rate for a coalescent, and every lineage branches at the same rate for a birth-death process, or AUTO to choose the one that fits the log densities better at the first fit", TreeHazard.COALESCENT, TreeHazard.values());
    final public Input<Integer> numKnotsInput = new Input<>("numKnots", "the number of knots above height zero, between which the logarithm of the rate is linear in the height", 20);
    final public Input<Integer> observeEveryInput = new Input<>("observeEvery", "the number of proposals between two observations of the state", 5);
    final public Input<Integer> minObservationsInput = new Input<>("minObservations", "the number of observations at which the rate is fitted for the first time, after which the fit is repeated whenever the number of observations doubles", 400);
    final public Input<Integer> maxObservationsInput = new Input<>("maxObservations", "the number of most recent observations that are kept", 1000);

    // the number of parameters the rate depends on: the logarithm of the population size and the growth rate
    private static final int NUM_PARAMETERS = 2;

    private LearnedHazardRate hazardRate;
    private long numProposals;
    private long nextFitSize;
    private boolean hasReportedHazard;

    // the two most recent time changes, by the parameters they were built for
    private final double[][] cachedParameters = new double[2][];
    private final HazardTimeChange[] cachedTimeChanges = new HazardTimeChange[2];
    private int nextSlot;

    private Mutations mutations;
    private Tree tree;
    private RealScalarParam<?> popSize;
    private RealScalarParam<?> growthRate;
    private RealScalarParam<?> clockRate;

    private double scaleFactor;

    // the heights of all nodes before the move, by node number
    private double[] oldHeights;

    // the parameters of the coalescent before and after the move
    private double oldPopSize;
    private double oldGrowthRate;
    private double newPopSize;
    private double newGrowthRate;

    @Override
    public void initAndValidate() {
        this.tree = this.treeInput.get();
        this.mutations = this.mutationsInput.get();
        this.popSize = this.getScalarParameter(this.popSizeInput.get());
        this.growthRate = this.getScalarParameter(this.growthRateInput.get());
        this.clockRate = this.getScalarParameter(this.clockRateInput.get());
        this.scaleFactor = this.scaleFactorInput.get();
        this.oldHeights = new double[this.tree.getNodeCount()];

        this.hazardRate = new LearnedHazardRate(this.hazardInput.get(), this.numKnotsInput.get(), NUM_PARAMETERS, this.maxObservationsInput.get());
        this.nextFitSize = this.minObservationsInput.get();
    }

    private RealScalarParam<?> getScalarParameter(StateNode parameter) {
        if (parameter instanceof RealScalarParam<?> scalar) {
            return scalar;
        }
        throw new IllegalArgumentException("The parameter " + parameter.getID() + " must be a real scalar parameter.");
    }

    @Override
    public double proposal() {
        double firstNormal = Randomizer.nextGaussian();
        double secondNormal = Randomizer.nextGaussian();
        double correlation = this.correlationInput.get();

        double logPopSizeShift = this.scaleFactor * this.sdLogPopSizeInput.get() * firstNormal;
        double growthRateShift = this.scaleFactor * this.sdGrowthRateInput.get()
                * (correlation * firstNormal + Math.sqrt(1.0 - correlation * correlation) * secondNormal);

        this.observeState();
        return this.warp(logPopSizeShift, growthRateShift);
    }

    /**
     * Shifts the logarithm of the population size and the growth rate by the given amounts, and maps the tree and
     * the clock rate accordingly. Returns the log Hastings ratio, which is the log Jacobian of the map from the old
     * to the new heights, the clock rate and the population size on the natural scale, plus one term for every
     * mutation. A step with the opposite shifts undoes the move and has the negated Hastings ratio.
     */
    public double warp(double logPopSizeShift, double growthRateShift) {
        if (!this.hazardRate.isFitted()) {
            return Double.NEGATIVE_INFINITY;
        }

        this.oldPopSize = this.popSize.get();
        this.oldGrowthRate = this.growthRate.get();
        this.newPopSize = this.oldPopSize * Math.exp(logPopSizeShift);
        this.newGrowthRate = this.oldGrowthRate + growthRateShift;
        if (!this.popSize.isValid(this.newPopSize) || !this.growthRate.isValid(this.newGrowthRate)) {
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

        // the step is symmetric in the logarithm of the population size, which adds the shift on the natural scale
        double logHastingsRatio = logJacobian + Math.log(lengthBefore / lengthAfter) + logPopSizeShift;
        logHastingsRatio += this.moveMutations(nodes);

        this.popSize.set(this.newPopSize);
        this.growthRate.set(this.newGrowthRate);
        this.clockRate.set(newClockRate);

        return logHastingsRatio;
    }

    /**
     * Moves every internal node, after the nodes below it, to the height whose time change above its older child
     * under the new parameters equals the time change it had under the old parameters. Returns the sum over the
     * internal nodes of the log ratio of the old to the new slope of the time change, which is the log Jacobian
     * of the map, or a value that is not finite if a node cannot be mapped.
     */
    private double warpHeights(Node node) {
        if (node.isLeaf()) {
            return 0.0;
        }

        double logJacobian = this.warpHeights(node.getLeft()) + this.warpHeights(node.getRight());

        double oldHeight = this.oldHeights[node.getNr()];
        double oldChildHeight = Math.max(this.oldHeights[node.getLeft().getNr()], this.oldHeights[node.getRight().getNr()]);
        double newChildHeight = Math.max(node.getLeft().getHeight(), node.getRight().getHeight());

        double newCoalescentTime = this.computeCoalescentTime(newChildHeight, this.newPopSize, this.newGrowthRate)
                + this.computeCoalescentTime(oldHeight, this.oldPopSize, this.oldGrowthRate)
                - this.computeCoalescentTime(oldChildHeight, this.oldPopSize, this.oldGrowthRate);
        double newHeight = this.computeHeight(newCoalescentTime, this.newPopSize, this.newGrowthRate);
        if (!(newHeight > newChildHeight)) {
            return Double.NaN;
        }

        node.setHeight(newHeight);

        return logJacobian
                + this.computeLogInversePopSize(oldHeight, this.oldPopSize, this.oldGrowthRate)
                - this.computeLogInversePopSize(newHeight, this.newPopSize, this.newGrowthRate);
    }

    private double computeCoalescentTime(double height, double popSize, double growthRate) {
        return this.getTimeChange(popSize, growthRate).computeTime(height);
    }

    private double computeHeight(double coalescentTime, double popSize, double growthRate) {
        return this.getTimeChange(popSize, growthRate).computeHeight(coalescentTime);
    }

    private double computeLogInversePopSize(double height, double popSize, double growthRate) {
        return this.getTimeChange(popSize, growthRate).computeLogRate(height);
    }

    /**
     * Returns the time change for the given parameters from one of the two most recent, as a move uses the old and the
     * new parameters alternately. A time change that is not valid yields NaN heights, which reject the move.
     */
    private HazardTimeChange getTimeChange(double popSize, double growthRate) {
        for (int slot = 0; slot < this.cachedTimeChanges.length; slot++) {
            double[] parameters = this.cachedParameters[slot];
            if (parameters != null && parameters[0] == popSize && parameters[1] == growthRate) {
                return this.cachedTimeChanges[slot];
            }
        }

        double[] parameters = {popSize, growthRate};
        HazardTimeChange timeChange = this.hazardRate.createTimeChange(new double[]{Math.log(popSize), growthRate});
        this.cachedParameters[this.nextSlot] = parameters;
        this.cachedTimeChanges[this.nextSlot] = timeChange;
        this.nextSlot = 1 - this.nextSlot;
        return timeChange;
    }

    private void clearTimeChanges() {
        this.cachedParameters[0] = null;
        this.cachedParameters[1] = null;
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
                        this.mapHeight(mutation.time(), oldStartHeight, oldEndHeight, newStartHeight, newEndHeight),
                        this.mapHeight(mutation.timeOfPreviousMutation(), oldStartHeight, oldEndHeight, newStartHeight, newEndHeight),
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
    private double mapHeight(double height, double oldStartHeight, double oldEndHeight, double newStartHeight, double newEndHeight) {
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

    /** Returns the learned rate, which tests and other operators may feed with observations. */
    public LearnedHazardRate getHazardRate() {
        return this.hazardRate;
    }

    /* Learning */

    /**
     * Records the state of every few proposals with the log density of the tree prior, and fits the rate when enough
     * observations are there for the first time or have doubled since the last fit. The fit only changes between
     * proposals, so a proposal and its reverse use the same rate.
     */
    private void observeState() {
        this.numProposals++;
        if (this.numProposals % this.observeEveryInput.get() != 0) {
            return;
        }

        this.observe(this.treePriorInput.get().getCurrentLogP());
        if (this.hazardRate.getNumObserved() >= this.nextFitSize) {
            this.hazardRate.fit();
            this.reportChosenHazard();
            this.clearTimeChanges();
            this.nextFitSize = 2 * this.hazardRate.getNumObserved();
        }
    }

    /** Announces the hazard that AUTO chose, with the residuals it chose by, once. */
    private void reportChosenHazard() {
        double[] residuals = this.hazardRate.getCandidateRmsResiduals();
        if (residuals != null && !this.hasReportedHazard) {
            this.hasReportedHazard = true;
            System.out.printf("%s: hazard %s chosen after %d observations, residual %.3g for COALESCENT and %.3g for BIRTH_DEATH%n",
                    this.getID(), this.hazardRate.getResolvedHazard(), this.hazardRate.getNumObserved(), residuals[0], residuals[1]);
        }
    }

    /** Records the current tree and parameters with the given log density of the tree prior. */
    public void observe(double logDensity) {
        List<Double> tipHeights = new ArrayList<>();
        List<Double> internalHeights = new ArrayList<>();
        for (Node node : this.treeInput.get().getNodesAsArray()) {
            (node.isLeaf() ? tipHeights : internalHeights).add(node.getHeight());
        }

        this.hazardRate.addObservation(this.toArray(tipHeights), this.toArray(internalHeights), this.readParameters(), logDensity);
    }

    private double[] toArray(List<Double> values) {
        return values.stream().mapToDouble(Double::doubleValue).toArray();
    }

    private double[] readParameters() {
        return new double[]{Math.log(((RealScalarParam<?>) this.popSizeInput.get()).get()), ((RealScalarParam<?>) this.growthRateInput.get()).get()};
    }

    /* Tuning */

    @Override
    public void optimize(double logAlpha) {
        if (this.optimiseInput.get() && this.hazardRate.isFitted()) {
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
