package emat.operators;

import beast.base.core.Description;
import beast.base.core.Input;
import beast.base.evolution.tree.Node;
import beast.base.inference.Distribution;
import beast.base.spec.inference.parameter.RealScalarParam;
import emat.helper.HazardTimeChange;
import emat.helper.LearnedHazardRate;
import emat.helper.TreeHazard;

import java.util.ArrayList;
import java.util.List;

@Description("Proposes the joint move of the CoalescentWarpOperator, but with a time change that it learns from the chain " +
        "instead of the integral of the inverse population size of the exponential growth coalescent. It fits the rate at which " +
        "the tree prior produces branching events, as a function of the height and of the two parameters, to the log density " +
        "that the tree prior gave the trees of the chain. The hazard says how the number of units at risk depends on the " +
        "number of lineages, and AUTO chooses between the two by the residual of their fits at the first fit. Proposals are rejected until the first fit.")
public class LearnedWarpOperator extends CoalescentWarpOperator {

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

    @Override
    public void initAndValidate() {
        super.initAndValidate();

        this.hazardRate = new LearnedHazardRate(this.hazardInput.get(), this.numKnotsInput.get(), NUM_PARAMETERS, this.maxObservationsInput.get());
        this.nextFitSize = this.minObservationsInput.get();
    }

    @Override
    public double proposal() {
        this.observeState();
        return super.proposal();
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

    /* Move */

    @Override
    public double warp(double logPopSizeShift, double growthRateShift) {
        if (!this.hazardRate.isFitted()) {
            return Double.NEGATIVE_INFINITY;
        }
        return super.warp(logPopSizeShift, growthRateShift);
    }

    @Override
    protected double computeCoalescentTime(double height, double popSize, double growthRate) {
        return this.getTimeChange(popSize, growthRate).computeTime(height);
    }

    @Override
    protected double computeHeight(double coalescentTime, double popSize, double growthRate) {
        return this.getTimeChange(popSize, growthRate).computeHeight(coalescentTime);
    }

    @Override
    protected double computeLogInversePopSize(double height, double popSize, double growthRate) {
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

    /* Tuning */

    @Override
    public void optimize(double logAlpha) {
        if (this.hazardRate.isFitted()) {
            super.optimize(logAlpha);
        }
    }

}
