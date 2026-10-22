package emat.operators.nodetimes.warp;

import beast.base.core.Description;
import beast.base.core.Input;
import adapters.Adapter;
import adaptiveoperators.CenteredMultivariateNormalSampler;
import beast.base.evolution.operator.TreeOperator;
import beast.base.evolution.tree.Node;
import beast.base.evolution.tree.Tree;
import beast.base.inference.StateNode;
import beast.base.internal.json.JSONException;
import beast.base.internal.json.JSONObject;
import beast.base.spec.evolution.branchratemodel.Base;
import beast.base.spec.evolution.branchratemodel.RandomLocalClockModel;
import beast.base.spec.evolution.branchratemodel.UCRelaxedClockModel;
import beast.base.inference.Distribution;
import beast.base.spec.inference.parameter.RealScalarParam;
import emat.helper.ThinnedEssTracker;
import emat.operators.nodetimes.BranchTimeMapper;
import emat.operators.nodetimes.NodeTimes;
import emat.state.Mutations;

import java.io.PrintWriter;
import java.io.StringWriter;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

@Description("Proposes a joint move of the parameters of the tree prior, the heights of all internal nodes and the " +
        "clock rate. The mutable values of the parameters take a Gaussian step whose covariance is learned from the chain, and every node keeps the interval to its older child in " +
        "time change, which the operator learns from the chain too. It fits the rate at which " +
        "the tree prior produces branching events, as a function of the height and of the parameters, to the log density " +
        "that the tree prior gave the trees of the chain. The hazard says how the number of units at risk depends on the " +
        "number of lineages, and AUTO chooses between the two by the residual of their fits at the first fit. " +
        "The mean rate of the clock model then follows from keeping the product of the mean rate and the tree length, weighted by " +
        "the relative rates of the branches. Proposals are rejected until the first fit.")
public class LearnedWarpOperator extends TreeOperator {

    final public Input<Mutations> mutationsInput = new Input<>("mutations", "the mutations to move along with the tree", Input.Validate.REQUIRED);
    final public Input<List<Adapter>> adaptersInput = new Input<>("adapter", "adapters of the parameters of the tree prior, which give the real representation the operator works on: the mutable values are moved, and the immutable values only inform the learned rate", new ArrayList<>());
    final public Input<Base> branchRateModelInput = new Input<>("branchRateModel", "the clock model, whose mean rate is set so that its product with the tree length, weighted by the relative rates of the branches, stays the same. The relative rates of the branches must not depend on the heights of the nodes, so that a normalised relaxed clock is not supported. Without a clock model or without a mean rate the rate is not changed");
    final public Input<Double> scaleFactorInput = new Input<>("scaleFactor", "the factor by which the learned standard deviations of the step are scaled, which is tuned if optimise is set, and which defaults to 2.38 / sqrt(d) for d mutable values");
    final public Input<Boolean> optimiseInput = new Input<>("optimise", "whether to tune the scale factor towards the target acceptance probability", true);

    final public Input<Distribution> treePriorInput = new Input<>("treePrior", "the tree prior, whose log density is what the rate is fitted to", Input.Validate.REQUIRED);
    final public Input<TreeHazard> hazardInput = new Input<>("hazard", "how the number of units at risk depends on the number of lineages: every pair of lineages merges at the same rate for a coalescent, and every lineage branches at the same rate for a birth-death process, or AUTO to choose the one that fits the log densities better at the first fit", TreeHazard.COALESCENT, TreeHazard.values());
    final public Input<Integer> numKnotsInput = new Input<>("numKnots", "the number of knots above height zero, between which the logarithm of the rate is linear in the height", 20);
    final public Input<Integer> observeEveryInput = new Input<>("observeEvery", "the number of proposals between two observations of the state", 5);
    final public Input<Integer> minObservationsInput = new Input<>("minObservations", "the number of observations at which the rate is fitted for the first time, after which the fit is repeated whenever the number of observations doubles", 400);
    final public Input<Double> minEssInput = new Input<>("minEss", "the effective sample size of the tree length, recorded at the observations, that must be exceeded before the rate is fitted for the first time, so that the fit does not learn from the burn-in", 20.0);
    final public Input<Integer> essThinningInput = new Input<>("essThinning", "the number of observations between two values of the trace of the tree length whose effective sample size is compared with minEss, which also delays the first check until there are 100 such values", 12);
    final public Input<Integer> maxObservationsInput = new Input<>("maxObservations", "the number of most recent observations that are kept", 1000);

    // the key under which the effective sample size of the tree length is kept in the state file
    private static final String ESS_KEY = "treeLengthEss";

    // the adapters have no use for the node, as the parameters of the tree prior do not belong to one
    private static final int NO_NODE = -1;

    private LearnedHazardRate hazardRate;
    private long numProposals;
    private long nextFitSize;
    private boolean hasReportedHazard;
    private boolean hasReportedEss;

    // the trace of the tree length at the observations, which tells whether the chain has left its starting state
    private ThinnedEssTracker treeLengthTracker;

    // the two most recent time changes, by the parameters they were built for
    private final double[][] cachedParameters = new double[2][];
    private final HazardTimeChange[] cachedTimeChanges = new HazardTimeChange[2];
    private int nextSlot;

    private Mutations mutations;
    private Tree tree;
    private List<Adapter> adapters;
    private int numMutable;
    private int numImmutable;

    // the mean rate of the clock model that is mapped along with the tree, or null if the rate is fixed
    private Base branchRateModel;
    private RealScalarParam<?> meanRate;

    // the learned distribution of the values that are moved, from which the zero-mean steps are drawn
    private CenteredMultivariateNormalSampler stepSampler;

    private double scaleFactor;

    // the heights of all nodes before the move, by node number
    private double[] oldHeights;

    // the mutable and immutable values of the parameters before and after the move, which the rate depends on
    private double[] oldFeatures;
    private double[] newFeatures;

    @Override
    public void initAndValidate() {
        this.tree = this.treeInput.get();
        this.mutations = this.mutationsInput.get();
        this.adapters = this.adaptersInput.get();
        this.initialiseClock();
        this.oldHeights = new double[this.tree.getNodeCount()];

        for (Adapter adapter : this.adapters) {
            this.numMutable += adapter.getNumMutable();
            this.numImmutable += adapter.getNumImmutable();
        }
        if (this.numMutable == 0) {
            throw new IllegalArgumentException("At least one adapter must have mutable values.");
        }

        Double scaleFactor = this.scaleFactorInput.get();
        this.scaleFactor = scaleFactor != null ? scaleFactor : 2.38 / Math.sqrt(this.numMutable);
        this.stepSampler = new CenteredMultivariateNormalSampler(this.numMutable);

        this.hazardRate = new LearnedHazardRate(this.hazardInput.get(), this.numKnotsInput.get(), this.numMutable + this.numImmutable, this.maxObservationsInput.get());
        this.nextFitSize = this.minObservationsInput.get();
        this.treeLengthTracker = new ThinnedEssTracker(this.essThinningInput.get());
    }

    /** Finds the mean rate to map, and rejects clock models whose relative branch rates change with the heights. */
    private void initialiseClock() {
        this.branchRateModel = this.branchRateModelInput.get();
        if (this.branchRateModel == null || this.branchRateModel.meanRateInput.get() == null) {
            return;
        }

        if (this.branchRateModel instanceof UCRelaxedClockModel relaxed && relaxed.normalizeInput.get()) {
            throw new IllegalArgumentException("The relative rates of a normalised relaxed clock depend on the heights, so set normalize to false.");
        }
        if (this.branchRateModel instanceof RandomLocalClockModel local && local.scalingInput.get()) {
            throw new IllegalArgumentException("The relative rates of a scaled random local clock depend on the heights, so set scaling to false.");
        }

        if (this.branchRateModel.meanRateInput.get() instanceof StateNode mean) {
            this.meanRate = this.getScalarParameter(mean);
        } else {
            throw new IllegalArgumentException("The mean rate of the clock model must be a real scalar parameter.");
        }
    }

    private RealScalarParam<?> getScalarParameter(StateNode parameter) {
        if (parameter instanceof RealScalarParam<?> scalar) {
            return scalar;
        }
        throw new IllegalArgumentException("The parameter " + parameter.getID() + " must be a real scalar parameter.");
    }

    @Override
    public double proposal() {
        this.stepSampler.record(new double[0], this.readMutable());
        this.observeState();

        double[] step = this.stepSampler.sampleConditionally(new double[0], this.scaleFactor);
        return this.warp(step);
    }

    /**
     * Adds the given step to the mutable values of the parameters, and maps the tree and the clock rate accordingly.
     * Returns the log Hastings ratio, which is the log Jacobian of the map from the old to the new heights and the
     * clock rate, plus one term for every mutation, plus the change of the log Jacobian corrections of the adapters,
     * which turn the symmetric step in the real representation into a step on the parameters. A step with the
     * opposite shifts undoes the move and has the negated Hastings ratio.
     */
    public double warp(double[] step) {
        if (!this.hazardRate.isFitted()) {
            return Double.NEGATIVE_INFINITY;
        }

        double[] oldMutable = this.readMutable();
        this.oldFeatures = this.readFeatures();
        double logCorrectionBefore = this.computeLogJacobianCorrection();

        if (!this.writeMutable(oldMutable, step)) {
            return Double.NEGATIVE_INFINITY;
        }
        this.newFeatures = this.readFeatures();
        double logCorrectionAfter = this.computeLogJacobianCorrection();

        Node[] nodes = this.tree.getNodesAsArray();
        for (Node node : nodes) {
            this.oldHeights[node.getNr()] = node.getHeight();
        }

        double[] branchRates = this.readBranchRates(nodes);
        double lengthBefore = this.computeWeightedLength(nodes, branchRates);
        double logJacobian = this.warpHeights(this.tree.getRoot());
        if (!Double.isFinite(logJacobian)) {
            return Double.NEGATIVE_INFINITY;
        }
        double lengthAfter = this.computeWeightedLength(nodes, branchRates);

        double logHastingsRatio = logJacobian + logCorrectionBefore - logCorrectionAfter;
        double newMeanRate = Double.NaN;
        if (this.meanRate != null) {
            newMeanRate = this.meanRate.get() * lengthBefore / lengthAfter;
            if (!this.meanRate.isValid(newMeanRate)) {
                return Double.NEGATIVE_INFINITY;
            }
            logHastingsRatio += Math.log(lengthBefore / lengthAfter);
        }
        logHastingsRatio += BranchTimeMapper.mapMutations(this.mutations, nodes, this.oldHeights, this);

        if (this.meanRate != null) {
            this.meanRate.set(newMeanRate);
        }

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

        double newCoalescentTime = this.computeCoalescentTime(newChildHeight, this.newFeatures)
                + this.computeCoalescentTime(oldHeight, this.oldFeatures)
                - this.computeCoalescentTime(oldChildHeight, this.oldFeatures);
        double newHeight = this.computeHeight(newCoalescentTime, this.newFeatures);
        if (!(newHeight > newChildHeight)) {
            return Double.NaN;
        }

        node.setHeight(newHeight);

        return logJacobian
                + this.computeLogRate(oldHeight, this.oldFeatures)
                - this.computeLogRate(newHeight, this.newFeatures);
    }

    private double computeCoalescentTime(double height, double[] features) {
        return this.getTimeChange(features).computeTime(height);
    }

    private double computeHeight(double coalescentTime, double[] features) {
        return this.getTimeChange(features).computeHeight(coalescentTime);
    }

    private double computeLogRate(double height, double[] features) {
        return this.getTimeChange(features).computeLogRate(height);
    }

    /**
     * Returns the time change for the given values of the parameters from one of the two most recent, as a move uses the old and the
     * new values alternately. A time change that is not valid yields NaN heights, which reject the move.
     */
    private HazardTimeChange getTimeChange(double[] features) {
        for (int slot = 0; slot < this.cachedTimeChanges.length; slot++) {
            if (Arrays.equals(this.cachedParameters[slot], features)) {
                return this.cachedTimeChanges[slot];
            }
        }

        HazardTimeChange timeChange = this.hazardRate.createTimeChange(features);
        this.cachedParameters[this.nextSlot] = features.clone();
        this.cachedTimeChanges[this.nextSlot] = timeChange;
        this.nextSlot = 1 - this.nextSlot;
        return timeChange;
    }

    private void clearTimeChanges() {
        this.cachedParameters[0] = null;
        this.cachedParameters[1] = null;
    }

    /**
     * Returns the rates of the branches above the nodes by node number, which are the same before and after the move as
     * they do not depend on the heights. Only their ratios matter, so it does not matter that the clock model may
     * not have caught up with the mean rate yet. The rates are all one if there is no mean rate to map.
     */
    private double[] readBranchRates(Node[] nodes) {
        double[] rates = new double[nodes.length];
        for (Node node : nodes) {
            rates[node.getNr()] = this.meanRate == null || node.isRoot() ? 1.0 : this.branchRateModel.getRateForBranch(node);
        }
        return rates;
    }

    /** Returns the sum over the branches of the length of the branch times its rate. */
    private double computeWeightedLength(Node[] nodes, double[] branchRates) {
        double length = 0.0;
        for (Node node : nodes) {
            length += branchRates[node.getNr()] * node.getLength();
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
     * observations are there for the first time or have doubled since the last fit, and the tree length has reached the
     * minimum effective sample size. The fit only changes between
     * proposals, so a proposal and its reverse use the same rate.
     */
    private void observeState() {
        this.numProposals++;
        if (this.numProposals % this.observeEveryInput.get() != 0) {
            return;
        }

        this.observe(this.treePriorInput.get().getCurrentLogP());
        this.treeLengthTracker.record(NodeTimes.computeTreeLength(this.tree));
        this.reportReachedEss();
        if (this.hazardRate.getNumObserved() >= this.nextFitSize && this.treeLengthTracker.hasReached(this.minEssInput.get())) {
            boolean hasFitted = this.hazardRate.fit();
            this.reportFit(hasFitted);
            this.reportChosenHazard();
            this.clearTimeChanges();

            // a failed fit is tried again after as many observations as the first fit waited for
            long numObserved = this.hazardRate.getNumObserved();
            this.nextFitSize = hasFitted ? 2 * numObserved : numObserved + this.minObservationsInput.get();
        }
    }

    /** Announces once that the tree length has reached the effective sample size that the first fit waits for. */
    private void reportReachedEss() {
        if (!this.hasReportedEss && this.treeLengthTracker.hasReached(this.minEssInput.get())) {
            this.hasReportedEss = true;
            System.out.printf("%s: tree length reached an ESS of %.1f after %d proposals%n",
                    this.getID(), this.treeLengthTracker.getEss(), this.numProposals);
        }
    }

    /** Announces the outcome of a fit, as a failed fit leaves the previous rate in place without any other sign. */
    private void reportFit(boolean hasFitted) {
        System.out.printf("%s: fit after %d observations %s, residual %.3g%s%n",
                this.getID(), this.hazardRate.getNumObserved(), hasFitted ? "succeeded" : "failed", this.hazardRate.getRmsResidual(),
                hasFitted ? "" : " (" + this.hazardRate.getFailureReason() + ")");
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

        this.hazardRate.addObservation(this.toArray(tipHeights), this.toArray(internalHeights), this.readFeatures(), logDensity);
    }

    private double[] toArray(List<Double> values) {
        return values.stream().mapToDouble(Double::doubleValue).toArray();
    }

    /* Parameters */

    /** Returns the mutable values followed by the immutable values of the adapters, which the learned rate depends on. */
    private double[] readFeatures() {
        double[] features = new double[this.numMutable + this.numImmutable];
        System.arraycopy(this.readMutable(), 0, features, 0, this.numMutable);

        int index = this.numMutable;
        for (Adapter adapter : this.adapters) {
            double[] immutable = adapter.getImmutable(NO_NODE);
            System.arraycopy(immutable, 0, features, index, immutable.length);
            index += immutable.length;
        }
        return features;
    }

    private double[] readMutable() {
        double[] mutable = new double[this.numMutable];

        int index = 0;
        for (Adapter adapter : this.adapters) {
            double[] values = adapter.getMutable(NO_NODE);
            System.arraycopy(values, 0, mutable, index, values.length);
            index += values.length;
        }
        return mutable;
    }

    /** Sets the mutable values to the old values plus the step, and returns whether an adapter took them all. */
    private boolean writeMutable(double[] oldMutable, double[] step) {
        int index = 0;
        for (Adapter adapter : this.adapters) {
            int numValues = adapter.getNumMutable();
            if (numValues == 0) {
                continue;
            }

            double[] values = new double[numValues];
            for (int i = 0; i < numValues; i++) {
                values[i] = oldMutable[index + i] + step[index + i];
            }

            try {
                adapter.update(values, NO_NODE);
            } catch (RuntimeException exception) {
                // the adapter cannot represent the values, as for a transform that leaves its domain
                return false;
            }
            index += numValues;
        }
        return true;
    }

    private double computeLogJacobianCorrection() {
        double logCorrection = 0.0;
        for (Adapter adapter : this.adapters) {
            logCorrection += adapter.getLogJacobianCorrection(NO_NODE);
        }
        return logCorrection;
    }

    @Override
    public List<StateNode> listStateNodes() {
        List<StateNode> stateNodes = new ArrayList<>(super.listStateNodes());
        if (this.meanRate != null) {
            stateNodes.add(this.meanRate);
        }
        for (Adapter adapter : this.adapters) {
            stateNodes.addAll(adapter.listStateNodes());
        }
        return stateNodes;
    }

    /* State File */

    /** Stores the effective sample size of the tree length with the tuning information, so that a resumed run does not wait for it again. */
    @Override
    public void storeToFile(PrintWriter out) {
        StringWriter tuning = new StringWriter();
        super.storeToFile(new PrintWriter(tuning));

        // the tuning information is a single JSON object, which gets one more entry
        String json = tuning.toString();
        out.print(json.substring(0, json.lastIndexOf('}')) + ",\"" + ESS_KEY + "\":" + this.treeLengthTracker.getEss() + "}");
    }

    @Override
    public void restoreFromFile(JSONObject o) {
        super.restoreFromFile(o);

        try {
            if (o.has(ESS_KEY)) {
                this.treeLengthTracker.restoreEss(o.getDouble(ESS_KEY));
            }
        } catch (JSONException exception) {
            // the effective sample size is learned again from the chain
            exception.printStackTrace();
        }
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
