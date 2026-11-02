package emat.operators.treeprior;

import adapters.Adapter;
import beast.base.core.Description;
import beast.base.core.Input;
import beast.base.evolution.tree.Tree;
import beast.base.inference.Distribution;
import beast.base.inference.Operator;
import beast.base.inference.State;
import beast.base.inference.StateNode;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

@Description("Proposes new values of the parameters of the tree prior by drawing them independently of their current " +
        "values from a distribution that is learned given statistics of the tree. The distribution is Gaussian in the " +
        "real representation of the adapters, with a mean that is a quadratic function of the logarithms of the tree length, " +
        "the tree height, quantiles of the heights of the internal nodes and the coalescent floor. It learns from regular " +
        "sweeps of slice sampling over the parameters with the tree fixed, which evaluate the distributions that depend " +
        "on the parameters. Proposals are rejected until the first fit.")
public class LearnedTreePriorOperator extends Operator {

    final public Input<Tree> treeInput = new Input<>("tree", "the tree whose statistics the parameters are drawn given", Input.Validate.REQUIRED);
    final public Input<State> stateInput = new Input<>("state", "the state, whose calculation nodes are brought up to date for every evaluation of a sweep", Input.Validate.REQUIRED);
    final public Input<List<Adapter>> adaptersInput = new Input<>("adapter", "adapters of the parameters of the tree prior, which give the real representation the operator works on: the mutable values are redrawn, and the immutable values are further conditions", new ArrayList<>());
    final public Input<List<Distribution>> distributionsInput = new Input<>("distribution", "the distributions that depend on the parameters, which are the tree prior and the priors of its parameters. Their log densities are what a sweep samples from, as the likelihood of the sequences does not depend on the parameters", new ArrayList<>());
    final public Input<Integer> sweepEveryInput = new Input<>("sweepEvery", "the number of proposals between two sweeps", 20);
    final public Input<Integer> numDrawsInput = new Input<>("numDraws", "the number of draws that a sweep records, each after a slice sampling step of every parameter", 4);
    final public Input<Integer> minRecordsInput = new Input<>("minRecords", "the number of recorded draws at which the distribution is fitted for the first time", 400);
    final public Input<Integer> maxRecordsInput = new Input<>("maxRecords", "the number of most recent draws that are kept", 4000);
    final public Input<Integer> burnInInput = new Input<>("burnIn", "the number of proposals before the first sweep, which are rejected", 2000);
    final public Input<Double> scaleFactorInput = new Input<>("scaleFactor", "the factor by which the learned covariance is multiplied, which widens the proposal for a target with heavier tails", 1.0);

    // the adapters have no use for the node, as the parameters of the tree prior do not belong to one
    private static final int NO_NODE = -1;

    protected Tree tree;
    protected State state;
    protected List<Adapter> adapters;
    private List<Distribution> distributions;
    private int numMutable;
    private int numImmutable;
    protected long numProposals;
    protected TreePriorSampler sampler;

    @Override
    public void initAndValidate() {
        this.tree = this.treeInput.get();
        this.state = this.stateInput.get();
        this.adapters = this.adaptersInput.get();
        this.distributions = this.distributionsInput.get();

        for (Adapter adapter : this.adapters) {
            this.numMutable += adapter.getNumMutable();
            this.numImmutable += adapter.getNumImmutable();
        }
        if (this.numMutable == 0) {
            throw new IllegalArgumentException("At least one adapter must have mutable values.");
        }
        if (this.distributions.isEmpty()) {
            throw new IllegalArgumentException("At least one distribution that depends on the parameters is required.");
        }

        this.sampler = new TreePriorSampler(this.numMutable, this.numImmutable, this.sweepEveryInput.get(), this.numDrawsInput.get(),
                this.minRecordsInput.get(), this.maxRecordsInput.get(), this.scaleFactorInput.get());
    }

    /**
     * Records the current values with a sweep if one is due, and redraws the mutable values given the tree. Returns the log
     * Hastings ratio, which is the change of the log Jacobian corrections of the adapters, plus the log density of the old
     * values given the new immutable values minus the log density of the new values given the old immutable values.
     */
    @Override
    public double proposal() {
        this.numProposals++;
        if (this.numProposals <= this.burnInInput.get()) {
            return Double.NEGATIVE_INFINITY;
        }

        double[] oldImmutable = this.readImmutable();
        double[] oldMutable = this.readMutable();
        this.sampler.record(this.tree, oldImmutable, oldMutable, this::evaluateLogTarget);
        if (!this.sampler.isFitted()) {
            return Double.NEGATIVE_INFINITY;
        }

        double[] newMutable = this.sampler.sample(this.tree, oldImmutable);
        double logCorrectionBefore = this.computeLogJacobianCorrection();
        if (!this.writeMutable(newMutable)) {
            return Double.NEGATIVE_INFINITY;
        }
        double logCorrectionAfter = this.computeLogJacobianCorrection();

        double[] newImmutable = this.readImmutable();
        if (!Arrays.stream(newImmutable).allMatch(Double::isFinite)) {
            return Double.NEGATIVE_INFINITY;
        }

        double logDensityForward = this.sampler.computeLogDensity(this.tree, oldImmutable, newMutable);
        double logDensityReverse = this.sampler.computeLogDensity(this.tree, newImmutable, oldMutable);
        return logCorrectionBefore - logCorrectionAfter + logDensityReverse - logDensityForward;
    }

    /**
     * Sets the parameters to the given values and returns the log density of their real representation under the
     * distributions that depend on them, which is their log density minus the log Jacobian corrections of the adapters.
     */
    protected double evaluateLogTarget(double[] mutable) {
        if (!this.writeMutable(mutable)) {
            return Double.NEGATIVE_INFINITY;
        }

        this.state.storeCalculationNodes();
        this.state.checkCalculationNodesDirtiness();

        double logDensity = 0.0;
        for (Distribution distribution : this.distributions) {
            logDensity += distribution.calculateLogP();
        }
        double logTarget = logDensity - this.computeLogJacobianCorrection();
        return Double.isNaN(logTarget) ? Double.NEGATIVE_INFINITY : logTarget;
    }

    protected double[] readMutable() {
        double[] mutable = new double[this.numMutable];

        int index = 0;
        for (Adapter adapter : this.adapters) {
            double[] values = adapter.getMutable(NO_NODE);
            System.arraycopy(values, 0, mutable, index, values.length);
            index += values.length;
        }
        return mutable;
    }

    protected double[] readImmutable() {
        double[] immutable = new double[this.numImmutable];

        int index = 0;
        for (Adapter adapter : this.adapters) {
            double[] values = adapter.getImmutable(NO_NODE);
            System.arraycopy(values, 0, immutable, index, values.length);
            index += values.length;
        }
        return immutable;
    }

    /** Sets the mutable values, and returns whether an adapter took them all. */
    protected boolean writeMutable(double[] mutable) {
        int index = 0;
        for (Adapter adapter : this.adapters) {
            int numValues = adapter.getNumMutable();
            if (numValues == 0) {
                continue;
            }

            try {
                adapter.update(Arrays.copyOfRange(mutable, index, index + numValues), NO_NODE);
            } catch (RuntimeException exception) {
                // the adapter cannot represent the values, as for a transform that leaves its domain
                return false;
            }
            index += numValues;
        }
        return true;
    }

    protected double computeLogJacobianCorrection() {
        double logCorrection = 0.0;
        for (Adapter adapter : this.adapters) {
            logCorrection += adapter.getLogJacobianCorrection(NO_NODE);
        }
        return logCorrection;
    }

    @Override
    public List<StateNode> listStateNodes() {
        List<StateNode> stateNodes = new ArrayList<>();
        for (Adapter adapter : this.adapters) {
            stateNodes.addAll(adapter.listStateNodes());
        }
        return stateNodes;
    }

}
