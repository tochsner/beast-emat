package emat.operators.nodetimes.batch;

import adapters.Adapter;
import beast.base.core.Description;
import beast.base.core.Input;
import beast.base.evolution.operator.TreeOperator;
import beast.base.evolution.tree.Node;
import beast.base.evolution.tree.Tree;
import beast.base.inference.StateNode;
import emat.operators.nodetimes.NodeTimes;
import emat.operators.nodetimes.batch.TruncatedDensityTransport.Tails;
import emat.prior.GeneticPrior;
import emat.state.Mutation;
import emat.state.Mutations;

import java.util.ArrayList;
import java.util.List;

@Description("Base of the operators that move the heights of many internal nodes at once in a coherent way. Every node " +
        "has a conditional density for its height within the range that keeps the topology and the times of all mutations " +
        "on its adjacent branches. It is the genetic prior times a learned quadratic approximation of the tree prior, and " +
        "it is transported to a reference distribution in which all nodes are moved together. The conditionals cancel " +
        "against the genetic prior, so only the error of the approximation of the tree prior enters the acceptance " +
        "probability, however far the nodes move.")
public abstract class TransportedNodeDisplacementOperator extends TreeOperator {

    final public Input<Mutations> mutationsInput = new Input<>("mutations", "");
    final public Input<GeneticPrior> geneticPriorInput = new Input<>("geneticPrior", "");
    final public Input<Double> stepSizeInput = new Input<>("stepSize", "the initial size of the coherent move, which is tuned if optimise is set. Its unit is given by the operator", 0.1);
    final public Input<Boolean> optimiseInput = new Input<>("optimise", "whether to tune the step size", true);
    final public Input<Double> targetAcceptanceInput = new Input<>("targetAcceptance", "acceptance probability towards which the step size is tuned", 0.6);
    final public Input<Boolean> learnTreePriorInput = new Input<>("learnTreePrior", "whether to learn an approximation of the tree prior of its own, instead of using the genetic prior only", true);
    final public Input<Integer> treePriorKnotsInput = new Input<>("treePriorKnots", "the number of knots along the height between which the slope of the approximation of the tree prior is interpolated", 20);
    final public Input<List<StateNode>> treePriorParametersInput = new Input<>("treePriorParameter", "real scalar, real vector or integer vector parameters of the tree prior, which the approximation of the tree prior may depend on", new ArrayList<>());
    final public Input<List<Adapter>> adaptersInput = new Input<>("adapter", "adapters of further parameters of the tree prior, whose mutable and immutable values the approximation of the tree prior may depend on. The operator never moves them", new ArrayList<>());
    final public Input<Integer> treePriorWarmUpInput = new Input<>("treePriorWarmUp", "the number of proposals after which the approximation of the tree prior is fitted for the first time", 500);
    final public Input<Boolean> scaleWeightByLogNumTipsInput = new Input<>("scaleWeightByLogNumTips", "whether to scale the weight by the natural logarithm of the number of tips n, so that the operator is selected with weight * ln(n). Otherwise, the weight is used as given", false);

    // the gain of the step size tuning decays with the number of proposals, but not below this value
    private static final double MIN_TUNING_GAIN = 0.01;
    private static final double MIN_STEP_SIZE = 1e-6;

    protected Tree tree;
    protected Mutations mutations;
    protected GeneticPrior geneticPrior;

    private double stepSize;

    // the factor ln(n) or one by which the weight of the operator is scaled
    private double weightFactor;

    // the approximation of the tree prior, or null to use the genetic prior only
    private LearnedTreePrior treePriorApproximation;
    private TreePriorParameterValues treePriorParameterValues;

    // the change of the approximate log tree prior over the nodes that moved in the last proposal, to second order
    private double approximateTreePriorChange;

    // the number of proposals the step size was tuned on
    private long numTuningSteps;

    /** The density of the height of a node within its range, and the transport between a height and its tails. */
    protected record Conditional(double decayRate, double midpoint, double minHeight, double maxHeight, double rate, double stiffness) {

        /** Returns the tails of the given height. */
        public Tails computeTails(double height) {
            return TruncatedDensityTransport.computeTails(this.rate, this.stiffness,
                    this.minHeight - this.midpoint, this.maxHeight - this.midpoint, height - this.midpoint);
        }

        /** Returns the height with the given tails. */
        public double computeHeight(Tails tails) {
            double offset = TruncatedDensityTransport.computeValue(this.rate, this.stiffness,
                    this.minHeight - this.midpoint, this.maxHeight - this.midpoint, tails);
            return Math.min(Math.max(this.midpoint + offset, this.minHeight), this.maxHeight);
        }

    }

    @Override
    public void initAndValidate() {
        this.tree = this.treeInput.get();
        this.mutations = this.mutationsInput.get();
        this.geneticPrior = this.geneticPriorInput.get();

        this.setCoercableParameterValue(this.stepSizeInput.get());
        this.weightFactor = this.scaleWeightByLogNumTipsInput.get() ? Math.log(this.tree.getLeafNodeCount()) : 1.0;

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
        this.approximateTreePriorChange = 0.0;
        if (this.treePriorApproximation != null) {
            this.treePriorParameterValues.read();
            this.treePriorApproximation.startObservation();
        }

        return this.moveNodes();
    }

    /** Moves the nodes and returns the log Hastings ratio, which is the sum of the values that {@link #displace} returns. */
    protected abstract double moveNodes();

    /** Returns the learned approximation of the tree prior, or null if the operator uses the genetic prior only. */
    protected LearnedTreePrior getTreePriorApproximation() {
        return this.treePriorApproximation;
    }

    /** Returns the values of the parameters of the tree prior that were read at the start of the proposal. */
    protected double[] getTreePriorParameters() {
        return this.treePriorParameterValues.getValues();
    }

    /** Adds to the change of the tree prior that the density of the proposal predicted, which the tree prior is learned from. */
    protected void addApproximateTreePriorChange(double change) {
        this.approximateTreePriorChange += change;
    }

    /** Returns whether the last proposal says something about the step size. Operators can opt out of tuning on proposals that hardly moved anything. */
    protected boolean isStepTuned() {
        return true;
    }

    /** Returns the largest step size, which depends on the unit of the step. */
    protected abstract double getMaxStepSize();

    /** Returns the conditional density of the height of the given node within the given range. */
    protected Conditional createConditional(Node node, double minHeight, double maxHeight) {
        double midpoint = (minHeight + maxHeight) / 2.0;
        double decayRate = NodeTimes.computeDecayRate(this.geneticPrior, node);

        double rate = decayRate;
        double stiffness = 0.0;
        if (this.treePriorApproximation != null) {
            double[] parameters = this.treePriorParameterValues.getValues();
            rate -= this.treePriorApproximation.getSlope(midpoint, parameters);
            stiffness = -this.treePriorApproximation.getUsableCurvature(minHeight, maxHeight, parameters);
        }

        return new Conditional(decayRate, midpoint, minHeight, maxHeight, rate, stiffness);
    }

    /**
     * Moves the given node to the given height, which is the transport of its old height under the conditional, and
     * returns its part of the log Hastings ratio. The conditional is the density exp(-k h + a(h)) of the genetic
     * prior with rate k times the approximation a of the tree prior. Moving by the transport changes the density
     * of the node by its ratio between the old and the new height, k (h' - h) - (a(h') - a(h)). Together with the
     * change of the genetic prior this leaves the change of the tree prior minus that of its approximation.
     */
    protected double displace(Node node, Conditional conditional, double newHeight) {
        double oldHeight = node.getHeight();

        double approximateChange = 0.0;
        if (this.treePriorApproximation != null) {
            approximateChange = this.treePriorApproximation.getChange(conditional.minHeight(), conditional.maxHeight(),
                    oldHeight, newHeight, this.treePriorParameterValues.getValues());
        }

        node.setHeight(newHeight);

        // the mutations on the branches below do not store the branch start, but the branches changed
        for (Node child : node.getChildren()) {
            List<Mutation> childMutations = this.mutations.getMutations(child);
            if (!childMutations.isEmpty()) {
                this.mutations.markBranchChanged(child, this);
            }
        }

        // remember the move for the fit of the approximation
        if (this.treePriorApproximation != null) {
            this.treePriorApproximation.addNode(conditional.midpoint(), this.treePriorParameterValues.getValues(),
                    oldHeight - conditional.midpoint(), newHeight - conditional.midpoint());
            this.approximateTreePriorChange += approximateChange;
        }

        return conditional.decayRate() * (newHeight - oldHeight) - approximateChange;
    }

    /* Tuning */

    /**
     * Passes the last proposal on to the approximation of the tree prior, and tunes the step size with a gain that
     * decays with the number of proposals down to a floor. The genetic prior cancels against its part of the
     * Hastings ratio, so the log acceptance ratio is the change of the log tree prior minus the change the
     * approximation predicted to second order. Adding the latter back gives the change of the log tree prior
     * over all nodes that moved.
     */
    @Override
    public void optimize(double logAlpha) {
        if (this.treePriorApproximation != null) {
            this.treePriorApproximation.finishObservation(logAlpha + this.approximateTreePriorChange);
        }

        if (this.optimiseInput.get() && this.isStepTuned()) {
            this.numTuningSteps++;
            double gain = Math.max(1.0 / (this.numTuningSteps + 1.0), MIN_TUNING_GAIN);
            double acceptance = Double.isNaN(logAlpha) ? 0.0 : Math.min(1.0, Math.exp(logAlpha));

            double delta = gain * (acceptance - this.getTargetAcceptanceProbability()) + Math.log(this.getCoercableParameterValue());
            this.setCoercableParameterValue(Math.exp(delta));
        }
    }

    @Override
    public double getTargetAcceptanceProbability() {
        return this.targetAcceptanceInput.get();
    }

    @Override
    public double getCoercableParameterValue() {
        return this.stepSize;
    }

    @Override
    public void setCoercableParameterValue(double value) {
        this.stepSize = Math.max(MIN_STEP_SIZE, Math.min(value, this.getMaxStepSize()));
    }

}
