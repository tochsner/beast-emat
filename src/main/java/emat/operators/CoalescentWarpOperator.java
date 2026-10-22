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

import java.util.ArrayList;
import java.util.List;

@Description("Proposes a joint move of the population size and the growth rate of an exponential growth coalescent, " +
        "the heights of all internal nodes and the clock rate. The parameters take a correlated Gaussian step, and every " +
        "node keeps the interval to its older child in coalescent time, the integral of the inverse population size. " +
        "The clock rate then follows from keeping the product of the clock rate and the tree length.")
public class CoalescentWarpOperator extends TreeOperator {

    // below this growth rate, the population size is treated as constant
    private static final double MIN_GROWTH_RATE = 1e-12;

    final public Input<Mutations> mutationsInput = new Input<>("mutations", "the mutations to move along with the tree", Input.Validate.REQUIRED);
    final public Input<StateNode> popSizeInput = new Input<>("popSize", "the population size at height zero of the exponential growth coalescent", Input.Validate.REQUIRED);
    final public Input<StateNode> growthRateInput = new Input<>("growthRate", "the growth rate of the exponential growth coalescent, with the population size exp(-rate * height) * popSize", Input.Validate.REQUIRED);
    final public Input<StateNode> clockRateInput = new Input<>("clockRate", "the clock rate, which is set so that its product with the tree length stays the same", Input.Validate.REQUIRED);
    final public Input<Double> sdLogPopSizeInput = new Input<>("sdLogPopSize", "the standard deviation of the step in the logarithm of the population size, before scaling", Input.Validate.REQUIRED);
    final public Input<Double> sdGrowthRateInput = new Input<>("sdGrowthRate", "the standard deviation of the step in the growth rate, before scaling", Input.Validate.REQUIRED);
    final public Input<Double> correlationInput = new Input<>("correlation", "the correlation between the two steps", Input.Validate.REQUIRED);
    final public Input<Double> scaleFactorInput = new Input<>("scaleFactor", "the factor by which both standard deviations are scaled, which is tuned if optimise is set", 1.0);
    final public Input<Boolean> optimiseInput = new Input<>("optimise", "whether to tune the scale factor towards the target acceptance probability", true);

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

        return this.warp(logPopSizeShift, growthRateShift);
    }

    /**
     * Shifts the logarithm of the population size and the growth rate by the given amounts, and maps the tree and
     * the clock rate accordingly. Returns the log Hastings ratio, which is the log Jacobian of the map from the old
     * to the new heights, the clock rate and the population size on the natural scale, plus one term for every
     * mutation. A step with the opposite shifts undoes the move and has the negated Hastings ratio.
     */
    public double warp(double logPopSizeShift, double growthRateShift) {
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
     * Moves every internal node, after the nodes below it, to the height whose coalescent time above its older child
     * under the new parameters equals the coalescent time it had under the old parameters. Returns the sum over the
     * internal nodes of the log ratio of the old to the new slope of the coalescent time, which is the log Jacobian
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

    /** Returns the integral from zero to the height of the inverse population size. */
    private double computeCoalescentTime(double height, double popSize, double growthRate) {
        if (Math.abs(growthRate) < MIN_GROWTH_RATE) {
            return height / popSize;
        }
        return Math.expm1(growthRate * height) / (growthRate * popSize);
    }

    /** Returns the height at which the integral of the inverse population size equals the given coalescent time. */
    private double computeHeight(double coalescentTime, double popSize, double growthRate) {
        if (Math.abs(growthRate) < MIN_GROWTH_RATE) {
            return coalescentTime * popSize;
        }
        return Math.log1p(growthRate * popSize * coalescentTime) / growthRate;
    }

    private double computeLogInversePopSize(double height, double popSize, double growthRate) {
        return growthRate * height - Math.log(popSize);
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

    /* Tuning */

    @Override
    public void optimize(double logAlpha) {
        if (this.optimiseInput.get()) {
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
