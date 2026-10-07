package emat.operators;

import beast.base.core.Description;
import beast.base.core.Input;
import beast.base.evolution.operator.TreeOperator;
import beast.base.evolution.tree.Node;
import beast.base.evolution.tree.Tree;
import beast.base.inference.StateNode;
import beast.base.spec.type.RealScalar;
import beast.base.spec.type.RealVector;
import beast.base.util.Randomizer;
import emat.helper.TreePriorApproximation;
import emat.prior.GeneticPrior;
import emat.state.Mutation;
import emat.state.Mutations;

import java.util.ArrayList;
import java.util.HashSet;
import java.util.List;
import java.util.Set;

@Description("Displaces the heights of a batch of internal nodes, each within the range that keeps the topology and " +
    "the times of all mutations on its adjacent branches. Every node draws its new height from the conditional of " +
    "the genetic prior times a learned approximation of the tree prior, so only the error of that approximation " +
    "enters the acceptance probability.")
public class BatchNodeDisplacementOperator extends TreeOperator {

    final public Input<Mutations> mutationsInput = new Input<>("mutations", "");
    final public Input<GeneticPrior> geneticPriorInput = new Input<>("geneticPrior", "");
    final public Input<Integer> batchSizeInput = new Input<>("batchSize", "", 16);
    final public Input<Boolean> optimiseInput = new Input<>("optimise", "", true);
    final public Input<Integer> treePriorDegreeInput = new Input<>("treePriorDegree", "the degree of the polynomial that approximates the log tree prior around the midpoint of the range of a node, or 0 to draw from the genetic prior only", 3);
    final public Input<Integer> treePriorMidpointDegreeInput = new Input<>("treePriorMidpointDegree", "the highest power of the midpoint of the range of a node that the approximation of the tree prior may depend on", 3);
    final public Input<List<StateNode>> treePriorParametersInput = new Input<>("treePriorParameter", "real scalar or vector parameters of the tree prior, which the approximation of the tree prior may depend on", new ArrayList<>());
    final public Input<Integer> treePriorWarmUpInput = new Input<>("treePriorWarmUp", "the number of proposals after which the approximation of the tree prior is fitted for the first time", 500);

    private Tree tree;
    private Mutations mutations;
    private GeneticPrior geneticPrior;
    private double batchSize;

    // the ridge penalty per proposal of the fit of the approximation of the tree prior
    private static final double TREE_PRIOR_RIDGE = 1e-8;

    // the numbers of the internal nodes, in the order they were visited for the last batch
    private int[] internalNodeNrs;

    // the approximation of the tree prior, or null to draw from the genetic prior only
    private TreePriorApproximation treePriorApproximation;

    // the values of the parameters of the tree prior at the start of the last proposal
    private double[] treePriorParameterValues;

    // the change of the approximate log tree prior over the nodes of the last batch, to first order
    private double approximateTreePriorChange;

    @Override
    public void initAndValidate() {
        this.tree = this.treeInput.get();
        this.mutations = this.mutationsInput.get();
        this.geneticPrior = this.geneticPriorInput.get();

        // BEAST numbers the internal nodes after the leaves
        int numLeaves = this.tree.getLeafNodeCount();
        this.internalNodeNrs = new int[this.tree.getInternalNodeCount()];
        for (int i = 0; i < this.internalNodeNrs.length; i++) {
            this.internalNodeNrs[i] = numLeaves + i;
        }

        this.setCoercableParameterValue(this.batchSizeInput.get());

        if (this.treePriorDegreeInput.get() > 0) {
            this.treePriorParameterValues = new double[this.countTreePriorParameterValues()];
            this.treePriorApproximation = new TreePriorApproximation(
                    this.treePriorDegreeInput.get(), this.treePriorMidpointDegreeInput.get(),
                    this.treePriorParameterValues.length, this.treePriorWarmUpInput.get(), TREE_PRIOR_RIDGE
            );
        }
    }

    @Override
    public double proposal() {
        List<Node> batch = this.selectBatch();

        this.approximateTreePriorChange = 0.0;
        if (this.treePriorApproximation != null) {
            this.readTreePriorParameterValues();
            this.treePriorApproximation.startObservation();
        }

        double logHastingsRatio = 0.0;
        for (Node node : batch) {
            logHastingsRatio += this.displace(node);
        }

        return logHastingsRatio;
    }

    /**
     * Selects a set of internal nodes to be changed. The set does not contain the root. No parent or child of a
     * node in the set is in the set as well. No node without room to move between its neighbours and the mutations
     * around it is in the set. The choice only depends on the topology, on the mutations and on the heights of
     * nodes outside the set, so it is symmetric and has no effect on the Hastings ratio.
     */
    private List<Node> selectBatch() {
        int targetSize = Math.toIntExact(Math.round(this.batchSize));

        List<Node> batch = new ArrayList<>(targetSize);
        Set<Node> selectedNodes = new HashSet<>();

        // visit the internal nodes in random order until the batch is full, skipping the ones that cannot join it

        int numInternalNodes = this.internalNodeNrs.length;
        for (int i = 0; i < numInternalNodes && batch.size() < targetSize; i++) {
            // shuffle the next node into place, so that every node is visited at most once
            int j = i + Randomizer.nextInt(numInternalNodes - i);
            int nodeNr = this.internalNodeNrs[j];
            this.internalNodeNrs[j] = this.internalNodeNrs[i];
            this.internalNodeNrs[i] = nodeNr;

            Node node = this.tree.getNode(nodeNr);
            if (node.isRoot()) {
                continue;
            }

            Node parent = node.getParent();
            Node left = node.getLeft();
            Node right = node.getRight();

            if (selectedNodes.contains(parent) || selectedNodes.contains(left) || selectedNodes.contains(right)) {
                continue;
            }

            if (this.computeMaxHeight(node) <= this.computeMinHeight(node)) {
                continue;
            }

            batch.add(node);
            selectedNodes.add(node);
        }

        return batch;
    }

    /**
     * Displaces the given node within its allowed range and returns the log Hastings ratio. The new height is drawn
     * from the density ∝ exp(-(k - s) h). The first factor exp(-k h) is the genetic prior as a function of the
     * height, which is exact as the mutations on the three neighbouring branches keep their times. The second
     * factor exp(s h) approximates the tree prior by its learned slope s at the midpoint of the range. Neither
     * the range nor the rates depend on the height of the node, so the reverse move draws from the same density.
     */
    private double displace(Node node) {
        double minHeight = this.computeMinHeight(node);
        double maxHeight = this.computeMaxHeight(node);
        double midpoint = (minHeight + maxHeight) / 2.0;

        // sample new node height

        double treePriorSlope = 0.0;
        if (this.treePriorApproximation != null) {
            treePriorSlope = this.treePriorApproximation.getSlope(midpoint, this.treePriorParameterValues);
        }

        double decayRate = this.computeDecayRate(node) - treePriorSlope;
        double newHeight = this.sampleTruncatedExponential(decayRate, minHeight, maxHeight);

        double oldHeight = node.getHeight();
        node.setHeight(newHeight);

        double oldHeightDensity = this.getTruncatedExponentialLogDensity(oldHeight, decayRate, minHeight, maxHeight);
        double newHeightDensity = this.getTruncatedExponentialLogDensity(newHeight, decayRate, minHeight, maxHeight);
        double logHastingsRatio = oldHeightDensity - newHeightDensity;

        // let the branches below start at the new height

        for (Node child : node.getChildren()) {
            this.moveBranchStart(child, newHeight);
        }

        // remember the move for the fit of the approximation

        if (this.treePriorApproximation != null) {
            this.treePriorApproximation.addNode(midpoint, this.treePriorParameterValues, oldHeight - midpoint, newHeight - midpoint);
            this.approximateTreePriorChange += treePriorSlope * (newHeight - oldHeight);
        }

        return logHastingsRatio;
    }

    /** Returns the number of values of the parameters of the tree prior, and checks that they are real. */
    private int countTreePriorParameterValues() {
        int numValues = 0;
        for (StateNode parameter : this.treePriorParametersInput.get()) {
            if (parameter instanceof RealScalar<?>) {
                numValues++;
            } else if (parameter instanceof RealVector<?> vector) {
                numValues += vector.size();
            } else {
                throw new IllegalArgumentException("The tree prior parameter " + parameter.getID() + " is neither a real scalar nor a real vector.");
            }
        }
        return numValues;
    }

    /** Reads the current values of the parameters of the tree prior. */
    private void readTreePriorParameterValues() {
        int index = 0;
        for (StateNode parameter : this.treePriorParametersInput.get()) {
            if (parameter instanceof RealScalar<?> scalar) {
                this.treePriorParameterValues[index++] = scalar.get();
            } else if (parameter instanceof RealVector<?> vector) {
                for (int i = 0; i < vector.size(); i++) {
                    this.treePriorParameterValues[index++] = vector.get(i);
                }
            }
        }
    }

    /**
     * Returns the lowest height the given node can take: above each child and above every mutation on the
     * branches below it. The mutations are sorted by descending height, so the first one of each branch is the
     * highest.
     */
    private double computeMinHeight(Node node) {
        double minHeight = Double.NEGATIVE_INFINITY;

        for (Node child : node.getChildren()) {
            minHeight = Math.max(minHeight, child.getHeight());

            List<Mutation> childMutations = this.mutations.getMutations(child);
            if (!childMutations.isEmpty()) {
                minHeight = Math.max(minHeight, childMutations.getFirst().time());
            }
        }

        return minHeight;
    }

    /**
     * Returns the highest height the given non-root node can take: below its parent and below every mutation on
     * the branch above it. The mutations are sorted by descending height, so the last one is the lowest.
     */
    private double computeMaxHeight(Node node) {
        double maxHeight = node.getParent().getHeight();

        List<Mutation> branchMutations = this.mutations.getMutations(node);
        if (!branchMutations.isEmpty()) {
            maxHeight = Math.min(maxHeight, branchMutations.getLast().time());
        }

        return maxHeight;
    }

    /**
     * Returns the rate k of the genetic prior as a function of the height h of the given non-root node,
     * G ∝ exp(-k h). The sequence at the node, with total mutation rate λ, spans the end of the branch above it
     * and the start of each branch below it. Raising the node by dh shortens the former by dh and lengthens each
     * of the latter by dh, so k = λ (Σ_c r_c - r) with the branch rates r_c of the children and r of the node.
     */
    private double computeDecayRate(Node node) {
        double childBranchRates = 0.0;
        for (Node child : node.getChildren()) {
            childBranchRates += this.geneticPrior.branchRateModel.getRateForBranch(child);
        }

        double branchRate = this.geneticPrior.branchRateModel.getRateForBranch(node);
        return this.geneticPrior.getTotalMutationRate(node) * (childBranchRates - branchRate);
    }

    /**
     * Moves the start of the branch above the given node to the given height. The mutation times stay, but the
     * first mutation at each site starts at the new branch start.
     */
    private void moveBranchStart(Node node, double branchStartHeight) {
        List<Mutation> branchMutations = this.mutations.getMutations(node);
        if (branchMutations.isEmpty()) {
            return;
        }

        Set<Integer> seenSites = new HashSet<>();
        List<Mutation> newBranchMutations = new ArrayList<>(branchMutations.size());

        for (Mutation mutation : branchMutations) {
            double timeOfPreviousMutation = seenSites.add(mutation.site())
                    ? branchStartHeight
                    : mutation.timeOfPreviousMutation();

            newBranchMutations.add(new Mutation(
                    mutation.nodeNr(), mutation.time(), timeOfPreviousMutation,
                    mutation.site(), mutation.oldState(), mutation.newState()
            ));
        }

        this.mutations.applyMutations(node, newBranchMutations, this);
    }

    /**
     * Samples a height from the density ∝ exp(-k h) truncated to [minHeight, maxHeight] by inverting its
     * cumulative distribution. The rate may be negative, in which case the density grows towards maxHeight, and
     * it falls back to a uniform draw if the density is flat on the range.
     */
    private double sampleTruncatedExponential(double decayRate, double minHeight, double maxHeight) {
        double width = maxHeight - minHeight;
        double u = Randomizer.nextDouble();

        double height;
        if (Math.abs(decayRate) * width < 1e-12) {
            height = minHeight + u * width;
        } else if (decayRate > 0.0) {
            height = minHeight - Math.log1p(u * Math.expm1(-decayRate * width)) / decayRate;
        } else {
            height = maxHeight - Math.log1p(u * Math.expm1(decayRate * width)) / decayRate;
        }

        // rounding must not push the height out of the range
        return Math.min(Math.max(height, minHeight), maxHeight);
    }

    /**
     * Returns the log of the normalised density ∝ exp(-k h) truncated to [minHeight, maxHeight] at the given
     * height. The density is measured from the end of the range where it is largest, so that a large rate of
     * either sign cannot overflow.
     */
    private double getTruncatedExponentialLogDensity(double height, double decayRate, double minHeight, double maxHeight) {
        double width = maxHeight - minHeight;

        if (Math.abs(decayRate) * width < 1e-12) {
            return -Math.log(width);
        } else if (decayRate > 0.0) {
            return -decayRate * (height - minHeight) + Math.log(decayRate) - Math.log(-Math.expm1(-decayRate * width));
        } else {
            return -decayRate * (height - maxHeight) + Math.log(-decayRate) - Math.log(-Math.expm1(decayRate * width));
        }
    }

    /* Tuning */

    /**
     * Passes the last proposal on to the approximation of the tree prior, and tunes the batch size. The genetic
     * prior cancels against its part of the Hastings ratio, so the log acceptance ratio is the change of the log
     * tree prior minus the change the approximation predicted to first order. Adding the latter back gives the
     * change of the log tree prior over the whole batch.
     */
    @Override
    public void optimize(double logAlpha) {
        if (this.treePriorApproximation != null) {
            this.treePriorApproximation.finishObservation(logAlpha + this.approximateTreePriorChange);
        }

        if (this.optimiseInput.get()) {
            double delta = this.calcDelta(logAlpha) + Math.log(this.getCoercableParameterValue());
            this.setCoercableParameterValue(Math.exp(delta));
        }
    }

    @Override
    public double getTargetAcceptanceProbability() {
        return 0.234;
    }

    @Override
    public double getCoercableParameterValue() {
        return this.batchSize;
    }

    @Override
    public void setCoercableParameterValue(double value) {
        // a batch holds at least one node and cannot hold more than the internal nodes below the root
        int maxBatchSize = Math.max(this.tree.getInternalNodeCount() - 1, 1);
        this.batchSize = Math.max(1.0, Math.min(value, maxBatchSize));
    }

}
