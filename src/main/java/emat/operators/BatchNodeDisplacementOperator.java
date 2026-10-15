package emat.operators;

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
import emat.helper.GlobalSplineTreePriorApproximation;
import emat.helper.LinearSplineTreePriorApproximation;
import emat.helper.LearnedTreePrior;
import emat.helper.TruncatedDensitySampler;
import emat.helper.TreePriorApproximation;
import emat.prior.GeneticPrior;
import emat.state.Mutation;
import emat.state.Mutations;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

@Description("Displaces the heights of a batch of internal nodes, each within the range that keeps the topology and " +
    "the times of all mutations on its adjacent branches. Every node draws its new height from the conditional of " +
    "the genetic prior times a learned quadratic approximation of the tree prior, so only the error of that approximation " +
    "enters the acceptance probability.")
public class BatchNodeDisplacementOperator extends TreeOperator {

    /** How the approximation of the tree prior depends on the midpoint of the range of a node. */
    public enum TreePriorShape {
        polynomial, spline, global
    }

    final public Input<Mutations> mutationsInput = new Input<>("mutations", "");
    final public Input<GeneticPrior> geneticPriorInput = new Input<>("geneticPrior", "");
    final public Input<Integer> batchSizeInput = new Input<>("batchSize", "the initial number of nodes that are displaced together, which is tuned if optimise is set, and which is capped at the number of internal nodes below the root", 128);
    final public Input<Boolean> optimiseInput = new Input<>("optimise", "", true);
    final public Input<Integer> treePriorDegreeInput = new Input<>("treePriorDegree", "the degree of the polynomial that approximates the log tree prior around the midpoint of the range of a node, or 0 to draw from the genetic prior only", 3);
    final public Input<TreePriorShape> treePriorShapeInput = new Input<>("treePriorShape", "how the approximation of the tree prior depends on the midpoint of the range of a node: as a polynomial in the midpoint, as a polynomial in the offset whose coefficients are linear splines in the midpoint, or as a single linear spline of the slope along the absolute height", TreePriorShape.polynomial, TreePriorShape.values());
    final public Input<Integer> treePriorMidpointDegreeInput = new Input<>("treePriorMidpointDegree", "the highest power of the midpoint of the range of a node that the approximation of the tree prior may depend on, if its shape is polynomial", 3);
    final public Input<Integer> treePriorKnotsInput = new Input<>("treePriorKnots", "the number of knots along the height between which the approximation of the tree prior is interpolated, if its shape is spline or global", 20);
    final public Input<List<StateNode>> treePriorParametersInput = new Input<>("treePriorParameter", "real scalar or vector parameters of the tree prior, which the approximation of the tree prior may depend on", new ArrayList<>());
    final public Input<Integer> treePriorWarmUpInput = new Input<>("treePriorWarmUp", "the number of proposals after which the approximation of the tree prior is fitted for the first time", 500);

    private Tree tree;
    private Mutations mutations;
    private GeneticPrior geneticPrior;
    private double batchSize;

    // the gain of the batch size tuning decays with the number of proposals, but not below this value, as the
    // approximation of the tree prior makes large batches acceptable and the batch size has to cover a wide range
    private static final double MIN_TUNING_GAIN = 0.01;

    // the ridge penalty per proposal of the fit of the approximation of the tree prior
    private static final double TREE_PRIOR_RIDGE = 1e-8;

    // the numbers of the internal nodes, in the order they were visited for the last batch
    private int[] internalNodeNrs;

    // the nodes of the last batch with the lowest and the highest height they can take, which do not change while
    // the batch is displaced as the heights and the mutation times that bound them belong to nodes outside the batch
    private Node[] batchNodes;
    private double[] batchMinHeights;
    private double[] batchMaxHeights;
    private int batchCount;

    // for every node number, whether the node is in the batch that is being selected
    private boolean[] isInBatch;

    // for every site, the stamp of the last branch on which a mutation at the site was seen
    private int[] siteStamps;
    private int siteStamp;

    // the approximation of the tree prior, or null to draw from the genetic prior only
    private LearnedTreePrior treePriorApproximation;

    // the values of the parameters of the tree prior at the start of the last proposal
    private double[] treePriorParameterValues;

    // the number of proposals the batch size was tuned on
    private long numTuningSteps;

    // whether the first fit of the approximation of the tree prior has been announced
    private boolean hasReportedFit;

    // the change of the approximate log tree prior over the nodes of the last batch, to second order
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

        this.batchNodes = new Node[this.internalNodeNrs.length];
        this.batchMinHeights = new double[this.internalNodeNrs.length];
        this.batchMaxHeights = new double[this.internalNodeNrs.length];
        this.isInBatch = new boolean[this.tree.getNodeCount()];
        this.siteStamps = new int[this.mutations.getReferenceSequence().length];

        this.setCoercableParameterValue(this.batchSizeInput.get());

        if (this.treePriorDegreeInput.get() > 0) {
            this.treePriorParameterValues = new double[this.countTreePriorParameterValues()];
            this.treePriorApproximation = this.createTreePriorApproximation();
        }
    }

    private LearnedTreePrior createTreePriorApproximation() {
        int degree = this.treePriorDegreeInput.get();
        int numParameters = this.treePriorParameterValues.length;
        int warmUp = this.treePriorWarmUpInput.get();

        if (this.treePriorShapeInput.get() == TreePriorShape.spline) {
            return new LinearSplineTreePriorApproximation(degree, this.treePriorKnotsInput.get(), numParameters, warmUp, TREE_PRIOR_RIDGE);
        }
        if (this.treePriorShapeInput.get() == TreePriorShape.global) {
            return new GlobalSplineTreePriorApproximation(this.treePriorKnotsInput.get(), numParameters, warmUp, TREE_PRIOR_RIDGE);
        }
        return new TreePriorApproximation(degree, this.treePriorMidpointDegreeInput.get(), numParameters, warmUp, TREE_PRIOR_RIDGE);
    }

    @Override
    public double proposal() {
        this.selectBatch();

        this.approximateTreePriorChange = 0.0;
        if (this.treePriorApproximation != null) {
            this.readTreePriorParameterValues();
            this.treePriorApproximation.startObservation();
        }

        double logHastingsRatio = 0.0;
        for (int i = 0; i < this.batchCount; i++) {
            logHastingsRatio += this.displace(this.batchNodes[i], this.batchMinHeights[i], this.batchMaxHeights[i]);
        }

        return logHastingsRatio;
    }

    /**
     * Selects a set of internal nodes to be changed, and the range of heights of every one of them. The set does
     * not contain the root. No parent or child of a node in the set is in the set as well. No node without room
     * to move between its neighbours and the mutations around it is in the set. The choice only depends on the
     * topology, on the mutations and on the heights of nodes outside the set, so it is symmetric and has no effect
     * on the Hastings ratio.
     */
    private void selectBatch() {
        int targetSize = Math.toIntExact(Math.round(this.batchSize));
        this.batchCount = 0;

        // visit the internal nodes in random order until the batch is full, skipping the ones that cannot join it

        int numInternalNodes = this.internalNodeNrs.length;
        for (int i = 0; i < numInternalNodes && this.batchCount < targetSize; i++) {
            // shuffle the next node into place, so that every node is visited at most once
            int j = i + Randomizer.nextInt(numInternalNodes - i);
            int nodeNr = this.internalNodeNrs[j];
            this.internalNodeNrs[j] = this.internalNodeNrs[i];
            this.internalNodeNrs[i] = nodeNr;

            Node node = this.tree.getNode(nodeNr);
            if (node.isRoot()) {
                continue;
            }

            Node left = node.getLeft();
            Node right = node.getRight();
            if (this.isInBatch[node.getParent().getNr()] || this.isInBatch[left.getNr()] || this.isInBatch[right.getNr()]) {
                continue;
            }

            double minHeight = this.computeMinHeight(node);
            double maxHeight = this.computeMaxHeight(node);
            if (maxHeight <= minHeight) {
                continue;
            }

            this.batchNodes[this.batchCount] = node;
            this.batchMinHeights[this.batchCount] = minHeight;
            this.batchMaxHeights[this.batchCount] = maxHeight;
            this.batchCount++;
            this.isInBatch[nodeNr] = true;
        }

        for (int i = 0; i < this.batchCount; i++) {
            this.isInBatch[this.batchNodes[i].getNr()] = false;
        }
    }

    /**
     * Displaces the given node within the given range and returns the log Hastings ratio. The new height h is
     * drawn from the density ∝ exp(-k h + a(h)). The first factor is the genetic prior as a function of the height,
     * which is exact as the mutations on the three neighbouring branches keep their times. The second factor
     * approximates the tree prior by the learned approximation a. Neither the range nor the rates depend on the
     * height of the node, so the reverse move draws from the same density and the normalisation cancels.
     */
    private double displace(Node node, double minHeight, double maxHeight) {
        double midpoint = (minHeight + maxHeight) / 2.0;

        double geneticDecayRate = this.computeDecayRate(node);
        double oldHeight = node.getHeight();
        double newHeight;
        double approximateChange = 0.0;

        // sample new node height

        if (this.treePriorApproximation == null) {
            newHeight = TruncatedDensitySampler.sampleExponential(geneticDecayRate, minHeight, maxHeight);
        } else {
            newHeight = this.treePriorApproximation.sampleHeight(geneticDecayRate, minHeight, maxHeight, this.treePriorParameterValues);
            approximateChange = this.treePriorApproximation.getChange(minHeight, maxHeight, oldHeight, newHeight, this.treePriorParameterValues);
        }

        double logHastingsRatio = geneticDecayRate * (newHeight - oldHeight) - approximateChange;

        node.setHeight(newHeight);

        // let the branches below start at the new height

        for (Node child : node.getChildren()) {
            this.moveBranchStart(child, newHeight);
        }

        // remember the move for the fit of the approximation

        if (this.treePriorApproximation != null) {
            this.treePriorApproximation.addNode(midpoint, this.treePriorParameterValues, oldHeight - midpoint, newHeight - midpoint);
            this.approximateTreePriorChange += approximateChange;
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
            } else if (parameter instanceof IntVector<?> vector) {
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
     * first mutation at each site starts at the new branch start. The other mutations are kept as they are.
     */
    private void moveBranchStart(Node node, double branchStartHeight) {
        List<Mutation> branchMutations = this.mutations.getMutations(node);
        int numMutations = branchMutations.size();
        if (numMutations == 0) {
            return;
        }

        // a site is marked as seen on this branch by the current stamp, which saves clearing the marks
        if (this.siteStamp == Integer.MAX_VALUE) {
            Arrays.fill(this.siteStamps, 0);
            this.siteStamp = 0;
        }
        this.siteStamp++;

        Mutation[] newBranchMutations = new Mutation[numMutations];
        for (int i = 0; i < numMutations; i++) {
            Mutation mutation = branchMutations.get(i);

            if (this.siteStamps[mutation.site()] == this.siteStamp) {
                newBranchMutations[i] = mutation;
            } else {
                this.siteStamps[mutation.site()] = this.siteStamp;
                newBranchMutations[i] = new Mutation(
                        mutation.nodeNr(), mutation.time(), branchStartHeight,
                        mutation.site(), mutation.oldState(), mutation.newState()
                );
            }
        }

        // an immutable list is not copied again when the mutations of the branch are set
        this.mutations.applyMutations(node, List.of(newBranchMutations), this);
    }

    /* Tuning */

    /**
     * Passes the last proposal on to the approximation of the tree prior, and tunes the batch size with a gain
     * that decays with the number of proposals down to a floor. The genetic prior cancels against its part of
     * the Hastings ratio, so the log acceptance ratio is the change of the log tree prior minus the change the
     * approximation predicted to second order. Adding the latter back gives the change of the log tree prior
     * over the whole batch.
     */
    @Override
    public void optimize(double logAlpha) {
        if (this.treePriorApproximation != null) {
            this.treePriorApproximation.finishObservation(logAlpha + this.approximateTreePriorChange);
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
