package emat.operators;

import beast.base.core.Description;
import beast.base.core.Input;
import beast.base.evolution.operator.TreeOperator;
import beast.base.evolution.tree.Node;
import beast.base.evolution.tree.Tree;
import beast.base.util.Randomizer;
import emat.prior.GeneticPrior;
import emat.state.Mutation;
import emat.state.Mutations;

import java.util.ArrayList;
import java.util.HashSet;
import java.util.List;
import java.util.Set;

@Description("Displaces the height of a random internal node within the range that keeps the topology and the " +
        "times of all mutations on its adjacent branches (docs/mcmc-moves.md §9.1). A non-root node draws its new " +
        "height from the conditional of the genetic prior, so only the tree prior enters the acceptance " +
        "probability. The root takes a Gaussian step instead.")
public class InnerNodeDisplacementOperator extends TreeOperator {

    final public Input<Mutations> mutationsInput = new Input<>("mutations", "the mutations to operate on", Input.Validate.REQUIRED);
    final public Input<GeneticPrior> geneticPriorInput = new Input<>("geneticPrior", "the genetic prior that defines the evolutionary model", Input.Validate.REQUIRED);

    Mutations mutations;
    GeneticPrior geneticPrior;
    Tree tree;

    @Override
    public void initAndValidate() {
        this.mutations = this.mutationsInput.get();
        this.geneticPrior = this.geneticPriorInput.get();
        this.tree = this.treeInput.get();
    }

    /**
     * Picks an internal node X uniformly, including the root, and displaces its height within
     * its allowed range. The mutations on the branches below X keep their times, and only the
     * first mutation at each site on them now starts at the new height of X.
     */
    @Override
    public double proposal() {
        int numLeaves = this.tree.getLeafNodeCount();
        int numInternalNodes = this.tree.getInternalNodeCount();
        if (numInternalNodes == 0) {
            return Double.NEGATIVE_INFINITY;
        }

        // BEAST numbers the internal nodes after the leaves
        Node node = this.tree.getNode(numLeaves + Randomizer.nextInt(numInternalNodes));

        double oldHeight = node.getHeight();
        double minHeight = this.computeMinHeight(node);

        double newHeight;
        double logHastingsRatio;

        if (node.isRoot()) {
            newHeight = this.proposeRootHeight(node);
            logHastingsRatio = 0.0;

            if (newHeight <= minHeight) {
                return Double.NEGATIVE_INFINITY;
            }
        } else {
            double maxHeight = this.computeMaxHeight(node);
            if (maxHeight <= minHeight) {
                return Double.NEGATIVE_INFINITY;
            }

            double decayRate = this.computeDecayRate(node);
            newHeight = this.sampleTruncatedExponential(decayRate, minHeight, maxHeight);

            // the range does not depend on the height of X, so the reverse move draws from the same density
            logHastingsRatio = decayRate * (newHeight - oldHeight);
        }

        node.setHeight(newHeight);
        for (Node child : node.getChildren()) {
            this.moveBranchStart(child, newHeight);
        }

        return logHastingsRatio;
    }

    /**
     * Returns the lowest height X can take: above each child and above every mutation on
     * the branches below X. The mutations are sorted by descending height, so the first one
     * of each branch is the highest.
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
     * Returns the highest height a non-root X can take: below its parent and below every
     * mutation on the branch above X. The mutations are sorted by descending height, so the
     * last one is the lowest.
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
     * Returns the rate k of the genetic prior as a function of the height h of a non-root X,
     * G ∝ exp(-k h). The sequence at X, with total mutation rate λ_X, spans the end of the
     * branch above X and the start of each branch below it. Raising X by dh shortens the
     * former by dh and lengthens each of the latter by dh, so k = λ_X (Σ_c r_c - r_X) with
     * branch rates r. For a strict clock this is λ_X r, i.e. exp(+λ_X t) in forward time.
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
     * Proposes a new root height by a Gaussian step with standard deviation 1 / (2 λ_X),
     * where λ_X is the total mutation rate of the root sequence times the mean branch rate
     * of its children. Neither depends on the root height, so the step is symmetric.
     */
    private double proposeRootHeight(Node root) {
        double childBranchRates = 0.0;
        for (Node child : root.getChildren()) {
            childBranchRates += this.geneticPrior.branchRateModel.getRateForBranch(child);
        }

        double mutationRate = this.geneticPrior.getTotalMutationRate(root) * childBranchRates / root.getChildCount();
        return root.getHeight() + Randomizer.nextGaussian() / (2.0 * mutationRate);
    }

    /**
     * Samples a height from the density ∝ exp(-k h) truncated to [minHeight, maxHeight] by
     * inverting its cumulative distribution. The rate may be negative, in which case the
     * density grows towards maxHeight, and it falls back to a uniform draw if the density is
     * flat on the range.
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
     * Moves the start of the branch above the given node to the given height. The mutation
     * times stay, but the first mutation at each site starts at the new branch start.
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

}
