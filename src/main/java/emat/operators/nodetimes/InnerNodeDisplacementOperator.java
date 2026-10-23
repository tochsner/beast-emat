package emat.operators.nodetimes;

import beast.base.core.Description;
import beast.base.core.Input;
import beast.base.evolution.operator.TreeOperator;
import beast.base.evolution.tree.Node;
import beast.base.evolution.tree.Tree;
import beast.base.util.Randomizer;
import emat.prior.GeneticPrior;
import emat.operators.nodetimes.batch.TruncatedDensitySampler;
import emat.state.Mutation;
import emat.state.Mutations;

import java.util.List;

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
        double minHeight = NodeTimes.computeMinHeight(this.mutations, node);

        double newHeight;
        double logHastingsRatio;

        if (node.isRoot()) {
            newHeight = this.proposeRootHeight(node);
            logHastingsRatio = 0.0;

            if (newHeight <= minHeight) {
                return Double.NEGATIVE_INFINITY;
            }
        } else {
            double maxHeight = NodeTimes.computeMaxHeight(this.mutations, node);
            if (maxHeight <= minHeight) {
                return Double.NEGATIVE_INFINITY;
            }

            double decayRate = NodeTimes.computeDecayRate(this.geneticPrior, node);
            newHeight = TruncatedDensitySampler.sampleExponential(decayRate, minHeight, maxHeight);

            // the range does not depend on the height of X, so the reverse move draws from the same density
            logHastingsRatio = decayRate * (newHeight - oldHeight);
        }

        node.setHeight(newHeight);
        for (Node child : node.getChildren()) {
            List<Mutation> childMutations = this.mutations.getMutations(child);
            if (!childMutations.isEmpty()) {
                this.mutations.markBranchChanged(child, this);
            }
        }

        return logHastingsRatio;
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

}
