package emat.operators.nodetimes;

import beast.base.evolution.tree.Node;
import beast.base.evolution.tree.Tree;
import emat.prior.GeneticPrior;
import emat.state.Mutation;
import emat.state.Mutations;

import java.util.List;

/** Computes the ranges and rates that govern the displacement of a node, and the length of a tree. */
public final class NodeTimes {

    private NodeTimes() {
    }

    /**
     * Returns the lowest height the given node can take: above each child and above every mutation on
     * the branches below it. The mutations are sorted by descending height, so the first one of each branch is the
     * highest.
     */
    public static double computeMinHeight(Mutations mutations, Node node) {
        double minHeight = Double.NEGATIVE_INFINITY;

        for (Node child : node.getChildren()) {
            minHeight = Math.max(minHeight, child.getHeight());

            List<Mutation> childMutations = mutations.getMutations(child);
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
    public static double computeMaxHeight(Mutations mutations, Node node) {
        double maxHeight = node.getParent().getHeight();

        List<Mutation> branchMutations = mutations.getMutations(node);
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
    public static double computeDecayRate(GeneticPrior geneticPrior, Node node) {
        double childBranchRates = 0.0;
        for (Node child : node.getChildren()) {
            childBranchRates += geneticPrior.branchRateModel.getRateForBranch(child);
        }

        double branchRate = geneticPrior.branchRateModel.getRateForBranch(node);
        return geneticPrior.getTotalMutationRate(node) * (childBranchRates - branchRate);
    }

    /** Returns the sum of the lengths of all branches. */
    public static double computeTreeLength(Tree tree) {
        double length = 0.0;
        for (Node node : tree.getNodesAsArray()) {
            length += node.getLength();
        }
        return length;
    }

}
