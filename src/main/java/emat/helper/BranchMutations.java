package emat.helper;

import emat.state.Mutation;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.HashSet;
import java.util.List;
import java.util.Map;
import java.util.Set;

/**
 * Joins and splits the mutation lists of branches when a node is removed from or inserted
 * into a branch, keeping the times of previous mutations consistent.
 */
public final class BranchMutations {

    /** The mutations above and below the height at which a branch was split. */
    public record Split(List<Mutation> upperMutations, List<Mutation> lowerMutations) {
    }

    private BranchMutations() {
    }

    /**
     * Joins an upper and a lower branch into one branch that belongs to the given node and
     * starts at the given height. The first mutation at a site on the lower branch now
     * follows the last mutation at that site on the upper branch, or the start of the
     * joined branch if there is none.
     */
    public static List<Mutation> joinBranches(List<Mutation> upperMutations, List<Mutation> lowerMutations,
                                              int nodeNr, double branchStartHeight) {
        List<Mutation> joinedMutations = new ArrayList<>();
        Map<Integer, Double> lastMutationTimeAtSite = new HashMap<>();

        for (Mutation mutation : upperMutations) {
            joinedMutations.add(new Mutation(
                    nodeNr, mutation.time(), mutation.timeOfPreviousMutation(),
                    mutation.site(), mutation.oldState(), mutation.newState()
            ));
            lastMutationTimeAtSite.put(mutation.site(), mutation.time());
        }

        Set<Integer> seenSites = new HashSet<>();
        for (Mutation mutation : lowerMutations) {
            double timeOfPreviousMutation = mutation.timeOfPreviousMutation();
            if (seenSites.add(mutation.site())) {
                timeOfPreviousMutation = lastMutationTimeAtSite.getOrDefault(mutation.site(), branchStartHeight);
            }

            joinedMutations.add(new Mutation(
                    nodeNr, mutation.time(), timeOfPreviousMutation,
                    mutation.site(), mutation.oldState(), mutation.newState()
            ));
        }

        return joinedMutations;
    }

    /**
     * Splits the mutations of a branch at the given height. The mutations above it move to
     * the upper node, and the others to the lower node, where the first mutation at a site
     * now starts at the split height.
     */
    public static Split splitBranch(List<Mutation> branchMutations, double splitHeight, int upperNodeNr, int lowerNodeNr) {
        List<Mutation> upperMutations = new ArrayList<>();
        List<Mutation> lowerMutations = new ArrayList<>();
        Set<Integer> seenLowerSites = new HashSet<>();

        for (Mutation mutation : branchMutations) {
            if (mutation.time() >= splitHeight) {
                upperMutations.add(new Mutation(
                        upperNodeNr, mutation.time(), mutation.timeOfPreviousMutation(),
                        mutation.site(), mutation.oldState(), mutation.newState()
                ));
                continue;
            }

            double timeOfPreviousMutation = seenLowerSites.add(mutation.site())
                    ? splitHeight
                    : mutation.timeOfPreviousMutation();

            lowerMutations.add(new Mutation(
                    lowerNodeNr, mutation.time(), timeOfPreviousMutation,
                    mutation.site(), mutation.oldState(), mutation.newState()
            ));
        }

        return new Split(upperMutations, lowerMutations);
    }

}
