package emat.helper;

import beast.base.evolution.tree.Node;
import beast.base.inference.Operator;
import emat.state.Mutation;
import emat.state.Mutations;

import java.util.ArrayList;
import java.util.Comparator;
import java.util.List;

/** Reads and replaces the mutations of a single site on a branch. */
public final class SiteMutations {

    private SiteMutations() {
    }

    /** Collects the mutations at the given site on the branch above the given node, sorted by descending height. */
    public static List<Mutation> collect(Mutations mutations, Node node, int site) {
        List<Mutation> siteMutations = new ArrayList<>();
        for (Mutation mutation : mutations.getMutations(node)) {
            if (mutation.site() == site) {
                siteMutations.add(mutation);
            }
        }
        return siteMutations;
    }

    /**
     * Replaces the mutations at the given site on the branch above the given node, keeping
     * the mutations at all other sites. Branches without mutations at the site before and
     * after are left untouched.
     */
    public static void replace(Mutations mutations, Node node, int site, List<Mutation> newSiteMutations, Operator operator) {
        List<Mutation> branchMutations = mutations.getMutations(node);

        // most branches have no mutations at the site before or after, so check this without allocating

        if (newSiteMutations.isEmpty() && !hasSiteMutation(branchMutations, site)) {
            return;
        }

        List<Mutation> newBranchMutations = new ArrayList<>();
        for (Mutation mutation : branchMutations) {
            if (mutation.site() != site) {
                newBranchMutations.add(mutation);
            }
        }

        newBranchMutations.addAll(newSiteMutations);
        newBranchMutations.sort(Comparator.comparingDouble(Mutation::time).reversed());
        mutations.applyMutations(node, newBranchMutations, operator);
    }

    /** Returns whether any of the given mutations is at the given site. */
    private static boolean hasSiteMutation(List<Mutation> branchMutations, int site) {
        for (int i = 0; i < branchMutations.size(); i++) {
            if (branchMutations.get(i).site() == site) {
                return true;
            }
        }
        return false;
    }

}
