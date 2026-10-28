package emat.operators.mutations;

import beast.base.evolution.tree.Node;
import beast.base.inference.Operator;
import emat.state.Mutation;
import emat.state.Mutations;
import emat.stochasticmapping.SiteHistorySampler;

import java.util.ArrayList;
import java.util.Comparator;
import java.util.List;
import java.util.function.IntPredicate;

/** Reads and replaces the mutations of single sites on a branch. */
public final class SiteMutations {

    private SiteMutations() {
    }

    /** Collects the mutations at the given site on the branch above the given node, sorted by descending height. */
    public static List<Mutation> collect(Mutations mutations, Node node, int site) {
        List<Mutation> siteMutations = new ArrayList<>();
        List<Mutation> branchMutations = mutations.getMutations(node);
        for (int i = 0; i < branchMutations.size(); i++) {
            Mutation mutation = branchMutations.get(i);
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
        replace(mutations, node, otherSite -> otherSite == site, newSiteMutations, operator);
    }

    /**
     * Replaces the mutations at the given sites on the branch above the given node by the
     * given mutations at these sites, keeping the mutations at all other sites. Branches
     * without mutations at the sites before and after are left untouched.
     */
    public static void replace(Mutations mutations, Node node, IntPredicate isReplaced, List<Mutation> newSiteMutations, Operator operator) {
        List<Mutation> branchMutations = mutations.getMutations(node);

        // most branches have no mutations at the sites before or after, so check this without allocating

        if (newSiteMutations.isEmpty() && !hasSiteMutation(branchMutations, isReplaced)) {
            return;
        }

        List<Mutation> newBranchMutations = new ArrayList<>();
        for (int i = 0; i < branchMutations.size(); i++) {
            Mutation mutation = branchMutations.get(i);
            if (!isReplaced.test(mutation.site())) {
                newBranchMutations.add(mutation);
            }
        }

        newBranchMutations.addAll(newSiteMutations);
        newBranchMutations.sort(Comparator.comparingDouble(Mutation::time).reversed());
        mutations.applyMutations(node, newBranchMutations, operator);
    }

    /**
     * Replaces the mutations at the given sites on every branch of the current region of the
     * given sampler by the given mutations, indexed like the region nodes.
     */
    public static void replaceRegion(Mutations mutations, SiteHistorySampler sampler, IntPredicate isReplaced, List<List<Mutation>> newRegionMutations, Operator operator) {
        for (int i = 0; i < sampler.getNumBranches(); i++) {
            replace(mutations, sampler.getBranchNode(i), isReplaced, newRegionMutations.get(i), operator);
        }
    }

    /** Returns whether any of the given mutations is at one of the given sites. */
    private static boolean hasSiteMutation(List<Mutation> branchMutations, IntPredicate isReplaced) {
        for (int i = 0; i < branchMutations.size(); i++) {
            if (isReplaced.test(branchMutations.get(i).site())) {
                return true;
            }
        }
        return false;
    }

}
