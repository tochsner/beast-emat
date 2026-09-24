package emat.helper;

import emat.state.Mutation;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

/**
 * Joins and splits the mutation lists of branches when a node is removed from or inserted
 * into a branch, keeping the times of previous mutations consistent. The per-site lookups
 * use arrays over all sites that are reused between calls, so an instance must not be
 * shared between threads.
 */
public final class BranchMutations {

    /** The mutations above and below the height at which a branch was split. */
    public record Split(List<Mutation> upperMutations, List<Mutation> lowerMutations) {
    }

    // an entry of the arrays below is only valid if its stamp equals the stamp of the current call, which avoids clearing them
    private final int[] upperStampOfSite;
    private final int[] lowerStampOfSite;
    private final double[] lastUpperTimeOfSite;
    private int stamp = 0;

    public BranchMutations(int numSites) {
        this.upperStampOfSite = new int[numSites];
        this.lowerStampOfSite = new int[numSites];
        this.lastUpperTimeOfSite = new double[numSites];
    }

    /**
     * Joins an upper and a lower branch into one branch that belongs to the given node and
     * starts at the given height. The first mutation at a site on the lower branch now
     * follows the last mutation at that site on the upper branch, or the start of the
     * joined branch if there is none.
     */
    public List<Mutation> joinBranches(List<Mutation> upperMutations, List<Mutation> lowerMutations,
                                       int nodeNr, double branchStartHeight) {
        int stamp = this.getNextStamp();
        List<Mutation> joinedMutations = new ArrayList<>(upperMutations.size() + lowerMutations.size());

        for (int i = 0; i < upperMutations.size(); i++) {
            Mutation mutation = upperMutations.get(i);
            joinedMutations.add(new Mutation(
                    nodeNr, mutation.time(), mutation.timeOfPreviousMutation(),
                    mutation.site(), mutation.oldState(), mutation.newState()
            ));

            // the mutations are sorted by descending height, so the last one at a site wins
            this.upperStampOfSite[mutation.site()] = stamp;
            this.lastUpperTimeOfSite[mutation.site()] = mutation.time();
        }

        for (int i = 0; i < lowerMutations.size(); i++) {
            Mutation mutation = lowerMutations.get(i);
            int site = mutation.site();

            double timeOfPreviousMutation = mutation.timeOfPreviousMutation();
            if (this.lowerStampOfSite[site] != stamp) {
                this.lowerStampOfSite[site] = stamp;
                timeOfPreviousMutation = this.upperStampOfSite[site] == stamp ? this.lastUpperTimeOfSite[site] : branchStartHeight;
            }

            joinedMutations.add(new Mutation(
                    nodeNr, mutation.time(), timeOfPreviousMutation,
                    site, mutation.oldState(), mutation.newState()
            ));
        }

        return joinedMutations;
    }

    /**
     * Splits the mutations of a branch at the given height. The mutations above it move to
     * the upper node, and the others to the lower node, where the first mutation at a site
     * now starts at the split height.
     */
    public Split splitBranch(List<Mutation> branchMutations, double splitHeight, int upperNodeNr, int lowerNodeNr) {
        int stamp = this.getNextStamp();

        // the mutations are sorted by descending height, so the upper ones come first
        int numUpperMutations = 0;
        while (numUpperMutations < branchMutations.size() && branchMutations.get(numUpperMutations).time() >= splitHeight) {
            numUpperMutations++;
        }

        List<Mutation> upperMutations = new ArrayList<>(numUpperMutations);
        List<Mutation> lowerMutations = new ArrayList<>(branchMutations.size() - numUpperMutations);

        for (int i = 0; i < numUpperMutations; i++) {
            Mutation mutation = branchMutations.get(i);
            upperMutations.add(new Mutation(
                    upperNodeNr, mutation.time(), mutation.timeOfPreviousMutation(),
                    mutation.site(), mutation.oldState(), mutation.newState()
            ));
        }

        for (int i = numUpperMutations; i < branchMutations.size(); i++) {
            Mutation mutation = branchMutations.get(i);
            int site = mutation.site();

            double timeOfPreviousMutation = mutation.timeOfPreviousMutation();
            if (this.lowerStampOfSite[site] != stamp) {
                this.lowerStampOfSite[site] = stamp;
                timeOfPreviousMutation = splitHeight;
            }

            lowerMutations.add(new Mutation(
                    lowerNodeNr, mutation.time(), timeOfPreviousMutation,
                    site, mutation.oldState(), mutation.newState()
            ));
        }

        return new Split(upperMutations, lowerMutations);
    }

    /** Returns a stamp that no entry holds yet, resetting all entries once the stamps run out. */
    private int getNextStamp() {
        if (this.stamp == Integer.MAX_VALUE) {
            Arrays.fill(this.upperStampOfSite, 0);
            Arrays.fill(this.lowerStampOfSite, 0);
            this.stamp = 0;
        }
        return ++this.stamp;
    }

}
