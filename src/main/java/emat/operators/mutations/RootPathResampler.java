package emat.operators.mutations;

import beast.base.evolution.tree.Node;
import beast.base.inference.Operator;
import emat.helper.MutationPaths;
import emat.helper.NodeStateLookup;
import emat.helper.RootSequence;
import emat.helper.SiteChanges;
import emat.operators.topology.BranchMutations;
import emat.prior.GeneticPrior;
import emat.state.Mutation;
import emat.state.Mutations;
import emat.stochasticmapping.BranchHistorySampler;
import emat.stochasticmapping.SubstitutionProcess;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

/**
 * Resamples the history on the path from the left root child S up through the root R and
 * down to the right root child X (docs/mcmc-moves.md §5 with P' = P), optionally for a new
 * root height. The sequences of S and X stay fixed. The model is time-reversible, so the path
 * is mapped as one branch that starts in the sequence of S and ends in the sequence of X, and
 * its part above R is mirrored onto the R–S branch with the direction of every mutation
 * reversed. The state at R becomes the new root state. The real model, including the change
 * of the root prior, enters through the genetic prior.
 * <p>
 * The path is mapped in units of branch rate times time, in which it has the length
 * r_X (t_R - t_X) + r_S (t_R - t_S) and the rate matrix s Q with s the site rate. A mutation
 * on a branch with rate r has the density in height of its unit position times r. The
 * Jukes-Cantor rate μ̃ = (λ(S) + λ(X)) / 2L only depends on the fixed sequences of S and X,
 * so the reverse move uses the same rate. Missations are not supported.
 */
public final class RootPathResampler {

    private final Mutations mutations;
    private final GeneticPrior geneticPrior;
    private final SubstitutionProcess process;
    private final BranchHistorySampler historySampler;
    private final BranchMutations mutationLinker;

    // reusable site maps for the changes from the root down to both root children and the resulting differing sites
    private final SiteChanges startChanges;
    private final SiteChanges endChanges;
    private final SiteChanges differingSites;

    // reusable lookup of the states at the end of the path for the sites whose end states agree
    private final NodeStateLookup endStates;

    public RootPathResampler(Mutations mutations, GeneticPrior geneticPrior, SubstitutionProcess process, BranchHistorySampler historySampler) {
        this.mutations = mutations;
        this.geneticPrior = geneticPrior;
        this.process = process;
        this.historySampler = historySampler;

        int numSites = mutations.getReferenceSequence().length;
        this.mutationLinker = new BranchMutations(numSites);
        this.startChanges = new SiteChanges(numSites);
        this.endChanges = new SiteChanges(numSites);
        this.differingSites = new SiteChanges(numSites);
        this.endStates = new NodeStateLookup(mutations, numSites);
    }

    /**
     * Moves the binary root to the given height and resamples the history on the path between
     * its children, applying the new mutations for the given operator. The substitution
     * process must be up to date. Returns the log of α_mut(n → o) / α_mut(o → n).
     */
    public double resample(Node root, double newHeight, Operator operator) {
        Node start = root.getLeft();
        Node end = root.getRight();

        double startRate = this.geneticPrior.branchRateModel.getRateForBranch(start);
        double endRate = this.geneticPrior.branchRateModel.getRateForBranch(end);
        double siteRate = this.process.getSiteRate();
        int numSites = this.mutations.getReferenceSequence().length;
        double jukesCantorRate = 0.5 * (this.geneticPrior.getTotalMutationRate(start) + this.geneticPrior.getTotalMutationRate(end)) / numSites;

        // find the sites that differ between S and X, which are the start and end of the path

        MutationPaths.collectChanges(this.mutations, root, start, start.getHeight(), this.startChanges);
        MutationPaths.collectChanges(this.mutations, root, end, end.getHeight(), this.endChanges);
        MutationPaths.combineChanges(this.startChanges, this.endChanges, this.differingSites);
        int numDifferingSites = this.differingSites.getSize();

        // the old path of the reverse move, in the units of the old root height

        RootPath oldPath = new RootPath(root.getHeight(), start, end, startRate, endRate);
        List<Mutation> oldPathMutations = oldPath.collectPathMutations(this.mutations.getMutations(start), this.mutations.getMutations(end));
        double logBackwardDensity = this.historySampler.computeLogBranchHistoryDensity(
                oldPathMutations, oldPath.getLength(), 0.0, siteRate, jukesCantorRate, numDifferingSites
        ) + oldPath.computeLogJacobian(oldPathMutations);

        // sample the new path while the tree still holds the sequence of X, which the move keeps

        root.setHeight(newHeight);
        RootPath newPath = new RootPath(newHeight, start, end, startRate, endRate);
        List<Mutation> newPathMutations = this.historySampler.sampleBranchHistory(
                end.getNr(), newPath.getLength(), 0.0, siteRate, jukesCantorRate, this.differingSites, this.endStates.resetFor(end)
        );
        double logForwardDensity = this.historySampler.computeLogBranchHistoryDensity(
                newPathMutations, newPath.getLength(), 0.0, siteRate, jukesCantorRate, numDifferingSites
        ) + newPath.computeLogJacobian(newPathMutations);

        // the root state at a site is the state of S, unless the mirrored part of the path changes it

        Map<Integer, Integer> newRootStates = new HashMap<>();
        for (Mutation mutation : this.mutations.getMutations(start)) {
            newRootStates.put(mutation.site(), mutation.newState());
        }

        // split the path at the root, mirroring the part above it onto the R–S branch

        List<Mutation> newStartMutations = new ArrayList<>();
        List<Mutation> newEndMutations = new ArrayList<>();
        for (Mutation mutation : newPathMutations) {
            if (newPath.isOnEndBranch(mutation)) {
                newEndMutations.add(newPath.mapToEndBranch(mutation));
                continue;
            }

            // the path runs through its mutations from S up to R, so the last one at a site sets the root state
            newRootStates.put(mutation.site(), mutation.newState());
            newStartMutations.add(newPath.mapToStartBranch(mutation));
        }

        this.mutations.applyMutations(start, this.mutationLinker.linkMutations(newStartMutations), operator);
        this.mutations.applyMutations(end, this.mutationLinker.linkMutations(newEndMutations), operator);
        RootSequence.setRootStates(this.mutations, root, newRootStates, operator);

        return logBackwardDensity - logForwardDensity;
    }

    /**
     * The path from the start child S up through the root R and down to the end child X, in
     * units of branch rate times time, measured from X. The R–X branch covers the positions
     * from 0 to d_X = r_X (t_R - t_X), and the R–S branch the positions up to d_X + d_S with
     * d_S = r_S (t_R - t_S), which run backwards in time from R to S.
     */
    private record RootPath(double rootHeight, Node start, Node end, double startRate, double endRate) {

        private double getEndLength() {
            return this.endRate * (this.rootHeight - this.end.getHeight());
        }

        private double getLength() {
            return this.getEndLength() + this.startRate * (this.rootHeight - this.start.getHeight());
        }

        /**
         * Returns the mutations of both branches as mutations of the path, sorted by
         * descending position. A mutation of the R–S branch runs in the opposite direction on
         * the path, so its old and new states are swapped.
         */
        private List<Mutation> collectPathMutations(List<Mutation> startMutations, List<Mutation> endMutations) {
            List<Mutation> pathMutations = new ArrayList<>(startMutations.size() + endMutations.size());

            double endLength = this.getEndLength();
            for (Mutation mutation : startMutations) {
                double position = endLength + this.startRate * (this.rootHeight - mutation.time());
                pathMutations.add(new Mutation(this.end.getNr(), position, Mutation.BRANCH_START, mutation.site(), mutation.newState(), mutation.oldState()));
            }
            for (Mutation mutation : endMutations) {
                double position = this.endRate * (mutation.time() - this.end.getHeight());
                pathMutations.add(new Mutation(this.end.getNr(), position, Mutation.BRANCH_START, mutation.site(), mutation.oldState(), mutation.newState()));
            }

            pathMutations.sort((first, second) -> Double.compare(second.time(), first.time()));
            return pathMutations;
        }

        /** Returns the log of the factor that turns the density of the positions of the given path mutations into the density of their heights. */
        private double computeLogJacobian(List<Mutation> pathMutations) {
            int numEndMutations = 0;
            for (Mutation mutation : pathMutations) {
                numEndMutations += this.isOnEndBranch(mutation) ? 1 : 0;
            }

            return numEndMutations * Math.log(this.endRate) + (pathMutations.size() - numEndMutations) * Math.log(this.startRate);
        }

        private boolean isOnEndBranch(Mutation pathMutation) {
            return pathMutation.time() < this.getEndLength();
        }

        /** Returns the given path mutation on the R–X branch as a mutation at its height, kept within the branch. */
        private Mutation mapToEndBranch(Mutation pathMutation) {
            double height = this.end.getHeight() + pathMutation.time() / this.endRate;
            height = Math.min(Math.max(height, this.end.getHeight()), this.rootHeight);
            return new Mutation(this.end.getNr(), height, Mutation.BRANCH_START, pathMutation.site(), pathMutation.oldState(), pathMutation.newState());
        }

        /** Returns the given path mutation on the R–S branch as a mutation at its height, in the direction from R to S and kept within the branch. */
        private Mutation mapToStartBranch(Mutation pathMutation) {
            double height = this.rootHeight - (pathMutation.time() - this.getEndLength()) / this.startRate;
            height = Math.min(Math.max(height, this.start.getHeight()), this.rootHeight);
            return new Mutation(this.start.getNr(), height, Mutation.BRANCH_START, pathMutation.site(), pathMutation.newState(), pathMutation.oldState());
        }

    }

}
