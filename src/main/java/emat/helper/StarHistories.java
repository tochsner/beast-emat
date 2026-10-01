package emat.helper;

import beast.base.evolution.tree.Node;
import emat.state.Mutation;
import emat.stochasticmapping.JukesCantorStochasticMapping;

import java.util.List;
import java.util.function.Function;
import java.util.function.IntUnaryOperator;

/**
 * Reusable helpers to resample the histories on a star of three branches around a centre
 * under approximate Jukes-Cantor stochastic mapping (docs/mcmc-moves.md §3.5). The star
 * states hold the outer ends in the columns TOP, FIRST and SECOND and the centre in the
 * column CENTRE, where the top end is the one above the centre and the other two lie below
 * it. The states are only held at sites where not all of them agree.
 */
public final class StarHistories {

    // the columns of the star states
    public static final int TOP = 0;
    public static final int FIRST = 1;
    public static final int SECOND = 2;
    public static final int CENTRE = 3;
    public static final int NUM_COLUMNS = CENTRE + 1;

    private final JukesCantorStochasticMapping stochasticMapping;

    // reusable site maps for the changes from the MRCA of the outer ends down to each of them and for the differing sites of a branch
    private final SiteChanges[] outerChanges;
    private final SiteChanges differingSites;

    public StarHistories(JukesCantorStochasticMapping stochasticMapping, int numSites) {
        this.stochasticMapping = stochasticMapping;
        this.outerChanges = new SiteChanges[]{new SiteChanges(numSites), new SiteChanges(numSites), new SiteChanges(numSites)};
        this.differingSites = new SiteChanges(numSites);
    }

    /**
     * Collects the states at the outer ends of a star at every site where they do not all
     * agree, from the changes on the paths from their MRCA down to each of them. The centre
     * states are left unset.
     */
    public void collectOuterStates(Function<Node, List<Mutation>> mutationsAboveNode, Node top, Node first, Node second,
                                   SiteStates states) {
        Node[] ends = {top, first, second};
        Node mrca = MutationPaths.findMrca(MutationPaths.findMrca(first, second), top);

        for (int i = 0; i < ends.length; i++) {
            MutationPaths.collectChanges(mutationsAboveNode, mrca, ends[i], ends[i].getHeight(), this.outerChanges[i]);
        }

        states.clear();
        int[] endStates = new int[ends.length];

        for (SiteChanges changes : this.outerChanges) {
            for (int slot = 0; slot < changes.getSize(); slot++) {
                int site = changes.getSite(slot);
                if (states.containsSite(site)) {
                    continue;
                }

                // an end without changes at the site keeps the state at the MRCA
                int mrcaState = changes.getStartState(slot);
                boolean isAgreeing = true;
                for (int i = 0; i < ends.length; i++) {
                    int endSlot = this.outerChanges[i].getSlot(site);
                    endStates[i] = endSlot >= 0 ? this.outerChanges[i].getEndState(endSlot) : mrcaState;
                    isAgreeing &= endStates[i] == endStates[0];
                }

                if (!isAgreeing) {
                    int statesSlot = states.addSite(site);
                    for (int i = 0; i < ends.length; i++) {
                        states.setState(statesSlot, i, endStates[i]);
                    }
                }
            }
        }
    }

    /**
     * Completes the states of a star with the states at its centre, which follow from the
     * states at the top end and the mutations on the branch from it down to the centre.
     * Sites where only the centre differs from the agreeing outer ends are added.
     */
    public void collectCentreStates(List<Mutation> topMutations, SiteStates states) {
        for (int slot = 0; slot < states.getSize(); slot++) {
            states.setState(slot, CENTRE, states.getState(slot, TOP));
        }

        // the mutations are sorted by descending height, so the first one at a site starts in the top state and the last one sets the centre state
        for (Mutation mutation : topMutations) {
            int slot = states.getSlot(mutation.site());
            if (slot < 0) {
                slot = states.addSite(mutation.site());
                states.setAllStates(slot, mutation.oldState());
            }
            states.setState(slot, CENTRE, mutation.newState());
        }
    }

    /**
     * Samples a new history on the branch above the given node, whose end states are held
     * by the given columns of the star states. At the sites without an entry, both ends are
     * in the state returned by the given lookup of the agreeing states. The returned
     * mutations are sorted by descending height.
     */
    public List<Mutation> sampleBranchHistory(Node node, double startHeight, double endHeight, double mutationRate,
                                              SiteStates states, int startColumn, int endColumn, IntUnaryOperator agreeingStates) {
        this.differingSites.clear();
        for (int slot = 0; slot < states.getSize(); slot++) {
            int startState = states.getState(slot, startColumn);
            int endState = states.getState(slot, endColumn);
            if (startState != endState) {
                this.differingSites.addSite(states.getSite(slot), startState, endState);
            }
        }

        return this.stochasticMapping.sampleBranchHistory(
                node.getNr(), startHeight, endHeight, mutationRate, this.differingSites,
                site -> {
                    int slot = states.getSlot(site);
                    return slot >= 0 ? states.getState(slot, startColumn) : agreeingStates.applyAsInt(site);
                }
        );
    }

    /**
     * Computes the log probability that sampleBranchHistory proposes the given history on a
     * branch whose end states are held by the given columns of the star states.
     */
    public double computeLogBranchHistoryDensity(List<Mutation> branchMutations, double startHeight, double endHeight,
                                                 double mutationRate, SiteStates states, int startColumn, int endColumn) {
        int numDifferingSites = 0;
        for (int slot = 0; slot < states.getSize(); slot++) {
            if (states.getState(slot, startColumn) != states.getState(slot, endColumn)) {
                numDifferingSites++;
            }
        }

        return this.stochasticMapping.computeLogBranchHistoryDensity(
                branchMutations, startHeight, endHeight, mutationRate, numDifferingSites
        );
    }

}
