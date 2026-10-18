package emat.stochasticmapping;

import beast.base.evolution.substitutionmodel.EigenDecomposition;
import beast.base.util.Randomizer;
import emat.helper.SiteChanges;
import emat.state.Mutation;

import java.util.ArrayList;
import java.util.Comparator;
import java.util.List;
import java.util.function.IntUnaryOperator;

/**
 * Samples the history of all sites on a branch given the states at both of its ends in two
 * steps, and computes the probability of proposing a given history. First, the sites that
 * receive mutations are chosen: every site whose end states differ, and every site whose end
 * states agree with the Jukes-Cantor probability r of a non-empty history with the given rate
 * μ̃, which is the same for all states. Then the history of every chosen site is drawn from its
 * exact conditional distribution under the rate matrix of the genetic prior, given its end
 * states and, for an agreeing site, that it is not empty. As r does not depend on the states,
 * the agreeing sites are chosen by geometric skipping, so the cost grows with the number of
 * mutations rather than with the genome length.
 */
public class HybridBranchHistories {

    private final int numStates;
    private final int numSites;
    private final UniformisedStochasticMapping siteMapping;

    // the rate matrix Q and its eigen decomposition set last
    private double[] rateMatrix;
    private EigenDecomposition eigenDecomposition;

    // reusable transition probabilities exp(s Q t) of the branch of the current call
    private final double[] transitionProbabilities;

    public HybridBranchHistories(int numStates, int numSites) {
        this.numStates = numStates;
        this.numSites = numSites;
        this.siteMapping = new UniformisedStochasticMapping(numStates);
        this.transitionProbabilities = new double[numStates * numStates];
    }

    /** Sets the rate matrix Q, stored row by row, and its eigen decomposition, which every following call scales by its rate scale. */
    public void setModel(double[] rateMatrix, EigenDecomposition eigenDecomposition) {
        this.rateMatrix = rateMatrix;
        this.eigenDecomposition = eigenDecomposition;
        this.siteMapping.setRateMatrix(rateMatrix);
    }

    /**
     * Samples the history of every site on the branch above the given node, which runs from
     * the start height down to the end height, under R = s Q for the given rate scale s, with
     * the agreeing sites chosen by Jukes-Cantor with the given rate μ̃. The differing sites map
     * every site whose end states differ to its states at the start and at the end. At every
     * other site, both end states equal the state returned by the given lookup, which is only
     * called for the chosen sites. The mutations are sorted by descending height.
     */
    public List<Mutation> sampleBranchHistory(int nodeNr, double branchStartHeight, double branchEndHeight, double rateScale,
                                              double selectionRate, SiteChanges differingSites, IntUnaryOperator agreeingStates) {
        List<Mutation> branchMutations = new ArrayList<>();
        double duration = branchStartHeight - branchEndHeight;
        if (duration <= 0.0) {
            // a branch of length zero cannot carry mutations, and its end states are equal
            return branchMutations;
        }

        this.computeTransitionProbabilities(duration, rateScale);

        // the differing sites always receive mutations

        for (int slot = 0; slot < differingSites.getSize(); slot++) {
            int startState = differingSites.getStartState(slot);
            int endState = differingSites.getEndState(slot);
            branchMutations.addAll(this.siteMapping.sampleSiteHistory(
                    nodeNr, differingSites.getSite(slot), branchStartHeight, branchEndHeight, startState, endState, rateScale,
                    this.transitionProbabilities[startState * this.numStates + endState]
            ));
        }

        // choose the agreeing sites by skipping over the ones that stay empty

        double selectionProbability = this.computeSelectionProbability(selectionRate * duration);
        if (selectionProbability > 0.0) {
            double logSkipProbability = Math.log1p(-selectionProbability);
            double site = 0.0;

            while (true) {
                site += Math.floor(Math.log(1.0 - Randomizer.nextDouble()) / logSkipProbability);
                if (site >= this.numSites) {
                    break;
                }

                int siteIndex = (int) site;
                if (!differingSites.containsSite(siteIndex)) {
                    int state = agreeingStates.applyAsInt(siteIndex);
                    branchMutations.addAll(this.siteMapping.sampleNonEmptySameStateHistory(
                            nodeNr, siteIndex, branchStartHeight, branchEndHeight, state, rateScale,
                            this.computeNonEmptyProbability(state, duration, rateScale)
                    ));
                }
                site++;
            }
        }

        branchMutations.sort(Comparator.comparingDouble(Mutation::time).reversed());
        return branchMutations;
    }

    /**
     * Computes the log probability that sampleBranchHistory proposes the given history on a
     * branch, given the number of sites whose end states differ. Every site with mutations
     * contributes the conditional density of its path, and the agreeing sites contribute the
     * probability of choosing exactly the ones with mutations.
     */
    public double computeLogBranchHistoryDensity(List<Mutation> branchMutations, double branchStartHeight, double branchEndHeight,
                                                 double rateScale, double selectionRate, int numDifferingSites) {
        double duration = branchStartHeight - branchEndHeight;
        if (duration <= 0.0) {
            return 0.0;
        }

        this.computeTransitionProbabilities(duration, rateScale);

        List<Mutation> sortedMutations = new ArrayList<>(branchMutations);
        sortedMutations.sort(Comparator.comparingInt(Mutation::site).thenComparing(Comparator.comparingDouble(Mutation::time).reversed()));

        double logDensity = 0.0;
        int numChosenAgreeingSites = 0;

        int start = 0;
        while (start < sortedMutations.size()) {
            int site = sortedMutations.get(start).site();
            int end = start;
            while (end < sortedMutations.size() && sortedMutations.get(end).site() == site) {
                end++;
            }

            List<Mutation> siteMutations = sortedMutations.subList(start, end);
            int startState = siteMutations.getFirst().oldState();
            int endState = siteMutations.getLast().newState();

            double conditionProbability;
            if (startState == endState) {
                conditionProbability = this.computeNonEmptyProbability(startState, duration, rateScale);
                numChosenAgreeingSites++;
            } else {
                conditionProbability = this.transitionProbabilities[startState * this.numStates + endState];
            }

            logDensity += this.siteMapping.computeLogSiteHistoryDensity(
                    siteMutations, branchStartHeight, branchEndHeight, startState, endState, rateScale, conditionProbability
            );
            start = end;
        }

        // the agreeing sites are chosen independently with the same probability

        double selectionProbability = this.computeSelectionProbability(selectionRate * duration);
        int numAgreeingSites = this.numSites - numDifferingSites;
        if (numChosenAgreeingSites > 0) {
            logDensity += numChosenAgreeingSites * Math.log(selectionProbability);
        }
        logDensity += (numAgreeingSites - numChosenAgreeingSites) * Math.log1p(-selectionProbability);

        return logDensity;
    }

    /**
     * Computes the Jukes-Cantor probability r that a site keeps its state with at least one
     * mutation, given that it keeps its state, after the given expected number of jumps x:
     * r = 1 - e^{-x} / P_ii(x), with P_ii(x) = 1/K + (K - 1)/K e^{-K x / (K - 1)} for K states.
     */
    private double computeSelectionProbability(double expectedJumps) {
        double keepProbability = 1.0 / this.numStates
                + (this.numStates - 1.0) / this.numStates * Math.exp(-expectedJumps * this.numStates / (this.numStates - 1.0));
        return Math.max(1.0 - Math.exp(-expectedJumps) / keepProbability, 0.0);
    }

    /** Computes the exact probability P_ii(t) - exp(R_ii t) of returning to the given state with at least one mutation. */
    private double computeNonEmptyProbability(int state, double duration, double rateScale) {
        int diagonal = state * this.numStates + state;
        return this.transitionProbabilities[diagonal] - Math.exp(rateScale * this.rateMatrix[diagonal] * duration);
    }

    private void computeTransitionProbabilities(double duration, double rateScale) {
        StochasticMapping.computeTransitionProbabilities(this.eigenDecomposition, rateScale * duration, this.transitionProbabilities);
    }

}
