package emat.stochasticmapping;

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
 * states agree with the probability r = Σ_a π_a p_a, where p_a is the exact probability under
 * the rate matrix s Q of the substitution process that a site that starts and ends in state a
 * has mutations. Then the history of every chosen site is drawn from its exact conditional
 * distribution given its end states and, for an agreeing site, that it is not empty. As r does
 * not depend on the states, the agreeing sites are chosen by geometric skipping, so the cost
 * grows with the number of mutations rather than with the genome length. The Jukes-Cantor rate
 * is not used.
 */
public class HybridBranchHistories implements BranchHistorySampler {

    private final SubstitutionProcess process;
    private final int numStates;
    private final int numSites;
    private final UniformisedStochasticMapping siteMapping;

    // reusable transition probabilities exp(s Q t) of the branch of the current call
    private final double[] transitionProbabilities;

    public HybridBranchHistories(SubstitutionProcess process, int numSites) {
        this.process = process;
        this.numStates = process.getNumStates();
        this.numSites = numSites;
        this.siteMapping = new UniformisedStochasticMapping(this.numStates);
        this.transitionProbabilities = new double[this.numStates * this.numStates];
    }

    /** Samples the history of every site on the branch with the paths of the chosen sites drawn exactly. */
    @Override
    public List<Mutation> sampleBranchHistory(int nodeNr, double branchStartHeight, double branchEndHeight, double rateScale,
                                              double jukesCantorRate, SiteChanges differingSites, IntUnaryOperator agreeingStates) {
        List<Mutation> branchMutations = new ArrayList<>();
        double duration = branchStartHeight - branchEndHeight;
        if (duration <= 0.0) {
            // a branch of length zero cannot carry mutations, and its end states are equal
            return branchMutations;
        }

        this.updateModel(duration, rateScale);

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

        double selectionProbability = this.computeSelectionProbability(duration, rateScale);
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
     * branch. Every site with mutations contributes the conditional density of its path, and
     * the agreeing sites contribute the probability of choosing exactly the ones with mutations.
     */
    @Override
    public double computeLogBranchHistoryDensity(List<Mutation> branchMutations, double branchStartHeight, double branchEndHeight,
                                                 double rateScale, double jukesCantorRate, int numDifferingSites) {
        double duration = branchStartHeight - branchEndHeight;
        if (duration <= 0.0) {
            return 0.0;
        }

        this.updateModel(duration, rateScale);

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

        double selectionProbability = this.computeSelectionProbability(duration, rateScale);
        int numAgreeingSites = this.numSites - numDifferingSites;
        if (numChosenAgreeingSites > 0) {
            logDensity += numChosenAgreeingSites * Math.log(selectionProbability);
        }
        logDensity += (numAgreeingSites - numChosenAgreeingSites) * Math.log1p(-selectionProbability);

        return logDensity;
    }

    /** Takes Q from the substitution process and computes the transition probabilities exp(s Q t) of the branch. */
    private void updateModel(double duration, double rateScale) {
        this.siteMapping.setRateMatrix(this.process.getRateMatrix());
        this.process.computeTransitionProbabilities(rateScale * duration, this.transitionProbabilities);
    }

    /**
     * Computes the probability r = Σ_a π_a p_a of choosing an agreeing site, where
     * p_a = (P_aa(t) - exp(s Q_aa t)) / P_aa(t) is the probability that a site that starts and
     * ends in state a has mutations. The frequencies π weight the states, so r depends only on
     * the model and the branch, and both directions of a move use the same r for a branch.
     */
    private double computeSelectionProbability(double duration, double rateScale) {
        double[] frequencies = this.process.getFrequencies();
        double selectionProbability = 0.0;
        for (int state = 0; state < this.numStates; state++) {
            double keepProbability = this.transitionProbabilities[state * this.numStates + state];
            selectionProbability += frequencies[state] * Math.max(this.computeNonEmptyProbability(state, duration, rateScale), 0.0) / keepProbability;
        }
        return Math.min(selectionProbability, 1.0);
    }

    /** Computes the exact probability P_ii(t) - exp(R_ii t) of returning to the given state with at least one mutation. */
    private double computeNonEmptyProbability(int state, double duration, double rateScale) {
        int diagonal = state * this.numStates + state;
        return this.transitionProbabilities[diagonal] - Math.exp(rateScale * this.process.getRateMatrix()[diagonal] * duration);
    }

}
