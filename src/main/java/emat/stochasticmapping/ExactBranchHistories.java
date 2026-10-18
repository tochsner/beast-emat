package emat.stochasticmapping;

import beast.base.evolution.substitutionmodel.EigenDecomposition;
import emat.helper.SiteChanges;
import emat.state.Mutation;

import java.util.ArrayList;
import java.util.Comparator;
import java.util.List;
import java.util.function.IntUnaryOperator;

/**
 * Samples the history of all sites on a branch given the states at both of its ends from its
 * exact conditional distribution under the rate matrix of the genetic prior, and computes the
 * probability of proposing a given history. Every site is mapped on its own by uniformisation,
 * so the cost grows with the genome length rather than with the number of differing sites.
 */
public class ExactBranchHistories {

    private final int numStates;
    private final int numSites;
    private final UniformisedStochasticMapping siteMapping;

    // the rate matrix Q and its eigen decomposition set last
    private double[] rateMatrix;
    private EigenDecomposition eigenDecomposition;

    // reusable transition probabilities exp(s Q t) of the branch of the current call
    private final double[] transitionProbabilities;

    public ExactBranchHistories(int numStates, int numSites) {
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
     * the start height down to the end height, under R = s Q for the given rate scale s. The
     * differing sites map every site whose end states differ to its states at the start and at
     * the end. At every other site, both end states equal the state returned by the given
     * lookup. The mutations are sorted by descending height.
     */
    public List<Mutation> sampleBranchHistory(int nodeNr, double branchStartHeight, double branchEndHeight, double rateScale,
                                              SiteChanges differingSites, IntUnaryOperator agreeingStates) {
        List<Mutation> branchMutations = new ArrayList<>();
        if (branchStartHeight <= branchEndHeight) {
            // a branch of length zero cannot carry mutations, and its end states are equal
            return branchMutations;
        }

        this.computeTransitionProbabilities(branchStartHeight - branchEndHeight, rateScale);

        for (int site = 0; site < this.numSites; site++) {
            int slot = differingSites.getSlot(site);
            int startState = slot >= 0 ? differingSites.getStartState(slot) : agreeingStates.applyAsInt(site);
            int endState = slot >= 0 ? differingSites.getEndState(slot) : startState;

            branchMutations.addAll(this.siteMapping.sampleSiteHistory(
                    nodeNr, site, branchStartHeight, branchEndHeight, startState, endState, rateScale,
                    this.transitionProbabilities[startState * this.numStates + endState]
            ));
        }

        branchMutations.sort(Comparator.comparingDouble(Mutation::time).reversed());
        return branchMutations;
    }

    /**
     * Computes the log probability that sampleBranchHistory proposes the given history on a
     * branch, given the states at its start, which the lookup returns for every site. The
     * states at the end follow from the start states and the mutations. Per site, this is the
     * density of its path under R = s Q divided by the probability of its end state.
     */
    public double computeLogBranchHistoryDensity(List<Mutation> branchMutations, double branchStartHeight, double branchEndHeight,
                                                 double rateScale, IntUnaryOperator startStates) {
        double duration = branchStartHeight - branchEndHeight;
        if (duration <= 0.0) {
            return 0.0;
        }

        this.computeTransitionProbabilities(duration, rateScale);

        // every site without mutations contributes the density of staying in its state, divided by the probability of keeping it

        double[] logEmptySiteDensities = new double[this.numStates];
        for (int state = 0; state < this.numStates; state++) {
            int diagonal = state * this.numStates + state;
            logEmptySiteDensities[state] = rateScale * this.rateMatrix[diagonal] * duration - Math.log(this.transitionProbabilities[diagonal]);
        }

        double logDensity = 0.0;
        for (int site = 0; site < this.numSites; site++) {
            logDensity += logEmptySiteDensities[startStates.applyAsInt(site)];
        }

        // the sites with mutations replace that contribution by the density of their path

        List<Mutation> sortedMutations = new ArrayList<>(branchMutations);
        sortedMutations.sort(Comparator.comparingInt(Mutation::site).thenComparing(Comparator.comparingDouble(Mutation::time).reversed()));

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

            logDensity -= logEmptySiteDensities[startState];
            logDensity += this.siteMapping.computeLogSiteHistoryDensity(
                    siteMutations, branchStartHeight, branchEndHeight, startState, endState, rateScale,
                    this.transitionProbabilities[startState * this.numStates + endState]
            );

            start = end;
        }

        return logDensity;
    }

    private void computeTransitionProbabilities(double duration, double rateScale) {
        StochasticMapping.computeTransitionProbabilities(this.eigenDecomposition, rateScale * duration, this.transitionProbabilities);
    }

}
