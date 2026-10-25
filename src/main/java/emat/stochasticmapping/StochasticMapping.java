package emat.stochasticmapping;

import beast.base.util.Randomizer;
import emat.state.Mutation;

import java.util.List;

/**
 * Samples the history of a single site on a branch given the states at both of its ends
 * (stochastic mapping), and computes the probability of proposing a given history.
 */
public interface StochasticMapping {

    /**
     * Sets the rate matrix Q, stored row by row, which every following call scales by the
     * rate scale of its branch.
     */
    void setRateMatrix(double[] rateMatrix);

    /**
     * Samples the history of the given site on the branch above the given node, which runs
     * from the start height down to the end height, given the states at both ends. The end
     * probability is the transition probability exp(s Q t) from the start to the end state.
     * The mutations are sorted by descending height.
     */
    List<Mutation> sampleSiteHistory(int nodeNr, int site, double branchStartHeight, double branchEndHeight,
                                     int startState, int endState, double rateScale, double endProbability);

    /**
     * Computes the log probability that sampleSiteHistory proposes the given history of a
     * single site on a branch, given the states at both ends and the same arguments.
     */
    double computeLogSiteHistoryDensity(List<Mutation> siteMutations, double branchStartHeight, double branchEndHeight,
                                        int startState, int endState, double rateScale, double endProbability);

    /**
     * Returns the rate λ of the jumps that sampleSameStateHistory expects, before the scaling
     * of a branch: on a branch with rate scale s and duration t, their number is drawn from
     * the unconditioned Poisson(λ s t) distribution.
     */
    double getJumpRate();

    /**
     * Samples the history on a branch whose ends have the same state, given a number of jumps
     * drawn from the unconditioned Poisson(λ s t) distribution of the branch. The draw is
     * accepted with the probability that the jumps return to the state, which turns the
     * Poisson distribution into the distribution conditional on the end state. Returns null
     * if the draw is rejected, in which case sampleSiteHistory provides an exact draw instead.
     */
    List<Mutation> sampleSameStateHistory(int nodeNr, int site, double branchStartHeight, double branchEndHeight,
                                          int state, int numJumps);

    /* Helpers */

    /** Samples an index with probability proportional to the given non-negative weights. */
    static int sampleIndex(double[] weights) {
        double totalWeight = 0.0;
        for (double weight : weights) {
            totalWeight += weight;
        }

        double threshold = Randomizer.nextDouble() * totalWeight;
        double cumulativeWeight = 0.0;
        for (int i = 0; i < weights.length; i++) {
            cumulativeWeight += weights[i];
            if (threshold < cumulativeWeight) {
                return i;
            }
        }

        // only reached through rounding, so return the last index with positive weight
        for (int i = weights.length - 1; i >= 0; i--) {
            if (weights[i] > 0.0) {
                return i;
            }
        }
        throw new RuntimeException("Cannot sample from weights that are all zero.");
    }

}
