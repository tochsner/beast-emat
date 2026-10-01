package emat.helper;

import beast.base.evolution.substitutionmodel.EigenDecomposition;
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

    /* Helpers */

    /**
     * Computes the transition probabilities P = exp(Q t) from the eigen decomposition of Q,
     * where the duration t already includes every rate scaling of the branch.
     */
    static double[] computeTransitionProbabilities(EigenDecomposition eigenDecomposition, double duration) {
        double[] eigenVectors = eigenDecomposition.getEigenVectors();
        double[] inverseEigenVectors = eigenDecomposition.getInverseEigenVectors();
        double[] eigenValues = eigenDecomposition.getEigenValues();
        int numStates = eigenValues.length;

        double[] expEigenValues = new double[numStates];
        for (int k = 0; k < numStates; k++) {
            expEigenValues[k] = Math.exp(eigenValues[k] * duration);
        }

        double[] probabilities = new double[numStates * numStates];
        for (int from = 0; from < numStates; from++) {
            for (int to = 0; to < numStates; to++) {
                double probability = 0.0;
                for (int k = 0; k < numStates; k++) {
                    probability += eigenVectors[from * numStates + k] * expEigenValues[k]
                            * inverseEigenVectors[k * numStates + to];
                }

                // clamp tiny negative values caused by rounding
                probabilities[from * numStates + to] = Math.max(probability, 0.0);
            }
        }

        return probabilities;
    }

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
