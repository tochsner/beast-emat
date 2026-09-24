package emat.helper;

import beast.base.evolution.substitutionmodel.EigenDecomposition;
import beast.base.util.Randomizer;
import emat.state.Mutation;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.Comparator;
import java.util.List;

/**
 * Samples the history of a single site on a branch given the states at both of its ends,
 * using uniformisation: with a rate μ* at least as large as every escape rate, the path is
 * a Poisson number of jumps of the chain B = I + R/μ*, where R is the rate matrix of the
 * branch, and jumps that stay in the same state are dropped.
 */
public class StochasticMapping {

    // safeguard against numerical problems when sampling the number of uniformised jumps
    private static final int MAX_JUMPS = 10000;

    final int numStates;

    public StochasticMapping(int numStates) {
        this.numStates = numStates;
    }

    /**
     * Computes the transition probabilities P = exp(Q t) from the eigen decomposition of Q,
     * where the duration t already includes every rate scaling of the branch.
     */
    public double[] computeTransitionProbabilities(EigenDecomposition eigenDecomposition, double duration) {
        double[] eigenVectors = eigenDecomposition.getEigenVectors();
        double[] inverseEigenVectors = eigenDecomposition.getInverseEigenVectors();
        double[] eigenValues = eigenDecomposition.getEigenValues();

        double[] expEigenValues = new double[this.numStates];
        for (int k = 0; k < this.numStates; k++) {
            expEigenValues[k] = Math.exp(eigenValues[k] * duration);
        }

        double[] probabilities = new double[this.numStates * this.numStates];
        for (int from = 0; from < this.numStates; from++) {
            for (int to = 0; to < this.numStates; to++) {
                double probability = 0.0;
                for (int k = 0; k < this.numStates; k++) {
                    probability += eigenVectors[from * this.numStates + k] * expEigenValues[k]
                            * inverseEigenVectors[k * this.numStates + to];
                }

                // clamp tiny negative values caused by rounding
                probabilities[from * this.numStates + to] = Math.max(probability, 0.0);
            }
        }

        return probabilities;
    }

    /**
     * Samples the history of every site on the branch above the given node, given the full
     * sequences at both ends, by mapping each site independently. The rate matrix R and the
     * transition probabilities exp(R t) must already include every rate scaling of the
     * branch. The mutations are sorted by descending height.
     */
    public List<Mutation> sampleSequenceHistory(int nodeNr, double branchStartHeight, double branchEndHeight,
                                                int[] startSequence, int[] endSequence,
                                                double[] rateMatrix, double[] transitionProbabilities) {
        List<Mutation> branchMutations = new ArrayList<>();

        for (int site = 0; site < startSequence.length; site++) {
            int startState = startSequence[site];
            int endState = endSequence[site];
            double endProbability = transitionProbabilities[startState * this.numStates + endState];

            branchMutations.addAll(this.sampleBranchHistory(
                    nodeNr, site, branchStartHeight, branchEndHeight,
                    startState, endState, rateMatrix, endProbability
            ));
        }

        branchMutations.sort(Comparator.comparingDouble(Mutation::time).reversed());
        return branchMutations;
    }

    /**
     * Computes the log probability that sampleSequenceHistory proposes the given history on a
     * branch, given the full sequences at both ends. Per site, this is the density of the
     * path under R, i.e. the rates of its jumps times the probability of no further jumps,
     * divided by the transition probability from the start to the end state.
     */
    public double computeLogSequenceHistoryDensity(List<Mutation> branchMutations, double branchStartHeight, double branchEndHeight,
                                                   int[] startSequence, int[] endSequence,
                                                   double[] rateMatrix, double[] transitionProbabilities) {
        double escapeRate = 0.0;
        for (int state : startSequence) {
            escapeRate -= rateMatrix[state * this.numStates + state];
        }

        // add the path densities of all sites at once, as the escape rates add up

        double logDensity = 0.0;
        double previousHeight = branchStartHeight;

        for (Mutation mutation : branchMutations) {
            logDensity -= escapeRate * (previousHeight - mutation.time());
            logDensity += Math.log(rateMatrix[mutation.oldState() * this.numStates + mutation.newState()]);

            escapeRate += rateMatrix[mutation.oldState() * this.numStates + mutation.oldState()]
                    - rateMatrix[mutation.newState() * this.numStates + mutation.newState()];
            previousHeight = mutation.time();
        }

        logDensity -= escapeRate * (previousHeight - branchEndHeight);

        // normalise by the transition probabilities of all sites

        for (int site = 0; site < startSequence.length; site++) {
            logDensity -= Math.log(transitionProbabilities[startSequence[site] * this.numStates + endSequence[site]]);
        }

        return logDensity;
    }

    /**
     * Samples the history of the given site on the branch above the given node, which runs
     * from the start height down to the end height, given the states at both ends. The rate
     * matrix R must already include every rate scaling of the branch, and the end probability
     * is the transition probability exp(R t) from the start to the end state. The number of
     * jumps is drawn conditional on the end state, then the jump times uniformly, and then
     * the states of the jumps one by one. The mutations are sorted by descending height.
     */
    public List<Mutation> sampleBranchHistory(int nodeNr, int site, double branchStartHeight, double branchEndHeight,
                                              int startState, int endState, double[] rateMatrix, double endProbability) {
        // evolution runs forwards in time from the branch start to the branch end

        double duration = branchStartHeight - branchEndHeight;

        List<Mutation> siteMutations = new ArrayList<>();
        if (duration <= 0.0) {
            // a branch of length zero cannot carry mutations, and its end states are equal
            return siteMutations;
        }

        // set up the uniformised chain for the rate matrix R

        double maxEscapeRate = 0.0;
        for (int state = 0; state < this.numStates; state++) {
            maxEscapeRate = Math.max(maxEscapeRate, -rateMatrix[state * this.numStates + state]);
        }
        if (maxEscapeRate == 0.0) {
            return siteMutations;
        }

        double[] jumpMatrix = new double[this.numStates * this.numStates];
        for (int from = 0; from < this.numStates; from++) {
            for (int to = 0; to < this.numStates; to++) {
                double identity = from == to ? 1.0 : 0.0;
                jumpMatrix[from * this.numStates + to] = identity + rateMatrix[from * this.numStates + to] / maxEscapeRate;
            }
        }

        // sample the number of jumps n with P(n | start, end) ∝ Poisson(n; μ* t) (B^n)_{start, end}

        double threshold = Randomizer.nextDouble() * endProbability;

        List<double[]> jumpMatrixPowers = new ArrayList<>();
        jumpMatrixPowers.add(this.getIdentityMatrix());

        double poissonProbability = Math.exp(-maxEscapeRate * duration);
        double cumulativeProbability = poissonProbability * jumpMatrixPowers.get(0)[startState * this.numStates + endState];
        int numJumps = 0;

        while (cumulativeProbability < threshold) {
            numJumps++;
            if (numJumps > MAX_JUMPS) {
                throw new RuntimeException("Could not sample the number of jumps on a branch of the site history.");
            }

            poissonProbability *= maxEscapeRate * duration / numJumps;
            double[] jumpMatrixPower = this.multiplyMatrices(jumpMatrixPowers.get(numJumps - 1), jumpMatrix);
            jumpMatrixPowers.add(jumpMatrixPower);

            cumulativeProbability += poissonProbability * jumpMatrixPower[startState * this.numStates + endState];
        }

        // sample the jump times forwards from the branch start

        double[] jumpTimes = new double[numJumps];
        for (int i = 0; i < numJumps; i++) {
            jumpTimes[i] = Randomizer.nextDouble() * duration;
        }
        Arrays.sort(jumpTimes);

        // sample the state after every jump, conditional on reaching the end state, and keep the real jumps

        int state = startState;
        double previousHeight = branchStartHeight;
        double[] weights = new double[this.numStates];

        for (int i = 0; i < numJumps; i++) {
            double[] remainingPower = jumpMatrixPowers.get(numJumps - i - 1);
            for (int nextState = 0; nextState < this.numStates; nextState++) {
                weights[nextState] = jumpMatrix[state * this.numStates + nextState]
                        * remainingPower[nextState * this.numStates + endState];
            }
            int nextState = sampleIndex(weights);

            if (nextState != state) {
                double height = branchStartHeight - jumpTimes[i];
                siteMutations.add(new Mutation(nodeNr, height, previousHeight, site, state, nextState));
                previousHeight = height;
                state = nextState;
            }
        }

        return siteMutations;
    }

    /** Samples an index with probability proportional to the given non-negative weights. */
    public static int sampleIndex(double[] weights) {
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

    /* Helpers */

    private double[] getIdentityMatrix() {
        double[] identity = new double[this.numStates * this.numStates];
        for (int state = 0; state < this.numStates; state++) {
            identity[state * this.numStates + state] = 1.0;
        }
        return identity;
    }

    private double[] multiplyMatrices(double[] left, double[] right) {
        double[] product = new double[this.numStates * this.numStates];
        for (int i = 0; i < this.numStates; i++) {
            for (int k = 0; k < this.numStates; k++) {
                double value = left[i * this.numStates + k];
                if (value == 0.0) {
                    continue;
                }
                for (int j = 0; j < this.numStates; j++) {
                    product[i * this.numStates + j] += value * right[k * this.numStates + j];
                }
            }
        }
        return product;
    }

}
