package emat.helper;

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
public class UniformisedStochasticMapping implements StochasticMapping {

    // safeguard against numerical problems when sampling the number of uniformised jumps
    private static final int MAX_JUMPS = 10000;

    final int numStates;

    // the rate matrix Q set last and its uniformised chain, which all branches share
    double[] rateMatrix;
    UniformisedChain uniformisedChain;

    public UniformisedStochasticMapping(int numStates) {
        this.numStates = numStates;
    }

    /* Single Sites */

    @Override
    public void setRateMatrix(double[] rateMatrix) {
        if (!Arrays.equals(rateMatrix, this.rateMatrix)) {
            this.rateMatrix = rateMatrix.clone();
            this.uniformisedChain = this.createUniformisedChain(this.rateMatrix);
        }
    }

    /** Samples the history exactly from its conditional distribution under R = s Q. */
    @Override
    public List<Mutation> sampleSiteHistory(int nodeNr, int site, double branchStartHeight, double branchEndHeight,
                                            int startState, int endState, double rateScale, double endProbability) {
        return this.sampleBranchHistory(
                nodeNr, site, branchStartHeight, branchEndHeight,
                startState, endState, this.uniformisedChain, rateScale, endProbability
        );
    }

    /**
     * Computes the density of the path under R = s Q, i.e. the rates of its jumps times the
     * probability of no further jumps, divided by the end probability.
     */
    @Override
    public double computeLogSiteHistoryDensity(List<Mutation> siteMutations, double branchStartHeight, double branchEndHeight,
                                               int startState, int endState, double rateScale, double endProbability) {
        double logDensity = 0.0;
        double previousHeight = branchStartHeight;
        int state = startState;

        for (Mutation mutation : siteMutations) {
            logDensity += rateScale * this.rateMatrix[state * this.numStates + state] * (previousHeight - mutation.time());
            logDensity += Math.log(rateScale * this.rateMatrix[mutation.oldState() * this.numStates + mutation.newState()]);

            state = mutation.newState();
            previousHeight = mutation.time();
        }

        logDensity += rateScale * this.rateMatrix[state * this.numStates + state] * (previousHeight - branchEndHeight);

        return logDensity - Math.log(endProbability);
    }

    /* Branch Histories */

    /**
     * Samples the history of the given site on the branch above the given node like above,
     * but with the rate matrix R = s Q given by its scale s and the uniformised chain of Q,
     * which can be shared by all branches. The number of jumps is drawn conditional on the
     * end state, then the jump times uniformly, and then the states of the jumps one by one.
     */
    public List<Mutation> sampleBranchHistory(int nodeNr, int site, double branchStartHeight, double branchEndHeight,
                                              int startState, int endState, UniformisedChain chain, double rateScale,
                                              double endProbability) {
        // evolution runs forwards in time from the branch start to the branch end

        double duration = branchStartHeight - branchEndHeight;

        List<Mutation> siteMutations = new ArrayList<>();
        if (duration <= 0.0) {
            // a branch of length zero cannot carry mutations, and its end states are equal
            return siteMutations;
        }

        double maxEscapeRate = rateScale * chain.uniformisationRate;
        if (maxEscapeRate == 0.0) {
            return siteMutations;
        }

        // sample the number of jumps n with P(n | start, end) ∝ Poisson(n; μ* t) (B^n)_{start, end}

        double threshold = Randomizer.nextDouble() * endProbability;

        double poissonProbability = Math.exp(-maxEscapeRate * duration);
        double cumulativeProbability = startState == endState ? poissonProbability : 0.0;
        int numJumps = 0;

        while (cumulativeProbability < threshold) {
            numJumps++;
            if (numJumps > MAX_JUMPS) {
                throw new RuntimeException("Could not sample the number of jumps on a branch of the site history.");
            }

            poissonProbability *= maxEscapeRate * duration / numJumps;
            cumulativeProbability += poissonProbability * chain.getJumpMatrixPower(numJumps)[startState * this.numStates + endState];
        }

        if (numJumps == 0) {
            return siteMutations;
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
            double[] remainingPower = chain.getJumpMatrixPower(numJumps - i - 1);
            for (int nextState = 0; nextState < this.numStates; nextState++) {
                weights[nextState] = chain.jumpMatrix[state * this.numStates + nextState]
                        * remainingPower[nextState * this.numStates + endState];
            }
            int nextState = StochasticMapping.sampleIndex(weights);

            if (nextState != state) {
                double height = branchStartHeight - jumpTimes[i];
                siteMutations.add(new Mutation(nodeNr, height, previousHeight, site, state, nextState));
                previousHeight = height;
                state = nextState;
            }
        }

        return siteMutations;
    }

    /** Creates the uniformised chain of the given rate matrix. */
    public UniformisedChain createUniformisedChain(double[] rateMatrix) {
        return new UniformisedChain(rateMatrix);
    }

    /* Uniformised Chain */

    /**
     * The uniformised chain of a rate matrix Q: the uniformisation rate μ* = max_i -Q_ii and
     * the jump matrix B = I + Q/μ*, together with the powers of B computed so far. Scaling Q
     * scales μ* but leaves B unchanged, so one chain serves every multiple of Q.
     */
    public class UniformisedChain {

        final double uniformisationRate;
        final double[] jumpMatrix;
        final List<double[]> jumpMatrixPowers = new ArrayList<>();

        UniformisedChain(double[] rateMatrix) {
            double maxEscapeRate = 0.0;
            for (int state = 0; state < UniformisedStochasticMapping.this.numStates; state++) {
                maxEscapeRate = Math.max(maxEscapeRate, -rateMatrix[state * UniformisedStochasticMapping.this.numStates + state]);
            }
            this.uniformisationRate = maxEscapeRate;

            this.jumpMatrix = this.getIdentityMatrix();
            if (maxEscapeRate > 0.0) {
                for (int i = 0; i < this.jumpMatrix.length; i++) {
                    this.jumpMatrix[i] += rateMatrix[i] / maxEscapeRate;
                }
            }

            this.jumpMatrixPowers.add(this.getIdentityMatrix());
        }

        /** Returns B^n, computing and caching the missing powers. */
        double[] getJumpMatrixPower(int n) {
            while (this.jumpMatrixPowers.size() <= n) {
                double[] previousPower = this.jumpMatrixPowers.get(this.jumpMatrixPowers.size() - 1);
                this.jumpMatrixPowers.add(this.multiplyMatrices(previousPower, this.jumpMatrix));
            }
            return this.jumpMatrixPowers.get(n);
        }

        private double[] getIdentityMatrix() {
            int numStates = UniformisedStochasticMapping.this.numStates;
            double[] identity = new double[numStates * numStates];
            for (int state = 0; state < numStates; state++) {
                identity[state * numStates + state] = 1.0;
            }
            return identity;
        }

        private double[] multiplyMatrices(double[] left, double[] right) {
            int numStates = UniformisedStochasticMapping.this.numStates;
            double[] product = new double[numStates * numStates];
            for (int i = 0; i < numStates; i++) {
                for (int k = 0; k < numStates; k++) {
                    double value = left[i * numStates + k];
                    if (value == 0.0) {
                        continue;
                    }
                    for (int j = 0; j < numStates; j++) {
                        product[i * numStates + j] += value * right[k * numStates + j];
                    }
                }
            }
            return product;
        }

    }

}
