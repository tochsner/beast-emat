package emat.stochasticmapping;

import beast.base.util.Randomizer;
import emat.state.Mutation;

import java.util.ArrayList;
import java.util.Arrays;
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

    private final int numStates;

    // the rate matrix Q set last, its uniformisation rate μ* = max_i -Q_ii and its jump matrix
    // B = I + Q/μ*, together with the powers of B computed so far. Scaling Q scales μ* but
    // leaves B unchanged, so they serve every branch.
    private double[] rateMatrix;
    private double uniformisationRate;
    private double[] jumpMatrix;
    private final List<double[]> jumpMatrixPowers = new ArrayList<>();

    public UniformisedStochasticMapping(int numStates) {
        this.numStates = numStates;
    }

    @Override
    public void setRateMatrix(double[] rateMatrix) {
        if (Arrays.equals(rateMatrix, this.rateMatrix)) {
            return;
        }
        this.rateMatrix = rateMatrix.clone();

        this.uniformisationRate = 0.0;
        for (int state = 0; state < this.numStates; state++) {
            this.uniformisationRate = Math.max(this.uniformisationRate, -rateMatrix[state * this.numStates + state]);
        }

        this.jumpMatrix = this.getIdentityMatrix();
        if (this.uniformisationRate > 0.0) {
            for (int i = 0; i < this.jumpMatrix.length; i++) {
                this.jumpMatrix[i] += rateMatrix[i] / this.uniformisationRate;
            }
        }

        this.jumpMatrixPowers.clear();
        this.jumpMatrixPowers.add(this.getIdentityMatrix());
    }

    /**
     * Samples the history exactly from its conditional distribution under R = s Q. The number
     * of jumps is drawn conditional on the end state, then the jump times uniformly, and then
     * the states of the jumps one by one.
     */
    @Override
    public List<Mutation> sampleSiteHistory(int nodeNr, int site, double branchStartHeight, double branchEndHeight,
                                            int startState, int endState, double rateScale, double endProbability) {
        // evolution runs forwards in time from the branch start to the branch end

        double duration = branchStartHeight - branchEndHeight;
        double maxEscapeRate = rateScale * this.uniformisationRate;

        List<Mutation> siteMutations = new ArrayList<>();
        if (duration <= 0.0 || maxEscapeRate == 0.0) {
            // without time or rate, no mutations can occur and the end states are equal
            return siteMutations;
        }

        int numJumps = this.sampleNumJumps(startState, endState, maxEscapeRate * duration, endProbability);
        if (numJumps == 0) {
            return siteMutations;
        }

        return this.sampleJumps(nodeNr, site, branchStartHeight, branchEndHeight, startState, endState, numJumps);
    }

    /**
     * Samples the uniformised jump chain of a site from its exact conditional distribution
     * under R = s Q given the states at both ends, like sampleSiteHistory, but without the jump
     * times. The returned array holds the start state followed by the state after every
     * uniformised jump, including the jumps that stay in the same state. Given its length, the
     * times of its jumps are independent and uniform on the branch.
     */
    public int[] sampleJumpChain(int startState, int endState, double duration, double rateScale, double endProbability) {
        double maxEscapeRate = rateScale * this.uniformisationRate;
        if (duration <= 0.0 || maxEscapeRate == 0.0) {
            return new int[]{startState};
        }

        int numJumps = this.sampleNumJumps(startState, endState, maxEscapeRate * duration, endProbability);
        return this.sampleJumpStates(startState, endState, numJumps);
    }

    /**
     * Samples the number of uniformised jumps n with P(n | start, end) ∝ Poisson(n; μ* t) (B^n)_{start, end},
     * where the end probability is the normalising constant.
     */
    private int sampleNumJumps(int startState, int endState, double expectedJumps, double endProbability) {
        double threshold = Randomizer.nextDouble() * endProbability;

        double poissonProbability = Math.exp(-expectedJumps);
        double cumulativeProbability = startState == endState ? poissonProbability : 0.0;
        int numJumps = 0;

        while (cumulativeProbability < threshold) {
            numJumps++;
            if (numJumps > MAX_JUMPS) {
                throw new RuntimeException("Could not sample the number of jumps on a branch of the site history.");
            }

            poissonProbability *= expectedJumps / numJumps;
            cumulativeProbability += poissonProbability * this.getJumpMatrixPower(numJumps)[startState * this.numStates + endState];
        }

        return numJumps;
    }

    /** Returns the uniformisation rate μ* of Q, before the scaling of a branch. */
    @Override
    public double getJumpRate() {
        return this.uniformisationRate;
    }

    /**
     * Samples the history on a branch whose ends have the same state, given a number of
     * uniformised jumps drawn from the unconditioned Poisson(μ* s t) distribution of the
     * branch. The draw is accepted with probability (B^n)_{state, state}, which turns the
     * Poisson distribution into the distribution of n conditional on the end state, and the
     * jumps are then sampled as in sampleSiteHistory. Returns null if the draw is rejected,
     * in which case sampleSiteHistory provides an exact draw instead.
     */
    @Override
    public List<Mutation> sampleSameStateHistory(int nodeNr, int site, double branchStartHeight, double branchEndHeight,
                                                 int state, int numJumps) {
        if (numJumps > MAX_JUMPS) {
            return null;
        }

        double acceptanceProbability = this.getJumpMatrixPower(numJumps)[state * this.numStates + state];
        if (Randomizer.nextDouble() >= acceptanceProbability) {
            return null;
        }

        return this.sampleJumps(nodeNr, site, branchStartHeight, branchEndHeight, state, state, numJumps);
    }

    /**
     * Samples the uniformised jump chain of a site exactly from its conditional distribution
     * under R = s Q, given that it starts and ends in the given state and has at least one
     * mutation. The non-empty probability is the probability of returning to the state with at
     * least one mutation, P_ii(t) - exp(R_ii t). A uniformised path is empty if all its jumps
     * stay in the state, which has probability B_ii^n for n jumps, so the number of jumps is
     * drawn with P(n) ∝ Poisson(n; μ* t) ((B^n)_ii - B_ii^n), and the states of the jumps
     * exclude the path that never leaves the state. The returned array holds the start state
     * followed by the state after every uniformised jump, including the jumps that stay in the
     * same state. Given its length, the times of its jumps are independent and uniform on the
     * branch.
     */
    public int[] sampleNonEmptySameStateChain(int state, double duration, double rateScale, double nonEmptyProbability) {
        int numJumps = this.sampleNonEmptyNumJumps(state, rateScale * this.uniformisationRate * duration, nonEmptyProbability);
        return this.sampleNonEmptyJumpStates(state, numJumps);
    }

    /**
     * Samples the number of uniformised jumps of a non-empty history that starts and ends in the
     * given state, with P(n) ∝ Poisson(n; μ* t) ((B^n)_ii - B_ii^n). At least two jumps are
     * needed to leave and return.
     */
    private int sampleNonEmptyNumJumps(int state, double expectedJumps, double nonEmptyProbability) {
        int diagonal = state * this.numStates + state;
        double selfJumpProbability = this.jumpMatrix[diagonal];

        double threshold = Randomizer.nextDouble() * nonEmptyProbability;
        double poissonProbability = Math.exp(-expectedJumps);
        double cumulativeProbability = 0.0;
        int numJumps = 0;

        while (cumulativeProbability < threshold || numJumps < 2) {
            numJumps++;
            if (numJumps > MAX_JUMPS) {
                throw new RuntimeException("Could not sample the number of jumps of a non-empty site history.");
            }

            poissonProbability *= expectedJumps / numJumps;
            cumulativeProbability += poissonProbability
                    * Math.max(this.getJumpMatrixPower(numJumps)[diagonal] - Math.pow(selfJumpProbability, numJumps), 0.0);
        }

        return numJumps;
    }

    /**
     * Samples the state after every one of the given number of uniformised jumps of a path that
     * starts and ends in the given state and leaves it at least once. Until the first mutation,
     * the path is still in the state and must not stay there. The returned array holds the
     * start state followed by the state after every jump.
     */
    private int[] sampleNonEmptyJumpStates(int state, int numJumps) {
        int diagonal = state * this.numStates + state;
        double selfJumpProbability = this.jumpMatrix[diagonal];

        int[] states = new int[numJumps + 1];
        states[0] = state;
        boolean hasMutated = false;
        double[] weights = new double[this.numStates];

        for (int i = 0; i < numJumps; i++) {
            int remainingJumps = numJumps - i - 1;
            double[] remainingPower = this.getJumpMatrixPower(remainingJumps);
            for (int nextState = 0; nextState < this.numStates; nextState++) {
                double reachProbability = remainingPower[nextState * this.numStates + state];
                if (!hasMutated && nextState == state) {
                    reachProbability = Math.max(reachProbability - Math.pow(selfJumpProbability, remainingJumps), 0.0);
                }
                weights[nextState] = this.jumpMatrix[states[i] * this.numStates + nextState] * reachProbability;
            }
            states[i + 1] = StochasticMapping.sampleIndex(weights);
            hasMutated |= states[i + 1] != states[i];
        }

        return states;
    }

    /**
     * Samples the times of the given number of uniformised jumps uniformly on the branch and
     * then the state after every jump, conditional on reaching the end state, and returns the
     * real jumps as mutations sorted by descending height.
     */
    private List<Mutation> sampleJumps(int nodeNr, int site, double branchStartHeight, double branchEndHeight,
                                       int startState, int endState, int numJumps) {
        double duration = branchStartHeight - branchEndHeight;

        // sample the jump times forwards from the branch start

        double[] jumpTimes = new double[numJumps];
        for (int i = 0; i < numJumps; i++) {
            jumpTimes[i] = Randomizer.nextDouble() * duration;
        }
        Arrays.sort(jumpTimes);

        // sample the state after every jump, conditional on reaching the end state, and keep the real jumps

        int[] states = this.sampleJumpStates(startState, endState, numJumps);

        List<Mutation> siteMutations = new ArrayList<>();
        double previousHeight = Mutation.BRANCH_START;
        for (int i = 0; i < numJumps; i++) {
            if (states[i + 1] != states[i]) {
                double height = branchStartHeight - jumpTimes[i];
                siteMutations.add(new Mutation(nodeNr, height, previousHeight, site, states[i], states[i + 1]));
                previousHeight = height;
            }
        }

        return siteMutations;
    }

    /**
     * Samples the state after every one of the given number of uniformised jumps, conditional on
     * reaching the end state. The returned array holds the start state followed by the state
     * after every jump.
     */
    private int[] sampleJumpStates(int startState, int endState, int numJumps) {
        int[] states = new int[numJumps + 1];
        states[0] = startState;
        double[] weights = new double[this.numStates];

        for (int i = 0; i < numJumps; i++) {
            double[] remainingPower = this.getJumpMatrixPower(numJumps - i - 1);
            for (int nextState = 0; nextState < this.numStates; nextState++) {
                weights[nextState] = this.jumpMatrix[states[i] * this.numStates + nextState]
                        * remainingPower[nextState * this.numStates + endState];
            }
            states[i + 1] = StochasticMapping.sampleIndex(weights);
        }

        return states;
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

    /* Jump Matrix */

    /** Returns B^n, computing and caching the missing powers. */
    private double[] getJumpMatrixPower(int n) {
        while (this.jumpMatrixPowers.size() <= n) {
            double[] previousPower = this.jumpMatrixPowers.getLast();
            this.jumpMatrixPowers.add(this.multiplyMatrices(previousPower, this.jumpMatrix));
        }
        return this.jumpMatrixPowers.get(n);
    }

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
