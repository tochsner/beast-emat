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
            cumulativeProbability += poissonProbability * this.getJumpMatrixPower(numJumps)[startState * this.numStates + endState];
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
            double[] remainingPower = this.getJumpMatrixPower(numJumps - i - 1);
            for (int nextState = 0; nextState < this.numStates; nextState++) {
                weights[nextState] = this.jumpMatrix[state * this.numStates + nextState]
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
