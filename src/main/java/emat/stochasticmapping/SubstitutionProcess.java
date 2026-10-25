package emat.stochasticmapping;

import beast.base.evolution.substitutionmodel.EigenDecomposition;
import beast.base.evolution.tree.Node;
import beast.base.evolution.tree.TreeInterface;
import beast.base.util.Randomizer;
import emat.prior.GeneticPrior;

import java.util.Arrays;

/**
 * The substitution process of the genetic prior as of the last update: the rate matrix Q, its
 * eigen decomposition, the site rate and the root frequencies. The samplers of mutation
 * histories read the model from here, so that it is fetched once per proposal. On the branch
 * above a node, the process runs with the rate matrix s Q, where the rate scale s is the
 * branch rate times the site rate.
 */
public final class SubstitutionProcess {

    private final GeneticPrior geneticPrior;
    private final TreeInterface tree;
    private final int numStates;

    // the model as of the last update; Q is only replaced by a new array if its values change
    private double[] rateMatrix;
    private EigenDecomposition eigenDecomposition;
    private double siteRate;
    private double[] frequencies;
    private double maxEscapeRate;

    public SubstitutionProcess(GeneticPrior geneticPrior, TreeInterface tree, int numStates) {
        this.geneticPrior = geneticPrior;
        this.tree = tree;
        this.numStates = numStates;
    }

    /**
     * Takes the current model from the genetic prior, which has to be done once per proposal.
     * The substitution model may overwrite its eigen decomposition objects in place, so they
     * are fetched again every time. Q keeps its array while its values stay the same, so
     * users can detect a change by comparing the array they saw last.
     */
    public void update() {
        Node root = this.tree.getRoot();

        this.eigenDecomposition = this.geneticPrior.substitutionModel.getEigenDecomposition(root);
        double[] newRateMatrix = this.geneticPrior.computeRateMatrix();
        if (!Arrays.equals(newRateMatrix, this.rateMatrix)) {
            this.rateMatrix = newRateMatrix;
            this.maxEscapeRate = 0.0;
            for (int state = 0; state < this.numStates; state++) {
                this.maxEscapeRate = Math.max(this.maxEscapeRate, -this.rateMatrix[state * this.numStates + state]);
            }
        }

        this.siteRate = this.geneticPrior.siteModel.getRateForCategory(0, root);
        this.frequencies = this.geneticPrior.substitutionModel.getFrequencies();
    }

    public int getNumStates() {
        return this.numStates;
    }

    /** Returns Q, stored row by row, before the scaling of a branch. */
    public double[] getRateMatrix() {
        return this.rateMatrix;
    }

    public double getSiteRate() {
        return this.siteRate;
    }

    public double[] getFrequencies() {
        return this.frequencies;
    }

    /** Returns the scale of Q on the branch above the given node: the branch rate times the site rate. */
    public double getRateScale(Node node) {
        return this.geneticPrior.branchRateModel.getRateForBranch(node) * this.siteRate;
    }

    /**
     * Computes the transition probabilities P = exp(Q t) into the given array of length n²,
     * where the duration t already includes every rate scaling of the branch.
     */
    public void computeTransitionProbabilities(double duration, double[] probabilities) {
        computeTransitionProbabilities(this.eigenDecomposition, duration, probabilities);
    }

    /* Uniformisation */

    /**
     * Returns the rate of the uniformised process: the largest escape rate of Q, which bounds
     * the escape rate of every state. Events of a Poisson process at this rate, of which each
     * is a jump of the chain with probability escape rate over this rate and otherwise
     * virtual, yield the same process as Q.
     */
    public double getMaxEscapeRate() {
        return this.maxEscapeRate;
    }

    /**
     * Resolves an event of the uniformised process in the given state: draws a threshold
     * uniformly up to the rate of the uniformised process, and returns the state whose
     * off-diagonal rate covers it, or the given state if the threshold lies beyond the escape
     * rate and the event is virtual.
     */
    public int sampleUniformisedJump(int state) {
        double threshold = Randomizer.nextDouble() * this.maxEscapeRate;

        double cumulativeRate = 0.0;
        for (int newState = 0; newState < this.numStates; newState++) {
            if (newState == state) {
                continue;
            }

            cumulativeRate += this.rateMatrix[state * this.numStates + newState];
            if (threshold < cumulativeRate) {
                return newState;
            }
        }

        return state;
    }

    /* Transition Probabilities */

    /**
     * Computes the transition probabilities P = exp(Q t) from the eigen decomposition of Q,
     * where the duration t already includes every rate scaling of the branch.
     */
    public static double[] computeTransitionProbabilities(EigenDecomposition eigenDecomposition, double duration) {
        int numStates = eigenDecomposition.getEigenValues().length;
        double[] probabilities = new double[numStates * numStates];
        computeTransitionProbabilities(eigenDecomposition, duration, probabilities);
        return probabilities;
    }

    /**
     * Computes the transition probabilities P = exp(Q t) like computeTransitionProbabilities,
     * but writes them into the given array of length n², which avoids an allocation per branch.
     */
    public static void computeTransitionProbabilities(EigenDecomposition eigenDecomposition, double duration, double[] probabilities) {
        double[] eigenVectors = eigenDecomposition.getEigenVectors();
        double[] inverseEigenVectors = eigenDecomposition.getInverseEigenVectors();
        double[] eigenValues = eigenDecomposition.getEigenValues();
        int numStates = eigenValues.length;

        // add up the contribution of one eigenvalue at a time, which needs only a single exponential at hand
        Arrays.fill(probabilities, 0.0);
        for (int k = 0; k < numStates; k++) {
            // a zero eigenvalue, which every rate matrix has, needs no exponential
            double expEigenValue = eigenValues[k] == 0.0 ? 1.0 : Math.exp(eigenValues[k] * duration);

            for (int from = 0; from < numStates; from++) {
                double scaledEigenVector = eigenVectors[from * numStates + k] * expEigenValue;
                for (int to = 0; to < numStates; to++) {
                    probabilities[from * numStates + to] += scaledEigenVector * inverseEigenVectors[k * numStates + to];
                }
            }
        }

        // clamp tiny negative values caused by rounding
        for (int i = 0; i < probabilities.length; i++) {
            probabilities[i] = Math.max(probabilities[i], 0.0);
        }
    }

}
