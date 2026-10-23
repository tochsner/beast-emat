package emat.stochasticmapping;

import beast.base.evolution.tree.Node;
import beast.base.util.Randomizer;
import emat.prior.GeneticPrior;
import emat.state.Mutation;

import java.util.List;

/**
 * Simulates the substitution process of the genetic prior forwards along single branches,
 * one site at a time. Given the state at the top of a branch, the history of a site whose
 * tips below are all missing does not depend on the data, so such a simulation is a draw
 * from its exact conditional distribution.
 */
public final class ForwardSiteSimulator {

    private final GeneticPrior geneticPrior;
    private final int numStates;

    // the model of the current proposal: Q, its largest escape rate and the site rate
    private double[] rateMatrix;
    private double maxEscapeRate;
    private double siteRate;

    public ForwardSiteSimulator(GeneticPrior geneticPrior, int numStates) {
        this.geneticPrior = geneticPrior;
        this.numStates = numStates;
    }

    /** Takes Q and the site rate from the genetic prior, which has to be done once per proposal. */
    public void updateModel(Node root) {
        this.rateMatrix = this.geneticPrior.computeRateMatrix();
        this.maxEscapeRate = 0.0;
        for (int state = 0; state < this.numStates; state++) {
            this.maxEscapeRate = Math.max(this.maxEscapeRate, -this.rateMatrix[state * this.numStates + state]);
        }
        this.siteRate = this.geneticPrior.siteModel.getRateForCategory(0, root);
    }

    /** Returns the scale of Q on the branch above the given node: the branch rate times the site rate. */
    public double getRateScale(Node node) {
        return this.geneticPrior.branchRateModel.getRateForBranch(node) * this.siteRate;
    }

    /**
     * Simulates a single site from the given start state at the top of the branch above the
     * given node down to the node, adds the mutations to the given list and returns the
     * state at the node. The waiting time until the next mutation is exponential with the
     * escape rate of the current state, and the new state is drawn proportional to the rates
     * of leaving the current state towards it.
     */
    public int simulateBranch(Node node, int site, int startState, double rateScale, List<Mutation> branchMutations) {
        double endHeight = node.getHeight();
        double previousHeight = node.getParent().getHeight();
        double timeOfPreviousMutation = Mutation.BRANCH_START;
        int state = startState;

        while (true) {
            double escapeRate = -this.rateMatrix[state * this.numStates + state] * rateScale;
            double height = previousHeight - Randomizer.nextExponential(escapeRate);
            if (!(height > endHeight)) {
                return state;
            }

            int newState = this.sampleNewState(state);
            branchMutations.add(new Mutation(node.getNr(), height, timeOfPreviousMutation, site, state, newState));

            previousHeight = height;
            timeOfPreviousMutation = height;
            state = newState;
        }
    }

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

    /** Draws the state that the given state mutates to, proportional to the off-diagonal rates of Q. */
    private int sampleNewState(int state) {
        double threshold = Randomizer.nextDouble() * -this.rateMatrix[state * this.numStates + state];

        int lastCandidate = -1;
        double cumulativeRate = 0.0;
        for (int newState = 0; newState < this.numStates; newState++) {
            if (newState == state) {
                continue;
            }

            lastCandidate = newState;
            cumulativeRate += this.rateMatrix[state * this.numStates + newState];
            if (threshold < cumulativeRate) {
                return newState;
            }
        }

        // only reached by rounding when the threshold lies at the very end
        return lastCandidate;
    }

}
