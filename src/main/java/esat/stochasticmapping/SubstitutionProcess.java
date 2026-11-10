package esat.stochasticmapping;

import beast.base.evolution.tree.Node;
import esat.helper.TipCompatibility;
import esat.prior.GeneticPrior;

/**
 * The substitution process of the genetic prior as of the last update: the transition
 * probabilities of every branch, the root frequencies and the constraints of the data on the
 * tips. The samplers of node states read the model from here, so that they sample from the
 * same model the prior evaluates.
 */
public final class SubstitutionProcess {

    private final GeneticPrior geneticPrior;
    private final int numStates;

    private double[] frequencies;

    public SubstitutionProcess(GeneticPrior geneticPrior) {
        this.geneticPrior = geneticPrior;
        this.numStates = geneticPrior.getTipCompatibility().getNumStates();
    }

    /** Takes the current model from the genetic prior, which has to be done once per proposal. */
    public void update() {
        this.geneticPrior.refreshModel();
        this.frequencies = this.geneticPrior.getFrequencies();
    }

    /**
     * Writes the transition probabilities of the branch above the given non-root node into
     * the given array, row-major by the state at the parent.
     */
    public void getTransitionProbabilities(Node node, double[] transitionProbabilities) {
        this.checkUpdated();
        this.geneticPrior.getTransitionProbabilities(node, transitionProbabilities);
    }

    public double[] getFrequencies() {
        this.checkUpdated();
        return this.frequencies;
    }

    private void checkUpdated() {
        if (this.frequencies == null) {
            throw new IllegalStateException("The substitution process has to be updated before it is used.");
        }
    }

    public TipCompatibility getTipCompatibility() {
        return this.geneticPrior.getTipCompatibility();
    }

    public int getNumStates() {
        return this.numStates;
    }

}
