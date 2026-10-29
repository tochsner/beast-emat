package emat.operators.nodetimes;

import beast.base.core.Description;
import beast.base.core.Input;

@Description("Moves a random internal node in the ratio parameterisation of the node heights like the ratio scale " +
        "operator, but tunes a separate scale factor for every mutation count class of the subtree that moves: " +
        "the mutations on its branches, in half-octaves of 1 + m. The mutations move linearly with their branches, " +
        "so the genetic prior of a stretch of the subtree by a factor ρ is close to ρ^m exp(-Λ ρ), whose width in " +
        "log ρ is 1 / √(1 + m). Every class starts from this width times a prior scale.")
public class MutationRatioScaleOperator extends RatioScaleOperator {

    final public Input<Double> priorScaleInput = new Input<>("priorScale", "the width of the kernel in log space for a subtree without mutations, which sets the initial width c / √(1 + m) of every mutation count class", 1.0);

    // the number of classes per doubling of 1 + m, and the number of doublings that are covered
    private static final int CLASSES_PER_OCTAVE = 2;
    private static final int NUM_OCTAVES = 30;

    /** Returns the number of mutation count classes, which cover any count up to 2^30. */
    @Override
    protected int computeNumClasses() {
        return CLASSES_PER_OCTAVE * NUM_OCTAVES;
    }

    /** Returns c / √(1 + m) for the count m in the middle of the given class. */
    @Override
    protected double computeInitialScaleFactor(int stepClass) {
        double logCount = (stepClass + 0.5) / CLASSES_PER_OCTAVE;
        return this.priorScaleInput.get() / Math.sqrt(Math.pow(2.0, logCount));
    }

    /** Returns the mutation count class ⌊2 log₂(1 + m)⌋ of the subtree of the last proposal. */
    @Override
    protected int computeStepClass() {
        int stepClass = (int) Math.floor(CLASSES_PER_OCTAVE * Math.log(1.0 + this.subtreeNumMutations) / Math.log(2.0));
        return Math.min(stepClass, this.computeNumClasses() - 1);
    }

    @Override
    protected String describeStepClasses() {
        return "mutation count class";
    }

    @Override
    protected String describeStepClass(int stepClass) {
        return Math.round(Math.pow(2.0, (double) stepClass / CLASSES_PER_OCTAVE) - 1.0) + "+ mutations";
    }

}
