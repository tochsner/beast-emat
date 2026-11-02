package emat.operators.nodetimes;

import beast.base.core.Description;
import beast.base.core.Input;
import beast.base.util.Randomizer;
import emat.diagnostics.StepAnnotations;

import java.util.Arrays;

@Description("An interval scale operator that is lifted: it carries a direction (grow or shrink the intervals), " +
        "proposes in that direction, keeps it as long as the proposals are accepted and reverses it on a rejection. " +
        "The chain on the pair of state and direction is non-reversible and leaves the posterior invariant, because the " +
        "direction is independent of the state under the target. The softness is the probability that a proposal " +
        "goes against the direction; with a softness of 0.5 the proposal is symmetric and the operator is reversible. " +
        "A proposal outside the allowed range of the tree length is rejected without evaluating it, which reverses the " +
        "direction and so reflects the walk at the end of the range.")
public class LiftedIntervalScaleOperator extends IntervalScaleOperator {

    final public Input<Double> softnessInput = new Input<>("softness", "probability that a proposal goes against the direction, in [0, 0.5]", 0.1);
    final public Input<Double> refreshProbabilityInput = new Input<>("refreshProbability", "probability per proposal to draw the direction anew", 0.01);
    final public Input<Double> targetAcceptanceInput = new Input<>("targetAcceptance", "acceptance probability towards which the scale factor is tuned", 0.6);
    final public Input<Double> minTreeLengthInput = new Input<>("minTreeLength", "proposals that make the tree shorter than this are rejected without evaluation", 0.0);
    final public Input<Double> maxTreeLengthInput = new Input<>("maxTreeLength", "proposals that make the tree longer than this are rejected without evaluation", Double.POSITIVE_INFINITY);

    private double softness;
    private double refreshProbability;
    private double targetAcceptance;
    private double minTreeLength;
    private double maxTreeLength;

    // the direction of the lifting: +1 scales the intervals up, -1 scales them down
    private int direction;
    private boolean isBackward;
    private int runLength;

    @Override
    public void initAndValidate() {
        super.initAndValidate();

        this.softness = this.softnessInput.get();
        this.refreshProbability = this.refreshProbabilityInput.get();
        this.targetAcceptance = this.targetAcceptanceInput.get();
        this.minTreeLength = this.minTreeLengthInput.get();
        this.maxTreeLength = this.maxTreeLengthInput.get();

        if (this.softness < 0.0 || this.softness > 0.5) {
            throw new IllegalArgumentException("the softness must lie in [0, 0.5]");
        }

        this.direction = Randomizer.nextBoolean() ? 1 : -1;
    }

    @Override
    public double proposal() {
        double logHastingsRatio = super.proposal();

        StepAnnotations.set(5, this.direction);
        StepAnnotations.set(6, this.isBackward ? 1.0 : 0.0);
        StepAnnotations.set(7, this.runLength);

        return logHastingsRatio;
    }

    /**
     * Draws the size of the move from the kernel and gives it the direction of the lifting, or, with
     * the probability of the softness, the opposite one. The reverse of a move along the direction is a
     * move along the reversed direction of the same size, so that the proposal densities cancel.
     */
    @Override
    protected double drawScaler() {
        if (Randomizer.nextDouble() < this.refreshProbability) {
            this.direction = Randomizer.nextBoolean() ? 1 : -1;
            this.runLength = 0;
        }

        double logSize = Math.abs(Math.log(super.drawScaler()));
        this.isBackward = Randomizer.nextDouble() < this.softness;
        int sign = this.isBackward ? -this.direction : this.direction;

        return Math.exp(sign * logSize);
    }

    @Override
    protected boolean isTreeLengthAllowed(double treeLength) {
        return treeLength >= this.minTreeLength && treeLength <= this.maxTreeLength;
    }

    /** Keeps the direction after an accepted proposal. */
    @Override
    public void accept() {
        super.accept();
        this.runLength++;
    }

    /** Reverses the direction after a rejected proposal, whatever the reason of the rejection. */
    @Override
    public void reject(int reason) {
        super.reject(reason);
        this.direction = -this.direction;
        this.runLength = 0;
    }

    @Override
    public double getTargetAcceptanceProbability() {
        return this.targetAcceptance;
    }

    @Override
    public String[] getAnnotationNames() {
        String[] names = super.getAnnotationNames();
        String[] lifted = Arrays.copyOf(names, names.length + 3);
        lifted[names.length] = "direction";
        lifted[names.length + 1] = "isBackward";
        lifted[names.length + 2] = "runLength";
        return lifted;
    }

}
