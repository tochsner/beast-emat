package emat.operators.treeprior;

import beast.base.evolution.tree.Tree;
import beast.base.util.Randomizer;

import java.util.Arrays;
import java.util.function.ToDoubleFunction;

/**
 * Samples the real representation of the parameters of a tree prior from a distribution that is learned from the chain,
 * given statistics of the tree and further conditions.
 *
 * <p>The distribution is a {@link QuadraticGaussianModel}. It learns from sweeps, which run slice sampling over the
 * parameters with the tree fixed, so that their draws follow the distribution of the parameters given this very tree rather
 * than the joint distribution of the chain, in which the parameters are confounded with the tree. A sweep only evaluates
 * the log target through the callback it is handed, and leaves the parameters at their current values.
 */
public class TreePriorSampler {

    private static final int MAX_STEPS_OUT = 10;
    private static final int MAX_SHRINKS = 50;
    private static final double ADAPTATION_OFFSET = 10.0;
    private static final double ADAPTATION_EXPONENT = 0.6;
    private static final double MIN_WINDOW_SIZE = 1.0E-12;
    private static final double MAX_WINDOW_SIZE = 1.0E12;

    private final int numMutable;
    private final int numImmutable;
    private final int sweepEvery;
    private final int numDrawsPerSweep;
    private final int minRecords;
    private final double covarianceScale;

    // the recorded draws, as a ring buffer
    private final double[][] recordedConditions;
    private final double[][] recordedValues;
    private int numStored;
    private int nextIndex;
    private long numRecorded;
    private long numRecordedAtFit;

    private final double[] windowSizes;
    private long numWindowUpdates;
    private long numRecordCalls;

    private QuadraticGaussianModel model;

    public TreePriorSampler(int numMutable, int numImmutable, int sweepEvery, int numDrawsPerSweep, int minRecords, int maxRecords, double covarianceScale) {
        this.numMutable = numMutable;
        this.numImmutable = numImmutable;
        this.sweepEvery = sweepEvery;
        this.numDrawsPerSweep = numDrawsPerSweep;
        this.minRecords = minRecords;
        this.covarianceScale = covarianceScale;

        this.recordedConditions = new double[maxRecords][];
        this.recordedValues = new double[maxRecords][];
        this.windowSizes = new double[numMutable];
        Arrays.fill(this.windowSizes, 1.0);
    }

    /** Returns whether the distribution has been fitted, which is the case once enough draws have been recorded. */
    public boolean isFitted() {
        return this.model != null;
    }

    /** Draws the values of the parameters given the tree and the immutable values. */
    public double[] sample(Tree tree, double[] immutable) {
        return this.sample(this.createConditions(tree, immutable));
    }

    /** Returns the log density of the values of the parameters given the tree and the immutable values. */
    public double computeLogDensity(Tree tree, double[] immutable, double[] mutable) {
        return this.computeLogDensity(this.createConditions(tree, immutable), mutable);
    }

    /** Draws the values of the parameters given conditions that were created from a tree and the immutable values. */
    public double[] sample(double[] conditions) {
        return this.model.sample(conditions, this.covarianceScale);
    }

    /** Returns the log density of the values of the parameters given conditions that were created from a tree and the immutable values. */
    public double computeLogDensity(double[] conditions, double[] mutable) {
        return this.model.computeLogDensity(conditions, mutable, this.covarianceScale);
    }

    /** Returns the conditions of the tree and the immutable values, which stay valid when the tree changes afterwards. */
    public double[] createConditions(Tree tree, double[] immutable) {
        double[] conditions = new double[TreeStatistics.COUNT + this.numImmutable];
        System.arraycopy(TreeStatistics.compute(tree), 0, conditions, 0, TreeStatistics.COUNT);
        System.arraycopy(immutable, 0, conditions, TreeStatistics.COUNT, this.numImmutable);
        return conditions;
    }

    /**
     * Counts a call and sweeps on every few calls, starting from the given current values. The callback must set the
     * parameters to the given values and return the log target density of their real representation, or negative infinity
     * if they are invalid. The sweep ends with a call of the callback at the current values, which leaves the parameters
     * and everything the callback keeps up to date as they were.
     */
    public void record(Tree tree, double[] immutable, double[] current, ToDoubleFunction<double[]> evaluateLogTarget) {
        this.numRecordCalls++;
        if (this.numRecordCalls % this.sweepEvery != 0) {
            return;
        }

        double[] conditions = this.createConditions(tree, immutable);
        double[] state = current.clone();
        double logTarget = evaluateLogTarget.applyAsDouble(state);
        if (Double.isFinite(logTarget)) {
            for (int draw = 0; draw < this.numDrawsPerSweep; draw++) {
                logTarget = this.updateCoordinates(state, logTarget, evaluateLogTarget);
                this.store(conditions, state);
            }
        }

        evaluateLogTarget.applyAsDouble(current);
        this.refitIfDue();
    }

    /** Updates every coordinate of the state by one slice sampling step, and returns the log target at the new state. */
    private double updateCoordinates(double[] state, double logTarget, ToDoubleFunction<double[]> evaluateLogTarget) {
        for (int coordinate = 0; coordinate < this.numMutable; coordinate++) {
            logTarget = this.updateCoordinate(state, coordinate, logTarget, evaluateLogTarget);
        }
        return logTarget;
    }

    /**
     * Samples the coordinate uniformly from the slice under the log target, by stepping out from a randomly placed window
     * and shrinking it on every point outside the slice. The coordinate is left as it is if either takes too many steps.
     */
    private double updateCoordinate(double[] state, int coordinate, double logTarget, ToDoubleFunction<double[]> evaluateLogTarget) {
        double start = state[coordinate];
        double logSliceHeight = logTarget + Math.log(Randomizer.nextDouble());
        double windowSize = this.windowSizes[coordinate];

        double lower = start - windowSize * Randomizer.nextDouble();
        double upper = lower + windowSize;
        int numStepsOut = 0;
        while (this.evaluateAt(state, coordinate, lower, evaluateLogTarget) >= logSliceHeight) {
            lower -= windowSize;
            if (++numStepsOut > MAX_STEPS_OUT) {
                state[coordinate] = start;
                return logTarget;
            }
        }
        while (this.evaluateAt(state, coordinate, upper, evaluateLogTarget) >= logSliceHeight) {
            upper += windowSize;
            if (++numStepsOut > MAX_STEPS_OUT) {
                state[coordinate] = start;
                return logTarget;
            }
        }

        int numShrinks = 0;
        while (numShrinks <= MAX_SHRINKS) {
            double candidate = lower + Randomizer.nextDouble() * (upper - lower);
            double logCandidate = this.evaluateAt(state, coordinate, candidate, evaluateLogTarget);
            if (logCandidate >= logSliceHeight) {
                this.adaptWindowSize(coordinate, numStepsOut, numShrinks);
                return logCandidate;
            }

            if (candidate < start) {
                lower = candidate;
            } else {
                upper = candidate;
            }
            numShrinks++;
        }

        state[coordinate] = start;
        return logTarget;
    }

    private double evaluateAt(double[] state, int coordinate, double value, ToDoubleFunction<double[]> evaluateLogTarget) {
        state[coordinate] = value;
        return evaluateLogTarget.applyAsDouble(state);
    }

    /** Moves the window of the coordinate towards one step out and one shrink per update, with a gain that decays. */
    private void adaptWindowSize(int coordinate, int numStepsOut, int numShrinks) {
        this.numWindowUpdates++;
        double gain = Math.pow(this.numWindowUpdates / (double) this.numMutable + ADAPTATION_OFFSET, -ADAPTATION_EXPONENT);
        double logWindowSize = Math.log(this.windowSizes[coordinate]) + gain * ((numStepsOut - 1.0) - (numShrinks - 1.0));
        this.windowSizes[coordinate] = Math.min(MAX_WINDOW_SIZE, Math.max(MIN_WINDOW_SIZE, Math.exp(logWindowSize)));
    }

    private void store(double[] conditions, double[] values) {
        this.recordedConditions[this.nextIndex] = conditions;
        this.recordedValues[this.nextIndex] = values.clone();
        this.nextIndex = (this.nextIndex + 1) % this.recordedConditions.length;
        this.numStored = Math.min(this.numStored + 1, this.recordedConditions.length);
        this.numRecorded++;
    }

    /** Fits the first time once enough draws are there, and again whenever the draws since the last fit are half of those it used. */
    private void refitIfDue() {
        boolean isFirstFit = this.model == null && this.numStored >= this.minRecords;
        boolean isRefit = this.model != null && this.numRecorded - this.numRecordedAtFit >= Math.max(this.minRecords, this.numStored / 2);
        if (!isFirstFit && !isRefit) {
            return;
        }

        QuadraticGaussianModel fitted = new QuadraticGaussianModel(TreeStatistics.COUNT + this.numImmutable, this.numMutable);
        try {
            fitted.fit(Arrays.copyOf(this.recordedConditions, this.numStored), Arrays.copyOf(this.recordedValues, this.numStored));
        } catch (RuntimeException exception) {
            // the draws cannot be fitted yet, so the previous model stays and the fit is tried again later
            return;
        }
        this.model = fitted;
        this.numRecordedAtFit = this.numRecorded;
    }

}
