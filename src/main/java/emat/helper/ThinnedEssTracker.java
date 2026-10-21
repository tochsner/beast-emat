package emat.helper;

import beast.base.spec.inference.util.ESS;

import java.util.ArrayList;
import java.util.List;

/**
 * Records the values of a single parameter and reports the effective sample size (ESS) of its thinned trace.
 *
 * <p>Only every {@code thinning}-th recorded value enters the trace, which saves memory and keeps the
 * autocorrelation of the trace short. The ESS is expensive, so it is only recomputed once the trace has grown by
 * a fixed factor; in between, the most recent value is returned.
 */
public class ThinnedEssTracker {

    private static final int FIRST_ESS_CHECK = 100;
    private static final double ESS_CHECK_GROWTH = 1.1;

    private final int thinning;
    private final List<Double> trace = new ArrayList<>();

    private int numRecorded;
    private int nextEssCheck = FIRST_ESS_CHECK;
    private double ess;

    public ThinnedEssTracker(int thinning) {
        if (thinning < 1) {
            throw new IllegalArgumentException("The thinning must be at least one.");
        }

        this.thinning = thinning;
    }

    /**
     * Records a value, which enters the trace if it falls on the thinning interval, and returns the current ESS of
     * the thinned trace.
     */
    public double record(double value) {
        this.numRecorded++;

        if (this.numRecorded % this.thinning == 0) {
            this.trace.add(value);
            this.updateEss();
        }

        return this.ess;
    }

    /**
     * The ESS of the thinned trace as of the last evaluation, which is zero as long as the trace is too short.
     */
    public double getEss() {
        return this.ess;
    }

    /**
     * Whether the ESS of the thinned trace has exceeded the given value.
     */
    public boolean hasReached(double targetEss) {
        return this.ess > targetEss;
    }

    private void updateEss() {
        // the ESS is only evaluated at geometrically spaced sample counts
        if (this.trace.size() < this.nextEssCheck) return;
        this.nextEssCheck = (int) Math.ceil(this.nextEssCheck * ESS_CHECK_GROWTH);

        double newEss = ESS.calcESS(this.trace);

        // a trace that has not moved yet has no autocorrelation to speak of
        this.ess = Double.isNaN(newEss) ? 0.0 : newEss;
    }
}
