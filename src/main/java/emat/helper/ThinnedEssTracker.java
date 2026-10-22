package emat.helper;

import beast.base.spec.inference.util.ESS;

import java.util.ArrayList;
import java.util.List;

/**
 * Records the values of a single parameter and reports the effective sample size (ESS) of its thinned trace.
 *
 * <p>Only every {@code thinning}-th recorded value enters the trace, which saves memory and keeps the
 * autocorrelation of the trace short. The first part of the trace is dropped as burn-in, so that the initial
 * approach to the stationary distribution does not count as autocorrelation. The ESS is expensive, so it is only
 * recomputed once the trace has grown by a fixed factor; in between, the most recent value is returned. An ESS
 * restored from an earlier run is kept as a floor, as the trace of that run is not available.
 */
public class ThinnedEssTracker {

    private static final int FIRST_ESS_CHECK = 100;
    private static final double ESS_CHECK_GROWTH = 1.1;
    private static final double BURN_IN_FRACTION = 0.1;

    private final int thinning;
    private final List<Double> trace = new ArrayList<>();

    private int numRecorded;
    private int nextEssCheck = FIRST_ESS_CHECK;
    private double ess;
    private double restoredEss;

    public ThinnedEssTracker(int thinning) {
        if (thinning < 1) {
            throw new IllegalArgumentException("The thinning must be at least one.");
        }

        this.thinning = thinning;
    }

    /**
     * Records a value, which enters the trace if it falls on the thinning interval, and returns the current ESS of
     * the thinned trace without its burn-in.
     */
    public double record(double value) {
        this.numRecorded++;

        if (this.numRecorded % this.thinning == 0) {
            this.trace.add(value);
            this.updateEss();
        }

        return this.getEss();
    }

    /**
     * The ESS of the thinned trace without its burn-in as of the last evaluation, which is zero as long as the trace
     * is too short.
     */
    public double getEss() {
        return Math.max(this.ess, this.restoredEss);
    }

    /**
     * Sets the ESS that an earlier run reached, which counts as the ESS until the trace of this run reaches a higher one.
     */
    public void restoreEss(double ess) {
        this.restoredEss = Double.isFinite(ess) ? Math.max(ess, 0.0) : 0.0;
    }

    /**
     * Whether the ESS of the thinned trace without its burn-in has exceeded the given value.
     */
    public boolean hasReached(double targetEss) {
        return this.getEss() > targetEss;
    }

    private void updateEss() {
        // the ESS is only evaluated at geometrically spaced sample counts
        if (this.trace.size() < this.nextEssCheck) return;
        this.nextEssCheck = (int) Math.ceil(this.nextEssCheck * ESS_CHECK_GROWTH);

        int burnIn = (int) (BURN_IN_FRACTION * this.trace.size());
        double newEss = ESS.calcESS(this.trace.subList(burnIn, this.trace.size()));

        // a trace that has not moved yet has no autocorrelation to speak of
        this.ess = Double.isNaN(newEss) ? 0.0 : newEss;
    }
}
