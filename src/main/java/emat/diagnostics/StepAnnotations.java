package emat.diagnostics;

import java.util.Arrays;

/**
 * Operator-specific measurements of the MCMC step in progress. An operator writes them while it
 * proposes, and the diagnostic MCMC reads and clears them once the step is resolved. A slot
 * that the operator did not write stays NaN. Each slot is named by {@link DiagnosedOperator}.
 */
public final class StepAnnotations {

    public static final int NUM_SLOTS = 10;

    private static final double[] VALUES = new double[NUM_SLOTS];

    static {
        Arrays.fill(VALUES, Double.NaN);
    }

    private StepAnnotations() {
    }

    /** Records a measurement of the current step in the given slot. */
    public static void set(int slot, double value) {
        VALUES[slot] = value;
    }

    /** Copies the measurements of the current step into the given array and clears them. */
    public static void drainInto(double[] target) {
        System.arraycopy(VALUES, 0, target, 0, NUM_SLOTS);
        Arrays.fill(VALUES, Double.NaN);
    }

    /** Clears the measurements of the current step. */
    public static void clear() {
        Arrays.fill(VALUES, Double.NaN);
    }

}
