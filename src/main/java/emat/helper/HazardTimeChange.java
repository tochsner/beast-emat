package emat.helper;

/**
 * The time change that is the integral from height zero of a rate whose logarithm is piecewise linear in the height.
 * Above the last knot the logarithm of the rate continues with the slope of the last segment.
 */
public class HazardTimeChange {

    // below this size of the exponent, the integral of an exponential is taken as linear
    private static final double SMALL_EXPONENT = 1e-9;

    private final double[] knotHeights;
    private final double[] knotLogRates;
    private final double[] knotTimes;
    private final double[] slopes;
    private final boolean isValid;

    /** @param knotHeights the increasing heights of the knots, starting at zero */
    public HazardTimeChange(double[] knotHeights, double[] knotLogRates) {
        int numSegments = knotHeights.length - 1;
        this.knotHeights = knotHeights.clone();
        this.knotLogRates = knotLogRates.clone();
        this.knotTimes = new double[knotHeights.length];
        this.slopes = new double[numSegments];

        boolean valid = true;
        for (int i = 0; i < numSegments; i++) {
            this.slopes[i] = (this.knotLogRates[i + 1] - this.knotLogRates[i]) / (this.knotHeights[i + 1] - this.knotHeights[i]);
            this.knotTimes[i + 1] = this.knotTimes[i] + this.computeIntegral(i, this.knotHeights[i + 1] - this.knotHeights[i]);
            valid &= Double.isFinite(this.knotTimes[i + 1]) && this.knotTimes[i + 1] > this.knotTimes[i];
        }
        this.isValid = valid;
    }

    /** Returns whether the knot times are finite and increasing. */
    public boolean isValid() {
        return this.isValid;
    }

    /** Returns the integral of the rate from zero to the height. */
    public double computeTime(double height) {
        int segment = this.findSegment(height);
        return this.knotTimes[segment] + this.computeIntegral(segment, height - this.knotHeights[segment]);
    }

    /**
     * Returns the height at which the integral of the rate equals the given time, which is the inverse of
     * {@link #computeTime}, or NaN if the time is never reached.
     */
    public double computeHeight(double time) {
        int low = 0;
        int high = this.slopes.length;
        while (high - low > 1) {
            int middle = (low + high) >>> 1;
            if (this.knotTimes[middle] <= time) {
                low = middle;
            } else {
                high = middle;
            }
        }

        double scaled = (time - this.knotTimes[low]) / Math.exp(this.knotLogRates[low]);
        double slope = this.slopes[low];
        if (Math.abs(slope * scaled) < SMALL_EXPONENT) {
            return this.knotHeights[low] + scaled;
        }

        double argument = slope * scaled;
        return argument > -1.0 ? this.knotHeights[low] + Math.log1p(argument) / slope : Double.NaN;
    }

    /** Returns the logarithm of the rate at the height. */
    public double computeLogRate(double height) {
        int segment = this.findSegment(height);
        return this.knotLogRates[segment] + this.slopes[segment] * (height - this.knotHeights[segment]);
    }

    /** Returns the integral of the rate over the given offset from the start of the segment. */
    private double computeIntegral(int segment, double offset) {
        double exponent = this.slopes[segment] * offset;
        double mean = Math.abs(exponent) < SMALL_EXPONENT ? 1.0 + 0.5 * exponent : Math.expm1(exponent) / exponent;
        return Math.exp(this.knotLogRates[segment]) * offset * mean;
    }

    /** Returns the segment that contains the height, where the last segment continues without end. */
    private int findSegment(double height) {
        int low = 0;
        int high = this.slopes.length;
        while (high - low > 1) {
            int middle = (low + high) >>> 1;
            if (this.knotHeights[middle] <= height) {
                low = middle;
            } else {
                high = middle;
            }
        }
        return low;
    }

}
