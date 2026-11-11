package esat.operators.nodetimes;

import emat.operators.nodetimes.batch.TruncatedDensitySampler;
import beast.base.util.Randomizer;

/**
 * Samples from and evaluates a density whose logarithm is linear between a sorted list of knots, which is given by the
 * heights of the knots and the unnormalised log density at them.
 */
public final class PiecewiseExponentialSampler {

    private PiecewiseExponentialSampler() {
    }

    /**
     * Draws a value from the density that interpolates the log densities linearly between the first numKnots knots.
     * The segments are picked by their mass and the value is then drawn from the truncated exponential density of the
     * segment. The log densities must be finite, and the heights strictly ascending.
     */
    public static double sample(double[] heights, double[] logDensities, int numKnots, double[] segmentMasses) {
        double maxLogDensity = Double.NEGATIVE_INFINITY;
        for (int i = 0; i < numKnots; i++) {
            maxLogDensity = Math.max(maxLogDensity, logDensities[i]);
        }

        double totalMass = 0.0;
        for (int i = 0; i < numKnots - 1; i++) {
            totalMass += computeSegmentMass(heights, logDensities, i, maxLogDensity);
            segmentMasses[i] = totalMass;
        }

        double target = Randomizer.nextDouble() * totalMass;
        int segment = 0;
        while (segment < numKnots - 2 && segmentMasses[segment] < target) {
            segment++;
        }

        double lower = heights[segment];
        double upper = heights[segment + 1];
        double slope = (logDensities[segment + 1] - logDensities[segment]) / (upper - lower);
        return TruncatedDensitySampler.sampleExponential(-slope, lower, upper);
    }

    /** Returns the mass of the density on the given segment, relative to the density exp(maxLogDensity). */
    private static double computeSegmentMass(double[] heights, double[] logDensities, int segment, double maxLogDensity) {
        double width = heights[segment + 1] - heights[segment];
        double lowerDensity = Math.exp(logDensities[segment] - maxLogDensity);
        double upperDensity = Math.exp(logDensities[segment + 1] - maxLogDensity);

        double logChange = logDensities[segment + 1] - logDensities[segment];
        if (Math.abs(logChange) < 1e-9) {
            return width * lowerDensity;
        }
        return width * (upperDensity - lowerDensity) / logChange;
    }

    /** Returns the log density at the given height, which lies between the first and the last of the first numKnots knots. */
    public static double evaluate(double[] heights, double[] logDensities, int numKnots, double height) {
        int low = 0;
        int high = numKnots - 1;
        while (high - low > 1) {
            int middle = (low + high) >>> 1;
            if (heights[middle] <= height) {
                low = middle;
            } else {
                high = middle;
            }
        }

        double fraction = (height - heights[low]) / (heights[high] - heights[low]);
        return logDensities[low] + fraction * (logDensities[high] - logDensities[low]);
    }

}
