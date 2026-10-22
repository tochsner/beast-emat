package emat.operators.nodetimes.batch;

import beast.base.util.Randomizer;

/** Draws from the densities exp(-a x) and exp(-a x - b x²) truncated to an interval. */
public final class TruncatedDensitySampler {

    private TruncatedDensitySampler() {
    }

    /**
     * Samples a value from the density ∝ exp(-rate x) truncated to [lower, upper] by inverting its cumulative
     * distribution. The rate may be negative, in which case the density grows towards the upper end, and it falls
     * back to a uniform draw if the density is flat on the interval.
     */
    public static double sampleExponential(double rate, double lower, double upper) {
        double width = upper - lower;
        double u = Randomizer.nextDouble();

        double value;
        if (Math.abs(rate) * width < 1e-12) {
            value = lower + u * width;
        } else if (rate > 0.0) {
            value = lower - Math.log1p(u * Math.expm1(-rate * width)) / rate;
        } else {
            value = upper - Math.log1p(u * Math.expm1(rate * width)) / rate;
        }

        // rounding must not push the value out of the interval
        return Math.min(Math.max(value, lower), upper);
    }

    /**
     * Samples a value from the density ∝ exp(-a x - b x²) with b > 0, truncated to [lower, upper]. The density is
     * a normal distribution around its mode μ = -a / (2b). Away from the mode it falls like exp(-r y - b y²) with
     * y the distance from the end of the interval that is nearest to the mode, and an interval that holds the mode
     * is split there into two such pieces that are picked by their mass.
     */
    public static double sampleGaussianTilt(double a, double b, double lower, double upper) {
        double mode = -a / (2.0 * b);

        if (mode <= lower) {
            return lower + sampleDecayingDistance(2.0 * b * (lower - mode), b, upper - lower);
        }
        if (mode >= upper) {
            return upper - sampleDecayingDistance(2.0 * b * (mode - upper), b, upper - lower);
        }

        double sqrtB = Math.sqrt(b);
        double massBelow = computeErf(sqrtB * (mode - lower));
        double massAbove = computeErf(sqrtB * (upper - mode));

        if (Randomizer.nextDouble() * (massBelow + massAbove) < massBelow) {
            return mode - sampleDecayingDistance(0.0, b, mode - lower);
        }
        return mode + sampleDecayingDistance(0.0, b, upper - mode);
    }

    /**
     * Samples y from the density ∝ exp(-r y - b y²) on [0, width] with r ≥ 0 and b > 0 by rejection. The
     * concave exponent lies below each of its tangents, so the density is below an exponential density
     * ∝ exp(-(r + 2 b y0) y) and the draw from it is accepted with probability exp(-b (y - y0)²). The tangent
     * point y0 = min(width, 1 / √b) keeps that probability high both for narrow and for wide intervals.
     */
    private static double sampleDecayingDistance(double r, double b, double width) {
        double tangentPoint = Math.min(width, 1.0 / Math.sqrt(b));
        double envelopeRate = r + 2.0 * b * tangentPoint;

        while (true) {
            double y = sampleExponential(envelopeRate, 0.0, width);
            double deviation = y - tangentPoint;
            if (Math.log(Randomizer.nextDouble()) < -b * deviation * deviation) {
                return y;
            }
        }
    }

    /** Returns the error function, from its series, which converges quickly for the arguments up to 6. */
    static double computeErf(double x) {
        if (x < 0.0) {
            return -computeErf(-x);
        }
        if (x > 6.0) {
            return 1.0;
        }

        double term = x;
        double sum = x;
        for (int n = 1; n < 200 && term > 1e-17 * sum; n++) {
            term *= 2.0 * x * x / (2 * n + 1);
            sum += term;
        }
        return 2.0 / Math.sqrt(Math.PI) * Math.exp(-x * x) * sum;
    }

}
