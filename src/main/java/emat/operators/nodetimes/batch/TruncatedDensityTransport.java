package emat.operators.nodetimes.batch;

import org.apache.commons.statistics.distribution.NormalDistribution;

/**
 * Transports a value that follows the density ∝ exp(-rate x - stiffness x²) on [lower, upper] to a uniform or a
 * standard normal value and back. The transport is the cumulative distribution function of the density, so it
 * is exact also for ranges that lie far out in the tail of the normal density, where the cumulative distribution
 * function itself would underflow. Every probability is given as its lower and its upper tail, so that a value
 * close to either end of the range keeps its precision.
 */
public final class TruncatedDensityTransport {

    /** The probability of a value to lie below (lower) and above (upper) a given position. The two sum to one. */
    public record Tails(double lower, double upper) {

        /** Returns the tails of a standard normal value. */
        public static Tails ofNormalScore(double score) {
            return new Tails(NORMAL.cumulativeProbability(score), NORMAL.survivalProbability(score));
        }

        /** Returns the tails of a uniform value. */
        public static Tails ofProbability(double probability) {
            return new Tails(probability, 1.0 - probability);
        }

        /** Returns the standard normal value of the tails, from the smaller tail for precision. */
        public double toNormalScore() {
            if (this.lower <= this.upper) {
                return NORMAL.inverseCumulativeProbability(Math.max(this.lower, MIN_TAIL));
            }
            return NORMAL.inverseSurvivalProbability(Math.max(this.upper, MIN_TAIL));
        }

        private Tails swap() {
            return new Tails(this.upper, this.lower);
        }

    }

    private static final NormalDistribution NORMAL = NormalDistribution.of(0.0, 1.0);

    // the smallest tail that is mapped to a normal value, which is about 37 standard deviations from zero
    private static final double MIN_TAIL = 1e-300;

    // below this, the exponent is flat on the range and the density is taken as uniform
    private static final double FLAT_EXPONENT = 1e-12;

    private static final double LOG_SQRT_TWO_PI = 0.5 * Math.log(2.0 * Math.PI);

    // from this distance to the mean on, the survival function of the normal is computed from its continued fraction
    private static final double CONTINUED_FRACTION_START = 30.0;

    private TruncatedDensityTransport() {
    }

    /**
     * Returns the tails of the given position for the density ∝ exp(-rate x - stiffness x²) on [lower, upper].
     * The stiffness must not be negative.
     */
    public static Tails computeTails(double rate, double stiffness, double lower, double upper, double x) {
        double width = upper - lower;
        double linear = rate + 2.0 * stiffness * lower;
        double distance = Math.min(Math.max(x - lower, 0.0), width);

        // the shape is evaluated from the end of the range that is nearer to the mode, where the density falls
        boolean isMirrored = isMirrored(linear, stiffness, width);
        if (isMirrored) {
            linear = -(linear + 2.0 * stiffness * width);
            distance = width - distance;
        }

        Tails tails;
        if (stiffness == 0.0) {
            tails = computeExponentialTails(linear, width, distance);
        } else {
            double mode = -linear / (2.0 * stiffness);
            double deviation = 1.0 / Math.sqrt(2.0 * stiffness);
            double lowerScore = -mode / deviation;
            double upperScore = (width - mode) / deviation;
            double score = (distance - mode) / deviation;

            tails = lowerScore >= 0.0
                    ? computeFarNormalTails(lowerScore, upperScore, score)
                    : computeNearNormalTails(lowerScore, upperScore, score);
        }

        return isMirrored ? tails.swap() : tails;
    }

    /**
     * Returns the position for the given tails, which is the inverse of {@link #computeTails} for the same
     * density and range.
     */
    public static double computeValue(double rate, double stiffness, double lower, double upper, Tails tails) {
        double width = upper - lower;
        double linear = rate + 2.0 * stiffness * lower;

        boolean isMirrored = isMirrored(linear, stiffness, width);
        if (isMirrored) {
            linear = -(linear + 2.0 * stiffness * width);
            tails = tails.swap();
        }

        double distance;
        if (stiffness == 0.0) {
            distance = computeExponentialDistance(linear, width, tails);
        } else {
            double mode = -linear / (2.0 * stiffness);
            double deviation = 1.0 / Math.sqrt(2.0 * stiffness);
            double lowerScore = -mode / deviation;
            double upperScore = (width - mode) / deviation;

            double score = lowerScore >= 0.0
                    ? computeFarNormalScore(lowerScore, upperScore, tails)
                    : computeNearNormalScore(lowerScore, upperScore, tails);
            distance = mode + deviation * score;
        }

        distance = Math.min(Math.max(distance, 0.0), width);
        return lower + (isMirrored ? width - distance : distance);
    }

    /** Returns whether the mode of the density in the distance from the lower end lies in the upper half of the range. */
    private static boolean isMirrored(double linear, double stiffness, double width) {
        if (stiffness == 0.0) {
            return linear < 0.0;
        }
        return -linear / (2.0 * stiffness) > width / 2.0;
    }

    /* Exponential density ∝ exp(-linear d) on [0, width] with a linear coefficient that is not negative */

    private static Tails computeExponentialTails(double linear, double width, double distance) {
        if (linear * width < FLAT_EXPONENT) {
            return Tails.ofProbability(distance / width);
        }

        double mass = -Math.expm1(-linear * width);
        double lower = -Math.expm1(-linear * distance) / mass;
        double upper = Math.exp(-linear * distance) * -Math.expm1(-linear * (width - distance)) / mass;
        return new Tails(lower, upper);
    }

    private static double computeExponentialDistance(double linear, double width, Tails tails) {
        if (linear * width < FLAT_EXPONENT) {
            return tails.lower() * width;
        }

        if (tails.lower() <= tails.upper()) {
            return -Math.log1p(tails.lower() * Math.expm1(-linear * width)) / linear;
        }

        double logMass = Math.log(-Math.expm1(-linear * width));
        return -computeLogSumExp(Math.log(tails.upper()) + logMass, -linear * width) / linear;
    }

    /* Normal density with the mode below the range, in standardised scores. The tails come from the log survival function */

    private static Tails computeFarNormalTails(double lowerScore, double upperScore, double score) {
        double logSurvivalLower = computeLogSurvival(lowerScore);
        double relativeScore = computeLogSurvival(score) - logSurvivalLower;
        double relativeUpper = computeLogSurvival(upperScore) - logSurvivalLower;

        double mass = -Math.expm1(relativeUpper);
        double lower = -Math.expm1(relativeScore) / mass;
        double upper = Math.exp(relativeScore) * -Math.expm1(relativeUpper - relativeScore) / mass;
        return new Tails(lower, upper);
    }

    private static double computeFarNormalScore(double lowerScore, double upperScore, Tails tails) {
        double logSurvivalLower = computeLogSurvival(lowerScore);
        double relativeUpper = computeLogSurvival(upperScore) - logSurvivalLower;

        double relativeScore;
        if (tails.lower() <= tails.upper()) {
            relativeScore = Math.log1p(tails.lower() * Math.expm1(relativeUpper));
        } else {
            double logMass = Math.log(-Math.expm1(relativeUpper));
            relativeScore = computeLogSumExp(Math.log(tails.upper()) + logMass, relativeUpper);
        }

        double score = computeInverseLogSurvival(logSurvivalLower + relativeScore);
        return Math.min(Math.max(score, lowerScore), upperScore);
    }

    /** Returns the logarithm of the survival function of the standard normal at a score that is not negative. */
    private static double computeLogSurvival(double score) {
        if (score < CONTINUED_FRACTION_START) {
            return Math.log(NORMAL.survivalProbability(score));
        }

        // the survival function is the density times the Mills ratio 1 / (x + 1 / (x + 2 / (x + 3 / (x + ...))))
        double fraction = score;
        for (int n = 40; n >= 1; n--) {
            fraction = score + n / fraction;
        }
        return -0.5 * score * score - LOG_SQRT_TWO_PI - Math.log(fraction);
    }

    /** Returns the score at which the logarithm of the survival function of the standard normal takes the given value. */
    private static double computeInverseLogSurvival(double logSurvival) {
        if (logSurvival > -700.0) {
            return NORMAL.inverseSurvivalProbability(Math.exp(logSurvival));
        }

        // start from the tail approximation Q(x) ≈ φ(x) / x, then polish by Newton steps on the concave log survival
        double score = Math.sqrt(-2.0 * logSurvival);
        for (int i = 0; i < 3; i++) {
            score = Math.sqrt(-2.0 * logSurvival - 2.0 * Math.log(score) - 2.0 * LOG_SQRT_TWO_PI);
        }
        for (int i = 0; i < 4; i++) {
            double logSurvivalAtScore = computeLogSurvival(score);
            double logDensity = -0.5 * score * score - LOG_SQRT_TWO_PI;
            double slope = -Math.exp(logDensity - logSurvivalAtScore);
            score -= (logSurvivalAtScore - logSurvival) / slope;
        }
        return score;
    }

    /* Normal density with the mode inside the range or above its lower end, in standardised scores */

    private static Tails computeNearNormalTails(double lowerScore, double upperScore, double score) {
        double survivalLower = NORMAL.survivalProbability(lowerScore);
        double survivalUpper = NORMAL.survivalProbability(upperScore);
        double mass = survivalLower - survivalUpper;

        double lowerMass = score < 0.0
                ? NORMAL.cumulativeProbability(score) - NORMAL.cumulativeProbability(lowerScore)
                : survivalLower - NORMAL.survivalProbability(score);
        double upperMass = NORMAL.survivalProbability(score) - survivalUpper;
        return new Tails(lowerMass / mass, upperMass / mass);
    }

    private static double computeNearNormalScore(double lowerScore, double upperScore, Tails tails) {
        double survivalLower = NORMAL.survivalProbability(lowerScore);
        double survivalUpper = NORMAL.survivalProbability(upperScore);
        double mass = survivalLower - survivalUpper;

        double score;
        if (tails.lower() <= tails.upper()) {
            score = NORMAL.inverseCumulativeProbability(NORMAL.cumulativeProbability(lowerScore) + tails.lower() * mass);
        } else {
            score = NORMAL.inverseSurvivalProbability(survivalUpper + tails.upper() * mass);
        }
        return Math.min(Math.max(score, lowerScore), upperScore);
    }

    /** Returns log(exp(a) + exp(b)). */
    private static double computeLogSumExp(double a, double b) {
        double max = Math.max(a, b);
        return max + Math.log1p(Math.exp(Math.min(a, b) - max));
    }

}
