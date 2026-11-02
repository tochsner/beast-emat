package emat.operators.nodetimes.batch;

import emat.operators.nodetimes.batch.TruncatedDensityTransport.Tails;
import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertTrue;

public class TruncatedDensityTransportTest {

    // rate, stiffness, lower, upper: flat, exponential in both directions, normal with the mode inside, below and above
    // the range, and the far tails where the normal survival function underflows
    private static final double[][] DENSITIES = {
            {0.0, 0.0, -0.5, 0.5},
            {3.0, 0.0, -0.5, 0.5},
            {-3.0, 0.0, -0.5, 0.5},
            {400.0, 0.0, -0.5, 0.5},
            {-400.0, 0.0, -0.5, 0.5},
            {0.0, 4.0, -0.5, 0.5},
            {2.0, 4.0, -0.5, 0.5},
            {-2.0, 4.0, -0.5, 0.5},
            {30.0, 4.0, -0.5, 0.5},
            {-30.0, 4.0, -0.5, 0.5},
            {200.0, 0.5, -0.05, 0.05},
            {-200.0, 0.5, -0.05, 0.05},
            {3000.0, 0.01, -0.5, 0.5},
            {-3000.0, 0.01, -0.5, 0.5},
            {50.0, 1e4, -0.3, 0.3},
            {1.0, 1e-3, -2.0, 2.0},
    };

    /** The tails of a position are probabilities that add up to one and grow and fall with the position. */
    @Test
    public void testTailsAreConsistentAndMonotone() {
        for (double[] density : DENSITIES) {
            double previousLower = -1.0;
            for (int i = 0; i <= 50; i++) {
                double x = density[2] + (density[3] - density[2]) * i / 50.0;
                Tails tails = TruncatedDensityTransport.computeTails(density[0], density[1], density[2], density[3], x);

                assertEquals(1.0, tails.lower() + tails.upper(), 1e-9, this.describe(density, x));
                assertTrue(tails.lower() >= previousLower, this.describe(density, x));
                assertTrue(tails.lower() >= 0.0 && tails.upper() >= 0.0, this.describe(density, x));
                previousLower = tails.lower();
            }
        }
    }

    /** Positions are recovered from their tails, also close to the ends of the range and in the far tails. */
    @Test
    public void testRoundTripRecoversPositions() {
        for (double[] density : DENSITIES) {
            for (int i = 1; i < 50; i++) {
                double x = density[2] + (density[3] - density[2]) * i / 50.0;
                Tails tails = TruncatedDensityTransport.computeTails(density[0], density[1], density[2], density[3], x);
                if (Math.min(tails.lower(), tails.upper()) < 1e-12) {
                    continue;
                }

                double recovered = TruncatedDensityTransport.computeValue(density[0], density[1], density[2], density[3], tails);
                assertEquals(x, recovered, 1e-7 * (density[3] - density[2]), this.describe(density, x));
            }
        }
    }

    /** Tails that are given as a normal score round trip through the positions as well. */
    @Test
    public void testNormalScoresRoundTrip() {
        for (double[] density : DENSITIES) {
            for (double score = -6.0; score <= 6.0; score += 0.5) {
                double x = TruncatedDensityTransport.computeValue(density[0], density[1], density[2], density[3], Tails.ofNormalScore(score));
                double recovered = TruncatedDensityTransport.computeTails(density[0], density[1], density[2], density[3], x).toNormalScore();

                // positions are only resolved down to rounding, which limits the scores in the steep tails
                double resolution = 1e-9 * (density[3] - density[2]);
                double spread = Math.abs(x - TruncatedDensityTransport.computeValue(density[0], density[1], density[2], density[3], Tails.ofNormalScore(score + 0.01)));
                if (spread > resolution) {
                    assertEquals(score, recovered, 1e-3, this.describe(density, x) + " score " + score);
                }
            }
        }
    }

    /** The transport is the cumulative distribution function of the density, which is compared to a numerical integral. */
    @Test
    public void testTailsMatchNumericalIntegral() {
        for (double[] density : DENSITIES) {
            int numCells = 200000;
            double width = density[3] - density[2];
            double step = width / numCells;

            // the log density relative to its maximum, to stay clear of underflow
            double maxLogDensity = Double.NEGATIVE_INFINITY;
            for (int i = 0; i <= numCells; i++) {
                maxLogDensity = Math.max(maxLogDensity, this.computeLogDensity(density, density[2] + i * step));
            }

            double[] cumulative = new double[numCells + 1];
            for (int i = 1; i <= numCells; i++) {
                double left = Math.exp(this.computeLogDensity(density, density[2] + (i - 1) * step) - maxLogDensity);
                double right = Math.exp(this.computeLogDensity(density, density[2] + i * step) - maxLogDensity);
                cumulative[i] = cumulative[i - 1] + 0.5 * (left + right) * step;
            }

            for (int i = 1; i < 20; i++) {
                int cell = numCells * i / 20;
                double x = density[2] + cell * step;
                Tails tails = TruncatedDensityTransport.computeTails(density[0], density[1], density[2], density[3], x);

                assertEquals(cumulative[cell] / cumulative[numCells], tails.lower(), 2e-6, this.describe(density, x));
            }
        }
    }

    private double computeLogDensity(double[] density, double x) {
        return -density[0] * x - density[1] * x * x;
    }

    private String describe(double[] density, double x) {
        return "rate " + density[0] + ", stiffness " + density[1] + ", range [" + density[2] + ", " + density[3] + "], x " + x;
    }

}
