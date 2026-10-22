package emat.operators.nodetimes.batch;

import beast.base.util.Randomizer;
import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.assertEquals;

public class TruncatedDensitySamplerTest {

    /**
     * The empirical distribution of the draws has to match the cumulative distribution computed by numerical
     * integration of exp(-a x - b x²), for modes inside the interval, far outside of it on either side, and for
     * intervals that are narrow or wide compared with the width of the density.
     */
    @Test
    public void testGaussianTiltMatchesDensity() {
        Randomizer.setSeed(5);

        double[][] cases = {
                // a, b, lower, upper
                {0.0, 4.0, -1.0, 1.0},
                {3.0, 2.0, -0.5, 2.0},
                {-40.0, 5.0, -1.0, 1.0},
                {40.0, 5.0, -1.0, 1.0},
                {2.0, 1e-3, -1.0, 1.0},
                {1.0, 1e4, -1.0, 3.0},
                {-300.0, 1e4, -0.01, 0.2},
                {1000.0, 0.5, 0.0, 10.0},
        };

        for (double[] c : cases) {
            this.assertDrawsMatchDensity(c[0], c[1], c[2], c[3]);
        }
    }

    private void assertDrawsMatchDensity(double a, double b, double lower, double upper) {
        int numDraws = 200000;
        int numBins = 20;

        // the mass of fine cells, shifted by the exponent at the mode for stability
        double mode = Math.min(Math.max(-a / (2.0 * b), lower), upper);
        int numCells = 20000;
        double[] cumulative = new double[numCells + 1];
        for (int i = 0; i < numCells; i++) {
            double x = lower + (i + 0.5) / numCells * (upper - lower);
            double exponent = -a * (x - mode) - b * (x * x - mode * mode);
            cumulative[i + 1] = cumulative[i] + Math.exp(exponent);
        }

        int[] counts = new int[numBins];
        for (int i = 0; i < numDraws; i++) {
            double x = TruncatedDensitySampler.sampleGaussianTilt(a, b, lower, upper);
            assertEquals(x, Math.min(Math.max(x, lower), upper), 0.0);
            counts[Math.min((int) ((x - lower) / (upper - lower) * numBins), numBins - 1)]++;
        }

        for (int bin = 0; bin < numBins; bin++) {
            int from = bin * numCells / numBins;
            int to = (bin + 1) * numCells / numBins;
            double expected = (cumulative[to] - cumulative[from]) / cumulative[numCells];
            assertEquals(expected, counts[bin] / (double) numDraws, 0.004, "a=" + a + ", b=" + b + ", bin " + bin);
        }
    }

}
