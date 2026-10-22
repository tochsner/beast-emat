package emat;

import beast.base.util.Randomizer;
import emat.helper.GlobalSplineTreePriorApproximation;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.Timeout;

import java.util.function.DoubleUnaryOperator;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertTrue;
import static org.junit.jupiter.api.Assertions.assertThrows;

public class GlobalSplineTreePriorApproximationTest {

    // the log tree prior of a node at height h for the parameter θ is θ (2 h - 1.5 h²), with the slope θ (2 - 3 h)
    private static double getLogTreePrior(double height, double theta) {
        return theta * (2.0 * height - 1.5 * height * height);
    }

    private GlobalSplineTreePriorApproximation fitApproximation(int numObservations) {
        return this.fitApproximation(numObservations, height -> getLogTreePrior(height, 1.0));
    }

    private GlobalSplineTreePriorApproximation fitApproximation(int numObservations, DoubleUnaryOperator logPrior) {
        GlobalSplineTreePriorApproximation approximation = new GlobalSplineTreePriorApproximation(12, 1, 200, 1e-8);

        for (int i = 0; i < numObservations; i++) {
            double theta = 0.5 + Randomizer.nextDouble();
            double[] parameters = {theta};

            approximation.startObservation();
            double change = 0.0;
            int numNodes = 1 + Randomizer.nextInt(3);
            for (int node = 0; node < numNodes; node++) {
                double oldHeight = Randomizer.nextDouble();
                double newHeight = Randomizer.nextDouble();
                double midpoint = (oldHeight + newHeight) / 2.0;
                approximation.addNode(midpoint, parameters, oldHeight - midpoint, newHeight - midpoint);
                change += theta * (logPrior.applyAsDouble(newHeight) - logPrior.applyAsDouble(oldHeight));
            }
            approximation.finishObservation(change);
        }

        return approximation;
    }

    /** The slope of a log tree prior that is linear in the height and in the parameter is learned exactly. */
    @Test
    public void testLearnsSlopeAndChange() {
        Randomizer.setSeed(1);

        GlobalSplineTreePriorApproximation approximation = new GlobalSplineTreePriorApproximation(12, 1, 200, 1e-8);
        assertFalse(approximation.isFitted());
        assertEquals(0.0, approximation.getChange(0.0, 1.0, 0.2, 0.7, new double[]{1.0}), 0.0);

        approximation = this.fitApproximation(3000);
        assertTrue(approximation.isFitted());

        for (double theta : new double[]{0.6, 1.0, 1.4}) {
            double[] parameters = {theta};
            assertEquals(theta * (2.0 - 3.0 * 0.4), approximation.getSlope(0.4, parameters), 1e-3);
            assertEquals(-3.0 * theta, approximation.getCurvature(0.4, parameters), 1e-2);
            assertEquals(getLogTreePrior(0.9, theta) - getLogTreePrior(0.15, theta),
                    approximation.getChange(0.0, 1.0, 0.15, 0.9, parameters), 1e-3);
        }
    }

    /**
     * The draws follow the density ∝ exp(-k h + a(h)) between the bounds, whatever the sign and the size of the
     * decay rate, including intervals that cross several knots and ranges that stick out beyond them.
     */
    @Test
    @Timeout(15)
    public void testSampledHeightsFollowTheApproximateDensity() {
        Randomizer.setSeed(3);
        GlobalSplineTreePriorApproximation approximation = this.fitApproximation(3000);

        double[][] cases = {
                // decay rate, lower, upper, θ
                {0.0, 0.0, 1.0, 1.0},
                {5.0, 0.1, 0.9, 0.7},
                {-8.0, 0.3, 0.6, 1.3},
                {60.0, 0.0, 1.0, 1.0},
                {-40.0, -0.5, 1.5, 1.2},
                {3.0, 0.45, 0.47, 1.0},
                // strong concavity previously stalled the single exponential proposal
                {0.0, 0.0, 1.0, 1000.0},
                {500.0, 0.0, 1.0, 1000.0},
                // convex residuals and linear extrapolation on either side of the knots
                {0.0, 0.0, 1.0, -20.0},
                {2.0, -2.0, -1.0, 1.0},
                {-2.0, 2.0, 3.0, 1.0},
        };

        this.assertSampledDensity(approximation, cases);
    }

    /** Refining one peak must preserve the relative masses of all peaks. */
    @Test
    @Timeout(15)
    public void testMultimodalDensity() {
        Randomizer.setSeed(17);
        GlobalSplineTreePriorApproximation approximation = this.fitApproximation(3000,
                height -> 20.0 * Math.cos(4.0 * Math.PI * height));
        this.assertSampledDensity(approximation, new double[][]{
                {0.0, 0.0, 1.0, 1.0},
                {10.0, 0.0, 1.0, 10.0}
        });
    }

    /** Non-finite inputs fail explicitly instead of entering a rejection loop that cannot accept. */
    @Test
    public void testNonFiniteEnvelopeFails() {
        Randomizer.setSeed(3);
        GlobalSplineTreePriorApproximation approximation = this.fitApproximation(3000);
        assertThrows(IllegalArgumentException.class,
                () -> approximation.sampleHeight(Double.NaN, 0.0, 1.0, new double[]{1.0}));
        assertThrows(ArithmeticException.class,
                () -> approximation.sampleHeight(0.0, 0.0, 1.0, new double[]{Double.NaN}));
    }

    private void assertSampledDensity(GlobalSplineTreePriorApproximation approximation, double[][] cases) {
        int numDraws = 100000;
        int numBins = 16;
        int numCells = 20000;

        for (double[] c : cases) {
            double decayRate = c[0];
            double lower = c[1];
            double upper = c[2];
            double[] parameters = {c[3]};

            double[] cumulative = new double[numCells + 1];
            double largest = Double.NEGATIVE_INFINITY;
            double[] exponents = new double[numCells];
            for (int i = 0; i < numCells; i++) {
                double h = lower + (i + 0.5) / numCells * (upper - lower);
                exponents[i] = -decayRate * (h - lower) + approximation.getChange(lower, upper, lower, h, parameters);
                largest = Math.max(largest, exponents[i]);
            }
            for (int i = 0; i < numCells; i++) {
                cumulative[i + 1] = cumulative[i] + Math.exp(exponents[i] - largest);
            }

            int[] counts = new int[numBins];
            for (int i = 0; i < numDraws; i++) {
                double height = approximation.sampleHeight(decayRate, lower, upper, parameters);
                assertTrue(height >= lower && height <= upper);
                counts[Math.min((int) ((height - lower) / (upper - lower) * numBins), numBins - 1)]++;
            }

            for (int bin = 0; bin < numBins; bin++) {
                double expected = (cumulative[(bin + 1) * numCells / numBins] - cumulative[bin * numCells / numBins]) / cumulative[numCells];
                assertEquals(expected, counts[bin] / (double) numDraws, 0.006, "case " + decayRate + ", " + lower + ", " + upper + ", bin " + bin);
            }
        }
    }

}
