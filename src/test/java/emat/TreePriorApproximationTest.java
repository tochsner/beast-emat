package emat;

import beast.base.util.Randomizer;
import emat.helper.TreePriorApproximation;
import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertTrue;

public class TreePriorApproximationTest {

    /**
     * Generates observations from a quadratic in the offset, with one to five nodes moving per observation. Its
     * linear coefficient is curved in the midpoint m and linear in a parameter θ,
     * c_1 = 0.5 - 2 m + 40 m^2 + 0.3 θ, and its quadratic coefficient is c_2 = -1 + 0.1 m. The midpoints, the
     * parameter and the offsets have very different magnitudes. The fitted slope must match c_1.
     */
    @Test
    public void testRecoversSlopeOfKnownPolynomial() {
        Randomizer.setSeed(1);
        TreePriorApproximation approximation = new TreePriorApproximation(2, 2, 1, 200, 1e-12);

        double[] parameters = new double[1];
        assertEquals(0.0, approximation.getSlope(1.0, new double[]{2.0}), 0.0);

        for (int i = 0; i < 5000; i++) {
            approximation.startObservation();

            // the parameter is shared by the nodes of an observation
            parameters[0] = 150.0 + 50.0 * Randomizer.nextDouble();

            double target = 0.0;
            int numNodes = 1 + Randomizer.nextInt(5);
            for (int node = 0; node < numNodes; node++) {
                double midpoint = 0.3 * Randomizer.nextDouble();
                double oldOffset = 0.02 * (Randomizer.nextDouble() - 0.5);
                double newOffset = 0.02 * (Randomizer.nextDouble() - 0.5);

                approximation.addNode(midpoint, parameters, oldOffset, newOffset);
                target += this.getLinearCoefficient(midpoint, parameters[0]) * (newOffset - oldOffset)
                        + this.getQuadraticCoefficient(midpoint) * (newOffset * newOffset - oldOffset * oldOffset);
            }

            approximation.finishObservation(target);
        }

        assertTrue(approximation.isFitted());
        assertEquals(5000, approximation.getNumObservations());

        for (double[] point : new double[][]{{0.01, 150.0}, {0.29, 200.0}, {0.1, 170.0}}) {
            double expectedSlope = this.getLinearCoefficient(point[0], point[1]);
            assertEquals(expectedSlope, approximation.getSlope(point[0], new double[]{point[1]}), 1e-4 * Math.abs(expectedSlope));
        }
    }

    /**
     * A slope that follows a bump over the midpoint cannot be fitted by a line in the midpoint, but closely by a
     * cubic. Outside the midpoints it was fitted on, the slope stays at its value at the edge of that range.
     */
    @Test
    public void testHigherMidpointDegreeFollowsCurvedSlope() {
        double[] noParameters = new double[0];
        double[] maxErrors = new double[2];
        TreePriorApproximation cubic = null;

        for (int variant = 0; variant < 2; variant++) {
            Randomizer.setSeed(1);
            TreePriorApproximation approximation = new TreePriorApproximation(1, variant == 0 ? 1 : 3, 0, 200, 1e-12);

            for (int i = 0; i < 5000; i++) {
                double midpoint = 100.0 + 200.0 * Randomizer.nextDouble();
                double oldOffset = 5.0 * (Randomizer.nextDouble() - 0.5);
                double newOffset = 5.0 * (Randomizer.nextDouble() - 0.5);

                approximation.startObservation();
                approximation.addNode(midpoint, noParameters, oldOffset, newOffset);
                approximation.finishObservation(this.getBumpSlope(midpoint) * (newOffset - oldOffset));
            }

            for (double midpoint = 105.0; midpoint <= 295.0; midpoint += 5.0) {
                maxErrors[variant] = Math.max(maxErrors[variant], Math.abs(approximation.getSlope(midpoint, noParameters) - this.getBumpSlope(midpoint)));
            }
            cubic = approximation;
        }

        assertTrue(maxErrors[0] > 0.2);
        assertTrue(maxErrors[1] < 0.05);
        assertEquals(cubic.getSlope(300.0, noParameters), cubic.getSlope(1000.0, noParameters), 1e-3);
        assertEquals(cubic.getSlope(100.0, noParameters), cubic.getSlope(-500.0, noParameters), 1e-3);
    }

    /**
     * A midpoint and a parameter that never vary and nodes that never move carry no information. The fit must
     * not fail on them, and the slope must stay finite.
     */
    @Test
    public void testDegenerateObservationsGiveFiniteSlope() {
        TreePriorApproximation approximation = new TreePriorApproximation(3, 3, 1, 50, 1e-8);
        double[] parameters = {5.0};

        for (int i = 0; i < 200; i++) {
            approximation.startObservation();
            approximation.addNode(1.0, parameters, 0.0, 0.0);
            approximation.finishObservation(0.0);
        }

        assertTrue(Double.isFinite(approximation.getSlope(1.0, parameters)));
        assertEquals(0.0, approximation.getSlope(1.0, parameters), 1e-12);
    }

    /** Observations without nodes or with a target that is not finite are not counted. */
    @Test
    public void testUnusableObservationsAreDropped() {
        TreePriorApproximation approximation = new TreePriorApproximation(1, 1, 0, 10, 1e-8);
        double[] noParameters = new double[0];

        approximation.startObservation();
        approximation.finishObservation(1.0);

        approximation.startObservation();
        approximation.addNode(1.0, noParameters, 0.0, 1.0);
        approximation.finishObservation(Double.NEGATIVE_INFINITY);

        approximation.startObservation();
        approximation.addNode(1.0, noParameters, 0.0, 1.0);
        approximation.discardObservation();
        approximation.finishObservation(1.0);

        assertEquals(0, approximation.getNumObservations());
        assertFalse(approximation.isFitted());
    }

    private double getLinearCoefficient(double midpoint, double parameter) {
        return 0.5 - 2.0 * midpoint + 40.0 * midpoint * midpoint + 0.3 * parameter;
    }

    private double getQuadraticCoefficient(double midpoint) {
        return -1.0 + 0.1 * midpoint;
    }

    /** Returns a slope that falls, bottoms out near a midpoint of 200 and rises again, lopsidedly. */
    private double getBumpSlope(double midpoint) {
        double distance = (midpoint - 200.0) / 100.0;
        return -1.0 + 2.0 * distance * distance - 0.5 * distance * distance * distance;
    }

}
