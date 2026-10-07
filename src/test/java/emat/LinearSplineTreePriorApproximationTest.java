package emat;

import beast.base.util.Randomizer;
import emat.helper.LinearSplineTreePriorApproximation;
import emat.helper.LearnedTreePrior;
import emat.helper.TreePriorApproximation;
import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertTrue;

public class LinearSplineTreePriorApproximationTest {

    /**
     * Generates observations from a quadratic in the offset, with one to five nodes moving per observation. Its
     * linear coefficient follows a narrow bump over the midpoint and is linear in a parameter θ. The slope fitted
     * with a linear spline must match it closely, and more closely than a cubic in the midpoint.
     */
    @Test
    public void testFollowsBumpOverMidpoint() {
        LearnedTreePrior spline = new LinearSplineTreePriorApproximation(2, 20, 1, 200, 1e-12);
        LearnedTreePrior cubic = new TreePriorApproximation(2, 3, 1, 200, 1e-12);
        this.observe(spline, 20000);
        this.observe(cubic, 20000);

        assertTrue(spline.isFitted());
        assertEquals(20000, spline.getNumObservations());

        double splineError = 0.0;
        double cubicError = 0.0;
        for (int i = 0; i <= 100; i++) {
            double midpoint = 0.01 + 0.28 * i / 100.0;
            double[] parameters = {150.0 + 0.5 * i};
            double expectedSlope = this.getLinearCoefficient(midpoint, parameters[0]);

            splineError = Math.max(splineError, Math.abs(spline.getSlope(midpoint, parameters) - expectedSlope));
            cubicError = Math.max(cubicError, Math.abs(cubic.getSlope(midpoint, parameters) - expectedSlope));
        }

        // the bump is 30 high
        assertTrue(splineError < 1.5, "error of the spline: " + splineError);
        assertTrue(cubicError > 4.0 * splineError, "error of the cubic: " + cubicError);
    }

    /** Outside the knots, the slope stays at its value at the nearest knot. */
    @Test
    public void testHoldsSlopeOutsideKnots() {
        LearnedTreePrior spline = new LinearSplineTreePriorApproximation(2, 20, 1, 200, 1e-12);
        this.observe(spline, 5000);

        double[] parameters = {170.0};
        assertEquals(spline.getSlope(0.3, parameters), spline.getSlope(5.0, parameters), 1e-9);
        assertEquals(spline.getSlope(0.0, parameters), spline.getSlope(-5.0, parameters), 1e-9);
    }

    /**
     * The slope is zero before the first fit, and it stays finite if all nodes share one midpoint, the parameter
     * never varies and some observations carry no information.
     */
    @Test
    public void testStaysFiniteWithDegenerateInput() {
        Randomizer.setSeed(3);
        LinearSplineTreePriorApproximation approximation = new LinearSplineTreePriorApproximation(3, 8, 1, 50, 1e-8);
        double[] parameters = {7.0};

        assertFalse(approximation.isFitted());
        assertEquals(0.0, approximation.getSlope(0.5, parameters), 0.0);

        for (int i = 0; i < 400; i++) {
            approximation.startObservation();
            double oldOffset = 0.1 * (Randomizer.nextDouble() - 0.5);
            double newOffset = i % 3 == 0 ? oldOffset : 0.1 * (Randomizer.nextDouble() - 0.5);
            approximation.addNode(0.5, parameters, oldOffset, newOffset);
            approximation.finishObservation(i % 7 == 0 ? Double.NEGATIVE_INFINITY : -2.0 * (newOffset - oldOffset));

            // an observation without nodes is dropped
            approximation.startObservation();
            approximation.finishObservation(1.0);
        }

        assertTrue(approximation.isFitted());
        assertEquals(-2.0, approximation.getSlope(0.5, parameters), 1e-3);
        assertTrue(Double.isFinite(approximation.getSlope(100.0, new double[]{-50.0})));
    }

    private void observe(LearnedTreePrior approximation, int numObservations) {
        Randomizer.setSeed(2);
        double[] parameters = new double[1];

        for (int i = 0; i < numObservations; i++) {
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
                        - 5.0 * (newOffset * newOffset - oldOffset * oldOffset);
            }

            approximation.finishObservation(target);
        }
    }

    private double getLinearCoefficient(double midpoint, double parameter) {
        double distance = (midpoint - 0.1) / 0.03;
        return 30.0 * Math.exp(-0.5 * distance * distance) + 0.1 * parameter;
    }

}
