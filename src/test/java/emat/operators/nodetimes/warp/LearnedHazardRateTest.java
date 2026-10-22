package emat.operators.nodetimes.warp;

import org.junit.jupiter.api.Test;

import java.util.Arrays;
import java.util.Random;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertTrue;

public class LearnedHazardRateTest {

    private static final double LOG_POP_SIZE = Math.log(10.0);
    private static final double GROWTH_RATE = 0.1;

    private LearnedHazardRate createFittedRate(int numTrees, TreeHazard hazard) {
        Random random = new Random(7);
        LearnedHazardRate rate = new LearnedHazardRate(hazard, 12, 2, 1000);
        for (int i = 0; i < numTrees; i++) {
            double logPopSize = LOG_POP_SIZE + 0.3 * random.nextGaussian();
            double growthRate = GROWTH_RATE + 0.05 * random.nextGaussian();
            CoalescentSimulation tree = new CoalescentSimulation(random, 60, Math.exp(logPopSize), growthRate);
            rate.addObservation(tree.tipHeights, tree.internalHeights, new double[]{logPopSize, growthRate}, tree.logDensity);
        }
        rate.fit();
        return rate;
    }

    /** The time change is the integral of the rate, and its height function is the inverse. */
    @Test
    public void testTimeChangeInvertsAndIntegrates() {
        HazardTimeChange change = new HazardTimeChange(new double[]{0.0, 1.0, 2.5, 4.0}, new double[]{-1.0, -0.5, -1.2, -1.5});

        assertTrue(change.isValid());
        for (double height : new double[]{0.0, 0.3, 1.0, 1.7, 2.5, 3.9, 6.0}) {
            assertEquals(height, change.computeHeight(change.computeTime(height)), 1e-9);
        }

        // a numerical integral of the rate agrees with the closed form
        double integral = 0.0;
        int steps = 20000;
        for (int i = 0; i < steps; i++) {
            integral += Math.exp(change.computeLogRate((i + 0.5) * 3.0 / steps)) * 3.0 / steps;
        }
        assertEquals(integral, change.computeTime(3.0), 1e-6);
    }

    /** The units at risk of the two hazards. */
    @Test
    public void testMultiplicities() {
        assertEquals(10.0, TreeHazard.COALESCENT.computeExposureMultiplicity(5));
        assertEquals(10.0, TreeHazard.COALESCENT.computeEventMultiplicity(5));
        assertEquals(5.0, TreeHazard.BIRTH_DEATH.computeExposureMultiplicity(5));
        assertEquals(4.0, TreeHazard.BIRTH_DEATH.computeEventMultiplicity(5));
    }

    /** Fitted to exact coalescent log densities, the rate recovers the logarithm of the inverse population size and its growth. */
    @Test
    public void testRecoversExponentialGrowthRate() {
        LearnedHazardRate rate = this.createFittedRate(400, TreeHazard.COALESCENT);

        assertTrue(rate.isFitted());
        assertTrue(rate.getRmsResidual() < 0.5, "rms " + rate.getRmsResidual());

        HazardTimeChange center = rate.createTimeChange(new double[]{LOG_POP_SIZE, GROWTH_RATE});
        HazardTimeChange largerPopulation = rate.createTimeChange(new double[]{LOG_POP_SIZE + 0.3, GROWTH_RATE});
        HazardTimeChange fasterGrowth = rate.createTimeChange(new double[]{LOG_POP_SIZE, GROWTH_RATE + 0.05});

        Random random = new Random(11);
        double[] heights = new CoalescentSimulation(random, 60, 10.0, GROWTH_RATE).internalHeights;
        Arrays.sort(heights);
        for (double fraction : new double[]{0.2, 0.5, 0.8}) {
            double height = heights[(int) (fraction * (heights.length - 1))];
            assertEquals(GROWTH_RATE * height - LOG_POP_SIZE, center.computeLogRate(height), 0.3);
            assertEquals(-0.3, largerPopulation.computeLogRate(height) - center.computeLogRate(height), 0.15);
            assertEquals(0.05 * height, fasterGrowth.computeLogRate(height) - center.computeLogRate(height), 0.15);
        }
    }

    /** A fit that is given the wrong units at risk leaves a visibly larger residual. */
    @Test
    public void testWrongHazardFitsWorse() {
        double right = this.createFittedRate(400, TreeHazard.COALESCENT).getRmsResidual();
        double wrong = this.createFittedRate(400, TreeHazard.BIRTH_DEATH).getRmsResidual();

        assertTrue(wrong > 3.0 * right, "right " + right + " wrong " + wrong);
    }

    /** Nothing is fitted from too few observations or from log densities that are not finite. */
    @Test
    public void testIgnoresDegenerateObservations() {
        LearnedHazardRate rate = new LearnedHazardRate(TreeHazard.COALESCENT, 8, 2, 100);
        rate.addObservation(new double[]{0, 0, 0}, new double[]{0.1, 0.2}, new double[]{0.0, 0.0}, Double.NaN);
        rate.fit();

        assertFalse(rate.isFitted());
        assertEquals(1, rate.getNumObserved());
    }

    /** AUTO chooses the coalescent hazard for coalescent log densities. */
    @Test
    public void testAutoChoosesCoalescent() {
        LearnedHazardRate rate = this.createFittedRate(400, TreeHazard.AUTO);

        assertEquals(TreeHazard.COALESCENT, rate.getResolvedHazard());
        assertTrue(rate.getCandidateRmsResiduals()[0] < rate.getCandidateRmsResiduals()[1]);
    }

    /** AUTO chooses the birth-death hazard for log densities that are those of a rate per lineage. */
    @Test
    public void testAutoChoosesBirthDeath() {
        Random random = new Random(9);
        LearnedHazardRate rate = new LearnedHazardRate(TreeHazard.AUTO, 12, 2, 1000);
        for (int i = 0; i < 400; i++) {
            double logScale = 0.3 * random.nextGaussian();
            double slope = 0.05 * random.nextGaussian();
            CoalescentSimulation tree = new CoalescentSimulation(random, 60, 10.0, 0.1);
            rate.addObservation(tree.tipHeights, tree.internalHeights, new double[]{logScale, slope}, this.computeBirthDeathLogDensity(tree, logScale, slope));
        }
        rate.fit();

        assertEquals(TreeHazard.BIRTH_DEATH, rate.getResolvedHazard());
        assertTrue(rate.getCandidateRmsResiduals()[1] < 0.5, "rms " + rate.getCandidateRmsResiduals()[1]);
        assertTrue(rate.getCandidateRmsResiduals()[0] > 3.0 * rate.getCandidateRmsResiduals()[1]);
    }

    /**
     * Returns the log density of a tree whose lineages each branch at the rate exp(logScale + slope h) / 2, which has
     * the number of lineages below an event minus one at risk at an event.
     */
    private double computeBirthDeathLogDensity(CoalescentSimulation tree, double logScale, double slope) {
        double[] heights = tree.internalHeights.clone();
        Arrays.sort(heights);

        double logDensity = 0.0;
        double previous = 0.0;
        for (int i = 0; i < heights.length; i++) {
            int lineagesBelow = heights.length + 1 - i;
            double height = heights[i];
            double integral = Math.exp(logScale - Math.log(2.0)) * Math.expm1(slope * height) / slope
                    - Math.exp(logScale - Math.log(2.0)) * Math.expm1(slope * previous) / slope;
            logDensity += Math.log(lineagesBelow - 1) + logScale - Math.log(2.0) + slope * height - lineagesBelow * integral;
            previous = height;
        }
        return logDensity;
    }

}
