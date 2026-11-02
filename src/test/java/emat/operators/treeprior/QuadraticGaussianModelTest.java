package emat.operators.treeprior;

import org.junit.jupiter.api.Test;

import java.util.Random;

import static org.junit.jupiter.api.Assertions.assertEquals;

public class QuadraticGaussianModelTest {

    private static final double NOISE = 0.3;

    @Test
    public void recoversQuadraticMeanAndNoise() {
        Random random = new Random(1);
        int n = 2000;
        double[][] conditions = new double[n][3];
        double[][] values = new double[n][2];
        for (int i = 0; i < n; i++) {
            for (int j = 0; j < 3; j++) {
                conditions[i][j] = random.nextGaussian();
            }
            values[i] = QuadraticGaussianModelTest.computeMean(conditions[i]);
            values[i][0] += NOISE * random.nextGaussian();
            values[i][1] += NOISE * random.nextGaussian();
        }

        QuadraticGaussianModel model = new QuadraticGaussianModel(3, 2);
        model.fit(conditions, values);

        double[] query = {0.5, -1.0, 0.7};
        double[] expected = QuadraticGaussianModelTest.computeMean(query);
        double[] mean = model.computeMean(query);
        assertEquals(expected[0], mean[0], 0.1);
        assertEquals(expected[1], mean[1], 0.1);

        double sumSquares = 0.0;
        for (int i = 0; i < 4000; i++) {
            double[] draw = model.sample(query, 1.0);
            sumSquares += (draw[0] - mean[0]) * (draw[0] - mean[0]);
        }
        assertEquals(NOISE, Math.sqrt(sumSquares / 4000), 0.04);
    }

    @Test
    public void logDensityIntegratesToOneInOneDimension() {
        Random random = new Random(2);
        double[][] conditions = new double[500][1];
        double[][] values = new double[500][1];
        for (int i = 0; i < 500; i++) {
            conditions[i][0] = random.nextGaussian();
            values[i][0] = conditions[i][0] + NOISE * random.nextGaussian();
        }
        QuadraticGaussianModel model = new QuadraticGaussianModel(1, 1);
        model.fit(conditions, values);

        double integral = 0.0;
        for (double x = -6.0; x < 6.0; x += 0.001) {
            integral += 0.001 * Math.exp(model.computeLogDensity(new double[]{0.2}, new double[]{x}, 1.0));
        }
        assertEquals(1.0, integral, 1.0E-3);
    }

    private static double[] computeMean(double[] c) {
        return new double[]{1.0 + c[0] - 0.5 * c[1] * c[1] + 0.4 * c[0] * c[2], -2.0 * c[2] + 0.3 * c[0] * c[0]};
    }

}
