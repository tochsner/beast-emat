package emat.operators.mutations;

import static org.junit.jupiter.api.Assertions.assertEquals;

/**
 * Estimates the mean of an autocorrelated series of MCMC values and the standard error of
 * this estimate by batch means: the series is cut into consecutive batches, which are long
 * enough to be nearly independent, and the standard error follows from the spread of the
 * batch means. The series must be exactly as long as the given number of values.
 */
class BatchMeans {

    // the number of standard errors by which an estimate may deviate from its expected value
    private static final double NUM_STANDARD_ERRORS = 5.0;

    // the smallest tolerance, for values that are (almost) constant, whose standard error is (almost) zero
    private static final double MIN_TOLERANCE = 1e-3;

    private final int batchLength;
    private final double[] batchSums;

    private int numValues = 0;

    BatchMeans(int numValues, int numBatches) {
        if (numValues % numBatches != 0) {
            throw new IllegalArgumentException("The number of values must be a multiple of the number of batches.");
        }
        this.batchLength = numValues / numBatches;
        this.batchSums = new double[numBatches];
    }

    void add(double value) {
        this.batchSums[this.numValues / this.batchLength] += value;
        this.numValues++;
    }

    /** Asserts that the mean is within a few standard errors of the expected value. */
    void assertMean(double expected, String message) {
        double tolerance = Math.max(NUM_STANDARD_ERRORS * this.computeStandardError(), MIN_TOLERANCE);
        assertEquals(expected, this.computeMean(), tolerance, message);
    }

    double computeMean() {
        double sum = 0.0;
        for (double batchSum : this.batchSums) {
            sum += batchSum;
        }
        return sum / (this.batchLength * this.batchSums.length);
    }

    /** Computes the standard error of the mean from the sample variance of the batch means. */
    double computeStandardError() {
        int numBatches = this.batchSums.length;
        double mean = this.computeMean();

        double sumSquares = 0.0;
        for (double batchSum : this.batchSums) {
            double deviation = batchSum / this.batchLength - mean;
            sumSquares += deviation * deviation;
        }
        return Math.sqrt(sumSquares / (numBatches - 1) / numBatches);
    }

}
