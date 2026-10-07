package emat.helper;

import org.apache.commons.math4.legacy.linear.Array2DRowRealMatrix;
import org.apache.commons.math4.legacy.linear.ArrayRealVector;
import org.apache.commons.math4.legacy.linear.CholeskyDecomposition;
import org.apache.commons.math4.legacy.linear.RealMatrix;
import org.apache.commons.math4.legacy.linear.RealVector;
import org.apache.commons.math4.legacy.linear.SingularValueDecomposition;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

/**
 * Learns a local approximation of the log tree prior as a function of the height of a single node. Around the
 * midpoint m of the range of the node, the log density is modelled by a polynomial in the offset x from the
 * midpoint, const + c_1 x + ... + c_n x^n. Every coefficient is a linear combination of the features of the
 * node: a constant, the powers m, m^2, ..., m^d of the midpoint, and the parameters of the tree prior. The
 * weights are fitted by least squares to observed changes of the log tree prior when one or several nodes move
 * at once, which the model explains as the sum of the changes of the polynomials of the moved nodes.
 *
 * <p>The offsets, the midpoints and the parameters are rescaled with scales that are fixed after a warm-up, so
 * that the fit is well conditioned whatever the units of the tree and of the parameters. The weights are
 * refitted at growing intervals, so the approximation changes less and less as observations come in.
 */
public class TreePriorApproximation {

    // the factor by which the number of observations grows between two fits
    private static final double REFIT_GROWTH = 1.5;

    private final int degree;
    private final int midpointDegree;
    private final int numParameters;
    private final int warmUp;
    private final double ridge;

    // per polynomial order, the features are the constant, the powers of the midpoint and the parameters
    private final int numFeatures;
    private final int numUnknowns;

    // the row of the current observation, in unscaled units during the warm-up and in scaled units after it
    private final double[] currentRow;
    private boolean hasCurrentNodes;

    // the unscaled observations of the warm-up
    private List<double[]> warmUpRows = new ArrayList<>();
    private List<Double> warmUpTargets = new ArrayList<>();

    // statistics over the nodes of the warm-up, from which the scales are derived
    private long numWarmUpNodes;
    private double midpointSum;
    private double midpointSquareSum;
    private double minMidpoint = Double.POSITIVE_INFINITY;
    private double maxMidpoint = Double.NEGATIVE_INFINITY;
    private final double[] parameterSums;
    private final double[] parameterSquareSums;
    private double offsetSquareSum;

    // the fixed scales: a scale of zero marks a midpoint or a parameter that did not vary
    private boolean hasScales;
    private double midpointMean;
    private double midpointScale;
    private double minScaledMidpoint;
    private double maxScaledMidpoint;
    private final double[] parameterMeans;
    private final double[] parameterScales;
    private double offsetScale;

    // the normal equations in scaled units
    private final double[][] xtx;
    private final double[] xty;
    private long numObservations;
    private long nextFit;

    // the weights of the linear coefficient c_1 for the scaled features
    private final double[] slopeWeights;
    private boolean isFitted;

    // the features of the node at hand, unscaled during the warm-up and scaled after it
    private final double[] nodeFeatures;

    /**
     * @param degree         the degree n of the polynomial in the offset
     * @param midpointDegree the highest power d of the midpoint among the features
     * @param numParameters  the number of parameters of the tree prior among the features
     * @param warmUp         the number of observations after which the scales are fixed and the first fit is made
     * @param ridge          the ridge penalty per observation, in scaled units
     */
    public TreePriorApproximation(int degree, int midpointDegree, int numParameters, int warmUp, double ridge) {
        if (degree < 1) {
            throw new IllegalArgumentException("The degree of the polynomial must be at least 1.");
        }
        if (midpointDegree < 0) {
            throw new IllegalArgumentException("The degree of the midpoint must not be negative.");
        }

        this.degree = degree;
        this.midpointDegree = midpointDegree;
        this.numParameters = numParameters;
        this.warmUp = Math.max(warmUp, 1);
        this.ridge = ridge;

        this.numFeatures = 1 + this.midpointDegree + this.numParameters;
        this.numUnknowns = this.degree * this.numFeatures;

        this.currentRow = new double[this.numUnknowns];
        this.parameterSums = new double[this.numParameters];
        this.parameterSquareSums = new double[this.numParameters];
        this.parameterMeans = new double[this.numParameters];
        this.parameterScales = new double[this.numParameters];
        this.xtx = new double[this.numUnknowns][this.numUnknowns];
        this.xty = new double[this.numUnknowns];
        this.slopeWeights = new double[this.numFeatures];
        this.nodeFeatures = new double[this.numFeatures];
    }

    /**
     * Returns the slope c_1 of the log tree prior at the given midpoint of the range of a node, for the given
     * values of the parameters of the tree prior. The slope is zero until the first fit.
     */
    public double getSlope(double midpoint, double[] parameters) {
        if (!this.isFitted) {
            return 0.0;
        }

        this.computeScaledFeatures(midpoint, parameters);

        double slope = 0.0;
        for (int j = 0; j < this.numFeatures; j++) {
            slope += this.slopeWeights[j] * this.nodeFeatures[j];
        }
        return slope / this.offsetScale;
    }

    public boolean isFitted() {
        return this.isFitted;
    }

    public long getNumObservations() {
        return this.numObservations;
    }

    /* Observations */

    /** Starts a new observation, which drops the nodes of an observation that was not finished. */
    public void startObservation() {
        this.discardObservation();
    }

    /**
     * Adds a node to the current observation that moved from the old to the new offset from the midpoint of its
     * range. The node contributes feature · (newOffset^i - oldOffset^i) to the factor of every weight.
     */
    public void addNode(double midpoint, double[] parameters, double oldOffset, double newOffset) {
        if (this.hasScales) {
            this.computeScaledFeatures(midpoint, parameters);
        } else {
            this.computeUnscaledFeatures(midpoint, parameters);
            this.collectWarmUpStatistics(midpoint, parameters, oldOffset, newOffset);
        }

        // after the warm-up the offsets are measured in units of their scale
        double scale = this.hasScales ? this.offsetScale : 1.0;
        double oldPower = 1.0;
        double newPower = 1.0;

        for (int order = 0; order < this.degree; order++) {
            oldPower *= oldOffset / scale;
            newPower *= newOffset / scale;
            double powerDifference = newPower - oldPower;

            int block = order * this.numFeatures;
            for (int j = 0; j < this.numFeatures; j++) {
                this.currentRow[block + j] += this.nodeFeatures[j] * powerDifference;
            }
        }

        this.hasCurrentNodes = true;
    }

    /**
     * Finishes the current observation with the observed change of the log tree prior, and refits the weights
     * if enough observations have come in since the last fit. An observation without nodes or with a target that
     * is not finite is dropped.
     */
    public void finishObservation(double target) {
        if (!this.hasCurrentNodes || !Double.isFinite(target)) {
            this.discardObservation();
            return;
        }

        if (!this.hasScales) {
            this.warmUpRows.add(this.currentRow.clone());
            this.warmUpTargets.add(target);

            if (this.warmUpRows.size() >= this.warmUp) {
                this.fixScales();
                for (int i = 0; i < this.warmUpRows.size(); i++) {
                    this.accumulate(this.scaleWarmUpRow(this.warmUpRows.get(i)), this.warmUpTargets.get(i));
                }
                this.warmUpRows = null;
                this.warmUpTargets = null;

                this.fit();
            }
        } else {
            this.accumulate(this.currentRow, target);
            if (this.numObservations >= this.nextFit) {
                this.fit();
            }
        }

        this.discardObservation();
    }

    /** Drops the current observation. */
    public void discardObservation() {
        Arrays.fill(this.currentRow, 0.0);
        this.hasCurrentNodes = false;
    }

    /* Features */

    /** Writes the constant, the powers of the midpoint and the parameters as they are into the features. */
    private void computeUnscaledFeatures(double midpoint, double[] parameters) {
        this.nodeFeatures[0] = 1.0;
        for (int power = 1; power <= this.midpointDegree; power++) {
            this.nodeFeatures[power] = this.nodeFeatures[power - 1] * midpoint;
        }
        for (int j = 0; j < this.numParameters; j++) {
            this.nodeFeatures[1 + this.midpointDegree + j] = parameters[j];
        }
    }

    /**
     * Writes the constant, the powers of the centred and scaled midpoint, and the centred and scaled parameters
     * into the features. The midpoint is held at the range seen during the warm-up, as a polynomial runs away
     * outside the range it was fitted on.
     */
    private void computeScaledFeatures(double midpoint, double[] parameters) {
        double scaledMidpoint = 0.0;
        if (this.midpointScale > 0.0) {
            scaledMidpoint = (midpoint - this.midpointMean) / this.midpointScale;
            scaledMidpoint = Math.max(this.minScaledMidpoint, Math.min(scaledMidpoint, this.maxScaledMidpoint));
        }

        this.nodeFeatures[0] = 1.0;
        for (int power = 1; power <= this.midpointDegree; power++) {
            this.nodeFeatures[power] = this.nodeFeatures[power - 1] * scaledMidpoint;
        }
        for (int j = 0; j < this.numParameters; j++) {
            this.nodeFeatures[1 + this.midpointDegree + j] = this.parameterScales[j] > 0.0
                    ? (parameters[j] - this.parameterMeans[j]) / this.parameterScales[j]
                    : 0.0;
        }
    }

    private void collectWarmUpStatistics(double midpoint, double[] parameters, double oldOffset, double newOffset) {
        this.numWarmUpNodes++;

        this.midpointSum += midpoint;
        this.midpointSquareSum += midpoint * midpoint;
        this.minMidpoint = Math.min(this.minMidpoint, midpoint);
        this.maxMidpoint = Math.max(this.maxMidpoint, midpoint);

        for (int j = 0; j < this.numParameters; j++) {
            this.parameterSums[j] += parameters[j];
            this.parameterSquareSums[j] += parameters[j] * parameters[j];
        }

        this.offsetSquareSum += (oldOffset * oldOffset + newOffset * newOffset) / 2.0;
    }

    /* Scales */

    /**
     * Fixes the scales from the nodes of the warm-up: the midpoint and every parameter are centred on their mean
     * and divided by their standard deviation, and the offsets are divided by their root mean square. A midpoint
     * or a parameter that did not vary is left out of the fit, as it cannot be told apart from the constant.
     */
    private void fixScales() {
        this.midpointMean = this.midpointSum / this.numWarmUpNodes;
        this.midpointScale = this.computeScale(this.midpointMean, this.midpointSquareSum / this.numWarmUpNodes);
        if (this.midpointScale > 0.0) {
            this.minScaledMidpoint = (this.minMidpoint - this.midpointMean) / this.midpointScale;
            this.maxScaledMidpoint = (this.maxMidpoint - this.midpointMean) / this.midpointScale;
        }

        for (int j = 0; j < this.numParameters; j++) {
            this.parameterMeans[j] = this.parameterSums[j] / this.numWarmUpNodes;
            this.parameterScales[j] = this.computeScale(this.parameterMeans[j], this.parameterSquareSums[j] / this.numWarmUpNodes);
        }

        double offsetScale = Math.sqrt(this.offsetSquareSum / this.numWarmUpNodes);
        this.offsetScale = offsetScale > 0.0 ? offsetScale : 1.0;

        this.hasScales = true;
    }

    /** Returns the standard deviation for the given mean and mean square, or zero if it is negligible. */
    private double computeScale(double mean, double meanSquare) {
        double scale = Math.sqrt(Math.max(meanSquare - mean * mean, 0.0));
        return scale > 1e-9 * Math.max(Math.abs(mean), 1e-300) ? scale : 0.0;
    }

    /**
     * Converts an unscaled row of the warm-up to scaled units. The row holds sums over the nodes of unscaled
     * features times power differences of the offsets. The powers of the centred and scaled midpoint follow from
     * the unscaled powers by the binomial theorem, ((m - μ) / σ)^p = σ^-p Σ_q C(p, q) (-μ)^(p - q) m^q, and a
     * centred parameter from the parameter and the constant.
     */
    private double[] scaleWarmUpRow(double[] row) {
        double[] scaledRow = new double[this.numUnknowns];

        double powerScale = 1.0;
        for (int order = 0; order < this.degree; order++) {
            powerScale *= this.offsetScale;
            int block = order * this.numFeatures;

            double constant = row[block];
            scaledRow[block] = constant / powerScale;

            if (this.midpointScale > 0.0) {
                for (int power = 1; power <= this.midpointDegree; power++) {
                    double sum = 0.0;
                    double binomial = 1.0;
                    for (int q = 0; q <= power; q++) {
                        sum += binomial * Math.pow(-this.midpointMean, power - q) * row[block + q];
                        binomial = binomial * (power - q) / (q + 1);
                    }
                    scaledRow[block + power] = sum / (Math.pow(this.midpointScale, power) * powerScale);
                }
            }

            for (int j = 0; j < this.numParameters; j++) {
                int index = block + 1 + this.midpointDegree + j;
                if (this.parameterScales[j] > 0.0) {
                    scaledRow[index] = (row[index] - this.parameterMeans[j] * constant) / (this.parameterScales[j] * powerScale);
                }
            }
        }

        return scaledRow;
    }

    /* Fitting */

    /** Adds a scaled observation to the normal equations. */
    private void accumulate(double[] scaledRow, double target) {
        for (int a = 0; a < this.numUnknowns; a++) {
            if (scaledRow[a] == 0.0) {
                continue;
            }
            for (int b = 0; b < this.numUnknowns; b++) {
                this.xtx[a][b] += scaledRow[a] * scaledRow[b];
            }
            this.xty[a] += scaledRow[a] * target;
        }

        this.numObservations++;
    }

    /**
     * Solves the normal equations with a ridge penalty for the weights, and keeps the weights of the linear
     * coefficient. A fit whose weights are not finite is ignored.
     */
    private void fit() {
        this.nextFit = (long) Math.ceil(this.numObservations * REFIT_GROWTH);

        RealMatrix matrix = new Array2DRowRealMatrix(this.xtx, true);
        double penalty = this.ridge * this.numObservations;
        for (int a = 0; a < this.numUnknowns; a++) {
            matrix.addToEntry(a, a, penalty);
        }
        RealVector rhs = new ArrayRealVector(this.xty, true);

        double[] weights;
        try {
            weights = new CholeskyDecomposition(matrix).getSolver().solve(rhs).toArray();
        } catch (RuntimeException exception) {
            // the matrix is not positive definite to working precision, so fall back to the pseudo-inverse
            weights = new SingularValueDecomposition(matrix).getSolver().solve(rhs).toArray();
        }

        for (double weight : weights) {
            if (!Double.isFinite(weight)) {
                return;
            }
        }

        // the weights of the linear coefficient are the first block
        System.arraycopy(weights, 0, this.slopeWeights, 0, this.numFeatures);
        this.isFitted = true;
    }

}
