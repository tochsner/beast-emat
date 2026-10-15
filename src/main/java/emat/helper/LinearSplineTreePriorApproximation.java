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
 * midpoint, const + c_1 x + ... + c_n x^n. Every coefficient is a linear spline in the midpoint, plus a linear
 * combination of the parameters of the tree prior: it is given by its values at a set of knots along the height
 * and interpolated linearly between them, which is a regression on one hat basis function per knot. The knots
 * are placed at quantiles of the midpoints, so that every interval between two knots holds a similar number of
 * nodes. The weights are fitted by ridge-penalised least squares to observed changes of the log tree prior when
 * one or several nodes move at once, which the model explains as the sum of the changes of the polynomials of
 * the moved nodes.
 *
 * <p>Compared with powers of the midpoint, the spline follows a slope that changes with the height in any shape,
 * and a midpoint outside the knots keeps the value at the nearest knot.
 *
 * <p>The knots and the scales of the offsets and of the parameters are fixed after a warm-up, and the weights
 * are refitted at growing intervals, so the approximation changes less and less as observations come in.
 */
public class LinearSplineTreePriorApproximation implements LearnedTreePrior {

    // the factor by which the number of observations grows between two fits
    private static final double REFIT_GROWTH = 1.5;

    private final int degree;
    private final int numKnots;
    private final int numParameters;
    private final int warmUp;
    private final double ridgePenalty;

    // per polynomial order, the features are the hat basis functions of the knots and the parameters
    private final int numFeatures;
    private final int numWeights;

    // the row of the design matrix for the current observation in scaled units, which is only filled after the warm-up
    private final double[] designRow;
    private boolean hasCurrentNodes;

    // the nodes of the current observation of the warm-up, as midpoint, old offset and new offset
    private final List<double[]> currentNodes = new ArrayList<>();
    private double[] currentParameters;

    // the observations of the warm-up
    private List<List<double[]>> warmUpNodes = new ArrayList<>();
    private List<double[]> warmUpParameters = new ArrayList<>();
    private List<Double> warmUpTargets = new ArrayList<>();

    // the fixed knots and scales: a scale of zero marks a parameter that did not vary
    private boolean hasKnotsAndScales;
    private double[] knots;
    private final double[] parameterMeans;
    private final double[] parameterScales;
    private double offsetScale;

    // the normal equations in scaled units: the Gram matrix XᵀX and the moment vector Xᵀy
    private final double[][] gramMatrix;
    private final double[] momentVector;
    private long numObservations;
    private long nextFit;

    // the weights of the linear coefficient c_1 for the knots and the scaled parameters
    private final double[] slopeWeights;

    // the weights of the quadratic coefficient c_2, which stay zero for a polynomial of degree one
    private final double[] curvatureWeights;
    private boolean isFitted;

    // the knot interval of the node at hand: the knot below its midpoint, and the weight of the knot above
    private int lowerKnotIndex;
    private double upperKnotWeight;

    /**
     * @param degree        the degree n of the polynomial in the offset
     * @param numKnots      the number of knots along the height
     * @param numParameters the number of parameters of the tree prior among the features
     * @param warmUp        the number of observations after which the knots are fixed and the first fit is made
     * @param ridgePenalty  the ridge penalty per observation, in scaled units
     */
    public LinearSplineTreePriorApproximation(int degree, int numKnots, int numParameters, int warmUp, double ridgePenalty) {
        if (degree < 1) {
            throw new IllegalArgumentException("The degree of the polynomial must be at least 1.");
        }
        if (numKnots < 1) {
            throw new IllegalArgumentException("There must be at least one knot.");
        }

        this.degree = degree;
        this.numKnots = numKnots;
        this.numParameters = numParameters;
        this.warmUp = Math.max(warmUp, 1);
        this.ridgePenalty = ridgePenalty;

        this.numFeatures = this.numKnots + this.numParameters;
        this.numWeights = this.degree * this.numFeatures;

        this.designRow = new double[this.numWeights];
        this.parameterMeans = new double[this.numParameters];
        this.parameterScales = new double[this.numParameters];
        this.gramMatrix = new double[this.numWeights][this.numWeights];
        this.momentVector = new double[this.numWeights];
        this.slopeWeights = new double[this.numFeatures];
        this.curvatureWeights = new double[this.numFeatures];
    }

    @Override
    public double getSlope(double midpoint, double[] parameters) {
        if (!this.isFitted) {
            return 0.0;
        }

        return this.evaluateCoefficient(this.slopeWeights, midpoint, parameters) / this.offsetScale;
    }

    @Override
    public double getCurvature(double midpoint, double[] parameters) {
        if (!this.isFitted) {
            return 0.0;
        }

        return this.evaluateCoefficient(this.curvatureWeights, midpoint, parameters) / (this.offsetScale * this.offsetScale);
    }

    /** Returns the coefficient with the given weights in scaled units at the midpoint and the parameters. */
    private double evaluateCoefficient(double[] weights, double midpoint, double[] parameters) {
        this.locateKnotInterval(midpoint);

        double coefficient = (1.0 - this.upperKnotWeight) * weights[this.lowerKnotIndex];
        if (this.upperKnotWeight > 0.0) {
            coefficient += this.upperKnotWeight * weights[this.lowerKnotIndex + 1];
        }
        for (int j = 0; j < this.numParameters; j++) {
            coefficient += weights[this.numKnots + j] * this.scaleParameter(j, parameters[j]);
        }
        return coefficient;
    }

    @Override
    public boolean isFitted() {
        return this.isFitted;
    }

    @Override
    public long getNumObservations() {
        return this.numObservations;
    }

    /* Observations */

    @Override
    public void startObservation() {
        this.discardObservation();
    }

    @Override
    public void addNode(double midpoint, double[] parameters, double oldOffset, double newOffset) {
        if (this.hasKnotsAndScales) {
            this.addToDesignRow(midpoint, parameters, oldOffset, newOffset);
        } else {
            // the knots are not known yet, so the node is kept as it is
            this.currentNodes.add(new double[]{midpoint, oldOffset, newOffset});
            this.currentParameters = parameters.clone();
        }

        this.hasCurrentNodes = true;
    }

    @Override
    public void finishObservation(double target) {
        if (!this.hasCurrentNodes || !Double.isFinite(target)) {
            this.discardObservation();
            return;
        }

        if (!this.hasKnotsAndScales) {
            this.warmUpNodes.add(new ArrayList<>(this.currentNodes));
            this.warmUpParameters.add(this.currentParameters);
            this.warmUpTargets.add(target);

            if (this.warmUpTargets.size() >= this.warmUp) {
                this.fixKnotsAndScales();
                this.replayWarmUp();
                this.fit();
            }
        } else {
            this.accumulateNormalEquations(target);
            if (this.numObservations >= this.nextFit) {
                this.fit();
            }
        }

        this.discardObservation();
    }

    @Override
    public void discardObservation() {
        Arrays.fill(this.designRow, 0.0);
        this.currentNodes.clear();
        this.hasCurrentNodes = false;
    }

    /* Features */

    /**
     * Adds a node to the design row of the current observation. The node contributes feature · (newOffset^i -
     * oldOffset^i) to the factor of every weight, with the offsets in units of their scale. Only the two knots
     * around the midpoint have a basis function that is not zero.
     */
    private void addToDesignRow(double midpoint, double[] parameters, double oldOffset, double newOffset) {
        this.locateKnotInterval(midpoint);

        double oldPower = 1.0;
        double newPower = 1.0;

        for (int order = 0; order < this.degree; order++) {
            oldPower *= oldOffset / this.offsetScale;
            newPower *= newOffset / this.offsetScale;
            double powerDifference = newPower - oldPower;

            int block = order * this.numFeatures;
            this.designRow[block + this.lowerKnotIndex] += (1.0 - this.upperKnotWeight) * powerDifference;
            if (this.upperKnotWeight > 0.0) {
                this.designRow[block + this.lowerKnotIndex + 1] += this.upperKnotWeight * powerDifference;
            }
            for (int j = 0; j < this.numParameters; j++) {
                this.designRow[block + this.numKnots + j] += this.scaleParameter(j, parameters[j]) * powerDifference;
            }
        }
    }

    /**
     * Finds the knot interval of the given midpoint: the knot below it, and how far it lies towards the knot above. A
     * midpoint outside the knots is held at the nearest knot.
     */
    private void locateKnotInterval(double midpoint) {
        int last = this.knots.length - 1;

        if (midpoint <= this.knots[0] || last == 0) {
            this.lowerKnotIndex = 0;
            this.upperKnotWeight = 0.0;
            return;
        }
        if (midpoint >= this.knots[last]) {
            this.lowerKnotIndex = last;
            this.upperKnotWeight = 0.0;
            return;
        }

        // binary search for the last knot that is not above the midpoint
        int low = 0;
        int high = last;
        while (high - low > 1) {
            int middle = (low + high) >>> 1;
            if (this.knots[middle] <= midpoint) {
                low = middle;
            } else {
                high = middle;
            }
        }

        this.lowerKnotIndex = low;
        this.upperKnotWeight = (midpoint - this.knots[low]) / (this.knots[high] - this.knots[low]);
    }

    private double scaleParameter(int index, double parameter) {
        return this.parameterScales[index] > 0.0
                ? (parameter - this.parameterMeans[index]) / this.parameterScales[index]
                : 0.0;
    }

    /* Scales */

    /**
     * Fixes the knots and the scales from the observations of the warm-up. The knots are distinct quantiles of
     * the midpoints of all nodes. Every parameter is centred on its mean and divided by its standard deviation
     * over the observations, and the offsets are divided by their root mean square. A parameter that did not
     * vary is left out of the fit, as it cannot be told apart from the values at the knots.
     */
    private void fixKnotsAndScales() {
        int numNodes = 0;
        for (List<double[]> nodes : this.warmUpNodes) {
            numNodes += nodes.size();
        }

        double[] midpoints = new double[numNodes];
        double offsetSquareSum = 0.0;
        int index = 0;
        for (List<double[]> nodes : this.warmUpNodes) {
            for (double[] node : nodes) {
                midpoints[index++] = node[0];
                offsetSquareSum += (node[1] * node[1] + node[2] * node[2]) / 2.0;
            }
        }
        Arrays.sort(midpoints);

        // knots at evenly spaced quantiles, keeping only the ones that differ

        double[] knots = new double[this.numKnots];
        int numDistinct = 0;
        for (int k = 0; k < this.numKnots; k++) {
            double quantile = this.numKnots > 1 ? (double) k / (this.numKnots - 1) : 0.5;
            double knot = midpoints[(int) Math.round(quantile * (numNodes - 1))];
            if (numDistinct == 0 || knot > knots[numDistinct - 1]) {
                knots[numDistinct++] = knot;
            }
        }
        this.knots = Arrays.copyOf(knots, numDistinct);

        int numObservations = this.warmUpParameters.size();
        for (int j = 0; j < this.numParameters; j++) {
            double sum = 0.0;
            double squareSum = 0.0;
            for (double[] parameters : this.warmUpParameters) {
                sum += parameters[j];
                squareSum += parameters[j] * parameters[j];
            }

            double mean = sum / numObservations;
            double scale = Math.sqrt(Math.max(squareSum / numObservations - mean * mean, 0.0));
            this.parameterMeans[j] = mean;
            this.parameterScales[j] = scale > 1e-9 * Math.max(Math.abs(mean), 1e-300) ? scale : 0.0;
        }

        double offsetScale = Math.sqrt(offsetSquareSum / numNodes);
        this.offsetScale = offsetScale > 0.0 ? offsetScale : 1.0;

        this.hasKnotsAndScales = true;
    }

    /** Adds the observations of the warm-up to the normal equations, now that the knots and scales are known. */
    private void replayWarmUp() {
        for (int i = 0; i < this.warmUpTargets.size(); i++) {
            Arrays.fill(this.designRow, 0.0);
            for (double[] node : this.warmUpNodes.get(i)) {
                this.addToDesignRow(node[0], this.warmUpParameters.get(i), node[1], node[2]);
            }
            this.accumulateNormalEquations(this.warmUpTargets.get(i));
        }

        this.warmUpNodes = null;
        this.warmUpParameters = null;
        this.warmUpTargets = null;
    }

    /* Fitting */

    /** Adds the design row of the current observation to the normal equations. */
    private void accumulateNormalEquations(double target) {
        for (int a = 0; a < this.numWeights; a++) {
            if (this.designRow[a] == 0.0) {
                continue;
            }
            for (int b = 0; b < this.numWeights; b++) {
                this.gramMatrix[a][b] += this.designRow[a] * this.designRow[b];
            }
            this.momentVector[a] += this.designRow[a] * target;
        }

        this.numObservations++;
    }

    /**
     * Solves the normal equations with a ridge penalty for the weights, and keeps the weights of the linear
     * coefficient. A fit whose weights are not finite is ignored.
     */
    private void fit() {
        this.nextFit = (long) Math.ceil(this.numObservations * REFIT_GROWTH);

        RealMatrix matrix = new Array2DRowRealMatrix(this.gramMatrix, true);
        double penalty = this.ridgePenalty * this.numObservations;
        for (int a = 0; a < this.numWeights; a++) {
            matrix.addToEntry(a, a, penalty);
        }
        RealVector rightHandSide = new ArrayRealVector(this.momentVector, true);

        double[] weights;
        try {
            weights = new CholeskyDecomposition(matrix).getSolver().solve(rightHandSide).toArray();
        } catch (RuntimeException exception) {
            // the matrix is not positive definite to working precision, so fall back to the pseudo-inverse
            weights = new SingularValueDecomposition(matrix).getSolver().solve(rightHandSide).toArray();
        }

        for (double weight : weights) {
            if (!Double.isFinite(weight)) {
                return;
            }
        }

        // the weights of the linear and the quadratic coefficient are the first two blocks
        System.arraycopy(weights, 0, this.slopeWeights, 0, this.numFeatures);
        if (this.degree >= 2) {
            System.arraycopy(weights, this.numFeatures, this.curvatureWeights, 0, this.numFeatures);
        }
        this.isFitted = true;
    }

}
