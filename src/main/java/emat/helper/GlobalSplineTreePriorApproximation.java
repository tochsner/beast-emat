package emat.helper;

import beast.base.util.Randomizer;
import org.apache.commons.math4.legacy.linear.Array2DRowRealMatrix;
import org.apache.commons.math4.legacy.linear.ArrayRealVector;
import org.apache.commons.math4.legacy.linear.CholeskyDecomposition;
import org.apache.commons.math4.legacy.linear.RealMatrix;
import org.apache.commons.math4.legacy.linear.RealVector;
import org.apache.commons.math4.legacy.linear.SingularValueDecomposition;

import java.util.ArrayDeque;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

/**
 * Learns a single global approximation of the log tree prior as a function of the height of a node. The slope
 * of the approximation is a linear spline in the absolute height: it is given by its values at a set of knots
 * and interpolated linearly between them, and constant beyond the first and the last knot. The value at every
 * knot is a linear combination of a constant and the parameters of the tree prior. The log tree prior is the
 * integral of the slope, a(h), so that a node that moves from one height to another changes the log tree prior
 * by a(new) - a(old).
 *
 * <p>In contrast to {@link LinearSplineTreePriorApproximation}, nothing is expanded around the midpoint of the
 * range of a node: a move is explained by the integral of the one shared slope curve over the heights it spans,
 * whatever the range, and the curvature is the change of the slope between the knots. The weights are fitted by
 * ridge-penalised least squares to observed changes of the log tree prior when one or several nodes move at
 * once, which the model explains as the sum of the changes of a over the moved nodes. The integrals of the hat
 * functions of the knots are known in closed form, so the model is linear in the weights.
 *
 * <p>The knots and the scales of the heights and of the parameters are derived from, and the weights are fitted
 * to, a window of the most recent observations. The tree changes a lot during the burn-in, so that older
 * observations describe a different log tree prior. The window holds the most recent half of all observations,
 * but at least a minimum number, and the weights are refitted after a tenth of all observations, but at least a
 * minimum number. So the approximation follows the early drift of the chain, and every refit changes it less and
 * less, which keeps the adaptation of the proposals diminishing. At most a fixed number of observations is kept
 * for the window: when it is full, every second one of them is dropped and only every second new one is kept.
 */
public class GlobalSplineTreePriorApproximation implements LearnedTreePrior {

    // the defaults for the least number of observations the fit looks back on and the least between two fits
    public static final int DEFAULT_MIN_WINDOW = 1000;
    public static final int DEFAULT_MIN_REFIT_INTERVAL = 5_000;

    // the window is the most recent part of all observations of this fraction, the interval between fits that of all
    private static final double WINDOW_FRACTION = 0.5;
    private static final double REFIT_FRACTION = 0.1;

    // the largest number of observations that are kept
    private static final int MAX_STORED_OBSERVATIONS = 5000;

    // the exponent changes by about this much over a piece of the envelope, unless there are too many pieces
    private static final double MAX_EXPONENT_CHANGE_PER_PIECE = 0.25;
    private static final int MAX_PIECES_PER_KNOT_INTERVAL = 64;

    private final int numKnots;
    private final int numParameters;
    private final int warmUp;
    private final double ridgePenalty;
    private final int minWindow;
    private final int minRefitInterval;
    private final int maxStoredObservations;

    // per knot, the features are the constant and the parameters
    private final int numFeatures;
    private final int numWeights;

    // the nodes of the current observation as old and new height, and the parameters they were moved under
    private final List<double[]> currentNodes = new ArrayList<>();
    private double[] currentParameters;

    /** A finished observation: its number, the old and new height of every moved node, the parameters and the target. */
    private record Observation(long number, double[][] nodes, double[] parameters, double target) {
    }

    // the observations of the window that are kept, oldest first, and the distance between the kept ones
    private final ArrayDeque<Observation> recentObservations = new ArrayDeque<>();
    private long keepEvery = 1;
    private long numObservationsSinceFit;

    // the knots and the scales of the last fit: a scale of zero marks a parameter that did not vary
    private double[] knots;
    private double[][] knotHatIntegrals;
    private final double[] parameterMeans;
    private final double[] parameterScales;
    private double heightScale;

    // the normal equations in scaled units: the Gram matrix XᵀX and the moment vector Xᵀy
    private final double[][] gramMatrix;
    private final double[] momentVector;
    private long numObservations;

    // the weights of the slope at the knots, as the constant and the coefficients of the scaled parameters
    private final double[] weights;
    private boolean isFitted;

    // the slope and its integral at the knots for the parameters of the last call
    private final double[] cachedParameters;
    private boolean hasCache;
    private final double[] knotSlopes;
    private final double[] knotIntegrals;

    // scratch space
    private final double[] designRow;
    private final double[] hatIntegrals;
    private final double[] hatIntegralsLow;
    private final double[] hatIntegralsHigh;
    private double[] pieceBounds = new double[0];
    private double[] pieceCeilings = new double[0];
    private double[] pieceMasses = new double[0];

    /**
     * @param numKnots      the number of knots along the height
     * @param numParameters the number of parameters of the tree prior among the features
     * @param warmUp        the number of observations after which the knots are fixed and the first fit is made
     * @param ridgePenalty  the ridge penalty per observation, in scaled units
     */
    public GlobalSplineTreePriorApproximation(int numKnots, int numParameters, int warmUp, double ridgePenalty) {
        this(numKnots, numParameters, warmUp, ridgePenalty, DEFAULT_MIN_WINDOW, DEFAULT_MIN_REFIT_INTERVAL);
    }

    /**
     * @param minWindow        the least number of most recent observations that the knots, the scales and the
     *                         weights are derived from, which is at least the warm-up
     * @param minRefitInterval the least number of observations between two fits after the first
     */
    public GlobalSplineTreePriorApproximation(int numKnots, int numParameters, int warmUp, double ridgePenalty, int minWindow, int minRefitInterval) {
        if (numKnots < 2) {
            throw new IllegalArgumentException("There must be at least two knots.");
        }

        this.numKnots = numKnots;
        this.numParameters = numParameters;
        this.warmUp = Math.max(warmUp, 1);
        this.ridgePenalty = ridgePenalty;
        this.minWindow = Math.max(minWindow, this.warmUp);
        this.minRefitInterval = Math.max(minRefitInterval, 1);
        this.maxStoredObservations = Math.max(MAX_STORED_OBSERVATIONS, this.minWindow);

        this.numFeatures = 1 + this.numParameters;
        this.numWeights = this.numKnots * this.numFeatures;

        this.parameterMeans = new double[this.numParameters];
        this.parameterScales = new double[this.numParameters];
        this.gramMatrix = new double[this.numWeights][this.numWeights];
        this.momentVector = new double[this.numWeights];
        this.weights = new double[this.numWeights];

        this.cachedParameters = new double[this.numParameters];
        this.knotSlopes = new double[this.numKnots];
        this.knotIntegrals = new double[this.numKnots];

        this.designRow = new double[this.numWeights];
        this.hatIntegrals = new double[this.numKnots];
        this.hatIntegralsLow = new double[this.numKnots];
        this.hatIntegralsHigh = new double[this.numKnots];
    }

    /* Approximation */

    @Override
    public double getSlope(double midpoint, double[] parameters) {
        if (!this.isFitted) {
            return 0.0;
        }

        this.updateCache(parameters);
        return this.computeSlope(midpoint);
    }

    @Override
    public double getCurvature(double midpoint, double[] parameters) {
        if (!this.isFitted || midpoint <= this.knots[0] || midpoint >= this.knots[this.numKnots - 1]) {
            return 0.0;
        }

        this.updateCache(parameters);
        int index = this.findKnotInterval(midpoint);
        return (this.knotSlopes[index + 1] - this.knotSlopes[index]) / (this.knots[index + 1] - this.knots[index]);
    }

    @Override
    public double getChange(double lower, double upper, double oldHeight, double newHeight, double[] parameters) {
        if (!this.isFitted) {
            return 0.0;
        }

        this.updateCache(parameters);
        return this.computeIntegral(newHeight) - this.computeIntegral(oldHeight);
    }

    /**
     * Draws a height from the density ∝ exp(-decayRate h + a(h)) between the bounds by rejection. The exponent is
     * piecewise quadratic, as its slope is linear between the knots. The interval is split at the knots and
     * every piece is split further until the exponent changes little over it. The density is below the constant
     * at the maximum of the exponent over the piece, so a draw from the piecewise constant envelope is accepted
     * with the probability exp(exponent - maximum).
     */
    @Override
    public double sampleHeight(double decayRate, double lower, double upper, double[] parameters) {
        if (!this.isFitted) {
            return TruncatedDensitySampler.sampleExponential(decayRate, lower, upper);
        }
        if (upper <= lower) {
            return lower;
        }

        this.updateCache(parameters);
        int numPieces = this.buildEnvelope(decayRate, lower, upper);

        double totalMass = 0.0;
        for (int i = 0; i < numPieces; i++) {
            totalMass += this.pieceMasses[i];
        }

        double referenceIntegral = this.computeIntegral(lower);
        while (true) {
            double u = Randomizer.nextDouble() * totalMass;
            int piece = 0;
            while (piece < numPieces - 1 && u > this.pieceMasses[piece]) {
                u -= this.pieceMasses[piece];
                piece++;
            }

            double pieceStart = this.pieceBounds[piece];
            double height = pieceStart + (this.pieceBounds[piece + 1] - pieceStart) * Randomizer.nextDouble();
            double exponent = this.computeExponent(decayRate, lower, referenceIntegral, height);
            if (Math.log(Randomizer.nextDouble()) < exponent - this.pieceCeilings[piece]) {
                return Math.min(Math.max(height, lower), upper);
            }
        }
    }

    /**
     * Fills the bounds, the exponents at the maxima and the masses of the envelope pieces and returns their
     * number. Exponents are measured from the value at the lower bound and masses from the largest exponent.
     */
    private int buildEnvelope(double decayRate, double lower, double upper) {
        // the pieces split at the knots inside the interval
        int firstKnot = this.findFirstKnotAbove(lower);
        int lastKnot = this.findFirstKnotAbove(upper);

        int numSegments = lastKnot - firstKnot + 1;
        int maxPieces = numSegments * MAX_PIECES_PER_KNOT_INTERVAL;
        if (this.pieceCeilings.length < maxPieces) {
            this.pieceBounds = new double[maxPieces + 1];
            this.pieceCeilings = new double[maxPieces];
            this.pieceMasses = new double[maxPieces];
        }

        double referenceIntegral = this.computeIntegral(lower);
        int numPieces = 0;
        double largestCeiling = Double.NEGATIVE_INFINITY;

        for (int segment = 0; segment < numSegments; segment++) {
            double start = segment == 0 ? lower : this.knots[firstKnot + segment - 1];
            double end = segment == numSegments - 1 ? upper : this.knots[firstKnot + segment];

            double steepness = Math.max(Math.abs(this.computeSlope(start) - decayRate), Math.abs(this.computeSlope(end) - decayRate));
            int numSubPieces = (int) Math.min(MAX_PIECES_PER_KNOT_INTERVAL,
                    Math.max(1.0, Math.ceil(steepness * (end - start) / MAX_EXPONENT_CHANGE_PER_PIECE)));

            for (int i = 0; i < numSubPieces; i++) {
                double pieceStart = start + (end - start) * i / numSubPieces;
                double pieceEnd = i == numSubPieces - 1 ? end : start + (end - start) * (i + 1) / numSubPieces;

                double ceiling = Math.max(
                        this.computeExponent(decayRate, lower, referenceIntegral, pieceStart),
                        this.computeExponent(decayRate, lower, referenceIntegral, pieceEnd));

                // the slope of the exponent is linear in the piece, so it has a maximum where it turns from + to -
                double startSlope = this.computeSlope(pieceStart) - decayRate;
                double endSlope = this.computeSlope(pieceEnd) - decayRate;
                if (startSlope > 0.0 && endSlope < 0.0) {
                    double vertex = pieceStart + (pieceEnd - pieceStart) * startSlope / (startSlope - endSlope);
                    ceiling = Math.max(ceiling, this.computeExponent(decayRate, lower, referenceIntegral, vertex));
                }

                this.pieceBounds[numPieces] = pieceStart;
                this.pieceBounds[numPieces + 1] = pieceEnd;
                this.pieceCeilings[numPieces] = ceiling;
                largestCeiling = Math.max(largestCeiling, ceiling);
                numPieces++;
            }
        }

        for (int i = 0; i < numPieces; i++) {
            this.pieceMasses[i] = (this.pieceBounds[i + 1] - this.pieceBounds[i]) * Math.exp(this.pieceCeilings[i] - largestCeiling);
        }
        return numPieces;
    }

    /** Returns -decayRate (h - lower) + a(h) - a(lower). */
    private double computeExponent(double decayRate, double lower, double referenceIntegral, double height) {
        return -decayRate * (height - lower) + this.computeIntegral(height) - referenceIntegral;
    }

    /** Returns the slope at the given height, in the units of the log tree prior per height. */
    private double computeSlope(double height) {
        if (height <= this.knots[0]) {
            return this.knotSlopes[0];
        }
        if (height >= this.knots[this.numKnots - 1]) {
            return this.knotSlopes[this.numKnots - 1];
        }

        int index = this.findKnotInterval(height);
        double fraction = (height - this.knots[index]) / (this.knots[index + 1] - this.knots[index]);
        return this.knotSlopes[index] + fraction * (this.knotSlopes[index + 1] - this.knotSlopes[index]);
    }

    /** Returns the integral of the slope from the first knot to the given height. */
    private double computeIntegral(double height) {
        int last = this.numKnots - 1;
        if (height <= this.knots[0]) {
            return this.knotSlopes[0] * (height - this.knots[0]);
        }
        if (height >= this.knots[last]) {
            return this.knotIntegrals[last] + this.knotSlopes[last] * (height - this.knots[last]);
        }

        int index = this.findKnotInterval(height);
        double width = this.knots[index + 1] - this.knots[index];
        double offset = height - this.knots[index];
        double slopeChange = this.knotSlopes[index + 1] - this.knotSlopes[index];
        return this.knotIntegrals[index] + this.knotSlopes[index] * offset + 0.5 * slopeChange * offset * offset / width;
    }

    /** Returns the index of the knot at or below the height, for a height within the knots. */
    private int findKnotInterval(double height) {
        int low = 0;
        int high = this.numKnots - 1;
        while (high - low > 1) {
            int middle = (low + high) >>> 1;
            if (this.knots[middle] <= height) {
                low = middle;
            } else {
                high = middle;
            }
        }
        return low;
    }

    /** Returns the index of the first knot above the height, or the number of knots if there is none. */
    private int findFirstKnotAbove(double height) {
        int low = 0;
        int high = this.numKnots;
        while (low < high) {
            int middle = (low + high) >>> 1;
            if (this.knots[middle] <= height) {
                low = middle + 1;
            } else {
                high = middle;
            }
        }
        return low;
    }

    /** Computes the slope and its integral at the knots for the parameters, unless they are cached already. */
    private void updateCache(double[] parameters) {
        if (this.hasCache && Arrays.equals(this.cachedParameters, parameters)) {
            return;
        }

        System.arraycopy(parameters, 0, this.cachedParameters, 0, this.numParameters);
        this.hasCache = true;

        for (int k = 0; k < this.numKnots; k++) {
            int block = k * this.numFeatures;
            double slope = this.weights[block];
            for (int j = 0; j < this.numParameters; j++) {
                slope += this.weights[block + 1 + j] * this.scaleParameter(j, parameters[j]);
            }
            this.knotSlopes[k] = slope / this.heightScale;
        }

        this.knotIntegrals[0] = 0.0;
        for (int k = 1; k < this.numKnots; k++) {
            this.knotIntegrals[k] = this.knotIntegrals[k - 1]
                    + 0.5 * (this.knotSlopes[k - 1] + this.knotSlopes[k]) * (this.knots[k] - this.knots[k - 1]);
        }
    }

    private double scaleParameter(int index, double parameter) {
        return this.parameterScales[index] > 0.0
                ? (parameter - this.parameterMeans[index]) / this.parameterScales[index]
                : 0.0;
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
        this.currentNodes.add(new double[]{midpoint + oldOffset, midpoint + newOffset});
        this.currentParameters = parameters.clone();
    }

    @Override
    public void finishObservation(double target) {
        if (this.currentNodes.isEmpty() || !Double.isFinite(target)) {
            this.discardObservation();
            return;
        }

        this.numObservations++;
        this.numObservationsSinceFit++;

        if (this.numObservations % this.keepEvery == 0) {
            this.recentObservations.addLast(new Observation(
                    this.numObservations, this.currentNodes.toArray(new double[0][]), this.currentParameters, target));
        }
        this.restrictToWindow();

        long refitInterval = Math.max(this.minRefitInterval, (long) (REFIT_FRACTION * this.numObservations));
        boolean isFirstFitDue = !this.isFitted && this.recentObservations.size() >= this.warmUp;
        if (isFirstFitDue || (this.isFitted && this.numObservationsSinceFit >= refitInterval)) {
            this.fit();
        }

        this.discardObservation();
    }

    /** Drops the observations that are older than the window, and thins the kept ones if there are too many. */
    private void restrictToWindow() {
        long windowSize = Math.max(this.minWindow, (long) (WINDOW_FRACTION * this.numObservations));
        while (!this.recentObservations.isEmpty() && this.recentObservations.peekFirst().number() <= this.numObservations - windowSize) {
            this.recentObservations.removeFirst();
        }

        if (this.recentObservations.size() > this.maxStoredObservations) {
            ArrayDeque<Observation> thinnedObservations = new ArrayDeque<>();
            boolean isKept = false;
            for (Observation observation : this.recentObservations) {
                isKept = !isKept;
                if (isKept) {
                    thinnedObservations.addLast(observation);
                }
            }

            this.recentObservations.clear();
            this.recentObservations.addAll(thinnedObservations);
            this.keepEvery *= 2;
        }
    }

    @Override
    public void discardObservation() {
        this.currentNodes.clear();
        this.currentParameters = null;
    }

    /* Scales */

    /**
     * Derives the knots and the scales from the recent observations. The knots are distinct quantiles of the
     * heights the nodes moved between. Every parameter is centred on its mean and divided by its standard
     * deviation over the observations, and the heights are measured in units of the root mean square of the
     * moves. A parameter that did not vary is left out of the fit, as it cannot be told apart from the constant.
     */
    private void fixKnotsAndScales() {
        int numMoves = 0;
        for (Observation observation : this.recentObservations) {
            numMoves += observation.nodes().length;
        }

        double[] heights = new double[2 * numMoves];
        double moveSquareSum = 0.0;
        int index = 0;
        for (Observation observation : this.recentObservations) {
            for (double[] node : observation.nodes()) {
                heights[index++] = node[0];
                heights[index++] = node[1];
                moveSquareSum += (node[1] - node[0]) * (node[1] - node[0]);
            }
        }
        Arrays.sort(heights);

        double moveScale = Math.sqrt(moveSquareSum / numMoves);
        this.heightScale = moveScale > 0.0 ? moveScale : 1.0;

        double range = heights[heights.length - 1] - heights[0];
        double minSpacing = (range > 0.0 ? range : 1.0) * 1e-9;

        this.knots = new double[this.numKnots];
        for (int k = 0; k < this.numKnots; k++) {
            double quantile = (double) k / (this.numKnots - 1);
            this.knots[k] = heights[(int) Math.round(quantile * (heights.length - 1))];
            if (k > 0 && this.knots[k] < this.knots[k - 1] + minSpacing) {
                this.knots[k] = this.knots[k - 1] + minSpacing;
            }
        }

        // the integral of the hat function of every knot from the first knot up to every knot
        this.knotHatIntegrals = new double[this.numKnots][this.numKnots];
        for (int k = 0; k < this.numKnots; k++) {
            for (int j = 1; j < this.numKnots; j++) {
                double left = j - 1 == k ? 1.0 : 0.0;
                double right = j == k ? 1.0 : 0.0;
                this.knotHatIntegrals[k][j] = this.knotHatIntegrals[k][j - 1] + 0.5 * (left + right) * (this.knots[j] - this.knots[j - 1]);
            }
        }

        int numParameterObservations = this.recentObservations.size();
        for (int j = 0; j < this.numParameters; j++) {
            double sum = 0.0;
            double squareSum = 0.0;
            for (Observation observation : this.recentObservations) {
                double value = observation.parameters()[j];
                sum += value;
                squareSum += value * value;
            }

            double mean = sum / numParameterObservations;
            double standardDeviation = Math.sqrt(Math.max(squareSum / numParameterObservations - mean * mean, 0.0));
            this.parameterMeans[j] = mean;
            this.parameterScales[j] = standardDeviation > 1e-9 * Math.max(Math.abs(mean), 1e-300) ? standardDeviation : 0.0;
        }

    }

    /* Fitting */

    /**
     * Writes the integrals over [from, to] of the hat function of every knot into the output, in units of the
     * height scale. Beyond the first and the last knot the hat functions of these knots are constant.
     */
    private void computeHatIntegralsBetween(double from, double to, double[] output) {
        this.computeHatIntegralsFromFirstKnot(to, this.hatIntegralsHigh);
        this.computeHatIntegralsFromFirstKnot(from, this.hatIntegralsLow);
        for (int k = 0; k < this.numKnots; k++) {
            output[k] = (this.hatIntegralsHigh[k] - this.hatIntegralsLow[k]) / this.heightScale;
        }
    }

    /** Writes the integrals of the hat function of every knot from the first knot up to the height. */
    private void computeHatIntegralsFromFirstKnot(double height, double[] output) {
        int last = this.numKnots - 1;

        if (height <= this.knots[0]) {
            Arrays.fill(output, 0.0);
            output[0] = height - this.knots[0];
            return;
        }

        if (height >= this.knots[last]) {
            for (int k = 0; k < this.numKnots; k++) {
                output[k] = this.knotHatIntegrals[k][last];
            }
            output[last] += height - this.knots[last];
            return;
        }

        int index = this.findKnotInterval(height);
        double width = this.knots[index + 1] - this.knots[index];
        double offset = height - this.knots[index];
        double fraction = offset / width;

        for (int k = 0; k < this.numKnots; k++) {
            output[k] = this.knotHatIntegrals[k][index];
        }

        // within the interval only the hat functions of its two knots are not zero
        output[index] += 0.5 * (2.0 - fraction) * offset;
        output[index + 1] += 0.5 * fraction * offset;
    }

    /** Adds an observation to the normal equations. */
    private void accumulate(double[][] nodes, double[] parameters, double target) {
        Arrays.fill(this.designRow, 0.0);

        for (double[] node : nodes) {
            this.computeHatIntegralsBetween(node[0], node[1], this.hatIntegrals);
            for (int k = 0; k < this.numKnots; k++) {
                int block = k * this.numFeatures;
                this.designRow[block] += this.hatIntegrals[k];
                for (int j = 0; j < this.numParameters; j++) {
                    this.designRow[block + 1 + j] += this.hatIntegrals[k] * this.scaleParameter(j, parameters[j]);
                }
            }
        }

        for (int a = 0; a < this.numWeights; a++) {
            if (this.designRow[a] == 0.0) {
                continue;
            }
            for (int b = 0; b < this.numWeights; b++) {
                this.gramMatrix[a][b] += this.designRow[a] * this.designRow[b];
            }
            this.momentVector[a] += this.designRow[a] * target;
        }
    }

    /**
     * Derives the knots and the scales from the recent observations, and solves the normal equations of these
     * observations with a ridge penalty for the weights. A fit whose weights are not finite is ignored, and the
     * knots and the scales of the last fit are kept.
     */
    private void fit() {
        this.numObservationsSinceFit = 0;

        double[] previousKnots = this.knots;
        double[][] previousKnotHatIntegrals = this.knotHatIntegrals;
        double[] previousParameterMeans = this.parameterMeans.clone();
        double[] previousParameterScales = this.parameterScales.clone();
        double previousHeightScale = this.heightScale;

        this.fixKnotsAndScales();

        for (double[] row : this.gramMatrix) {
            Arrays.fill(row, 0.0);
        }
        Arrays.fill(this.momentVector, 0.0);
        for (Observation observation : this.recentObservations) {
            this.accumulate(observation.nodes(), observation.parameters(), observation.target());
        }

        RealMatrix matrix = new Array2DRowRealMatrix(this.gramMatrix, true);
        double penalty = this.ridgePenalty * this.recentObservations.size();
        for (int a = 0; a < this.numWeights; a++) {
            matrix.addToEntry(a, a, penalty);
        }
        RealVector rightHandSide = new ArrayRealVector(this.momentVector, true);

        double[] solution;
        try {
            solution = new CholeskyDecomposition(matrix).getSolver().solve(rightHandSide).toArray();
        } catch (RuntimeException exception) {
            // the matrix is not positive definite to working precision, so fall back to the pseudo-inverse
            solution = new SingularValueDecomposition(matrix).getSolver().solve(rightHandSide).toArray();
        }

        for (double weight : solution) {
            if (!Double.isFinite(weight)) {
                this.knots = previousKnots;
                this.knotHatIntegrals = previousKnotHatIntegrals;
                System.arraycopy(previousParameterMeans, 0, this.parameterMeans, 0, this.numParameters);
                System.arraycopy(previousParameterScales, 0, this.parameterScales, 0, this.numParameters);
                this.heightScale = previousHeightScale;
                return;
            }
        }

        System.arraycopy(solution, 0, this.weights, 0, this.numWeights);
        this.hasCache = false;
        this.isFitted = true;
    }

}
