package emat.operators.nodetimes.warp;

import org.apache.commons.math4.legacy.core.Pair;
import org.apache.commons.math4.legacy.fitting.leastsquares.LeastSquaresBuilder;
import org.apache.commons.math4.legacy.fitting.leastsquares.LeastSquaresProblem;
import org.apache.commons.math4.legacy.fitting.leastsquares.LevenbergMarquardtOptimizer;
import org.apache.commons.math4.legacy.linear.Array2DRowRealMatrix;
import org.apache.commons.math4.legacy.linear.ArrayRealVector;
import org.apache.commons.math4.legacy.linear.RealMatrix;
import org.apache.commons.math4.legacy.linear.RealVector;

import java.util.ArrayDeque;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

/**
 * Learns from observed trees and the log density the tree prior gave them how the rate of its branching events
 * depends on the height and on its parameters, without the formula of the prior.
 *
 * <p>The log density of a tree is modelled as the sum over the branching events of the logarithm of the number of
 * units at risk and of the rate at the height of the event, minus the integral of the rate over the heights, weighted
 * by the number of units at risk while the lineages are present, plus a constant that is linear in the parameters. The
 * number of units at risk follows from the number of lineages by the {@link TreeHazard}. The logarithm of the rate is
 * piecewise linear in the height between knots at quantiles of the heights of the internal nodes, and linear in the
 * standardised parameters at every knot. The weights are fitted to the log densities by Levenberg-Marquardt least
 * squares with a small ridge penalty on the weights of the parameters.
 *
 * <p>The fit uses the most recent half of the stored observations, so that it follows the drift of the chain.
 */
public class LearnedHazardRate {

    // the penalty on the weights of the standardised parameters, relative to the squared error of a log density
    private static final double RIDGE = 1e-3;

    // below this size of the difference of two log rates, the integral of the rate is expanded in a series
    private static final double SMALL_DIFFERENCE = 1e-3;

    private static final int MAX_EVALUATIONS = 200;

    // the hazard that the fit uses, which is chosen at the first fit if the hazard is AUTO
    private TreeHazard resolvedHazard;
    private double[] candidateRmsResiduals;
    private final int numKnots;
    private final int numParameters;
    private final int maxObservations;

    private final ArrayDeque<Observation> observations = new ArrayDeque<>();
    private long numObserved;

    // the fit: the knots above zero, the standardisation of the parameters and the weights
    private boolean isFitted;
    private double[] knotHeights;
    private final double[] parameterMeans;
    private final double[] parameterScales;
    private double[][] knotWeights;
    private double[] interceptWeights;
    private double rmsResidual = Double.NaN;

    /** A tree by the sorted heights of its tips and of its internal nodes, with the parameters and log density it was seen with. */
    private record Observation(double[] parameters, double[] tipHeights, double[] internalHeights, double logDensity) {
    }

    /** The branching events and the intervals with units at risk of a tree, for one hazard. */
    private record Events(double[] heights, double[] logMultiplicities, double[] pieceStarts, double[] pieceEnds, double[] pieceMultiplicities) {
    }

    /** The result of a fit, which is kept only if it is the one that is chosen. */
    private record Fit(double[] knots, double[] means, double[] scales, double[] weights, double rmsResidual) {
    }

    /** An observation, located on the segments between the knots of a fit. */
    private record Located(double[] features, int[] eventSegments, double[] eventFractions, double eventLogMultiplicitySum,
                           int[] pieceSegments, double[] pieceStartFractions, double[] pieceEndFractions,
                           double[] pieceWidths, double[] pieceMultiplicities, double logDensity) {
    }

    /**
     * @param numKnots        the number of knots above height zero
     * @param numParameters   the number of parameters of the tree prior that the rate may depend on
     * @param maxObservations the number of most recent observations that are kept
     */
    public LearnedHazardRate(TreeHazard hazard, int numKnots, int numParameters, int maxObservations) {
        if (numKnots < 2) {
            throw new IllegalArgumentException("There must be at least two knots above zero.");
        }

        this.resolvedHazard = hazard == TreeHazard.AUTO ? null : hazard;
        this.numKnots = numKnots;
        this.numParameters = numParameters;
        this.maxObservations = maxObservations;
        this.parameterMeans = new double[numParameters];
        this.parameterScales = new double[numParameters];
    }

    public boolean isFitted() {
        return this.isFitted;
    }

    /** Returns the number of observations that were made, including those that are no longer kept. */
    public long getNumObserved() {
        return this.numObserved;
    }

    /** Returns the root mean square of the residuals of the last fit to the log densities. */
    public double getRmsResidual() {
        return this.rmsResidual;
    }

    /* Observations */

    /**
     * Records a tree, given by the heights of its tips and of its internal nodes, with the parameters of the tree
     * prior and the log density the prior gave it. A log density that is not finite is ignored.
     */
    public void addObservation(double[] tipHeights, double[] internalHeights, double[] parameters, double logDensity) {
        this.numObserved++;
        if (!Double.isFinite(logDensity)) {
            return;
        }

        this.observations.addLast(this.createObservation(tipHeights, internalHeights, parameters, logDensity));
        if (this.observations.size() > this.maxObservations) {
            this.observations.removeFirst();
        }
    }

    private Observation createObservation(double[] tipHeights, double[] internalHeights, double[] parameters, double logDensity) {
        double[] tips = tipHeights.clone();
        double[] internals = internalHeights.clone();
        Arrays.sort(tips);
        Arrays.sort(internals);

        return new Observation(parameters.clone(), tips, internals, logDensity);
    }

    /** Walks up the tree and records the branching events and the intervals with at least one unit at risk. */
    private Events collectEvents(Observation observation, TreeHazard hazard) {
        double[] tips = observation.tipHeights();
        double[] internals = observation.internalHeights();

        List<double[]> pieces = new ArrayList<>();
        double[] logMultiplicities = new double[internals.length];
        int numLineages = 0;
        int tipIndex = 0;
        int internalIndex = 0;
        double previous = 0.0;

        while (internalIndex < internals.length) {
            boolean isTip = tipIndex < tips.length && tips[tipIndex] <= internals[internalIndex];
            double height = isTip ? tips[tipIndex] : internals[internalIndex];

            double multiplicity = hazard.computeExposureMultiplicity(numLineages);
            if (height > previous && multiplicity > 0.0) {
                pieces.add(new double[]{previous, height, multiplicity});
            }

            if (isTip) {
                numLineages++;
                tipIndex++;
            } else {
                logMultiplicities[internalIndex++] = Math.log(hazard.computeEventMultiplicity(numLineages));
                numLineages--;
            }
            previous = height;
        }

        double[] pieceStarts = new double[pieces.size()];
        double[] pieceEnds = new double[pieces.size()];
        double[] pieceMultiplicities = new double[pieces.size()];
        for (int i = 0; i < pieces.size(); i++) {
            pieceStarts[i] = pieces.get(i)[0];
            pieceEnds[i] = pieces.get(i)[1];
            pieceMultiplicities[i] = pieces.get(i)[2];
        }

        return new Events(internals, logMultiplicities, pieceStarts, pieceEnds, pieceMultiplicities);
    }

    /* Fitting */

    /**
     * Fits the weights to the most recent half of the observations. If the hazard is AUTO, the first fit tries both
     * hazards and keeps the one with the smaller residual for good. A fit that fails or is not finite is ignored and
     * the previous fit is kept.
     */
    public void fit() {
        int numUsed = Math.min(this.observations.size(), Math.max(this.observations.size() / 2, 10));
        List<Observation> all = new ArrayList<>(this.observations);
        List<Observation> used = all.subList(all.size() - numUsed, all.size());
        if (used.size() < 3) {
            return;
        }

        if (this.resolvedHazard != null) {
            this.keepFit(this.fitWithHazard(used, this.resolvedHazard));
            return;
        }

        Fit coalescentFit = this.fitWithHazard(used, TreeHazard.COALESCENT);
        Fit birthDeathFit = this.fitWithHazard(used, TreeHazard.BIRTH_DEATH);
        if (coalescentFit == null && birthDeathFit == null) {
            return;
        }

        boolean isCoalescent = birthDeathFit == null || (coalescentFit != null && coalescentFit.rmsResidual() <= birthDeathFit.rmsResidual());
        this.resolvedHazard = isCoalescent ? TreeHazard.COALESCENT : TreeHazard.BIRTH_DEATH;
        this.candidateRmsResiduals = new double[]{
                coalescentFit == null ? Double.NaN : coalescentFit.rmsResidual(),
                birthDeathFit == null ? Double.NaN : birthDeathFit.rmsResidual()};
        this.keepFit(isCoalescent ? coalescentFit : birthDeathFit);
    }

    /** Returns the hazard the fit uses, or null if it is AUTO and has not been chosen yet. */
    public TreeHazard getResolvedHazard() {
        return this.resolvedHazard;
    }

    /** Returns the residuals of the fits with the coalescent and the birth-death hazard from which AUTO chose, or null. */
    public double[] getCandidateRmsResiduals() {
        return this.candidateRmsResiduals;
    }

    private Fit fitWithHazard(List<Observation> used, TreeHazard hazard) {
        double[] knots = this.computeKnots(used);
        double[] means = new double[this.numParameters];
        double[] scales = new double[this.numParameters];
        this.computeStandardisation(used, means, scales);
        if (knots == null) {
            return null;
        }

        List<Located> located = new ArrayList<>();
        for (Observation observation : used) {
            located.add(this.locate(observation, this.collectEvents(observation, hazard), knots, means, scales));
        }

        int numFeatures = 1 + this.numParameters;
        int numKnotsTotal = knots.length;
        int numRidge = numKnotsTotal * this.numParameters;
        int numRows = located.size() + numRidge;

        double[] start = this.computeStart(located, numKnotsTotal, numFeatures);
        double[] target = new double[numRows];
        for (int o = 0; o < located.size(); o++) {
            target[o] = located.get(o).logDensity();
        }

        double ridgeRoot = Math.sqrt(RIDGE);
        try {
            LeastSquaresProblem problem = new LeastSquaresBuilder()
                    .start(start)
                    .model(point -> this.evaluate(point, located, numKnotsTotal, numFeatures, numRidge, ridgeRoot))
                    .target(target)
                    .lazyEvaluation(false)
                    .maxEvaluations(MAX_EVALUATIONS)
                    .maxIterations(MAX_EVALUATIONS)
                    .build();
            double[] weights = new LevenbergMarquardtOptimizer().optimize(problem).getPoint().toArray();

            for (double weight : weights) {
                if (!Double.isFinite(weight)) {
                    return null;
                }
            }

            return new Fit(knots, means, scales, weights, this.computeRmsResidual(weights, located, numKnotsTotal, numFeatures, target));
        } catch (RuntimeException exception) {
            // the optimiser did not converge or met a value that is not finite, so there is no fit
            return null;
        }
    }

    /** Returns the knots: zero and the quantiles of the heights of the internal nodes, or null if they cannot be told apart. */
    private double[] computeKnots(List<Observation> used) {
        int count = 0;
        for (Observation observation : used) {
            count += observation.internalHeights().length;
        }

        double[] heights = new double[count];
        int index = 0;
        for (Observation observation : used) {
            for (double height : observation.internalHeights()) {
                heights[index++] = height;
            }
        }
        Arrays.sort(heights);

        double range = heights[count - 1];
        if (!(range > 0.0)) {
            return null;
        }

        double minSpacing = range * 1e-6;
        double[] knots = new double[this.numKnots + 1];
        for (int k = 1; k <= this.numKnots; k++) {
            double quantile = (double) k / this.numKnots;
            knots[k] = Math.max(heights[(int) Math.round(quantile * (count - 1))], knots[k - 1] + minSpacing);
        }
        return knots;
    }

    /** Computes the mean and the scale of every parameter; a parameter that did not vary gets the scale one. */
    private void computeStandardisation(List<Observation> used, double[] means, double[] scales) {
        for (int j = 0; j < this.numParameters; j++) {
            double sum = 0.0;
            double squareSum = 0.0;
            for (Observation observation : used) {
                double value = observation.parameters()[j];
                sum += value;
                squareSum += value * value;
            }

            double mean = sum / used.size();
            double standardDeviation = Math.sqrt(Math.max(squareSum / used.size() - mean * mean, 0.0));
            means[j] = mean;
            scales[j] = standardDeviation > 1e-9 * Math.max(Math.abs(mean), 1e-300) ? standardDeviation : 1.0;
        }
    }

    /** Places the events and the intervals of an observation on the segments between the knots. */
    private Located locate(Observation observation, Events events, double[] knots, double[] means, double[] scales) {
        double[] features = new double[1 + this.numParameters];
        features[0] = 1.0;
        for (int j = 0; j < this.numParameters; j++) {
            features[1 + j] = (observation.parameters()[j] - means[j]) / scales[j];
        }

        int numEvents = events.heights().length;
        int[] eventSegments = new int[numEvents];
        double[] eventFractions = new double[numEvents];
        double logMultiplicitySum = 0.0;
        for (int e = 0; e < numEvents; e++) {
            eventSegments[e] = this.findSegment(knots, events.heights()[e]);
            eventFractions[e] = this.computeFraction(knots, eventSegments[e], events.heights()[e]);
            logMultiplicitySum += events.logMultiplicities()[e];
        }

        // an interval is cut at the knots, as the rate is exponential of a linear function within a segment
        List<double[]> cut = new ArrayList<>();
        for (int p = 0; p < events.pieceStarts().length; p++) {
            double current = events.pieceStarts()[p];
            double end = events.pieceEnds()[p];
            while (current < end) {
                int segment = this.findSegment(knots, current);
                double segmentEnd = segment == knots.length - 2 ? end : Math.min(end, knots[segment + 1]);
                if (!(segmentEnd > current)) {
                    break;
                }
                cut.add(new double[]{segment, this.computeFraction(knots, segment, current),
                        this.computeFraction(knots, segment, segmentEnd), segmentEnd - current, events.pieceMultiplicities()[p]});
                current = segmentEnd;
            }
        }

        int[] pieceSegments = new int[cut.size()];
        double[] startFractions = new double[cut.size()];
        double[] endFractions = new double[cut.size()];
        double[] widths = new double[cut.size()];
        double[] multiplicities = new double[cut.size()];
        for (int i = 0; i < cut.size(); i++) {
            pieceSegments[i] = (int) cut.get(i)[0];
            startFractions[i] = cut.get(i)[1];
            endFractions[i] = cut.get(i)[2];
            widths[i] = cut.get(i)[3];
            multiplicities[i] = cut.get(i)[4];
        }

        return new Located(features, eventSegments, eventFractions, logMultiplicitySum,
                pieceSegments, startFractions, endFractions, widths, multiplicities, observation.logDensity());
    }

    /** Returns the segment of the knots that contains the height, where the last segment continues without end. */
    private int findSegment(double[] knots, double height) {
        int low = 0;
        int high = knots.length - 1;
        while (high - low > 1) {
            int middle = (low + high) >>> 1;
            if (knots[middle] <= height) {
                low = middle;
            } else {
                high = middle;
            }
        }
        return low;
    }

    private double computeFraction(double[] knots, int segment, double height) {
        return (height - knots[segment]) / (knots[segment + 1] - knots[segment]);
    }

    /** Starts from a rate that is the same at all heights and parameters, with the constant that fits the mean. */
    private double[] computeStart(List<Located> located, int numKnotsTotal, int numFeatures) {
        double events = 0.0;
        double exposure = 0.0;
        for (Located observation : located) {
            events += observation.eventSegments().length;
            for (int i = 0; i < observation.pieceWidths().length; i++) {
                exposure += observation.pieceWidths()[i] * observation.pieceMultiplicities()[i];
            }
        }
        double logRate = Math.log(events / exposure);

        double[] start = new double[(numKnotsTotal + 1) * numFeatures];
        for (int k = 0; k < numKnotsTotal; k++) {
            start[k * numFeatures] = logRate;
        }

        double sum = 0.0;
        for (Located observation : located) {
            double fitted = observation.eventLogMultiplicitySum() + observation.eventSegments().length * logRate;
            for (int i = 0; i < observation.pieceWidths().length; i++) {
                fitted -= observation.pieceMultiplicities()[i] * observation.pieceWidths()[i] * Math.exp(logRate);
            }
            sum += observation.logDensity() - fitted;
        }
        start[numKnotsTotal * numFeatures] = sum / located.size();

        return start;
    }

    /**
     * Returns the modelled log densities and their derivatives with respect to the weights, followed by the ridge
     * terms. The log rate at a knot is the product of the features and the weights of the knot.
     */
    private Pair<RealVector, RealMatrix> evaluate(RealVector point, List<Located> located, int numKnotsTotal,
                                                  int numFeatures, int numRidge, double ridgeRoot) {
        double[] weights = point.toArray();
        int numRows = located.size() + numRidge;
        double[] values = new double[numRows];
        double[][] jacobian = new double[numRows][weights.length];

        double[] logRates = new double[numKnotsTotal];
        double[] derivatives = new double[numKnotsTotal];
        int interceptStart = numKnotsTotal * numFeatures;

        for (int o = 0; o < located.size(); o++) {
            Located observation = located.get(o);
            double[] features = observation.features();

            for (int k = 0; k < numKnotsTotal; k++) {
                double logRate = 0.0;
                for (int f = 0; f < numFeatures; f++) {
                    logRate += weights[k * numFeatures + f] * features[f];
                }
                logRates[k] = logRate;
            }
            Arrays.fill(derivatives, 0.0);

            double value = observation.eventLogMultiplicitySum();
            for (int f = 0; f < numFeatures; f++) {
                value += weights[interceptStart + f] * features[f];
            }

            for (int e = 0; e < observation.eventSegments().length; e++) {
                int segment = observation.eventSegments()[e];
                double fraction = observation.eventFractions()[e];
                value += (1.0 - fraction) * logRates[segment] + fraction * logRates[segment + 1];
                derivatives[segment] += 1.0 - fraction;
                derivatives[segment + 1] += fraction;
            }

            for (int i = 0; i < observation.pieceWidths().length; i++) {
                int segment = observation.pieceSegments()[i];
                double startFraction = observation.pieceStartFractions()[i];
                double endFraction = observation.pieceEndFractions()[i];
                double multiplicity = observation.pieceMultiplicities()[i];
                double width = observation.pieceWidths()[i];

                double startLogRate = (1.0 - startFraction) * logRates[segment] + startFraction * logRates[segment + 1];
                double endLogRate = (1.0 - endFraction) * logRates[segment] + endFraction * logRates[segment + 1];
                double difference = endLogRate - startLogRate;
                double scale = width * Math.exp(startLogRate);
                double mean = this.computeMeanOfExponential(difference);
                double slope = this.computeSlopeOfMean(difference);

                value -= multiplicity * scale * mean;
                double derivativeEnd = multiplicity * scale * slope;
                double derivativeStart = multiplicity * scale * (mean - slope);
                derivatives[segment] -= (1.0 - startFraction) * derivativeStart + (1.0 - endFraction) * derivativeEnd;
                derivatives[segment + 1] -= startFraction * derivativeStart + endFraction * derivativeEnd;
            }

            values[o] = value;
            for (int k = 0; k < numKnotsTotal; k++) {
                for (int f = 0; f < numFeatures; f++) {
                    jacobian[o][k * numFeatures + f] = derivatives[k] * features[f];
                }
            }
            for (int f = 0; f < numFeatures; f++) {
                jacobian[o][interceptStart + f] = features[f];
            }
        }

        // the ridge penalty on the weights of the parameters, not on the constants
        int row = located.size();
        for (int k = 0; k < numKnotsTotal && numRidge > 0; k++) {
            for (int f = 1; f < numFeatures; f++) {
                values[row] = ridgeRoot * weights[k * numFeatures + f];
                jacobian[row][k * numFeatures + f] = ridgeRoot;
                row++;
            }
        }

        return new Pair<>(new ArrayRealVector(values, false), new Array2DRowRealMatrix(jacobian, false));
    }

    /** Returns (exp(d) - 1) / d, the mean of exp(x d) over x from zero to one. */
    private double computeMeanOfExponential(double difference) {
        if (Math.abs(difference) < SMALL_DIFFERENCE) {
            return 1.0 + difference * (0.5 + difference * (1.0 / 6.0 + difference / 24.0));
        }
        return Math.expm1(difference) / difference;
    }

    /** Returns the derivative of {@link #computeMeanOfExponential} with respect to its argument. */
    private double computeSlopeOfMean(double difference) {
        if (Math.abs(difference) < SMALL_DIFFERENCE) {
            return 0.5 + difference * (1.0 / 3.0 + difference * (1.0 / 8.0 + difference / 30.0));
        }
        return (Math.exp(difference) * (difference - 1.0) + 1.0) / (difference * difference);
    }

    private double computeRmsResidual(double[] weights, List<Located> located, int numKnotsTotal, int numFeatures, double[] target) {
        double[] model = this.evaluate(new ArrayRealVector(weights, false), located, numKnotsTotal, numFeatures, 0, 0.0).getFirst().toArray();
        double squareSum = 0.0;
        for (int o = 0; o < located.size(); o++) {
            squareSum += (target[o] - model[o]) * (target[o] - model[o]);
        }
        return Math.sqrt(squareSum / located.size());
    }

    private void keepFit(Fit fit) {
        if (fit == null) {
            return;
        }

        int numFeatures = 1 + this.numParameters;
        int numKnotsTotal = fit.knots().length;
        this.knotWeights = new double[numKnotsTotal][numFeatures];
        for (int k = 0; k < numKnotsTotal; k++) {
            System.arraycopy(fit.weights(), k * numFeatures, this.knotWeights[k], 0, numFeatures);
        }

        this.knotHeights = fit.knots();
        System.arraycopy(fit.means(), 0, this.parameterMeans, 0, this.numParameters);
        System.arraycopy(fit.scales(), 0, this.parameterScales, 0, this.numParameters);
        this.interceptWeights = Arrays.copyOfRange(fit.weights(), numKnotsTotal * numFeatures, fit.weights().length);
        this.rmsResidual = fit.rmsResidual();
        this.isFitted = true;
    }

    /* Evaluation */

    /** Returns the time change of the learned rate for the given parameters of the tree prior. */
    public HazardTimeChange createTimeChange(double[] parameters) {
        double[] logRates = new double[this.knotHeights.length];
        for (int k = 0; k < this.knotHeights.length; k++) {
            double logRate = this.knotWeights[k][0];
            for (int j = 0; j < this.numParameters; j++) {
                logRate += this.knotWeights[k][1 + j] * (parameters[j] - this.parameterMeans[j]) / this.parameterScales[j];
            }
            logRates[k] = logRate;
        }

        return new HazardTimeChange(this.knotHeights, logRates);
    }

}
