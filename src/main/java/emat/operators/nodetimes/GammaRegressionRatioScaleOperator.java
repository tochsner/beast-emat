package emat.operators.nodetimes;

import beast.base.core.Description;
import beast.base.core.Input;
import beast.base.evolution.tree.Node;
import beast.base.inference.StateNode;
import beast.base.spec.type.IntVector;
import beast.base.spec.type.RealScalar;
import beast.base.spec.type.RealVector;
import org.apache.commons.statistics.distribution.GammaDistribution;
import org.apache.commons.statistics.distribution.NormalDistribution;

import java.util.ArrayList;
import java.util.List;

@Description("Moves a random internal node in the ratio parameterisation of the node heights, with the ratio " +
        "r = (h - τ) / (h_p - τ) of its height and the height of its parent, both above the oldest tip τ below the node. " +
        "The odds r / (1 - r) follow a gamma distribution that is learned online, and the step is a step of the kernel in the " +
        "standard normal space of its univariate Rosenblatt transform z = Φ⁻¹(F(odds)). The nodes are grouped by the number " +
        "of mutations on the branch of the node and the branches of its children, which no move changes, and every group has " +
        "its own distribution. The logarithms of the shape and the scale of the gamma distribution are an intercept plus a " +
        "linear combination of the standardised values of the tree prior parameters, which no move changes either. They are " +
        "learned by stochastic gradient ascent on the gamma log-likelihood of the odds of the visited states, with a step " +
        "that decays with the number of observations, which keeps the adaptation diminishing.")
public class GammaRegressionRatioScaleOperator extends AnchoredRatioScaleOperator {

    final public Input<List<StateNode>> treePriorParametersInput = new Input<>("treePriorParameter", "real scalar, real vector or integer vector parameters of the tree prior, on which the shape and the scale of the gamma distribution depend linearly in log space", new ArrayList<>());
    final public Input<Integer> minObservationsInput = new Input<>("minObservations", "number of observed odds of a mutation group after which its intercepts are first set by the maximum likelihood fit of a gamma distribution that ignores the tree prior parameters", 100);
    final public Input<Double> learningRateInput = new Input<>("learningRate", "initial step of the stochastic gradient ascent of the coefficients", 0.01);
    final public Input<Double> learningDecayInput = new Input<>("learningDecay", "number of observations of a group after which its step has decayed to 1 / sqrt(2) of the initial step", 1000.0);

    private static final int NUM_GROUPS = 16;
    private static final double MAX_ABS_GRADIENT = 5.0;
    private static final double MAX_ABS_STANDARDISED_VALUE = 5.0;
    private static final double MAX_ABS_LOG_SHAPE = 6.0;
    private static final double MAX_ABS_LOG_SCALE = 12.0;

    private static final NormalDistribution NORMAL = NormalDistribution.of(0.0, 1.0);

    private int numCovariates;

    // the running mean and sum of squared deviations of every tree prior parameter value, used to standardise it
    private long numCovariateObservations;
    private double[] covariateMeans;
    private double[] covariateSquaredDeviations;

    // the raw and the standardised values of the tree prior parameters in the last proposal, the latter with a leading one for the intercept
    private double[] covariateValues;
    private double[] features;

    // per mutation group, the coefficients of the logarithm of the shape and of the scale, intercept first
    private double[][] logShapeCoefficients;
    private double[][] logScaleCoefficients;

    // per mutation group, the number of observed odds, the sum of the odds and of their logarithms, and whether the coefficients are set
    private long[] numObservations;
    private double[] sumOdds;
    private double[] sumLogOdds;
    private boolean[] isFitted;

    // the node and the gamma distribution of the odds of the last proposal
    private Node selectedNode;
    private GammaDistribution gamma;

    @Override
    public void initAndValidate() {
        super.initAndValidate();

        this.numCovariates = this.countCovariates();
        this.covariateMeans = new double[this.numCovariates];
        this.covariateSquaredDeviations = new double[this.numCovariates];
        this.covariateValues = new double[this.numCovariates];
        this.features = new double[this.numCovariates + 1];

        this.logShapeCoefficients = new double[NUM_GROUPS][this.numCovariates + 1];
        this.logScaleCoefficients = new double[NUM_GROUPS][this.numCovariates + 1];
        this.numObservations = new long[NUM_GROUPS];
        this.sumOdds = new double[NUM_GROUPS];
        this.sumLogOdds = new double[NUM_GROUPS];
        this.isFitted = new boolean[NUM_GROUPS];
    }

    @Override
    protected double getReferenceHeight(Node node) {
        return this.oldestTipHeights[node.getNr()];
    }

    @Override
    protected boolean isAnchoredAtRoot() {
        return false;
    }

    @Override
    protected double proposeHeights(Node selectedNode, double scaler) {
        this.selectedNode = selectedNode;
        return super.proposeHeights(selectedNode, scaler);
    }

    /**
     * Learns from the odds of the current state, maps them to the normal space of the gamma distribution of
     * the group of the selected node, steps there by the log of the scaler and maps back.
     */
    @Override
    protected double proposeRatio(double ratio, double scaler) {
        double odds = ratio / (1.0 - ratio);
        int group = this.computeGroup(this.selectedNode);

        this.readFeatures();
        this.learnGamma(group, odds);
        this.gamma = this.createGamma(group);

        double newZ = this.mapOddsToNormal(odds) + Math.log(scaler);
        double newOdds = this.mapNormalToOdds(newZ);
        return newOdds / (1.0 + newOdds);
    }

    /**
     * In the normal space z of the Rosenblatt transform, the step is symmetric. The density of the ratio r
     * in the normal space is φ(z) |dr/dz|, with the density f(odds) / (1 - r)² of the ratio from the gamma
     * density of the odds, so the log Hastings ratio is
     * log f_r(r) - log φ(z) - log f_r(r') + log φ(z'). The gamma distribution is the same in both
     * directions, since the group and the tree prior parameters do not change.
     */
    @Override
    protected double computeLogHastingsRatio(double ratio, double newRatio) {
        double logTransport = this.computeLogTransportDensity(ratio);
        double newLogTransport = this.computeLogTransportDensity(newRatio);
        return logTransport - newLogTransport;
    }

    /** Returns log f_r(r) - log φ(z), the log derivative of the normal variable z in the ratio r. */
    private double computeLogTransportDensity(double ratio) {
        double odds = ratio / (1.0 - ratio);
        double z = this.mapOddsToNormal(odds);
        double logRatioDensity = this.gamma.logDensity(odds) - 2.0 * Math.log1p(-ratio);
        return logRatioDensity - NORMAL.logDensity(z);
    }

    /** Returns Φ⁻¹(F(odds)), computed from the survival function in the upper half for precision. */
    private double mapOddsToNormal(double odds) {
        double probability = this.gamma.cumulativeProbability(odds);
        if (probability < 0.5) {
            return NORMAL.inverseCumulativeProbability(probability);
        }
        return NORMAL.inverseSurvivalProbability(this.gamma.survivalProbability(odds));
    }

    /** Returns F⁻¹(Φ(z)), the inverse of {@link #mapOddsToNormal}. */
    private double mapNormalToOdds(double z) {
        if (z < 0.0) {
            return this.gamma.inverseCumulativeProbability(NORMAL.cumulativeProbability(z));
        }
        return this.gamma.inverseSurvivalProbability(NORMAL.survivalProbability(z));
    }

    /* Grouping */

    /**
     * Returns the group of the node, the base-2 logarithm of one plus the number of mutations on its branch
     * and on the branches of its children, rounded down. A move changes the times of the mutations, not their number.
     */
    private int computeGroup(Node node) {
        int numMutations = this.countBranchMutations(node);
        for (Node child : node.getChildren()) {
            numMutations += this.countBranchMutations(child);
        }
        return Math.min(31 - Integer.numberOfLeadingZeros(numMutations + 1), NUM_GROUPS - 1);
    }

    private int countBranchMutations(Node node) {
        return node.isRoot() ? 0 : this.mutations.getMutations(node).size();
    }

    /* Tree prior parameters */

    private int countCovariates() {
        int numValues = 0;
        for (StateNode parameter : this.treePriorParametersInput.get()) {
            if (parameter instanceof RealScalar<?>) {
                numValues++;
            } else if (parameter instanceof RealVector<?> vector) {
                numValues += vector.size();
            } else if (parameter instanceof IntVector<?> vector) {
                numValues += vector.size();
            } else {
                throw new IllegalArgumentException("The tree prior parameter " + parameter.getID() + " is neither a real scalar nor a real or integer vector.");
            }
        }
        return numValues;
    }

    /** Reads the current values of the tree prior parameters, updates their running moments and standardises them into the features. */
    private void readFeatures() {
        int index = 0;
        for (StateNode parameter : this.treePriorParametersInput.get()) {
            if (parameter instanceof RealScalar<?> scalar) {
                this.covariateValues[index++] = scalar.get();
            } else if (parameter instanceof RealVector<?> vector) {
                for (int i = 0; i < vector.size(); i++) {
                    this.covariateValues[index++] = vector.get(i);
                }
            } else if (parameter instanceof IntVector<?> vector) {
                for (int i = 0; i < vector.size(); i++) {
                    this.covariateValues[index++] = vector.get(i);
                }
            }
        }

        this.numCovariateObservations++;
        this.features[0] = 1.0;
        for (int i = 0; i < this.numCovariates; i++) {
            double delta = this.covariateValues[i] - this.covariateMeans[i];
            this.covariateMeans[i] += delta / this.numCovariateObservations;
            this.covariateSquaredDeviations[i] += delta * (this.covariateValues[i] - this.covariateMeans[i]);

            double standardDeviation = Math.sqrt(this.covariateSquaredDeviations[i] / Math.max(this.numCovariateObservations - 1, 1));
            double standardised = standardDeviation > 1e-12 ? (this.covariateValues[i] - this.covariateMeans[i]) / standardDeviation : 0.0;
            this.features[i + 1] = Math.max(Math.min(standardised, MAX_ABS_STANDARDISED_VALUE), -MAX_ABS_STANDARDISED_VALUE);
        }
    }

    /* Learning */

    /**
     * Adds the odds to the statistics of the group. Once the group has enough observations, its intercepts are set
     * to the maximum likelihood fit of a gamma distribution, and from then on, every observation moves the
     * coefficients by one step along the gradient of the gamma log-likelihood of the odds.
     */
    private void learnGamma(int group, double odds) {
        this.numObservations[group]++;
        this.sumOdds[group] += odds;
        this.sumLogOdds[group] += Math.log(odds);

        if (!this.isFitted[group]) {
            if (this.numObservations[group] >= this.minObservationsInput.get()) {
                this.fitIntercepts(group);
            }
            return;
        }

        double logShape = this.computeLinearPredictor(this.logShapeCoefficients[group], MAX_ABS_LOG_SHAPE);
        double logScale = this.computeLinearPredictor(this.logScaleCoefficients[group], MAX_ABS_LOG_SCALE);
        double shape = Math.exp(logShape);
        double scale = Math.exp(logScale);

        double logShapeGradient = shape * (Math.log(odds) - this.computeDigamma(shape) - logScale);
        double logScaleGradient = odds / scale - shape;
        double step = this.learningRateInput.get() / Math.sqrt(1.0 + this.numObservations[group] / this.learningDecayInput.get());
        for (int i = 0; i < this.features.length; i++) {
            this.logShapeCoefficients[group][i] += step * this.clipGradient(logShapeGradient) * this.features[i];
            this.logScaleCoefficients[group][i] += step * this.clipGradient(logScaleGradient) * this.features[i];
        }
    }

    /**
     * Sets the intercepts of the group to the maximum likelihood fit of a gamma distribution, with the shape from the
     * closed-form approximation of Minka for s = log(mean) - mean(log), and keeps them if the statistics are degenerate.
     */
    private void fitIntercepts(int group) {
        double count = this.numObservations[group];
        double mean = this.sumOdds[group] / count;
        double s = Math.log(mean) - this.sumLogOdds[group] / count;
        if (!(s > 0.0) || !Double.isFinite(s)) {
            return;
        }

        double shape = (3.0 - s + Math.sqrt((s - 3.0) * (s - 3.0) + 24.0 * s)) / (12.0 * s);
        double scale = mean / shape;
        if (shape > 0.0 && scale > 0.0 && Double.isFinite(shape) && Double.isFinite(scale)) {
            this.logShapeCoefficients[group][0] = Math.log(shape);
            this.logScaleCoefficients[group][0] = Math.log(scale);
            this.isFitted[group] = true;
        }
    }

    private GammaDistribution createGamma(int group) {
        if (!this.isFitted[group]) {
            return GammaDistribution.of(1.0, 1.0);
        }

        double shape = Math.exp(this.computeLinearPredictor(this.logShapeCoefficients[group], MAX_ABS_LOG_SHAPE));
        double scale = Math.exp(this.computeLinearPredictor(this.logScaleCoefficients[group], MAX_ABS_LOG_SCALE));
        return GammaDistribution.of(shape, scale);
    }

    /** Returns the intercept plus the linear combination of the standardised tree prior parameters, limited to a sane range. */
    private double computeLinearPredictor(double[] coefficients, double maxAbsValue) {
        double predictor = 0.0;
        for (int i = 0; i < coefficients.length; i++) {
            predictor += coefficients[i] * this.features[i];
        }
        return Math.max(Math.min(predictor, maxAbsValue), -maxAbsValue);
    }

    private double clipGradient(double gradient) {
        return Math.max(Math.min(gradient, MAX_ABS_GRADIENT), -MAX_ABS_GRADIENT);
    }

    /** Returns ψ(x), from the recurrence ψ(x) = ψ(x + 1) - 1 / x up to x ≥ 6 and the asymptotic series there. */
    private double computeDigamma(double x) {
        double result = 0.0;
        while (x < 6.0) {
            result -= 1.0 / x;
            x += 1.0;
        }

        double inverseSquare = 1.0 / (x * x);
        return result + Math.log(x) - 0.5 / x
                - inverseSquare * (1.0 / 12.0 - inverseSquare * (1.0 / 120.0 - inverseSquare * (1.0 / 252.0)));
    }

}
