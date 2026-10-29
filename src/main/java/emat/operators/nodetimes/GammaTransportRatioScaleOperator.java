package emat.operators.nodetimes;

import beast.base.core.Description;
import beast.base.core.Input;
import beast.base.evolution.tree.Node;
import org.apache.commons.statistics.distribution.GammaDistribution;
import org.apache.commons.statistics.distribution.NormalDistribution;

import java.util.Arrays;

@Description("Moves a random internal node in the ratio parameterisation of the node heights, with the ratio " +
        "r = (h - τ) / (h_p - τ) of its height and the height of its parent, both above the oldest tip τ below the node. " +
        "For every clade size class, a gamma distribution is learned for the odds r / (1 - r) of the nodes of that class " +
        "from the states of the chain. The odds are mapped to a standard normal variable by the univariate Rosenblatt " +
        "transform z = Φ⁻¹(F(odds)), moved by a step from the kernel in z, and mapped back. The step is tuned for every " +
        "class like the scale factor of the ratio operator. The fit changes less and less as the chain grows, " +
        "which keeps the adaptation diminishing.")
public class GammaTransportRatioScaleOperator extends AnchoredRatioScaleOperator {

    final public Input<Integer> minObservationsInput = new Input<>("minObservations", "number of observed odds of a clade size class after which its gamma distribution is first fitted", 100);
    final public Input<Integer> refitIntervalInput = new Input<>("refitInterval", "number of observed odds of a clade size class between two refits of its gamma distribution", 100);

    private static final int MAX_NUM_SIZE_CLASSES = 32;

    private static final NormalDistribution NORMAL = NormalDistribution.of(0.0, 1.0);

    // per clade size class, the number of observed odds, the sum of the odds and of their logarithms
    private long[] numObservations;
    private double[] sumOdds;
    private double[] sumLogOdds;

    // per clade size class, the fitted gamma distribution of the odds, which is a standard exponential before the first fit
    private GammaDistribution[] gammas;

    @Override
    public void initAndValidate() {
        super.initAndValidate();

        this.numObservations = new long[MAX_NUM_SIZE_CLASSES];
        this.sumOdds = new double[MAX_NUM_SIZE_CLASSES];
        this.sumLogOdds = new double[MAX_NUM_SIZE_CLASSES];
        this.gammas = new GammaDistribution[MAX_NUM_SIZE_CLASSES];
        Arrays.fill(this.gammas, GammaDistribution.of(1.0, 1.0));
    }

    @Override
    protected double getReferenceHeight(Node node) {
        return this.oldestTipHeights[node.getNr()];
    }

    @Override
    protected boolean isAnchoredAtRoot() {
        return false;
    }

    /**
     * Observes the odds of the current state for the learned distribution, maps them to the normal space of the
     * distribution of the clade size class, steps there by the log of the scaler and maps back.
     */
    @Override
    protected double proposeRatio(double ratio, double scaler) {
        double odds = ratio / (1.0 - ratio);
        this.observeOdds(odds);

        GammaDistribution gamma = this.gammas[this.lastSizeClass];
        double newZ = this.mapOddsToNormal(gamma, odds) + Math.log(scaler);
        double newOdds = this.mapNormalToOdds(gamma, newZ);
        return newOdds / (1.0 + newOdds);
    }

    /**
     * In the normal space z of the Rosenblatt transform, the step is symmetric. The density of the ratio r
     * in the normal space is φ(z) |dr/dz|, with the density f(odds) / (1 - r)² of the ratio from the gamma
     * density of the odds, so the log Hastings ratio is
     * log f_r(r) - log φ(z) - log f_r(r') + log φ(z').
     */
    @Override
    protected double computeLogHastingsRatio(double ratio, double newRatio) {
        GammaDistribution gamma = this.gammas[this.lastSizeClass];
        double odds = ratio / (1.0 - ratio);
        double newOdds = newRatio / (1.0 - newRatio);

        double logTransport = this.computeLogTransportDensity(gamma, odds, ratio);
        double newLogTransport = this.computeLogTransportDensity(gamma, newOdds, newRatio);
        return logTransport - newLogTransport;
    }

    /** Returns log f_r(r) - log φ(z), the log derivative of the normal variable z in the ratio r. */
    private double computeLogTransportDensity(GammaDistribution gamma, double odds, double ratio) {
        double z = this.mapOddsToNormal(gamma, odds);
        double logRatioDensity = gamma.logDensity(odds) - 2.0 * Math.log1p(-ratio);
        return logRatioDensity - NORMAL.logDensity(z);
    }

    /** Returns Φ⁻¹(F(odds)), computed from the survival function in the upper half for precision. */
    private double mapOddsToNormal(GammaDistribution gamma, double odds) {
        double probability = gamma.cumulativeProbability(odds);
        if (probability < 0.5) {
            return NORMAL.inverseCumulativeProbability(probability);
        }
        return NORMAL.inverseSurvivalProbability(gamma.survivalProbability(odds));
    }

    /** Returns F⁻¹(Φ(z)), the inverse of {@link #mapOddsToNormal}. */
    private double mapNormalToOdds(GammaDistribution gamma, double z) {
        if (z < 0.0) {
            return gamma.inverseCumulativeProbability(NORMAL.cumulativeProbability(z));
        }
        return gamma.inverseSurvivalProbability(NORMAL.survivalProbability(z));
    }

    /** Adds the odds to the statistics of the clade size class of the last proposal and refits its gamma distribution when due. */
    private void observeOdds(double odds) {
        int sizeClass = this.lastSizeClass;
        this.numObservations[sizeClass]++;
        this.sumOdds[sizeClass] += odds;
        this.sumLogOdds[sizeClass] += Math.log(odds);

        long numObservations = this.numObservations[sizeClass];
        boolean isDue = numObservations == this.minObservationsInput.get()
                || (numObservations > this.minObservationsInput.get() && numObservations % this.refitIntervalInput.get() == 0);
        if (isDue) {
            this.fitGamma(sizeClass);
        }
    }

    /**
     * Fits the gamma distribution of the clade size class by maximum likelihood, with the shape from the
     * closed-form approximation of Minka for s = log(mean) - mean(log), and keeps the old fit if the
     * statistics are degenerate.
     */
    private void fitGamma(int sizeClass) {
        double count = this.numObservations[sizeClass];
        double mean = this.sumOdds[sizeClass] / count;
        double s = Math.log(mean) - this.sumLogOdds[sizeClass] / count;
        if (!(s > 0.0) || !Double.isFinite(s)) {
            return;
        }

        double shape = (3.0 - s + Math.sqrt((s - 3.0) * (s - 3.0) + 24.0 * s)) / (12.0 * s);
        double scale = mean / shape;
        if (shape > 0.0 && scale > 0.0 && Double.isFinite(shape) && Double.isFinite(scale)) {
            this.gammas[sizeClass] = GammaDistribution.of(shape, scale);
        }
    }

}
