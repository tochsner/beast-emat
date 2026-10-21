package emat.operators.mdspr;

import beast.base.evolution.tree.Node;
import beast.base.evolution.tree.Tree;

import java.util.Arrays;

/**
 * Only picks tips, in proportion to a power of the autocorrelation time of the size of their
 * sister clade, so that tips that rarely change their sister are picked more often. The time is
 * estimated from the history of the chain: every given number of proposals, the sister clade
 * sizes of all tips are sampled, and the weights are refreshed. The weights only change in
 * {@link #prepare()}. They depend on the whole history, with the influence of a new sample
 * shrinking as the samples add up.
 */
public class SisterActSubtreeSelector extends SubtreeSelector {

    private final double power;
    private final double floor;
    private final int interval;
    private final int warmup;

    // the weight of every tip, which only changes with the samples of the sister clade sizes
    private final double[] tipWeights;

    // the number of calls so far that count towards the sampling interval, the number of samples so far, and the number of those that count towards the estimates
    private long numCalls;
    private long numSamples;
    private int numEstimateSamples;

    // per tip, the running sums over the estimating samples of the log size of the sister clade, its square and the squared change since the previous sample, and the previous log size
    private final double[] sums;
    private final double[] squareSums;
    private final double[] stepSquareSums;
    private final double[] lastLogSizes;

    /**
     * @param power    the power to which the estimated autocorrelation time is raised. With 0, all tips are picked equally often; larger values concentrate the picks on the slowest tips
     * @param floor    the share of the picks that is spread uniformly over all tips, so that every tip keeps being picked
     * @param interval the number of proposals between two samples of the sister clade sizes
     * @param warmup   the number of samples that are discarded before the estimation starts, while all tips are picked equally often
     */
    public SisterActSubtreeSelector(Tree tree, double power, double floor, int interval, int warmup) {
        super(tree);
        if (power < 0.0) {
            throw new IllegalArgumentException("sisterActPower must not be negative.");
        }
        if (floor < 0.0 || floor > 1.0) {
            throw new IllegalArgumentException("sisterActFloor must be between 0 and 1.");
        }
        if (interval < 1) {
            throw new IllegalArgumentException("sisterActInterval must be positive.");
        }
        if (warmup < 0) {
            throw new IllegalArgumentException("sisterActWarmup must not be negative.");
        }

        this.power = power;
        this.floor = floor;
        this.interval = interval;
        this.warmup = warmup;

        int numNodes = tree.getNodeCount();
        this.tipWeights = new double[numNodes];
        Arrays.fill(this.tipWeights, 1.0 / tree.getLeafNodeCount());
        this.sums = new double[numNodes];
        this.squareSums = new double[numNodes];
        this.stepSquareSums = new double[numNodes];
        this.lastLogSizes = new double[numNodes];
    }

    @Override
    public void prepare() {
        boolean isSampling = this.numCalls % this.interval == 0;
        this.numCalls++;
        if (!isSampling) {
            return;
        }

        this.numSamples++;
        if (this.numSamples <= this.warmup) {
            return;
        }

        this.sampleSisterSizes();
        this.updateTipWeights();
    }

    @Override
    protected double computeWeight(Node node) {
        return node.isLeaf() ? this.tipWeights[node.getNr()] : 0.0;
    }

    @Override
    public int getNumSamples() {
        return this.numEstimateSamples;
    }

    /** Adds the log size of the sister clade of every tip to its running sums. */
    private void sampleSisterSizes() {
        int[] leafCounts = this.computeLeafCounts();

        int numTips = this.tree.getLeafNodeCount();
        for (int nr = 0; nr < numTips; nr++) {
            Node tip = this.tree.getNode(nr);
            double logSize = Math.log(leafCounts[this.getSister(tip).getNr()]);

            if (this.numEstimateSamples > 0) {
                double step = logSize - this.lastLogSizes[nr];
                this.stepSquareSums[nr] += step * step;
            }
            this.sums[nr] += logSize;
            this.squareSums[nr] += logSize * logSize;
            this.lastLogSizes[nr] = logSize;
        }
        this.numEstimateSamples++;
    }

    private Node getSister(Node tip) {
        Node parent = tip.getParent();
        return parent.getChild(0) == tip ? parent.getChild(1) : parent.getChild(0);
    }

    /**
     * Estimates the autocorrelation time of the log sister clade size of every tip as
     * 2 var / E[step²], which is 1 / (1 - r) for the lag-1 autocorrelation r, and sets the
     * weight of the tip to the estimate raised to the power. A tip whose sister has not
     * changed has the time 1, and the estimate is limited to between 1/2 and the number of
     * samples. The weights are normalised and then mixed with a uniform share over the tips.
     * Until three samples are in, all tips keep the same weight.
     */
    private void updateTipWeights() {
        int numSamples = this.numEstimateSamples;
        if (numSamples < 3) {
            return;
        }

        int numTips = this.tree.getLeafNodeCount();
        double maxLogWeight = Double.NEGATIVE_INFINITY;
        for (int nr = 0; nr < numTips; nr++) {
            double mean = this.sums[nr] / numSamples;
            double variance = this.squareSums[nr] / numSamples - mean * mean;
            double meanStepSquare = this.stepSquareSums[nr] / (numSamples - 1);

            double act = variance > 1e-12 && meanStepSquare > 0.0 ? 2.0 * variance / meanStepSquare : 1.0;
            double logWeight = this.power * Math.log(Math.min(Math.max(act, 0.5), numSamples));
            this.tipWeights[nr] = logWeight;
            maxLogWeight = Math.max(maxLogWeight, logWeight);
        }

        double totalWeight = 0.0;
        for (int nr = 0; nr < numTips; nr++) {
            this.tipWeights[nr] = Math.exp(this.tipWeights[nr] - maxLogWeight);
            totalWeight += this.tipWeights[nr];
        }
        for (int nr = 0; nr < numTips; nr++) {
            this.tipWeights[nr] = (1.0 - this.floor) * this.tipWeights[nr] / totalWeight + this.floor / numTips;
        }
    }
}
