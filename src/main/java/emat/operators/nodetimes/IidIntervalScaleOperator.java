package emat.operators.nodetimes;

import beast.base.core.Description;
import beast.base.core.Input;
import beast.base.evolution.tree.Node;
import emat.diagnostics.DiagnosedOperator;
import emat.diagnostics.StepAnnotations;

import java.util.Arrays;
import java.util.Random;

@Description("Diagnostic proposal that scales the interval between every internal node and its older child by its own " +
        "independent log-normal factor, and maps the mutations linearly within their branches. The Hastings ratio is " +
        "not computed: the proposal is meant to be probed for the change of the posterior and never used in a chain. It " +
        "draws its random numbers from its own generator, so that it leaves the random stream of the chain untouched.")
public class IidIntervalScaleOperator extends ScaleTunedOperator implements DiagnosedOperator {

    final public Input<Double> logStandardDeviationInput = new Input<>("logStandardDeviation", "standard deviation of the log of the factor of every interval", 0.03);
    final public Input<Long> seedInput = new Input<>("seed", "seed of the generator of the factors", 12345L);

    private Random random;

    @Override
    public void initAndValidate() {
        super.initAndValidate();
        this.random = new Random(this.seedInput.get());
    }

    /**
     * Multiplies the interval of every internal node by exp(sd z) with an independent standard normal z, places every
     * node above its higher child and returns zero as the log Hastings ratio.
     */
    @Override
    public double proposal() {
        Node[] nodes = this.tree.getNodesAsArray();
        double[] oldHeights = new double[nodes.length];
        for (Node node : nodes) {
            oldHeights[node.getNr()] = node.getHeight();
        }

        double sd = this.logStandardDeviationInput.get();
        double[] logFactors = new double[nodes.length];
        double[] intervals = new double[nodes.length];
        double lengthBefore = NodeTimes.computeTreeLength(this.tree);

        this.scaleIntervals(this.tree.getRoot(), sd, oldHeights, logFactors, intervals);

        this.mutations.setSanityCheckingEnabled(false);
        try {
            BranchTimeMapper.mapMutations(this.mutations, nodes, oldHeights, this);
        } finally {
            this.mutations.setSanityCheckingEnabled(true);
        }

        this.annotate(nodes, oldHeights, logFactors, intervals, NodeTimes.computeTreeLength(this.tree) / lengthBefore);

        return 0.0;
    }

    /**
     * Scales the interval of the node after everything below it, where the old interval is measured against the old
     * height of the higher child. Records the log factor and the old interval by node number.
     */
    private void scaleIntervals(Node node, double sd, double[] oldHeights, double[] logFactors, double[] intervals) {
        if (node.isLeaf()) {
            return;
        }

        this.scaleIntervals(node.getLeft(), sd, oldHeights, logFactors, intervals);
        this.scaleIntervals(node.getRight(), sd, oldHeights, logFactors, intervals);

        double oldInterval = oldHeights[node.getNr()] - Math.max(oldHeights[node.getLeft().getNr()], oldHeights[node.getRight().getNr()]);
        double logFactor = sd * this.random.nextGaussian();

        node.setHeight(Math.max(node.getLeft().getHeight(), node.getRight().getHeight()) + oldInterval * Math.exp(logFactor));
        logFactors[node.getNr()] = logFactor;
        intervals[node.getNr()] = oldInterval;
    }

    @Override
    public String[] getAnnotationNames() {
        return new String[]{"meanLogFactor", "intervalWeightedMeanLogFactor", "logTreeLengthRatio", "slopeOnHeight",
                "meanLogFactorYoungThird", "meanLogFactorOldThird", "rootLogFactor", "meanLogFactorShortQuarter",
                "meanLogFactorLongQuarter", "logIntervalSumRatio"};
    }

    /** Writes summaries of the log factors of the internal nodes: overall, weighted by the interval, by position and by size. */
    private void annotate(Node[] nodes, double[] oldHeights, double[] logFactors, double[] intervals, double lengthRatio) {
        int numInternal = this.tree.getInternalNodeCount();
        Node[] internalNodes = Arrays.stream(nodes).filter(node -> !node.isLeaf()).toArray(Node[]::new);

        double sumFactor = 0.0;
        double sumWeighted = 0.0;
        double sumIntervals = 0.0;
        double sumNewIntervals = 0.0;
        for (Node node : internalNodes) {
            double logFactor = logFactors[node.getNr()];
            double interval = intervals[node.getNr()];
            sumFactor += logFactor;
            sumWeighted += interval * logFactor;
            sumIntervals += interval;
            sumNewIntervals += interval * Math.exp(logFactor);
        }

        // the slope of the log factor on the old height in units of the standard deviation of the heights
        double meanHeight = Arrays.stream(internalNodes).mapToDouble(node -> oldHeights[node.getNr()]).average().orElse(0.0);
        double covariance = 0.0;
        double varianceHeight = 0.0;
        for (Node node : internalNodes) {
            double centred = oldHeights[node.getNr()] - meanHeight;
            covariance += centred * logFactors[node.getNr()];
            varianceHeight += centred * centred;
        }
        double slope = covariance / Math.sqrt(varianceHeight * numInternal);

        Node[] byHeight = Arrays.stream(internalNodes).sorted((a, b) -> Double.compare(oldHeights[a.getNr()], oldHeights[b.getNr()])).toArray(Node[]::new);
        Node[] byInterval = Arrays.stream(internalNodes).sorted((a, b) -> Double.compare(intervals[a.getNr()], intervals[b.getNr()])).toArray(Node[]::new);

        StepAnnotations.set(0, sumFactor / numInternal);
        StepAnnotations.set(1, sumWeighted / sumIntervals);
        StepAnnotations.set(2, Math.log(lengthRatio));
        StepAnnotations.set(3, slope);
        StepAnnotations.set(4, this.meanLogFactor(byHeight, 0, numInternal / 3, logFactors));
        StepAnnotations.set(5, this.meanLogFactor(byHeight, numInternal - numInternal / 3, numInternal, logFactors));
        StepAnnotations.set(6, logFactors[this.tree.getRoot().getNr()]);
        StepAnnotations.set(7, this.meanLogFactor(byInterval, 0, numInternal / 4, logFactors));
        StepAnnotations.set(8, this.meanLogFactor(byInterval, numInternal - numInternal / 4, numInternal, logFactors));
        StepAnnotations.set(9, Math.log(sumNewIntervals / sumIntervals));
    }

    private double meanLogFactor(Node[] sortedNodes, int from, int to, double[] logFactors) {
        double sum = 0.0;
        for (int i = from; i < to; i++) {
            sum += logFactors[sortedNodes[i].getNr()];
        }
        return sum / (to - from);
    }

}
