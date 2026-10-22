package emat.helper;

import beast.base.core.Description;
import beast.base.evolution.tree.Node;
import beast.base.evolution.tree.Tree;
import emat.state.Mutation;
import emat.state.Mutations;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

@Description("Summaries of a tree and its mutations that could carry the slow part of the posterior.")
public class TreeSummaries {

    private final Tree tree;
    private final Mutations mutations;

    public TreeSummaries(Tree tree, Mutations mutations) {
        this.tree = tree;
        this.mutations = mutations;
    }

    public static final String[] NAMES = {
            "logTreeLength", "logRootHeight", "logCoalescentExposure", "logSumInternalHeights", "logMeanInternalHeight",
            "logInternalHeightQ10", "logInternalHeightQ25", "logInternalHeightQ50", "logInternalHeightQ75", "logInternalHeightQ90",
            "meanInternalOverRoot", "logExternalLength", "logInternalLength", "logExternalOverInternal",
            "logLineagesAt10", "logLineagesAt25", "logLineagesAt50", "logLineagesAt75",
            "logSumSquaredIntervals", "logLongestInterval", "logSmallestInterval", "logIntervalSum",
            "logSackin", "logColless", "logCherries",
            "logMutationCount", "logMeanMutationHeight", "fractionExternalMutations", "logMutationsPerLength",
            "logHeightOfLastCoalescenceCluster", "logSecondRootInterval", "logMeanTipDepth"
    };

    /**
     * Computes summaries of the tree and its mutations that could carry the slow part of the posterior.
     * Heights are summarised on the log scale because the sampler moves them multiplicatively.
     */
    public double[] computeSummaries() {
        Node[] nodes = this.tree.getNodesAsArray();
        double rootHeight = this.tree.getRoot().getHeight();

        List<Double> internalHeights = new ArrayList<>();
        double treeLength = 0.0;
        double externalLength = 0.0;
        double internalLength = 0.0;
        int numCherries = 0;
        double sumSquaredIntervals = 0.0;
        double longestInterval = 0.0;
        double smallestInterval = Double.POSITIVE_INFINITY;
        double intervalSum = 0.0;
        double sumTipDepth = 0.0;
        int numTips = 0;

        for (Node node : nodes) {
            if (!node.isRoot()) {
                double length = node.getParent().getHeight() - node.getHeight();
                treeLength += length;
                if (node.isLeaf()) {
                    externalLength += length;
                } else {
                    internalLength += length;
                }
            }

            if (node.isLeaf()) {
                numTips++;
                sumTipDepth += rootHeight - node.getHeight();
                continue;
            }

            internalHeights.add(node.getHeight());

            double interval = node.getHeight() - Math.max(node.getLeft().getHeight(), node.getRight().getHeight());
            sumSquaredIntervals += interval * interval;
            longestInterval = Math.max(longestInterval, interval);
            smallestInterval = Math.min(smallestInterval, interval);
            intervalSum += interval;

            if (node.getLeft().isLeaf() && node.getRight().isLeaf()) {
                numCherries++;
            }
        }

        double[] sortedHeights = internalHeights.stream().mapToDouble(Double::doubleValue).sorted().toArray();
        double sumHeights = Arrays.stream(sortedHeights).sum();
        double meanHeight = sumHeights / sortedHeights.length;

        double[] imbalance = this.computeSackinAndColless();

        int numMutations = 0;
        double sumMutationHeight = 0.0;
        int numExternalMutations = 0;
        if (this.mutations != null) {
            for (Node node : nodes) {
                if (node.isRoot()) continue;
                for (Mutation mutation : this.mutations.getMutations(node)) {
                    numMutations++;
                    sumMutationHeight += mutation.time();
                    if (node.isLeaf()) numExternalMutations++;
                }
            }
        }

        double secondRootInterval = Math.max(
                this.tree.getRoot().getLeft().isLeaf() ? 0.0 : this.tree.getRoot().getLeft().getHeight(),
                this.tree.getRoot().getRight().isLeaf() ? 0.0 : this.tree.getRoot().getRight().getHeight());

        return new double[]{
                Math.log(treeLength), Math.log(rootHeight), Math.log(this.computeCoalescentExposure(nodes)),
                Math.log(sumHeights), Math.log(meanHeight),
                Math.log(this.computeQuantile(sortedHeights, 0.10)), Math.log(this.computeQuantile(sortedHeights, 0.25)),
                Math.log(this.computeQuantile(sortedHeights, 0.50)), Math.log(this.computeQuantile(sortedHeights, 0.75)),
                Math.log(this.computeQuantile(sortedHeights, 0.90)),
                meanHeight / rootHeight, Math.log(externalLength), Math.log(internalLength), Math.log(externalLength / internalLength),
                Math.log(this.countLineagesAt(nodes, 0.10 * rootHeight)), Math.log(this.countLineagesAt(nodes, 0.25 * rootHeight)),
                Math.log(this.countLineagesAt(nodes, 0.50 * rootHeight)), Math.log(this.countLineagesAt(nodes, 0.75 * rootHeight)),
                Math.log(sumSquaredIntervals), Math.log(longestInterval), Math.log(smallestInterval), Math.log(intervalSum),
                Math.log(imbalance[0]), Math.log(1.0 + imbalance[1]), Math.log(numCherries),
                Math.log(numMutations), Math.log(sumMutationHeight / Math.max(1, numMutations)),
                numExternalMutations / (double) Math.max(1, numMutations), Math.log(numMutations / treeLength),
                Math.log(this.computeHeightOfLargestCluster()), Math.log(Math.max(secondRootInterval, 1.0E-12)),
                Math.log(sumTipDepth / numTips)
        };
    }

    private double computeQuantile(double[] sorted, double probability) {
        return sorted[(int) Math.min(sorted.length - 1, Math.floor(probability * sorted.length))];
    }

    /**
     * The coalescent exposure is the sum over the intervals between consecutive tips and coalescences of the number of
     * pairs of lineages times the length of the interval.
     */
    private double computeCoalescentExposure(Node[] nodes) {
        double[] events = new double[nodes.length];
        int[] changes = new int[nodes.length];
        Integer[] order = new Integer[nodes.length];
        for (int i = 0; i < nodes.length; i++) {
            events[i] = nodes[i].getHeight();
            changes[i] = nodes[i].isLeaf() ? 1 : -1;
            order[i] = i;
        }
        Arrays.sort(order, (a, b) -> events[a] != events[b] ? Double.compare(events[a], events[b]) : Integer.compare(changes[b], changes[a]));

        double exposure = 0.0;
        int numLineages = 0;
        double previousHeight = events[order[0]];
        for (int index : order) {
            exposure += 0.5 * numLineages * (numLineages - 1.0) * (events[index] - previousHeight);
            previousHeight = events[index];
            numLineages += changes[index];
        }
        return exposure;
    }

    private int countLineagesAt(Node[] nodes, double height) {
        int numLineages = 0;
        for (Node node : nodes) {
            if (!node.isRoot() && node.getHeight() <= height && node.getParent().getHeight() > height) {
                numLineages++;
            }
        }
        return Math.max(1, numLineages);
    }

    /**
     * Returns the Sackin index, the sum of the number of internal nodes above every tip, and the Colless index, the
     * sum over all internal nodes of the absolute difference between the numbers of tips below the two children.
     */
    private double[] computeSackinAndColless() {
        double[] result = new double[2];
        this.collectImbalance(this.tree.getRoot(), 0, result);
        return result;
    }

    private int collectImbalance(Node node, int depth, double[] result) {
        if (node.isLeaf()) {
            result[0] += depth;
            return 1;
        }

        int left = this.collectImbalance(node.getLeft(), depth + 1, result);
        int right = this.collectImbalance(node.getRight(), depth + 1, result);
        result[1] += Math.abs(left - right);
        return left + right;
    }

    /**
     * Returns the height of the highest node whose two children are both internal, which is where the tree stops
     * being made of tips attached to a single spine.
     */
    private double computeHeightOfLargestCluster() {
        double height = 0.0;
        for (Node node : this.tree.getNodesAsArray()) {
            if (!node.isLeaf() && !node.getLeft().isLeaf() && !node.getRight().isLeaf()) {
                height = Math.max(height, node.getHeight());
            }
        }
        return Math.max(height, 1.0E-12);
    }

}
