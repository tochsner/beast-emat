package emat.operators.treeprior;

import beast.base.evolution.tree.Node;
import beast.base.evolution.tree.Tree;

import java.util.Arrays;

/**
 * Summarises a tree by the logarithms of its length, its height, quantiles of the heights of its internal nodes and its
 * coalescent floor, which is the length of all branches after the oldest tip.
 */
public final class TreeStatistics {

    private static final double[] QUANTILES = {0.05, 0.1, 0.25, 0.5, 0.75, 0.9, 0.95};
    private static final double MIN_VALUE = 1.0E-12;

    /** The number of statistics that {@link #compute} returns. */
    public static final int COUNT = 3 + QUANTILES.length;

    private TreeStatistics() {
    }

    /** Returns the logarithms of the tree length, the tree height, the quantiles of the internal node heights and the floor. */
    public static double[] compute(Tree tree) {
        Node[] nodes = tree.getNodesAsArray();

        double oldestTipHeight = 0.0;
        int numInternal = 0;
        for (Node node : nodes) {
            if (node.isLeaf()) {
                oldestTipHeight = Math.max(oldestTipHeight, node.getHeight());
            } else {
                numInternal++;
            }
        }

        double[] internalHeights = new double[numInternal];
        double length = 0.0;
        double floor = 0.0;
        int index = 0;
        for (Node node : nodes) {
            if (!node.isLeaf()) {
                internalHeights[index++] = node.getHeight();
            }
            if (!node.isRoot()) {
                length += node.getLength();
                floor += Math.max(0.0, Math.min(node.getParent().getHeight(), oldestTipHeight) - node.getHeight());
            }
        }
        Arrays.sort(internalHeights);

        double[] statistics = new double[COUNT];
        statistics[0] = length;
        statistics[1] = tree.getRoot().getHeight();
        for (int i = 0; i < QUANTILES.length; i++) {
            statistics[2 + i] = TreeStatistics.computeQuantile(internalHeights, QUANTILES[i]);
        }
        statistics[COUNT - 1] = floor;

        for (int i = 0; i < COUNT; i++) {
            statistics[i] = Math.log(Math.max(statistics[i], MIN_VALUE));
        }
        return statistics;
    }

    /** Returns the quantile of the sorted values by linear interpolation between the closest ranks. */
    private static double computeQuantile(double[] sorted, double probability) {
        double position = probability * (sorted.length - 1);
        int lower = (int) Math.floor(position);
        int upper = Math.min(lower + 1, sorted.length - 1);
        return sorted[lower] + (position - lower) * (sorted[upper] - sorted[lower]);
    }

}
