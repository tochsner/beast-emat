package emat.tree;

import beast.base.core.Description;
import beast.base.evolution.tree.Node;
import beast.base.evolution.tree.Tree;
import beast.base.evolution.tree.TreeIntervals;

@Description("Extracts the intervals from a binary tree. The order of the nodes by height is kept between " +
        "recalculations and re-sorted with insertion sort, which takes linear time when only a few node " +
        "heights have changed. Unlike TreeIntervals, the lineages added and removed in each interval are " +
        "not recorded.")
public class InsertionSortTreeIntervals extends TreeIntervals {

    @Override
    protected void calculateIntervals() {
        Tree tree = this.treeInput.get();
        final int nodeCount = tree.getNodeCount();
        final int leafCount = tree.getLeafNodeCount();

        if (this.times == null || this.times.length != nodeCount) {
            this.times = new double[nodeCount];
            this.indices = new int[nodeCount];
            for (int i = 0; i < nodeCount; i++) {
                this.indices[i] = i;
            }
        }
        if (this.intervals == null || this.intervals.length != nodeCount) {
            this.intervals = new double[nodeCount];
            this.storedIntervals = new double[nodeCount];
            this.lineageCounts = new int[nodeCount];
            this.storedLineageCounts = new int[nodeCount];
        }

        Node[] nodes = tree.getNodesAsArray();
        for (int i = 0; i < nodeCount; i++) {
            this.times[i] = nodes[i].getHeight();
        }
        this.sortIndices();

        // leaves are numbered below leafCount, and every internal node merges two lineages into one
        double start = this.times[this.indices[0]];
        int numLines = 0;
        int nodeNo = 0;
        this.intervalCount = 0;
        while (nodeNo < nodeCount) {
            int lineagesRemoved = 0;
            int lineagesAdded = 0;

            final double finish = this.times[this.indices[nodeNo]];
            double next;

            do {
                final boolean isLeaf = this.indices[nodeNo] < leafCount;
                nodeNo += 1;
                if (isLeaf) {
                    lineagesAdded += 1;
                } else {
                    lineagesRemoved += 1;
                    // without a multifurcation limit, each coalescence is a separate event
                    if (this.multifurcationLimit == 0.0) {
                        break;
                    }
                }

                if (nodeNo < nodeCount) {
                    next = this.times[this.indices[nodeNo]];
                } else {
                    break;
                }
            } while (Math.abs(next - finish) <= this.multifurcationLimit);

            // sample event
            if (lineagesAdded > 0) {
                if (this.intervalCount > 0 || finish - start > this.multifurcationLimit) {
                    this.intervals[this.intervalCount] = finish - start;
                    this.lineageCounts[this.intervalCount] = numLines;
                    this.intervalCount += 1;
                }
                start = finish;
            }
            numLines += lineagesAdded;

            // coalescent event
            if (lineagesRemoved > 0) {
                this.intervals[this.intervalCount] = finish - start;
                this.lineageCounts[this.intervalCount] = numLines;
                this.intervalCount += 1;
                start = finish;
            }
            numLines -= lineagesRemoved;
        }

        this.intervalsKnown = true;
    }

    /**
     * Sorts the indices by ascending time with insertion sort, starting from their current
     * order. This takes time linear in the number of indices plus the number of pairs that are
     * out of order, so it is fast if the indices are already nearly sorted.
     */
    protected void sortIndices() {
        final double[] times = this.times;
        final int[] indices = this.indices;
        for (int i = 1; i < indices.length; i++) {
            final int index = indices[i];
            final double time = times[index];
            int j = i - 1;
            while (j >= 0 && times[indices[j]] > time) {
                indices[j + 1] = indices[j];
                j--;
            }
            indices[j + 1] = index;
        }
    }

}
