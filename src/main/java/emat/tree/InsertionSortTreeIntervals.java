package emat.tree;

import beast.base.core.Description;
import beast.base.evolution.tree.Node;
import beast.base.evolution.tree.Tree;
import beast.base.evolution.tree.TreeIntervals;

@Description("Extracts the intervals from a binary tree. The order of the nodes by height is kept between " +
        "recalculations and re-sorted with insertion sort, which takes linear time when only a few node " +
        "heights have changed. On an IncrementalStoreTree, only the nodes flagged by the tree are moved " +
        "within the order, and only the intervals between their old and new positions are updated, stored " +
        "and restored. Unlike TreeIntervals, the lineages added and removed in each interval are not recorded.")
public class InsertionSortTreeIntervals extends TreeIntervals {

    private static final int[] NO_COUNTS = new int[0];
    private static final double[] NO_INTERVALS = new double[0];

    // the times in the order of the sorted indices, so that sorting and walking the intervals read them in sequence
    protected double[] sortedTimes;

    // the position of every node in the sorted indices
    protected int[] positions;

    // the sorted indices and times at the last store, next to the stored intervals and lineage counts of the base class
    protected int[] storedIndices;
    protected double[] storedSortedTimes;

    /**
     * The range of positions whose sorted indices, sorted times, intervals or lineage counts
     * might differ from the stored ones. The range is empty if the first position is after the
     * last.
     */
    protected int firstChangedPosition = 0;
    protected int lastChangedPosition = -1;

    /**
     * Nodes whose height might have changed since the last calculation.
     */
    private int[] pendingNodeNrs;
    private boolean[] isNodePending;
    private int numPendingNodes;

    // whether the sorted order might not match the tree apart from the pending nodes
    private boolean requiresFullUpdate = true;

    // whether the stored arrays might not match the tree as it was at the last store
    private boolean isStoreOutdated = true;

    @Override
    protected boolean requiresRecalculation() {
        this.collectPendingNodes();
        return super.requiresRecalculation();
    }

    /**
     * Records the nodes flagged by the tree, as these are the only ones whose heights might have
     * changed since the tree was last cleaned. Without flagged nodes, the next calculation sorts
     * all nodes.
     */
    private void collectPendingNodes() {
        if (this.requiresFullUpdate) {
            return;
        }
        if (!(this.treeInput.get() instanceof IncrementalStoreTree tree) || !tree.areFlaggedNodesComplete()
                || tree.getNodeCount() != this.isNodePending.length) {
            this.requireFullUpdate();
            return;
        }

        int[] flaggedNodeNrs = tree.getFlaggedNodeNrs();
        int numFlaggedNodes = tree.getFlaggedNodeCount();
        for (int i = 0; i < numFlaggedNodes; i++) {
            int nodeNr = flaggedNodeNrs[i];
            if (!this.isNodePending[nodeNr]) {
                this.isNodePending[nodeNr] = true;
                this.pendingNodeNrs[this.numPendingNodes++] = nodeNr;
            }
        }

        // moving many nodes one by one is slower than sorting all of them
        if (this.numPendingNodes > this.pendingNodeNrs.length / 4) {
            this.requireFullUpdate();
        }
    }

    @Override
    public void setIntervalsUnknown() {
        this.requireFullUpdate();
        super.setIntervalsUnknown();
    }

    private void requireFullUpdate() {
        this.requiresFullUpdate = true;
        this.clearPendingNodes();
    }

    private void clearPendingNodes() {
        for (int i = 0; i < this.numPendingNodes; i++) {
            this.isNodePending[this.pendingNodeNrs[i]] = false;
        }
        this.numPendingNodes = 0;
    }

    @Override
    protected void calculateIntervals() {
        Tree tree = this.treeInput.get();
        final int nodeCount = tree.getNodeCount();

        if (this.indices == null || this.indices.length != nodeCount) {
            this.allocateArrays(nodeCount);
        }

        if (this.requiresFullUpdate || this.multifurcationLimit >= 0.0) {
            this.updateAllNodes(tree);
        } else {
            this.updatePendingNodes(tree);
        }

        this.intervalsKnown = true;
    }

    private void allocateArrays(int nodeCount) {
        this.indices = new int[nodeCount];
        this.positions = new int[nodeCount];
        this.sortedTimes = new double[nodeCount];
        for (int i = 0; i < nodeCount; i++) {
            this.indices[i] = i;
        }
        this.storedIndices = new int[nodeCount];
        this.storedSortedTimes = new double[nodeCount];

        this.intervals = new double[nodeCount];
        this.storedIntervals = new double[nodeCount];
        this.lineageCounts = new int[nodeCount];
        this.storedLineageCounts = new int[nodeCount];

        this.pendingNodeNrs = new int[nodeCount];
        this.isNodePending = new boolean[nodeCount];
        this.numPendingNodes = 0;

        this.requiresFullUpdate = true;
        this.isStoreOutdated = true;
    }

    // full update

    /**
     * Sorts all nodes by height and recomputes all intervals.
     */
    private void updateAllNodes(Tree tree) {
        Node[] nodes = tree.getNodesAsArray();
        final int nodeCount = nodes.length;

        for (int i = 0; i < nodeCount; i++) {
            this.sortedTimes[i] = nodes[this.indices[i]].getHeight();
        }
        this.sortIndices();
        for (int i = 0; i < nodeCount; i++) {
            this.positions[this.indices[i]] = i;
        }

        if (this.multifurcationLimit < 0.0) {
            this.computeIntervals(0, nodeCount - 1, tree.getLeafNodeCount());
        } else {
            this.computeMergedIntervals(tree.getLeafNodeCount());
        }

        this.markChanged(0, nodeCount - 1);
        this.clearPendingNodes();
        this.requiresFullUpdate = false;
    }

    /**
     * Sorts the indices together with their times by ascending time with insertion sort,
     * starting from their current order. This takes time linear in the number of indices plus
     * the number of pairs that are out of order, so it is fast if the indices are already nearly
     * sorted.
     */
    protected void sortIndices() {
        final int[] indices = this.indices;
        final double[] sortedTimes = this.sortedTimes;
        for (int i = 1; i < indices.length; i++) {
            final int index = indices[i];
            final double time = sortedTimes[i];

            // the indices before i are sorted already
            int j = i - 1;
            while (j >= 0 && sortedTimes[j] > time) {
                indices[j + 1] = indices[j];
                sortedTimes[j + 1] = sortedTimes[j];
                j--;
            }
            indices[j + 1] = index;
            sortedTimes[j + 1] = time;
        }
    }

    /**
     * Computes the intervals when the events within the multifurcation limit are merged. Leaves
     * are numbered below the leaf count, and every internal node merges two lineages into one.
     */
    private void computeMergedIntervals(int leafCount) {
        final int nodeCount = this.indices.length;

        double start = this.sortedTimes[0];
        int numLines = 0;
        int nodeNo = 0;
        this.intervalCount = 0;
        while (nodeNo < nodeCount) {
            int lineagesRemoved = 0;
            int lineagesAdded = 0;

            final double finish = this.sortedTimes[nodeNo];
            double next;

            do {
                final boolean isLeaf = this.indices[nodeNo] < leafCount;
                nodeNo += 1;
                if (isLeaf) {
                    lineagesAdded += 1;
                } else {
                    lineagesRemoved += 1;
                    // with a zero multifurcation limit, each coalescence is a separate event
                    if (this.multifurcationLimit == 0.0) {
                        break;
                    }
                }

                if (nodeNo < nodeCount) {
                    next = this.sortedTimes[nodeNo];
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
    }

    // incremental update

    /**
     * Moves the pending nodes whose heights changed to their new positions in the sorted order,
     * and recomputes the intervals between their old and new positions.
     */
    private void updatePendingNodes(Tree tree) {
        Node[] nodes = tree.getNodesAsArray();

        int firstMovedPosition = nodes.length;
        int lastMovedPosition = -1;
        for (int i = 0; i < this.numPendingNodes; i++) {
            final int nodeNr = this.pendingNodeNrs[i];
            this.isNodePending[nodeNr] = false;

            final double height = nodes[nodeNr].getHeight();
            final int position = this.positions[nodeNr];
            if (height == this.sortedTimes[position]) {
                continue;
            }

            final int newPosition = this.moveNode(position, height);
            firstMovedPosition = Math.min(firstMovedPosition, Math.min(position, newPosition));
            lastMovedPosition = Math.max(lastMovedPosition, Math.max(position, newPosition));
        }
        this.numPendingNodes = 0;

        if (firstMovedPosition <= lastMovedPosition) {
            // the interval after the last moved position starts at a moved node
            final int lastPosition = Math.min(lastMovedPosition + 1, nodes.length - 1);
            this.computeIntervals(firstMovedPosition, lastPosition, tree.getLeafNodeCount());
            this.markChanged(firstMovedPosition, lastPosition);
        }
    }

    /**
     * Gives the node at the given position the given time and shifts it to keep the times
     * sorted, assuming that all other times are sorted. Returns the new position of the node.
     */
    private int moveNode(int position, double time) {
        final int[] indices = this.indices;
        final double[] sortedTimes = this.sortedTimes;
        final int[] positions = this.positions;
        final int nodeNr = indices[position];

        int j = position;
        while (j > 0 && sortedTimes[j - 1] > time) {
            indices[j] = indices[j - 1];
            sortedTimes[j] = sortedTimes[j - 1];
            positions[indices[j]] = j;
            j--;
        }
        while (j < indices.length - 1 && sortedTimes[j + 1] <= time) {
            indices[j] = indices[j + 1];
            sortedTimes[j] = sortedTimes[j + 1];
            positions[indices[j]] = j;
            j++;
        }
        indices[j] = nodeNr;
        sortedTimes[j] = time;
        positions[nodeNr] = j;
        return j;
    }

    /**
     * Computes the intervals ending at the given range of positions when every node is a
     * separate event. The first interval has length zero, and the lineage count before every
     * position follows from the leaves and internal nodes before it.
     */
    private void computeIntervals(int firstPosition, int lastPosition, int leafCount) {
        final int[] indices = this.indices;
        final double[] sortedTimes = this.sortedTimes;
        final double[] intervals = this.intervals;
        final int[] lineageCounts = this.lineageCounts;

        int start = firstPosition;
        if (start == 0) {
            intervals[0] = 0.0;
            lineageCounts[0] = 0;
            start = 1;
        }
        int numLines = lineageCounts[start - 1];
        for (int i = start; i <= lastPosition; i++) {
            numLines += indices[i - 1] < leafCount ? 1 : -1;
            intervals[i] = sortedTimes[i] - sortedTimes[i - 1];
            lineageCounts[i] = numLines;
        }
        this.intervalCount = indices.length;
    }

    private void markChanged(int firstPosition, int lastPosition) {
        if (this.firstChangedPosition > this.lastChangedPosition) {
            this.firstChangedPosition = firstPosition;
            this.lastChangedPosition = lastPosition;
        } else {
            this.firstChangedPosition = Math.min(this.firstChangedPosition, firstPosition);
            this.lastChangedPosition = Math.max(this.lastChangedPosition, lastPosition);
        }
    }

    // store and restore

    /**
     * Copies the changed positions into the stored arrays, which makes them equal to the current
     * ones without copying the unchanged positions.
     */
    @Override
    protected void store() {
        if (this.indices != null) {
            this.copyChangedPositions(this.indices, this.sortedTimes, this.intervals, this.lineageCounts,
                    this.storedIndices, this.storedSortedTimes, this.storedIntervals, this.storedLineageCounts);
        }
        this.isStoreOutdated = this.requiresFullUpdate || this.numPendingNodes > 0;

        // the stored intervals and lineage counts are up to date already, so the base class copies nothing
        int[] lineageCounts = this.lineageCounts;
        double[] intervals = this.intervals;
        this.lineageCounts = NO_COUNTS;
        this.intervals = NO_INTERVALS;
        super.store();
        this.lineageCounts = lineageCounts;
        this.intervals = intervals;
    }

    /**
     * Copies the stored arrays back into the changed positions. If the sorted order did not
     * match the tree at the last store, the stored arrays do not match the restored tree, and
     * the next calculation sorts all nodes.
     */
    @Override
    protected void restore() {
        if (this.indices != null) {
            int firstPosition = this.firstChangedPosition;
            int lastPosition = this.lastChangedPosition;
            this.copyChangedPositions(this.storedIndices, this.storedSortedTimes, this.storedIntervals,
                    this.storedLineageCounts, this.indices, this.sortedTimes, this.intervals, this.lineageCounts);
            for (int i = firstPosition; i <= lastPosition; i++) {
                this.positions[this.indices[i]] = i;
            }
        }
        if (this.isStoreOutdated) {
            this.requireFullUpdate();
            this.intervalsKnown = false;
        }

        // the current and stored intervals and lineage counts are equal, so swapping them changes nothing
        super.restore();
    }

    /**
     * Copies the changed positions of the given arrays into the target arrays and empties the
     * range of changed positions.
     */
    private void copyChangedPositions(int[] indices, double[] sortedTimes, double[] intervals, int[] lineageCounts,
                                      int[] targetIndices, double[] targetSortedTimes, double[] targetIntervals,
                                      int[] targetLineageCounts) {
        if (this.firstChangedPosition <= this.lastChangedPosition) {
            final int first = this.firstChangedPosition;
            final int length = this.lastChangedPosition - first + 1;
            System.arraycopy(indices, first, targetIndices, first, length);
            System.arraycopy(sortedTimes, first, targetSortedTimes, first, length);
            System.arraycopy(intervals, first, targetIntervals, first, length);
            System.arraycopy(lineageCounts, first, targetLineageCounts, first, length);
        }
        this.firstChangedPosition = 0;
        this.lastChangedPosition = -1;
    }

}
