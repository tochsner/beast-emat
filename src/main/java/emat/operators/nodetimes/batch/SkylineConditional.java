package emat.operators.nodetimes.batch;

import beast.base.evolution.tree.Tree;
import beast.base.spec.type.IntVector;
import beast.base.spec.type.RealVector;
import beast.base.util.Randomizer;

import java.util.Arrays;

/**
 * The exact log Bayesian skyline density as a function of the height of a single internal node, with all other
 * heights fixed. Group g ends at the coalescent event of rank c_g, the cumulative group size, so its boundary is
 * the height of the node clamped to the window between the events of ranks c_g - 1 and c_g among the other
 * nodes. Outside the windows, a node at height h with k lineages just below it has the slope -(k - 1) / N(h).
 * Within the window of group g, the node is the boundary itself and the slope is
 * -(C(k, 2) / N_g - C(k - 1, 2) / N_{g + 1}), which is quadratic in the number of lineages. The sum of the terms
 * -log N over the coalescent events does not depend on the heights, as every group holds a fixed number of them.
 * So the log density is piecewise linear in the height, with breakpoints at the tips, the other internal nodes and
 * the ends of the windows.
 */
class SkylineConditional {

    private final Tree tree;
    private final RealVector<?> popSizes;
    private final IntVector<?> groupSizes;

    private final double[] tipHeights;

    // the sorted heights of all internal nodes, which follow every move of a node
    private final double[] internalHeights;

    // the population and the cumulative group sizes at the start of the batch
    private final double[] sizes;
    private final int[] cumulativeGroupSizes;

    // the pieces of the log density of the last node: their ends, slopes and the log density at their start
    private double[] pieceEnds = new double[16];
    private double[] pieceSlopes = new double[16];
    private double[] pieceStarts = new double[16];
    private int numPieces;
    private double pieceLower;

    // scratch space for the breakpoints and the masses of the pieces
    private double[] breakpoints = new double[16];
    private double[] logMasses = new double[16];

    SkylineConditional(Tree tree, RealVector<?> popSizes, IntVector<?> groupSizes) {
        this.tree = tree;
        this.popSizes = popSizes;
        this.groupSizes = groupSizes;

        int numLeaves = tree.getLeafNodeCount();
        this.tipHeights = new double[numLeaves];
        for (int i = 0; i < numLeaves; i++) {
            this.tipHeights[i] = tree.getNode(i).getHeight();
        }
        Arrays.sort(this.tipHeights);

        this.internalHeights = new double[tree.getInternalNodeCount()];
        this.sizes = new double[popSizes.size()];
        this.cumulativeGroupSizes = new int[groupSizes.size()];
    }

    /** Reads the heights of the internal nodes and the parameters of the skyline before a batch. */
    void startBatch() {
        int numLeaves = this.tree.getLeafNodeCount();
        for (int i = 0; i < this.internalHeights.length; i++) {
            this.internalHeights[i] = this.tree.getNode(numLeaves + i).getHeight();
        }
        Arrays.sort(this.internalHeights);

        for (int i = 0; i < this.sizes.length; i++) {
            this.sizes[i] = this.popSizes.get(i);
        }
        int sum = 0;
        for (int i = 0; i < this.cumulativeGroupSizes.length; i++) {
            sum += this.groupSizes.get(i);
            this.cumulativeGroupSizes[i] = sum;
        }
    }

    /**
     * Draws a new height for the node at the given height from the density ∝ exp(-decayRate h) times the skyline
     * density between the bounds, and returns it. The change of the log skyline density from the old to the new
     * height is available from {@link #getChange} until the next draw.
     */
    double sampleHeight(double oldHeight, double decayRate, double lower, double upper) {
        this.computePieces(oldHeight, lower, upper);

        // the log masses of the pieces under the tilted density
        double largest = Double.NEGATIVE_INFINITY;
        double start = 0.0;
        double pieceStart = lower;
        for (int j = 0; j < this.numPieces; j++) {
            double width = this.pieceEnds[j] - pieceStart;
            double rate = this.pieceSlopes[j] - decayRate;
            this.logMasses[j] = start + Math.log(width) + computeLogMeanExp(rate * width);
            largest = Math.max(largest, this.logMasses[j]);
            start += rate * width;
            pieceStart = this.pieceEnds[j];
        }

        double total = 0.0;
        for (int j = 0; j < this.numPieces; j++) {
            total += Math.exp(this.logMasses[j] - largest);
        }
        double u = Randomizer.nextDouble() * total;
        int index = 0;
        while (index < this.numPieces - 1) {
            u -= Math.exp(this.logMasses[index] - largest);
            if (u <= 0.0) {
                break;
            }
            index++;
        }

        double pieceLowerEnd = index == 0 ? lower : this.pieceEnds[index - 1];
        return TruncatedDensitySampler.sampleExponential(decayRate - this.pieceSlopes[index], pieceLowerEnd, this.pieceEnds[index]);
    }

    /** Returns log ((e^x - 1) / x), the log mean of e^(x u) for u uniform on [0, 1]. */
    private static double computeLogMeanExp(double x) {
        if (Math.abs(x) < 1e-12) {
            return 0.5 * x;
        }
        if (x > 0.0) {
            return x + Math.log(-Math.expm1(-x) / x);
        }
        return Math.log(Math.expm1(x) / x);
    }

    /**
     * Splits the range of the node into pieces of constant slope of the log skyline density. Reads the heights of
     * the other nodes from the sorted heights, which still hold the node at its old height.
     */
    private void computePieces(double oldHeight, double lower, double upper) {
        int ownIndex = Arrays.binarySearch(this.internalHeights, oldHeight);
        int numGroups = this.cumulativeGroupSizes.length;

        // the breakpoints inside the range
        int numBreakpoints = 0;
        numBreakpoints = this.addBreakpoints(this.tipHeights, -1, lower, upper, numBreakpoints);
        numBreakpoints = this.addBreakpoints(this.internalHeights, ownIndex, lower, upper, numBreakpoints);
        for (int g = 0; g < numGroups - 1; g++) {
            int rank = this.cumulativeGroupSizes[g] - 1;
            numBreakpoints = this.addBreakpoint(this.getOtherHeight(rank - 1, ownIndex), lower, upper, numBreakpoints);
            numBreakpoints = this.addBreakpoint(this.getOtherHeight(rank, ownIndex), lower, upper, numBreakpoints);
        }
        numBreakpoints = this.appendBreakpoint(upper, numBreakpoints);
        Arrays.sort(this.breakpoints, 0, numBreakpoints);

        // one piece per stretch between neighbouring breakpoints
        this.ensurePieceCapacity(numBreakpoints);
        this.numPieces = 0;
        this.pieceLower = lower;
        double start = lower;
        double logDensity = 0.0;
        for (int i = 0; i < numBreakpoints; i++) {
            double end = this.breakpoints[i];
            if (end <= start) {
                continue;
            }
            double slope = this.computeSlope(0.5 * (start + end), oldHeight, ownIndex);
            this.pieceEnds[this.numPieces] = end;
            this.pieceSlopes[this.numPieces] = slope;
            this.pieceStarts[this.numPieces] = logDensity;
            this.numPieces++;
            logDensity += slope * (end - start);
            start = end;
        }
    }

    private int addBreakpoints(double[] sorted, int skipIndex, double lower, double upper, int numBreakpoints) {
        int from = countBelow(sorted, lower);
        for (int i = from; i < sorted.length && sorted[i] < upper; i++) {
            if (i != skipIndex) {
                numBreakpoints = this.addBreakpoint(sorted[i], lower, upper, numBreakpoints);
            }
        }
        return numBreakpoints;
    }

    private int addBreakpoint(double height, double lower, double upper, int numBreakpoints) {
        if (height <= lower || height >= upper) {
            return numBreakpoints;
        }
        return this.appendBreakpoint(height, numBreakpoints);
    }

    private int appendBreakpoint(double height, int numBreakpoints) {
        if (numBreakpoints == this.breakpoints.length) {
            this.breakpoints = Arrays.copyOf(this.breakpoints, 2 * numBreakpoints);
        }
        this.breakpoints[numBreakpoints] = height;
        return numBreakpoints + 1;
    }

    /** Returns the height of the given rank among the internal nodes other than the node, or ±∞ beyond them. */
    private double getOtherHeight(int rank, int ownIndex) {
        if (rank < 0) {
            return Double.NEGATIVE_INFINITY;
        }
        int index = rank < ownIndex ? rank : rank + 1;
        return index < this.internalHeights.length ? this.internalHeights[index] : Double.POSITIVE_INFINITY;
    }

    /** Returns the slope of the log skyline density for the node at the given height, between two breakpoints. */
    private double computeSlope(double height, double oldHeight, int ownIndex) {
        int otherBelow = countBelow(this.internalHeights, height) - (oldHeight < height ? 1 : 0);
        int lineages = countBelow(this.tipHeights, height) - otherBelow;

        int group = 0;
        int numGroups = this.cumulativeGroupSizes.length;
        for (int g = 0; g < numGroups - 1; g++) {
            int rank = this.cumulativeGroupSizes[g] - 1;
            if (height > this.getOtherHeight(rank, ownIndex)) {
                group++;
            } else if (height > this.getOtherHeight(rank - 1, ownIndex)) {
                // the node is the boundary of group g
                double sizeBelow = this.sizes[g];
                double sizeAbove = this.sizes[g + 1];
                return -(lineages * (lineages - 1) / 2.0 / sizeBelow - (lineages - 1) * (lineages - 2) / 2.0 / sizeAbove);
            }
        }
        return -(lineages - 1) / this.sizes[Math.min(group, this.sizes.length - 1)];
    }

    private void ensurePieceCapacity(int capacity) {
        if (capacity > this.pieceEnds.length) {
            int length = Math.max(capacity, 2 * this.pieceEnds.length);
            this.pieceEnds = Arrays.copyOf(this.pieceEnds, length);
            this.pieceSlopes = Arrays.copyOf(this.pieceSlopes, length);
            this.pieceStarts = Arrays.copyOf(this.pieceStarts, length);
            this.logMasses = Arrays.copyOf(this.logMasses, length);
        }
    }

    /** Returns the change of the log skyline density from the old to the new height over the pieces of the last draw. */
    double getChange(double oldHeight, double newHeight) {
        return this.computeLogDensity(newHeight) - this.computeLogDensity(oldHeight);
    }

    private double computeLogDensity(double height) {
        double start = this.pieceLower;
        for (int j = 0; j < this.numPieces; j++) {
            if (height <= this.pieceEnds[j] || j == this.numPieces - 1) {
                return this.pieceStarts[j] + this.pieceSlopes[j] * (height - start);
            }
            start = this.pieceEnds[j];
        }
        return 0.0;
    }

    /** Moves the node in the sorted heights from the old to the new height. */
    void moveNode(double oldHeight, double newHeight) {
        double[] heights = this.internalHeights;
        int index = Arrays.binarySearch(heights, oldHeight);
        System.arraycopy(heights, index + 1, heights, index, heights.length - index - 1);
        int insert = Arrays.binarySearch(heights, 0, heights.length - 1, newHeight);
        if (insert < 0) {
            insert = -insert - 1;
        }
        System.arraycopy(heights, insert, heights, insert + 1, heights.length - insert - 1);
        heights[insert] = newHeight;
    }

    /** Returns the number of the sorted values below the given one. */
    private static int countBelow(double[] sorted, double value) {
        int index = Arrays.binarySearch(sorted, value);
        if (index < 0) {
            return -index - 1;
        }
        while (index > 0 && sorted[index - 1] == value) {
            index--;
        }
        return index;
    }

}
