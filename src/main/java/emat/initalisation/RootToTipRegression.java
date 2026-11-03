package emat.initalisation;

/**
 * The least-squares regression of the root-to-tip distances of a tree on the sampling dates of
 * its tips, with the root placed anywhere on the tree. The best root is the one with the
 * smallest residual sum of squares among the roots with a positive slope, as in TempEst. Sums
 * over the tips inside and outside of every subtree make each root position cost constant time.
 */
public class RootToTipRegression {

    private final BinaryTopology topology;
    private final double[] branchLengths;
    private final double[] dates;

    // the variance of the dates, times the number of tips
    private final double dateSumOfSquares;

    /**
     * Sets up the regression on the given topology with the given length of the branch above
     * every node and the given date of every tip.
     */
    public RootToTipRegression(BinaryTopology topology, double[] branchLengths, double[] dates) {
        this.topology = topology;
        this.branchLengths = branchLengths;
        this.dates = dates;
        this.dateSumOfSquares = this.computeDateSumOfSquares();
    }

    private double computeDateSumOfSquares() {
        double meanDate = 0.0;
        for (double date : this.dates) {
            meanDate += date;
        }
        meanDate /= this.dates.length;

        double sumOfSquares = 0.0;
        for (double date : this.dates) {
            sumOfSquares += (date - meanDate) * (date - meanDate);
        }
        return sumOfSquares;
    }

    /**
     * Returns the slope of the best root, or zero if all tips have the same date or no root
     * gives a positive slope. A root on the branch above a node is tried at both ends of the
     * branch and where the residual sum of squares, which is quadratic in the position, is
     * smallest.
     */
    public double computeBestRootSlope() {
        if (this.dateSumOfSquares == 0.0) {
            return 0.0;
        }

        TipSums[] insideSums = this.computeInsideSums();
        TipSums[] outsideSums = this.computeOutsideSums(insideSums);

        Fit bestFit = null;
        for (int node = 0; node < this.topology.getNumNodes(); node++) {
            if (node != this.topology.getRoot()) {
                Fit fit = this.fitBestRootOnBranch(insideSums[node], outsideSums[node], this.branchLengths[node]);
                bestFit = this.chooseBetterFit(bestFit, fit);
            }
        }
        return bestFit == null ? 0.0 : bestFit.slope();
    }

    /** Returns the sums over the tips below every node, with the distances measured from the node. */
    private TipSums[] computeInsideSums() {
        TipSums[] insideSums = new TipSums[this.topology.getNumNodes()];
        for (int node : this.topology.getPostorder()) {
            if (node < this.topology.getNumTips()) {
                insideSums[node] = TipSums.ofTip(this.dates[node]);
                continue;
            }

            TipSums sums = TipSums.EMPTY;
            for (int child : this.topology.getChildren(node)) {
                sums = sums.add(insideSums[child].moveAway(this.branchLengths[child]));
            }
            insideSums[node] = sums;
        }
        return insideSums;
    }

    /** Returns the sums over the tips not below every node, with the distances measured from the node. */
    private TipSums[] computeOutsideSums(TipSums[] insideSums) {
        TipSums[] outsideSums = new TipSums[this.topology.getNumNodes()];
        outsideSums[this.topology.getRoot()] = TipSums.EMPTY;

        for (int node : this.topology.getPreorder()) {
            int[] children = this.topology.getChildren(node);
            for (int i = 0; i < children.length; i++) {
                int child = children[i];
                int sibling = children[1 - i];
                TipSums sums = outsideSums[node].add(insideSums[sibling].moveAway(this.branchLengths[sibling]));
                outsideSums[child] = sums.moveAway(this.branchLengths[child]);
            }
        }
        return outsideSums;
    }

    /**
     * Returns the best fit with a positive slope among the roots on the branch above a node, or
     * null if there is none. A root at distance x above the node is x further from the tips
     * below the node and x closer to the others.
     */
    private Fit fitBestRootOnBranch(TipSums inside, TipSums outside, double branchLength) {
        Fit bottom = this.fitRootAt(inside, outside, 0.0);
        if (branchLength <= 0.0) {
            return this.chooseBetterFit(null, bottom);
        }

        Fit middle = this.fitRootAt(inside, outside, 0.5 * branchLength);
        Fit top = this.fitRootAt(inside, outside, branchLength);
        Fit best = this.chooseBetterFit(this.chooseBetterFit(null, bottom), top);

        double vertex = this.findVertex(branchLength, bottom, middle, top);
        return Double.isNaN(vertex) ? best : this.chooseBetterFit(best, this.fitRootAt(inside, outside, vertex));
    }

    /** Returns the fit for a root at the given distance above the node. */
    private Fit fitRootAt(TipSums inside, TipSums outside, double position) {
        return this.fitRegression(inside.moveAway(position).add(outside.moveAway(-position)));
    }

    /** Returns the least-squares fit of the distances on the dates of all tips. */
    private Fit fitRegression(TipSums sums) {
        double distanceSumOfSquares = sums.sumOfSquaredDistances() - sums.sumOfDistances() * sums.sumOfDistances() / sums.numTips();
        double crossSum = sums.sumOfDateDistanceProducts() - sums.sumOfDates() * sums.sumOfDistances() / sums.numTips();

        double slope = crossSum / this.dateSumOfSquares;
        double residualSumOfSquares = distanceSumOfSquares - crossSum * slope;
        return new Fit(slope, residualSumOfSquares);
    }

    /**
     * Returns the candidate if it has a positive slope and a smaller residual sum of squares than
     * the best fit so far, which may be null, and the best fit otherwise.
     */
    private Fit chooseBetterFit(Fit best, Fit candidate) {
        boolean isBetter = candidate != null && candidate.slope() > 0.0
                && (best == null || candidate.residualSumOfSquares() < best.residualSumOfSquares());
        return isBetter ? candidate : best;
    }

    /**
     * Returns the position of the smallest residual sum of squares on the branch, found from the
     * parabola through the fits at the bottom, middle and top of the branch, or NaN if the
     * parabola has no minimum inside the branch.
     */
    private double findVertex(double branchLength, Fit bottom, Fit middle, Fit top) {
        double half = 0.5 * branchLength;
        double curvature = (bottom.residualSumOfSquares() - 2.0 * middle.residualSumOfSquares() + top.residualSumOfSquares()) / (2.0 * half * half);
        if (curvature <= 0.0) {
            return Double.NaN;
        }

        double gradient = (middle.residualSumOfSquares() - bottom.residualSumOfSquares()) / half - curvature * half;
        double position = -gradient / (2.0 * curvature);
        return position > 0.0 && position < branchLength ? position : Double.NaN;
    }

    /**
     * Sums over a set of tips of their dates and of their distances from a point of the tree.
     */
    record TipSums(double numTips, double sumOfDates, double sumOfDistances, double sumOfSquaredDistances,
                   double sumOfDateDistanceProducts) {

        static final TipSums EMPTY = new TipSums(0.0, 0.0, 0.0, 0.0, 0.0);

        static TipSums ofTip(double date) {
            return new TipSums(1.0, date, 0.0, 0.0, 0.0);
        }

        TipSums add(TipSums other) {
            return new TipSums(
                    this.numTips + other.numTips,
                    this.sumOfDates + other.sumOfDates,
                    this.sumOfDistances + other.sumOfDistances,
                    this.sumOfSquaredDistances + other.sumOfSquaredDistances,
                    this.sumOfDateDistanceProducts + other.sumOfDateDistanceProducts
            );
        }

        /** Returns the sums for a point that is the given length further from every tip. */
        TipSums moveAway(double length) {
            return new TipSums(
                    this.numTips,
                    this.sumOfDates,
                    this.sumOfDistances + this.numTips * length,
                    this.sumOfSquaredDistances + 2.0 * length * this.sumOfDistances + this.numTips * length * length,
                    this.sumOfDateDistanceProducts + length * this.sumOfDates
            );
        }
    }

    /** The slope and residual sum of squares of a regression. */
    record Fit(double slope, double residualSumOfSquares) {
    }

}
