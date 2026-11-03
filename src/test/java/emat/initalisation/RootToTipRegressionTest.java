package emat.initalisation;

import org.junit.jupiter.api.Test;

import java.util.Random;

import static org.junit.jupiter.api.Assertions.assertEquals;

public class RootToTipRegressionTest {

    /**
     * Checks that the slope of a tree with a strict clock of rate 0.01 is recovered. The root of
     * ((0,1),(2,3)) is at date 1990, the parent of 0 and 1 at 1994, the parent of 2 and 3 at
     * 1996, and the tips at 2000, 2003, 2001 and 2004.
     */
    @Test
    public void testStrictClockRate() {
        BinaryTopology topology = new BinaryTopology(4, new int[]{4, 4, 5, 5, 6, 6, -1});
        double[] branchLengths = {0.06, 0.09, 0.05, 0.08, 0.04, 0.06, 0.0};
        double[] dates = {2000.0, 2003.0, 2001.0, 2004.0};

        double slope = new RootToTipRegression(topology, branchLengths, dates).computeBestRootSlope();

        assertEquals(0.01, slope, 1e-12);
    }

    /** Checks that contemporaneous tips give no slope. */
    @Test
    public void testContemporaneousTipsGiveNoSlope() {
        BinaryTopology topology = new BinaryTopology(3, new int[]{3, 3, 4, 4, -1});
        double[] branchLengths = {0.05, 0.07, 0.14, 0.05, 0.0};
        double[] dates = {2000.0, 2000.0, 2000.0};

        assertEquals(0.0, new RootToTipRegression(topology, branchLengths, dates).computeBestRootSlope());
    }

    /**
     * Checks the best root against a search over a fine grid of root positions on every branch of
     * random trees with a noisy clock, where the root-to-tip distances are computed directly.
     */
    @Test
    public void testBestRootMatchesGridSearch() {
        Random random = new Random(3);
        for (int replicate = 0; replicate < 20; replicate++) {
            int numTips = 8;
            BinaryTopology topology = this.createRandomTopology(numTips, random);
            double[] nodeDates = new double[topology.getNumNodes()];
            double[] branchLengths = new double[topology.getNumNodes()];
            for (int node : topology.getPreorder()) {
                int parent = topology.getParent(node);
                if (parent >= 0) {
                    double duration = 0.5 + 1.5 * random.nextDouble();
                    nodeDates[node] = nodeDates[parent] + duration;
                    branchLengths[node] = 0.01 * duration * (0.5 + random.nextDouble());
                }
            }
            double[] dates = new double[numTips];
            System.arraycopy(nodeDates, 0, dates, 0, numTips);

            double slope = new RootToTipRegression(topology, branchLengths, dates).computeBestRootSlope();
            double expectedSlope = this.searchGrid(topology, branchLengths, dates, 2000);

            assertEquals(expectedSlope, slope, 1e-3 * expectedSlope);
        }
    }

    /** Returns a random topology built by joining random pairs of the remaining subtrees. */
    private BinaryTopology createRandomTopology(int numTips, Random random) {
        int[] parents = new int[2 * numTips - 1];
        int[] subtrees = new int[numTips];
        for (int tip = 0; tip < numTips; tip++) {
            subtrees[tip] = tip;
        }

        int numSubtrees = numTips;
        for (int join = numTips; join < parents.length; join++) {
            int first = random.nextInt(numSubtrees);
            int firstNode = subtrees[first];
            subtrees[first] = subtrees[--numSubtrees];
            int second = random.nextInt(numSubtrees);
            parents[firstNode] = join;
            parents[subtrees[second]] = join;
            subtrees[second] = join;
        }
        parents[parents.length - 1] = -1;
        return new BinaryTopology(numTips, parents);
    }

    /**
     * Returns the slope of the root position with the smallest residual sum of squares among the
     * positions with a positive slope, or zero if there is none. On a tree with a clock the best
     * position has a clearly positive slope, so the result does not depend on how positions close
     * to a zero slope are treated.
     */
    private double searchGrid(BinaryTopology topology, double[] branchLengths, double[] dates, int numPoints) {
        double bestSlope = 0.0;
        double bestResiduals = Double.POSITIVE_INFINITY;
        for (int node = 0; node < topology.getNumNodes(); node++) {
            if (node == topology.getRoot()) {
                continue;
            }
            for (int point = 0; point <= numPoints; point++) {
                double position = branchLengths[node] * point / numPoints;
                double[] distances = this.computeRootToTipDistances(topology, branchLengths, node, position);
                double[] fit = this.fitRegression(dates, distances);
                if (fit[0] > 0.0 && fit[1] < bestResiduals) {
                    bestSlope = fit[0];
                    bestResiduals = fit[1];
                }
            }
        }
        return bestSlope;
    }

    /** Returns the distance of every tip from a root at the given position above the node. */
    private double[] computeRootToTipDistances(BinaryTopology topology, double[] branchLengths, int node, double position) {
        double[] distances = new double[topology.getNumTips()];
        for (int tip = 0; tip < topology.getNumTips(); tip++) {
            double distanceToNode = this.computeDistanceToAncestor(topology, branchLengths, tip, node);
            if (!Double.isNaN(distanceToNode)) {
                distances[tip] = distanceToNode + position;
            } else {
                int parent = topology.getParent(node);
                double distanceToParent = this.computeDistanceBetween(topology, branchLengths, tip, parent);
                distances[tip] = distanceToParent + branchLengths[node] - position;
            }
        }
        return distances;
    }

    /** Returns the distance from the tip up to the ancestor, or NaN if the node is not an ancestor. */
    private double computeDistanceToAncestor(BinaryTopology topology, double[] branchLengths, int tip, int ancestor) {
        double distance = 0.0;
        int node = tip;
        while (node != ancestor) {
            if (node == topology.getRoot()) {
                return Double.NaN;
            }
            distance += branchLengths[node];
            node = topology.getParent(node);
        }
        return distance;
    }

    /** Returns the path length between the tip and another node through their common ancestor. */
    private double computeDistanceBetween(BinaryTopology topology, double[] branchLengths, int tip, int other) {
        int ancestor = other;
        double otherToAncestor = 0.0;
        while (Double.isNaN(this.computeDistanceToAncestor(topology, branchLengths, tip, ancestor))) {
            otherToAncestor += branchLengths[ancestor];
            ancestor = topology.getParent(ancestor);
        }
        return otherToAncestor + this.computeDistanceToAncestor(topology, branchLengths, tip, ancestor);
    }

    /** Returns the slope and the residual sum of squares of the regression of y on x. */
    private double[] fitRegression(double[] x, double[] y) {
        double meanX = 0.0;
        double meanY = 0.0;
        for (int i = 0; i < x.length; i++) {
            meanX += x[i] / x.length;
            meanY += y[i] / x.length;
        }

        double sxy = 0.0;
        double sxx = 0.0;
        double syy = 0.0;
        for (int i = 0; i < x.length; i++) {
            sxy += (x[i] - meanX) * (y[i] - meanY);
            sxx += (x[i] - meanX) * (x[i] - meanX);
            syy += (y[i] - meanY) * (y[i] - meanY);
        }
        double slope = sxy / sxx;
        return new double[]{slope, syy - slope * sxy};
    }

}
