package emat.operators.nodetimes.batch;

import beast.base.evolution.tree.Node;
import beast.base.evolution.tree.TreeIntervals;
import beast.base.evolution.tree.TreeParser;
import beast.base.spec.domain.PositiveInt;
import beast.base.spec.domain.PositiveReal;
import beast.base.spec.evolution.tree.coalescent.BayesianSkyline;
import beast.base.spec.inference.parameter.IntSimplexParam;
import beast.base.spec.inference.parameter.RealVectorParam;
import beast.base.util.Randomizer;
import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertTrue;

public class SkylineConditionalTest {

    // tips at different heights, with internal nodes close to each other so that the group boundaries matter
    private static final String NEWICK = "(((((A:0.3,B:0.1):0.2,C:0.45):0.1,(D:0.2,E:0.35):0.4):0.3,F:0.9):0.5,((G:0.15,H:0.4):0.6,I:1.0):0.3);";

    private final RealVectorParam<PositiveReal> popSizes = new RealVectorParam<>(new double[]{0.3, 2.5, 0.8}, PositiveReal.INSTANCE);
    private final IntSimplexParam<PositiveInt> groupSizes = new IntSimplexParam<>(new int[]{3, 2, 3}, PositiveInt.INSTANCE, 8);

    /** Returns the log skyline density of the tree, from fresh intervals. */
    private double computeLogSkyline(TreeParser tree) {
        TreeIntervals intervals = new TreeIntervals();
        intervals.initByName("tree", tree);
        BayesianSkyline skyline = new BayesianSkyline();
        skyline.initByName("treeIntervals", intervals, "popSizes", this.popSizes, "groupSizes", this.groupSizes);
        return skyline.calculateLogP();
    }

    private TreeParser createTree() {
        TreeParser tree = new TreeParser();
        tree.initByName("newick", NEWICK, "IsLabelledNewick", true, "adjustTipHeights", false);
        return tree;
    }

    /**
     * The change that the conditional reports matches the change of the BayesianSkyline when nodes move one
     * after the other between their children and their parent, also across the group boundaries.
     */
    @Test
    public void testChangeMatchesSkyline() {
        Randomizer.setSeed(5);
        TreeParser tree = this.createTree();
        SkylineConditional conditional = new SkylineConditional(tree, this.popSizes, this.groupSizes);

        double maxError = 0.0;
        for (int i = 0; i < 2000; i++) {
            conditional.startBatch();
            for (int j = 0; j < 4; j++) {
                Node node = tree.getNode(tree.getLeafNodeCount() + Randomizer.nextInt(tree.getInternalNodeCount()));
                if (node.isRoot()) {
                    continue;
                }

                double lower = Math.max(node.getLeft().getHeight(), node.getRight().getHeight());
                double upper = node.getParent().getHeight();
                double oldHeight = node.getHeight();
                double before = this.computeLogSkyline(tree);

                double newHeight = conditional.sampleHeight(oldHeight, 2.0 * Randomizer.nextDouble() - 1.0, lower, upper);
                double change = conditional.getChange(oldHeight, newHeight);
                conditional.moveNode(oldHeight, newHeight);
                node.setHeight(newHeight);

                maxError = Math.max(maxError, Math.abs(this.computeLogSkyline(tree) - before - change));
            }
        }
        assertEquals(0.0, maxError, 1e-9);
    }

    /** The drawn heights follow exp(-k h) times the skyline density, measured with the BayesianSkyline itself. */
    @Test
    public void testSampledHeightsFollowTheConditional() {
        Randomizer.setSeed(7);
        TreeParser tree = this.createTree();
        SkylineConditional conditional = new SkylineConditional(tree, this.popSizes, this.groupSizes);

        int numDraws = 200000;
        int numBins = 20;
        int numCells = 4000;
        for (Node node : tree.getInternalNodes()) {
            if (node.isRoot()) {
                continue;
            }
            for (double decayRate : new double[]{0.0, 3.0, -2.0}) {
                double lower = Math.max(node.getLeft().getHeight(), node.getRight().getHeight());
                double upper = node.getParent().getHeight();
                double oldHeight = node.getHeight();

                // the target density on a fine grid
                double[] cumulative = new double[numCells + 1];
                double[] exponents = new double[numCells];
                double largest = Double.NEGATIVE_INFINITY;
                for (int i = 0; i < numCells; i++) {
                    node.setHeight(lower + (i + 0.5) / numCells * (upper - lower));
                    exponents[i] = -decayRate * node.getHeight() + this.computeLogSkyline(tree);
                    largest = Math.max(largest, exponents[i]);
                }
                node.setHeight(oldHeight);
                for (int i = 0; i < numCells; i++) {
                    cumulative[i + 1] = cumulative[i] + Math.exp(exponents[i] - largest);
                }

                conditional.startBatch();
                int[] counts = new int[numBins];
                for (int i = 0; i < numDraws; i++) {
                    double height = conditional.sampleHeight(oldHeight, decayRate, lower, upper);
                    assertTrue(height >= lower && height <= upper);
                    counts[Math.min((int) ((height - lower) / (upper - lower) * numBins), numBins - 1)]++;
                }

                for (int bin = 0; bin < numBins; bin++) {
                    double expected = (cumulative[(bin + 1) * numCells / numBins] - cumulative[bin * numCells / numBins]) / cumulative[numCells];
                    assertEquals(expected, counts[bin] / (double) numDraws, 0.005, "node " + node.getNr() + ", rate " + decayRate + ", bin " + bin);
                }
            }
        }
    }

}
