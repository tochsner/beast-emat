package emat;

import beast.base.evolution.alignment.Alignment;
import beast.base.evolution.alignment.Sequence;
import beast.base.evolution.tree.Node;
import beast.base.evolution.tree.TreeParser;
import beast.base.spec.domain.PositiveReal;
import beast.base.spec.domain.Real;
import beast.base.spec.inference.parameter.RealScalarParam;
import emat.initalisation.ParsimonyMutationsInitialiser;
import emat.operators.QuantileWarpOperator;
import emat.state.Mutation;
import emat.state.Mutations;
import org.apache.commons.math4.legacy.linear.Array2DRowRealMatrix;
import org.apache.commons.math4.legacy.linear.LUDecomposition;
import org.junit.jupiter.api.Test;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertTrue;

public class QuantileWarpOperatorTest {

    private static final double POP_SIZE = 200.0;
    private static final double GROWTH_RATE = 8.0;
    private static final double CLOCK_RATE = 0.001;
    private static final int NUM_LEVELS = 21;

    private TreeParser tree;
    private Mutations mutations;
    private RealScalarParam<PositiveReal> popSize;
    private RealScalarParam<Real> growthRate;
    private RealScalarParam<PositiveReal> clockRate;
    private QuantileWarpOperator operator;

    private void setUp(List<Sequence> sequences) {
        Alignment alignment = new Alignment();
        alignment.initByName("sequence", sequences, "dataType", "nucleotide");

        this.tree = new TreeParser();
        this.tree.initByName("newick", "(((A:0.01,B:0.01):0.02,C:0.03):0.04,D:0.07);", "IsLabelledNewick", true, "adjustTipHeights", false);

        this.mutations = new Mutations();
        this.mutations.initByName("tree", this.tree, "alignment", alignment);
        ParsimonyMutationsInitialiser initialiser = new ParsimonyMutationsInitialiser();
        initialiser.initByName("mutations", this.mutations);
        initialiser.initStateNodes();

        this.popSize = new RealScalarParam<>(POP_SIZE, PositiveReal.INSTANCE);
        this.growthRate = new RealScalarParam<>(GROWTH_RATE, Real.INSTANCE);
        this.clockRate = new RealScalarParam<>(CLOCK_RATE, PositiveReal.INSTANCE);

        this.operator = new QuantileWarpOperator();
        this.operator.initByName("weight", 1.0, "tree", this.tree, "mutations", this.mutations,
                "popSize", this.popSize, "growthRate", this.growthRate, "clockRate", this.clockRate,
                "sdLogPopSize", 0.08, "sdGrowthRate", 0.3, "correlation", 0.9, "optimise", false);

        // observations whose quantiles grow with the population size and fall with the growth rate
        for (int i = 0; i < 60; i++) {
            double logPopSize = Math.log(POP_SIZE) + 0.3 * Math.sin(1.7 * i);
            double growth = GROWTH_RATE + 2.0 * Math.cos(2.3 * i);
            double[] logQuantiles = new double[NUM_LEVELS];
            for (int q = 0; q < NUM_LEVELS; q++) {
                double level = (q + 1.0) / (NUM_LEVELS + 1.0);
                logQuantiles[q] = Math.log(0.002 + 0.12 * level * level) + (0.4 * level) * (logPopSize - Math.log(POP_SIZE)) - 0.03 * (1.0 - level) * (growth - GROWTH_RATE);
            }
            this.operator.observe(logPopSize, growth, logQuantiles);
        }
        this.operator.fit();
    }

    private void setUpWithMutations() {
        this.setUp(List.of(new Sequence("A", "ACGTAC"), new Sequence("B", "ACGTTC"), new Sequence("C", "ACCTTC"), new Sequence("D", "TCCTTA")));
    }

    private void setUpWithoutMutations() {
        this.setUp(List.of(new Sequence("A", "ACGTAC"), new Sequence("B", "ACGTAC"), new Sequence("C", "ACGTAC"), new Sequence("D", "ACGTAC")));
    }

    /** A proposal is rejected until the map has been fitted. */
    @Test
    public void testRejectsBeforeFit() {
        this.setUpWithMutations();
        QuantileWarpOperator unfitted = new QuantileWarpOperator();
        unfitted.initByName("weight", 1.0, "tree", this.tree, "mutations", this.mutations,
                "popSize", this.popSize, "growthRate", this.growthRate, "clockRate", this.clockRate,
                "sdLogPopSize", 0.08, "sdGrowthRate", 0.3, "correlation", 0.9);

        assertEquals(Double.NEGATIVE_INFINITY, unfitted.proposal());
    }

    /** The clock rate keeps the product with the tree length, and the parameters take the shifted values. */
    @Test
    public void testClockRateKeepsMutationRateTimesTreeLength() {
        this.setUpWithMutations();
        double before = this.clockRate.get() * this.computeTreeLength();

        double logHastingsRatio = this.operator.warp(0.2, -0.7);

        assertTrue(Double.isFinite(logHastingsRatio));
        assertEquals(before, this.clockRate.get() * this.computeTreeLength(), 1e-12);
        assertEquals(POP_SIZE * Math.exp(0.2), this.popSize.get(), 1e-9);
        assertEquals(GROWTH_RATE - 0.7, this.growthRate.get(), 1e-9);
    }

    /** A step of zero leaves the tree alone and has a Hastings ratio of zero. */
    @Test
    public void testZeroStepIsIdentity() {
        this.setUpWithMutations();
        double[] oldHeights = this.getHeights();

        double logHastingsRatio = this.operator.warp(0.0, 0.0);

        assertEquals(0.0, logHastingsRatio, 1e-9);
        assertTrue(Arrays.equals(oldHeights, this.getHeights()) || this.maxDifference(oldHeights, this.getHeights()) < 1e-12);
    }

    /** The opposite step restores the tree, the mutations and the parameters, and negates the Hastings ratio. */
    @Test
    public void testOppositeStepUndoesMove() {
        this.setUpWithMutations();
        double[] oldHeights = this.getHeights();
        List<List<Mutation>> oldMutations = this.getMutations();

        double forward = this.operator.warp(0.3, 1.1);
        double backward = this.operator.warp(-0.3, -1.1);

        assertTrue(Double.isFinite(forward));
        assertTrue(this.maxDifference(oldHeights, this.getHeights()) > 1e-6 || Math.abs(forward) > 0.0);
        assertEquals(0.0, forward + backward, 1e-9);
        assertEquals(POP_SIZE, this.popSize.get(), 1e-9);
        assertEquals(GROWTH_RATE, this.growthRate.get(), 1e-9);
        assertEquals(CLOCK_RATE, this.clockRate.get(), 1e-15);
        assertEquals(0.0, this.maxDifference(oldHeights, this.getHeights()), 1e-12);

        List<List<Mutation>> newMutations = this.getMutations();
        for (int i = 0; i < oldMutations.size(); i++) {
            for (int j = 0; j < oldMutations.get(i).size(); j++) {
                assertEquals(oldMutations.get(i).get(j).time(), newMutations.get(i).get(j).time(), 1e-12);
            }
        }
    }

    /** Without mutations, the Hastings ratio is the log determinant of the map of the internal heights and the clock rate plus the shift. */
    @Test
    public void testHastingsRatioMatchesNumericalJacobian() {
        this.setUpWithoutMutations();
        double logPopSizeShift = 0.25;
        double growthRateShift = -0.9;

        List<Node> internalNodes = new ArrayList<>();
        for (Node node : this.tree.getNodesAsArray()) {
            if (!node.isLeaf()) {
                internalNodes.add(node);
            }
        }
        double[] baseHeights = internalNodes.stream().mapToDouble(Node::getHeight).toArray();

        int size = internalNodes.size() + 1;
        double[][] jacobian = new double[size][size];
        double epsilon = 1e-7;
        double logHastingsRatio = 0.0;
        for (int column = -1; column < size; column++) {
            double[] heights = baseHeights.clone();
            double clock = CLOCK_RATE;
            if (column >= 0 && column < size - 1) {
                heights[column] += epsilon;
            } else if (column == size - 1) {
                clock += epsilon * CLOCK_RATE;
            }
            double[] output = this.warpFrom(internalNodes, heights, clock, logPopSizeShift, growthRateShift);
            if (column == -1) {
                logHastingsRatio = output[size];
                continue;
            }
            double[] base = this.warpFrom(internalNodes, baseHeights, CLOCK_RATE, logPopSizeShift, growthRateShift);
            double step = column == size - 1 ? epsilon * CLOCK_RATE : epsilon;
            for (int row = 0; row < size; row++) {
                jacobian[row][column] = (output[row] - base[row]) / step;
            }
        }

        double logDeterminant = Math.log(Math.abs(new LUDecomposition(new Array2DRowRealMatrix(jacobian)).getDeterminant()));
        assertEquals(logDeterminant + logPopSizeShift, logHastingsRatio, 1e-4);
    }

    /** Sets the internal heights and the clock rate, warps, and returns the new heights, the clock rate and the Hastings ratio. */
    private double[] warpFrom(List<Node> internalNodes, double[] heights, double clock, double logPopSizeShift, double growthRateShift) {
        for (int i = 0; i < heights.length; i++) {
            internalNodes.get(i).setHeight(heights[i]);
        }
        this.popSize.set(POP_SIZE);
        this.growthRate.set(GROWTH_RATE);
        this.clockRate.set(clock);

        double logHastingsRatio = this.operator.warp(logPopSizeShift, growthRateShift);
        double[] output = new double[heights.length + 2];
        for (int i = 0; i < heights.length; i++) {
            output[i] = internalNodes.get(i).getHeight();
        }
        output[heights.length] = this.clockRate.get();
        output[heights.length + 1] = logHastingsRatio;
        return output;
    }

    private double computeTreeLength() {
        double length = 0.0;
        for (Node node : this.tree.getNodesAsArray()) {
            length += node.getLength();
        }
        return length;
    }

    private double maxDifference(double[] first, double[] second) {
        double difference = 0.0;
        for (int i = 0; i < first.length; i++) {
            difference = Math.max(difference, Math.abs(first[i] - second[i]));
        }
        return difference;
    }

    private double[] getHeights() {
        return Arrays.stream(this.tree.getNodesAsArray()).mapToDouble(Node::getHeight).toArray();
    }

    private List<List<Mutation>> getMutations() {
        List<List<Mutation>> branchMutations = new ArrayList<>();
        for (Node node : this.tree.getNodesAsArray()) {
            branchMutations.add(new ArrayList<>(this.mutations.getMutations(node)));
        }
        return branchMutations;
    }

}
