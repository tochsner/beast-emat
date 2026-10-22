package emat;

import beast.base.evolution.alignment.Alignment;
import beast.base.evolution.alignment.Sequence;
import beast.base.evolution.tree.Node;
import beast.base.evolution.tree.TreeParser;
import beast.base.spec.domain.PositiveReal;
import beast.base.spec.domain.Real;
import beast.base.spec.inference.parameter.RealScalarParam;
import emat.initalisation.ParsimonyMutationsInitialiser;
import emat.operators.CoalescentWarpOperator;
import emat.state.Mutation;
import emat.state.Mutations;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;

import java.util.ArrayList;
import java.util.List;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertTrue;

public class CoalescentWarpOperatorTest {

    private static final double POP_SIZE = 200.0;
    private static final double GROWTH_RATE = 8.0;
    private static final double CLOCK_RATE = 0.001;
    private static final double TOLERANCE = 1e-10;

    private TreeParser tree;
    private Mutations mutations;
    private RealScalarParam<PositiveReal> popSize;
    private RealScalarParam<Real> growthRate;
    private RealScalarParam<PositiveReal> clockRate;
    private CoalescentWarpOperator operator;

    @BeforeEach
    public void setUp() {
        Alignment alignment = new Alignment();
        alignment.initByName("sequence", List.of(
                new Sequence("A", "ACGTAC"), new Sequence("B", "ACGTTC"), new Sequence("C", "ACCTTC"), new Sequence("D", "TCCTTA")
        ), "dataType", "nucleotide");

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

        this.operator = new CoalescentWarpOperator();
        this.operator.initByName("weight", 1.0, "tree", this.tree, "mutations", this.mutations,
                "popSize", this.popSize, "growthRate", this.growthRate, "clockRate", this.clockRate,
                "sdLogPopSize", 0.08, "sdGrowthRate", 0.3, "correlation", 0.9, "optimise", false);
    }

    /** The product of the clock rate and the tree length is the quantity the clock rate is set to keep. */
    @Test
    public void testClockRateKeepsMutationRateTimesTreeLength() {
        double before = this.clockRate.get() * this.computeTreeLength();

        double logHastingsRatio = this.operator.warp(0.2, -0.7);

        assertTrue(Double.isFinite(logHastingsRatio));
        assertEquals(before, this.clockRate.get() * this.computeTreeLength(), 1e-12);
        assertEquals(POP_SIZE * Math.exp(0.2), this.popSize.get(), TOLERANCE);
        assertEquals(GROWTH_RATE - 0.7, this.growthRate.get(), TOLERANCE);
    }

    /** Every internal node keeps its coalescent time above its older child, which the map is built to preserve. */
    @Test
    public void testIntervalsKeepTheirCoalescentTime() {
        double[] oldIntervals = this.computeCoalescentIntervals(POP_SIZE, GROWTH_RATE);

        this.operator.warp(0.15, -0.5);

        double[] newIntervals = this.computeCoalescentIntervals(this.popSize.get(), this.growthRate.get());
        for (int i = 0; i < oldIntervals.length; i++) {
            assertEquals(oldIntervals[i], newIntervals[i], 1e-12);
        }
    }

    /** The opposite step restores the tree, the mutations and the parameters, and negates the Hastings ratio. */
    @Test
    public void testOppositeStepUndoesMove() {
        double[] oldHeights = this.getHeights();
        List<List<Mutation>> oldMutations = this.getMutations();

        double forward = this.operator.warp(0.3, 1.1);
        double backward = this.operator.warp(-0.3, -1.1);

        assertEquals(0.0, forward + backward, 1e-9);
        assertEquals(POP_SIZE, this.popSize.get(), 1e-9);
        assertEquals(GROWTH_RATE, this.growthRate.get(), 1e-9);
        assertEquals(CLOCK_RATE, this.clockRate.get(), 1e-15);

        double[] newHeights = this.getHeights();
        for (int i = 0; i < oldHeights.length; i++) {
            assertEquals(oldHeights[i], newHeights[i], 1e-12);
        }
        List<List<Mutation>> newMutations = this.getMutations();
        for (int i = 0; i < oldMutations.size(); i++) {
            assertEquals(oldMutations.get(i).size(), newMutations.get(i).size());
            for (int j = 0; j < oldMutations.get(i).size(); j++) {
                assertEquals(oldMutations.get(i).get(j).time(), newMutations.get(i).get(j).time(), 1e-12);
            }
        }
    }

    private double[] computeCoalescentIntervals(double popSize, double growthRate) {
        List<Double> intervals = new ArrayList<>();
        for (Node node : this.tree.getNodesAsArray()) {
            if (!node.isLeaf()) {
                double childHeight = Math.max(node.getLeft().getHeight(), node.getRight().getHeight());
                intervals.add(this.computeCoalescentTime(node.getHeight(), popSize, growthRate)
                        - this.computeCoalescentTime(childHeight, popSize, growthRate));
            }
        }
        return intervals.stream().mapToDouble(Double::doubleValue).toArray();
    }

    private double computeCoalescentTime(double height, double popSize, double growthRate) {
        return Math.expm1(growthRate * height) / (growthRate * popSize);
    }

    private double computeTreeLength() {
        double length = 0.0;
        for (Node node : this.tree.getNodesAsArray()) {
            length += node.getLength();
        }
        return length;
    }

    private double[] getHeights() {
        return java.util.Arrays.stream(this.tree.getNodesAsArray()).mapToDouble(Node::getHeight).toArray();
    }

    private List<List<Mutation>> getMutations() {
        List<List<Mutation>> branchMutations = new ArrayList<>();
        for (Node node : this.tree.getNodesAsArray()) {
            branchMutations.add(new ArrayList<>(this.mutations.getMutations(node)));
        }
        return branchMutations;
    }

}
