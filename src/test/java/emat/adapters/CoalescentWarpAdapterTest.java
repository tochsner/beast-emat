package emat.adapters;

import beast.base.evolution.alignment.Alignment;
import beast.base.evolution.alignment.Sequence;
import beast.base.evolution.tree.TreeParser;
import beast.base.spec.domain.PositiveReal;
import beast.base.spec.domain.Real;
import beast.base.spec.inference.parameter.RealScalarParam;
import emat.initalisation.ParsimonyMutationsInitialiser;
import emat.operators.CoalescentWarpOperator;
import emat.state.Mutations;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;

import java.util.List;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertTrue;

public class CoalescentWarpAdapterTest {

    private static final double POP_SIZE = 200.0;
    private static final double GROWTH_RATE = 8.0;
    private static final double CLOCK_RATE = 0.001;

    private TreeParser tree;
    private Mutations mutations;
    private RealScalarParam<PositiveReal> popSize;
    private RealScalarParam<Real> growthRate;
    private RealScalarParam<PositiveReal> clockRate;
    private CoalescentWarpAdapter adapter;
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

        this.adapter = new CoalescentWarpAdapter();
        this.adapter.initByName("tree", this.tree, "mutations", this.mutations, "popSize", this.popSize,
                "growthRate", this.growthRate, "clockRate", this.clockRate);

        this.operator = new CoalescentWarpOperator();
        this.operator.initByName("weight", 1.0, "tree", this.tree, "mutations", this.mutations,
                "popSize", this.popSize, "growthRate", this.growthRate, "clockRate", this.clockRate,
                "sdLogPopSize", 0.08, "sdGrowthRate", 0.3, "correlation", 0.9, "optimise", false);
    }

    /** The coordinates are the logarithm of the population size and the growth rate. */
    @Test
    public void testMutableCoordinates() {
        double[] mutable = this.adapter.getMutable(0);

        assertEquals(2, this.adapter.getNumMutable());
        assertEquals(Math.log(POP_SIZE), mutable[0], 1e-12);
        assertEquals(GROWTH_RATE, mutable[1], 1e-12);
    }

    /** The correction before the update minus the correction after it is the Hastings ratio of the warp operator. */
    @Test
    public void testCorrectionDifferenceIsHastingsRatioOfWarp() {
        double[][] steps = {{0.2, -0.7}, {-0.15, 0.5}, {0.05, 1.3}};

        for (double[] step : steps) {
            double correctionBefore = this.adapter.getLogJacobianCorrection(0);
            double logHastingsRatio = this.operator.warp(step[0], step[1]);
            double correctionAfter = this.adapter.getLogJacobianCorrection(0);

            assertTrue(Double.isFinite(logHastingsRatio));
            assertEquals(logHastingsRatio, correctionBefore - correctionAfter, 1e-9);
        }
    }

    /** Updating the coordinates gives the same state as the warp operator, and the correction difference matches. */
    @Test
    public void testUpdateMovesStateToGivenCoordinates() {
        double[] mutable = this.adapter.getMutable(0);
        double correctionBefore = this.adapter.getLogJacobianCorrection(0);
        double product = this.clockRate.get() * this.computeTreeLength();

        this.adapter.update(new double[]{mutable[0] + 0.25, mutable[1] - 0.9}, 0);

        assertEquals(POP_SIZE * Math.exp(0.25), this.popSize.get(), 1e-9);
        assertEquals(GROWTH_RATE - 0.9, this.growthRate.get(), 1e-9);
        assertEquals(product, this.clockRate.get() * this.computeTreeLength(), 1e-12);
        assertTrue(Double.isFinite(correctionBefore - this.adapter.getLogJacobianCorrection(0)));
    }

    /** A step that cannot be mapped is made invalid by an infinite correction, and the next proposal starts clean. */
    @Test
    public void testUnmappableStepIsInvalid() {
        double[] mutable = this.adapter.getMutable(0);

        this.adapter.update(new double[]{mutable[0], -1.0e6}, 0);
        assertEquals(Double.POSITIVE_INFINITY, this.adapter.getLogJacobianCorrection(0));

        this.adapter.getMutable(0);
        assertTrue(Double.isFinite(this.adapter.getLogJacobianCorrection(0)));
    }

    private double computeTreeLength() {
        double length = 0.0;
        for (beast.base.evolution.tree.Node node : this.tree.getNodesAsArray()) {
            length += node.getLength();
        }
        return length;
    }

}
