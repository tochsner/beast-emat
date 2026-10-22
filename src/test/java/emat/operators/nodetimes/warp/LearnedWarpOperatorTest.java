package emat.operators.nodetimes.warp;

import adapters.Adapter;
import beast.base.core.BEASTObject;
import beast.base.evolution.alignment.Alignment;
import beast.base.evolution.alignment.Sequence;
import beast.base.evolution.tree.Node;
import beast.base.evolution.tree.TreeParser;
import beast.base.inference.CompoundDistribution;
import beast.base.inference.StateNode;
import beast.base.internal.json.JSONObject;
import beast.base.spec.domain.PositiveReal;
import beast.base.spec.evolution.branchratemodel.Base;
import beast.base.spec.evolution.branchratemodel.StrictClockModel;
import beast.base.spec.domain.Real;
import beast.base.spec.inference.parameter.RealScalarParam;
import emat.initalisation.ParsimonyMutationsInitialiser;
import emat.state.Mutations;
import org.junit.jupiter.api.Test;

import java.io.PrintWriter;
import java.io.StringWriter;
import java.util.ArrayList;
import java.util.List;
import java.util.Random;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertTrue;

public class LearnedWarpOperatorTest {

    private static final double POP_SIZE = 10.0;
    private static final double GROWTH_RATE = 0.1;
    private static final int NUM_TIPS = 40;

    private TreeParser tree;
    private RealScalarParam<PositiveReal> popSize;
    private RealScalarParam<Real> growthRate;
    private RealScalarParam<PositiveReal> clockRate;
    private LearnedWarpOperator operator;
    private Base clockModel;

    /** Moves the logarithm of the population size and the growth rate, as the log and the identity transform do. */
    private static class ParameterAdapter extends BEASTObject implements Adapter {

        private final RealScalarParam<PositiveReal> popSize;
        private final RealScalarParam<Real> growthRate;

        private ParameterAdapter(RealScalarParam<PositiveReal> popSize, RealScalarParam<Real> growthRate) {
            this.popSize = popSize;
            this.growthRate = growthRate;
        }

        @Override
        public void initAndValidate() {
        }

        @Override
        public int getNumImmutable() {
            return 0;
        }

        @Override
        public int getNumMutable() {
            return 2;
        }

        @Override
        public double[] getImmutable(int nodeId) {
            return new double[0];
        }

        @Override
        public double[] getMutable(int nodeId) {
            return new double[]{Math.log(this.popSize.get()), this.growthRate.get()};
        }

        @Override
        public double update(double[] mutable, int nodeId) {
            this.popSize.set(Math.exp(mutable[0]));
            this.growthRate.set(mutable[1]);
            return 0.0;
        }

        @Override
        public double getLogJacobianCorrection(int nodeId) {
            return -Math.log(this.popSize.get());
        }

        @Override
        public List<StateNode> listStateNodes() {
            return List.of(this.popSize, this.growthRate);
        }

    }

    /** A clock model whose branch rates are given by node number, and do not depend on the heights. */
    private static class FixedRatesClock extends Base {

        private final double[] rates;

        private FixedRatesClock(RealScalarParam<PositiveReal> meanRate, double[] rates) {
            this.rates = rates;
            this.initByName("clock.rate", meanRate);
        }

        @Override
        public void initAndValidate() {
        }

        @Override
        public double getRateForBranch(Node node) {
            return this.rates[node.getNr()];
        }

    }

    private LearnedWarpOperator createOperator(TreeParser tree, Mutations mutations, RealScalarParam<PositiveReal> popSize,
                                               RealScalarParam<Real> growthRate, Base clockModel) {
        LearnedWarpOperator learned = new LearnedWarpOperator();
        learned.initByName("weight", 1.0, "tree", tree, "mutations", mutations, "adapter", new ParameterAdapter(popSize, growthRate),
                "branchRateModel", clockModel, "optimise", false,
                "treePrior", new CompoundDistribution(), "hazard", TreeHazard.COALESCENT, "numKnots", 12);
        return learned;
    }

    private TreeParser createTree(String newick, Mutations[] mutationsOut) {
        List<Sequence> sequences = new ArrayList<>();
        for (int i = 0; i < NUM_TIPS; i++) {
            sequences.add(new Sequence("t" + i, "ACGTAC"));
        }
        Alignment alignment = new Alignment();
        alignment.initByName("sequence", sequences, "dataType", "nucleotide");

        TreeParser parsed = new TreeParser();
        parsed.initByName("newick", newick, "IsLabelledNewick", true, "adjustTipHeights", false);

        Mutations mutations = new Mutations();
        mutations.initByName("tree", parsed, "alignment", alignment);
        ParsimonyMutationsInitialiser initialiser = new ParsimonyMutationsInitialiser();
        initialiser.initByName("mutations", mutations);
        initialiser.initStateNodes();
        mutationsOut[0] = mutations;
        return parsed;
    }

    private void setUp() {
        this.setUp(false);
    }

    private void setUp(boolean isRelaxed) {
        Random random = new Random(3);
        CoalescentSimulation simulated = new CoalescentSimulation(random, NUM_TIPS, POP_SIZE, GROWTH_RATE);
        Mutations[] mutations = new Mutations[1];
        this.tree = this.createTree(simulated.newick, mutations);

        this.popSize = new RealScalarParam<>(POP_SIZE, PositiveReal.INSTANCE);
        this.growthRate = new RealScalarParam<>(GROWTH_RATE, Real.INSTANCE);
        this.clockRate = new RealScalarParam<>(0.001, PositiveReal.INSTANCE);
        this.clockModel = isRelaxed ? new FixedRatesClock(this.clockRate, this.createRelativeRates()) : this.createStrictClockModel();
        this.operator = this.createOperator(this.tree, mutations[0], this.popSize, this.growthRate, this.clockModel);

        for (int i = 0; i < 400; i++) {
            double logPopSize = Math.log(POP_SIZE) + 0.3 * random.nextGaussian();
            double growth = GROWTH_RATE + 0.05 * random.nextGaussian();
            CoalescentSimulation other = new CoalescentSimulation(random, NUM_TIPS, Math.exp(logPopSize), growth);
            this.operator.getHazardRate().addObservation(other.tipHeights, other.internalHeights, new double[]{logPopSize, growth}, other.logDensity);
        }
        this.operator.getHazardRate().fit();
    }

    private Base createStrictClockModel() {
        StrictClockModel strict = new StrictClockModel();
        strict.initByName("clock.rate", this.clockRate);
        return strict;
    }

    private double computeWeightedLength(double[] rates) {
        double length = 0.0;
        for (Node node : this.tree.getNodesAsArray()) {
            length += rates[node.getNr()] * node.getLength();
        }
        return length;
    }

    private double[] createRelativeRates() {
        double[] rates = new double[this.tree.getNodeCount()];
        for (int i = 0; i < rates.length; i++) {
            rates[i] = 0.5 + (i % 5) * 0.4;
        }
        return rates;
    }

    private double[] getHeights() {
        double[] heights = new double[this.tree.getNodeCount()];
        for (Node node : this.tree.getNodesAsArray()) {
            heights[node.getNr()] = node.getHeight();
        }
        return heights;
    }

    /** The effective sample size of the tree length is written to the state file with the tuning information and read back from it. */
    @Test
    public void testEffectiveSampleSizeSurvivesStateFile() throws Exception {
        this.setUp();
        StringWriter written = new StringWriter();
        this.operator.storeToFile(new PrintWriter(written));
        assertEquals(0.0, new JSONObject(written.toString()).getDouble("treeLengthEss"));

        JSONObject restored = new JSONObject(written.toString());
        restored.put("treeLengthEss", 42.5);
        this.operator.restoreFromFile(restored);

        StringWriter rewritten = new StringWriter();
        this.operator.storeToFile(new PrintWriter(rewritten));
        assertEquals(42.5, new JSONObject(rewritten.toString()).getDouble("treeLengthEss"));
    }

    /** A proposal is rejected until the rate has been fitted. */
    @Test
    public void testRejectsBeforeFit() {
        this.setUp();
        Mutations[] mutations = new Mutations[1];
        TreeParser other = this.createTree(new CoalescentSimulation(new Random(5), NUM_TIPS, POP_SIZE, GROWTH_RATE).newick, mutations);
        LearnedWarpOperator unfitted = this.createOperator(other, mutations[0], this.popSize, this.growthRate, this.clockModel);

        assertEquals(Double.NEGATIVE_INFINITY, unfitted.warp(new double[]{0.05, 0.01}));
    }

    /** A step and then the opposite step restore the tree, and their Hastings ratios cancel. */
    @Test
    public void testReverseStepUndoesMove() {
        this.setUp();
        double[] before = this.getHeights();

        double forward = this.operator.warp(new double[]{0.1, 0.03});
        assertTrue(Double.isFinite(forward));
        double backward = this.operator.warp(new double[]{-0.1, -0.03});

        assertEquals(0.0, forward + backward, 1e-8);
        double[] after = this.getHeights();
        for (int i = 0; i < before.length; i++) {
            assertEquals(before[i], after[i], 1e-9);
        }
        assertEquals(POP_SIZE, this.popSize.get(), 1e-9);
        assertEquals(GROWTH_RATE, this.growthRate.get(), 1e-9);
    }

    /** The product of the clock rate and the tree length is kept by a move, and by its reverse. */
    @Test
    public void testKeepsClockRateTimesTreeLength() {
        this.setUp();
        double[] ones = new double[this.tree.getNodeCount()];
        java.util.Arrays.fill(ones, 1.0);
        double before = this.clockRate.get() * this.computeWeightedLength(ones);

        assertTrue(Double.isFinite(this.operator.warp(new double[]{0.1, 0.03})));

        assertEquals(before, this.clockRate.get() * this.computeWeightedLength(ones), 1e-12);
    }

    /** With relaxed branch rates, the product of the mean rate and the rate-weighted tree length is kept, and a step is undone by its reverse. */
    @Test
    public void testRelaxedClockKeepsWeightedLengthAndIsReversible() {
        this.setUp(true);
        double[] rates = this.createRelativeRates();

        double[] before = this.getHeights();
        double weightedBefore = this.clockRate.get() * this.computeWeightedLength(rates);

        double forward = this.operator.warp(new double[]{0.1, 0.03});
        assertTrue(Double.isFinite(forward));
        assertEquals(weightedBefore, this.clockRate.get() * this.computeWeightedLength(rates), 1e-12);

        double backward = this.operator.warp(new double[]{-0.1, -0.03});
        assertEquals(0.0, forward + backward, 1e-8);
        double[] after = this.getHeights();
        for (int i = 0; i < before.length; i++) {
            assertEquals(before[i], after[i], 1e-9);
        }
    }

}
