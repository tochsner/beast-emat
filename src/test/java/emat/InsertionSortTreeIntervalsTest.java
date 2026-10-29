package emat;

import beast.base.evolution.alignment.Taxon;
import beast.base.evolution.alignment.TaxonSet;
import beast.base.evolution.operator.Exchange;
import beast.base.evolution.operator.ScaleOperator;
import beast.base.evolution.operator.SubtreeSlide;
import beast.base.evolution.operator.Uniform;
import beast.base.evolution.operator.WilsonBalding;
import beast.base.evolution.tree.Node;
import beast.base.evolution.tree.TreeIntervals;
import beast.base.evolution.tree.TreeParser;
import beast.base.evolution.tree.coalescent.Coalescent;
import beast.base.evolution.tree.coalescent.ConstantPopulation;
import beast.base.inference.CompoundDistribution;
import beast.base.inference.Operator;
import beast.base.inference.State;
import beast.base.inference.parameter.RealParameter;
import beast.base.util.Randomizer;
import emat.tree.IncrementalStoreTree;
import emat.tree.InsertionSortTreeIntervals;
import org.junit.jupiter.api.Test;

import java.util.ArrayList;
import java.util.List;

import static org.junit.jupiter.api.Assertions.assertEquals;

public class InsertionSortTreeIntervalsTest {

    private static final int NUM_TIPS = 50;
    private static final int NUM_MOVES = 2000;
    private static final int NUM_STEPS = 20000;

    /**
     * Repeatedly moves the height of a random internal node of a serially sampled tree, with
     * some tips sampled at the same time, and checks that the intervals and lineage counts
     * match those of TreeIntervals after every move.
     */
    @Test
    public void testMatchesTreeIntervals() {
        this.assertMatchesTreeIntervals(new InsertionSortTreeIntervals());
    }

    private void assertMatchesTreeIntervals(TreeIntervals actual) {
        Randomizer.setSeed(1);

        TreeParser tree = new TreeParser();
        tree.initByName("newick", this.createRandomNewick(), "IsLabelledNewick", true, "adjustTipHeights", false);

        TreeIntervals expected = new TreeIntervals();
        expected.initByName("tree", tree);
        actual.initByName("tree", tree);

        this.assertSameIntervals(expected, actual);

        for (int move = 0; move < NUM_MOVES; move++) {
            Node node = tree.getNode(NUM_TIPS + Randomizer.nextInt(NUM_TIPS - 1));
            double lower = Math.max(node.getLeft().getHeight(), node.getRight().getHeight());
            double upper = node.isRoot() ? lower + 1.0 : node.getParent().getHeight();
            node.setHeight(lower + Randomizer.nextDouble() * (upper - lower));

            expected.setIntervalsUnknown();
            actual.setIntervalsUnknown();
            this.assertSameIntervals(expected, actual);
        }
    }

    /**
     * Runs a chain of core tree operators with random acceptances on an incremental store tree,
     * which makes the intervals update only the flagged nodes, and checks that the intervals
     * match those of TreeIntervals after every proposal and every restore. Some proposals are
     * not evaluated before they are accepted or rejected, as happens when no distribution
     * needs the intervals.
     */
    @Test
    public void testMatchesTreeIntervalsUnderStoreAndRestore() {
        Randomizer.setSeed(2);

        TreeParser parser = new TreeParser();
        parser.initByName("newick", this.createRandomNewick(), "IsLabelledNewick", true, "adjustTipHeights", false);
        IncrementalStoreTree tree = new IncrementalStoreTree();
        List<Taxon> taxa = new ArrayList<>();
        for (String taxon : parser.getTaxaNames()) {
            taxa.add(new Taxon(taxon));
        }
        tree.initByName("taxonset", new TaxonSet(taxa));
        tree.assignFrom(parser);

        TreeIntervals expected = new TreeIntervals();
        expected.initByName("tree", tree);
        InsertionSortTreeIntervals actual = new InsertionSortTreeIntervals();
        actual.initByName("tree", tree);

        // the coalescents connect the intervals to the state, which stores, restores and accepts them
        CompoundDistribution posterior = new CompoundDistribution();
        posterior.initByName("distribution", List.of(this.createCoalescent(expected), this.createCoalescent(actual)));

        State state = new State();
        state.initByName("stateNode", tree);
        state.initialise();
        state.setPosterior(posterior);
        state.setEverythingDirty(true);
        state.checkCalculationNodesDirtiness();
        posterior.calculateLogP();
        this.assertSameIntervals(expected, actual);
        state.acceptCalculationNodes();
        state.setEverythingDirty(false);

        List<Operator> operators = this.createOperators(tree);
        for (long i = 0; i < NUM_STEPS; i++) {
            state.store(i);
            Operator operator = operators.get(Randomizer.nextInt(operators.size()));
            double logHastingsRatio = operator.proposal();

            // like MCMC, a failed proposal only restores the state nodes, as no calculation node was stored
            if (logHastingsRatio == Double.NEGATIVE_INFINITY) {
                state.restore();
            } else {
                state.storeCalculationNodes();
                state.checkCalculationNodesDirtiness();
                if (Randomizer.nextDouble() < 0.8) {
                    posterior.calculateLogP();
                    expected.setIntervalsUnknown();
                    this.assertSameIntervals(expected, actual);
                }

                if (Randomizer.nextDouble() < 0.5) {
                    state.restore();
                    state.restoreCalculationNodes();
                } else {
                    state.acceptCalculationNodes();
                }
            }
            state.setEverythingDirty(false);

            if (Randomizer.nextDouble() < 0.8) {
                expected.setIntervalsUnknown();
                this.assertSameIntervals(expected, actual);
            }
        }
    }

    private Coalescent createCoalescent(TreeIntervals intervals) {
        ConstantPopulation population = new ConstantPopulation();
        population.initByName("popSize", new RealParameter("1.0"));

        Coalescent coalescent = new Coalescent();
        coalescent.initByName("treeIntervals", intervals, "populationModel", population);
        return coalescent;
    }

    private List<Operator> createOperators(IncrementalStoreTree tree) {
        SubtreeSlide subtreeSlide = new SubtreeSlide();
        subtreeSlide.initByName("tree", tree, "weight", 1.0, "size", 0.5, "optimise", false);

        Exchange wideExchange = new Exchange();
        wideExchange.initByName("tree", tree, "weight", 1.0, "isNarrow", false);

        WilsonBalding wilsonBalding = new WilsonBalding();
        wilsonBalding.initByName("tree", tree, "weight", 1.0);

        Uniform uniform = new Uniform();
        uniform.initByName("tree", tree, "weight", 1.0);

        ScaleOperator scaleOperator = new ScaleOperator();
        scaleOperator.initByName("tree", tree, "weight", 1.0, "scaleFactor", 0.9, "optimise", false);

        ScaleOperator rootScaleOperator = new ScaleOperator();
        rootScaleOperator.initByName("tree", tree, "weight", 1.0, "scaleFactor", 0.9, "rootOnly", true, "optimise", false);

        return List.of(subtreeSlide, wideExchange, wilsonBalding, uniform, uniform, uniform, scaleOperator, rootScaleOperator);
    }

    private void assertSameIntervals(TreeIntervals expected, TreeIntervals actual) {
        assertEquals(expected.getIntervalCount(), actual.getIntervalCount());
        for (int i = 0; i < expected.getIntervalCount(); i++) {
            assertEquals(expected.getInterval(i), actual.getInterval(i), 1e-12);
            assertEquals(expected.getLineageCount(i), actual.getLineageCount(i));
        }
    }

    /**
     * Creates a random tree by coalescing random pairs of lineages at increasing times, with the
     * tips sampled at one of a few shared times.
     */
    private String createRandomNewick() {
        double[] sampleTimes = {0.0, 0.2, 0.5};

        List<String> lineages = new ArrayList<>();
        List<Double> heights = new ArrayList<>();
        for (int i = 0; i < NUM_TIPS; i++) {
            lineages.add("t" + i);
            heights.add(sampleTimes[Randomizer.nextInt(sampleTimes.length)]);
        }

        double time = 0.5;
        while (lineages.size() > 1) {
            time += Randomizer.nextExponential(1.0);
            int a = Randomizer.nextInt(lineages.size());
            String left = lineages.remove(a) + ":" + (time - heights.remove(a));
            int b = Randomizer.nextInt(lineages.size());
            String right = lineages.remove(b) + ":" + (time - heights.remove(b));
            lineages.add("(" + left + "," + right + ")");
            heights.add(time);
        }
        return lineages.get(0) + ";";
    }
}
