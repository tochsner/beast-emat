package esat.stochasticmapping;

import beast.base.evolution.alignment.Alignment;
import beast.base.evolution.alignment.Sequence;
import beast.base.evolution.tree.Node;
import beast.base.evolution.tree.Tree;
import beast.base.evolution.tree.TreeParser;
import beast.base.inference.Operator;
import beast.base.inference.State;
import beast.base.spec.domain.PositiveReal;
import beast.base.spec.evolution.branchratemodel.StrictClockModel;
import beast.base.spec.evolution.sitemodel.SiteModel;
import beast.base.spec.evolution.substitutionmodel.Frequencies;
import beast.base.spec.evolution.substitutionmodel.HKY;
import beast.base.spec.inference.parameter.RealScalarParam;
import beast.base.spec.inference.parameter.SimplexParam;
import beast.base.util.Randomizer;
import esat.helper.TipCompatibility;
import esat.prior.GeneticPrior;
import esat.state.NodeStates;
import esat.state.StateChanges;
import org.junit.jupiter.api.Test;

import java.util.ArrayList;
import java.util.List;
import java.util.Random;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertTrue;

public class NodeStateSamplerTest {

    private static final String NEWICK = "(((A:0.3,B:0.3):0.4,(C:0.5,D:0.5):0.2):0.3,(E:0.6,F:0.6):0.4);";
    private static final String SHORT_NEWICK = "(((A:0.003,B:0.003):0.004,(C:0.005,D:0.005):0.002):0.003,(E:0.006,F:0.006):0.004);";
    private static final int NUM_STATES = 4;
    private static final int NUM_SAMPLES = 20000;

    // the alignment has ambiguous (R, Y) and missing (N, -) sites
    private static final List<Sequence> SEQUENCES = List.of(
            new Sequence("A", "ACGTACGTACGT"),
            new Sequence("B", "ACRAACGNNCGT"),
            new Sequence("C", "TCGAACCTAC-A"),
            new Sequence("D", "TCGTACCTAGGA"),
            new Sequence("E", "GCGTTNNNNNGT"),
            new Sequence("F", "GCGYTCGAACTT")
    );

    // three sites, so that every configuration of a node can be enumerated
    private static final List<Sequence> SHORT_SEQUENCES = List.of(
            new Sequence("A", "ACG"),
            new Sequence("B", "ANR"),
            new Sequence("C", "TCA"),
            new Sequence("D", "TGA"),
            new Sequence("E", "GNN"),
            new Sequence("F", "GCY")
    );

    /** The tree, the states, the model and the sampler of a test. */
    private record Fixture(Tree tree, NodeStates nodeStates, GeneticPrior geneticPrior, HKY hky,
                           RealScalarParam<PositiveReal> clockRate, SubstitutionProcess substitutionProcess,
                           NodeStateSampler sampler) {
    }

    private final Random random = new Random(17);

    private final Operator operator = new Operator() {
        @Override
        public void initAndValidate() {
        }

        @Override
        public double proposal() {
            return 0.0;
        }
    };

    /* Density */

    /**
     * Compares the density of the sampler to the sum of the log full conditionals over the
     * sites, which the test computes from the substitution model itself, for the current
     * states and for random states of every node.
     */
    @Test
    public void testLogDensityMatchesConditionals() {
        Fixture fixture = this.createFixture(this.parseTree(NEWICK), SEQUENCES);

        for (int i = 0; i < 10; i++) {
            this.initialiseStates(fixture, this.drawCompatibleSequences(fixture));
            fixture.substitutionProcess().update();

            for (Node node : fixture.tree().getNodesAsArray()) {
                double[][] conditionals = this.computeConditionals(fixture, node);

                int[] currentSequence = this.getSequence(fixture.nodeStates(), node);
                assertEquals(
                        this.computeLogDensity(conditionals, currentSequence),
                        fixture.sampler().computeLogDensity(node, this.toChanges(fixture, node, currentSequence)),
                        1e-9
                );

                int[] randomSequence = this.drawCompatibleSequences(fixture)[node.getNr()];
                if (node.isLeaf()) {
                    // the observed sites are fixed
                    this.copyObservedSites(fixture, node, currentSequence, randomSequence);
                }
                assertEquals(
                        this.computeLogDensity(conditionals, randomSequence),
                        fixture.sampler().computeLogDensity(node, this.toChanges(fixture, node, randomSequence)),
                        1e-9
                );
            }
        }
    }

    /** Sums the density of every node over all of its configurations at three sites, which must give one. */
    @Test
    public void testLogDensityNormalises() {
        Fixture fixture = this.createFixture(this.parseTree(NEWICK), SHORT_SEQUENCES);
        this.initialiseStates(fixture, this.drawCompatibleSequences(fixture));
        fixture.substitutionProcess().update();

        for (Node node : fixture.tree().getNodesAsArray()) {
            double total = 0.0;
            for (int configuration = 0; configuration < 64; configuration++) {
                int[] sequence = {configuration % 4, (configuration / 4) % 4, configuration / 16};
                total += Math.exp(fixture.sampler().computeLogDensity(node, this.toChanges(fixture, node, sequence)));
            }
            assertEquals(1.0, total, 1e-12, "Node " + node.getNr());
        }
    }

    /* Sampling */

    /**
     * Compares the frequencies of the sampled states at every site with the full
     * conditional, at branch lengths at which the thinning visits most sites.
     */
    @Test
    public void testSamplesMatchConditionals() {
        Fixture fixture = this.createFixture(this.parseTree(NEWICK), SEQUENCES);
        this.initialiseStates(fixture, this.drawCompatibleSequences(fixture));
        fixture.substitutionProcess().update();

        Randomizer.setSeed(5);
        for (Node node : fixture.tree().getNodesAsArray()) {
            this.assertSamplesMatchConditionals(fixture, node);
        }
    }

    /**
     * Compares the frequencies of the sampled states with the full conditional on a long
     * alignment with short branches, on which the thinning skips most sites and changes are
     * rare.
     */
    @Test
    public void testSamplesMatchConditionalsOnShortBranches() {
        List<Sequence> sequences = this.createLongSequences(200);
        Fixture fixture = this.createFixture(this.parseTree(SHORT_NEWICK), sequences);
        this.initialiseStates(fixture, this.drawSimilarSequences(fixture));
        fixture.substitutionProcess().update();

        Randomizer.setSeed(6);
        for (Node node : fixture.tree().getNodesAsArray()) {
            this.assertSamplesMatchConditionals(fixture, node);
        }
    }

    /**
     * Samples the given node repeatedly and compares the counts of the states at every site
     * with their expectation by a chi-square statistic. Cells with a small expectation are
     * pooled into one, and cells without probability must stay empty.
     */
    private void assertSamplesMatchConditionals(Fixture fixture, Node node) {
        double[][] conditionals = this.computeConditionals(fixture, node);
        int[] reference = fixture.nodeStates().getReferenceSequence();
        int numSites = reference.length;

        long[][] counts = new long[numSites][NUM_STATES];
        for (int i = 0; i < NUM_SAMPLES; i++) {
            StateChanges changes = fixture.sampler().sample(node);
            for (int site = 0; site < numSites; site++) {
                counts[site][changes.getStateOrDefault(site, reference[site])]++;
            }
        }

        double chiSquare = 0.0;
        int numCells = 0;
        double pooledObserved = 0.0;
        double pooledExpected = 0.0;
        for (int site = 0; site < numSites; site++) {
            for (int state = 0; state < NUM_STATES; state++) {
                double expected = NUM_SAMPLES * conditionals[site][state];
                double observed = counts[site][state];
                if (expected == 0.0) {
                    assertEquals(0.0, observed, "Impossible state " + state + " at site " + site + " of node " + node.getNr());
                } else if (expected < 5.0) {
                    pooledObserved += observed;
                    pooledExpected += expected;
                } else {
                    chiSquare += (observed - expected) * (observed - expected) / expected;
                    numCells++;
                }
            }
        }
        if (pooledExpected > 0.0) {
            chiSquare += (pooledObserved - pooledExpected) * (pooledObserved - pooledExpected) / pooledExpected;
            numCells++;
        }

        // the number of cells bounds the degrees of freedom, and the margin lies far in the tail
        double bound = numCells + 6.0 * Math.sqrt(2.0 * numCells) + 10.0;
        assertTrue(chiSquare < bound, "Chi-square " + chiSquare + " above " + bound + " for node " + node.getNr());
    }

    /* Gibbs updates */

    /**
     * Applies the samples of random nodes in the store, accept and restore cycle of MCMC. The
     * prior must stay consistent with a computation from scratch, and as the sampler draws
     * from the full conditional, the change of the prior must equal the change of the density.
     */
    @Test
    public void testGibbsUpdatesMatchPrior() {
        Fixture fixture = this.createFixture(this.parseTree(NEWICK), SEQUENCES);
        this.initialiseStates(fixture, this.drawCompatibleSequences(fixture));
        GeneticPrior geneticPrior = fixture.geneticPrior();
        NodeStates nodeStates = fixture.nodeStates();

        State state = new State();
        state.initByName("stateNode", fixture.tree(), "stateNode", nodeStates);
        state.initialise();
        state.setPosterior(geneticPrior);
        state.setEverythingDirty(true);
        double logP = geneticPrior.calculateLogP();
        state.setEverythingDirty(false);

        Randomizer.setSeed(7);
        for (long i = 0; i < 3000; i++) {
            Node node = fixture.tree().getNode(this.random.nextInt(fixture.tree().getNodeCount()));
            state.store(i);

            fixture.substitutionProcess().update();
            StateChanges currentChanges = node.isRoot() ? StateChanges.empty(node.getNr()) : nodeStates.getChanges(node);
            StateChanges sampledChanges = fixture.sampler().sample(node);
            double logDensityRatio = fixture.sampler().computeLogDensity(node, sampledChanges)
                    - fixture.sampler().computeLogDensity(node, currentChanges);

            if (node.isRoot()) {
                nodeStates.applyRootStates(sampledChanges, this.operator);
            } else {
                nodeStates.applyChanges(node, sampledChanges, this.operator);
            }

            state.storeCalculationNodes();
            state.checkCalculationNodesDirtiness();
            double newLogP = geneticPrior.calculateLogP();

            assertTrue(Double.isFinite(newLogP));
            assertEquals(this.computeLogPFromScratch(geneticPrior), newLogP, 1e-9);
            assertEquals(logDensityRatio, newLogP - logP, 1e-8, "Node " + node.getNr());

            if (this.random.nextBoolean()) {
                state.acceptCalculationNodes();
                logP = newLogP;
            } else {
                state.restore();
                state.restoreCalculationNodes();
                assertEquals(logP, geneticPrior.getCurrentLogP(), 1e-9);
            }

            state.setEverythingDirty(false);
        }
    }

    private double computeLogPFromScratch(GeneticPrior geneticPrior) {
        GeneticPrior freshGeneticPrior = new GeneticPrior();
        freshGeneticPrior.initByName(
                "data", geneticPrior.dataInput.get(),
                "tree", geneticPrior.treeInput.get(),
                "siteModel", geneticPrior.siteModelInput.get(),
                "branchRateModel", geneticPrior.branchRateModelInput.get(),
                "nodeStates", geneticPrior.nodeStatesInput.get()
        );
        return freshGeneticPrior.calculateLogP();
    }

    /* Exact conditionals */

    /**
     * Computes the full conditional of the states of the given node at every site from the
     * transition probabilities of the substitution model. The observed sites of a tip keep
     * their current state.
     */
    private double[][] computeConditionals(Fixture fixture, Node node) {
        NodeStates nodeStates = fixture.nodeStates();
        TipCompatibility tipCompatibility = fixture.geneticPrior().getTipCompatibility();
        double[] frequencies = fixture.hky().getFrequencies();
        double rate = fixture.clockRate().get();

        double[] upperMatrix = new double[NUM_STATES * NUM_STATES];
        if (!node.isRoot()) {
            fixture.hky().getTransitionProbabilities(node, node.getParent().getHeight(), node.getHeight(), rate, upperMatrix);
        }
        List<double[]> childMatrices = new ArrayList<>();
        for (Node child : node.getChildren()) {
            double[] matrix = new double[NUM_STATES * NUM_STATES];
            fixture.hky().getTransitionProbabilities(child, node.getHeight(), child.getHeight(), rate, matrix);
            childMatrices.add(matrix);
        }

        double[][] conditionals = new double[nodeStates.getNumSites()][NUM_STATES];
        for (int site = 0; site < conditionals.length; site++) {
            int compatibleStates = tipCompatibility.getCompatibleStates(node, site);
            if (Integer.bitCount(compatibleStates) == 1) {
                conditionals[site][nodeStates.getState(node, site)] = 1.0;
                continue;
            }

            double total = 0.0;
            for (int state = 0; state < NUM_STATES; state++) {
                double weight = node.isRoot()
                        ? frequencies[state]
                        : upperMatrix[nodeStates.getState(node.getParent(), site) * NUM_STATES + state];
                for (int c = 0; c < childMatrices.size(); c++) {
                    weight *= childMatrices.get(c)[state * NUM_STATES + nodeStates.getState(node.getChild(c), site)];
                }
                if ((compatibleStates & (1 << state)) == 0) {
                    weight = 0.0;
                }
                conditionals[site][state] = weight;
                total += weight;
            }
            for (int state = 0; state < NUM_STATES; state++) {
                conditionals[site][state] /= total;
            }
        }
        return conditionals;
    }

    private double computeLogDensity(double[][] conditionals, int[] sequence) {
        double logDensity = 0.0;
        for (int site = 0; site < sequence.length; site++) {
            logDensity += Math.log(conditionals[site][sequence[site]]);
        }
        return logDensity;
    }

    /* Fixtures */

    private TreeParser parseTree(String newick) {
        TreeParser tree = new TreeParser();
        tree.initByName(
                "newick", newick,
                "IsLabelledNewick", true,
                "adjustTipHeights", false
        );
        return tree;
    }

    private Fixture createFixture(Tree tree, List<Sequence> sequences) {
        Alignment alignment = new Alignment();
        alignment.initByName("sequence", sequences, "dataType", "nucleotide");

        NodeStates nodeStates = new NodeStates();
        nodeStates.initByName("tree", tree, "alignment", alignment);

        Frequencies frequencies = new Frequencies();
        frequencies.initByName("frequencies", new SimplexParam(new double[]{0.1, 0.2, 0.3, 0.4}));

        RealScalarParam<PositiveReal> kappa = new RealScalarParam<>(3.0, PositiveReal.INSTANCE);
        HKY hky = new HKY();
        hky.initByName("kappa", kappa, "frequencies", frequencies);

        SiteModel siteModel = new SiteModel();
        siteModel.initByName("substModel", hky);

        RealScalarParam<PositiveReal> clockRate = new RealScalarParam<>(0.5, PositiveReal.INSTANCE);
        StrictClockModel clockModel = new StrictClockModel();
        clockModel.initByName("clock.rate", clockRate);

        GeneticPrior geneticPrior = new GeneticPrior();
        geneticPrior.initByName(
                "data", alignment,
                "tree", tree,
                "siteModel", siteModel,
                "branchRateModel", clockModel,
                "nodeStates", nodeStates
        );

        SubstitutionProcess substitutionProcess = new SubstitutionProcess(geneticPrior);
        NodeStateSampler sampler = new NodeStateSampler(nodeStates, substitutionProcess);
        return new Fixture(tree, nodeStates, geneticPrior, hky, clockRate, substitutionProcess, sampler);
    }

    /**
     * Creates tips that differ from a common random sequence at a few sites, with a run of
     * missing sites, gaps and ambiguous sites.
     */
    private List<Sequence> createLongSequences(int numSites) {
        String nucleotides = "ACGT";
        char[] base = new char[numSites];
        for (int site = 0; site < numSites; site++) {
            base[site] = nucleotides.charAt(this.random.nextInt(NUM_STATES));
        }

        List<Sequence> sequences = new ArrayList<>();
        for (String taxon : List.of("A", "B", "C", "D", "E", "F")) {
            char[] sequence = base.clone();
            for (int i = 0; i < 5; i++) {
                sequence[this.random.nextInt(numSites)] = nucleotides.charAt(this.random.nextInt(NUM_STATES));
            }
            for (int i = 0; i < 5; i++) {
                int site = this.random.nextInt(numSites);
                sequence[site] = sequence[site] == 'A' || sequence[site] == 'G' ? 'R' : 'Y';
            }
            int start = this.random.nextInt(numSites - 40);
            for (int site = start; site < start + 30; site++) {
                sequence[site] = this.random.nextInt(10) == 0 ? '-' : 'N';
            }
            sequences.add(new Sequence(taxon, new String(sequence)));
        }
        return sequences;
    }

    /**
     * Draws a random sequence for the root, copies it to the inner nodes with a few changes,
     * and gives every tip the state of its parent at the latent sites where it is compatible.
     */
    private int[][] drawSimilarSequences(Fixture fixture) {
        TipCompatibility tipCompatibility = fixture.geneticPrior().getTipCompatibility();
        int numSites = fixture.nodeStates().getNumSites();
        int[][] sequences = new int[fixture.tree().getNodeCount()][];

        // the nodes are in preorder, so every parent comes before its children
        for (Node node : fixture.tree().getRoot().getAllChildNodesAndSelf()) {
            int[] sequence = new int[numSites];
            for (int site = 0; site < numSites; site++) {
                int compatibleStates = tipCompatibility.getCompatibleStates(node, site);
                int parentState = node.isRoot() ? this.random.nextInt(NUM_STATES) : sequences[node.getParent().getNr()][site];
                boolean isChanged = !node.isRoot() && this.random.nextInt(50) == 0;
                sequence[site] = (compatibleStates & (1 << parentState)) != 0 && !isChanged
                        ? parentState
                        : this.drawCompatibleState(compatibleStates);
            }
            sequences[node.getNr()] = sequence;
        }
        return sequences;
    }

    /** Draws a random sequence for every node, which for the tips is compatible with their data. */
    private int[][] drawCompatibleSequences(Fixture fixture) {
        TipCompatibility tipCompatibility = fixture.geneticPrior().getTipCompatibility();
        int numSites = fixture.nodeStates().getNumSites();

        int[][] sequences = new int[fixture.tree().getNodeCount()][numSites];
        for (Node node : fixture.tree().getNodesAsArray()) {
            for (int site = 0; site < numSites; site++) {
                sequences[node.getNr()][site] = this.drawCompatibleState(tipCompatibility.getCompatibleStates(node, site));
            }
        }
        return sequences;
    }

    private int drawCompatibleState(int compatibleStates) {
        while (true) {
            int state = this.random.nextInt(NUM_STATES);
            if ((compatibleStates & (1 << state)) != 0) {
                return state;
            }
        }
    }

    private void copyObservedSites(Fixture fixture, Node tip, int[] from, int[] to) {
        TipCompatibility tipCompatibility = fixture.geneticPrior().getTipCompatibility();
        for (int site = 0; site < from.length; site++) {
            if (Integer.bitCount(tipCompatibility.getCompatibleStates(tip, site)) == 1) {
                to[site] = from[site];
            }
        }
    }

    /** Sets the states of every node to the given sequences, with the sequence of the root as the reference. */
    private void initialiseStates(Fixture fixture, int[][] sequences) {
        int[] rootSequence = sequences[fixture.tree().getRoot().getNr()];
        List<StateChanges> changes = new ArrayList<>();
        for (int nodeNr = 0; nodeNr < sequences.length; nodeNr++) {
            changes.add(StateChanges.fromSequence(nodeNr, sequences[nodeNr], rootSequence));
        }
        fixture.nodeStates().initialiseStates(rootSequence, changes);
    }

    /** Returns the changes of the given node with the given sequence, relative to the current reference. */
    private StateChanges toChanges(Fixture fixture, Node node, int[] sequence) {
        return StateChanges.fromSequence(node.getNr(), sequence, fixture.nodeStates().getReferenceSequence());
    }

    private int[] getSequence(NodeStates nodeStates, Node node) {
        int[] sequence = new int[nodeStates.getNumSites()];
        for (int site = 0; site < sequence.length; site++) {
            sequence[site] = nodeStates.getState(node, site);
        }
        return sequence;
    }

}
