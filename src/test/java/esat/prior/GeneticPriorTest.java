package esat.prior;

import beast.base.evolution.alignment.Alignment;
import beast.base.evolution.alignment.Sequence;
import beast.base.evolution.alignment.Taxon;
import beast.base.evolution.alignment.TaxonSet;
import beast.base.evolution.tree.Node;
import beast.base.evolution.tree.Tree;
import beast.base.evolution.tree.TreeParser;
import beast.base.inference.Operator;
import beast.base.inference.State;
import beast.base.spec.domain.PositiveReal;
import beast.base.spec.evolution.branchratemodel.StrictClockModel;
import beast.base.spec.evolution.likelihood.TreeLikelihood;
import beast.base.spec.evolution.sitemodel.SiteModel;
import beast.base.spec.evolution.substitutionmodel.Frequencies;
import beast.base.spec.evolution.substitutionmodel.HKY;
import beast.base.spec.inference.parameter.RealScalarParam;
import beast.base.spec.inference.parameter.SimplexParam;
import emat.tree.IncrementalStoreTree;
import esat.helper.TipCompatibility;
import esat.state.NodeStates;
import esat.state.BranchChanges;
import org.junit.jupiter.api.Test;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.Random;
import java.util.function.Supplier;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertTrue;

public class GeneticPriorTest {

    private static final String NEWICK = "(((A:0.3,B:0.3):0.4,(C:0.5,D:0.5):0.2):0.3,(E:0.6,F:0.6):0.4);";
    private static final int NUM_STATES = 4;

    // the alignment has ambiguous (R, Y) and missing (N, -) sites
    private static final List<Sequence> SEQUENCES = List.of(
            new Sequence("A", "ACGTACGTACGT"),
            new Sequence("B", "ACRAACGNNCGT"),
            new Sequence("C", "TCGAACCTAC-A"),
            new Sequence("D", "TCGTACCTAGGA"),
            new Sequence("E", "GCGTTNNNNNGT"),
            new Sequence("F", "GCGYTCGAACTT")
    );

    /** The tree, the states and the model of a test, with the parameters an operator can change. */
    private record Fixture(Tree tree, Alignment alignment, NodeStates nodeStates, GeneticPrior geneticPrior,
                           RealScalarParam<PositiveReal> kappa, RealScalarParam<PositiveReal> clockRate) {
    }

    private final Random random = new Random(11);

    private final Operator operator = new Operator() {
        @Override
        public void initAndValidate() {
        }

        @Override
        public double proposal() {
            return 0.0;
        }
    };

    /* Correctness */

    /**
     * Compares the prior to a sum over sites of the log root frequency and the log transition
     * probabilities, which come from the substitution model itself instead of the eigen
     * decomposition of the prior.
     */
    @Test
    public void testMatchesSumOverSites() {
        Fixture fixture = this.createFixture(this.parseTree(NEWICK), SEQUENCES);

        for (int i = 0; i < 20; i++) {
            int[][] sequences = this.drawCompatibleSequences(fixture);
            this.initialiseStates(fixture, sequences);

            assertEquals(this.computeLogPOverSites(fixture, sequences), fixture.geneticPrior().calculateLogP(), 1e-9);
        }
    }

    private double computeLogPOverSites(Fixture fixture, int[][] sequences) {
        Tree tree = fixture.tree();
        GeneticPrior geneticPrior = fixture.geneticPrior();
        double[] frequencies = geneticPrior.substitutionModel.getFrequencies();
        double rate = fixture.clockRate().get();
        int[] rootSequence = sequences[tree.getRoot().getNr()];

        double logP = 0.0;
        for (int state : rootSequence) {
            logP += Math.log(frequencies[state]);
        }

        double[] matrix = new double[NUM_STATES * NUM_STATES];
        for (Node node : tree.getNodesAsArray()) {
            if (node.isRoot()) {
                continue;
            }

            geneticPrior.substitutionModel.getTransitionProbabilities(node, node.getParent().getHeight(), node.getHeight(), rate, matrix);
            int[] parentSequence = sequences[node.getParent().getNr()];
            int[] nodeSequence = sequences[node.getNr()];
            for (int site = 0; site < nodeSequence.length; site++) {
                logP += Math.log(matrix[parentSequence[site] * NUM_STATES + nodeSequence[site]]);
            }
        }

        return logP;
    }

    /**
     * Sums the prior over every state of the inner nodes and every latent state of the tips at
     * a single site, which must give the tree likelihood with ambiguities.
     */
    @Test
    public void testMarginalMatchesTreeLikelihood() {
        Tree tree = this.parseTree("((A:0.3,B:0.3):0.4,(C:0.5,D:0.5):0.2);");
        List<Sequence> sequences = List.of(
                new Sequence("A", "A"),
                new Sequence("B", "R"),
                new Sequence("C", "N"),
                new Sequence("D", "C")
        );
        Fixture fixture = this.createFixture(tree, sequences);

        // enumerate every compatible assignment of states to the nodes

        TipCompatibility tipCompatibility = fixture.geneticPrior().getTipCompatibility();
        Node[] nodes = tree.getNodesAsArray();
        int numAssignments = (int) Math.pow(NUM_STATES, nodes.length);

        double maxLogP = Double.NEGATIVE_INFINITY;
        List<Double> logPs = new ArrayList<>();
        for (int assignment = 0; assignment < numAssignments; assignment++) {
            int[][] states = new int[nodes.length][1];
            boolean isCompatible = true;
            int remainder = assignment;
            for (Node node : nodes) {
                int state = remainder % NUM_STATES;
                remainder /= NUM_STATES;
                states[node.getNr()][0] = state;
                isCompatible &= (tipCompatibility.getCompatibleStates(node, 0) & (1 << state)) != 0;
            }
            if (!isCompatible) {
                continue;
            }

            this.initialiseStates(fixture, states);
            double logP = fixture.geneticPrior().calculateLogP();
            logPs.add(logP);
            maxLogP = Math.max(maxLogP, logP);
        }
        assertEquals(4 * 4 * 4 * 2 * 4, logPs.size());

        double sum = 0.0;
        for (double logP : logPs) {
            sum += Math.exp(logP - maxLogP);
        }
        double marginalLogP = maxLogP + Math.log(sum);

        TreeLikelihood treeLikelihood = new TreeLikelihood();
        treeLikelihood.initByName(
                "data", fixture.alignment(),
                "tree", tree,
                "siteModel", fixture.geneticPrior().siteModel,
                "branchRateModel", fixture.geneticPrior().branchRateModel,
                "useAmbiguities", true
        );

        assertEquals(treeLikelihood.calculateLogP(), marginalLogP, 1e-9);
    }

    /** A tip in a state its data is not compatible with makes the prior zero, which a restore undoes. */
    @Test
    public void testIncompatibleTipState() {
        Fixture fixture = this.createFixture(this.parseTree(NEWICK), SEQUENCES);
        int[][] sequences = this.drawCompatibleSequences(fixture);
        this.initialiseStates(fixture, sequences);

        NodeStates nodeStates = fixture.nodeStates();
        GeneticPrior geneticPrior = fixture.geneticPrior();
        double oldLogP = geneticPrior.calculateLogP();
        nodeStates.setEverythingDirty(false);

        // site 0 of tip A is an unambiguous A
        Node tipA = this.getTip(fixture.tree(), "A");
        int[] sequence = sequences[tipA.getNr()].clone();
        sequence[0] = 1;

        geneticPrior.store();
        this.setSequence(nodeStates, tipA, sequence);
        assertEquals(Double.NEGATIVE_INFINITY, geneticPrior.calculateLogP());

        nodeStates.restoreStateNode();
        geneticPrior.restore();
        nodeStates.setEverythingDirty(false);

        assertEquals(oldLogP, geneticPrior.getCurrentLogP());
        assertEquals(oldLogP, geneticPrior.calculateLogP(), 1e-12);
    }

    /* Caching */

    @Test
    public void testIncrementalMatchesFullComputation() {
        this.runChain(this.parseTree(NEWICK));
    }

    /**
     * Runs the same chain on an incremental store tree, for which the genetic prior only
     * visits the nodes flagged by the tree and the branches marked dirty by the node states.
     */
    @Test
    public void testIncrementalMatchesFullComputationOnIncrementalStoreTree() {
        TreeParser parsedTree = this.parseTree(NEWICK);
        List<Taxon> taxa = parsedTree.getTaxonset().asStringList().stream().map(Taxon::new).toList();
        IncrementalStoreTree tree = new IncrementalStoreTree();
        tree.initByName("taxonset", new TaxonSet(taxa));
        tree.assignFrom(parsedTree);
        this.runChain(tree);
    }

    /**
     * Drives the state through the store, accept and restore cycle of MCMC with random
     * proposals of every kind, accepting at random, and checks after every step that the
     * incrementally updated prior matches a computation from scratch.
     */
    private void runChain(Tree tree) {
        Fixture fixture = this.createFixture(tree, SEQUENCES);
        this.initialiseStates(fixture, this.drawCompatibleSequences(fixture));
        GeneticPrior geneticPrior = fixture.geneticPrior();

        State state = new State();
        state.initByName(
                "stateNode", tree,
                "stateNode", fixture.nodeStates(),
                "stateNode", fixture.kappa(),
                "stateNode", fixture.clockRate()
        );
        state.initialise();
        state.setPosterior(geneticPrior);
        state.setEverythingDirty(true);
        geneticPrior.calculateLogP();
        state.setEverythingDirty(false);

        List<Supplier<String>> proposals = List.of(
                () -> this.proposeInnerNodeChange(fixture),
                () -> this.proposeRootChange(fixture),
                () -> this.proposeTipChange(fixture),
                () -> this.proposeHeightChange(fixture),
                () -> this.proposeParameterChange(fixture)
        );

        for (long i = 0; i < 5000; i++) {
            state.store(i);
            String proposal = proposals.get(this.random.nextInt(proposals.size())).get();

            state.storeCalculationNodes();
            state.checkCalculationNodesDirtiness();

            double newLogP = geneticPrior.calculateLogP();
            assertEquals(this.computeLogPFromScratch(geneticPrior), newLogP, 1e-9, "Proposal of " + proposal);

            if (newLogP != Double.NEGATIVE_INFINITY && this.random.nextBoolean()) {
                state.acceptCalculationNodes();
            } else {
                state.restore();
                state.restoreCalculationNodes();
                assertEquals(this.computeLogPFromScratch(geneticPrior), geneticPrior.getCurrentLogP(), 1e-9, "Rejection of " + proposal);
            }

            state.setEverythingDirty(false);
        }
    }

    private String proposeInnerNodeChange(Fixture fixture) {
        Node[] nodes = fixture.tree().getNodesAsArray();
        Node node = nodes[fixture.tree().getLeafNodeCount() + this.random.nextInt(fixture.tree().getInternalNodeCount())];
        if (node.isRoot()) {
            return this.proposeRootChange(fixture);
        }

        NodeStates nodeStates = fixture.nodeStates();
        int[] sequence = this.getSequence(nodeStates, node);
        for (int site = 0; site < sequence.length; site++) {
            if (this.random.nextDouble() < 0.3) {
                sequence[site] = this.random.nextInt(NUM_STATES);
            }
        }
        this.setSequence(nodeStates, node, sequence);
        return "inner node change";
    }

    private String proposeRootChange(Fixture fixture) {
        int numSites = fixture.nodeStates().getNumSites();
        int site = this.random.nextInt(numSites);
        fixture.nodeStates().setStates(fixture.tree().getRoot(), new int[]{site}, new int[]{this.random.nextInt(NUM_STATES)}, this.operator);
        return "root change";
    }

    /** Changes a latent state of a random tip among its compatible states, and rarely to an incompatible state. */
    private String proposeTipChange(Fixture fixture) {
        Node tip = fixture.tree().getNode(this.random.nextInt(fixture.tree().getLeafNodeCount()));
        NodeStates nodeStates = fixture.nodeStates();
        TipCompatibility tipCompatibility = fixture.geneticPrior().getTipCompatibility();

        int[] sequence = this.getSequence(nodeStates, tip);
        int site = this.random.nextInt(sequence.length);
        sequence[site] = this.random.nextDouble() < 0.1
                ? this.random.nextInt(NUM_STATES)
                : this.drawCompatibleState(tipCompatibility.getCompatibleStates(tip, site));

        this.setSequence(nodeStates, tip, sequence);
        return "tip change";
    }

    /** Moves a random inner node uniformly between its oldest child and its parent, or above its oldest child for the root. */
    private String proposeHeightChange(Fixture fixture) {
        Tree tree = fixture.tree();
        Node node = tree.getNode(tree.getLeafNodeCount() + this.random.nextInt(tree.getInternalNodeCount()));

        double lower = Math.max(node.getLeft().getHeight(), node.getRight().getHeight());
        double upper = node.isRoot() ? lower + 1.0 : node.getParent().getHeight();
        node.setHeight(lower + (upper - lower) * (0.01 + 0.98 * this.random.nextDouble()));
        return "height change";
    }

    private String proposeParameterChange(Fixture fixture) {
        RealScalarParam<PositiveReal> parameter = this.random.nextBoolean() ? fixture.kappa() : fixture.clockRate();
        parameter.set(parameter.get() * Math.exp(0.5 * (this.random.nextDouble() - 0.5)));
        return "parameter change";
    }

    /**
     * Computes the genetic prior of the current tree and states from scratch with a new
     * genetic prior on the same inputs, whose first calculation recomputes every branch.
     */
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

        return new Fixture(tree, alignment, nodeStates, geneticPrior, kappa, clockRate);
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

    /** Sets the states of every node to the given sequences. */
    private void initialiseStates(Fixture fixture, int[][] sequences) {
        int[] rootSequence = sequences[fixture.tree().getRoot().getNr()];
        List<BranchChanges> changes = new ArrayList<>();
        for (int nodeNr = 0; nodeNr < sequences.length; nodeNr++) {
            Node node = fixture.tree().getNode(nodeNr);
            changes.add(node.isRoot()
                    ? BranchChanges.empty(nodeNr)
                    : BranchChanges.fromSequences(nodeNr, sequences[node.getParent().getNr()], sequences[nodeNr]));
        }
        fixture.nodeStates().initialiseStates(rootSequence, changes);
        assertTrue(fixture.nodeStates().isEveryBranchDirty());
    }

    /** Sets the states of the given node to the given sequence, keeping those of every other node. */
    private void setSequence(NodeStates nodeStates, Node node, int[] sequence) {
        int[] current = nodeStates.materializeState(node);
        int[] sites = new int[sequence.length];
        int[] states = new int[sequence.length];
        int size = 0;
        for (int site = 0; site < sequence.length; site++) {
            if (sequence[site] != current[site]) {
                sites[size] = site;
                states[size] = sequence[site];
                size++;
            }
        }
        nodeStates.setStates(node, Arrays.copyOf(sites, size), Arrays.copyOf(states, size), this.operator);
    }

    private int[] getSequence(NodeStates nodeStates, Node node) {
        return nodeStates.materializeState(node);
    }

    private Node getTip(Tree tree, String taxon) {
        for (Node node : tree.getExternalNodes()) {
            if (node.getID().equals(taxon)) {
                return node;
            }
        }
        throw new IllegalArgumentException("Unknown taxon " + taxon);
    }

}
