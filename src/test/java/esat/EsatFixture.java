package esat;

import beast.base.evolution.alignment.Alignment;
import beast.base.evolution.alignment.Sequence;
import beast.base.evolution.alignment.Taxon;
import beast.base.evolution.alignment.TaxonSet;
import beast.base.evolution.tree.Node;
import beast.base.evolution.tree.Tree;
import beast.base.evolution.tree.TreeParser;
import beast.base.inference.Operator;
import beast.base.spec.domain.PositiveReal;
import beast.base.spec.evolution.branchratemodel.StrictClockModel;
import beast.base.spec.evolution.sitemodel.SiteModel;
import beast.base.spec.evolution.substitutionmodel.Frequencies;
import beast.base.spec.evolution.substitutionmodel.HKY;
import beast.base.spec.inference.parameter.RealScalarParam;
import beast.base.spec.inference.parameter.SimplexParam;
import emat.tree.IncrementalStoreTree;
import esat.helper.TipCompatibility;
import esat.prior.GeneticPrior;
import esat.state.NodeStates;
import esat.state.StateChanges;

import java.util.ArrayList;
import java.util.List;
import java.util.Random;

/**
 * A tree with node states and a genetic prior under HKY with kappa 3, the frequencies
 * {0.1, 0.2, 0.3, 0.4} and a strict clock with rate 0.5, for the tests of the samplers.
 */
public record EsatFixture(Tree tree, Alignment alignment, NodeStates nodeStates, GeneticPrior geneticPrior, HKY hky,
                          RealScalarParam<PositiveReal> clockRate) {

    public static final int NUM_STATES = 4;

    /** An operator that does nothing, for edits of the node states outside of an MCMC run. */
    public static final Operator NO_OPERATOR = new Operator() {
        @Override
        public void initAndValidate() {
        }

        @Override
        public double proposal() {
            return 0.0;
        }
    };

    public static EsatFixture create(String newick, List<Sequence> sequences) {
        return create(parseTree(newick), sequences);
    }

    public static TreeParser parseTree(String newick) {
        TreeParser tree = new TreeParser();
        tree.initByName(
                "newick", newick,
                "IsLabelledNewick", true,
                "adjustTipHeights", false
        );
        return tree;
    }

    /** Returns the tree of the given newick string as a tree that only stores its changed nodes. */
    public static IncrementalStoreTree parseIncrementalTree(String newick) {
        TreeParser parsedTree = parseTree(newick);
        List<Taxon> taxa = parsedTree.getTaxonset().asStringList().stream().map(Taxon::new).toList();
        IncrementalStoreTree tree = new IncrementalStoreTree();
        tree.initByName("taxonset", new TaxonSet(taxa));
        tree.assignFrom(parsedTree);
        return tree;
    }

    public static EsatFixture create(Tree tree, List<Sequence> sequences) {
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

        return new EsatFixture(tree, alignment, nodeStates, geneticPrior, hky, clockRate);
    }

    /** Draws a random sequence for every node, which for the tips is compatible with their data. */
    public int[][] drawCompatibleSequences(Random random) {
        TipCompatibility tipCompatibility = this.geneticPrior.getTipCompatibility();
        int numSites = this.nodeStates.getNumSites();

        int[][] sequences = new int[this.tree.getNodeCount()][numSites];
        for (Node node : this.tree.getNodesAsArray()) {
            for (int site = 0; site < numSites; site++) {
                int compatibleStates = tipCompatibility.getCompatibleStates(node, site);
                int state;
                do {
                    state = random.nextInt(NUM_STATES);
                } while ((compatibleStates & (1 << state)) == 0);
                sequences[node.getNr()][site] = state;
            }
        }
        return sequences;
    }

    /** Sets the states of every node to the given sequences, with the sequence of the root as the reference. */
    public void initialiseStates(int[][] sequences) {
        int[] rootSequence = sequences[this.tree.getRoot().getNr()];
        List<StateChanges> changes = new ArrayList<>();
        for (int nodeNr = 0; nodeNr < sequences.length; nodeNr++) {
            changes.add(StateChanges.fromSequence(nodeNr, sequences[nodeNr], rootSequence));
        }
        this.nodeStates.initialiseStates(rootSequence, changes);
    }

    /** Returns the full sequence of every node. */
    public int[][] getSequences() {
        int[][] sequences = new int[this.tree.getNodeCount()][this.nodeStates.getNumSites()];
        for (Node node : this.tree.getNodesAsArray()) {
            for (int site = 0; site < sequences[node.getNr()].length; site++) {
                sequences[node.getNr()][site] = this.nodeStates.getState(node, site);
            }
        }
        return sequences;
    }

    /** Returns whether the states of every tip are compatible with its data. */
    public boolean areTipsCompatible() {
        TipCompatibility tipCompatibility = this.geneticPrior.getTipCompatibility();
        for (Node tip : this.tree.getExternalNodes()) {
            if (!tipCompatibility.isCompatible(tip, this.nodeStates.getChanges(tip), this.nodeStates.getReferenceSequence())) {
                return false;
            }
        }
        return true;
    }

    /**
     * Computes the genetic prior of the current tree and states from scratch with a new
     * genetic prior on the same inputs, whose first calculation recomputes every branch.
     */
    public double computeLogPFromScratch() {
        GeneticPrior freshGeneticPrior = new GeneticPrior();
        freshGeneticPrior.initByName(
                "data", this.geneticPrior.dataInput.get(),
                "tree", this.geneticPrior.treeInput.get(),
                "siteModel", this.geneticPrior.siteModelInput.get(),
                "branchRateModel", this.geneticPrior.branchRateModelInput.get(),
                "nodeStates", this.geneticPrior.nodeStatesInput.get()
        );
        return freshGeneticPrior.calculateLogP();
    }

}
