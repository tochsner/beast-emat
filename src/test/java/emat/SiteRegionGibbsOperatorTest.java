package emat;

import beast.base.evolution.alignment.Alignment;
import beast.base.evolution.alignment.Sequence;
import beast.base.evolution.substitutionmodel.SubstitutionModel;
import beast.base.evolution.tree.Node;
import beast.base.evolution.tree.TreeParser;
import beast.base.inference.State;
import beast.base.spec.domain.PositiveReal;
import beast.base.spec.evolution.branchratemodel.StrictClockModel;
import beast.base.spec.evolution.sitemodel.SiteModel;
import beast.base.spec.evolution.substitutionmodel.Frequencies;
import beast.base.spec.evolution.substitutionmodel.HKY;
import beast.base.spec.evolution.substitutionmodel.JukesCantor;
import beast.base.spec.inference.parameter.RealScalarParam;
import beast.base.spec.inference.parameter.SimplexParam;
import beast.base.util.Randomizer;
import emat.helper.MutationPaths;
import emat.initalisation.ParsimonyMutationsInitialiser;
import emat.operators.SiteRegionGibbsOperator;
import emat.operators.SiteRegionGibbsOperator.Selection;
import emat.prior.GeneticPrior;
import emat.state.Mutations;
import emat.stochasticmapping.StochasticMapping;
import org.junit.jupiter.api.Test;

import java.util.Arrays;
import java.util.List;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertTrue;

public class SiteRegionGibbsOperatorTest {

    // the maximum numbers of nodes to test: the star, a region of two nodes, and the whole tree
    private static final int[] MAX_NODES = {1, 2, 100};

    @Test
    public void testNodeStatesMatchExactPosterior() {
        for (int maxNodes : MAX_NODES) {
            this.checkNodeStatesMatchExactPosterior(maxNodes);
        }
    }

    @Test
    public void testAnchoredSelectionMatchesExactPosterior() {
        for (int maxNodes : MAX_NODES) {
            this.checkTwoSitesMatchExactPosterior(maxNodes, Selection.MUTATION_ANCHORED, false);
        }
    }

    @Test
    public void testParsimonySitesMatchExactPosterior() {
        for (int maxNodes : MAX_NODES) {
            this.checkTwoSitesMatchExactPosterior(maxNodes, Selection.UNIFORM, true);
            this.checkTwoSitesMatchExactPosterior(maxNodes, Selection.MUTATION_ANCHORED, true);
        }
    }

    @Test
    public void testUniformSelectionMatchesEnumeratedPosterior() {
        for (int maxNodes : MAX_NODES) {
            this.checkMatchesEnumeratedPosterior(maxNodes, Selection.UNIFORM);
        }
    }

    @Test
    public void testAnchoredSelectionMatchesEnumeratedPosterior() {
        for (int maxNodes : MAX_NODES) {
            this.checkMatchesEnumeratedPosterior(maxNodes, Selection.MUTATION_ANCHORED);
        }
    }

    @Test
    public void testTipStatesStayCompatible() {
        for (int maxNodes : MAX_NODES) {
            this.checkTipStatesStayCompatible(maxNodes, Selection.UNIFORM);
            this.checkTipStatesStayCompatible(maxNodes, Selection.MUTATION_ANCHORED);
        }
    }

    /**
     * On the tree ((A:0.3,B:0.5):0.4,C:0.9) with tips A, C and G under HKY, runs the
     * operator with uniform selection, which alternates between the regions around the root
     * and around the inner node, and checks the sampled joint states of both nodes against
     * their exact posterior P(r, x) ∝ π_r P_rx(0.4) P_xA(0.3) P_xC(0.5) P_rG(0.9).
     */
    private void checkNodeStatesMatchExactPosterior(int maxNodes) {
        Randomizer.setSeed(1);

        Frequencies frequencies = new Frequencies();
        frequencies.initByName("frequencies", new SimplexParam(new double[]{0.1, 0.2, 0.3, 0.4}));
        HKY hky = new HKY();
        hky.initByName("kappa", new RealScalarParam<>(4.0, PositiveReal.INSTANCE), "frequencies", frequencies);

        TestSetup setup = new TestSetup(
                List.of(new Sequence("A", "A"), new Sequence("B", "C"), new Sequence("C", "G")),
                "((A:0.3,B:0.5):0.4,C:0.9);",
                hky, maxNodes, Selection.UNIFORM
        );

        Node root = setup.tree.getRoot();
        Node inner = root.getLeft().isLeaf() ? root.getRight() : root.getLeft();

        // run the operator and collect the joint states of the root and the inner node

        int numBurnin = 1000;
        int numSamples = 400000;

        double[] stateCounts = new double[16];
        for (int i = 0; i < numBurnin + numSamples; i++) {
            assertEquals(Double.POSITIVE_INFINITY, setup.operator.proposal());

            if (i >= numBurnin) {
                int rootState = MutationPaths.getState(setup.mutations, root, 0);
                int innerState = MutationPaths.getState(setup.mutations, inner, 0);
                stateCounts[rootState * 4 + innerState]++;
            }
        }

        // compare with the exact posterior

        double[] stationaryFrequencies = hky.getFrequencies();
        double[] rootToInner = this.getTransitionProbabilities(hky, 0.4);
        double[] innerToA = this.getTransitionProbabilities(hky, 0.3);
        double[] innerToB = this.getTransitionProbabilities(hky, 0.5);
        double[] rootToC = this.getTransitionProbabilities(hky, 0.9);

        double[] probabilities = new double[16];
        double normalisation = 0.0;
        for (int rootState = 0; rootState < 4; rootState++) {
            for (int innerState = 0; innerState < 4; innerState++) {
                double weight = stationaryFrequencies[rootState]
                        * rootToInner[rootState * 4 + innerState]
                        * innerToA[innerState * 4]
                        * innerToB[innerState * 4 + 1]
                        * rootToC[rootState * 4 + 2];
                probabilities[rootState * 4 + innerState] = weight;
                normalisation += weight;
            }
        }

        for (int i = 0; i < 16; i++) {
            assertEquals(probabilities[i] / normalisation, stateCounts[i] / numSamples, 0.005,
                    "Max nodes " + maxNodes + ", root state " + i / 4 + ", inner state " + i % 4 + ".");
        }
    }

    /**
     * On the tree ((A:0.3,B:0.5):0.4,C:0.9) under HKY with a variable site (tips A, C, G) and
     * a constant site (all A), runs the operator with the given selection strategy and site
     * weighting under Metropolis-Hastings acceptance, and checks the sampled joint states of the root and the
     * inner node at both sites against their exact posterior
     * P(r, x) ∝ π_r P_rx(0.4) P_x,a(0.3) P_x,b(0.5) P_r,c(0.9).
     */
    private void checkTwoSitesMatchExactPosterior(int maxNodes, Selection selection, boolean parsimonySites) {
        Randomizer.setSeed(3);

        Frequencies frequencies = new Frequencies();
        frequencies.initByName("frequencies", new SimplexParam(new double[]{0.1, 0.2, 0.3, 0.4}));
        HKY hky = new HKY();
        hky.initByName("kappa", new RealScalarParam<>(4.0, PositiveReal.INSTANCE), "frequencies", frequencies);

        TestSetup setup = new TestSetup(
                List.of(new Sequence("A", "AA"), new Sequence("B", "CA"), new Sequence("C", "GA")),
                "((A:0.3,B:0.5):0.4,C:0.9);",
                hky, maxNodes, selection, parsimonySites
        );

        Node root = setup.tree.getRoot();
        Node inner = root.getLeft().isLeaf() ? root.getRight() : root.getLeft();

        // run the operator with Metropolis-Hastings acceptance and collect the joint states of the root and the inner node

        int numBurnin = 1000;
        // with parsimony weights, the constant site is picked rarely and needs a longer chain
        int numSamples = parsimonySites ? 4000000 : 800000;

        double[][] stateCounts = new double[2][16];
        int numAccepted = 0;

        for (int i = 0; i < numBurnin + numSamples; i++) {
            if (setup.runMetropolisHastingsStep(i)) {
                numAccepted++;
            }

            if (i >= numBurnin) {
                for (int site = 0; site < 2; site++) {
                    int rootState = MutationPaths.getState(setup.mutations, root, site);
                    int innerState = MutationPaths.getState(setup.mutations, inner, site);
                    stateCounts[site][rootState * 4 + innerState]++;
                }
            }
        }

        if (selection == Selection.MUTATION_ANCHORED) {
            assertTrue(numAccepted < numBurnin + numSamples, "Anchored selection should reject some moves.");
        }

        // compare with the exact posterior

        int[][] tipStates = {{0, 1, 2}, {0, 0, 0}};

        double[] stationaryFrequencies = hky.getFrequencies();
        double[] rootToInner = this.getTransitionProbabilities(hky, 0.4);
        double[] innerToA = this.getTransitionProbabilities(hky, 0.3);
        double[] innerToB = this.getTransitionProbabilities(hky, 0.5);
        double[] rootToC = this.getTransitionProbabilities(hky, 0.9);

        for (int site = 0; site < 2; site++) {
            int[] tips = tipStates[site];

            double[] probabilities = new double[16];
            double normalisation = 0.0;
            for (int rootState = 0; rootState < 4; rootState++) {
                for (int innerState = 0; innerState < 4; innerState++) {
                    double weight = stationaryFrequencies[rootState]
                            * rootToInner[rootState * 4 + innerState]
                            * innerToA[innerState * 4 + tips[0]]
                            * innerToB[innerState * 4 + tips[1]]
                            * rootToC[rootState * 4 + tips[2]];
                    probabilities[rootState * 4 + innerState] = weight;
                    normalisation += weight;
                }
            }

            for (int i = 0; i < 16; i++) {
                assertEquals(probabilities[i] / normalisation, stateCounts[site][i] / numSamples, 0.005,
                        selection + ", parsimony sites " + parsimonySites + ", max nodes " + maxNodes + ", site " + site + ", root state " + i / 4 + ", inner state " + i % 4 + ".");
            }
        }
    }

    /**
     * On a larger tree with ambiguous and missing characters, checks after every move that
     * the state of every tip stays compatible with its data, and that the states of inner
     * nodes and of tips with uncertain data do change. Invalid mutation lists are also
     * caught by the sanity checks in Mutations.
     */
    private void checkTipStatesStayCompatible(int maxNodes, Selection selection) {
        Randomizer.setSeed(2);

        TestSetup setup = new TestSetup(
                List.of(
                        new Sequence("A", "ACGTR"),
                        new Sequence("B", "ACGAN"),
                        new Sequence("C", "TCRAA"),
                        new Sequence("D", "TCGT-"),
                        new Sequence("E", "GCGTA")
                ),
                "(((A:0.3,B:0.3):0.4,(C:0.5,D:0.5):0.2):0.3,E:1.0);",
                new JukesCantor(), maxNodes, selection
        );

        int numTips = setup.tree.getLeafNodeCount();

        int numChangedInnerStates = 0;
        int numChangedTipStates = 0;

        for (int i = 0; i < 5000; i++) {
            int[] oldStates = setup.getNodeStates();
            setup.runMetropolisHastingsStep(i);
            int[] newStates = setup.getNodeStates();

            for (int tipNr = 0; tipNr < numTips; tipNr++) {
                Node tip = setup.tree.getNode(tipNr);
                int taxonNr = setup.alignment.getTaxonIndex(tip.getID());

                for (int site = 0; site < 5; site++) {
                    int tipState = newStates[tipNr * 5 + site];
                    int code = setup.alignment.getPattern(taxonNr, setup.alignment.getPatternIndex(site));
                    int[] compatibleStates = setup.alignment.getDataType().getStatesForCode(code);

                    boolean isCompatible = false;
                    for (int state : compatibleStates) {
                        isCompatible |= state == tipState;
                    }
                    assertTrue(isCompatible, "Tip " + tip.getID() + " is inconsistent at site " + site + ".");

                    if (tipState != oldStates[tipNr * 5 + site]) {
                        assertTrue(compatibleStates.length > 1, "Tip " + tip.getID() + " changed its certain state at site " + site + ".");
                        numChangedTipStates++;
                    }
                }
            }

            if (!Arrays.equals(oldStates, numTips * 5, oldStates.length, newStates, numTips * 5, newStates.length)) {
                numChangedInnerStates++;
            }
        }

        assertTrue(numChangedInnerStates > 0, "The operator never changed the state of an inner node.");
        assertTrue(numChangedTipStates > 0, "The operator never changed the state of a tip with uncertain data.");
    }

    /**
     * On a five-tip tree with nested clades and an ambiguous tip (R, i.e. A or G) under HKY,
     * runs the operator with Metropolis-Hastings acceptance and checks the sampled states of
     * every node against their exact posterior marginals, obtained by enumerating all
     * assignments of states to the nodes that are compatible with the tips.
     */
    private void checkMatchesEnumeratedPosterior(int maxNodes, Selection selection) {
        Randomizer.setSeed(4);

        Frequencies frequencies = new Frequencies();
        frequencies.initByName("frequencies", new SimplexParam(new double[]{0.1, 0.2, 0.3, 0.4}));
        HKY hky = new HKY();
        hky.initByName("kappa", new RealScalarParam<>(4.0, PositiveReal.INSTANCE), "frequencies", frequencies);

        TestSetup setup = new TestSetup(
                List.of(
                        new Sequence("A", "A"),
                        new Sequence("B", "C"),
                        new Sequence("C", "C"),
                        new Sequence("D", "G"),
                        new Sequence("E", "R")
                ),
                "(((A:0.3,B:0.3):0.4,(C:0.5,D:0.5):0.2):0.3,E:1.0);",
                hky, maxNodes, selection
        );

        int numNodes = setup.tree.getNodeCount();

        // run the operator and collect the states of the nodes

        int numBurnin = 1000;
        int numSamples = 2000000;

        double[][] stateCounts = new double[numNodes][4];
        for (int i = 0; i < numBurnin + numSamples; i++) {
            setup.runMetropolisHastingsStep(i);

            if (i >= numBurnin) {
                int[] nodeStates = setup.getNodeStates();
                for (int nodeNr = 0; nodeNr < numNodes; nodeNr++) {
                    stateCounts[nodeNr][nodeStates[nodeNr]]++;
                }
            }
        }

        // enumerate the states of all nodes, encoded as the base-4 digits of the assignment

        double[] stationaryFrequencies = hky.getFrequencies();

        double[][] transitionProbabilities = new double[numNodes][];
        boolean[][] isCompatible = new boolean[numNodes][4];
        for (Node node : setup.tree.getNodesAsArray()) {
            if (!node.isRoot()) {
                transitionProbabilities[node.getNr()] = this.getTransitionProbabilities(hky, node.getLength());
            }

            if (node.isLeaf()) {
                int code = setup.alignment.getPattern(setup.alignment.getTaxonIndex(node.getID()), setup.alignment.getPatternIndex(0));
                for (int state : setup.alignment.getDataType().getStatesForCode(code)) {
                    isCompatible[node.getNr()][state] = true;
                }
            } else {
                Arrays.fill(isCompatible[node.getNr()], true);
            }
        }

        int[] nodeStates = new int[numNodes];
        double[][] probabilities = new double[numNodes][4];
        double normalisation = 0.0;

        for (int assignment = 0; assignment < 1 << (2 * numNodes); assignment++) {
            double weight = 1.0;
            for (int nodeNr = 0; nodeNr < numNodes; nodeNr++) {
                nodeStates[nodeNr] = (assignment >> (2 * nodeNr)) & 3;
                if (!isCompatible[nodeNr][nodeStates[nodeNr]]) {
                    weight = 0.0;
                }
            }

            for (Node node : setup.tree.getNodesAsArray()) {
                weight *= node.isRoot()
                        ? stationaryFrequencies[nodeStates[node.getNr()]]
                        : transitionProbabilities[node.getNr()][nodeStates[node.getParent().getNr()] * 4 + nodeStates[node.getNr()]];
            }

            normalisation += weight;
            for (int nodeNr = 0; nodeNr < numNodes; nodeNr++) {
                probabilities[nodeNr][nodeStates[nodeNr]] += weight;
            }
        }

        for (int nodeNr = 0; nodeNr < numNodes; nodeNr++) {
            for (int state = 0; state < 4; state++) {
                assertEquals(probabilities[nodeNr][state] / normalisation, stateCounts[nodeNr][state] / numSamples, 0.006,
                        "Max nodes " + maxNodes + ", node " + nodeNr + ", state " + state + ".");
            }
        }
    }

    /* Helpers */

    private double[] getTransitionProbabilities(SubstitutionModel substitutionModel, double time) {
        return StochasticMapping.computeTransitionProbabilities(substitutionModel.getEigenDecomposition(null), time);
    }

    /* Test Setup */

    /**
     * Builds the tree, alignment, mutations (initialised by parsimony), a genetic prior with
     * the given substitution model and clock rate 1, and the operator with the given maximum
     * number of nodes and selection strategy.
     */
    private static class TestSetup {

        final Alignment alignment;
        final TreeParser tree;
        final Mutations mutations;
        final GeneticPrior geneticPrior;
        final SiteRegionGibbsOperator operator;

        // the state and the current genetic prior of the Metropolis-Hastings chain, created with its first step
        State state;
        double logP;

        TestSetup(List<Sequence> sequences, String newick, SubstitutionModel substitutionModel,
                  int maxNodes, Selection selection) {
            this(sequences, newick, substitutionModel, maxNodes, selection, false);
        }

        TestSetup(List<Sequence> sequences, String newick, SubstitutionModel substitutionModel,
                  int maxNodes, Selection selection, boolean parsimonySites) {
            this.alignment = new Alignment();
            this.alignment.initByName("sequence", sequences, "dataType", "nucleotide");

            this.tree = new TreeParser();
            this.tree.initByName("newick", newick, "IsLabelledNewick", true, "adjustTipHeights", false);

            this.mutations = new Mutations();
            this.mutations.initByName("tree", this.tree, "alignment", this.alignment);

            ParsimonyMutationsInitialiser initialiser = new ParsimonyMutationsInitialiser();
            initialiser.initByName("mutations", this.mutations);
            initialiser.initStateNodes();

            SiteModel siteModel = new SiteModel();
            siteModel.initByName("substModel", substitutionModel);

            StrictClockModel clockModel = new StrictClockModel();
            clockModel.initByName("clock.rate", new RealScalarParam<>(1.0, PositiveReal.INSTANCE));

            this.geneticPrior = new GeneticPrior();
            this.geneticPrior.initByName(
                    "data", this.alignment,
                    "tree", this.tree,
                    "siteModel", siteModel,
                    "branchRateModel", clockModel,
                    "mutations", this.mutations
            );

            this.operator = new SiteRegionGibbsOperator();
            this.operator.initByName("weight", 1.0, "mutations", this.mutations, "geneticPrior", this.geneticPrior,
                    "maxNodes", maxNodes, "selection", selection, "parsimonySites", parsimonySites);
        }

        /**
         * Runs one step of the operator with Metropolis-Hastings acceptance under the genetic
         * prior and returns whether the proposal was accepted.
         */
        boolean runMetropolisHastingsStep(int stepNr) {
            if (this.state == null) {
                this.state = new State();
                this.state.initByName("stateNode", this.tree, "stateNode", this.mutations);
                this.state.initialise();
                this.state.setPosterior(this.geneticPrior);
                this.state.setEverythingDirty(true);
                this.logP = this.geneticPrior.calculateLogP();
                this.state.setEverythingDirty(false);
            }

            this.state.store(stepNr);
            double logHastingsRatio = this.operator.proposal();

            this.state.storeCalculationNodes();
            this.state.checkCalculationNodesDirtiness();
            double newLogP = this.geneticPrior.calculateLogP();

            boolean isAccepted = Math.log(Randomizer.nextDouble()) < newLogP - this.logP + logHastingsRatio;
            if (isAccepted) {
                this.logP = newLogP;
                this.state.acceptCalculationNodes();
            } else {
                this.state.restore();
                this.state.restoreCalculationNodes();
            }
            this.state.setEverythingDirty(false);

            return isAccepted;
        }

        /** Returns the states of all nodes at all sites, flattened by node and then site. */
        int[] getNodeStates() {
            int numSites = this.mutations.getReferenceSequence().length;
            int numNodes = this.tree.getNodeCount();

            int[] states = new int[numNodes * numSites];
            for (int nodeNr = 0; nodeNr < numNodes; nodeNr++) {
                for (int site = 0; site < numSites; site++) {
                    states[nodeNr * numSites + site] = MutationPaths.getState(this.mutations, this.tree.getNode(nodeNr), site);
                }
            }
            return states;
        }

    }

}
