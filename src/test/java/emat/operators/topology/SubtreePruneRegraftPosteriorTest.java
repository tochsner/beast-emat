package emat.operators.topology;

import beast.base.evolution.alignment.Alignment;
import beast.base.evolution.alignment.Sequence;
import beast.base.evolution.datatype.Nucleotide;
import beast.base.evolution.substitutionmodel.SubstitutionModel;
import beast.base.evolution.tree.Node;
import beast.base.evolution.tree.TreeParser;
import beast.base.inference.Operator;
import beast.base.inference.OperatorSchedule;
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
import emat.operators.mutations.BranchReformOperator;
import emat.prior.GeneticPrior;
import emat.state.Mutations;
import emat.stochasticmapping.StochasticMapping;
import org.junit.jupiter.api.Test;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.TreeSet;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertTrue;

public class SubtreePruneRegraftPosteriorTest {

    private static final int NUM_BURNIN = 20000;
    private static final int NUM_SAMPLES = 1000000;
    private static final int NUM_GRID_POINTS = 400;

    private static final double ROOT_HEIGHT = 1.0;
    private static final String[] TAXA = {"A", "B", "C", "D"};
    // B and D have missing data, whose states the moves resample when they prune these tips
    private static final String[] SEQUENCES = {"ACGTC", "ACGAN", "TCGAC", "TNNAT"};
    // B and D also have partially ambiguous data, whose states the moves resample among the compatible ones
    private static final String[] AMBIGUOUS_SEQUENCES = {"AAGTC", "WWKAN", "TTGAC", "TNNAY"};
    private static final Nucleotide NUCLEOTIDE = new Nucleotide();

    // the transition probabilities of the posterior computation per number of half grid steps, stored row by row
    private double[][] halfStepTransitionProbabilities;

    /**
     * Runs mdSPR, which resample only the branch above the
     * pruned subtree and the missing states of a pruned tip, together with branch reform,
     * which also changes the root sequence. The rooted topologies then follow the
     * Felsenstein likelihood with uniform root frequencies, integrated over the heights of
     * the two internal nodes below the root.
     */
    @Test
    public void testSubtreeTopologiesMatchPosterior() {
        Randomizer.setSeed(11);
        Setup setup = this.createSetup();

        MutationDirectedSprOperator mdSprOperator = new MutationDirectedSprOperator();
        mdSprOperator.initByName("weight", 1.0, "tree", setup.tree, "mutations", setup.mutations, "geneticPrior", setup.geneticPrior);

        BranchReformOperator branchReformOperator = new BranchReformOperator();
        branchReformOperator.initByName("weight", 1.0, "mutations", setup.mutations, "geneticPrior", setup.geneticPrior);

        Map<String, Double> expectedFrequencies = this.computeTopologyPosterior(setup, null);
        Map<String, Integer> topologyCounts = this.runChain(
                setup, List.of(mdSprOperator, branchReformOperator)
        );
        this.assertFrequencies(expectedFrequencies, topologyCounts);
    }

    /**
     * Runs mdSPR, which resample the states of a pruned
     * tip among the compatible ones where its data is partially ambiguous, together with
     * branch reform. The rooted topologies follow the Felsenstein likelihood whose tip
     * partials are one at every compatible state.
     */
    @Test
    public void testAmbiguousTipsMatchPosterior() {
        Randomizer.setSeed(13);
        Setup setup = this.createSetup(new JukesCantor(), 1.0, AMBIGUOUS_SEQUENCES);

        MutationDirectedSprOperator mdSprOperator = new MutationDirectedSprOperator();
        mdSprOperator.initByName("weight", 1.0, "tree", setup.tree, "mutations", setup.mutations, "geneticPrior", setup.geneticPrior,
                "windowProbability", 0.3);

        BranchReformOperator branchReformOperator = new BranchReformOperator();
        branchReformOperator.initByName("weight", 1.0, "mutations", setup.mutations, "geneticPrior", setup.geneticPrior);

        Map<String, Double> expectedFrequencies = this.computeTopologyPosterior(setup, null);
        Map<String, Integer> topologyCounts = this.runChain(
                setup, List.of(mdSprOperator, branchReformOperator)
        );
        this.assertFrequencies(expectedFrequencies, topologyCounts);
    }

    /**
     * Runs mdSPR under HKY with unequal frequencies and a clock rate other than one, where the
     * Jukes-Cantor proposals differ from the model. The move does not change the root
     * sequence, so the posterior is conditioned on it.
     */
    @Test
    public void testHkyTopologiesMatchPosterior() {
        Randomizer.setSeed(23);

        Frequencies frequencies = new Frequencies();
        frequencies.initByName("frequencies", new SimplexParam(new double[]{0.1, 0.2, 0.3, 0.4}));
        HKY hky = new HKY();
        hky.initByName("kappa", new RealScalarParam<>(4.0, PositiveReal.INSTANCE), "frequencies", frequencies);
        Setup setup = this.createSetup(hky, 0.7);

        MutationDirectedSprOperator mdSprOperator = new MutationDirectedSprOperator();
        mdSprOperator.initByName("weight", 1.0, "tree", setup.tree, "mutations", setup.mutations, "geneticPrior", setup.geneticPrior,
                "windowProbability", 0.5, "fullExplorationProbability", 0.1);

        int[] rootSequence = new int[SEQUENCES[0].length()];
        for (int site = 0; site < rootSequence.length; site++) {
            rootSequence[site] = MutationPaths.getState(setup.mutations, setup.tree.getRoot(), site);
        }

        Map<String, Double> expectedFrequencies = this.computeTopologyPosterior(setup, rootSequence);
        Map<String, Integer> topologyCounts = this.runChain(setup, List.of(mdSprOperator));
        this.assertFrequencies(expectedFrequencies, topologyCounts);
    }

    /**
     * Runs mdSPR with local explorations only, whose thresholds are drawn from the geometric
     * window, together with branch reform. The rooted topologies follow the same posterior
     * as with the other subtree moves.
     */
    @Test
    public void testWindowedMdSprTopologiesMatchPosterior() {
        Randomizer.setSeed(13);
        Setup setup = this.createSetup();

        MutationDirectedSprOperator mdSprOperator = new MutationDirectedSprOperator();
        mdSprOperator.initByName("weight", 1.0, "tree", setup.tree, "mutations", setup.mutations, "geneticPrior", setup.geneticPrior,
                "fullExplorationProbability", 0.0, "windowProbability", 0.5);

        BranchReformOperator branchReformOperator = new BranchReformOperator();
        branchReformOperator.initByName("weight", 1.0, "mutations", setup.mutations, "geneticPrior", setup.geneticPrior);

        Map<String, Double> expectedFrequencies = this.computeTopologyPosterior(setup, null);
        Map<String, Integer> topologyCounts = this.runChain(setup, List.of(mdSprOperator, branchReformOperator));
        this.assertFrequencies(expectedFrequencies, topologyCounts);
    }

    /**
     * Runs mdSPR with local explorations only that also take the shortcuts between tips with
     * close data, together with branch reform. The data of the tips differ at one to three
     * sites, so both distances connect some but not all pairs of tips. The rooted
     * topologies follow the same posterior as with the other subtree moves, also where the
     * data of the tips are partially ambiguous.
     */
    @Test
    public void testShortcutMdSprTopologiesMatchPosterior() {
        for (String[] sequences : List.of(SEQUENCES, AMBIGUOUS_SEQUENCES)) {
            for (int shortcutDistance : new int[]{1, 2}) {
                Randomizer.setSeed(19);
                this.assertShortcutMdSprMatchesPosterior(this.createSetup(new JukesCantor(), 1.0, sequences), shortcutDistance);
            }
        }
    }

    private void assertShortcutMdSprMatchesPosterior(Setup setup, int shortcutDistance) {
        MutationDirectedSprOperator mdSprOperator = new MutationDirectedSprOperator();
        mdSprOperator.initByName("weight", 1.0, "tree", setup.tree, "mutations", setup.mutations, "geneticPrior", setup.geneticPrior,
                "fullExplorationProbability", 0.0, "windowProbability", 0.5, "shortcutDistance", shortcutDistance);

        BranchReformOperator branchReformOperator = new BranchReformOperator();
        branchReformOperator.initByName("weight", 1.0, "mutations", setup.mutations, "geneticPrior", setup.geneticPrior);

        Map<String, Double> expectedFrequencies = this.computeTopologyPosterior(setup, null);
        Map<String, Integer> topologyCounts = this.runChain(setup, List.of(mdSprOperator, branchReformOperator));
        this.assertFrequencies(expectedFrequencies, topologyCounts);
        assertTrue(mdSprOperator.getDiagnostics().get("mdspr_shortcut_tips") > 0.0);
    }

    /**
     * Runs mdSPR with every strategy of picking the subtrees, together with branch reform.
     * The rooted topologies follow the same posterior as with the uniform choice.
     */
    @Test
    public void testSubtreeSelectionsMatchPosterior() {
        for (MutationDirectedSprOperator.SubtreeSelection subtreeSelection : MutationDirectedSprOperator.SubtreeSelection.values()) {
            this.assertSubtreeSelectionMatchesPosterior(subtreeSelection);
        }
    }

    private void assertSubtreeSelectionMatchesPosterior(MutationDirectedSprOperator.SubtreeSelection subtreeSelection) {
        Randomizer.setSeed(1);
        Setup setup = this.createSetup();

        MutationDirectedSprOperator mdSprOperator = new MutationDirectedSprOperator();
        mdSprOperator.initByName("weight", 1.0, "tree", setup.tree, "mutations", setup.mutations, "geneticPrior", setup.geneticPrior,
                "subtreeSelection", subtreeSelection);

        BranchReformOperator branchReformOperator = new BranchReformOperator();
        branchReformOperator.initByName("weight", 1.0, "mutations", setup.mutations, "geneticPrior", setup.geneticPrior);

        Map<String, Double> expectedFrequencies = this.computeTopologyPosterior(setup, null);
        Map<String, Integer> topologyCounts = this.runChain(setup, List.of(mdSprOperator, branchReformOperator));
        this.assertFrequencies(expectedFrequencies, topologyCounts);
    }

    /** Checks that the weight of mdSPR is scaled by the logarithm of the number of tips if requested. */
    @Test
    public void testWeightScalesWithNumTips() {
        Setup setup = this.createSetup();

        MutationDirectedSprOperator unscaledOperator = new MutationDirectedSprOperator();
        unscaledOperator.initByName("weight", 100.0, "tree", setup.tree, "mutations", setup.mutations, "geneticPrior", setup.geneticPrior);
        assertEquals(100.0, unscaledOperator.getWeight(), 1e-12);

        MutationDirectedSprOperator scaledOperator = new MutationDirectedSprOperator();
        scaledOperator.initByName("weight", 100.0, "tree", setup.tree, "mutations", setup.mutations, "geneticPrior", setup.geneticPrior,
                "scaleWeightByLogNumTips", true);
        assertEquals(100.0 * Math.log(4), scaledOperator.getWeight(), 1e-12);
    }

    private record Setup(Alignment alignment, TreeParser tree, Mutations mutations, GeneticPrior geneticPrior,
                         SubstitutionModel substitutionModel, double clockRate) {
    }

    /** Creates the alignment, a starting tree with parsimonious mutations and the genetic prior under Jukes-Cantor with clock rate one. */
    private Setup createSetup() {
        return this.createSetup(new JukesCantor(), 1.0);
    }

    /** Creates the alignment, a starting tree with parsimonious mutations and the genetic prior under the given model and clock rate. */
    private Setup createSetup(SubstitutionModel substitutionModel, double clockRate) {
        return this.createSetup(substitutionModel, clockRate, SEQUENCES);
    }

    /** Creates the setup of the given model and clock rate for the given sequences of the taxa. */
    private Setup createSetup(SubstitutionModel substitutionModel, double clockRate, String[] taxonSequences) {
        Alignment alignment = new Alignment();
        List<Sequence> sequences = new ArrayList<>();
        for (int i = 0; i < TAXA.length; i++) {
            sequences.add(new Sequence(TAXA[i], taxonSequences[i]));
        }
        alignment.initByName("sequence", sequences, "dataType", "nucleotide");

        TreeParser tree = new TreeParser();
        tree.initByName("newick", "(((A:0.4,B:0.4):0.3,C:0.7):0.3,D:1.0);", "IsLabelledNewick", true, "adjustTipHeights", false);

        Mutations mutations = new Mutations();
        mutations.initByName("tree", tree, "alignment", alignment);

        ParsimonyMutationsInitialiser initialiser = new ParsimonyMutationsInitialiser();
        initialiser.initByName("mutations", mutations);
        initialiser.initStateNodes();

        SiteModel siteModel = new SiteModel();
        siteModel.initByName("substModel", substitutionModel);

        StrictClockModel clockModel = new StrictClockModel();
        clockModel.initByName("clock.rate", new RealScalarParam<>(clockRate, PositiveReal.INSTANCE));

        GeneticPrior geneticPrior = new GeneticPrior();
        geneticPrior.initByName("data", alignment, "tree", tree, "siteModel", siteModel, "branchRateModel", clockModel, "mutations", mutations);

        return new Setup(alignment, tree, mutations, geneticPrior, substitutionModel, clockRate);
    }

    private void assertFrequencies(Map<String, Double> expectedFrequencies, Map<String, Integer> topologyCounts) {
        for (Map.Entry<String, Double> entry : expectedFrequencies.entrySet()) {
            double frequency = (double) topologyCounts.getOrDefault(entry.getKey(), 0) / NUM_SAMPLES;
            assertEquals(entry.getValue(), frequency, 0.01, "Frequency of " + entry.getKey());
        }
    }

    /**
     * Runs a Metropolis-Hastings chain with the given operators against the genetic prior,
     * driving the state through the same store, accept and restore cycle as EmatMCMC, and
     * counts the sampled rooted topologies after the burn-in.
     */
    private Map<String, Integer> runChain(Setup setup, List<Operator> operators) {
        TreeParser tree = setup.tree();
        Mutations mutations = setup.mutations();
        GeneticPrior geneticPrior = setup.geneticPrior();

        State state = new State();
        state.initByName("stateNode", tree, "stateNode", mutations);
        state.initialise();
        state.setPosterior(geneticPrior);
        state.setEverythingDirty(true);

        double oldLogP = geneticPrior.calculateLogP();
        state.setEverythingDirty(false);

        // the operators need a schedule to count their accepted and rejected proposals
        OperatorSchedule operatorSchedule = new OperatorSchedule();
        operatorSchedule.initAndValidate();
        for (Operator operator : operators) {
            operator.setOperatorSchedule(operatorSchedule);
        }

        Map<String, Integer> topologyCounts = new HashMap<>();

        for (long i = 0; i < NUM_BURNIN + NUM_SAMPLES; i++) {
            state.store(i);
            Operator operator = operators.get(Randomizer.nextInt(operators.size()));

            double logHastingsRatio = operator.proposal();
            if (logHastingsRatio == Double.NEGATIVE_INFINITY) {
                operator.reject(-2);
                state.restore();
            } else {
                state.storeCalculationNodes();
                state.checkCalculationNodesDirtiness();

                double newLogP = geneticPrior.calculateLogP();
                if (Math.log(Randomizer.nextDouble()) < newLogP - oldLogP + logHastingsRatio) {
                    oldLogP = newLogP;
                    state.acceptCalculationNodes();
                    operator.accept();
                } else {
                    operator.reject(0);
                    state.restore();
                    state.restoreCalculationNodes();
                }
            }
            state.setEverythingDirty(false);

            if (i >= NUM_BURNIN) {
                topologyCounts.merge(this.getTopology(tree.getRoot()), 1, Integer::sum);
            }
        }

        return topologyCounts;
    }

    /* Exact Posterior */

    /**
     * Computes the posterior of every rooted topology on the four taxa by integrating the
     * likelihood over the heights of the internal nodes on a grid, with the density of the
     * heights being uniform below the root, under the substitution model and clock rate of
     * the given setup. The likelihood is conditioned on the given root sequence, or averages
     * over uniform root states if there is none.
     */
    private Map<String, Double> computeTopologyPosterior(Setup setup, int[] rootSequence) {
        Alignment alignment = setup.alignment();
        double step = ROOT_HEIGHT / NUM_GRID_POINTS;

        // every branch on the grid is a multiple of half a step long
        this.halfStepTransitionProbabilities = new double[2 * NUM_GRID_POINTS + 1][];
        for (int k = 0; k <= 2 * NUM_GRID_POINTS; k++) {
            this.halfStepTransitionProbabilities[k] = StochasticMapping.computeTransitionProbabilities(
                    setup.substitutionModel().getEigenDecomposition(null), setup.clockRate() * k * 0.5 * step
            );
        }

        int[][] tipStates = new int[TAXA.length][];
        for (int i = 0; i < TAXA.length; i++) {
            tipStates[i] = alignment.getCounts().get(alignment.getTaxonIndex(TAXA[i])).stream().mapToInt(Integer::intValue).toArray();
        }

        Map<String, Double> weights = new HashMap<>();

        for (int[] order : this.getTaxonOrders()) {
            int a = order[0];
            int b = order[1];
            int c = order[2];
            int d = order[3];

            // balanced ((a,b),(c,d)): both internal heights range freely below the root

            if (a < b && c < d && a < c) {
                double weight = 0.0;
                for (int i = 0; i < NUM_GRID_POINTS; i++) {
                    for (int j = 0; j < NUM_GRID_POINTS; j++) {
                        double u = (i + 0.5) * step;
                        double v = (j + 0.5) * step;
                        weight += this.computeBalancedLikelihood(tipStates, rootSequence, a, b, c, d, u, v) * step * step;
                    }
                }
                weights.put(this.formatBalanced(a, b, c, d), weight);
            }

            // caterpillar (((a,b),c),d): the cherry is below the node that joins c

            if (a < b) {
                double weight = 0.0;
                for (int i = 0; i < NUM_GRID_POINTS; i++) {
                    for (int j = 0; j < NUM_GRID_POINTS; j++) {
                        double u = (i + 0.5) * step;
                        double v = (j + 0.5) * step;
                        if (u < v) {
                            weight += this.computeCaterpillarLikelihood(tipStates, rootSequence, a, b, c, d, u, v) * step * step;
                        }
                    }
                }
                weights.put(this.formatCaterpillar(a, b, c, d), weight);
            }
        }

        double totalWeight = weights.values().stream().mapToDouble(Double::doubleValue).sum();
        weights.replaceAll((topology, weight) -> weight / totalWeight);
        return weights;
    }

    private double computeBalancedLikelihood(int[][] tipStates, int[] rootSequence, int a, int b, int c, int d, double u, double v) {
        double likelihood = 1.0;
        for (int site = 0; site < SEQUENCES[0].length(); site++) {
            double[] left = this.computeCherryPartials(tipStates, site, a, b, u);
            double[] right = this.computeCherryPartials(tipStates, site, c, d, v);
            double[] root = this.multiply(this.propagate(left, ROOT_HEIGHT - u), this.propagate(right, ROOT_HEIGHT - v));
            likelihood *= this.computeRootLikelihood(root, rootSequence, site);
        }
        return likelihood;
    }

    private double computeCaterpillarLikelihood(int[][] tipStates, int[] rootSequence, int a, int b, int c, int d, double u, double v) {
        double likelihood = 1.0;
        for (int site = 0; site < SEQUENCES[0].length(); site++) {
            double[] cherry = this.computeCherryPartials(tipStates, site, a, b, u);
            double[] inner = this.multiply(this.propagate(cherry, v - u), this.propagate(this.getTipPartials(tipStates, site, c), v));
            double[] root = this.multiply(this.propagate(inner, ROOT_HEIGHT - v), this.propagate(this.getTipPartials(tipStates, site, d), ROOT_HEIGHT));
            likelihood *= this.computeRootLikelihood(root, rootSequence, site);
        }
        return likelihood;
    }

    /** Returns the partial at the root for the given root state, or its average over uniform root states if there is none. */
    private double computeRootLikelihood(double[] rootPartials, int[] rootSequence, int site) {
        if (rootSequence != null) {
            return rootPartials[rootSequence[site]];
        }
        return 0.25 * (rootPartials[0] + rootPartials[1] + rootPartials[2] + rootPartials[3]);
    }

    private double[] computeCherryPartials(int[][] tipStates, int site, int a, int b, double height) {
        return this.multiply(
                this.propagate(this.getTipPartials(tipStates, site, a), height),
                this.propagate(this.getTipPartials(tipStates, site, b), height)
        );
    }

    /** Returns the partials of a tip, which are one for every state that is compatible with its data. */
    private double[] getTipPartials(int[][] tipStates, int site, int taxon) {
        double[] partials = new double[4];
        for (int state : NUCLEOTIDE.getStatesForCode(tipStates[taxon][site])) {
            partials[state] = 1.0;
        }
        return partials;
    }

    /** Returns the partials at the top of a branch of the given length, a multiple of half a grid step, from those at its bottom. */
    private double[] propagate(double[] partials, double length) {
        int numHalfSteps = (int) Math.round(2.0 * length * NUM_GRID_POINTS / ROOT_HEIGHT);
        double[] transitionProbabilities = this.halfStepTransitionProbabilities[numHalfSteps];

        double[] result = new double[4];
        for (int from = 0; from < 4; from++) {
            for (int to = 0; to < 4; to++) {
                result[from] += transitionProbabilities[from * 4 + to] * partials[to];
            }
        }
        return result;
    }

    private double[] multiply(double[] left, double[] right) {
        double[] result = new double[4];
        for (int state = 0; state < 4; state++) {
            result[state] = left[state] * right[state];
        }
        return result;
    }

    /* Topologies */

    /** Returns all orders of the taxon indices. */
    private List<int[]> getTaxonOrders() {
        List<int[]> orders = new ArrayList<>();
        for (int a = 0; a < 4; a++) {
            for (int b = 0; b < 4; b++) {
                for (int c = 0; c < 4; c++) {
                    int d = 6 - a - b - c;
                    if (a != b && a != c && b != c && d != a && d != b && d != c) {
                        orders.add(new int[]{a, b, c, d});
                    }
                }
            }
        }
        return orders;
    }

    private String formatBalanced(int a, int b, int c, int d) {
        return this.formatPair("(" + TAXA[a] + "," + TAXA[b] + ")", "(" + TAXA[c] + "," + TAXA[d] + ")");
    }

    private String formatCaterpillar(int a, int b, int c, int d) {
        String cherry = "(" + TAXA[a] + "," + TAXA[b] + ")";
        return this.formatPair(this.formatPair(cherry, TAXA[c]), TAXA[d]);
    }

    /** Formats two subtrees in a canonical order, so that every rooted topology has one string. */
    private String formatPair(String left, String right) {
        return left.compareTo(right) < 0 ? "(" + left + "," + right + ")" : "(" + right + "," + left + ")";
    }

    /** Returns the canonical string of the rooted topology below the given node. */
    private String getTopology(Node node) {
        if (node.isLeaf()) {
            return node.getID();
        }

        TreeSet<String> children = new TreeSet<>();
        for (Node child : node.getChildren()) {
            children.add(this.getTopology(child));
        }
        return this.formatPair(children.first(), children.last());
    }

}
