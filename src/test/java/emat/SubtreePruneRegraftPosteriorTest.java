package emat;

import beast.base.evolution.alignment.Alignment;
import beast.base.evolution.alignment.Sequence;
import beast.base.evolution.tree.Node;
import beast.base.evolution.tree.TreeParser;
import beast.base.inference.Operator;
import beast.base.inference.State;
import beast.base.spec.domain.PositiveReal;
import beast.base.spec.evolution.branchratemodel.StrictClockModel;
import beast.base.spec.evolution.sitemodel.SiteModel;
import beast.base.spec.evolution.substitutionmodel.JukesCantor;
import beast.base.spec.inference.parameter.RealScalarParam;
import beast.base.util.Randomizer;
import emat.helper.MutationPaths;
import emat.initalisation.ParsimonyMutationsInitialiser;
import emat.operators.BranchReformOperator;
import emat.operators.MutationDirectedSprOperator;
import emat.operators.SubtreeSlideOperator;
import emat.operators.WilsonBaldingOperator;
import emat.prior.GeneticPrior;
import emat.state.Mutations;
import org.junit.jupiter.api.Test;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.TreeSet;

import static org.junit.jupiter.api.Assertions.assertEquals;

public class SubtreePruneRegraftPosteriorTest {

    private static final int NUM_BURNIN = 20000;
    private static final int NUM_SAMPLES = 1000000;
    private static final int NUM_GRID_POINTS = 400;

    private static final double ROOT_HEIGHT = 1.0;
    private static final String[] TAXA = {"A", "B", "C", "D"};
    // B and D have missing data, whose states the moves resample when they prune these tips
    private static final String[] SEQUENCES = {"ACGTC", "ACGAN", "TCGAC", "TNNAT"};

    /**
     * Runs Wilson–Balding and subtree slide with neighbourhood resampling against the
     * genetic prior under Jukes-Cantor on four tips at height 0, two of them with missing
     * data. Neither operator changes the root height or the root sequence, so the rooted
     * topologies follow the Felsenstein likelihood given the root sequence, integrated over
     * the heights of the two internal nodes below the root.
     */
    @Test
    public void testNeighbourhoodTopologiesMatchPosterior() {
        Randomizer.setSeed(7);
        Setup setup = this.createSetup();

        WilsonBaldingOperator wilsonBaldingOperator = new WilsonBaldingOperator();
        wilsonBaldingOperator.initByName("weight", 1.0, "tree", setup.tree, "mutations", setup.mutations,
                "geneticPrior", setup.geneticPrior, "resampleNeighbourhood", true);

        SubtreeSlideOperator subtreeSlideOperator = new SubtreeSlideOperator();
        subtreeSlideOperator.initByName("weight", 1.0, "tree", setup.tree, "mutations", setup.mutations,
                "geneticPrior", setup.geneticPrior, "size", 0.3, "optimise", false, "resampleNeighbourhood", true);

        // the root sequence stays fixed, so the posterior is conditioned on it

        int[] rootSequence = new int[SEQUENCES[0].length()];
        for (int site = 0; site < rootSequence.length; site++) {
            rootSequence[site] = MutationPaths.getState(setup.mutations, setup.tree.getRoot(), site);
        }

        Map<String, Double> expectedFrequencies = this.computeTopologyPosterior(setup.alignment, rootSequence);
        Map<String, Integer> topologyCounts = this.runChain(setup, List.of(wilsonBaldingOperator, subtreeSlideOperator));
        this.assertFrequencies(expectedFrequencies, topologyCounts);
    }

    /**
     * Runs Wilson–Balding, subtree slide and mdSPR, which resample only the branch above the
     * pruned subtree and the missing states of a pruned tip, together with branch reform,
     * which also changes the root sequence. The rooted topologies then follow the
     * Felsenstein likelihood with uniform root frequencies, integrated over the heights of
     * the two internal nodes below the root.
     */
    @Test
    public void testSubtreeTopologiesMatchPosterior() {
        Randomizer.setSeed(11);
        Setup setup = this.createSetup();

        WilsonBaldingOperator wilsonBaldingOperator = new WilsonBaldingOperator();
        wilsonBaldingOperator.initByName("weight", 1.0, "tree", setup.tree, "mutations", setup.mutations, "geneticPrior", setup.geneticPrior);

        SubtreeSlideOperator subtreeSlideOperator = new SubtreeSlideOperator();
        subtreeSlideOperator.initByName("weight", 1.0, "tree", setup.tree, "mutations", setup.mutations,
                "geneticPrior", setup.geneticPrior, "size", 0.3, "optimise", false);

        MutationDirectedSprOperator mdSprOperator = new MutationDirectedSprOperator();
        mdSprOperator.initByName("weight", 1.0, "tree", setup.tree, "mutations", setup.mutations, "geneticPrior", setup.geneticPrior);

        BranchReformOperator branchReformOperator = new BranchReformOperator();
        branchReformOperator.initByName("weight", 1.0, "mutations", setup.mutations, "geneticPrior", setup.geneticPrior);

        Map<String, Double> expectedFrequencies = this.computeTopologyPosterior(setup.alignment, null);
        Map<String, Integer> topologyCounts = this.runChain(
                setup, List.of(wilsonBaldingOperator, subtreeSlideOperator, mdSprOperator, branchReformOperator)
        );
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

        Map<String, Double> expectedFrequencies = this.computeTopologyPosterior(setup.alignment, null);
        Map<String, Integer> topologyCounts = this.runChain(setup, List.of(mdSprOperator, branchReformOperator));
        this.assertFrequencies(expectedFrequencies, topologyCounts);
    }

    private record Setup(Alignment alignment, TreeParser tree, Mutations mutations, GeneticPrior geneticPrior) {
    }

    /** Creates the alignment, a starting tree with parsimonious mutations and the genetic prior under Jukes-Cantor. */
    private Setup createSetup() {
        Alignment alignment = new Alignment();
        List<Sequence> sequences = new ArrayList<>();
        for (int i = 0; i < TAXA.length; i++) {
            sequences.add(new Sequence(TAXA[i], SEQUENCES[i]));
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
        siteModel.initByName("substModel", new JukesCantor());

        StrictClockModel clockModel = new StrictClockModel();
        clockModel.initByName("clock.rate", new RealScalarParam<>(1.0, PositiveReal.INSTANCE));

        GeneticPrior geneticPrior = new GeneticPrior();
        geneticPrior.initByName("data", alignment, "tree", tree, "siteModel", siteModel, "branchRateModel", clockModel, "mutations", mutations);

        return new Setup(alignment, tree, mutations, geneticPrior);
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

        Map<String, Integer> topologyCounts = new HashMap<>();

        for (long i = 0; i < NUM_BURNIN + NUM_SAMPLES; i++) {
            state.store(i);
            Operator operator = operators.get(Randomizer.nextInt(operators.size()));

            double logHastingsRatio = operator.proposal();
            if (logHastingsRatio == Double.NEGATIVE_INFINITY) {
                state.restore();
            } else {
                state.storeCalculationNodes();
                state.checkCalculationNodesDirtiness();

                double newLogP = geneticPrior.calculateLogP();
                if (Math.log(Randomizer.nextDouble()) < newLogP - oldLogP + logHastingsRatio) {
                    oldLogP = newLogP;
                    state.acceptCalculationNodes();
                } else {
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
     * heights being uniform below the root. The likelihood is conditioned on the given root
     * sequence, or averages over uniform root states if there is none.
     */
    private Map<String, Double> computeTopologyPosterior(Alignment alignment, int[] rootSequence) {
        int[][] tipStates = new int[TAXA.length][];
        for (int i = 0; i < TAXA.length; i++) {
            tipStates[i] = alignment.getCounts().get(alignment.getTaxonIndex(TAXA[i])).stream().mapToInt(Integer::intValue).toArray();
        }

        Map<String, Double> weights = new HashMap<>();
        double step = ROOT_HEIGHT / NUM_GRID_POINTS;

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

    /** Returns the partials of a tip, which are one for every state at a missing site. */
    private double[] getTipPartials(int[][] tipStates, int site, int taxon) {
        double[] partials = new double[4];
        int state = tipStates[taxon][site];
        if (state < 4) {
            partials[state] = 1.0;
        } else {
            Arrays.fill(partials, 1.0);
        }
        return partials;
    }

    /** Returns the partials at the top of a branch of the given length from those at its bottom under Jukes-Cantor. */
    private double[] propagate(double[] partials, double length) {
        double stayProbability = 0.25 * (1.0 + 3.0 * Math.exp(-4.0 / 3.0 * length));
        double changeProbability = 0.25 * (1.0 - Math.exp(-4.0 / 3.0 * length));

        double[] result = new double[4];
        for (int from = 0; from < 4; from++) {
            for (int to = 0; to < 4; to++) {
                result[from] += (from == to ? stayProbability : changeProbability) * partials[to];
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
