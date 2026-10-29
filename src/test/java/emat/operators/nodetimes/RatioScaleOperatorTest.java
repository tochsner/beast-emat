package emat.operators.nodetimes;

import beast.base.evolution.alignment.Alignment;
import beast.base.evolution.alignment.Sequence;
import beast.base.evolution.tree.Node;
import beast.base.evolution.tree.TreeParser;
import beast.base.spec.domain.PositiveReal;
import beast.base.spec.evolution.branchratemodel.StrictClockModel;
import beast.base.spec.evolution.sitemodel.SiteModel;
import beast.base.spec.evolution.substitutionmodel.JukesCantor;
import beast.base.spec.inference.parameter.RealScalarParam;
import beast.base.util.Randomizer;
import emat.initalisation.ParsimonyMutationsInitialiser;
import emat.prior.GeneticPrior;
import emat.state.Mutation;
import emat.state.Mutations;
import org.junit.jupiter.api.Test;

import java.util.ArrayList;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;

import static org.junit.jupiter.api.Assertions.assertEquals;

public class RatioScaleOperatorTest {

    private static final List<Sequence> SEQUENCES = List.of(
            new Sequence("A", "ACGT"),
            new Sequence("B", "ACGA"),
            new Sequence("C", "CCGA"),
            new Sequence("D", "CTTA")
    );

    // a caterpillar (((A,B),C),D) whose tips C and D are sampled before A and B
    private static final String NEWICK = "(((A:0.5,B:0.5):0.5,C:0.7):0.5,D:1.3);";

    private static final int NUM_BURNIN = 10000;
    private static final int NUM_SAMPLES = 400000;

    private static final int NUM_GRID_POINTS = 600;
    private static final double MAX_GRID_HEIGHT = 4.0;

    @Test
    public void testFixedHistoryMatchesExactPosterior() {
        this.assertFixedHistoryMatchesExactPosterior(new RatioScaleOperator(), "optimise", false);
    }

    /**
     * The step of a proposal depends only on the size of the picked clade, which the move
     * keeps, so tuning the step of every clade size class still targets the exact posterior.
     */
    @Test
    public void testTunedFixedHistoryMatchesExactPosterior() {
        this.assertFixedHistoryMatchesExactPosterior(new RatioScaleOperator(), "optimise", true);
    }

    /**
     * The step of a proposal depends only on the number of mutations of the picked subtree,
     * which the move keeps, so tuning the step of every mutation count class still targets
     * the exact posterior.
     */
    @Test
    public void testMutationClassesMatchExactPosterior() {
        this.assertFixedHistoryMatchesExactPosterior(new MutationRatioScaleOperator(), "optimise", true);
    }

    /**
     * On a four-tip caterpillar with serially sampled tips and an exponential prior on the
     * root height, runs a Metropolis-Hastings chain with only the given operator, which
     * keeps the parsimony history and the relative positions of its mutations. Under
     * Jukes-Cantor with rate 1, the genetic prior is exp(-L T) up to the mutation times, with
     * L sites and tree length T, and keeping the m_b mutations of every branch b at their
     * relative positions contributes len_b^m_b. The internal node heights then follow
     * exp(-h_root - L T) Π_b len_b^m_b, whose means are integrated on a grid.
     */
    void assertFixedHistoryMatchesExactPosterior(ScaleTunedOperator operator, Object... inputs) {
        Randomizer.setSeed(1);

        Alignment alignment = new Alignment();
        alignment.initByName("sequence", SEQUENCES, "dataType", "nucleotide");

        TreeParser tree = new TreeParser();
        tree.initByName("newick", NEWICK, "IsLabelledNewick", true, "adjustTipHeights", false);

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

        // the given inputs override the defaults, which leave the scale factor untuned
        Map<String, Object> operatorInputs = new LinkedHashMap<>();
        operatorInputs.put("weight", 1.0);
        operatorInputs.put("tree", tree);
        operatorInputs.put("mutations", mutations);
        operatorInputs.put("scaleFactor", 0.5);
        operatorInputs.put("optimise", false);
        for (int i = 0; i < inputs.length; i += 2) {
            operatorInputs.put((String) inputs[i], inputs[i + 1]);
        }
        List<Object> initArguments = new ArrayList<>();
        operatorInputs.forEach((name, value) -> {
            initArguments.add(name);
            initArguments.add(value);
        });
        operator.initByName(initArguments.toArray());

        double[] expectedMeans = this.integrateMeanHeights(tree, mutations, SEQUENCES.getFirst().getData().length());
        double[] sampledMeans = this.runChain(tree, mutations, geneticPrior, operator);

        for (int i = 0; i < expectedMeans.length; i++) {
            assertEquals(expectedMeans[i], sampledMeans[i], 0.01);
        }
    }

    /**
     * Integrates the means of the internal node heights h1 < h2 < h3 of the caterpillar,
     * from the cherry to the root, by the midpoint rule on a regular grid.
     */
    private double[] integrateMeanHeights(TreeParser tree, Mutations mutations, int numSites) {
        Node cherry = this.getCommonAncestor(tree, "A", "B");
        Node middle = cherry.getParent();
        Node root = tree.getRoot();

        Node[] internalNodes = {cherry, middle, root};
        double[] heights = new double[tree.getNodeCount()];
        for (Node node : tree.getNodesAsArray()) {
            heights[node.getNr()] = node.getHeight();
        }

        double stepSize = MAX_GRID_HEIGHT / NUM_GRID_POINTS;
        double normalisation = 0.0;
        double[] means = new double[internalNodes.length];

        for (int i = 0; i < NUM_GRID_POINTS; i++) {
            for (int j = 0; j < NUM_GRID_POINTS; j++) {
                for (int k = 0; k < NUM_GRID_POINTS; k++) {
                    heights[cherry.getNr()] = (i + 0.5) * stepSize;
                    heights[middle.getNr()] = (j + 0.5) * stepSize;
                    heights[root.getNr()] = (k + 0.5) * stepSize;

                    double logDensity = this.computeLogDensity(tree, mutations, heights, numSites);
                    if (logDensity == Double.NEGATIVE_INFINITY) {
                        continue;
                    }

                    double density = Math.exp(logDensity);
                    normalisation += density;
                    for (int n = 0; n < internalNodes.length; n++) {
                        means[n] += density * heights[internalNodes[n].getNr()];
                    }
                }
            }
        }

        for (int n = 0; n < internalNodes.length; n++) {
            means[n] /= normalisation;
        }
        return means;
    }

    /** Returns the most recent common ancestor of the two tips with the given names. */
    private Node getCommonAncestor(TreeParser tree, String firstName, String secondName) {
        Node first = tree.getNode(tree.getTaxonset().getTaxonIndex(firstName));
        Node second = tree.getNode(tree.getTaxonset().getTaxonIndex(secondName));
        return first.getParent() == second.getParent() ? first.getParent() : null;
    }

    /**
     * Returns the log of exp(-h_root - L T) Π_b len_b^m_b at the given heights, or negative
     * infinity if a node is not above its children.
     */
    private double computeLogDensity(TreeParser tree, Mutations mutations, double[] heights, int numSites) {
        double logDensity = -heights[tree.getRoot().getNr()];

        for (Node node : tree.getNodesAsArray()) {
            if (node.isRoot()) {
                continue;
            }

            double length = heights[node.getParent().getNr()] - heights[node.getNr()];
            if (length <= 0.0) {
                return Double.NEGATIVE_INFINITY;
            }

            logDensity += -numSites * length + mutations.getMutations(node).size() * Math.log(length);
        }

        return logDensity;
    }

    /**
     * Runs a Metropolis-Hastings chain with the given operator against the genetic prior and
     * the prior exp(-h) on the root height h, and returns the mean heights of the cherry, the
     * middle node and the root after the burn-in.
     */
    private double[] runChain(TreeParser tree, Mutations mutations, GeneticPrior geneticPrior, ScaleTunedOperator operator) {
        Node cherry = this.getCommonAncestor(tree, "A", "B");
        Node[] internalNodes = {cherry, cherry.getParent(), tree.getRoot()};
        Node[] nodes = tree.getNodesAsArray();

        double logTarget = -tree.getRoot().getHeight() + geneticPrior.calculateLogP();
        double[] sumHeights = new double[internalNodes.length];

        for (int i = 0; i < NUM_BURNIN + NUM_SAMPLES; i++) {
            double[] oldHeights = new double[nodes.length];
            List<List<Mutation>> oldMutations = new ArrayList<>();
            for (Node node : nodes) {
                oldHeights[node.getNr()] = node.getHeight();
                oldMutations.add(mutations.getMutations(node));
            }

            double logHastingsRatio = operator.proposal();

            boolean isAccepted = false;
            double logAlpha = Double.NEGATIVE_INFINITY;
            if (logHastingsRatio != Double.NEGATIVE_INFINITY) {
                double newLogTarget = -tree.getRoot().getHeight() + geneticPrior.calculateLogP();
                logAlpha = newLogTarget - logTarget + logHastingsRatio;
                if (Math.log(Randomizer.nextDouble()) < logAlpha) {
                    logTarget = newLogTarget;
                    isAccepted = true;
                }
            }

            // like the MCMC, the operator tunes itself after every proposal, which does nothing unless optimise is set
            operator.optimize(logAlpha);

            if (!isAccepted) {
                // restore all heights first, as the mutations are checked against their branches
                for (Node node : nodes) {
                    if (!node.isLeaf()) {
                        node.setHeight(oldHeights[node.getNr()]);
                    }
                }
                for (Node node : nodes) {
                    mutations.applyMutations(node, oldMutations.get(node.getNr()), operator);
                }
            }

            if (i >= NUM_BURNIN) {
                for (int n = 0; n < internalNodes.length; n++) {
                    sumHeights[n] += internalNodes[n].getHeight();
                }
            }
        }

        for (int n = 0; n < internalNodes.length; n++) {
            sumHeights[n] /= NUM_SAMPLES;
        }
        return sumHeights;
    }

}
