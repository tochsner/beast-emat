package emat.operators.nodetimes;

import beast.base.evolution.alignment.Alignment;
import beast.base.evolution.alignment.Sequence;
import beast.base.evolution.tree.Node;
import beast.base.evolution.tree.TreeParser;
import beast.base.inference.Operator;
import beast.base.inference.OperatorSchedule;
import beast.base.spec.domain.PositiveReal;
import beast.base.spec.evolution.branchratemodel.StrictClockModel;
import beast.base.spec.evolution.sitemodel.SiteModel;
import beast.base.spec.evolution.substitutionmodel.JukesCantor;
import beast.base.spec.inference.parameter.RealScalarParam;
import beast.base.util.Randomizer;
import emat.initalisation.ParsimonyMutationsInitialiser;
import emat.operators.mutations.SiteHistoryGibbsOperator;
import emat.prior.GeneticPrior;
import emat.state.Mutation;
import emat.state.Mutations;
import org.junit.jupiter.api.Test;

import java.util.ArrayList;
import java.util.List;

import static org.junit.jupiter.api.Assertions.assertEquals;

public class TiltedIntervalScaleOperatorTest {

    private static final String[] SEQUENCES = {"ACG", "ACT", "CCT"};

    private static final int NUM_BURNIN = 2000;
    private static final int NUM_SAMPLES = 1000000;

    /**
     * On the three-tip tree ((A,B),C) with an exponential prior on both node heights, runs a Metropolis-Hastings
     * chain that cycles through the tilted and the plain interval scale operator and the site history Gibbs operator.
     * The tilt keeps the tree length, so the plain operator is needed to reach the whole space. The chain targets
     * p(h1, h2) ∝ exp(-h1 - h2) L(h1, h2) for h1 < h2 with the Jukes-Cantor likelihood L, whose means are compared
     * with the sampled means.
     */
    @Test
    public void testThreeTipsMatchExactPosteriorForDepth() {
        this.assertChainMatchesExactPosterior(TiltedIntervalScaleOperator.Coordinate.DEPTH, 1);
    }

    /** The same chain with the tilt by the rank of the node heights. */
    @Test
    public void testThreeTipsMatchExactPosteriorForRank() {
        this.assertChainMatchesExactPosterior(TiltedIntervalScaleOperator.Coordinate.RANK, 2);
    }

    private void assertChainMatchesExactPosterior(TiltedIntervalScaleOperator.Coordinate coordinate, long seed) {
        Randomizer.setSeed(seed);

        Alignment alignment = new Alignment();
        List<Sequence> sequences = List.of(new Sequence("A", SEQUENCES[0]), new Sequence("B", SEQUENCES[1]), new Sequence("C", SEQUENCES[2]));
        alignment.initByName("sequence", sequences, "dataType", "nucleotide");

        TreeParser tree = new TreeParser();
        tree.initByName("newick", "((A:1.0,B:1.0):0.5,C:1.5);", "IsLabelledNewick", true, "adjustTipHeights", false);

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

        TiltedIntervalScaleOperator tiltOperator = new TiltedIntervalScaleOperator();
        tiltOperator.initByName("weight", 1.0, "tree", tree, "mutations", mutations, "scaleFactor", 0.5, "optimise", false, "coordinate", coordinate);
        tiltOperator.setOperatorSchedule(new OperatorSchedule());
        IntervalScaleOperator scaleOperator = new IntervalScaleOperator();
        scaleOperator.initByName("weight", 1.0, "tree", tree, "mutations", mutations, "scaleFactor", 0.5, "optimise", false);
        scaleOperator.setOperatorSchedule(new OperatorSchedule());
        SiteHistoryGibbsOperator gibbsOperator = new SiteHistoryGibbsOperator();
        gibbsOperator.initByName("weight", 1.0, "mutations", mutations, "geneticPrior", geneticPrior);

        Node root = tree.getRoot();
        Node cherry = root.getLeft().isLeaf() ? root.getRight() : root.getLeft();

        double sumCherryHeights = 0.0;
        double sumRootHeights = 0.0;
        int numAcceptedTilts = 0;

        for (int i = 0; i < NUM_BURNIN + NUM_SAMPLES; i++) {
            // the Gibbs operator draws from the exact conditional, so it is always accepted

            gibbsOperator.proposal();

            Operator operator = i % 2 == 0 ? tiltOperator : scaleOperator;
            boolean isAccepted = this.runMetropolisHastingsStep(tree, mutations, geneticPrior, operator);
            numAcceptedTilts += isAccepted && operator == tiltOperator ? 1 : 0;

            if (i >= NUM_BURNIN) {
                sumCherryHeights += cherry.getHeight();
                sumRootHeights += root.getHeight();
            }
        }

        assertEquals(true, numAcceptedTilts > 1000, "accepted tilts " + numAcceptedTilts);

        double[] expectedHeights = this.computeExactMeanHeights();
        assertEquals(expectedHeights[0], sumCherryHeights / NUM_SAMPLES, 0.02);
        assertEquals(expectedHeights[1], sumRootHeights / NUM_SAMPLES, 0.03);
    }

    /** Proposes with the given operator and accepts or restores the heights and mutations. Returns whether the proposal was accepted. */
    private boolean runMetropolisHastingsStep(TreeParser tree, Mutations mutations, GeneticPrior geneticPrior, Operator operator) {
        Node[] nodes = tree.getNodesAsArray();
        double[] oldHeights = new double[nodes.length];
        List<List<Mutation>> oldMutations = new ArrayList<>();
        for (Node node : nodes) {
            oldHeights[node.getNr()] = node.getHeight();
            oldMutations.add(mutations.getMutations(node));
        }
        double logTarget = this.computeLogTarget(tree, geneticPrior);

        double logHastingsRatio = operator.proposal();
        double newLogTarget = logHastingsRatio == Double.NEGATIVE_INFINITY ? Double.NEGATIVE_INFINITY : this.computeLogTarget(tree, geneticPrior);

        if (Math.log(Randomizer.nextDouble()) < newLogTarget - logTarget + logHastingsRatio) {
            operator.accept();
            return true;
        }

        operator.reject(logHastingsRatio == Double.NEGATIVE_INFINITY ? -2 : 0);
        for (Node node : nodes) {
            node.setHeight(oldHeights[node.getNr()]);
        }
        for (Node node : nodes) {
            mutations.applyMutations(node, oldMutations.get(node.getNr()), operator);
        }
        return false;
    }

    private double computeLogTarget(TreeParser tree, GeneticPrior geneticPrior) {
        double sumInternalHeights = 0.0;
        for (Node node : tree.getInternalNodes()) {
            sumInternalHeights += node.getHeight();
        }
        return -sumInternalHeights + geneticPrior.calculateLogP();
    }

    /** Returns the exact posterior means of the heights of the cherry and the root, integrated by the midpoint rule on a grid. */
    private double[] computeExactMeanHeights() {
        int numSteps = 1500;
        double maxHeight = 15.0;
        double stepSize = maxHeight / numSteps;

        double normalisation = 0.0;
        double meanCherryHeight = 0.0;
        double meanRootHeight = 0.0;

        for (int i = 0; i < numSteps; i++) {
            double cherryHeight = (i + 0.5) * stepSize;
            for (int j = i; j < numSteps; j++) {
                double rootHeight = (j + 0.5) * stepSize;
                double weight = (j == i ? 0.5 : 1.0) * Math.exp(-cherryHeight - rootHeight) * this.computeLikelihood(cherryHeight, rootHeight);

                normalisation += weight;
                meanCherryHeight += weight * cherryHeight;
                meanRootHeight += weight * rootHeight;
            }
        }

        return new double[]{meanCherryHeight / normalisation, meanRootHeight / normalisation};
    }

    /** Returns the Jukes-Cantor likelihood of the three-tip tree ((A,B),C) with the given heights of the cherry and the root. */
    private double computeLikelihood(double cherryHeight, double rootHeight) {
        double likelihood = 1.0;

        for (int site = 0; site < SEQUENCES[0].length(); site++) {
            char a = SEQUENCES[0].charAt(site);
            char b = SEQUENCES[1].charAt(site);
            char c = SEQUENCES[2].charAt(site);

            double siteLikelihood = 0.0;
            for (char rootState : "ACGT".toCharArray()) {
                for (char cherryState : "ACGT".toCharArray()) {
                    siteLikelihood += 0.25
                            * this.getJukesCantorProbability(rootState == cherryState, rootHeight - cherryHeight)
                            * this.getJukesCantorProbability(cherryState == a, cherryHeight)
                            * this.getJukesCantorProbability(cherryState == b, cherryHeight)
                            * this.getJukesCantorProbability(rootState == c, rootHeight);
                }
            }
            likelihood *= siteLikelihood;
        }

        return likelihood;
    }

    /** Returns the Jukes-Cantor probability of ending in the same or in a given other state over the given time. */
    private double getJukesCantorProbability(boolean isSame, double time) {
        double decay = Math.exp(-4.0 * time / 3.0);
        return isSame ? 0.25 + 0.75 * decay : 0.25 - 0.25 * decay;
    }

}
