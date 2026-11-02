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
import static org.junit.jupiter.api.Assertions.assertTrue;

/**
 * Checks the interval scale operators on the four-tip tree ((A,B),(C,D)), where the tip D is sampled later than the
 * others. A move can then change which of the two cherries is the older child of the root, so that the older child
 * changes, and the chain has to sample the exact posterior including the probability of either order.
 */
public class OrientationChangeIntervalScaleTest {

    private static final String[] SEQUENCES = {"ACGT", "ACTT", "CCTA", "CGTA"};
    private static final double TIP_HEIGHT_OF_D = 0.5;

    private static final int NUM_BURNIN = 2000;
    private static final int NUM_SAMPLES = 800000;

    @Test
    public void testTiltedDepthMatchesExactPosterior() {
        this.assertChainMatchesExactPosterior(1, (tree, mutations) -> {
            TiltedIntervalScaleOperator depthTilt = new TiltedIntervalScaleOperator();
            depthTilt.initByName("weight", 1.0, "tree", tree, "mutations", mutations, "scaleFactor", 0.5, "optimise", false, "coordinate", TiltedIntervalScaleOperator.Coordinate.DEPTH);
            return depthTilt;
        });
    }

    @Test
    public void testMutationNoiseMatchesExactPosterior() {
        this.assertChainMatchesExactPosterior(23, (tree, mutations) -> {
            MutationNoiseIntervalScaleOperator noise = new MutationNoiseIntervalScaleOperator();
            noise.initByName("weight", 1.0, "tree", tree, "mutations", mutations, "scaleFactor", 0.5, "optimise", false);
            return noise;
        });
    }

    @Test
    public void testBandedMatchesExactPosterior() {
        this.assertChainMatchesExactPosterior(2, (tree, mutations) -> {
            BandedIntervalScaleOperator banded = new BandedIntervalScaleOperator();
            banded.initByName("weight", 1.0, "tree", tree, "mutations", mutations, "scaleFactor", 0.5, "optimise", false);
            return banded;
        });
    }

    private interface OperatorFactory {
        Operator create(TreeParser tree, Mutations mutations);
    }

    /**
     * Runs a chain that cycles through the operator under test, the plain interval scale operator, the tilt by the rank
     * (which changes the ratio of the two cherry heights) and the site history Gibbs operator, and compares the means of the
     * node heights and the probability that the first cherry is the higher one with the exact posterior.
     */
    private void assertChainMatchesExactPosterior(long seed, OperatorFactory factory) {
        Randomizer.setSeed(seed);

        Alignment alignment = new Alignment();
        List<Sequence> sequences = new ArrayList<>();
        for (int i = 0; i < SEQUENCES.length; i++) {
            sequences.add(new Sequence("ABCD".substring(i, i + 1), SEQUENCES[i]));
        }
        alignment.initByName("sequence", sequences, "dataType", "nucleotide");

        TreeParser tree = new TreeParser();
        tree.initByName("newick", "((A:1.5,B:1.5):0.5,(C:1.1,D:0.6):0.9);", "IsLabelledNewick", true, "adjustTipHeights", false);
        Node tipD = null;
        for (Node node : tree.getNodesAsArray()) {
            if (node.isLeaf() && node.getID().equals("D")) {
                tipD = node;
            }
        }
        assertEquals(TIP_HEIGHT_OF_D, tipD.getHeight(), 1e-9);

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

        Operator operatorUnderTest = factory.create(tree, mutations);
        operatorUnderTest.setOperatorSchedule(new OperatorSchedule());
        IntervalScaleOperator scaleOperator = new IntervalScaleOperator();
        scaleOperator.initByName("weight", 1.0, "tree", tree, "mutations", mutations, "scaleFactor", 0.5, "optimise", false);
        scaleOperator.setOperatorSchedule(new OperatorSchedule());
        TiltedIntervalScaleOperator rankTilt = new TiltedIntervalScaleOperator();
        rankTilt.initByName("weight", 1.0, "tree", tree, "mutations", mutations, "scaleFactor", 0.5, "optimise", false, "coordinate", TiltedIntervalScaleOperator.Coordinate.RANK);
        rankTilt.setOperatorSchedule(new OperatorSchedule());
        SiteHistoryGibbsOperator gibbsOperator = new SiteHistoryGibbsOperator();
        gibbsOperator.initByName("weight", 1.0, "mutations", mutations, "geneticPrior", geneticPrior);

        Node root = tree.getRoot();
        Node cherryAB = this.findCherry(root, "A");
        Node cherryCD = this.findCherry(root, "C");

        double sumAB = 0.0;
        double sumCD = 0.0;
        double sumRoot = 0.0;
        double numAHigher = 0.0;
        int numAcceptedChanges = 0;
        Operator[] operators = {operatorUnderTest, scaleOperator, rankTilt};

        for (int i = 0; i < NUM_BURNIN + NUM_SAMPLES; i++) {
            // the Gibbs operator draws from the exact conditional, so it is always accepted

            gibbsOperator.proposal();

            Operator operator = operators[i % operators.length];
            boolean wasABHigher = cherryAB.getHeight() > cherryCD.getHeight();
            boolean isAccepted = this.runMetropolisHastingsStep(tree, mutations, geneticPrior, operator);
            boolean isABHigher = cherryAB.getHeight() > cherryCD.getHeight();
            numAcceptedChanges += isAccepted && operator == operatorUnderTest && wasABHigher != isABHigher ? 1 : 0;

            if (i >= NUM_BURNIN) {
                sumAB += cherryAB.getHeight();
                sumCD += cherryCD.getHeight();
                sumRoot += root.getHeight();
                numAHigher += isABHigher ? 1.0 : 0.0;
            }
        }

        assertTrue(numAcceptedChanges > 100, "accepted changes of the older cherry " + numAcceptedChanges);

        double[] expected = this.computeExactPosterior();
        assertEquals(expected[0], sumAB / NUM_SAMPLES, 0.05);
        assertEquals(expected[1], sumCD / NUM_SAMPLES, 0.05);
        assertEquals(expected[2], sumRoot / NUM_SAMPLES, 0.06);
        assertEquals(expected[3], numAHigher / NUM_SAMPLES, 0.02);
    }

    private Node findCherry(Node root, String tipId) {
        for (Node child : List.of(root.getLeft(), root.getRight())) {
            if (child.getLeft().getID().equals(tipId) || child.getRight().getID().equals(tipId)) {
                return child;
            }
        }
        throw new IllegalStateException("No cherry has the tip " + tipId);
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

    /**
     * Returns the exact posterior means of the heights of the cherries (A,B) and (C,D) and of the root, and the probability
     * that the cherry (A,B) is higher, with an exponential prior on the internal heights, integrated by the midpoint rule.
     */
    private double[] computeExactPosterior() {
        int numSteps = 100;
        double maxHeight = 8.0;
        double stepSize = maxHeight / numSteps;

        double normalisation = 0.0;
        double meanAB = 0.0;
        double meanCD = 0.0;
        double meanRoot = 0.0;
        double probabilityABHigher = 0.0;

        for (int i = 0; i < numSteps; i++) {
            double heightAB = (i + 0.5) * stepSize;
            for (int j = 0; j < numSteps; j++) {
                double heightCD = (j + 0.5) * stepSize;
                if (heightCD <= TIP_HEIGHT_OF_D) {
                    continue;
                }

                for (int k = Math.max(i, j); k < numSteps; k++) {
                    double rootHeight = (k + 0.5) * stepSize;
                    double weight = (k == Math.max(i, j) ? 0.5 : 1.0) * Math.exp(-heightAB - heightCD - rootHeight)
                            * this.computeLikelihood(heightAB, heightCD, rootHeight);

                    normalisation += weight;
                    meanAB += weight * heightAB;
                    meanCD += weight * heightCD;
                    meanRoot += weight * rootHeight;
                    probabilityABHigher += heightAB > heightCD ? weight : 0.0;
                }
            }
        }

        return new double[]{meanAB / normalisation, meanCD / normalisation, meanRoot / normalisation, probabilityABHigher / normalisation};
    }

    /** Returns the Jukes-Cantor likelihood of the tree ((A,B),(C,D)) with the given heights and the late tip D. */
    private double computeLikelihood(double heightAB, double heightCD, double rootHeight) {
        double likelihood = 1.0;

        for (int site = 0; site < SEQUENCES[0].length(); site++) {
            char[] states = {SEQUENCES[0].charAt(site), SEQUENCES[1].charAt(site), SEQUENCES[2].charAt(site), SEQUENCES[3].charAt(site)};

            double siteLikelihood = 0.0;
            for (char rootState : "ACGT".toCharArray()) {
                for (char stateAB : "ACGT".toCharArray()) {
                    double partAB = this.getJukesCantorProbability(rootState == stateAB, rootHeight - heightAB)
                            * this.getJukesCantorProbability(stateAB == states[0], heightAB)
                            * this.getJukesCantorProbability(stateAB == states[1], heightAB);

                    for (char stateCD : "ACGT".toCharArray()) {
                        siteLikelihood += 0.25 * partAB
                                * this.getJukesCantorProbability(rootState == stateCD, rootHeight - heightCD)
                                * this.getJukesCantorProbability(stateCD == states[2], heightCD)
                                * this.getJukesCantorProbability(stateCD == states[3], heightCD - TIP_HEIGHT_OF_D);
                    }
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
