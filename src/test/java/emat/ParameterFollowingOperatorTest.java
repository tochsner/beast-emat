package emat;

import beast.base.evolution.alignment.Alignment;
import beast.base.evolution.alignment.Sequence;
import beast.base.evolution.tree.Node;
import beast.base.evolution.tree.TreeParser;
import beast.base.inference.Operator;
import beast.base.internal.json.JSONException;
import beast.base.spec.domain.PositiveReal;
import beast.base.spec.domain.UnitInterval;
import beast.base.spec.evolution.branchratemodel.StrictClockModel;
import beast.base.spec.evolution.sitemodel.SiteModel;
import beast.base.spec.evolution.substitutionmodel.JukesCantor;
import beast.base.spec.inference.parameter.RealScalarParam;
import beast.base.util.Randomizer;
import emat.helper.ParameterFollowing;
import emat.initalisation.ParsimonyMutationsInitialiser;
import emat.operators.BatchNodeDisplacementOperator;
import emat.operators.ParameterFollowingOperator;
import emat.prior.GeneticPrior;
import emat.state.Mutation;
import emat.state.Mutations;
import org.junit.jupiter.api.Test;

import java.util.ArrayList;
import java.util.List;

import static org.junit.jupiter.api.Assertions.assertArrayEquals;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

public class ParameterFollowingOperatorTest {

    private static final String SIX_TIP_NEWICK = "((((A:1.0,B:0.7):0.8,C:1.8):0.7,D:2.2):1.0,(E:1.5,F:1.1):2.0);";
    private static final List<String> SIX_TIP_SEQUENCES = List.of("AGTCACGT", "CTTCACGA", "CGAGACTT", "CGAGTCTT", "TGAGTAAT", "TGCGTAAC");

    /** The state of a test chain and the operator that moves its tree. */
    private record Chain(TreeParser tree, Mutations mutations, GeneticPrior geneticPrior, RealScalarParam<PositiveReal> theta,
                         ParameterFollowing following, Operator treeOperator) {
    }

    /** The means and variances of θ and of the tree length over a chain, and the slope of log θ on log length. */
    private record Moments(double meanTheta, double varianceTheta, double meanLength, double varianceLength, double slope) {
    }

    /** With all coefficients at zero, the wrapper proposes exactly what the wrapped operator proposes. */
    @Test
    public void testReducesToWrappedOperatorWithZeroCoefficients() {
        Chain plain = this.createChain(false, 0.0, false);
        Chain wrapped = this.createChain(true, 0.0, false);

        Randomizer.setSeed(5);
        double[] plainRatios = new double[200];
        for (int i = 0; i < plainRatios.length; i++) {
            plainRatios[i] = plain.treeOperator().proposal();
        }

        Randomizer.setSeed(5);
        double[] wrappedRatios = new double[200];
        for (int i = 0; i < wrappedRatios.length; i++) {
            wrappedRatios[i] = wrapped.treeOperator().proposal();
        }

        assertArrayEquals(plainRatios, wrappedRatios, 0.0);
        assertArrayEquals(this.getHeights(plain.tree()), this.getHeights(wrapped.tree()), 0.0);
        assertEquals(2.0, wrapped.theta().get(), 0.0);
    }

    /**
     * The target is exp(-θ Σ h) over the internal nodes below the root, times the genetic prior and a log-normal
     * prior on θ, so that θ and the tree length depend on each other. A chain in which θ follows the tree with a
     * fixed coefficient must sample the same distribution as a chain with the plain operator. A wrong Jacobian
     * of the shift would show in the moments of θ.
     */
    @Test
    public void testSamplesSameDistributionWithFixedCoefficient() {
        Moments plain = this.runChain(this.createChain(false, 0.0, false), 600000, 11);
        Moments following = this.runChain(this.createChain(true, -1.5, false), 600000, 12);

        assertEquals(plain.meanTheta(), following.meanTheta(), 0.03 * plain.meanTheta());
        assertEquals(plain.varianceTheta(), following.varianceTheta(), 0.08 * plain.varianceTheta());
        assertEquals(plain.meanLength(), following.meanLength(), 0.01 * plain.meanLength());
        assertEquals(plain.varianceLength(), following.varianceLength(), 0.08 * plain.varianceLength());
    }

    /**
     * When the coefficient is learned, the chain still samples the same distribution, and the coefficient ends
     * up at the regression slope of log θ on the log tree length.
     */
    @Test
    public void testLearnsRegressionSlope() {
        Moments plain = this.runChain(this.createChain(false, 0.0, false), 600000, 21);

        Chain chain = this.createChain(true, 0.0, true);
        Moments following = this.runChain(chain, 600000, 22);

        assertTrue(chain.following().isFitted());
        // the slope is weak on this small tree, so its estimate varies by about a tenth between chains
        assertEquals(plain.slope(), chain.following().getCoefficients()[0], 0.3 * Math.abs(plain.slope()));
        assertEquals(plain.meanTheta(), following.meanTheta(), 0.03 * plain.meanTheta());
        assertEquals(plain.varianceLength(), following.varianceLength(), 0.08 * plain.varianceLength());
    }

    /** A shift that would take a parameter out of its domain rejects the proposal and leaves the parameter as it is. */
    @Test
    public void testRejectsShiftOutOfDomain() {
        Randomizer.setSeed(3);
        Chain chain = this.createChain(false, 0.0, false);

        RealScalarParam<UnitInterval> share = new RealScalarParam<>(0.9, UnitInterval.INSTANCE);
        ParameterFollowing following = new ParameterFollowing();
        following.initByName("tree", chain.tree(), "parameter", share, "optimise", false, "coefficient", List.of(50.0));

        ParameterFollowingOperator operator = new ParameterFollowingOperator();
        operator.initByName("weight", 1.0, "operator", chain.treeOperator(), "following", following);

        int numRejected = 0;
        for (int i = 0; i < 200; i++) {
            double oldShare = share.get();
            if (operator.proposal() == Double.NEGATIVE_INFINITY) {
                numRejected++;
                assertEquals(oldShare, share.get(), 0.0);
            }
            assertTrue(share.get() > 0.0 && share.get() <= 1.0);
        }
        assertTrue(numRejected > 0);
    }

    /** The batch operator cannot be wrapped when its draws depend on a followed parameter. */
    @Test
    public void testRefusesBatchOperatorThatUsesFollowedParameter() {
        Chain chain = this.createChain(false, 0.0, false);

        BatchNodeDisplacementOperator batchOperator = new BatchNodeDisplacementOperator();
        batchOperator.initByName("weight", 1.0, "tree", chain.tree(), "mutations", chain.mutations(), "geneticPrior", chain.geneticPrior(),
                "treePriorParameter", chain.theta());

        ParameterFollowing following = new ParameterFollowing();
        following.initByName("tree", chain.tree(), "parameter", chain.theta());

        ParameterFollowingOperator operator = new ParameterFollowingOperator();
        assertThrows(RuntimeException.class, () -> operator.initByName("weight", 1.0, "operator", batchOperator, "following", following));
    }

    /** What has been learned survives the state file. */
    @Test
    public void testRestoresLearnedState() throws JSONException {
        Chain chain = this.createChain(true, 0.0, true);
        this.runChain(chain, 20000, 31);
        assertTrue(chain.following().isFitted());

        ParameterFollowing restored = new ParameterFollowing();
        restored.initByName("tree", chain.tree(), "parameter", chain.theta());
        restored.restoreFromJson(chain.following().storeToJson());

        assertTrue(restored.isFitted());
        assertEquals(chain.following().getNumObservations(), restored.getNumObservations());
        assertArrayEquals(chain.following().getCoefficients(), restored.getCoefficients(), 0.0);
    }

    /* Chains */

    /**
     * Runs a Metropolis-Hastings chain that alternates between the tree operator and a random walk on log θ, and
     * returns the moments over its second half.
     */
    private Moments runChain(Chain chain, int numSteps, long seed) {
        Randomizer.setSeed(seed);
        TreeParser tree = chain.tree();
        RealScalarParam<PositiveReal> theta = chain.theta();

        double logTarget = this.computeLogTarget(chain);
        double sumTheta = 0.0, sumThetaSquares = 0.0, sumLength = 0.0, sumLengthSquares = 0.0;
        double sumLogTheta = 0.0, sumLogLength = 0.0, sumLogLengthSquares = 0.0, sumLogProducts = 0.0;
        int numSamples = 0;

        for (int i = 0; i < numSteps; i++) {
            double[] oldHeights = this.getHeights(tree);
            double oldTheta = theta.get();
            List<List<Mutation>> oldMutations = new ArrayList<>();
            for (Node node : tree.getNodesAsArray()) {
                oldMutations.add(chain.mutations().getMutations(node));
            }

            boolean isTreeMove = Randomizer.nextBoolean();
            double logHastingsRatio;
            if (isTreeMove) {
                logHastingsRatio = chain.treeOperator().proposal();
            } else {
                double newTheta = oldTheta * Math.exp(0.6 * (Randomizer.nextDouble() - 0.5));
                theta.set(newTheta);
                logHastingsRatio = Math.log(newTheta / oldTheta);
            }

            double logAlpha = Double.NEGATIVE_INFINITY;
            double newLogTarget = Double.NaN;
            if (logHastingsRatio != Double.NEGATIVE_INFINITY) {
                newLogTarget = this.computeLogTarget(chain);
                logAlpha = newLogTarget - logTarget + logHastingsRatio;
            }

            if (Math.log(Randomizer.nextDouble()) < logAlpha) {
                logTarget = newLogTarget;
            } else {
                theta.set(oldTheta);
                for (Node node : tree.getNodesAsArray()) {
                    if (!node.isLeaf()) {
                        node.setHeight(oldHeights[node.getNr()]);
                    }
                }
                for (Node node : tree.getNodesAsArray()) {
                    chain.mutations().applyMutations(node, oldMutations.get(node.getNr()), chain.treeOperator());
                }
                chain.geneticPrior().calculateLogP();
            }
            if (isTreeMove) {
                chain.treeOperator().optimize(logAlpha);
            }

            if (i >= numSteps / 2) {
                double length = this.computeTreeLength(tree);
                sumTheta += theta.get();
                sumThetaSquares += theta.get() * theta.get();
                sumLength += length;
                sumLengthSquares += length * length;
                sumLogTheta += Math.log(theta.get());
                sumLogLength += Math.log(length);
                sumLogLengthSquares += Math.log(length) * Math.log(length);
                sumLogProducts += Math.log(theta.get()) * Math.log(length);
                numSamples++;
            }
        }

        double meanTheta = sumTheta / numSamples;
        double meanLength = sumLength / numSamples;
        double meanLogLength = sumLogLength / numSamples;
        double slope = (sumLogProducts / numSamples - meanLogLength * sumLogTheta / numSamples)
                / (sumLogLengthSquares / numSamples - meanLogLength * meanLogLength);

        return new Moments(meanTheta, sumThetaSquares / numSamples - meanTheta * meanTheta,
                meanLength, sumLengthSquares / numSamples - meanLength * meanLength, slope);
    }

    /** Returns the log of exp(-θ Σ h) times the genetic prior and a log-normal prior on θ. */
    private double computeLogTarget(Chain chain) {
        double theta = chain.theta().get();

        double sumHeights = 0.0;
        for (Node node : chain.tree().getInternalNodes()) {
            if (!node.isRoot()) {
                sumHeights += node.getHeight();
            }
        }

        double logTheta = Math.log(theta) - Math.log(2.0);
        double logPrior = -logTheta * logTheta / (2.0 * 0.5 * 0.5) - Math.log(theta);

        return -theta * sumHeights + chain.geneticPrior().calculateLogP() + logPrior;
    }

    private double computeTreeLength(TreeParser tree) {
        double length = 0.0;
        for (Node node : tree.getNodesAsArray()) {
            if (!node.isRoot()) {
                length += node.getLength();
            }
        }
        return length;
    }

    private double[] getHeights(TreeParser tree) {
        double[] heights = new double[tree.getNodeCount()];
        for (Node node : tree.getNodesAsArray()) {
            heights[node.getNr()] = node.getHeight();
        }
        return heights;
    }

    /**
     * Creates the six-tip chain with the batch operator as its tree operator, wrapped if θ follows the tree. The
     * batch operator draws from the genetic prior only.
     */
    private Chain createChain(boolean isFollowing, double coefficient, boolean isLearned) {
        // the initial history is drawn at random and stays fixed, so every chain must start from the same one
        Randomizer.setSeed(1);

        List<Sequence> taxa = new ArrayList<>();
        for (int i = 0; i < SIX_TIP_SEQUENCES.size(); i++) {
            taxa.add(new Sequence(String.valueOf((char) ('A' + i)), SIX_TIP_SEQUENCES.get(i)));
        }

        Alignment alignment = new Alignment();
        alignment.initByName("sequence", taxa, "dataType", "nucleotide");

        TreeParser tree = new TreeParser();
        tree.initByName("newick", SIX_TIP_NEWICK, "IsLabelledNewick", true, "adjustTipHeights", false);

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
        geneticPrior.calculateLogP();

        RealScalarParam<PositiveReal> theta = new RealScalarParam<>(2.0, PositiveReal.INSTANCE);

        BatchNodeDisplacementOperator batchOperator = new BatchNodeDisplacementOperator();
        batchOperator.initByName("weight", 1.0, "tree", tree, "mutations", mutations, "geneticPrior", geneticPrior,
                "batchSize", 2, "optimise", false, "treePriorDegree", 0);

        if (!isFollowing) {
            return new Chain(tree, mutations, geneticPrior, theta, null, batchOperator);
        }

        ParameterFollowing following = new ParameterFollowing();
        if (isLearned) {
            following.initByName("tree", tree, "parameter", theta, "warmUp", 500);
        } else {
            following.initByName("tree", tree, "parameter", theta, "optimise", false, "coefficient", List.of(coefficient));
        }

        ParameterFollowingOperator operator = new ParameterFollowingOperator();
        operator.initByName("weight", 1.0, "operator", batchOperator, "following", following);

        return new Chain(tree, mutations, geneticPrior, theta, following, operator);
    }

}
