package emat.operators.nodetimes.batch;

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
import java.util.function.Function;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertTrue;

public class TransportedNodeDisplacementOperatorTest {

    private static final String SIX_TIP_NEWICK = "((((A:1.0,B:0.7):0.8,C:1.8):0.7,D:2.2):1.0,(E:1.5,F:1.1):2.0);";
    private static final List<String> SIX_TIP_SEQUENCES = List.of("AGTCACGT", "CTTCACGA", "CGAGACTT", "CGAGTCTT", "TGAGTAAT", "TGCGTAAC");

    /** The tree, its mutations and the genetic prior of a test chain. */
    private record Chain(TreeParser tree, Mutations mutations, GeneticPrior geneticPrior) {
    }

    /** Creates a configuration of an operator on a chain, without the approximation of the tree prior. */
    private interface OperatorFactory extends Function<Chain, TransportedNodeDisplacementOperator> {
    }

    private final OperatorFactory quantileShift = chain -> this.createQuantileShift(chain, false);
    private final OperatorFactory persistentQuantileShift = chain -> this.createQuantileShift(chain, true);
    private final OperatorFactory gaussianRotation = chain -> this.createGaussianRotation(chain, 0.2);
    private final OperatorFactory reversibleGaussianRotation = chain -> this.createGaussianRotation(chain, 1.0);

    /**
     * Without the approximation of the tree prior, the change of the genetic prior is cancelled exactly by the
     * Hastings ratio, also when many nodes move at once and the histories change between the proposals.
     */
    @Test
    public void testGeneticPriorCancelsWithoutApproximation() {
        for (OperatorFactory factory : List.of(this.quantileShift, this.persistentQuantileShift, this.gaussianRotation, this.reversibleGaussianRotation)) {
            Randomizer.setSeed(1);
            Chain chain = this.createSixTipChain();
            GeneticPrior geneticPrior = chain.geneticPrior();
            TransportedNodeDisplacementOperator operator = factory.apply(chain);

            SiteHistoryGibbsOperator gibbsOperator = new SiteHistoryGibbsOperator();
            gibbsOperator.initByName("weight", 1.0, "mutations", chain.mutations(), "geneticPrior", geneticPrior);

            double logGeneticPrior = geneticPrior.calculateLogP();
            int numMultiNodeProposals = 0;

            for (int i = 0; i < 5000; i++) {
                if (i % 5 == 0) {
                    // the Gibbs operator draws from the exact conditional, so it is always accepted
                    gibbsOperator.proposal();
                    logGeneticPrior = geneticPrior.calculateLogP();
                }

                double[] oldHeights = this.getHeights(chain.tree());
                double logHastingsRatio = operator.proposal();
                if (logHastingsRatio == Double.NEGATIVE_INFINITY) {
                    operator.reject(-2);
                    continue;
                }
                double newLogGeneticPrior = geneticPrior.calculateLogP();

                int numMovedNodes = 0;
                for (Node node : chain.tree().getNodesAsArray()) {
                    if (node.getHeight() != oldHeights[node.getNr()]) {
                        numMovedNodes++;
                        assertTrue(!node.isRoot());
                    }
                }
                if (numMovedNodes > 1) {
                    numMultiNodeProposals++;
                }

                assertEquals(0.0, newLogGeneticPrior - logGeneticPrior + logHastingsRatio, 1e-8);
                logGeneticPrior = newLogGeneticPrior;
                operator.accept();
            }

            assertTrue(numMultiNodeProposals > 1000, factory.toString());
        }
    }

    /**
     * With the tree prior exp(-θ Σ h) over the heights of the internal nodes below the root, the chain of every
     * operator must sample the same distribution of the node heights as the batch operator alone, which draws every
     * node from its exact conditional. The operators move the nodes along one collective coordinate only, so they
     * leave the rest of the node heights as they are, and a chain of only one of them does not explore the target.
     * They are used next to operators that move the nodes individually, and the chain here mixes each of them with the
     * batch operator, which makes one proposal in five. The histories of the sites are the same and stay fixed in all chains. The comparison
     * fails if the transport or the Hastings ratio is wrong, or if the persistent direction or momentum breaks the
     * invariance.
     */
    @Test
    public void testChainMixedWithBatchOperatorMatchesBatchOperator() {
        double theta = 2.0;
        int numBurnin = 5000;
        int numSamples = 400000;

        // all chains start from the same heights and histories, which the initialiser does not give twice
        Chain chain = this.createSixTipChain();
        double[] initialHeights = this.getHeights(chain.tree());
        List<List<Mutation>> initialMutations = this.getMutations(chain);

        BatchNodeDisplacementOperator batchOperator = new BatchNodeDisplacementOperator();
        batchOperator.initByName("weight", 1.0, "tree", chain.tree(), "mutations", chain.mutations(),
                "geneticPrior", chain.geneticPrior(), "batchSize", 2, "optimise", false, "learnTreePrior", false);
        Randomizer.setSeed(3);
        double[] referenceMeans = this.runChain(List.of(batchOperator), chain, theta, numBurnin, numSamples);

        List<String> names = List.of("quantile shift", "persistent quantile shift", "gaussian rotation", "reversible gaussian rotation");
        List<OperatorFactory> factories = List.of(this.quantileShift, this.persistentQuantileShift, this.gaussianRotation, this.reversibleGaussianRotation);
        for (int k = 0; k < factories.size(); k++) {
            Operator operator = factories.get(k).apply(chain);
            this.restoreState(chain, initialHeights, initialMutations, operator);
            Randomizer.setSeed(10 + k);
            double[] means = this.runChain(List.of(operator, operator, operator, operator, batchOperator), chain, theta, numBurnin, numSamples);

            for (Node node : chain.tree().getNodesAsArray()) {
                assertEquals(referenceMeans[node.getNr()], means[node.getNr()], 0.004, "node " + node.getNr() + " of " + names.get(k));
            }
        }
    }

    /**
     * The proposals of the rotation are accepted at a high rate when the tree prior is exp(-θ Σ h) and the
     * approximation learns it, and the rate does not fall with the number of nodes that move together.
     */
    @Test
    public void testLearnedTreePriorKeepsAcceptanceHigh() {
        Randomizer.setSeed(5);
        double theta = 2.0;

        for (boolean isRotation : List.of(false, true)) {
            Chain chain = this.createSixTipChain();
            TransportedNodeDisplacementOperator operator = isRotation
                    ? this.createGaussianRotation(chain, 0.2, true)
                    : this.createQuantileShift(chain, false, true);

            int numAccepted = 0;
            int numProposals = 0;
            double logTarget = this.computeLogTarget(chain, theta);

            for (int i = 0; i < 30000; i++) {
                double[] oldHeights = this.getHeights(chain.tree());
                List<List<Mutation>> oldMutations = this.getMutations(chain);

                double logHastingsRatio = operator.proposal();
                boolean isAccepted = false;
                double logAlpha = Double.NEGATIVE_INFINITY;
                if (logHastingsRatio != Double.NEGATIVE_INFINITY) {
                    double newLogTarget = this.computeLogTarget(chain, theta);
                    logAlpha = newLogTarget - logTarget + logHastingsRatio;
                    isAccepted = Math.log(Randomizer.nextDouble()) < logAlpha;
                    if (isAccepted) {
                        logTarget = newLogTarget;
                    }
                }

                if (isAccepted) {
                    operator.accept();
                } else {
                    this.restoreState(chain, oldHeights, oldMutations, operator);
                    operator.reject(logHastingsRatio == Double.NEGATIVE_INFINITY ? -2 : 0);
                }
                operator.optimize(logAlpha);

                if (i >= 20000 && logHastingsRatio != Double.NEGATIVE_INFINITY) {
                    numProposals++;
                    if (isAccepted) {
                        numAccepted++;
                    }
                }
            }

            assertTrue(numAccepted > 0.4 * numProposals, "acceptance " + (double) numAccepted / numProposals + " of rotation " + isRotation);
        }
    }

    /* Chains */

    /**
     * Runs the Metropolis-Hastings chain, in which each proposal comes from an operator of the given list drawn
     * uniformly, so that an operator that is listed several times is drawn more often, and returns the mean height
     * of every node.
     */
    private double[] runChain(List<Operator> operators, Chain chain, double theta, int numBurnin, int numSamples) {
        for (Operator operator : operators) {
            operator.setOperatorSchedule(new OperatorSchedule());
        }

        double[] sums = new double[chain.tree().getNodeCount()];
        double logTarget = this.computeLogTarget(chain, theta);

        for (int i = 0; i < numBurnin + numSamples; i++) {
            Operator operator = operators.get(Randomizer.nextInt(operators.size()));
            double[] oldHeights = this.getHeights(chain.tree());
            List<List<Mutation>> oldMutations = this.getMutations(chain);

            double logHastingsRatio = operator.proposal();
            boolean isAccepted = false;
            if (logHastingsRatio != Double.NEGATIVE_INFINITY) {
                double newLogTarget = this.computeLogTarget(chain, theta);
                isAccepted = Math.log(Randomizer.nextDouble()) < newLogTarget - logTarget + logHastingsRatio;
                if (isAccepted) {
                    logTarget = newLogTarget;
                }
            }

            if (isAccepted) {
                operator.accept();
            } else {
                this.restoreState(chain, oldHeights, oldMutations, operator);
                operator.reject(logHastingsRatio == Double.NEGATIVE_INFINITY ? -2 : 0);
            }

            if (i >= numBurnin) {
                for (Node node : chain.tree().getNodesAsArray()) {
                    sums[node.getNr()] += node.getHeight();
                }
            }
        }

        for (int i = 0; i < sums.length; i++) {
            sums[i] /= numSamples;
        }
        return sums;
    }

    /* Setup */

    private TransportedNodeDisplacementOperator createQuantileShift(Chain chain, boolean isPersistent) {
        return this.createQuantileShift(chain, isPersistent, false);
    }

    private TransportedNodeDisplacementOperator createQuantileShift(Chain chain, boolean isPersistent, boolean learnTreePrior) {
        QuantileShiftNodeDisplacementOperator operator = new QuantileShiftNodeDisplacementOperator();
        operator.initByName("weight", 1.0, "tree", chain.tree(), "mutations", chain.mutations(), "geneticPrior", chain.geneticPrior(),
                "stepSize", 0.3, "optimise", learnTreePrior, "learnTreePrior", learnTreePrior, "persistent", isPersistent,
                "treePriorWarmUp", 300, "treePriorKnots", 6);
        operator.setOperatorSchedule(new OperatorSchedule());
        return operator;
    }

    private TransportedNodeDisplacementOperator createGaussianRotation(Chain chain, double refresh) {
        return this.createGaussianRotation(chain, refresh, false);
    }

    private TransportedNodeDisplacementOperator createGaussianRotation(Chain chain, double refresh, boolean learnTreePrior) {
        GaussianRotationNodeDisplacementOperator operator = new GaussianRotationNodeDisplacementOperator();
        operator.initByName("weight", 1.0, "tree", chain.tree(), "mutations", chain.mutations(), "geneticPrior", chain.geneticPrior(),
                "stepSize", 0.4, "optimise", learnTreePrior, "learnTreePrior", learnTreePrior, "refresh", refresh,
                "treePriorWarmUp", 300, "treePriorKnots", 6);
        operator.setOperatorSchedule(new OperatorSchedule());
        return operator;
    }

    /** Sets up the six-tip tree with its parsimony history under Jukes-Cantor with rate 1. */
    private Chain createSixTipChain() {
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

        return new Chain(tree, mutations, geneticPrior);
    }

    /* State */

    /** Returns the genetic prior plus the log of exp(-θ Σ h) over the internal nodes below the root. */
    private double computeLogTarget(Chain chain, double theta) {
        double sumHeights = 0.0;
        for (Node node : chain.tree().getInternalNodes()) {
            if (!node.isRoot()) {
                sumHeights += node.getHeight();
            }
        }
        return -theta * sumHeights + chain.geneticPrior().calculateLogP();
    }

    private void restoreState(Chain chain, double[] heights, List<List<Mutation>> branchMutations, Operator operator) {
        for (Node node : chain.tree().getNodesAsArray()) {
            node.setHeight(heights[node.getNr()]);
            chain.mutations().applyMutations(node, branchMutations.get(node.getNr()), operator);
        }
        chain.geneticPrior().calculateLogP();
    }

    private double[] getHeights(TreeParser tree) {
        double[] heights = new double[tree.getNodeCount()];
        for (Node node : tree.getNodesAsArray()) {
            heights[node.getNr()] = node.getHeight();
        }
        return heights;
    }

    private List<List<Mutation>> getMutations(Chain chain) {
        List<List<Mutation>> branchMutations = new ArrayList<>();
        for (int nodeNr = 0; nodeNr < chain.tree().getNodeCount(); nodeNr++) {
            branchMutations.add(chain.mutations().getMutations(chain.tree().getNode(nodeNr)));
        }
        return branchMutations;
    }

}
