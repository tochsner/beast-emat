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
import emat.prior.GeneticPrior;
import emat.state.Mutation;
import emat.state.Mutations;
import org.junit.jupiter.api.Test;

import java.util.ArrayList;
import java.util.List;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertTrue;

public class QuantileIntervalScaleOperatorTest {

    private static final String SIX_TIP_NEWICK = "((((A:1.0,B:0.7):0.8,C:1.8):0.7,D:2.2):1.0,(E:1.5,F:1.1):2.0);";
    private static final List<String> SIX_TIP_SEQUENCES = List.of("AGTCACGT", "CTTCACGA", "CGAGACTT", "CGAGTCTT", "TGAGTAAT", "TGCGTAAC");

    private static final double THETA = 2.0;

    /** The tree, its mutations and the genetic prior of a test chain. */
    private record Chain(TreeParser tree, Mutations mutations, GeneticPrior geneticPrior) {
    }

    /**
     * With the tree prior exp(-θ Σ h) over the heights of all internal nodes, the chain of the quantile interval
     * operator must sample the same distribution of the node heights as a random walk on the heights, which targets
     * the density of the heights with the histories of the sites integrated out. The quantile interval operator is
     * mixed with the batch operator, which moves the nodes below the root individually: a shared shift of the
     * quantiles does not explore the target alone. The histories of the sites are the same in all chains and stay
     * fixed. The quantile interval operator gets four of five proposals, so that an error in its Hastings ratio is
     * not diluted.
     */
    @Test
    public void testChainMatchesRandomWalkOnHeights() {
        Chain chain = this.createSixTipChain();
        double[] initialHeights = this.getHeights(chain.tree());
        List<List<Mutation>> initialMutations = this.getMutations(chain);

        Randomizer.setSeed(3);
        double[] referenceMeans = this.runRandomWalk(chain, 3000000);

        Operator batchOperator = this.createBatchOperator(chain);

        // a shared shift of the scores mixes slowly on its own, so the configurations mix operators that let intervals with
        // large scores stay as they are with others that move more of them; these have to keep the target, too
        List<List<Operator>> configurations = List.of(
                List.of(this.createQuantileIntervalOperator(chain, false, 6.0), this.createQuantileIntervalOperator(chain, false, 0.8)),
                List.of(this.createQuantileIntervalOperator(chain, true, 6.0), this.createQuantileIntervalOperator(chain, true, 1.5)),
                List.of(this.createQuantileIntervalOperator(chain, false, 6.0), this.createQuantileIntervalOperator(chain, false, 1.5)));

        for (int k = 0; k < configurations.size(); k++) {
            List<Operator> quantileOperators = configurations.get(k);
            this.restoreState(chain, initialHeights, initialMutations, quantileOperators.get(0));
            Randomizer.setSeed(10 + k);

            List<Operator> operators = new ArrayList<>();
            for (int i = 0; i < 4; i++) {
                operators.add(quantileOperators.get(i % quantileOperators.size()));
            }
            operators.add(batchOperator);
            double[] means = this.runChain(operators, chain, 5000, 400000);

            for (Node node : chain.tree().getNodesAsArray()) {
                if (!node.isLeaf()) {
                    assertEquals(referenceMeans[node.getNr()], means[node.getNr()], 0.008, "node " + node.getNr() + ", configuration " + k);
                }
            }
        }
    }

    /** A shift followed by the shift in the opposite direction restores the heights, and the log Hastings ratios of both cancel. */
    @Test
    public void testOppositeShiftReversesMove() {
        Randomizer.setSeed(2);
        Chain chain = this.createSixTipChain();
        double[] shift = new double[1];
        QuantileIntervalScaleOperator operator = new QuantileIntervalScaleOperator() {
            @Override
            protected double drawScoreShift() {
                return shift[0];
            }
        };
        operator.initByName("weight", 1.0, "tree", chain.tree(), "mutations", chain.mutations(), "geneticPrior", chain.geneticPrior(),
                "optimise", false, "learnTreePrior", false);
        operator.setOperatorSchedule(new OperatorSchedule());
        chain.geneticPrior().calculateLogP();

        double[] initialHeights = this.getHeights(chain.tree());
        for (double step : new double[]{0.05, 0.5, 2.0}) {
            shift[0] = step;
            double forward = operator.proposal();
            operator.accept();

            shift[0] = -step;
            double backward = operator.proposal();
            operator.accept();

            double[] finalHeights = this.getHeights(chain.tree());
            for (int i = 0; i < finalHeights.length; i++) {
                assertEquals(initialHeights[i], finalHeights[i], 1e-6, "node " + i + " step " + step);
            }
            assertEquals(0.0, forward + backward, 1e-6, "step " + step);
        }
    }

    /** The acceptance of the operator is high when the tree prior is learned, since the genetic prior is approximated well. */
    @Test
    public void testAcceptanceIsHigh() {
        Randomizer.setSeed(5);
        Chain chain = this.createSixTipChain();
        Operator operator = this.createQuantileIntervalOperator(chain, true);
        operator.setOperatorSchedule(new OperatorSchedule());

        int numAccepted = 0;
        int numProposals = 0;
        double logTarget = this.computeLogTarget(chain);

        for (int i = 0; i < 30000; i++) {
            double[] oldHeights = this.getHeights(chain.tree());
            List<List<Mutation>> oldMutations = this.getMutations(chain);

            double logHastingsRatio = operator.proposal();
            boolean isAccepted = false;
            double logAlpha = Double.NEGATIVE_INFINITY;
            if (logHastingsRatio != Double.NEGATIVE_INFINITY) {
                double newLogTarget = this.computeLogTarget(chain);
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

        assertTrue(numAccepted > 0.2 * numProposals, "acceptance " + (double) numAccepted / numProposals);
    }

    /* Chains */

    /**
     * Runs the Metropolis-Hastings chain, in which each proposal comes from an operator of the given list drawn
     * uniformly, and returns the mean height of every node.
     */
    private double[] runChain(List<Operator> operators, Chain chain, int numBurnin, int numSamples) {
        for (Operator operator : operators) {
            operator.setOperatorSchedule(new OperatorSchedule());
        }

        double[] sums = new double[chain.tree().getNodeCount()];
        double logTarget = this.computeLogTarget(chain);

        for (int i = 0; i < numBurnin + numSamples; i++) {
            Operator operator = operators.get(Randomizer.nextInt(operators.size()));
            double[] oldHeights = this.getHeights(chain.tree());
            List<List<Mutation>> oldMutations = this.getMutations(chain);

            double logHastingsRatio = operator.proposal();
            boolean isAccepted = false;
            if (logHastingsRatio != Double.NEGATIVE_INFINITY) {
                double newLogTarget = this.computeLogTarget(chain);
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

    /**
     * Samples the node heights with a random walk on them from the density ∏ L^m exp(-Λ L) exp(-θ Σ h) over the
     * branch lengths L with m mutations each, which is the density of the heights with the relative positions of the
     * mutations integrated out. It does not use any of the operators.
     */
    private double[] runRandomWalk(Chain chain, int numSamples) {
        TreeParser tree = chain.tree();
        int numNodes = tree.getNodeCount();
        double[] heights = this.getHeights(tree);
        int[] numMutations = new int[numNodes];
        for (Node node : tree.getNodesAsArray()) {
            numMutations[node.getNr()] = chain.mutations().getMutations(node).size();
        }
        double escapeRate = SIX_TIP_SEQUENCES.get(0).length();

        double[] sums = new double[numNodes];
        double logDensity = this.computeLogHeightDensity(tree, heights, numMutations, escapeRate);
        for (int i = 0; i < 5000 + numSamples; i++) {
            Node node = tree.getInternalNodes().get(Randomizer.nextInt(tree.getInternalNodeCount()));
            double oldHeight = heights[node.getNr()];
            heights[node.getNr()] = oldHeight + 0.3 * (2.0 * Randomizer.nextDouble() - 1.0);
            double newLogDensity = this.computeLogHeightDensity(tree, heights, numMutations, escapeRate);
            if (Math.log(Randomizer.nextDouble()) < newLogDensity - logDensity) {
                logDensity = newLogDensity;
            } else {
                heights[node.getNr()] = oldHeight;
            }
            if (i >= 5000) {
                for (int j = 0; j < numNodes; j++) {
                    sums[j] += heights[j];
                }
            }
        }
        for (int j = 0; j < numNodes; j++) {
            sums[j] /= numSamples;
        }
        return sums;
    }

    private double computeLogHeightDensity(TreeParser tree, double[] heights, int[] numMutations, double escapeRate) {
        double logDensity = 0.0;
        for (Node node : tree.getNodesAsArray()) {
            if (node.isRoot()) {
                continue;
            }
            double length = heights[node.getParent().getNr()] - heights[node.getNr()];
            if (length <= 0.0) {
                return Double.NEGATIVE_INFINITY;
            }
            logDensity += numMutations[node.getNr()] * Math.log(length) - escapeRate * length;
        }
        for (Node node : tree.getInternalNodes()) {
            logDensity -= THETA * heights[node.getNr()];
        }
        return logDensity;
    }

    /* Setup */

    private Operator createBatchOperator(Chain chain) {
        BatchNodeDisplacementOperator operator = new BatchNodeDisplacementOperator();
        operator.initByName("weight", 1.0, "tree", chain.tree(), "mutations", chain.mutations(),
                "geneticPrior", chain.geneticPrior(), "batchSize", 2, "optimise", false, "learnTreePrior", false);
        return operator;
    }

    private Operator createQuantileIntervalOperator(Chain chain, boolean isLearningTreePrior) {
        return this.createQuantileIntervalOperator(chain, isLearningTreePrior, 6.0);
    }

    private Operator createQuantileIntervalOperator(Chain chain, boolean isLearningTreePrior, double maxScore) {
        QuantileIntervalScaleOperator operator = new QuantileIntervalScaleOperator() {
            @Override
            protected double getMaxScore() {
                return maxScore;
            }
        };
        operator.initByName("weight", 1.0, "tree", chain.tree(), "mutations", chain.mutations(), "geneticPrior", chain.geneticPrior(),
                "stepSize", 0.8, "optimise", isLearningTreePrior, "targetAcceptance", 0.3, "learnTreePrior", isLearningTreePrior,
                "treePriorWarmUp", 300, "treePriorKnots", 6);
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

    /** Returns the genetic prior plus the log of exp(-θ Σ h) over all internal nodes. */
    private double computeLogTarget(Chain chain) {
        double sumHeights = 0.0;
        for (Node node : chain.tree().getInternalNodes()) {
            sumHeights += node.getHeight();
        }
        return -THETA * sumHeights + chain.geneticPrior().calculateLogP();
    }

    private void restoreState(Chain chain, double[] heights, List<List<Mutation>> branchMutations, Operator operator) {
        // all heights first, as the mutations of a branch have to lie between its ends
        for (Node node : chain.tree().getNodesAsArray()) {
            node.setHeight(heights[node.getNr()]);
        }
        for (Node node : chain.tree().getNodesAsArray()) {
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
