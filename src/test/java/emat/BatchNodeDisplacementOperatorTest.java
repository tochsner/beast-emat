package emat;

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
import emat.operators.BatchNodeDisplacementOperator;
import emat.operators.SiteHistoryGibbsOperator;
import emat.prior.GeneticPrior;
import emat.state.Mutation;
import emat.state.Mutations;
import org.junit.jupiter.api.Test;

import java.util.ArrayList;
import java.util.List;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertTrue;

public class BatchNodeDisplacementOperatorTest {

    private static final String SIX_TIP_NEWICK = "((((A:1.0,B:0.7):0.8,C:1.8):0.7,D:2.2):1.0,(E:1.5,F:1.1):2.0);";
    private static final List<String> SIX_TIP_SEQUENCES = List.of("AGTCACGT", "CTTCACGA", "CGAGACTT", "CGAGTCTT", "TGAGTAAT", "TGCGTAAC");

    // nucleotide states
    private static final int A = 0;
    private static final int C = 1;
    private static final int G = 2;

    /** The tree, its mutations and the genetic prior of a test chain. */
    private record Chain(TreeParser tree, Mutations mutations, GeneticPrior geneticPrior) {
    }

    /**
     * Without the approximation of the tree prior, the change of the genetic prior is cancelled exactly by the
     * Hastings ratio, also when several nodes move and the histories change between the proposals. No node moves
     * together with its parent or one of its children, and the root never moves.
     */
    @Test
    public void testGeneticPriorCancelsWithoutApproximation() {
        Randomizer.setSeed(1);
        Chain chain = this.createSixTipChain();
        TreeParser tree = chain.tree();
        GeneticPrior geneticPrior = chain.geneticPrior();

        BatchNodeDisplacementOperator operator = new BatchNodeDisplacementOperator();
        operator.initByName("weight", 1.0, "tree", tree, "mutations", chain.mutations(), "geneticPrior", geneticPrior,
                "batchSize", 3, "optimise", false, "treePriorDegree", 0);

        SiteHistoryGibbsOperator gibbsOperator = new SiteHistoryGibbsOperator();
        gibbsOperator.initByName("weight", 1.0, "mutations", chain.mutations(), "geneticPrior", geneticPrior);

        double logGeneticPrior = geneticPrior.calculateLogP();
        int numMultiNodeProposals = 0;

        for (int i = 0; i < 20000; i++) {
            if (i % 5 == 0) {
                // the Gibbs operator draws from the exact conditional, so it is always accepted
                gibbsOperator.proposal();
                logGeneticPrior = geneticPrior.calculateLogP();
            }

            double[] oldHeights = this.getHeights(tree);
            double logHastingsRatio = operator.proposal();
            double newLogGeneticPrior = geneticPrior.calculateLogP();

            int numMovedNodes = 0;
            for (Node node : tree.getNodesAsArray()) {
                if (node.getHeight() == oldHeights[node.getNr()]) {
                    continue;
                }

                numMovedNodes++;
                assertTrue(!node.isRoot());
                assertEquals(oldHeights[node.getParent().getNr()], node.getParent().getHeight(), 0.0);
                for (Node child : node.getChildren()) {
                    assertEquals(oldHeights[child.getNr()], child.getHeight(), 0.0);
                }
            }
            if (numMovedNodes > 1) {
                numMultiNodeProposals++;
            }

            assertEquals(0.0, newLogGeneticPrior - logGeneticPrior + logHastingsRatio, 1e-9);
            logGeneticPrior = newLogGeneticPrior;
        }

        assertTrue(numMultiNodeProposals > 10000);
    }

    /**
     * On the tree ((A,B)X,C)R with a fixed history, the mutations confine X to [0.5, 1.5], and under Jukes-Cantor
     * with rate 1 and 3 sites the genetic prior falls with the height of X at rate 3. With a tree prior
     * exp(-3 h_X), the height of X follows the truncated exponential with rate 6. The operator draws from the
     * genetic prior alone at first, and learns the slope -3 of the tree prior from the acceptance ratios. The
     * sampled mean must match, and once the slope is learned every proposal must be accepted.
     */
    @Test
    public void testLearnsConstantTreePriorSlope() {
        Randomizer.setSeed(1);

        Alignment alignment = new Alignment();
        alignment.initByName("sequence", List.of(
                new Sequence("A", "GAA"), new Sequence("B", "GAC"), new Sequence("C", "CCC")
        ), "dataType", "nucleotide");

        TreeParser tree = new TreeParser();
        tree.initByName("newick", "((A:1.0,B:1.0):1.0,C:2.0);", "IsLabelledNewick", true, "adjustTipHeights", false);

        Mutations mutations = new Mutations();
        mutations.initByName("tree", tree, "alignment", alignment);
        mutations.initialiseMutations(this.createThreeTipHistory(tree));

        GeneticPrior geneticPrior = this.createGeneticPrior(alignment, tree, mutations);

        BatchNodeDisplacementOperator operator = new BatchNodeDisplacementOperator();
        operator.initByName("weight", 1.0, "tree", tree, "mutations", mutations, "geneticPrior", geneticPrior,
                "batchSize", 1, "optimise", false, "treePriorDegree", 2, "treePriorWarmUp", 200);

        Node x = this.getInternalChild(tree.getRoot());
        double treePriorRate = 3.0;

        int numBurnin = 2000;
        int numSamples = 200000;
        double logTarget = -treePriorRate * x.getHeight() + geneticPrior.calculateLogP();
        double sumHeights = 0.0;
        int numEarlyAccepted = 0;
        int numLateAccepted = 0;

        for (int i = 0; i < numBurnin + numSamples; i++) {
            double oldHeight = x.getHeight();
            List<List<Mutation>> oldMutations = this.getMutations(tree, mutations);

            double logHastingsRatio = operator.proposal();
            double newLogTarget = -treePriorRate * x.getHeight() + geneticPrior.calculateLogP();
            double logAlpha = newLogTarget - logTarget + logHastingsRatio;

            if (Math.log(Randomizer.nextDouble()) < logAlpha) {
                logTarget = newLogTarget;
                if (i < 200) {
                    numEarlyAccepted++;
                } else if (i >= numBurnin) {
                    numLateAccepted++;
                }
            } else {
                x.setHeight(oldHeight);
                this.restoreMutations(tree, mutations, oldMutations, operator);
                geneticPrior.calculateLogP();
            }
            operator.optimize(logAlpha);

            if (i >= numBurnin) {
                sumHeights += x.getHeight();
            }
        }

        assertEquals(this.getTruncatedExponentialMean(6.0, 0.5, 1.5), sumHeights / numSamples, 0.005);
        assertTrue(numEarlyAccepted < 190);
        assertEquals(numSamples, numLateAccepted);
    }

    /**
     * On the six-tip tree with a fixed history, the tree prior exp(-θ Σ h) over the internal nodes below the root
     * has the slope -θ in every height, and θ is redrawn from time to time. With θ as a parameter of the tree
     * prior, the operator must learn the slope as a function of θ from batches of several nodes, so that in the
     * end every proposal is accepted whatever the value of θ.
     */
    @Test
    public void testLearnsTreePriorSlopeFromParameter() {
        Randomizer.setSeed(1);
        Chain chain = this.createSixTipChain();
        TreeParser tree = chain.tree();
        Mutations mutations = chain.mutations();
        GeneticPrior geneticPrior = chain.geneticPrior();

        RealScalarParam<PositiveReal> theta = new RealScalarParam<>(2.0, PositiveReal.INSTANCE);

        BatchNodeDisplacementOperator operator = new BatchNodeDisplacementOperator();
        operator.initByName("weight", 1.0, "tree", tree, "mutations", mutations, "geneticPrior", geneticPrior,
                "batchSize", 3, "optimise", false, "treePriorDegree", 3, "treePriorWarmUp", 500, "treePriorParameter", theta);

        int numProposals = 60000;
        int numLateProposals = 0;
        int numLateAccepted = 0;
        double logTarget = this.computeLogTreePrior(tree, theta.get()) + geneticPrior.calculateLogP();

        for (int i = 0; i < numProposals; i++) {
            if (i % 20 == 0) {
                theta.set(1.0 + 3.0 * Randomizer.nextDouble());
                logTarget = this.computeLogTreePrior(tree, theta.get()) + geneticPrior.calculateLogP();
            }

            double[] oldHeights = this.getHeights(tree);
            List<List<Mutation>> oldMutations = this.getMutations(tree, mutations);

            double logHastingsRatio = operator.proposal();
            double newLogTarget = this.computeLogTreePrior(tree, theta.get()) + geneticPrior.calculateLogP();
            double logAlpha = newLogTarget - logTarget + logHastingsRatio;

            boolean isAccepted = Math.log(Randomizer.nextDouble()) < logAlpha;
            if (isAccepted) {
                logTarget = newLogTarget;
            } else {
                for (Node node : tree.getNodesAsArray()) {
                    if (!node.isLeaf()) {
                        node.setHeight(oldHeights[node.getNr()]);
                    }
                }
                this.restoreMutations(tree, mutations, oldMutations, operator);
                geneticPrior.calculateLogP();
            }
            operator.optimize(logAlpha);

            if (i >= numProposals / 2) {
                numLateProposals++;
                if (isAccepted) {
                    numLateAccepted++;
                }
            }
        }

        assertTrue(numLateAccepted >= 0.999 * numLateProposals);
    }

    /* Helpers */

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

        return new Chain(tree, mutations, this.createGeneticPrior(alignment, tree, mutations));
    }

    private GeneticPrior createGeneticPrior(Alignment alignment, TreeParser tree, Mutations mutations) {
        SiteModel siteModel = new SiteModel();
        siteModel.initByName("substModel", new JukesCantor());

        StrictClockModel clockModel = new StrictClockModel();
        clockModel.initByName("clock.rate", new RealScalarParam<>(1.0, PositiveReal.INSTANCE));

        GeneticPrior geneticPrior = new GeneticPrior();
        geneticPrior.initByName("data", alignment, "tree", tree, "siteModel", siteModel, "branchRateModel", clockModel, "mutations", mutations);
        return geneticPrior;
    }

    /**
     * Creates a history for the tree ((A,B)X,C)R with the reference GAA of the first tip: the root sequence is
     * AAA, X gains G0 at height 1.5, B gains C2 at height 0.5, and C gains C0, C1 and C2 at heights 1.5, 1.2
     * and 0.8.
     */
    private List<List<Mutation>> createThreeTipHistory(TreeParser tree) {
        Node root = tree.getRoot();
        Node x = this.getInternalChild(root);
        Node b = this.getNodeByTaxon(tree, "B");
        Node c = this.getNodeByTaxon(tree, "C");

        List<List<Mutation>> history = new ArrayList<>();
        for (int i = 0; i < tree.getNodeCount(); i++) {
            history.add(new ArrayList<>());
        }

        history.get(root.getNr()).add(new Mutation(root.getNr(), 2.0, 2.0, 0, G, A));
        history.get(x.getNr()).add(new Mutation(x.getNr(), 1.5, 2.0, 0, A, G));
        history.get(b.getNr()).add(new Mutation(b.getNr(), 0.5, 1.0, 2, A, C));
        history.get(c.getNr()).addAll(List.of(
                new Mutation(c.getNr(), 1.5, 2.0, 0, A, C),
                new Mutation(c.getNr(), 1.2, 2.0, 1, A, C),
                new Mutation(c.getNr(), 0.8, 2.0, 2, A, C)
        ));

        return history;
    }

    /** Returns the log of exp(-θ Σ h) over the internal nodes below the root. */
    private double computeLogTreePrior(TreeParser tree, double theta) {
        double sumHeights = 0.0;
        for (Node node : tree.getInternalNodes()) {
            if (!node.isRoot()) {
                sumHeights += node.getHeight();
            }
        }
        return -theta * sumHeights;
    }

    private double[] getHeights(TreeParser tree) {
        double[] heights = new double[tree.getNodeCount()];
        for (Node node : tree.getNodesAsArray()) {
            heights[node.getNr()] = node.getHeight();
        }
        return heights;
    }

    private List<List<Mutation>> getMutations(TreeParser tree, Mutations mutations) {
        List<List<Mutation>> branchMutations = new ArrayList<>();
        for (int nodeNr = 0; nodeNr < tree.getNodeCount(); nodeNr++) {
            branchMutations.add(mutations.getMutations(tree.getNode(nodeNr)));
        }
        return branchMutations;
    }

    private void restoreMutations(TreeParser tree, Mutations mutations, List<List<Mutation>> branchMutations, BatchNodeDisplacementOperator operator) {
        for (Node node : tree.getNodesAsArray()) {
            mutations.applyMutations(node, branchMutations.get(node.getNr()), operator);
        }
    }

    private Node getInternalChild(Node node) {
        return node.getLeft().isLeaf() ? node.getRight() : node.getLeft();
    }

    private Node getNodeByTaxon(TreeParser tree, String taxon) {
        for (Node node : tree.getExternalNodes()) {
            if (node.getID().equals(taxon)) {
                return node;
            }
        }
        throw new IllegalArgumentException("Unknown taxon " + taxon + ".");
    }

    /** Returns the mean of the density ∝ exp(-k h) truncated to [minHeight, maxHeight]. */
    private double getTruncatedExponentialMean(double decayRate, double minHeight, double maxHeight) {
        double width = maxHeight - minHeight;
        double decay = Math.exp(-decayRate * width);
        return minHeight + 1.0 / decayRate - width * decay / (1.0 - decay);
    }

}
