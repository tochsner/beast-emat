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
import emat.operators.InnerNodeDisplacementOperator;
import emat.prior.GeneticPrior;
import emat.state.Mutation;
import emat.state.Mutations;
import org.junit.jupiter.api.Test;

import java.util.ArrayList;
import java.util.List;

import static org.junit.jupiter.api.Assertions.assertEquals;

public class InnerNodeDisplacementOperatorTest {

    private static final int NUM_BURNIN = 1000;
    private static final int NUM_SAMPLES = 200000;

    // nucleotide states
    private static final int A = 0;
    private static final int C = 1;
    private static final int G = 2;

    /**
     * On the tree ((A,B)X,C)R with a fixed history, runs a Metropolis-Hastings chain with the
     * operator against the genetic prior and an exponential prior on each internal height.
     * The mutations confine X to [0.5, 1.5] and R to [1.5, ∞), so both heights are
     * independent. Under Jukes-Cantor with rate 1 each of the 3 sites leaves its state at
     * rate 1, so λ = 3 everywhere. Raising X lengthens one branch and shortens another at the
     * same rate, so h_X follows the truncated exponential with rate λ + 1 = 4. Raising R
     * lengthens both root branches, so h_R - 1.5 follows the exponential with rate 2λ + 1 = 7.
     */
    @Test
    public void testHeightsMatchTruncatedExponentials() {
        Randomizer.setSeed(1);

        Alignment alignment = new Alignment();
        alignment.initByName("sequence", List.of(
                new Sequence("A", "GAA"), new Sequence("B", "GAC"), new Sequence("C", "CCC")
        ), "dataType", "nucleotide");

        TreeParser tree = new TreeParser();
        tree.initByName("newick", "((A:1.0,B:1.0):1.0,C:2.0);", "IsLabelledNewick", true, "adjustTipHeights", false);

        Mutations mutations = new Mutations();
        mutations.initByName("tree", tree, "alignment", alignment);
        mutations.initialiseMutations(this.createHistory(tree));

        SiteModel siteModel = new SiteModel();
        siteModel.initByName("substModel", new JukesCantor());

        StrictClockModel clockModel = new StrictClockModel();
        clockModel.initByName("clock.rate", new RealScalarParam<>(1.0, PositiveReal.INSTANCE));

        GeneticPrior geneticPrior = new GeneticPrior();
        geneticPrior.initByName("data", alignment, "tree", tree, "siteModel", siteModel, "branchRateModel", clockModel, "mutations", mutations);

        InnerNodeDisplacementOperator operator = new InnerNodeDisplacementOperator();
        operator.initByName("weight", 1.0, "tree", tree, "mutations", mutations, "geneticPrior", geneticPrior);

        // run the chain

        Node root = tree.getRoot();
        Node x = this.getInternalChild(root);

        double logTarget = -x.getHeight() - root.getHeight() + geneticPrior.calculateLogP();
        double sumXHeights = 0.0;
        double sumRootHeights = 0.0;

        for (int i = 0; i < NUM_BURNIN + NUM_SAMPLES; i++) {
            double oldXHeight = x.getHeight();
            double oldRootHeight = root.getHeight();
            List<List<Mutation>> oldMutations = new ArrayList<>();
            for (Node node : tree.getNodesAsArray()) {
                oldMutations.add(mutations.getMutations(node));
            }

            double logHastingsRatio = operator.proposal();

            if (logHastingsRatio != Double.NEGATIVE_INFINITY) {
                double newLogTarget = -x.getHeight() - root.getHeight() + geneticPrior.calculateLogP();

                if (Math.log(Randomizer.nextDouble()) < newLogTarget - logTarget + logHastingsRatio) {
                    logTarget = newLogTarget;
                } else {
                    x.setHeight(oldXHeight);
                    root.setHeight(oldRootHeight);
                    for (Node node : tree.getNodesAsArray()) {
                        mutations.applyMutations(node, oldMutations.get(node.getNr()), operator);
                    }
                    geneticPrior.calculateLogP();
                }
            }

            if (i >= NUM_BURNIN) {
                sumXHeights += x.getHeight();
                sumRootHeights += root.getHeight();
            }
        }

        double expectedXHeight = this.getTruncatedExponentialMean(4.0, 0.5, 1.5);
        double expectedRootHeight = 1.5 + 1.0 / 7.0;

        assertEquals(expectedXHeight, sumXHeights / NUM_SAMPLES, 0.005);
        assertEquals(expectedRootHeight, sumRootHeights / NUM_SAMPLES, 0.005);
    }

    /**
     * Creates a history for the tree ((A,B)X,C)R with the reference GAA of the first tip:
     * the root sequence is AAA, X gains G0 at height 1.5, B gains C2 at height 0.5, and C
     * gains C0, C1 and C2 at heights 1.5, 1.2 and 0.8.
     */
    private List<List<Mutation>> createHistory(TreeParser tree) {
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
